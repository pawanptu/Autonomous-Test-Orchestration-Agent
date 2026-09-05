"""Defensive JSON handling for LLM output.

These cover the deterministic helpers only. Nothing here asserts anything about
what a model *would* say - that is not a testable property - only that whatever
it says is handled without crashing the node.
"""

from __future__ import annotations

import pytest

from llm.json_utils import (
    JSONParseError,
    RETRY_INSTRUCTION,
    build_retry_messages,
    coerce_list,
    find_json_span,
    loads_lenient,
    parse_model,
    repair_common_json_issues,
    strip_code_fences,
)


class TestStripCodeFences:
    def test_plain_json_untouched(self):
        assert strip_code_fences('{"a": 1}') == '{"a": 1}'

    def test_removes_json_fence(self):
        assert strip_code_fences('```json\n{"a": 1}\n```') == '{"a": 1}'

    def test_removes_bare_fence(self):
        assert strip_code_fences('```\n{"a": 1}\n```') == '{"a": 1}'

    def test_removes_reasoning_tags(self):
        raw = "<think>I should answer with JSON</think>\n{\"a\": 1}"
        assert strip_code_fences(raw) == '{"a": 1}'

    def test_empty_input(self):
        assert strip_code_fences("") == ""


class TestFindJsonSpan:
    def test_extracts_object_from_prose(self):
        assert find_json_span('Sure! {"a": 1} hope that helps') == '{"a": 1}'

    def test_extracts_array(self):
        assert find_json_span("here: [1, 2, 3] done") == "[1, 2, 3]"

    def test_brace_inside_string_does_not_confuse_scanner(self):
        raw = '{"msg": "shows {0} results", "n": 1}'
        assert find_json_span(f"prefix {raw} suffix") == raw

    def test_escaped_quote_inside_string(self):
        raw = '{"msg": "he said \\"hi\\"", "n": 1}'
        assert find_json_span(raw) == raw

    def test_nested_objects(self):
        raw = '{"a": {"b": {"c": [1, {"d": 2}]}}}'
        assert find_json_span(raw) == raw

    def test_returns_none_when_no_json(self):
        assert find_json_span("no json at all") is None

    def test_unterminated_returns_remainder_for_repair(self):
        span = find_json_span('{"a": [1, 2')
        assert span is not None and span.startswith("{")


class TestRepair:
    def test_removes_trailing_comma_in_object(self):
        assert repair_common_json_issues('{"a": 1,}') == '{"a": 1}'

    def test_removes_trailing_comma_in_array(self):
        assert repair_common_json_issues("[1, 2,]") == "[1, 2]"

    def test_converts_python_literals(self):
        out = repair_common_json_issues('{"a": True, "b": False, "c": None}')
        assert "true" in out and "false" in out and "null" in out

    def test_does_not_touch_true_inside_a_string(self):
        out = repair_common_json_issues('{"a": "True story"}')
        assert '"True story"' in out

    def test_normalises_smart_quotes(self):
        assert '"a"' in repair_common_json_issues('{“a”: 1}')

    def test_strips_line_comments(self):
        assert repair_common_json_issues('{"a": 1} // done') == '{"a": 1}'

    def test_balances_truncated_object(self):
        assert repair_common_json_issues('{"a": [1, 2') == '{"a": [1, 2]}'


class TestLoadsLenient:
    def test_strict_json(self):
        assert loads_lenient('{"a": 1}') == {"a": 1}

    def test_fenced_with_prose_and_python_literals(self):
        raw = 'Here you go:\n```json\n{"a": [1, 2,], "b": True, "c": None}\n```\nEnjoy!'
        assert loads_lenient(raw) == {"a": [1, 2], "b": True, "c": None}

    def test_truncated_response_is_recovered(self):
        assert loads_lenient('{"flows": [{"id": "F001", "name": "abc') == {
            "flows": [{"id": "F001", "name": "abc"}]
        }

    def test_array_root(self):
        assert loads_lenient("[1, 2, 3]") == [1, 2, 3]

    @pytest.mark.parametrize("bad", ["", "   ", "no json here", None])
    def test_unparsable_raises_with_raw_attached(self, bad):
        with pytest.raises(JSONParseError) as excinfo:
            loads_lenient(bad)
        assert hasattr(excinfo.value, "raw")


class TestParseModel:
    def test_validates_into_a_model(self):
        from graph.state import TestStep

        step = parse_model('{"action": "click", "target": "the button"}', TestStep)
        assert step.action == "click" and step.target == "the button"

    def test_schema_violation_raises_json_parse_error(self):
        from graph.state import TestStep

        with pytest.raises(JSONParseError):
            parse_model('{"action": "teleport"}', TestStep)


class TestCoerceList:
    def test_passes_a_list_through(self):
        assert coerce_list([1, 2]) == [1, 2]

    def test_unwraps_named_key(self):
        assert coerce_list({"flows": [1, 2]}, "flows") == [1, 2]

    def test_unwraps_common_fallback_keys(self):
        assert coerce_list({"items": [1]}) == [1]

    def test_wraps_a_bare_object(self):
        assert coerce_list({"id": "F001"}) == [{"id": "F001"}]

    def test_none_becomes_empty(self):
        assert coerce_list(None) == []


class TestRetryMessages:
    def test_appends_bad_output_and_repair_instruction(self):
        original = [{"role": "user", "content": "give me JSON"}]
        messages = build_retry_messages(original, "not json", "Expecting value")

        assert len(messages) == 3
        assert messages[0] == original[0]
        assert messages[1]["role"] == "assistant"
        # The invalid output must be kept: without it the model has no idea
        # what it is being asked to correct.
        assert messages[1]["content"] == "not json"
        assert messages[2]["role"] == "user"
        assert "Expecting value" in messages[2]["content"]
        assert "invalid JSON" in messages[2]["content"]

    def test_original_messages_are_not_mutated(self):
        original = [{"role": "user", "content": "x"}]
        build_retry_messages(original, "bad", "err")
        assert len(original) == 1

    def test_instruction_template_has_the_error_placeholder(self):
        assert "{error}" in RETRY_INSTRUCTION
