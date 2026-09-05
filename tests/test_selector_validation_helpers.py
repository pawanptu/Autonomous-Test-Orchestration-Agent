"""Selector resolution helpers.

Everything covered here is the part of selector resolution that runs *without*
a browser: turning a plain-language target into candidate strategies, emitting
valid Python for each, ranking probed candidates, and parsing an emitted
expression back into its parts (which is how the Healer re-probes a locator).

The live probing itself needs Chromium and is exercised by the end-to-end run,
not here.
"""

from __future__ import annotations

import ast

import pytest

from browser.sandbox import validate_test_source
from agents.generator import _regex_literal
from browser.selectors import (
    STRATEGY_PRIORITY,
    SelectorCandidate,
    build_candidates,
    escape_py_string,
    expression_for,
    extract_quoted,
    fallback_order,
    guess_role,
    normalize_intent,
    parse_expression,
    pick_best,
    score_candidate,
    strategy_rank,
)


class TestNormalizeIntent:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("the 'Add to Basket' button", "add to basket"),
            ('the "Sign In" button', "sign in"),
            ("the search input", "search"),
            ("the email field", "email"),
            ("the main heading", "heading"),
            ("", ""),
        ],
    )
    def test_reduces_to_discriminating_words(self, raw, expected):
        assert normalize_intent(raw) == expected

    def test_quoted_text_wins_over_surrounding_words(self):
        assert normalize_intent("click the 'Proceed to Checkout' control") == "proceed to checkout"

    def test_only_stopwords_yields_empty(self):
        assert normalize_intent("the the a an") == ""


class TestExtractQuoted:
    def test_straight_quotes(self):
        assert extract_quoted("click 'Buy now' please") == ["Buy now"]

    def test_double_quotes(self):
        assert extract_quoted('click "Buy now"') == ["Buy now"]

    def test_curly_quotes(self):
        assert extract_quoted("click ‘Buy now’") == ["Buy now"]

    def test_none_present(self):
        assert extract_quoted("click the button") == []


class TestGuessRole:
    @pytest.mark.parametrize(
        "intent,role",
        [
            ("the submit button", "button"),
            ("the navigation link", "link"),
            ("the page heading", "heading"),
            ("the password field", "textbox"),
            ("the search box", "searchbox"),
            ("the terms checkbox", "checkbox"),
            ("the country dropdown", "combobox"),
            ("the error message", "alert"),
        ],
    )
    def test_wording_drives_the_role(self, intent, role):
        assert guess_role(intent) == role

    def test_action_drives_the_role_when_wording_is_neutral(self):
        assert guess_role("the widget", "fill") == "textbox"
        assert guess_role("the widget", "click") == "button"

    def test_unknown_action_still_returns_something(self):
        assert guess_role("thing", "teleport") == "button"


class TestFallbackOrder:
    def test_form_actions_lead_with_label_and_placeholder(self):
        order = fallback_order("fill")
        assert order[0] == "testid"
        assert order.index("label") < order.index("role")
        assert order.index("placeholder") < order.index("css")

    def test_click_uses_the_default_priority(self):
        assert fallback_order("click") == STRATEGY_PRIORITY

    def test_assertions_prefer_role_and_text(self):
        order = fallback_order("assert_text")
        assert order.index("text") < order.index("placeholder")

    def test_every_order_is_a_permutation_of_the_known_strategies(self):
        for action in ("fill", "click", "assert_text", "press", "wait_for"):
            assert set(fallback_order(action)) == set(STRATEGY_PRIORITY)

    def test_strategy_rank_orders_most_specific_first(self):
        assert strategy_rank("testid") < strategy_rank("role") < strategy_rank("css")

    def test_unknown_strategy_sorts_last(self):
        assert strategy_rank("nonsense") == len(STRATEGY_PRIORITY)


class TestExpressionFor:
    @pytest.mark.parametrize(
        "strategy,value,expected",
        [
            ("testid", "submit", 'page.get_by_test_id("submit")'),
            ("label", "Email", 'page.get_by_label("Email")'),
            ("placeholder", "Search", 'page.get_by_placeholder("Search")'),
            ("text", "Buy now", 'page.get_by_text("Buy now")'),
            ("alt", "Logo", 'page.get_by_alt_text("Logo")'),
            ("title", "Close", 'page.get_by_title("Close")'),
            ("css", "input[type='email']", "page.locator(\"input[type='email']\")"),
        ],
    )
    def test_renders_each_strategy(self, strategy, value, expected):
        assert expression_for(strategy, value) == expected

    def test_role_includes_the_name_filter(self):
        assert expression_for("role", "Buy", role="button") == (
            'page.get_by_role("button", name="Buy")'
        )

    def test_role_without_a_name(self):
        assert expression_for("role", "", role="heading") == 'page.get_by_role("heading")'

    def test_unknown_strategy_raises(self):
        with pytest.raises(ValueError):
            expression_for("telepathy", "x")

    @pytest.mark.parametrize(
        "value",
        ['Add to "Basket"', "back\\slash", "line\nbreak", "tab\there", "quote'inside"],
    )
    def test_every_emitted_expression_is_valid_python(self, value):
        # This is the property that matters: emitted source goes straight into
        # a generated test file, so it must always parse.
        ast.parse(expression_for("role", value, role="button"))
        ast.parse(expression_for("text", value))

    def test_escaping_round_trips_through_python(self):
        value = 'He said "hi"\\there'
        rendered = f'x = "{escape_py_string(value)}"'
        namespace: dict = {}
        exec(compile(rendered, "<test>", "exec"), namespace)  # noqa: S102
        assert namespace["x"] == value


class TestScoring:
    def candidate(self, **kwargs) -> SelectorCandidate:
        return SelectorCandidate(
            strategy=kwargs.pop("strategy", "role"),
            expression=kwargs.pop("expression", 'page.get_by_role("button", name="Buy")'),
            match_count=kwargs.pop("match_count", 1),
            visible=kwargs.pop("visible", True),
            **kwargs,
        )

    def test_zero_matches_is_disqualifying(self):
        assert score_candidate(self.candidate(match_count=0), "buy") < 0

    def test_one_visible_match_beats_one_hidden_match(self):
        visible = score_candidate(self.candidate(visible=True), "buy")
        hidden = score_candidate(self.candidate(visible=False), "buy")
        assert visible > hidden

    def test_one_match_beats_many(self):
        one = score_candidate(self.candidate(match_count=1), "buy")
        many = score_candidate(self.candidate(match_count=7), "buy")
        assert one > many

    def test_more_specific_strategy_wins_a_tie(self):
        testid = score_candidate(
            self.candidate(strategy="testid", expression='page.get_by_test_id("buy")'), "buy"
        )
        css = score_candidate(
            self.candidate(strategy="css", expression='page.locator("button")'), "buy"
        )
        assert testid > css

    def test_pick_best_returns_none_when_nothing_matched(self):
        assert pick_best([self.candidate(match_count=0)], "buy") is None

    def test_pick_best_returns_the_top_scorer(self):
        best = pick_best(
            [
                self.candidate(strategy="css", match_count=9, visible=True,
                               expression='page.locator("button")'),
                self.candidate(strategy="testid", match_count=1, visible=True,
                               expression='page.get_by_test_id("buy")'),
            ],
            "buy",
        )
        assert best is not None and best.strategy == "testid"

    def test_pick_best_on_an_empty_list(self):
        assert pick_best([], "buy") is None


class TestBuildCandidates:
    def test_produces_ordered_candidates(self):
        candidates = build_candidates(intent="the 'Sign In' button", action="click")
        assert candidates
        assert all(c.expression for c in candidates)

    def test_quoted_literal_appears_in_the_leading_candidates(self):
        candidates = build_candidates(intent="the 'Add to Basket' button", action="click")
        assert any("Add to Basket" in c.expression for c in candidates[:4])

    def test_password_intent_gets_the_conventional_css_fallback(self):
        candidates = build_candidates(intent="the password field", action="fill")
        assert any("input[type='password']" in c.expression for c in candidates)

    def test_email_intent_gets_the_conventional_css_fallback(self):
        candidates = build_candidates(intent="the email field", action="fill")
        assert any("input[type='email']" in c.expression for c in candidates)

    def test_page_inventory_drives_real_candidates(self):
        inventory = {
            "inputs": [{"label": "Email address", "placeholder": "you@example.com", "name": "email"}],
            "buttons": [],
            "headings": [],
        }
        candidates = build_candidates(
            intent="the email field", action="fill", page_inventory=inventory
        )
        # A label that genuinely exists on the page beats a guessed one.
        assert any("Email address" in c.expression for c in candidates)

    def test_button_inventory_drives_role_candidates(self):
        inventory = {
            "inputs": [],
            "buttons": [{"text": "Proceed to checkout", "aria_label": ""}],
            "headings": [],
        }
        candidates = build_candidates(
            intent="the checkout button", action="click", page_inventory=inventory
        )
        assert any("Proceed to checkout" in c.expression for c in candidates)

    def test_respects_the_candidate_cap(self):
        candidates = build_candidates(intent="the submit button", action="click", max_candidates=3)
        assert len(candidates) <= 3

    def test_no_duplicate_expressions(self):
        candidates = build_candidates(intent="the search input", action="fill")
        expressions = [c.expression for c in candidates]
        assert len(expressions) == len(set(expressions))

    def test_empty_intent_still_returns_a_role_fallback(self):
        assert build_candidates(intent="", action="click") != []

    def test_every_candidate_is_valid_python(self):
        for intent in ("the 'Buy \"now\"' button", "the search input", "the error message"):
            for candidate in build_candidates(intent=intent, action="click"):
                ast.parse(candidate.expression)


class TestParseExpression:
    @pytest.mark.parametrize(
        "expression,strategy,value",
        [
            ('page.get_by_role("button", name="Buy now")', "role", "Buy now"),
            ('page.get_by_test_id("submit")', "testid", "submit"),
            ('page.get_by_label("Email")', "label", "Email"),
            ('page.get_by_placeholder("Search")', "placeholder", "Search"),
            ('page.get_by_text("Welcome")', "text", "Welcome"),
            ('page.locator("#main")', "css", "#main"),
        ],
    )
    def test_round_trips_each_strategy(self, expression, strategy, value):
        parsed = parse_expression(expression)
        assert parsed is not None
        assert parsed.strategy == strategy
        assert parsed.value == value

    def test_keeps_the_role_name(self):
        parsed = parse_expression('page.get_by_role("heading", name="Welcome")')
        assert parsed is not None and parsed.role == "heading"

    def test_tolerates_a_first_suffix(self):
        assert parse_expression('page.get_by_text("Item").first') is not None

    def test_handles_an_escaped_quote(self):
        parsed = parse_expression('page.get_by_text("say \\"hi\\"")')
        assert parsed is not None and parsed.value == 'say "hi"'

    @pytest.mark.parametrize(
        "bad", ["", "not an expression", "page.click()", "driver.find_element('x')"]
    )
    def test_unparsable_returns_none(self, bad):
        assert parse_expression(bad) is None


class TestGeneratedSourceAudit:
    """The AST gate that stands between model output and ``exec``."""

    GOOD = (
        "async def test_flow(page, ctx):\n"
        '    await page.goto("https://example.com/")\n'
        '    await expect(page.get_by_role("heading")).to_be_visible(timeout=8000)\n'
    )

    def test_accepts_well_formed_source(self):
        assert validate_test_source(self.GOOD).ok is True

    def test_rejects_empty_source(self):
        assert validate_test_source("").ok is False

    def test_rejects_a_syntax_error(self):
        assert validate_test_source("async def test_flow(page, ctx:\n    pass").ok is False

    def test_rejects_a_missing_entry_point(self):
        assert validate_test_source("async def other(page, ctx):\n    await page.goto('/')\n").ok is False

    def test_rejects_a_non_async_entry_point(self):
        assert validate_test_source("def test_flow(page, ctx):\n    pass\n").ok is False

    def test_rejects_the_wrong_signature(self):
        source = "async def test_flow(browser):\n    await browser.goto('/')\n"
        assert validate_test_source(source).ok is False

    @pytest.mark.parametrize(
        "line",
        [
            "    import os",
            "    import subprocess",
            "    from pathlib import Path",
            '    exec("print(1)")',
            '    eval("1+1")',
            '    open("/etc/passwd")',
            "    x = page.__class__.__bases__",
            "    __import__('os')",
        ],
    )
    def test_rejects_forbidden_constructs(self, line):
        source = f"async def test_flow(page, ctx):\n{line}\n    await page.goto('/')\n"
        assert validate_test_source(source).ok is False

    def test_allows_a_permitted_import(self):
        source = (
            "async def test_flow(page, ctx):\n"
            "    import re\n"
            '    await page.goto("https://example.com/")\n'
            '    assert re.match("a", "abc")\n'
        )
        assert validate_test_source(source).ok is True

    def test_rejects_source_that_never_awaits(self):
        source = "async def test_flow(page, ctx):\n    x = 1\n"
        assert validate_test_source(source).ok is False

    def test_warns_but_accepts_source_with_no_assertion(self):
        source = "async def test_flow(page, ctx):\n    await page.goto('https://example.com/')\n"
        verdict = validate_test_source(source)
        assert verdict.ok is True
        assert any("assertion" in w for w in verdict.warnings)


class TestUrlAssertionPatterns:
    def test_escaped_url_matches_the_original_url(self):
        import re

        url = "https://example.com/catalog/item-1?color=red"
        assert re.fullmatch(_regex_literal(url), url)
