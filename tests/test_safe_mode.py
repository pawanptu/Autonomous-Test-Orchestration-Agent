"""Safe mode must make irreversible actions impossible by default.

The agent performs each destructive action *twice* if nothing stops it: once
while exploring the application and again when the generated test runs. These
tests pin three properties:

1. destructive intent is recognised from the text of a step;
2. a blocked step is compiled into an explicit failure, never dropped; and
3. the operator can authorise a named category without unlocking the rest.

The third matters most in practice: a team that wants checkout coverage on a
throwaway environment must be able to get it without also authorising
"delete account".
"""

from __future__ import annotations

import pytest

from agents.generator import compile_steps_to_source, is_simple_flow, screen_flow_for_safety
from graph.state import SelectorValidation, TestFlow, TestStep
from safe_actions import (
    DestructiveCategory,
    SafetyPolicy,
    classify_action,
    data_marker,
    evaluate_action,
    is_marked,
    is_safe_to_explore,
    mark_value,
    parse_categories,
)


class TestClassification:
    @pytest.mark.parametrize(
        "text,expected",
        [
            ("click the Place Order button", DestructiveCategory.CHECKOUT),
            ("Complete the purchase", DestructiveCategory.CHECKOUT),
            ("Pay now with the saved card", DestructiveCategory.PAYMENT),
            ("enter the credit card number", DestructiveCategory.PAYMENT),
            ("delete my account", DestructiveCategory.ACCOUNT_CANCELLATION),
            ("cancel my subscription", DestructiveCategory.ACCOUNT_CANCELLATION),
            ("reset my password", DestructiveCategory.PASSWORD_RESET),
            ("send the invoice email", DestructiveCategory.EMAIL_SEND),
            ("delete the uploaded document", DestructiveCategory.DELETE),
            ("transfer funds to the payee", DestructiveCategory.IRREVERSIBLE_SUBMIT),
        ],
    )
    def test_destructive_text_is_classified(self, text, expected):
        category, matched = classify_action(text)
        assert category is expected
        assert matched

    @pytest.mark.parametrize(
        "text",
        [
            "click the Login button",
            "fill the search box",
            "add the book to the cart",
            "assert the heading is visible",
            "open the product listing",
        ],
    )
    def test_ordinary_actions_are_not_destructive(self, text):
        assert classify_action(text) == (None, "")

    @pytest.mark.parametrize(
        "text",
        [
            "remove the item from the cart",
            "clear the search filter",
            "cancel the dialog",
            "delete the search query",
        ],
    )
    def test_benign_lookalikes_are_not_blocked(self, text):
        """A cart is a scratch surface; clearing a filter destroys nothing.

        Over-blocking here would make the agent unable to test the ordinary
        interactions that make up most of an application.
        """
        assert classify_action(text) == (None, "")

    def test_intent_is_read_across_all_fragments_of_a_step(self):
        """The destructive verb may live in the target rather than the action."""
        category, _ = classify_action("click", "the Place Order button", None)
        assert category is DestructiveCategory.CHECKOUT


class TestPolicyGate:
    def test_safe_mode_blocks_by_default(self):
        decision = evaluate_action(SafetyPolicy(), "click Place Order")
        assert decision.blocked
        assert decision.reason == "destructive-action-blocked"
        assert decision.category is DestructiveCategory.CHECKOUT

    def test_blocked_reason_is_explicit_and_actionable(self):
        """A blocked action must say what to do about it, not merely refuse."""
        decision = evaluate_action(SafetyPolicy(), "delete the record")
        assert "AUTHORIZED_DESTRUCTIVE_ACTIONS" in decision.detail
        assert "SAFE_MODE=false" in decision.detail

    def test_named_category_is_authorised_without_unlocking_the_rest(self):
        policy = SafetyPolicy(authorized=frozenset({DestructiveCategory.CHECKOUT}))
        assert evaluate_action(policy, "click Place Order").allowed
        assert evaluate_action(policy, "delete my account").blocked

    def test_disabling_safe_mode_permits_every_category(self):
        policy = SafetyPolicy(safe_mode=False)
        assert evaluate_action(policy, "delete my account").allowed
        assert evaluate_action(policy, "pay now").allowed

    def test_exploration_uses_the_same_bar_as_execution(self):
        """Discovery must not perform what execution would be blocked from doing.

        Otherwise the crawl places the order and the generated test places it
        again, which is the duplicate-irreversible-action failure this whole
        mechanism exists to prevent.
        """
        policy = SafetyPolicy()
        assert not is_safe_to_explore(policy, "Place Order")
        assert is_safe_to_explore(policy, "Show more results")


class TestCategoryParsing:
    def test_comma_separated_names_parse(self):
        assert parse_categories("checkout, payment") == frozenset(
            {DestructiveCategory.CHECKOUT, DestructiveCategory.PAYMENT}
        )

    def test_hyphen_and_space_forms_normalise(self):
        assert parse_categories(["account-cancellation"]) == frozenset(
            {DestructiveCategory.ACCOUNT_CANCELLATION}
        )

    def test_unknown_names_are_ignored_rather_than_raising(self):
        """A typo must not take the service down, and must fail *closed*."""
        assert parse_categories("checkout,nonsense") == frozenset(
            {DestructiveCategory.CHECKOUT}
        )

    def test_empty_input_authorises_nothing(self):
        assert parse_categories("") == frozenset()


class TestGeneratorIntegration:
    @staticmethod
    def _checkout_flow() -> TestFlow:
        return TestFlow(
            id="F1",
            name="Checkout",
            steps=[
                TestStep(action="goto", target="https://shop.test/cart"),
                TestStep(
                    action="click",
                    target="the Place Order button",
                    description="click Place Order",
                ),
                TestStep(action="assert_text", target="Thank you", description="confirmed"),
            ],
        )

    def test_destructive_step_is_screened_before_code_exists(self):
        blocked, reasons = screen_flow_for_safety(self._checkout_flow(), SafetyPolicy())
        assert set(blocked) == {1}
        assert "checkout" in reasons[0]

    def test_assertions_are_never_destructive(self):
        """An assertion observes; it cannot change server state.

        Without this carve-out a step asserting "the Delete button is visible"
        would be blocked, which would make the agent unable to test that
        destructive controls are rendered correctly.
        """
        flow = TestFlow(
            id="F2",
            name="Delete button renders",
            steps=[
                TestStep(
                    action="assert_visible",
                    target="the Delete account button",
                    description="delete account control is visible",
                )
            ],
        )
        blocked, _ = screen_flow_for_safety(flow, SafetyPolicy())
        assert blocked == {}

    def test_blocked_step_compiles_to_an_explicit_failure(self):
        """The step must fail loudly, not vanish.

        A test that silently dropped its checkout step and then reported success
        would claim coverage of a flow nobody exercised - strictly worse than
        having no test at all.
        """
        flow = self._checkout_flow()
        blocked, _ = screen_flow_for_safety(flow, SafetyPolicy())
        validations = [
            SelectorValidation(
                step_index=1, intent="Place Order", chosen='page.get_by_role("button")', valid=True
            )
        ]
        source, warnings, blocking = compile_steps_to_source(
            flow, validations, blocked_steps=blocked
        )
        assert blocking is None
        assert "BLOCKED BY SAFE MODE" in source
        assert "raise AssertionError" in source
        assert any("was not executed" in w for w in warnings)

    def test_unblocked_flow_compiles_without_a_raise(self):
        flow = TestFlow(
            id="F3",
            name="Search",
            steps=[
                TestStep(action="goto", target="https://shop.test/"),
                TestStep(action="fill", target="the search box", value="python"),
            ],
        )
        blocked, _ = screen_flow_for_safety(flow, SafetyPolicy())
        validations = [
            SelectorValidation(
                step_index=1, intent="search", chosen='page.get_by_label("Search")', valid=True
            )
        ]
        source, _, blocking = compile_steps_to_source(flow, validations, blocked_steps=blocked)
        assert blocking is None
        assert "BLOCKED BY SAFE MODE" not in source


class TestDeterministicCompilationGate:
    """Simple flows must skip the model; ambiguous ones must not."""

    def test_fully_resolved_flow_is_simple(self):
        flow = TestFlow(
            id="F4",
            name="Login",
            steps=[
                TestStep(action="goto", target="https://app.test/login"),
                TestStep(action="fill", target="email", value="a@b.test"),
                TestStep(action="assert_visible", target="dashboard"),
            ],
        )
        validations = [
            SelectorValidation(step_index=1, intent="email", chosen="page.get_by_label('Email')", valid=True)
        ]
        assert is_simple_flow(flow, validations) is True

    def test_unresolved_locator_makes_a_flow_non_simple(self):
        """An unresolved locator is where a model's judgment actually earns its cost."""
        flow = TestFlow(
            id="F5",
            name="Login",
            steps=[
                TestStep(action="goto", target="https://app.test/login"),
                TestStep(action="fill", target="email", value="a@b.test"),
            ],
        )
        validations = [SelectorValidation(step_index=1, intent="email", valid=False)]
        assert is_simple_flow(flow, validations) is False

    def test_empty_flow_is_not_simple(self):
        assert is_simple_flow(TestFlow(id="F6", name="x"), []) is False


class TestDataMarkers:
    def test_marker_is_derived_from_the_run_id(self):
        marker = data_marker("run_abc123def456")
        assert marker.startswith("atoa")
        assert is_marked(f"Widget {marker}")

    def test_email_marking_preserves_the_address_shape(self):
        marked = mark_value("qa@example.test", "run_abc123", field_type="email")
        assert "@" in marked and marked.count("@") == 1
        assert is_marked(marked)

    @pytest.mark.parametrize("field_type", ["number", "date", "password", "tel"])
    def test_formats_the_marker_would_corrupt_are_left_alone(self, field_type):
        assert mark_value("12345", "run_abc123", field_type=field_type) == "12345"

    def test_plain_text_is_tagged(self):
        assert is_marked(mark_value("Acme Corp", "run_abc123"))
