"""Deterministic application of the risk rubric.

The rubric decides the order of the final report, so the mapping from flow
wording to HIGH/MEDIUM/LOW is the thing worth pinning down. The LLM path is not
tested here - it is not deterministic - but the fallback and back-fill logic
that guarantees "every flow has exactly one classification" is.
"""

from __future__ import annotations

import pytest

from differentiation.risk_ranking import (
    RISK_WEIGHT,
    classify_by_rubric,
    risk_map,
    risk_order,
    risk_summary,
)
from graph.state import FlowCategory, RiskClassification, RiskLevel
from graph.state import TestFlow as Flow
from graph.state import TestStep as Step


def flow(name: str, **kwargs) -> Flow:
    return Flow(
        id=kwargs.pop("id", "F001"),
        name=name,
        category=kwargs.pop("category", FlowCategory.HAPPY_PATH),
        steps=kwargs.pop("steps", [Step(action="goto", target="/")]),
        expected_outcome=kwargs.pop("expected_outcome", "something observable happens"),
        url=kwargs.pop("url", "https://example.com/"),
        business_hints=kwargs.pop("business_hints", []),
        **kwargs,
    )


class TestHighRisk:
    @pytest.mark.parametrize(
        "name",
        [
            "Complete checkout with a saved card",
            "Submit payment details",
            "Cart persists across a page reload",
            "Sign in with valid credentials",
            "Log in and reach the dashboard",
            "Create an account",
            "Reset a forgotten password",
            "Delete the user account",
            "Cancel an existing order",
            "Update stored personal information",
        ],
    )
    def test_money_auth_and_destructive_flows_are_high(self, name):
        assert classify_by_rubric(flow(name)).risk is RiskLevel.HIGH

    def test_rationale_cites_the_rubric(self):
        verdict = classify_by_rubric(flow("Complete checkout"))
        assert verdict.rubric_cite.startswith("HIGH:")
        assert "rubric" in verdict.rationale.lower()

    def test_high_wins_over_a_low_signal_in_the_same_flow(self):
        # A flow that touches both checkout and a footer link is a checkout flow.
        verdict = classify_by_rubric(flow("Reach checkout from the footer link"))
        assert verdict.risk is RiskLevel.HIGH


class TestMediumRisk:
    @pytest.mark.parametrize(
        "name",
        [
            "Search the catalogue for a title",
            "Open a product detail page",
            "Filter results by category",
            "Update the profile display name",
            "Submit the contact form",
        ],
    )
    def test_supporting_flows_are_medium(self, name):
        assert classify_by_rubric(flow(name)).risk is RiskLevel.MEDIUM


class TestLowRisk:
    @pytest.mark.parametrize(
        "name",
        [
            "Open the About page",
            "Read a blog article",
            "Toggle dark mode",
            "Check the footer links render",
        ],
    )
    def test_cosmetic_and_static_flows_are_low(self, name):
        assert classify_by_rubric(flow(name)).risk is RiskLevel.LOW


class TestDefaulting:
    def test_unrecognised_flow_defaults_to_medium(self):
        verdict = classify_by_rubric(flow("Zzyzx qwertyuiop"))
        assert verdict.risk is RiskLevel.MEDIUM
        assert verdict.source == "default"
        assert "MEDIUM" in verdict.rubric_cite

    def test_signal_in_steps_is_considered_not_just_the_name(self):
        verdict = classify_by_rubric(
            flow(
                "Flow seven",
                steps=[
                    Step(action="goto", target="/"),
                    Step(action="click", target="the Add to Basket button"),
                ],
            )
        )
        assert verdict.risk is RiskLevel.HIGH

    def test_signal_in_business_hints_is_considered(self):
        verdict = classify_by_rubric(flow("Flow eight", business_hints=["authentication"]))
        assert verdict.risk is RiskLevel.HIGH

    def test_accepts_a_plain_dict(self):
        verdict = classify_by_rubric(
            {"id": "F009", "name": "Pay for the order", "steps": [], "expected_outcome": ""}
        )
        assert verdict.risk is RiskLevel.HIGH
        assert verdict.flow_id == "F009"

    def test_fallback_source_is_labelled(self):
        assert classify_by_rubric(flow("Complete checkout")).source == "rubric-fallback"


class TestOrdering:
    def _classified(self) -> list[RiskClassification]:
        return [
            RiskClassification(flow_id="F001", risk=RiskLevel.LOW),
            RiskClassification(flow_id="F002", risk=RiskLevel.HIGH),
            RiskClassification(flow_id="F003", risk=RiskLevel.MEDIUM),
            RiskClassification(flow_id="F004", risk=RiskLevel.HIGH),
        ]

    def test_risk_order_puts_high_first_and_is_stable(self):
        assert risk_order(self._classified()) == ["F002", "F004", "F003", "F001"]

    def test_risk_map_keys_by_flow_id(self):
        assert risk_map(self._classified())["F002"] is RiskLevel.HIGH

    def test_summary_counts_each_level(self):
        assert risk_summary(self._classified()) == {"high": 2, "medium": 1, "low": 1}

    def test_sort_key_ranks_high_lowest(self):
        assert RiskLevel.HIGH.sort_key < RiskLevel.MEDIUM.sort_key < RiskLevel.LOW.sort_key

    def test_weights_are_monotonic(self):
        assert RISK_WEIGHT[RiskLevel.HIGH] > RISK_WEIGHT[RiskLevel.MEDIUM] > RISK_WEIGHT[RiskLevel.LOW]

    def test_empty_input_is_handled(self):
        assert risk_order([]) == []
        assert risk_summary([]) == {"high": 0, "medium": 0, "low": 0}
