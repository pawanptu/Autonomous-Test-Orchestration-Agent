"""Report ordering.

The report is ordered by business risk, and within a risk band by how bad the
outcome was. A report sorted by flow index or by pass/fail tells a manager
nothing; this ordering is the differentiating claim, so it is pinned down here.
"""

from __future__ import annotations

import pytest

from graph.state import FlowReportRow, RiskLevel
from graph.state import TestStatus as Status
from reports.generator import sort_flows_by_risk


def row(
    flow_id: str,
    risk: RiskLevel,
    status: Status = Status.PASSED,
    **kwargs,
) -> FlowReportRow:
    return FlowReportRow(
        flow_id=flow_id,
        flow_name=kwargs.pop("flow_name", f"Flow {flow_id}"),
        category=kwargs.pop("category", "happy_path"),
        risk=risk,
        status=status,
        **kwargs,
    )


class TestPrimaryRiskOrdering:
    def test_high_medium_low(self):
        rows = [
            row("F001", RiskLevel.LOW),
            row("F002", RiskLevel.HIGH),
            row("F003", RiskLevel.MEDIUM),
        ]
        assert [r.flow_id for r in sort_flows_by_risk(rows)] == ["F002", "F003", "F001"]

    def test_risk_beats_outcome_across_bands(self):
        # A passing HIGH flow still outranks a failing LOW flow: the reader
        # should see the risky area first regardless of this run's outcome.
        rows = [
            row("LOWFAIL", RiskLevel.LOW, Status.FAILED),
            row("HIGHPASS", RiskLevel.HIGH, Status.PASSED),
        ]
        assert [r.flow_id for r in sort_flows_by_risk(rows)] == ["HIGHPASS", "LOWFAIL"]

    def test_all_same_risk_preserves_input_order_for_equal_outcomes(self):
        rows = [row(f"F{i:03d}", RiskLevel.MEDIUM) for i in range(1, 6)]
        assert [r.flow_id for r in sort_flows_by_risk(rows)] == [
            "F001", "F002", "F003", "F004", "F005"
        ]


class TestSecondaryOutcomeOrdering:
    def test_within_a_band_the_worst_outcome_comes_first(self):
        rows = [
            row("PASS", RiskLevel.HIGH, Status.PASSED),
            row("VISUAL", RiskLevel.HIGH, Status.PASSED, visual_regression=True),
            row("HEALED", RiskLevel.HIGH, Status.HEALED, healed=True),
            row("REVIEW", RiskLevel.HIGH, Status.PASSED, needs_human_review=True),
            row("FAIL", RiskLevel.HIGH, Status.FAILED),
        ]
        ordered = [r.flow_id for r in sort_flows_by_risk(rows)]
        assert ordered == ["FAIL", "REVIEW", "HEALED", "VISUAL", "PASS"]

    def test_a_failure_that_also_needs_review_ranks_as_a_failure(self):
        # Both are rank 0: a failing test is a failing test, and the review flag
        # only breaks ties between non-failures. Input order then decides.
        rows = [
            row("REVIEW", RiskLevel.HIGH, Status.FAILED, needs_human_review=True),
            row("FAIL", RiskLevel.HIGH, Status.FAILED),
            row("PASS", RiskLevel.HIGH, Status.PASSED),
        ]
        assert [r.flow_id for r in sort_flows_by_risk(rows)] == ["REVIEW", "FAIL", "PASS"]

    def test_skipped_sorts_last_within_its_band(self):
        rows = [
            row("SKIP", RiskLevel.HIGH, Status.SKIPPED),
            row("PASS", RiskLevel.HIGH, Status.PASSED),
        ]
        assert [r.flow_id for r in sort_flows_by_risk(rows)] == ["PASS", "SKIP"]

    def test_error_ranks_with_failure(self):
        rows = [
            row("PASS", RiskLevel.MEDIUM, Status.PASSED),
            row("ERROR", RiskLevel.MEDIUM, Status.ERROR),
        ]
        assert [r.flow_id for r in sort_flows_by_risk(rows)][0] == "ERROR"

    def test_a_high_risk_failure_is_always_the_first_row(self):
        rows = [
            row("F001", RiskLevel.LOW, Status.FAILED),
            row("F002", RiskLevel.MEDIUM, Status.FAILED),
            row("F003", RiskLevel.HIGH, Status.PASSED),
            row("F004", RiskLevel.HIGH, Status.FAILED),
            row("F005", RiskLevel.MEDIUM, Status.PASSED),
        ]
        assert sort_flows_by_risk(rows)[0].flow_id == "F004"


class TestPurity:
    def test_does_not_mutate_the_input_list(self):
        rows = [row("F001", RiskLevel.LOW), row("F002", RiskLevel.HIGH)]
        original = list(rows)
        sort_flows_by_risk(rows)
        assert rows == original

    def test_returns_a_new_list(self):
        rows = [row("F001", RiskLevel.HIGH)]
        assert sort_flows_by_risk(rows) is not rows

    def test_every_row_survives_the_sort(self):
        rows = [row(f"F{i:03d}", RiskLevel.MEDIUM) for i in range(20)]
        assert len(sort_flows_by_risk(rows)) == 20

    def test_empty_input(self):
        assert sort_flows_by_risk([]) == []

    def test_single_row(self):
        rows = [row("F001", RiskLevel.HIGH)]
        assert [r.flow_id for r in sort_flows_by_risk(rows)] == ["F001"]

    def test_sort_is_idempotent(self):
        rows = [
            row("F001", RiskLevel.LOW, Status.FAILED),
            row("F002", RiskLevel.HIGH, Status.PASSED),
            row("F003", RiskLevel.MEDIUM, Status.HEALED, healed=True),
        ]
        once = sort_flows_by_risk(rows)
        assert [r.flow_id for r in sort_flows_by_risk(once)] == [r.flow_id for r in once]


class TestRenderingUsesTheOrder:
    """The renderers must not re-sort or the ordering claim is cosmetic."""

    def _report(self):
        from graph.state import FinalReport

        return FinalReport(
            run_id="run_test",
            target_url="https://example.com",
            executive_summary="test",
            flows=sort_flows_by_risk(
                [
                    row("LOWROW", RiskLevel.LOW, Status.PASSED, flow_name="Footer links"),
                    row("HIGHROW", RiskLevel.HIGH, Status.FAILED, flow_name="Checkout"),
                    row("MEDROW", RiskLevel.MEDIUM, Status.PASSED, flow_name="Search"),
                ]
            ),
        )

    def test_markdown_lists_high_risk_first(self):
        from reports.generator import render_markdown

        markdown = render_markdown(self._report())
        assert markdown.index("Checkout") < markdown.index("Search") < markdown.index("Footer")

    def test_html_lists_high_risk_first(self):
        from reports.generator import render_html

        html = render_html(self._report())
        assert html.index("Checkout") < html.index("Search") < html.index("Footer")

    def test_html_escapes_interpolated_text(self):
        from graph.state import FinalReport
        from reports.generator import render_html

        report = FinalReport(
            run_id="run_test",
            target_url="https://example.com",
            executive_summary="<script>alert('xss')</script>",
        )
        html = render_html(report)
        assert "<script>alert" not in html
        assert "&lt;script&gt;" in html
