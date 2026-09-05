"""Deterministic offline provider used only when ``LLM_OFFLINE_MODE=true``.

This is **not** the agent. It is a plumbing harness: it lets the LangGraph
wiring, the FastAPI layer, the Streamlit UI and the browser stack be exercised
end to end on a machine with no API key, and it lets the sanity tests assert
that the re-plan branch and the low-confidence healer branch actually route.

Every run made in this mode is stamped ``llm_provider = "offline-stub"`` in the
report and carries a loud limitation entry, so a stub run can never be mistaken
for a real one.

The stub deliberately returns an *incomplete* plan on the first planning call
and a complete one after the coverage gate sends feedback back, so that the
re-plan loop is visible in a keyless demo.
"""

from __future__ import annotations

import json
import re
from typing import Any, Sequence

from llm.client import LLMResponse, LLMUsage, ModelRole
from llm.json_utils import find_json_span

Message = dict[str, str]

STUB_MODEL = "offline-stub"


class OfflineStubProvider:
    """Answers prompts from canned templates, keyed off the system prompt."""

    name = "offline-stub"

    def __init__(self) -> None:
        self._planner_calls = 0

    def resolve_model(self, role: ModelRole) -> str:
        return f"{STUB_MODEL}-{role.value}"

    async def complete(
        self,
        messages: Sequence[Message],
        *,
        model: str,
        temperature: float,
        max_tokens: int,
        json_mode: bool,
    ) -> LLMResponse:
        system = next((m["content"] for m in messages if m.get("role") == "system"), "")
        user = "\n".join(m["content"] for m in messages if m.get("role") == "user")
        text = self._dispatch(system, user)
        return LLMResponse(
            text=text,
            model=model,
            provider=self.name,
            usage=LLMUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0),
            latency_s=0.0,
        )

    async def aclose(self) -> None:  # pragma: no cover - nothing to close
        return None

    # ------------------------------------------------------------------
    def _dispatch(self, system: str, user: str) -> str:
        if "You are the Planner sub-agent" in system:
            return self._plan(user)
        if "coverage-evaluation gate" in system:
            return self._coverage(user)
        if "risk-ranking stage" in system:
            return self._risk(user)
        if "You are the Generator sub-agent" in system:
            return self._generate(user)
        if "You are the Healer sub-agent" in system:
            return self._heal(user)
        if "You are the bug packager" in system:
            return self._bug(user)
        if "meta-orchestrator" in system:
            return json.dumps(
                {
                    "action": "proceed",
                    "confidence": 0.5,
                    "rationale": "offline stub always proceeds",
                    "feedback": "",
                }
            )
        if "reporting stage" in system:
            return json.dumps(
                {
                    "executive_summary": (
                        "This run was produced in LLM_OFFLINE_MODE. No model "
                        "reasoning took place; the pipeline plumbing was "
                        "exercised with deterministic stub responses."
                    ),
                    "business_impact": (
                        "Not applicable: offline stub run, for smoke-testing only."
                    ),
                    "limitations": [
                        "LLM_OFFLINE_MODE was enabled - every judgment in this "
                        "report came from a hard-coded stub, not from a model."
                    ],
                }
            )
        return json.dumps({"note": "offline stub has no handler for this prompt"})

    # ------------------------------------------------------------------
    @staticmethod
    def _site_map(user: str) -> dict[str, Any]:
        marker = "CRAWL RESULT (structure only):"
        if marker in user:
            span = find_json_span(user.split(marker, 1)[1])
            if span:
                try:
                    return json.loads(span)
                except json.JSONDecodeError:
                    pass
        return {}

    def _plan(self, user: str) -> str:
        site = self._site_map(user)
        target = site.get("target_url") or "http://localhost/"
        pages = site.get("pages") or [{"url": target, "title": "Home"}]
        login_detected = bool(site.get("login_detected"))
        is_revision = "REVISION REQUIRED" in user
        self._planner_calls += 1

        flows: list[dict[str, Any]] = []
        for index, page in enumerate(pages[:3], start=1):
            url = page.get("url", target)
            title = page.get("title") or f"Page {index}"
            flows.append(
                {
                    "id": f"F{index:03d}",
                    "name": f"Open {title} and confirm it renders",
                    "category": "happy_path",
                    "url": url,
                    "requires_auth": bool(page.get("is_protected")),
                    "business_hints": ["navigation"],
                    "expected_outcome": "the page loads and its main heading is visible",
                    "steps": [
                        {
                            "action": "goto",
                            "target": url,
                            "value": None,
                            "description": f"open {title}",
                        },
                        {
                            "action": "assert_visible",
                            "target": "the main heading",
                            "value": None,
                            "description": "the primary heading renders",
                        },
                    ],
                }
            )

        if is_revision:
            # Second pass: add the edge case and error state the gate demanded.
            flows.append(
                {
                    "id": f"F{len(flows) + 1:03d}",
                    "name": "Submit an empty search query",
                    "category": "edge_case",
                    "url": target,
                    "requires_auth": False,
                    "business_hints": ["search"],
                    "expected_outcome": "the page stays usable and does not error",
                    "steps": [
                        {"action": "goto", "target": target, "value": None,
                         "description": "open the landing page"},
                        {"action": "assert_visible", "target": "the page body",
                         "value": None, "description": "the page is still rendered"},
                    ],
                }
            )
            flows.append(
                {
                    "id": f"F{len(flows) + 1:03d}",
                    "name": "Request a URL that does not exist",
                    "category": "error_state",
                    "url": target.rstrip("/") + "/this-page-does-not-exist-xyz",
                    "requires_auth": False,
                    "business_hints": ["error handling"],
                    "expected_outcome": "a not-found page is shown rather than a blank screen",
                    "steps": [
                        {"action": "goto",
                         "target": target.rstrip("/") + "/this-page-does-not-exist-xyz",
                         "value": None, "description": "open a missing URL"},
                        {"action": "assert_visible", "target": "the page body",
                         "value": None, "description": "something is rendered"},
                    ],
                }
            )
            if login_detected:
                flows.append(
                    {
                        "id": f"F{len(flows) + 1:03d}",
                        "name": "Reject invalid credentials",
                        "category": "error_state",
                        "url": site.get("login_url") or target,
                        "requires_auth": False,
                        "business_hints": ["authentication"],
                        "expected_outcome": "an error message appears and the user stays signed out",
                        "steps": [
                            {"action": "goto", "target": site.get("login_url") or target,
                             "value": None, "description": "open the login page"},
                            {"action": "fill", "target": "the username field",
                             "value": "nobody@example.invalid", "description": "type a fake user"},
                            {"action": "fill", "target": "the password field",
                             "value": "definitely-not-the-password",
                             "description": "type a fake password"},
                            {"action": "click", "target": "the submit button",
                             "value": None, "description": "submit"},
                            {"action": "assert_visible", "target": "the error message",
                             "value": None, "description": "an error is shown"},
                        ],
                    }
                )

        return json.dumps(
            {
                "summary": "Offline stub plan derived from the crawl structure.",
                "auth_flow_present": login_detected,
                "ecommerce_like": bool(site.get("ecommerce_signals")),
                "discovered_areas": [p.get("title") or p.get("url") for p in pages[:5]],
                "flows": flows,
            }
        )

    def _coverage(self, user: str) -> str:
        has_error_state = '"error_state"' in user
        has_edge_case = '"edge_case"' in user
        passed = has_error_state and has_edge_case
        checks = [
            {"id": "C1", "requirement": "happy path per area", "satisfied": True,
             "evidence": "stub: happy paths present"},
            {"id": "C2", "requirement": "at least one edge case",
             "satisfied": has_edge_case,
             "evidence": "stub: edge_case category present" if has_edge_case else "stub: none found"},
            {"id": "C3", "requirement": "at least one error state",
             "satisfied": has_error_state,
             "evidence": "stub: error_state present" if has_error_state else "stub: none found"},
            {"id": "C4", "requirement": "auth happy path and invalid credentials",
             "satisfied": True, "evidence": "not applicable in stub mode"},
            {"id": "C5", "requirement": "cart/checkout coverage", "satisfied": True,
             "evidence": "not applicable in stub mode"},
            {"id": "C6", "requirement": "flows state expected outcomes", "satisfied": True,
             "evidence": "stub: every flow has expected_outcome"},
        ]
        missing = [c["requirement"] for c in checks if not c["satisfied"]]
        return json.dumps(
            {
                "passed": passed,
                "score": round(sum(1 for c in checks if c["satisfied"]) / len(checks), 2),
                "confidence": 0.5,
                "rationale": "Offline stub rubric check.",
                "checks": checks,
                "missing": missing,
                "feedback": (
                    ""
                    if passed
                    else "Add at least one edge_case flow and one error_state flow, "
                    "including a 404/empty-state check."
                ),
            }
        )

    def _risk(self, user: str) -> str:
        ids = re.findall(r'"id":\s*"([^"]+)"', user)
        classifications = []
        for flow_id in ids:
            block = user.split(f'"{flow_id}"', 1)[-1][:400].lower()
            if any(word in block for word in ("login", "auth", "checkout", "payment", "basket", "cart")):
                risk, cite = "high", "HIGH: authentication / checkout"
            elif any(word in block for word in ("search", "detail", "filter", "profile", "form")):
                risk, cite = "medium", "MEDIUM: search / detail / form"
            else:
                risk, cite = "low", "LOW: static content"
            classifications.append(
                {
                    "flow_id": flow_id,
                    "risk": risk,
                    "rationale": f"Offline stub keyword match -> {cite}.",
                    "rubric_cite": cite,
                    "confidence": 0.4,
                }
            )
        return json.dumps({"classifications": classifications})

    def _generate(self, user: str) -> str:
        # The stub returns an empty source so the Generator falls through to
        # its deterministic compiler, which is the honest path here: the stub
        # cannot write meaningful test code.
        return json.dumps(
            {
                "test_source": "",
                "notes": [
                    "offline stub produced no source; the deterministic "
                    "compiler fallback will render the validated steps"
                ],
            }
        )

    def _heal(self, user: str) -> str:
        """Structural heuristic mirroring :func:`agents.healer._fallback_classification`.

        Only two signals are read, both of them structural rather than
        semantic: an assertion mismatch on a page whose locators still resolve
        points at the application, and a timeout points at the script. Anything
        else is UNKNOWN, which routes to human review. This is the same logic
        the real Healer falls back to when the model is unreachable, so the
        keyless path exercises the same branches a real run would.
        """
        lowered = user.lower()
        timeout_like = "timeout" in lowered
        assertion_like = "assertionerror" in lowered or '"failure_kind": "assertion"' in lowered
        locators_resolve = '"present": true' in lowered

        if assertion_like and locators_resolve:
            classification, confidence = "GENUINE_DEFECT", 0.7
            rationale = (
                "Offline stub heuristic: the locators still resolve on the live page but "
                "an assertion did not hold, which points at the application rather than "
                "the script."
            )
        elif timeout_like:
            classification, confidence = "SCRIPT_ISSUE", 0.55
            rationale = "Offline stub heuristic: the failure is a timeout."
        else:
            classification, confidence = "UNKNOWN", 0.3
            rationale = "Offline stub cannot classify this failure from the available evidence."

        return json.dumps(
            {
                "classification": classification,
                "confidence": confidence,
                "rationale": rationale,
                "signals": {
                    "selector_present_in_dom": locators_resolve or None,
                    "failure_kind": (
                        "assertion" if assertion_like else "timeout" if timeout_like else "unknown"
                    ),
                    "spa_race_suspected": False,
                    "captcha_or_bot_wall": False,
                    "auth_wall": False,
                },
                "proposed_fix": None,
                "bug_title": "Expected content did not appear on the page under test",
                "bug_description": (
                    "## Summary\nAn assertion failed while the target locators were still "
                    "present on the page.\n\n*Produced in LLM_OFFLINE_MODE: the classification "
                    "is a structural heuristic, not a model judgment.*"
                ),
            }
        )

    def _bug(self, user: str) -> str:
        """Build a ticket from the flow and failure the packager handed us.

        A stub should never emit *worse* output than the structured data it was
        given. The prose here is mechanical rather than written, but the title,
        steps, expected and actual are all real values pulled out of the prompt,
        so a keyless demo still produces an actionable ticket.
        """
        flow = self._json_after(user, "FLOW:") or {}
        failure = self._json_after(user, "FAILURE:") or {}
        healer = self._json_after(user, "HEALER VERDICT:") or {}

        name = str(flow.get("name") or "an application flow")
        expected = str(flow.get("expected_outcome") or "the flow's stated outcome")
        actual = str(failure.get("error_message") or "the assertion did not hold").strip()
        risk_line = user.split("\n", 1)[0]
        high_risk = "high" in risk_line.lower()

        steps = []
        for index, step in enumerate(flow.get("steps") or [], start=1):
            if not isinstance(step, dict):
                continue
            description = step.get("description") or f"{step.get('action')} {step.get('target')}"
            steps.append(f"{description}".strip())
        if not steps:
            steps = [f"Open {flow.get('url') or 'the affected page'} and replay the flow"]

        title = f"{name}: expected outcome did not hold"[:90]
        description = (
            f"## Summary\n{name} does not produce its expected outcome.\n\n"
            f"**Expected.** {expected}\n\n"
            f"**Actual.** {actual[:400]}\n\n"
            f"**Classification.** {healer.get('classification', 'GENUINE_DEFECT')} at "
            f"confidence {healer.get('confidence', 0)}.\n\n"
            "*Ticket assembled in LLM_OFFLINE_MODE: the fields below are taken verbatim "
            "from the captured flow and failure, but the prose was not written by a model.*"
        )
        return json.dumps(
            {
                "title": title,
                "description": description,
                "steps_to_reproduce": steps[:12],
                "expected": expected,
                "actual": actual[:400],
                "severity": "critical" if high_risk else "major",
                "labels": ["bug", "auto-filed", "offline-stub"],
            }
        )

    @staticmethod
    def _json_after(text: str, marker: str) -> dict[str, Any] | None:
        """Parse the JSON block that follows ``marker`` in a prompt."""
        if marker not in text:
            return None
        span = find_json_span(text.split(marker, 1)[1])
        if not span:
            return None
        try:
            parsed = json.loads(span)
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, dict) else None
