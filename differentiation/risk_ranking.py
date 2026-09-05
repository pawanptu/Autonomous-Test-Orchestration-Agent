"""Business-risk classification for test flows.

Risk is assigned once, immediately after the plan is approved, and then carried
through every later stage: generation order, execution order, healing priority
and - most visibly - the ordering of the final report. A report sorted by flow
index tells a manager nothing; a report sorted by risk tells them what to look
at first.

Two implementations, one rubric
-------------------------------
The 70B model does the classification because "is this flow business-critical"
is a judgment call. But the rubric also exists as a deterministic keyword
classifier (:func:`classify_by_rubric`), which serves three purposes:

* it is the fallback when the model is unavailable, rate-limited, or returns a
  flow id that was not asked for;
* it back-fills any flow the model silently dropped, so the invariant
  "every flow has exactly one classification" always holds;
* it is unit-testable, which the model is not.

Because both paths read the same rubric text (:data:`llm.prompts.RISK_RUBRIC_TEXT`)
they cannot drift apart.
"""

from __future__ import annotations

import re
from typing import Any, Callable, Iterable, Sequence

from graph.state import RiskClassification, RiskLevel, TestFlow
from llm.client import LLMClient, ModelRole
from llm.json_utils import JSONParseError, coerce_list
from llm.prompts import RISK_SYSTEM, risk_user
from logging_setup import get_logger

log = get_logger("aivor.risk")

EmitFn = Callable[[str, str, str | None, float | None], None]
"""``(summary, detail, risk, confidence)``"""

# --------------------------------------------------------------------------
# The rubric, as matchable patterns. Each entry carries the citation that the
# rationale must quote, so a deterministic classification is as auditable as an
# LLM one.
# --------------------------------------------------------------------------
HIGH_RISK_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\bcheckout\b|\bplace (an? )?order\b|\border confirmation\b", re.I),
     "HIGH: checkout"),
    (re.compile(r"\bpayment\b|\bpay\b|\bcredit card\b|\bbilling\b|\bstripe\b|\bpaypal\b", re.I),
     "HIGH: payment"),
    (re.compile(r"\bcart\b|\bbasket\b|\bbag\b", re.I), "HIGH: cart persistence"),
    (re.compile(r"\b(log ?in|sign ?in|authenticat|session|credential)\w*\b", re.I),
     "HIGH: authentication"),
    (re.compile(r"\bsign ?up\b|\bregist(er|ration)\b|\bcreate (an )?account\b", re.I),
     "HIGH: signup"),
    (re.compile(r"\b(password|credential)\s+reset\b|\bforgot(ten)?\s+password\b"
                r"|\breset\b[^.]{0,30}\bpassword\b|\bchange (my )?password\b", re.I),
     "HIGH: password reset"),
    (re.compile(r"\bpii\b|\bpersonal (data|information)\b|\bssn\b|\baddress book\b", re.I),
     "HIGH: PII"),
    (re.compile(r"\bdelete\b|\bremove account\b|\bwipe\b"
                r"|\bcancel\b[^.]{0,30}\b(order|subscription|booking|reservation)\b", re.I),
     "HIGH: destructive action"),
    (re.compile(r"\blog ?out\b|\bsign ?out\b", re.I), "HIGH: authentication"),
)

MEDIUM_RISK_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\bsearch\b|\bquery\b|\blookup\b", re.I), "MEDIUM: search"),
    (re.compile(r"\bproduct detail\b|\bdetail page\b|\bitem page\b|\bview (the )?product\b", re.I),
     "MEDIUM: product detail"),
    (re.compile(r"\bfilter\b|\bsort\b|\bfacet\b|\bcategory\b|\bpaginat", re.I),
     "MEDIUM: filters"),
    (re.compile(r"\bprofile\b|\bsettings\b|\baccount details\b|\bpreferences\b", re.I),
     "MEDIUM: profile update"),
    (re.compile(r"\bform\b|\bsubmit\b|\bcontact us\b|\bnewsletter\b|\bsubscribe\b", re.I),
     "MEDIUM: non-payment form submit"),
    (re.compile(r"\breview\b|\brating\b|\bcomment\b", re.I), "MEDIUM: user-generated content"),
)

LOW_RISK_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\bfooter\b|\bheader\b(?!.*\bnav\w* to\b)", re.I), "LOW: footer"),
    (re.compile(r"\babout( us)?\b|\bcontact info\b|\bterms\b|\bprivacy polic", re.I),
     "LOW: about / static content"),
    (re.compile(r"\bblog\b|\barticle\b|\bnews\b|\bpress\b", re.I), "LOW: blog"),
    (re.compile(r"\btheme\b|\bdark mode\b|\blanguage (switch|toggle)\b", re.I),
     "LOW: theme toggle"),
    (re.compile(r"\bstatic\b|\bcosmetic\b|\blanding page renders\b|\bhomepage loads\b", re.I),
     "LOW: static content"),
    (re.compile(r"\bnavigat\w*\b|\bmenu\b|\bbreadcrumb\b", re.I), "LOW: cosmetic navigation"),
)

RISK_WEIGHT: dict[RiskLevel, float] = {
    RiskLevel.HIGH: 3.0,
    RiskLevel.MEDIUM: 2.0,
    RiskLevel.LOW: 1.0,
}


def _flow_text(flow: TestFlow | dict[str, Any]) -> str:
    """All the wording a rubric match may consider, as one lowercase blob."""
    if isinstance(flow, dict):
        name = flow.get("name", "")
        outcome = flow.get("expected_outcome", "")
        url = flow.get("url", "")
        hints = flow.get("business_hints", []) or []
        steps = flow.get("steps", []) or []
        step_text = " ".join(
            f"{s.get('target', '')} {s.get('description', '')} {s.get('value', '') or ''}"
            for s in steps
            if isinstance(s, dict)
        )
    else:
        name = flow.name
        outcome = flow.expected_outcome
        url = flow.url
        hints = list(flow.business_hints)
        step_text = " ".join(f"{s.target} {s.description} {s.value or ''}" for s in flow.steps)
    return " ".join([name, outcome, url, " ".join(hints), step_text])


def classify_by_rubric(flow: TestFlow | dict[str, Any]) -> RiskClassification:
    """Deterministic rubric application. Unknown defaults to MEDIUM.

    High patterns are checked first and win outright: a flow that touches both
    checkout and a footer link is a checkout flow.
    """
    flow_id = flow.get("id", "") if isinstance(flow, dict) else flow.id
    flow_name = flow.get("name", "") if isinstance(flow, dict) else flow.name
    blob = _flow_text(flow)

    for patterns, level in (
        (HIGH_RISK_PATTERNS, RiskLevel.HIGH),
        (MEDIUM_RISK_PATTERNS, RiskLevel.MEDIUM),
        (LOW_RISK_PATTERNS, RiskLevel.LOW),
    ):
        for pattern, citation in patterns:
            match = pattern.search(blob)
            if match:
                return RiskClassification(
                    flow_id=flow_id,
                    flow_name=flow_name,
                    risk=level,
                    rationale=(
                        f"Matched {match.group(0)!r} in the flow definition, which the "
                        f"rubric classifies as {citation}."
                    ),
                    rubric_cite=citation,
                    confidence=0.7 if level is RiskLevel.HIGH else 0.6,
                    source="rubric-fallback",
                )

    return RiskClassification(
        flow_id=flow_id,
        flow_name=flow_name,
        risk=RiskLevel.MEDIUM,
        rationale="No rubric keyword matched; the rubric defaults unknown flows to MEDIUM.",
        rubric_cite="UNKNOWN -> MEDIUM",
        confidence=0.4,
        source="default",
    )


def _coerce_level(value: Any) -> RiskLevel:
    text = str(value or "").strip().lower()
    if text in ("high", "critical", "h"):
        return RiskLevel.HIGH
    if text in ("low", "l", "minor"):
        return RiskLevel.LOW
    return RiskLevel.MEDIUM


async def rank_flows(
    llm: LLMClient,
    flows: Sequence[TestFlow],
    *,
    emit: EmitFn | None = None,
) -> list[RiskClassification]:
    """Classify every flow, LLM-first with rubric back-fill.

    Guarantees exactly one classification per flow, in the input order. A model
    failure degrades to the deterministic rubric rather than aborting the run;
    the ``source`` field records which path produced each verdict.
    """
    report = emit or (lambda summary, detail="", risk=None, confidence=None: None)
    if not flows:
        return []

    by_id: dict[str, RiskClassification] = {}

    try:
        payload = await llm.complete_json(
            ModelRole.REASONING,
            [
                {"role": "system", "content": RISK_SYSTEM},
                {"role": "user", "content": risk_user([f.model_dump(mode="json") for f in flows])},
            ],
            task="risk_ranking",
        )
        raw = coerce_list(payload, "classifications")
        known = {f.id: f for f in flows}
        for item in raw:
            if not isinstance(item, dict):
                continue
            flow_id = str(item.get("flow_id") or item.get("id") or "").strip()
            if flow_id not in known:
                continue
            by_id[flow_id] = RiskClassification(
                flow_id=flow_id,
                flow_name=known[flow_id].name,
                risk=_coerce_level(item.get("risk")),
                rationale=str(item.get("rationale") or "")[:500],
                rubric_cite=str(item.get("rubric_cite") or "")[:120],
                confidence=_clamp(item.get("confidence")),
                source="llm",
            )
    except (JSONParseError, Exception) as exc:  # noqa: B014 - deliberate broad catch
        log.warning("risk ranking via LLM failed, falling back to the rubric: %s", exc)
        report(
            "Risk ranking: model unavailable, applying the deterministic rubric",
            f"{type(exc).__name__}: {exc}",
            None,
            0.4,
        )

    classifications: list[RiskClassification] = []
    for flow in flows:
        verdict = by_id.get(flow.id)
        if verdict is None:
            verdict = classify_by_rubric(flow)
            log.info("flow %s back-filled by rubric -> %s", flow.id, verdict.risk.value)
        classifications.append(verdict)
        report(
            f"Risk: {flow.name} = {verdict.risk.value.upper()}",
            verdict.rationale,
            verdict.risk.value,
            verdict.confidence,
        )

    return classifications


def risk_map(classifications: Iterable[RiskClassification]) -> dict[str, RiskLevel]:
    return {c.flow_id: c.risk for c in classifications}


def risk_order(classifications: Sequence[RiskClassification]) -> list[str]:
    """Flow ids ordered high risk first, stable within a level."""
    indexed = list(enumerate(classifications))
    indexed.sort(key=lambda pair: (pair[1].risk.sort_key, pair[0]))
    return [c.flow_id for _, c in indexed]


def risk_summary(classifications: Sequence[RiskClassification]) -> dict[str, int]:
    counts = {level.value: 0 for level in RiskLevel}
    for item in classifications:
        counts[item.risk.value] += 1
    return counts


def _clamp(value: Any, default: float = 0.5) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return default
