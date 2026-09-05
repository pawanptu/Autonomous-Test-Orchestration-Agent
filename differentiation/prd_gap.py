"""PRD-to-test-plan gap analysis, behind ``ENABLE_PRD_GAP_ANALYSIS``.

The question this module answers is the one a product manager actually asks:
*"the agent wrote twelve tests - which of my written requirements are still
untested?"* Coverage rubrics judge the shape of a plan (does it have an edge
case, does it have an error state); this judges the plan against the
requirements the business wrote down.

Two matchers, one contract
--------------------------
The good matcher is semantic: embed every flow, embed every requirement, and
take the nearest flow. ChromaDB gives us that with a *local, in-memory*
client - no server, no hosted service, no paid API - but it is a heavyweight
optional dependency that is deliberately **not** in ``requirements.txt``. So it
is imported lazily inside a ``try/except`` and, when it is absent or misbehaves,
the module degrades to a deterministic stopworded token-overlap matcher.

That degradation is never silent. :func:`analyse_prd_gaps` returns the method it
actually used alongside the items, the caller stamps it into the report, and
:data:`METHOD_DESCRIPTIONS` carries a one-line honest explanation of what each
method can and cannot conclude. A keyword match is genuinely weaker than an
embedding match - it cannot see that "shopper can pay with a saved card" and
"checkout using the stored payment method" are the same requirement - and the
report says so rather than presenting both as equivalent evidence.

Honest limitations of both paths
--------------------------------
* This is a *similarity* signal, not a proof of coverage. A flow can be the
  nearest neighbour of a requirement and still assert nothing about it. Items
  are therefore reported as "covered / not covered **by similarity**".
* The optional ChromaDB path uses Chroma's bundled local embedding model, which
  Chroma fetches into its own cache the first time it is used. When that cache
  is cold and unavailable the call raises and we fall through to the keyword
  matcher, which needs nothing at all.
* PRD text is user-supplied, so every requirement string is routed through
  :func:`security.redact_text` before it is placed in a :class:`PRDGapItem`
  that will be serialised into the report.
"""

from __future__ import annotations

import re
from typing import Any, Sequence

from graph.state import PRDGapItem, TestFlow, TestPlan
from logging_setup import get_logger
from security import redact_text

log = get_logger("aivor.prd_gap")

# --------------------------------------------------------------------------
# Method labels. The caller records these verbatim in the report so a reader
# can tell a semantic verdict from a lexical one.
# --------------------------------------------------------------------------
METHOD_CHROMA: str = "chromadb-embeddings"
METHOD_KEYWORD: str = "keyword-overlap"
METHOD_DISABLED: str = "disabled"
METHOD_NO_PRD: str = "no-prd"
METHOD_NO_PLAN: str = "no-plan"

METHOD_DESCRIPTIONS: dict[str, str] = {
    METHOD_CHROMA: (
        "Requirements and flows were embedded with ChromaDB's local in-memory "
        "client and matched by vector similarity."
    ),
    METHOD_KEYWORD: (
        "ChromaDB is not installed, so requirements were matched to flows by "
        "deterministic stopworded token overlap. This misses paraphrases and "
        "synonyms; treat an uncovered verdict as a prompt to look, not as proof."
    ),
    METHOD_DISABLED: "PRD gap analysis is disabled (ENABLE_PRD_GAP_ANALYSIS=false).",
    METHOD_NO_PRD: "No PRD text was supplied, so there was nothing to compare the plan against.",
    METHOD_NO_PLAN: (
        "No test plan was available when the analysis ran, so every extracted "
        "requirement is reported as uncovered by construction."
    ),
}

# --------------------------------------------------------------------------
# Requirement extraction
# --------------------------------------------------------------------------
MAX_REQUIREMENTS: int = 60
"""Hard cap. A 200-page PRD would otherwise produce an unreadable report table
and, on the ChromaDB path, an embedding call per line."""

MIN_REQUIREMENT_CHARS: int = 25
"""Below this a line is a heading, a table cell or a fragment, not a statement."""

MAX_REQUIREMENT_CHARS: int = 300

DEFAULT_SIMILARITY_THRESHOLD: float = 0.25

_COLLECTION_NAME: str = "aivor_prd_gap_flows"
_MAX_DOC_CHARS: int = 2000

# Leading list markers: bullet glyphs, ``1.`` / ``1)`` / ``(1)`` numbering,
# multi-level ``3.2.1`` section numbers, and markdown task checkboxes. Bare
# numbers are only treated as markers when punctuation or a second level
# follows, so "100 concurrent users must be supported" keeps its "100".
_MARKER_RE = re.compile(
    r"^\s*(?:"
    r"[-*+•‣●◦⁃∙]\s*"
    r"|\[[ xX]?\]\s*"
    r"|\d{1,3}(?:\.\d{1,3})+\s+"
    r"|\d{1,3}[.)]\s+"
    r"|\(\d{1,3}\)\s*"
    r")"
)

_HEADING_RE = re.compile(r"^\s*#{1,6}\s+")
_TABLE_ROW_RE = re.compile(r"^\s*\|")
_MODAL_RE = re.compile(
    r"\b(?:must(?:\s+not)?|shall(?:\s+not)?|should(?:\s+not)?|will(?:\s+not)?"
    r"|needs?\s+to|is\s+able\s+to|are\s+able\s+to|has\s+to|have\s+to"
    r"|is\s+required\s+to|are\s+required\s+to)\b",
    re.IGNORECASE,
)
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")
_EMPHASIS_RE = re.compile(r"[*_`]{1,3}")
_WHITESPACE_RE = re.compile(r"\s+")

_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9'\-]*")

STOPWORDS: frozenset[str] = frozenset(
    {
        "the", "a", "an", "and", "or", "but", "if", "then", "than", "that", "this",
        "these", "those", "there", "here", "for", "with", "without", "from", "into",
        "onto", "upon", "about", "over", "under", "between", "within", "of", "to",
        "in", "on", "at", "by", "as", "is", "are", "was", "were", "be", "been",
        "being", "am", "do", "does", "did", "done", "can", "could", "may", "might",
        "must", "shall", "should", "will", "would", "have", "has", "had", "not",
        "no", "yes", "it", "its", "they", "them", "their", "we", "our", "us", "you",
        "your", "i", "he", "she", "his", "her", "who", "whom", "which", "what",
        "when", "where", "why", "how", "all", "any", "each", "every", "some", "such",
        "only", "own", "same", "so", "too", "very", "just", "also", "able", "via",
        "per", "e.g", "i.e", "etc", "shown", "given", "user", "users", "system",
        "application", "app", "page", "site", "website", "feature", "support",
        "supported", "provide", "provided", "allow", "allows", "allowed", "ensure",
        "ensures", "need", "needs", "required", "requirement", "requirements",
    }
)
"""Function words plus PRD boilerplate ("the system shall allow the user to...").

Leaving "system", "user" and "allow" in would make every requirement look like
every flow, which is exactly the failure mode that makes a naive keyword matcher
useless."""


def _stem(token: str) -> str:
    """Crude suffix stripping so "results" and "result" match.

    Deliberately not a real stemmer: both sides of every comparison go through
    the same function, so consistency matters far more than linguistic accuracy.
    """
    for suffix in ("ing", "ies", "ed", "es", "s"):
        if token.endswith(suffix) and len(token) > len(suffix) + 2:
            if suffix == "ies":
                return token[:-3] + "y"
            return token[: -len(suffix)]
    return token


def _tokens(text: str) -> set[str]:
    """Lowercase, stopworded, lightly stemmed content tokens."""
    out: set[str] = set()
    for raw in _TOKEN_RE.findall(text.lower()):
        token = raw.strip("'-")
        if not token or token in STOPWORDS:
            continue
        if len(token) < 3 and not token.isdigit():
            continue
        stemmed = _stem(token)
        if stemmed and stemmed not in STOPWORDS:
            out.add(stemmed)
    return out


def _clean_line(line: str) -> str:
    """Strip list markers, markdown emphasis and collapse whitespace."""
    text = _MARKER_RE.sub("", line)
    text = _EMPHASIS_RE.sub("", text)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text.strip(" -:;–—")


def split_requirements(prd_text: str) -> list[str]:
    """Split a PRD into candidate requirement statements.

    Three shapes are recognised, because that is how requirements are actually
    written: bullet lines, numbered/sectioned lines, and prose sentences that
    contain a modal ("must", "shall", "should", "will", "needs to", "is able
    to"). Everything else - headings, table rows, narrative filler - is dropped,
    because a heading is a topic and a topic cannot be covered or uncovered.

    The result is order-preserving, case-insensitively de-duplicated, free of
    fragments shorter than :data:`MIN_REQUIREMENT_CHARS`, and capped at
    :data:`MAX_REQUIREMENTS` items.
    """
    if not prd_text or not prd_text.strip():
        return []

    ordered: dict[str, str] = {}

    def add(candidate: str) -> None:
        text = candidate.strip()
        if len(text) < MIN_REQUIREMENT_CHARS:
            return
        if len(text) > MAX_REQUIREMENT_CHARS:
            text = text[: MAX_REQUIREMENT_CHARS - 1].rstrip() + "…"
        key = re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()
        if key and key not in ordered:
            ordered[key] = text

    for raw_line in prd_text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        if len(ordered) >= MAX_REQUIREMENTS:
            break
        line = raw_line.rstrip()
        if not line.strip():
            continue
        if _HEADING_RE.match(line) or _TABLE_ROW_RE.match(line):
            continue

        is_listed = bool(_MARKER_RE.match(line))
        cleaned = _clean_line(line)
        if not cleaned:
            continue

        if is_listed:
            add(cleaned)
            continue

        # Prose: keep only the sentences that state an obligation.
        for sentence in _SENTENCE_SPLIT_RE.split(cleaned):
            if _MODAL_RE.search(sentence):
                add(sentence.strip())

    requirements = list(ordered.values())[:MAX_REQUIREMENTS]
    log.debug("extracted %d candidate requirements from the PRD", len(requirements))
    return requirements


# --------------------------------------------------------------------------
# Deterministic matcher
# --------------------------------------------------------------------------
def keyword_overlap(requirement: str, flow_text: str) -> float:
    """Stopworded lexical similarity between a requirement and a flow, in 0..1.

    Not a pure Jaccard index. A flow document (name, outcome and every step) is
    an order of magnitude longer than a one-sentence requirement, so pure
    Jaccard would sit near zero for *every* pair and no threshold would separate
    anything. The score is therefore weighted mostly toward how much of the
    *requirement* the flow accounts for, with a Jaccard term retained so that a
    flow which mentions everything cannot claim to cover everything.
    """
    requirement_tokens = _tokens(requirement or "")
    flow_tokens = _tokens(flow_text or "")
    if not requirement_tokens or not flow_tokens:
        return 0.0
    shared = requirement_tokens & flow_tokens
    if not shared:
        return 0.0
    coverage = len(shared) / len(requirement_tokens)
    jaccard = len(shared) / len(requirement_tokens | flow_tokens)
    return round(max(0.0, min(1.0, 0.7 * coverage + 0.3 * jaccard)), 4)


def flow_document(flow: TestFlow) -> str:
    """The text that represents one flow to either matcher.

    Name, expected outcome, business hints and every step's description and
    target. The step targets are included deliberately: they carry the element
    nouns ("the Add to Basket button") that a requirement is most likely to
    share vocabulary with.
    """
    parts: list[str] = [flow.name or "", flow.expected_outcome or ""]
    parts.extend(hint for hint in flow.business_hints if hint)
    for step in flow.steps:
        parts.append(f"{step.description or ''} {step.target or ''}".strip())
    text = " ".join(part for part in parts if part).strip()
    return text[:_MAX_DOC_CHARS]


def _flow_label(flow: TestFlow) -> str:
    """Human-facing identity of a flow inside a gap item.

    The flow *name* is what a report reader recognises; the id is the fallback
    for a malformed plan. Redacted because flow names are model output derived
    from page text.
    """
    return redact_text(flow.name.strip() or flow.id or "unnamed flow")


# --------------------------------------------------------------------------
# Optional semantic matcher
# --------------------------------------------------------------------------
def _chroma_nearest(
    requirements: Sequence[str],
    flows: Sequence[TestFlow],
) -> list[tuple[int, float]] | None:
    """Nearest flow index and similarity per requirement, or ``None``.

    ``None`` means "this path is unavailable, use the deterministic matcher" -
    ChromaDB is not installed, the embedding model could not be prepared, or the
    client raised. Every failure mode is treated identically on purpose: this is
    an optional enhancement and it may never take a run down with it.
    """
    try:
        import chromadb  # noqa: PLC0415 - optional dependency, imported lazily by design
    except ImportError:
        log.info("chromadb is not installed; PRD gap analysis uses keyword overlap")
        return None
    except Exception as exc:  # pragma: no cover - broken install, bad wheel
        log.warning("chromadb could not be imported (%s); falling back to keyword overlap", exc)
        return None

    client = None
    try:
        client = chromadb.EphemeralClient()
        cosine_space = True
        try:
            collection = client.create_collection(
                name=_COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )
        except Exception:
            # Older/newer Chroma builds disagree about how the space is
            # configured. Fall back to the backend default and convert the
            # distance accordingly rather than reporting a wrong similarity.
            cosine_space = False
            collection = client.create_collection(name=_COLLECTION_NAME)

        documents: list[str] = []
        ids: list[str] = []
        index_by_id: dict[str, int] = {}
        for index, flow in enumerate(flows):
            document = flow_document(flow)
            if not document:
                continue
            doc_id = (flow.id or "").strip() or f"flow-{index}"
            if doc_id in index_by_id:  # ids must be unique; a duplicate plan id
                doc_id = f"{doc_id}-{index}"
            documents.append(document)
            ids.append(doc_id)
            index_by_id[doc_id] = index

        if not documents:
            return None

        collection.add(documents=documents, ids=ids)
        response = collection.query(query_texts=list(requirements), n_results=1)
    except Exception as exc:
        log.warning("chromadb matching failed (%s); falling back to keyword overlap", exc)
        return None
    finally:
        if client is not None:
            try:
                client.delete_collection(_COLLECTION_NAME)
            except Exception:  # pragma: no cover - ephemeral client, nothing persists
                log.debug("could not drop the ephemeral chroma collection", exc_info=True)

    result_ids = response.get("ids") or []
    result_distances = response.get("distances") or []
    out: list[tuple[int, float]] = []
    for position in range(len(requirements)):
        try:
            row_ids = result_ids[position] or []
            row_distances = result_distances[position] or []
            if not row_ids:
                out.append((-1, 0.0))
                continue
            flow_index = index_by_id.get(str(row_ids[0]), -1)
            distance = float(row_distances[0]) if row_distances else 1.0
            out.append((flow_index, _distance_to_similarity(distance, cosine_space)))
        except (IndexError, TypeError, ValueError):
            log.debug("unexpected chroma result row at %d", position, exc_info=True)
            out.append((-1, 0.0))
    return out


def _distance_to_similarity(distance: float, cosine_space: bool) -> float:
    """Convert a vector distance into a 0..1 similarity.

    With the cosine space explicitly configured the distance is ``1 - cos``, so
    the similarity is its complement. When the collection fell back to the
    backend's default metric we only know the distance is non-negative and
    smaller-is-better, so ``1 / (1 + d)`` is used: monotonic, bounded, and
    honest about being a rank rather than a calibrated score.
    """
    try:
        value = float(distance)
    except (TypeError, ValueError):
        return 0.0
    if value < 0.0:
        value = 0.0
    similarity = (1.0 - value) if cosine_space else (1.0 / (1.0 + value))
    return round(max(0.0, min(1.0, similarity)), 4)


# --------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------
def analyse_prd_gaps(
    *,
    prd_text: str | None,
    plan: TestPlan | None,
    enabled: bool,
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
) -> tuple[list[PRDGapItem], str]:
    """Match every extracted requirement to its nearest planned flow.

    Returns ``(items, method)`` where ``method`` is one of
    :data:`METHOD_CHROMA`, :data:`METHOD_KEYWORD`, :data:`METHOD_DISABLED`,
    :data:`METHOD_NO_PRD` or :data:`METHOD_NO_PLAN`. The caller is expected to
    put the method in the report next to the items, because "uncovered by
    keyword overlap" and "uncovered by embeddings" are different claims.

    ``METHOD_NO_PLAN`` still returns one item per requirement, all uncovered:
    with no plan nothing *is* covered, and the extracted requirement list is
    itself the useful output. A single malformed flow or requirement never
    aborts the analysis; it is logged and skipped.
    """
    if not enabled:
        return [], METHOD_DISABLED

    requirements = split_requirements(prd_text or "")
    if not requirements:
        return [], METHOD_NO_PRD

    flows = list(plan.flows) if plan is not None else []
    if not flows:
        items = [
            PRDGapItem(
                requirement=redact_text(text),
                covered=False,
                best_match_flow=None,
                similarity=0.0,
            )
            for text in requirements
        ]
        return items, METHOD_NO_PLAN

    try:
        threshold = max(0.0, min(1.0, float(similarity_threshold)))
    except (TypeError, ValueError):
        threshold = DEFAULT_SIMILARITY_THRESHOLD

    nearest = _chroma_nearest(requirements, flows)
    method = METHOD_CHROMA if nearest is not None else METHOD_KEYWORD

    documents: list[str] = []
    if nearest is None:
        for flow in flows:
            try:
                documents.append(flow_document(flow))
            except Exception:  # pragma: no cover - defensive, a bad flow is skipped
                log.debug("could not build a document for a flow", exc_info=True)
                documents.append("")

    items: list[PRDGapItem] = []
    for position, requirement in enumerate(requirements):
        try:
            if nearest is not None:
                flow_index, similarity = nearest[position]
            else:
                flow_index, similarity = _best_keyword_match(requirement, documents)

            best_flow = flows[flow_index] if 0 <= flow_index < len(flows) else None
            items.append(
                PRDGapItem(
                    requirement=redact_text(requirement),
                    covered=bool(best_flow is not None and similarity >= threshold),
                    best_match_flow=_flow_label(best_flow) if best_flow is not None else None,
                    similarity=round(float(similarity), 3),
                )
            )
        except Exception:  # one bad requirement must not kill the analysis
            log.warning("could not analyse a requirement; recording it as uncovered",
                        exc_info=True)
            items.append(
                PRDGapItem(
                    requirement=redact_text(requirement)[:MAX_REQUIREMENT_CHARS],
                    covered=False,
                    best_match_flow=None,
                    similarity=0.0,
                )
            )

    log.info(
        "PRD gap analysis: %d requirements, %d covered, method=%s",
        len(items),
        sum(1 for item in items if item.covered),
        method,
    )
    return items, method


def _best_keyword_match(requirement: str, documents: Sequence[str]) -> tuple[int, float]:
    """Index of and score for the best lexical match. ``(-1, 0.0)`` if none."""
    best_index = -1
    best_score = 0.0
    for index, document in enumerate(documents):
        score = keyword_overlap(requirement, document)
        if score > best_score:
            best_index = index
            best_score = score
    return best_index, best_score


def summarise_gaps(items: Sequence[PRDGapItem], method: str = "") -> dict[str, Any]:
    """Counts for the report header and the UI badge.

    ``method`` is passed in rather than derived, because it is a property of the
    *analysis* and not of the items: a list of zero items means something very
    different under ``no-prd`` than under ``keyword-overlap``.
    """
    total = len(items)
    covered = sum(1 for item in items if item.covered)
    return {
        "requirements": total,
        "covered": covered,
        "uncovered": total - covered,
        "coverage_pct": round((covered / total) * 100.0, 1) if total else 0.0,
        "method": method,
        "method_note": METHOD_DESCRIPTIONS.get(method, ""),
    }
