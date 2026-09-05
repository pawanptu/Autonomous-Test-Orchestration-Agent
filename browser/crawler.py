"""Breadth-first exploration of the target application.

The crawler is the agent's only source of ground truth about the application.
Everything downstream - the plan, the coverage rubric's notion of "primary
areas", every resolved selector - is derived from what this module actually
observed in a real browser.

Boundaries, all enforced here rather than trusted to the LLM:

* same-origin only (configurable), depth-limited, page-count capped;
* logout links are never followed - one careless click would destroy the
  authenticated session mid-crawl;
* ``mailto:``, ``tel:``, ``javascript:``, fragment-only and obvious binary
  download links are skipped;
* when credentials were supplied, login happens **first**, so protected pages
  are reachable; when login fails the crawl continues over public pages and the
  site map is flagged ``auth_blocked`` for the report.
"""

from __future__ import annotations

import asyncio
import hashlib
import heapq
import itertools
import re
import time
from typing import Any, Callable
from urllib.parse import parse_qsl, urldefrag, urlencode, urljoin, urlparse

from browser.login import looks_like_logout
from browser.selectors import collect_inventory
from config import Settings, get_settings
from graph.state import DiscoveredPage, SiteMap
from logging_setup import get_logger
from safe_actions import SafetyPolicy, is_safe_to_explore
from security import sanitize_url

log = get_logger("aivor.crawler")

ProgressFn = Callable[[str, str], None]

SKIP_SCHEMES = ("mailto:", "tel:", "javascript:", "data:", "blob:", "sms:", "ftp:")
SKIP_EXTENSIONS = (
    ".pdf", ".zip", ".tar", ".gz", ".rar", ".7z", ".exe", ".dmg", ".pkg",
    ".mp4", ".mp3", ".avi", ".mov", ".wav", ".png", ".jpg", ".jpeg", ".gif",
    ".svg", ".ico", ".webp", ".css", ".js", ".json", ".xml", ".rss", ".woff",
    ".woff2", ".ttf", ".eot", ".csv", ".xlsx", ".doc", ".docx",
)

ECOMMERCE_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\badd to (cart|basket|bag)\b", re.I), "add-to-cart control"),
    (re.compile(r"\b(shopping )?(cart|basket)\b", re.I), "cart/basket surface"),
    (re.compile(r"\bcheckout\b", re.I), "checkout surface"),
    (re.compile(r"\b(payment|billing|credit card)\b", re.I), "payment surface"),
    (re.compile(r"\bplace (your )?order\b", re.I), "order placement"),
    (re.compile(r"[$£€]\s?\d+[.,]\d{2}", re.I), "priced items"),
    (re.compile(r"\bwishlist\b", re.I), "wishlist"),
    (re.compile(r"\bquantity\b", re.I), "quantity selector"),
)

LOGIN_WALL_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bplease (log|sign) ?in\b", re.I),
    re.compile(r"\byou must be logged in\b", re.I),
    re.compile(r"\bsession (has )?expired\b", re.I),
    re.compile(r"\b401\b|\bunauthori[sz]ed\b", re.I),
)


# Query parameters that identify a marketing campaign or a click, never a
# distinct page. Leaving them in the visited key makes the same page look new
# on every inbound link and burns the whole crawl budget on one document.
TRACKING_PARAMS: frozenset[str] = frozenset(
    {
        "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
        "utm_id", "utm_source_platform", "utm_creative_format",
        "gclid", "gbraid", "wbraid", "dclid", "fbclid", "msclkid", "twclid",
        "igshid", "mc_cid", "mc_eid", "_ga", "_gl", "yclid", "ttclid",
        "ref", "referrer", "referer", "source", "campaign",
        "s_kwcid", "ef_id", "trk", "trkCampaign", "sc_cid", "cmpid", "icid",
    }
)

# Parameters that genuinely select content and must survive canonicalisation.
# Pagination and filters produce distinct pages worth visiting; dropping them
# would collapse an entire catalogue into its first page.
MEANINGFUL_PARAMS: frozenset[str] = frozenset(
    {
        "page", "p", "offset", "start", "cursor", "limit", "per_page",
        "q", "query", "search", "keyword", "term",
        "sort", "order", "order_by", "direction",
        "filter", "category", "cat", "tag", "type", "status", "view",
        "id", "sku", "product", "slug", "lang", "locale",
    }
)


def canonicalize_url(url: str, *, drop_tracking: bool = True) -> str:
    """Canonical form used for the visited set and the site graph.

    Drops the fragment and any trailing slash so that ``/about``, ``/about/``
    and ``/about#team`` are one page rather than three crawl budget entries,
    lowercases the host, removes tracking parameters, and sorts the surviving
    query parameters so that ``?a=1&b=2`` and ``?b=2&a=1`` are one key.

    Unknown parameters are *kept*. A crawler that discarded every parameter it
    did not recognise would silently stop exploring any application whose
    routing is query-driven.
    """
    if not url:
        return ""
    clean, _ = urldefrag(url)
    parts = urlparse(clean)
    path = parts.path.rstrip("/") or "/"
    netloc = parts.netloc.lower()

    query = ""
    if parts.query:
        kept: list[tuple[str, str]] = []
        for key, value in parse_qsl(parts.query, keep_blank_values=True):
            if drop_tracking and key.lower() in TRACKING_PARAMS:
                continue
            kept.append((key, value))
        kept.sort()
        query = urlencode(kept)
    return f"{parts.scheme}://{netloc}{path}" + (f"?{query}" if query else "")


def normalize_url(url: str) -> str:
    """Backwards-compatible alias for :func:`canonicalize_url`.

    Retained because the memory keys, the regression radar and the visual
    baselines all persist values produced by this name across runs.
    """
    return canonicalize_url(url)


# Surfaces worth reaching first, highest value first. The crawl budget is small
# (12 pages by default) so the order in which links are dequeued decides what
# the agent is able to test at all. Authentication, checkout and account
# management carry the business risk; a press release does not.
SURFACE_PRIORITY: tuple[tuple[int, re.Pattern[str], str], ...] = (
    (
        100,
        re.compile(r"(?:^|/|\b)(login|log[\s-]?in|signin|sign[\s-]?in|auth|sso)(?:/|$|\?|\b)", re.I),
        "authentication",
    ),
    (
        95,
        re.compile(r"(?:^|/|\b)(register|signup|sign[\s-]?up|join|create[\s-]account)(?:/|$|\?|\b)", re.I),
        "registration",
    ),
    (
        90,
        re.compile(r"(?:^|/|\b)(checkout|check[\s-]out|payment|billing|place[\s-]order|purchase)(?:/|$|\?|\b)", re.I),
        "checkout",
    ),
    (
        85,
        re.compile(r"(?:^|/|\b)(cart|basket|add[\s-]to[\s-](cart|basket|bag))(?:/|$|\?|\b)", re.I),
        "cart",
    ),
    (
        80,
        re.compile(r"(?:^|/|\b)(account|profile|settings|preferences|dashboard|admin)(?:/|$|\?|\b)", re.I),
        "account",
    ),
    (
        75,
        re.compile(r"(?:^|/|\b)(search|find)(?:/|$|\?|\b)|[?&](q|query|search)=", re.I),
        "search",
    ),
    (
        65,
        re.compile(r"(?:^|/|\b)(products?|items?|detail|listings?|catalogue|catalog)(?:/|$|\?|\b)", re.I),
        "catalogue",
    ),
    (
        60,
        re.compile(r"(?:^|/|\b)(contact|support|help|feedback)(?:/|$|\?|\b)", re.I),
        "support",
    ),
    (
        30,
        re.compile(r"(?:^|/|\b)(blog|news|press|article|about|terms|privacy|legal)(?:/|$|\?|\b)", re.I),
        "content",
    ),
)


def surface_priority(url: str, link_text: str = "") -> tuple[int, str]:
    """Score a candidate URL by how much testable risk it is likely to carry.

    Returns the score and the label of the surface that matched. The link text
    is considered as well as the path, because single-page applications route
    through opaque paths where the only signal is what the link says.
    """
    blob = f"{url} {link_text}".strip()
    for score, pattern, label in SURFACE_PRIORITY:
        if pattern.search(blob):
            return score, label
    return 50, "general"


def same_origin(a: str, b: str) -> bool:
    pa, pb = urlparse(a), urlparse(b)
    return (pa.scheme, pa.netloc.lower()) == (pb.scheme, pb.netloc.lower())


def should_follow(href: str, base_url: str, *, same_origin_only: bool, link_text: str = "") -> bool:
    """Crawl-boundary decision for one discovered link."""
    if not href or href.strip().startswith("#"):
        return False
    lowered = href.strip().lower()
    if lowered.startswith(SKIP_SCHEMES):
        return False
    if looks_like_logout(href, link_text):
        return False
    absolute = urljoin(base_url, href)
    parts = urlparse(absolute)
    if parts.scheme not in ("http", "https"):
        return False
    if parts.path.lower().endswith(SKIP_EXTENSIONS):
        return False
    if same_origin_only and not same_origin(absolute, base_url):
        return False
    return True


async def crawl(
    page: Any,
    *,
    start_url: str,
    settings: Settings | None = None,
    progress: ProgressFn | None = None,
    auth_blocked: bool = False,
) -> SiteMap:
    """Breadth-first crawl from ``start_url``. Never raises.

    ``page`` must already carry the authenticated session when one exists, so
    that protected pages return their real content instead of a login wall.
    """
    cfg = settings or get_settings()
    emit = progress or (lambda summary, detail="": None)
    safety = SafetyPolicy.from_settings(cfg)

    site = SiteMap(target_url=start_url, auth_blocked=auth_blocked)
    origin = f"{urlparse(start_url).scheme}://{urlparse(start_url).netloc}"

    # Max-heap keyed by (-priority, depth, tiebreak): the highest-value surface
    # discovered so far is always visited next, and ties resolve shallow-first
    # so the crawl still fans out rather than tunnelling down one branch.
    counter = itertools.count()
    frontier: list[tuple[int, int, int, str, str]] = []
    heapq.heappush(frontier, (-100, 0, next(counter), start_url, "entry point"))

    visited: set[str] = set()
    signals: set[str] = set()
    surfaces: dict[str, str] = {}
    deadline = time.monotonic() + max(cfg.crawl_max_seconds, 1.0)
    budget_exhausted = ""

    while frontier and len(site.pages) < cfg.crawl_max_pages:
        if time.monotonic() > deadline:
            budget_exhausted = (
                f"discovery stopped after the {cfg.crawl_max_seconds:.0f}s time budget "
                f"with {len(site.pages)} page(s) mapped and {len(frontier)} still queued"
            )
            log.info(budget_exhausted)
            break

        neg_priority, depth, _, url, label = heapq.heappop(frontier)
        key = canonicalize_url(url)
        if not key or key in visited:
            continue
        visited.add(key)

        record = await _visit(page, url, depth, cfg)
        if record is None:
            site.notes.append(f"could not load {sanitize_url(url)}")
            continue
        surfaces[canonicalize_url(record.url)] = label

        site.pages.append(record)
        emit(
            f"Crawled {sanitize_url(record.url)} "
            f"({len(record.links)} links, {len(record.forms)} forms, depth {depth})",
            f"title={record.title!r} protected={record.is_protected} "
            f"surface={label} priority={-neg_priority}",
        )

        # Expand menus, tabs, accordions and dialogs so that single-page
        # applications reveal the controls a link-only crawl never sees. Every
        # candidate is screened by safe mode first, so discovery cannot place an
        # order that the generated test would then place a second time.
        if cfg.crawl_interact:
            revealed = await _expand_interactive(page, record, cfg, safety)
            if revealed:
                emit(
                    f"Revealed {len(revealed)} hidden control(s) on "
                    f"{sanitize_url(record.url)}",
                    "expanded menus/tabs/accordions in a disposable context",
                )

        # Login discovery: the first page with a password field wins.
        if not site.login_detected:
            for form in record.forms:
                if form.get("has_password"):
                    site.login_detected = True
                    site.login_url = record.url
                    emit(
                        f"Login form discovered at {sanitize_url(record.url)}",
                        "auth coverage becomes mandatory for the coverage gate",
                    )
                    break

        blob = " ".join(
            [record.text_excerpt]
            + [b.get("text", "") for b in record.buttons if isinstance(b, dict)]
            + record.links[:80]
        )
        for pattern, signal_label in ECOMMERCE_PATTERNS:
            if pattern.search(blob):
                signals.add(signal_label)

        if depth < cfg.crawl_max_depth:
            for href in record.links:
                absolute = urljoin(record.url, href)
                if canonicalize_url(absolute) in visited:
                    continue
                if len(visited) + len(frontier) >= cfg.crawl_max_pages * 3:
                    break
                if cfg.crawl_adaptive:
                    score, surface = surface_priority(absolute, record.link_text_for(href))
                else:
                    # Preserve the historical breadth-first ordering when the
                    # adaptive strategy is switched off: equal priority, so the
                    # heap degenerates to insertion order within a depth.
                    score, surface = 50, "general"
                heapq.heappush(
                    frontier, (-score, depth + 1, next(counter), absolute, surface)
                )

    site.ecommerce_signals = sorted(signals)
    site.surfaces = dict(sorted(surfaces.items()))
    if not site.pages:
        site.notes.append(
            "the crawl reached no pages at all; the target may be unreachable, "
            "blocked by a bot wall, or require credentials that were not supplied"
        )
    if budget_exhausted:
        site.notes.append(budget_exhausted)
    strategy = "adaptive (highest-risk surface first)" if cfg.crawl_adaptive else "breadth-first"
    site.notes.append(
        f"crawl budget: {len(site.pages)}/{cfg.crawl_max_pages} pages, "
        f"depth<={cfg.crawl_max_depth}, <={cfg.crawl_max_seconds:.0f}s, "
        f"origin={origin}, strategy={strategy}"
    )
    covered = sorted(set(surfaces.values()))
    if covered:
        site.notes.append("surfaces reached: " + ", ".join(covered))
    return site


# --------------------------------------------------------------------------
# SPA expansion
# --------------------------------------------------------------------------
#: Controls that reveal more of the page when clicked, without navigating.
_EXPANDABLE_JS = """
() => {
  const out = [];
  const sel = [
    '[aria-expanded="false"]',
    '[role="tab"]:not([aria-selected="true"])',
    'summary',
    'button[aria-haspopup]',
    '[data-toggle]', '[data-bs-toggle]',
    '.accordion-toggle', '.dropdown-toggle', '.menu-toggle', '.nav-toggle',
  ].join(',');
  const seen = new Set();
  document.querySelectorAll(sel).forEach((el, i) => {
    if (out.length >= 40) return;
    const rect = el.getBoundingClientRect();
    if (rect.width <= 0 || rect.height <= 0) return;
    const text = (el.innerText || el.getAttribute('aria-label') || '').trim().slice(0, 60);
    const key = text + '|' + el.tagName;
    if (seen.has(key)) return;
    seen.add(key);
    out.push({ index: i, text: text, tag: el.tagName.toLowerCase() });
  });
  return out;
}
"""


async def _expand_interactive(
    page: Any,
    record: DiscoveredPage,
    cfg: Settings,
    safety: SafetyPolicy,
) -> list[str]:
    """Click safe expanders and fold any newly revealed controls into ``record``.

    This runs on the crawl page, which is already a disposable context: the
    caller discards it after discovery, so state these clicks create does not
    leak into test execution.

    Only *expanding* controls are touched - things that reveal existing markup
    rather than submit anything - and each is additionally screened by safe
    mode. Anything that classifies as destructive is left alone and recorded.
    """
    revealed: list[str] = []
    try:
        candidates = await page.evaluate(_EXPANDABLE_JS)
    except Exception as exc:
        log.debug("expander discovery failed on %s: %s", sanitize_url(record.url), exc)
        return revealed

    before = {b.get("text", "") for b in record.buttons if isinstance(b, dict)}
    clicked = 0
    for candidate in candidates or []:
        if clicked >= cfg.crawl_max_interactions_per_page:
            break
        if not isinstance(candidate, dict):
            continue
        text = str(candidate.get("text") or "")
        if not is_safe_to_explore(safety, text):
            record.skipped_controls.append(f"{text[:60]} (destructive, not explored)")
            continue
        if looks_like_logout(text, text):
            record.skipped_controls.append(f"{text[:60]} (logout, not explored)")
            continue
        try:
            locator = page.locator(
                '[aria-expanded="false"], [role="tab"], summary, button[aria-haspopup], '
                "[data-toggle], [data-bs-toggle]"
            ).nth(int(candidate.get("index", 0)))
            await locator.click(timeout=1500, no_wait_after=True)
            clicked += 1
            await page.wait_for_timeout(120)
        except Exception:
            # A control that will not click is not an error: the page may have
            # re-rendered underneath us. Move on to the next candidate.
            continue

    if not clicked:
        return revealed

    try:
        inventory = await collect_inventory(page)
    except Exception as exc:
        log.debug("post-expansion inventory failed: %s", exc)
        return revealed

    for button in inventory.get("buttons", []):
        if not isinstance(button, dict):
            continue
        text = button.get("text", "")
        if text and text not in before:
            record.buttons.append(button)
            revealed.append(text)
    known_inputs = {(i.get("name"), i.get("label")) for i in record.inputs}
    for field in inventory.get("inputs", []):
        if isinstance(field, dict) and (field.get("name"), field.get("label")) not in known_inputs:
            record.inputs.append(field)
            revealed.append(field.get("label") or field.get("name") or "input")
    return revealed[:40]


async def _visit(page: Any, url: str, depth: int, cfg: Settings) -> DiscoveredPage | None:
    """Load one page and extract its structure. Returns ``None`` if unusable."""
    try:
        response = await page.goto(
            url, timeout=cfg.crawl_page_timeout_ms, wait_until="domcontentloaded"
        )
    except Exception as exc:
        log.info("skipping %s: %s", sanitize_url(url), type(exc).__name__)
        return None

    try:
        await page.wait_for_timeout(cfg.crawl_settle_ms)
    except Exception:
        pass

    record = DiscoveredPage(url=_current_url(page) or url, depth=depth)
    record.status = getattr(response, "status", None) if response is not None else None

    try:
        record.title = (await page.title() or "").strip()[:160]
    except Exception:
        record.title = ""

    inventory = await collect_inventory(page)
    record.inputs = [i for i in inventory.get("inputs", []) if isinstance(i, dict)]
    record.buttons = inventory.get("buttons", [])
    record.forms = inventory.get("forms", [])
    record.headings = [
        h.get("text", "") for h in inventory.get("headings", []) if isinstance(h, dict)
    ]

    try:
        links = await page.evaluate(
            "() => Array.from(document.querySelectorAll('a[href]'))"
            ".map(a => [a.getAttribute('href') || '', (a.innerText || '').trim().slice(0,60)])"
            ".slice(0, 200)"
        )
    except Exception:
        links = []

    kept: list[str] = []
    for href, text in links or []:
        if should_follow(href, record.url, same_origin_only=cfg.crawl_same_origin_only, link_text=text):
            absolute = urljoin(record.url, href)
            if absolute not in kept:
                kept.append(absolute)
                if text:
                    record.link_texts[absolute] = str(text)[:60]
    record.links = kept[:80]

    try:
        text = await page.evaluate("() => (document.body ? document.body.innerText : '')")
    except Exception:
        text = ""
    record.text_excerpt = " ".join((text or "").split())[:1500]

    record.is_protected = bool(
        any(p.search(record.text_excerpt) for p in LOGIN_WALL_PATTERNS)
        or (record.status is not None and record.status in (401, 403))
    )
    record.dom_fingerprint = fingerprint_page(record)
    return record


def fingerprint_page(record: DiscoveredPage) -> str:
    """Stable hash of a page's *interactive* structure.

    Deliberately built from the shape of the page - the identity of its form
    fields, its buttons and its headings - and not from its text. A catalogue
    page whose prices changed is the same page for caching purposes; a page
    that grew a new form field is not.

    Used to key the discovery and selector caches, and captured before selector
    validation so that a rerun can report "the page changed underneath us"
    instead of blaming the selector.
    """
    parts: list[str] = [canonicalize_url(record.url)]
    for field in record.inputs[:40]:
        if isinstance(field, dict):
            parts.append(
                "i:"
                + "|".join(
                    str(field.get(k) or "")
                    for k in ("type", "name", "id", "label", "placeholder")
                )
            )
    for button in record.buttons[:40]:
        if isinstance(button, dict):
            parts.append("b:" + str(button.get("text") or button.get("aria_label") or ""))
    for form in record.forms[:20]:
        if isinstance(form, dict):
            parts.append(f"f:{form.get('method')}|{form.get('field_count')}|{form.get('has_password')}")
    parts.extend(f"h:{h}" for h in record.headings[:20])
    digest = hashlib.sha256("\n".join(parts).encode("utf-8", "replace")).hexdigest()
    return digest[:16]


def _current_url(page: Any) -> str:
    try:
        return page.url or ""
    except Exception:  # pragma: no cover
        return ""


def summarise_for_prompt(site: SiteMap, max_pages: int = 10) -> dict[str, Any]:
    """Compact, prompt-safe projection of the site map.

    Full inventories are far too large for a prompt and mostly redundant; the
    Planner needs the shape of the app, not every attribute of every node.
    """
    pages: list[dict[str, Any]] = []
    for record in site.pages[:max_pages]:
        pages.append(
            {
                "url": sanitize_url(record.url),
                "title": record.title,
                "depth": record.depth,
                "status": record.status,
                "is_protected": record.is_protected,
                "headings": record.headings[:6],
                "forms": [
                    {
                        "method": f.get("method"),
                        "field_count": f.get("field_count"),
                        "has_password": f.get("has_password"),
                    }
                    for f in record.forms[:5]
                ],
                "inputs": [
                    {
                        "type": i.get("type"),
                        "name": i.get("name"),
                        "label": i.get("label"),
                        "placeholder": i.get("placeholder"),
                        "required": i.get("required"),
                    }
                    for i in record.inputs[:12]
                ],
                "buttons": [
                    (b.get("text") or b.get("aria_label") or "")[:60]
                    for b in record.buttons[:15]
                    if isinstance(b, dict)
                ],
                "text_excerpt": record.text_excerpt[:400],
            }
        )
    return {
        "target_url": sanitize_url(site.target_url),
        "page_count": len(site.pages),
        "login_detected": site.login_detected,
        "login_url": sanitize_url(site.login_url or "") or None,
        "auth_blocked": site.auth_blocked,
        "ecommerce_signals": site.ecommerce_signals,
        "pages": pages,
    }


def inventory_for_url(site: SiteMap, url: str) -> dict[str, Any]:
    """The crawler's element inventory for the page closest to ``url``.

    Selector resolution uses this to propose locators drawn from elements that
    genuinely exist rather than from guesses.
    """
    if not url:
        return {}
    target = normalize_url(url)
    for record in site.pages:
        if normalize_url(record.url) == target:
            return _inventory_of(record)
    # Fall back to the closest path prefix, then to the landing page.
    best: DiscoveredPage | None = None
    best_score = -1
    for record in site.pages:
        score = _common_prefix_len(normalize_url(record.url), target)
        if score > best_score:
            best, best_score = record, score
    return _inventory_of(best) if best else {}


def _inventory_of(record: DiscoveredPage) -> dict[str, Any]:
    return {
        "inputs": record.inputs,
        "buttons": record.buttons,
        "headings": [{"text": h} for h in record.headings],
        "forms": record.forms,
    }


def _common_prefix_len(a: str, b: str) -> int:
    count = 0
    for x, y in zip(a, b):
        if x != y:
            break
        count += 1
    return count
