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
import re
from collections import deque
from typing import Any, Callable, Iterable
from urllib.parse import urldefrag, urljoin, urlparse

from browser.login import looks_like_logout
from browser.selectors import collect_inventory
from config import Settings, get_settings
from graph.state import DiscoveredPage, SiteMap
from logging_setup import get_logger
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


def normalize_url(url: str) -> str:
    """Canonical form used for the visited set.

    Drops the fragment and any trailing slash so that ``/about``, ``/about/``
    and ``/about#team`` are one page rather than three crawl budget entries.
    """
    if not url:
        return ""
    clean, _ = urldefrag(url)
    parts = urlparse(clean)
    path = parts.path.rstrip("/") or "/"
    netloc = parts.netloc.lower()
    query = parts.query
    return f"{parts.scheme}://{netloc}{path}" + (f"?{query}" if query else "")


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

    site = SiteMap(target_url=start_url, auth_blocked=auth_blocked)
    origin = f"{urlparse(start_url).scheme}://{urlparse(start_url).netloc}"

    queue: deque[tuple[str, int]] = deque([(start_url, 0)])
    visited: set[str] = set()
    signals: set[str] = set()

    while queue and len(site.pages) < cfg.crawl_max_pages:
        url, depth = queue.popleft()
        key = normalize_url(url)
        if not key or key in visited:
            continue
        visited.add(key)

        record = await _visit(page, url, depth, cfg)
        if record is None:
            site.notes.append(f"could not load {sanitize_url(url)}")
            continue

        site.pages.append(record)
        emit(
            f"Crawled {sanitize_url(record.url)} "
            f"({len(record.links)} links, {len(record.forms)} forms, depth {depth})",
            f"title={record.title!r} protected={record.is_protected}",
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
        for pattern, label in ECOMMERCE_PATTERNS:
            if pattern.search(blob):
                signals.add(label)

        if depth < cfg.crawl_max_depth:
            for href in record.links:
                absolute = urljoin(record.url, href)
                if normalize_url(absolute) in visited:
                    continue
                if len(visited) + len(queue) >= cfg.crawl_max_pages * 3:
                    break
                queue.append((absolute, depth + 1))

    site.ecommerce_signals = sorted(signals)
    if not site.pages:
        site.notes.append(
            "the crawl reached no pages at all; the target may be unreachable, "
            "blocked by a bot wall, or require credentials that were not supplied"
        )
    site.notes.append(
        f"crawl budget: {len(site.pages)}/{cfg.crawl_max_pages} pages, "
        f"depth<={cfg.crawl_max_depth}, origin={origin}"
    )
    return site


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
    return record


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
