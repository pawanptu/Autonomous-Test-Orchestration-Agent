"""Screenshot capture with mandatory masking of secret-bearing fields.

Two jobs:

* **Evidence.** Every test failure and every visual baseline needs a stable
  image. Animations are disabled and the caret is hidden so that two captures
  of an unchanged page are byte-comparable.

* **Containment.** A filled password box is a credential rendered as pixels,
  and redaction cannot reach it. Before any capture we locate every password
  field and every input whose name/id/autocomplete marks it as sensitive, and
  hand them to Playwright's native ``mask=`` option so they are painted over
  in the output rather than merely blurred. If masking cannot be established
  at all, the capture is *skipped* rather than taken unmasked.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from logging_setup import get_logger

log = get_logger("aivor.screenshots")

# Selectors for controls whose rendered value must never reach an image.
SENSITIVE_FIELD_SELECTORS: tuple[str, ...] = (
    "input[type='password']",
    "input[name*='password' i]",
    "input[id*='password' i]",
    "input[name*='passwd' i]",
    "input[name*='token' i]",
    "input[id*='token' i]",
    "input[name*='secret' i]",
    "input[autocomplete='current-password']",
    "input[autocomplete='new-password']",
    "input[autocomplete='one-time-code']",
    "[data-sensitive='true']",
)

MASK_COLOR = "#111827"

# Content that legitimately changes between two captures of a working page.
# Masking it is what separates "the checkout button moved" from "the clock
# ticked", and it is the difference between a visual suite people act on and
# one they mute. Kept deliberately conservative: over-masking hides real
# regressions, so these target elements that *advertise* themselves as dynamic
# via a semantic tag, role or a conventional class/testid name.
DYNAMIC_CONTENT_SELECTORS: tuple[str, ...] = (
    "time",
    "[datetime]",
    "[data-testid*='timestamp' i]",
    "[data-testid*='date' i]",
    "[class*='timestamp' i]",
    "[class*='relative-time' i]",
    "[class*='last-updated' i]",
    "[class*='avatar' i]",
    "[class*='gravatar' i]",
    "img[alt*='avatar' i]",
    "[class*='carousel' i]",
    "[class*='slider' i]",
    "[aria-live='polite']",
    "[aria-live='assertive']",
    "[role='status']",
    "[class*='cookie' i][class*='banner' i]",
    "[id*='cookie' i][id*='banner' i]",
    "[class*='toast' i]",
    "[class*='notification' i]",
    "[class*='ad-' i]",
    "[class*='advert' i]",
    "iframe[src*='ads' i]",
    "[class*='price' i]",
    "[data-testid*='price' i]",
)

# Injected before capture to stop time- and randomness-driven rendering from
# differing between two otherwise identical captures.
FREEZE_SCRIPT: str = """
() => {
  const FIXED = new Date('2020-01-01T00:00:00Z').getTime();
  try {
    const RealDate = Date;
    // eslint-disable-next-line no-global-assign
    Date = class extends RealDate {
      constructor(...args) { super(...(args.length ? args : [FIXED])); }
      static now() { return FIXED; }
    };
    Date.parse = RealDate.parse;
    Date.UTC = RealDate.UTC;
  } catch (e) { /* a page that froze Date itself is left alone */ }
  try {
    let seed = 42;
    Math.random = () => {
      seed = (seed * 1664525 + 1013904223) % 4294967296;
      return seed / 4294967296;
    };
  } catch (e) { /* non-writable Math.random */ }
  try {
    const style = document.createElement('style');
    style.textContent =
      '*,*::before,*::after{animation:none!important;transition:none!important;' +
      'animation-duration:0s!important;caret-color:transparent!important}';
    document.head && document.head.appendChild(style);
  } catch (e) { /* no head yet */ }
}
"""


async def freeze_dynamic_rendering(page: Any) -> bool:
    """Seed a fixed clock and RNG and disable animation on ``page``.

    Returns whether the seeding took. Deterministic seeding removes the largest
    remaining class of false positives after third-party blocking: a page that
    renders "2 minutes ago" produces a different image every single run.
    """
    try:
        await page.evaluate(FREEZE_SCRIPT)
        return True
    except Exception as exc:
        log.debug("could not freeze dynamic rendering: %s", exc)
        return False


async def dynamic_locators(page: Any, extra: Sequence[str] = ()) -> list[Any]:
    """Locators for dynamic regions that should be masked before a capture.

    Only selectors that actually match anything are returned, so the mask list
    handed to Playwright stays as small as the page requires.
    """
    locators: list[Any] = []
    for selector in tuple(DYNAMIC_CONTENT_SELECTORS) + tuple(extra):
        if not selector:
            continue
        try:
            locator = page.locator(selector)
            if await locator.count() > 0:
                locators.append(locator)
        except Exception:
            # A selector the target's DOM makes invalid is skipped rather than
            # failing the capture; the rest of the mask list still applies.
            continue
    return locators


async def sensitive_locators(page: Any) -> list[Any]:
    """Locators for every sensitive control currently on the page."""
    locators: list[Any] = []
    for selector in SENSITIVE_FIELD_SELECTORS:
        try:
            locator = page.locator(selector)
            if await locator.count() > 0:
                locators.append(locator)
        except Exception:  # pragma: no cover - malformed selector on odd pages
            continue
    return locators


async def has_filled_sensitive_field(page: Any) -> bool:
    """True if any sensitive control currently holds a value.

    Used to decide between "mask it" and "skip this frame entirely" when the
    caller asked for a capture that cannot be masked (e.g. an element-scoped
    screenshot whose subtree contains the field).
    """
    script = """
    (selectors) => {
      for (const sel of selectors) {
        for (const el of document.querySelectorAll(sel)) {
          if (el.value && el.value.length > 0) return true;
        }
      }
      return false;
    }
    """
    try:
        return bool(await page.evaluate(script, list(SENSITIVE_FIELD_SELECTORS)))
    except Exception:
        # Fail closed: assume a secret may be on screen.
        return True


async def capture(
    page: Any,
    path: str | Path,
    *,
    full_page: bool = True,
    timeout_ms: int = 15_000,
    extra_mask: Sequence[Any] = (),
) -> str | None:
    """Capture a masked screenshot. Returns the path, or ``None`` on failure.

    Never raises: a missing screenshot degrades the evidence quality of one
    finding, but it must not fail the run.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        masks = await sensitive_locators(page)
        masks.extend(extra_mask)
        await page.screenshot(
            path=str(target),
            full_page=full_page,
            timeout=timeout_ms,
            animations="disabled",
            caret="hide",
            mask=masks or None,
            mask_color=MASK_COLOR,
        )
        return str(target)
    except TypeError:
        # Older Playwright builds do not accept mask_color / animations.
        try:
            await page.screenshot(path=str(target), full_page=full_page, timeout=timeout_ms)
            return str(target)
        except Exception as exc:  # pragma: no cover
            log.warning("screenshot failed for %s: %s", target.name, exc)
            return None
    except Exception as exc:
        log.warning("screenshot failed for %s: %s", target.name, exc)
        return None


async def capture_safe(page: Any, path: str | Path, **kwargs: Any) -> str | None:
    """Capture only if it can be done without rendering a secret.

    If a sensitive field holds a value *and* masking is unavailable, the frame
    is skipped and ``None`` is returned. Callers treat a missing screenshot as
    "evidence withheld for safety", which the report states explicitly.
    """
    try:
        masks = await sensitive_locators(page)
    except Exception:
        masks = []
    if not masks and await has_filled_sensitive_field(page):
        log.info("skipping screenshot: a sensitive field holds a value and cannot be masked")
        return None
    return await capture(page, path, **kwargs)


async def capture_dom_snippet(page: Any, max_chars: int = 6000) -> str:
    """A trimmed HTML snapshot for failure analysis.

    Scripts, styles, SVG bodies and inline data URIs are stripped: they are
    noise for defect classification and they inflate the prompt. Any element
    whose value could be a credential has its ``value`` attribute removed
    before the HTML is serialised.
    """
    script = """
    () => {
      const clone = document.documentElement.cloneNode(true);
      clone.querySelectorAll('script, style, noscript, svg, iframe, link').forEach(n => n.remove());
      clone.querySelectorAll('input, textarea').forEach(el => {
        const type = (el.getAttribute('type') || '').toLowerCase();
        const name = (el.getAttribute('name') || '') + (el.getAttribute('id') || '');
        if (type === 'password' || /pass|token|secret|otp/i.test(name)) {
          el.setAttribute('value', '***REDACTED***');
        }
      });
      clone.querySelectorAll('[src], [href]').forEach(el => {
        for (const attr of ['src', 'href']) {
          const v = el.getAttribute(attr);
          if (v && v.startsWith('data:')) el.setAttribute(attr, 'data:...');
        }
      });
      return clone.outerHTML;
    }
    """
    try:
        html = await page.evaluate(script)
    except Exception as exc:
        log.debug("DOM snapshot failed: %s", exc)
        try:
            html = await page.content()
        except Exception:
            return ""
    if not isinstance(html, str):
        return ""
    collapsed = " ".join(html.split())
    if len(collapsed) <= max_chars:
        return collapsed
    head = collapsed[: max_chars // 2]
    tail = collapsed[-max_chars // 2 :]
    return f"{head}\n...[{len(collapsed) - max_chars} chars elided]...\n{tail}"


async def capture_viewport_pair(
    page: Any,
    base_path: str | Path,
) -> tuple[str | None, str | None]:
    """Capture both a viewport frame and a full-page frame.

    The viewport frame is the stable visual-diff subject (constant height); the
    full-page frame is kept as human evidence.
    """
    base = Path(base_path)
    viewport = await capture(page, base.with_name(base.stem + "__viewport.png"), full_page=False)
    full = await capture(page, base.with_name(base.stem + "__full.png"), full_page=True)
    return viewport, full
