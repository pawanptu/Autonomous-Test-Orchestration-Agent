"""Playwright lifecycle management shared by the crawler, generator and runner.

One :class:`BrowserSession` owns one Playwright driver and one browser process
for the whole run. Contexts are cheap and disposable; the browser is not, and
launching one per flow would dominate the pipeline's wall-clock time.

Authenticated state travels as a ``storage_state`` file rather than as
credentials: once :mod:`browser.login` has signed in, every later context is
created from that file and no downstream component ever needs the password.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator

from config import Settings, get_settings
from logging_setup import get_logger
from security import sanitize_url
from target_policy import TargetPolicy, evaluate_target, log_decision

log = get_logger("aivor.session")

# A boring, stable UA. Some targets serve a different DOM to headless Chrome's
# default UA string, which would make selectors resolved during generation fail
# during execution.
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"
)

LAUNCH_ARGS = [
    "--disable-dev-shm-usage",
    "--no-sandbox",
    "--disable-blink-features=AutomationControlled",
]


class BrowserSession:
    """Owns the Playwright driver and the browser process for one run."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._playwright: Any | None = None
        self._browser: Any | None = None
        self.storage_state_path: str | None = None

    async def start(self) -> "BrowserSession":
        from playwright.async_api import async_playwright

        self._playwright = await async_playwright().start()
        self._browser = await self._playwright.chromium.launch(
            headless=self.settings.headless,
            slow_mo=self.settings.slow_mo_ms or 0,
            args=LAUNCH_ARGS,
        )
        log.info(
            "browser launched (headless=%s, viewport=%dx%d)",
            self.settings.headless,
            self.settings.viewport_width,
            self.settings.viewport_height,
        )
        return self

    async def stop(self) -> None:
        try:
            if self._browser is not None:
                await self._browser.close()
        except Exception:  # pragma: no cover - browser already gone
            log.debug("browser close failed", exc_info=True)
        finally:
            self._browser = None
        try:
            if self._playwright is not None:
                await self._playwright.stop()
        except Exception:  # pragma: no cover
            log.debug("playwright stop failed", exc_info=True)
        finally:
            self._playwright = None

    async def __aenter__(self) -> "BrowserSession":
        return await self.start()

    async def __aexit__(self, *exc_info: Any) -> None:
        await self.stop()

    # ------------------------------------------------------------------
    @property
    def browser(self) -> Any:
        if self._browser is None:
            raise RuntimeError("BrowserSession.start() has not been called")
        return self._browser

    async def new_context(self, *, use_storage_state: bool = True, **overrides: Any) -> Any:
        """Create a context, inheriting the authenticated session if we have one."""
        options: dict[str, Any] = {
            "viewport": {
                "width": self.settings.viewport_width,
                "height": self.settings.viewport_height,
            },
            "user_agent": DEFAULT_USER_AGENT,
            # Certificate errors are fatal unless the operator opted in. A test
            # target served over a broken TLS chain may be an interception, and
            # silently accepting it would mean typing credentials into it.
            "ignore_https_errors": self.settings.allow_insecure_tls,
            "locale": self.settings.visual_locale,
            "timezone_id": self.settings.visual_timezone,
            # Freeze the clock-driven parts of a page so visual baselines are
            # comparable between runs.
            "reduced_motion": "reduce",
        }
        if use_storage_state and self.storage_state_path and Path(self.storage_state_path).exists():
            options["storage_state"] = self.storage_state_path
        options.update(overrides)

        context = await self.browser.new_context(**options)
        context.set_default_timeout(self.settings.action_timeout_ms)
        context.set_default_navigation_timeout(self.settings.nav_timeout_ms)
        return context

    @asynccontextmanager
    async def page(self, *, use_storage_state: bool = True, **overrides: Any) -> AsyncIterator[Any]:
        """Yield a fresh page on a fresh context, closing both afterwards."""
        context = await self.new_context(use_storage_state=use_storage_state, **overrides)
        page = await context.new_page()
        try:
            yield page
        finally:
            try:
                await context.close()
            except Exception:  # pragma: no cover
                log.debug("context close failed", exc_info=True)


# --------------------------------------------------------------------------
# Target admission guard
# --------------------------------------------------------------------------
class NavigationGuard:
    """Re-applies the target policy to every navigation the browser attempts.

    A pre-flight check on the operator-supplied URL is not sufficient. The
    browser follows redirects on its own, so ``https://public.example`` may end
    up fetching ``http://127.0.0.1:9200``; a link on a crawled page may point
    anywhere; and a name may resolve differently on a later lookup. Each of
    those is a fresh navigation request, and each one is checked here.

    Only *navigation* requests are gated. Subresources (images, XHR, fonts) are
    left alone: blocking those would break the rendering the agent is there to
    observe, and they cannot redirect the top-level document to a new origin.

    Decisions are memoised per URL for the lifetime of the guard so that a
    crawl of 12 pages on one host performs one name resolution, not twelve.
    """

    def __init__(self, policy: TargetPolicy, *, context_label: str = "navigation") -> None:
        self.policy = policy
        self.context_label = context_label
        self.blocked: list[dict[str, Any]] = []
        self._cache: dict[str, bool] = {}
        self._lock = asyncio.Lock()

    async def allows(self, url: str) -> bool:
        """Whether ``url`` may be navigated to, resolving DNS off the event loop."""
        async with self._lock:
            cached = self._cache.get(url)
        if cached is not None:
            return cached
        decision = await asyncio.to_thread(evaluate_target, url, self.policy)
        log_decision(decision, context=self.context_label)
        if not decision.allowed:
            self.blocked.append(decision.audit())
        async with self._lock:
            self._cache[url] = decision.allowed
        return decision.allowed

    async def handle(self, route: Any, request: Any) -> None:
        """Playwright route handler: abort inadmissible navigations."""
        try:
            is_navigation = bool(request.is_navigation_request())
            url = request.url
        except Exception:  # pragma: no cover - request already torn down
            await route.continue_()
            return
        if not is_navigation:
            await route.continue_()
            return
        if await self.allows(url):
            await route.continue_()
            return
        log.warning(
            "aborting navigation to %s: refused by the target policy",
            sanitize_url(url),
        )
        await route.abort("blockedbyclient")


async def install_navigation_guard(
    context: Any,
    policy: TargetPolicy | None = None,
    *,
    settings: Settings | None = None,
    context_label: str = "navigation",
) -> NavigationGuard:
    """Attach a :class:`NavigationGuard` to every request on ``context``."""
    cfg = settings or get_settings()
    guard = NavigationGuard(policy or TargetPolicy.from_settings(cfg), context_label=context_label)
    await context.route("**/*", guard.handle)
    return guard


# Hosts and path fragments that serve analytics, advertising, session replay and
# chat widgets. They are the dominant source of false-positive visual diffs (a
# rotating ad creative changes every capture) and they slow every page load.
THIRD_PARTY_BLOCKLIST: tuple[str, ...] = (
    "google-analytics.com",
    "googletagmanager.com",
    "doubleclick.net",
    "googlesyndication.com",
    "googleadservices.com",
    "facebook.net",
    "facebook.com/tr",
    "connect.facebook",
    "hotjar.com",
    "hotjar.io",
    "fullstory.com",
    "mouseflow.com",
    "clarity.ms",
    "segment.com",
    "segment.io",
    "mixpanel.com",
    "amplitude.com",
    "intercom.io",
    "intercomcdn.com",
    "drift.com",
    "crisp.chat",
    "tawk.to",
    "zendesk.com/embeddable",
    "livechatinc.com",
    "sentry.io",
    "bugsnag.com",
    "newrelic.com",
    "nr-data.net",
    "optimizely.com",
    "adroll.com",
    "criteo.com",
    "taboola.com",
    "outbrain.com",
    "scorecardresearch.com",
    "quantserve.com",
)


async def install_third_party_blocker(
    context: Any, extra: tuple[str, ...] = ()
) -> list[str]:
    """Block analytics, ad and chat-widget traffic on ``context``.

    Returns the list of blocked URLs, which the visual report cites as evidence
    that a diff was compared against a page with the noisy surfaces removed.
    """
    blocked: list[str] = []
    patterns = THIRD_PARTY_BLOCKLIST + tuple(extra)

    async def _handle(route: Any, request: Any) -> None:
        try:
            url = request.url
        except Exception:  # pragma: no cover
            await route.continue_()
            return
        lowered = url.lower()
        if any(fragment in lowered for fragment in patterns):
            if len(blocked) < 200:
                blocked.append(url)
            await route.abort("blockedbyclient")
            return
        await route.continue_()

    await context.route("**/*", _handle)
    return blocked


def attach_console_capture(page: Any, sink: list[str], limit: int = 50) -> None:
    """Record console errors and page exceptions into ``sink``.

    Console output is a first-class defect signal: a flow can pass its DOM
    assertions while the app throws on every render. The Healer sees this list.
    """

    def _on_console(message: Any) -> None:
        try:
            if message.type in ("error", "warning") and len(sink) < limit:
                sink.append(f"[console.{message.type}] {str(message.text)[:300]}")
        except Exception:  # pragma: no cover
            pass

    def _on_page_error(error: Any) -> None:
        if len(sink) < limit:
            sink.append(f"[pageerror] {str(error)[:300]}")

    page.on("console", _on_console)
    page.on("pageerror", _on_page_error)
