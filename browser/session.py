"""Playwright lifecycle management shared by the crawler, generator and runner.

One :class:`BrowserSession` owns one Playwright driver and one browser process
for the whole run. Contexts are cheap and disposable; the browser is not, and
launching one per flow would dominate the pipeline's wall-clock time.

Authenticated state travels as a ``storage_state`` file rather than as
credentials: once :mod:`browser.login` has signed in, every later context is
created from that file and no downstream component ever needs the password.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator

from config import Settings, get_settings
from logging_setup import get_logger

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
            "ignore_https_errors": True,
            "locale": "en-US",
            "timezone_id": "UTC",
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
