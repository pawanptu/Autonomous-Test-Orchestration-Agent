"""Redirect-time re-checking and TLS posture at the browser boundary.

A pre-flight check on the operator-supplied URL is not sufficient. The browser
follows redirects on its own, so ``https://public.example`` can end up fetching
``http://127.0.0.1:9200``, and a crawled link can point anywhere. The guard
re-applies the target policy to every navigation request, which is where the
redirect hop is caught.

Playwright is stubbed. These tests are about the guard's decisions, and driving
a real browser to prove them would make the suite slow and network-dependent
without testing anything extra.
"""

from __future__ import annotations

import pytest

from browser.session import NavigationGuard, install_third_party_blocker
from config import Settings
from target_policy import TargetPolicy


class FakeRequest:
    def __init__(self, url: str, *, navigation: bool = True) -> None:
        self.url = url
        self._navigation = navigation

    def is_navigation_request(self) -> bool:
        return self._navigation


class FakeRoute:
    """Records whether the request was allowed through or aborted."""

    def __init__(self) -> None:
        self.continued = False
        self.aborted_with: str | None = None

    async def continue_(self) -> None:
        self.continued = True

    async def abort(self, reason: str = "failed") -> None:
        self.aborted_with = reason


class FakeContext:
    def __init__(self) -> None:
        self.handler = None

    async def route(self, pattern, handler):
        self.handler = handler


@pytest.fixture
def stub_dns(monkeypatch):
    def _install(mapping: dict[str, tuple[str, ...]]):
        def fake_resolve(host: str) -> tuple[str, ...]:
            import ipaddress

            text = (host or "").strip().lower().strip("[]")
            try:
                ipaddress.ip_address(text)
                return (text,)
            except ValueError:
                return mapping.get(text, ())

        monkeypatch.setattr("target_policy.resolve_host", fake_resolve)

    return _install


class TestNavigationGuard:
    async def test_a_public_navigation_is_allowed(self, stub_dns):
        stub_dns({"example.com": ("93.184.216.34",)})
        guard = NavigationGuard(TargetPolicy())
        route = FakeRoute()
        await guard.handle(route, FakeRequest("https://example.com/"))
        assert route.continued is True
        assert not guard.blocked

    async def test_a_redirect_to_loopback_is_aborted(self, stub_dns):
        """The redirect hop is a fresh navigation request, and is re-checked.

        This is the case a single pre-flight check cannot catch: the operator
        supplied a perfectly ordinary public URL.
        """
        stub_dns({"public.example": ("93.184.216.34",)})
        guard = NavigationGuard(TargetPolicy())
        route = FakeRoute()
        await guard.handle(route, FakeRequest("http://127.0.0.1:9200/_search"))
        assert route.continued is False
        assert route.aborted_with == "blockedbyclient"
        assert guard.blocked[0]["reason"] == "loopback-address"

    async def test_a_redirect_to_cloud_metadata_is_aborted(self):
        guard = NavigationGuard(TargetPolicy(allow_private=True))
        route = FakeRoute()
        await guard.handle(
            route, FakeRequest("http://169.254.169.254/latest/meta-data/")
        )
        assert route.aborted_with == "blockedbyclient"
        assert guard.blocked[0]["reason"] == "metadata-endpoint"

    async def test_dns_rebinding_on_a_later_hop_is_caught(self, stub_dns):
        stub_dns({"rebind.example": ("127.0.0.1",)})
        guard = NavigationGuard(TargetPolicy())
        route = FakeRoute()
        await guard.handle(route, FakeRequest("https://rebind.example/step2"))
        assert route.aborted_with == "blockedbyclient"

    async def test_subresources_are_not_gated(self, stub_dns):
        """Blocking images and XHR would break the rendering under test.

        A subresource cannot redirect the top-level document to a new origin,
        so it is not part of the SSRF surface this guard defends.
        """
        stub_dns({})
        guard = NavigationGuard(TargetPolicy())
        route = FakeRoute()
        await guard.handle(
            route, FakeRequest("http://10.0.0.5/logo.png", navigation=False)
        )
        assert route.continued is True

    async def test_private_targets_pass_once_overridden(self, stub_dns):
        stub_dns({})
        guard = NavigationGuard(TargetPolicy(allow_private=True))
        route = FakeRoute()
        await guard.handle(route, FakeRequest("http://127.0.0.1:3000/"))
        assert route.continued is True

    async def test_decisions_are_memoised_per_url(self, stub_dns, monkeypatch):
        """A twelve-page crawl of one host must not perform twelve lookups."""
        calls = {"count": 0}

        def counting_resolve(host: str) -> tuple[str, ...]:
            calls["count"] += 1
            return ("93.184.216.34",)

        monkeypatch.setattr("target_policy.resolve_host", counting_resolve)
        guard = NavigationGuard(TargetPolicy())
        for _ in range(5):
            await guard.allows("https://example.com/page")
        assert calls["count"] == 1

    async def test_blocked_audit_entries_are_credential_free(self):
        guard = NavigationGuard(TargetPolicy())
        await guard.allows("http://user:hunter2@169.254.169.254/?token=abcdefghij")
        assert "hunter2" not in str(guard.blocked)
        assert "abcdefghij" not in str(guard.blocked)


class TestTlsPosture:
    def test_certificate_errors_are_fatal_by_default(self):
        """A broken TLS chain on a test target may be an interception.

        Accepting it silently would mean typing a credential into whatever
        answered.
        """
        assert Settings().allow_insecure_tls is False

    def test_insecure_tls_is_opt_in(self):
        assert Settings(allow_insecure_tls=True).allow_insecure_tls is True

    def test_the_setting_reaches_the_browser_context(self):
        """``ignore_https_errors`` must follow the setting, not be hardcoded."""
        import inspect

        from browser.session import BrowserSession

        source = inspect.getsource(BrowserSession.new_context)
        assert '"ignore_https_errors": self.settings.allow_insecure_tls' in source

    def test_tls_posture_is_visible_in_the_feature_flags(self):
        """An operator must be able to see this is on from the run record."""
        assert Settings().feature_flags()["ALLOW_INSECURE_TLS"] is False


class TestThirdPartyBlocker:
    async def test_analytics_and_chat_widgets_are_blocked(self):
        """These are the dominant source of false-positive visual diffs."""
        context = FakeContext()
        blocked = await install_third_party_blocker(context)
        for url in (
            "https://www.google-analytics.com/collect",
            "https://static.hotjar.com/c/hotjar.js",
            "https://widget.intercom.io/widget/abc",
        ):
            route = FakeRoute()
            await context.handler(route, FakeRequest(url, navigation=False))
            assert route.aborted_with == "blockedbyclient", url
        assert len(blocked) == 3

    async def test_first_party_resources_are_untouched(self):
        context = FakeContext()
        await install_third_party_blocker(context)
        route = FakeRoute()
        await context.handler(
            route, FakeRequest("https://app.example.com/main.css", navigation=False)
        )
        assert route.continued is True

    async def test_extra_patterns_are_honoured(self):
        context = FakeContext()
        await install_third_party_blocker(context, extra=("noisy.example",))
        route = FakeRoute()
        await context.handler(
            route, FakeRequest("https://noisy.example/banner.gif", navigation=False)
        )
        assert route.aborted_with == "blockedbyclient"
