"""The target admission policy is the agent's SSRF boundary.

The agent resolves an operator-supplied hostname and drives a real browser at
it, following redirects on its own. Everything in this file exists to pin the
rule that a URL is admitted on the *address it resolves to*, not on how it
looks, and that the one class of address which is never admissible - cloud
instance metadata - has no override at all.

DNS is stubbed throughout. A test that depended on live resolution would be a
test of the network, and would fail in CI for reasons that have nothing to do
with the policy.
"""

from __future__ import annotations

import ipaddress

import pytest

from target_policy import (
    METADATA_ADDRESSES,
    TargetBlocked,
    TargetPolicy,
    classify_address,
    enforce_target,
    evaluate_target,
    resolve_host,
)


@pytest.fixture
def stub_dns(monkeypatch):
    """Point hostnames at chosen addresses without touching the network."""

    def _install(mapping: dict[str, tuple[str, ...]]):
        def fake_resolve(host: str) -> tuple[str, ...]:
            text = (host or "").strip().lower().strip("[]")
            # An IP literal still resolves to itself, exactly as the real
            # function does; only name lookups are served from the mapping.
            try:
                ipaddress.ip_address(text)
                return (text,)
            except ValueError:
                return mapping.get(text, ())

        monkeypatch.setattr("target_policy.resolve_host", fake_resolve)

    return _install


class TestClassifyAddress:
    @pytest.mark.parametrize(
        "address,expected",
        [
            ("8.8.8.8", "public"),
            ("93.184.216.34", "public"),
            ("127.0.0.1", "loopback"),
            ("::1", "loopback"),
            ("10.0.0.5", "private"),
            ("172.16.4.2", "private"),
            ("192.168.1.1", "private"),
            ("169.254.10.1", "link-local"),
            ("0.0.0.0", "unspecified"),
            ("224.0.0.1", "multicast"),
            ("240.0.0.1", "reserved"),
            ("not-an-ip", "invalid"),
        ],
    )
    def test_classification(self, address, expected):
        assert classify_address(address) == expected

    @pytest.mark.parametrize("address", sorted(METADATA_ADDRESSES))
    def test_metadata_beats_every_other_class(self, address):
        """169.254.169.254 is also link-local; the stricter label must win."""
        assert classify_address(address) == "metadata"

    def test_ipv4_mapped_ipv6_is_judged_on_the_embedded_address(self):
        """``::ffff:127.0.0.1`` is loopback, however it is spelled.

        Left unhandled this is a complete policy bypass: the v6 wrapper reports
        as a global address while the packet goes to localhost.
        """
        assert classify_address("::ffff:127.0.0.1") == "loopback"
        assert classify_address("::ffff:10.0.0.1") == "private"


class TestSchemeAndShape:
    @pytest.mark.parametrize(
        "url", ["file:///etc/passwd", "gopher://x/", "data:text/html,x", "ftp://h/f"]
    )
    def test_non_http_schemes_are_refused(self, url):
        decision = evaluate_target(url)
        assert not decision.allowed
        assert decision.reason == "scheme-not-allowed"

    def test_missing_host_is_refused(self):
        assert evaluate_target("http:///path").reason == "no-host"

    def test_empty_url_is_refused(self):
        assert evaluate_target("").reason == "empty-url"


class TestPrivateAddressBlocking:
    @pytest.mark.parametrize(
        "url",
        [
            "http://127.0.0.1:3000/",
            "http://localhost:8000/",
            "http://10.0.0.5/admin",
            "http://192.168.1.10/",
            "http://[::1]:9200/",
        ],
    )
    def test_blocked_by_default(self, url, stub_dns):
        stub_dns({"localhost": ("127.0.0.1",)})
        decision = evaluate_target(url)
        assert not decision.allowed
        assert decision.reason.endswith("-address")

    @pytest.mark.parametrize(
        "url", ["http://127.0.0.1:3000/", "http://localhost:8000/", "http://10.0.0.5/"]
    )
    def test_allowed_with_explicit_override(self, url, stub_dns):
        stub_dns({"localhost": ("127.0.0.1",)})
        decision = evaluate_target(url, TargetPolicy(allow_private=True))
        assert decision.allowed
        assert decision.overridden is True
        assert decision.reason == "allowed-by-override"

    def test_public_target_needs_no_override(self, stub_dns):
        stub_dns({"example.com": ("93.184.216.34",)})
        decision = evaluate_target("https://example.com/app")
        assert decision.allowed
        assert decision.overridden is False
        assert decision.category == "public"


class TestMetadataEndpoints:
    @pytest.mark.parametrize(
        "url",
        [
            "http://169.254.169.254/latest/meta-data/iam/security-credentials/",
            "http://[fd00:ec2::254]/latest/",
            "http://100.100.100.200/",
        ],
    )
    def test_metadata_addresses_are_refused(self, url):
        assert not evaluate_target(url).allowed

    def test_metadata_hostname_is_refused(self):
        decision = evaluate_target("http://metadata.google.internal/computeMetadata/v1/")
        assert not decision.allowed
        assert decision.reason == "metadata-endpoint"

    def test_metadata_has_no_override(self):
        """The private-target override must not unlock credential endpoints.

        This is the single most important assertion in the file: it is the
        difference between "operator tested localhost" and "operator exfiltrated
        the instance role".
        """
        permissive = TargetPolicy(allow_private=True, allow_insecure_tls=True)
        decision = evaluate_target("http://169.254.169.254/latest/", permissive)
        assert not decision.allowed
        assert "no override" in decision.detail

    def test_public_name_resolving_to_metadata_is_refused(self, stub_dns):
        stub_dns({"innocent.example": ("169.254.169.254",)})
        decision = evaluate_target(
            "https://innocent.example/", TargetPolicy(allow_private=True)
        )
        assert not decision.allowed
        assert decision.reason == "metadata-endpoint"


class TestDnsRebinding:
    def test_public_name_resolving_to_loopback_is_refused(self, stub_dns):
        """The classic rebinding shape: the name is public, the address is not."""
        stub_dns({"rebind.example": ("127.0.0.1",)})
        decision = evaluate_target("https://rebind.example/")
        assert not decision.allowed
        assert decision.reason == "loopback-address"

    def test_mixed_answers_are_refused_on_the_worst_address(self, stub_dns):
        """One public and one private answer is still a block.

        Which address the browser picks is not ours to predict, so a name that
        can resolve to a private host is treated as one that will.
        """
        stub_dns({"mixed.example": ("93.184.216.34", "10.0.0.5")})
        decision = evaluate_target("https://mixed.example/")
        assert not decision.allowed
        assert decision.reason == "private-address"

    def test_unresolvable_host_fails_closed(self, stub_dns):
        stub_dns({})
        decision = evaluate_target("https://nonexistent.invalid/")
        assert not decision.allowed
        assert decision.reason == "unresolvable-host"


class TestAllowlist:
    def test_exact_host_matches(self, stub_dns):
        stub_dns({"app.example.com": ("93.184.216.34",)})
        policy = TargetPolicy(allowlist=("app.example.com",))
        assert evaluate_target("https://app.example.com/", policy).allowed

    def test_wildcard_matches_subdomain_and_apex(self, stub_dns):
        stub_dns(
            {"api.example.com": ("93.184.216.34",), "example.com": ("93.184.216.34",)}
        )
        policy = TargetPolicy(allowlist=("*.example.com",))
        assert evaluate_target("https://api.example.com/", policy).allowed
        assert evaluate_target("https://example.com/", policy).allowed

    def test_host_outside_the_allowlist_is_refused(self, stub_dns):
        stub_dns({"evil.test": ("93.184.216.34",)})
        policy = TargetPolicy(allowlist=("*.example.com",))
        decision = evaluate_target("https://evil.test/", policy)
        assert not decision.allowed
        assert decision.reason == "not-allowlisted"

    def test_allowlist_does_not_relax_address_rules(self, stub_dns):
        """An allowlist narrows what is reachable; it never widens it.

        Allowlisting a name that resolves to a private address must still
        require the private-target override, or the allowlist would become a
        second, quieter way to enable SSRF.
        """
        stub_dns({"internal.example.com": ("10.0.0.5",)})
        policy = TargetPolicy(allowlist=("*.example.com",))
        decision = evaluate_target("https://internal.example.com/", policy)
        assert not decision.allowed
        assert decision.reason == "private-address"

    def test_cidr_entry_matches_ip_literal(self):
        policy = TargetPolicy(allowlist=("10.0.0.0/8",), allow_private=True)
        assert evaluate_target("http://10.1.2.3/", policy).allowed


class TestEnforceAndAudit:
    def test_enforce_raises_with_the_decision_attached(self):
        with pytest.raises(TargetBlocked) as excinfo:
            enforce_target("http://169.254.169.254/")
        assert excinfo.value.reason == "metadata-endpoint"
        assert excinfo.value.decision.category == "metadata"

    def test_enforce_returns_the_decision_when_allowed(self, stub_dns):
        stub_dns({"example.com": ("93.184.216.34",)})
        assert enforce_target("https://example.com/").allowed

    def test_audit_record_never_leaks_credentials(self):
        """A target URL is a routine accidental credential channel."""
        decision = evaluate_target("http://user:hunter2@169.254.169.254/?token=abcdefgh")
        audit = decision.audit()
        assert "hunter2" not in str(audit)
        assert "abcdefgh" not in str(audit)

    def test_resolve_host_returns_ip_literals_unchanged(self):
        assert resolve_host("10.0.0.1") == ("10.0.0.1",)
        assert resolve_host("") == ()
