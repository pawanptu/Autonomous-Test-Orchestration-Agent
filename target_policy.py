"""Target admission policy: which URLs this agent is allowed to drive a browser at.

The agent takes a URL from an operator and then autonomously fetches it, follows
its redirects, and executes generated code against whatever answers. That makes
the target URL a server-side request forgery (SSRF) surface: an operator who
pastes ``http://169.254.169.254/latest/meta-data/iam/security-credentials/`` is
asking the agent to read cloud instance credentials and put them in a report.

This module is the single admission gate. Every entry point that turns a string
into navigation - the ``POST /run`` body, the CLI, the crawler's redirect
handling - routes through :func:`evaluate_target`.

What is enforced
----------------
* **Scheme.** Only ``http`` and ``https``. No ``file:``, ``data:``, ``gopher:``.
* **Name resolution.** The hostname is resolved to every address it answers
  with, and *each* address is classified. A public name that resolves to
  ``127.0.0.1`` (DNS rebinding) is blocked on the address, not the name.
* **Address class.** Loopback, private, link-local, reserved, multicast and
  unspecified addresses are refused unless the operator explicitly opts in.
* **Cloud metadata.** The well-known metadata addresses are refused
  unconditionally - see :data:`METADATA_ADDRESSES` for why there is no override.
* **Allowlist.** When an operator configures one, the host must match it. An
  allowlist narrows what is reachable; it never widens the address rules.

Redirects
---------
A single pre-flight check is not enough: ``https://public.example`` may 302 to
``http://127.0.0.1:9200``. :func:`evaluate_target` is therefore re-run on every
navigation and every redirect hop by :mod:`browser.session`, so the address
class is re-checked against the host actually being contacted.

Why metadata has no override
----------------------------
Every other block here has a legitimate use: testing a staging box on a private
VLAN, or an app on ``localhost:3000``, is ordinary work. Reading the cloud
metadata service is not a web-application test under any configuration, and the
data it returns is credential material. Allowing an override would create a
one-flag path from "operator pasted a URL" to "instance role keys in a report".
"""

from __future__ import annotations

import ipaddress
import socket
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Final
from urllib.parse import urlsplit

from logging_setup import get_logger
from security import sanitize_url

log = get_logger("aivor.target")

ALLOWED_SCHEMES: Final[frozenset[str]] = frozenset({"http", "https"})

# Cloud instance metadata services. These are link-local or private addresses
# that return credential material to any process that can reach them.
METADATA_ADDRESSES: Final[frozenset[str]] = frozenset(
    {
        "169.254.169.254",  # AWS IMDS, Azure IMDS, GCP, DigitalOcean, Oracle
        "169.254.170.2",  # AWS ECS task metadata
        "100.100.100.200",  # Alibaba Cloud
        "192.0.0.192",  # Oracle Cloud legacy
        "fd00:ec2::254",  # AWS IMDS over IPv6
    }
)

# Hostnames that front a metadata service. Blocked by name as well as by
# address, because the name may resolve differently inside a given VPC.
METADATA_HOSTNAMES: Final[frozenset[str]] = frozenset(
    {
        "metadata.google.internal",
        "metadata.goog",
        "instance-data",
        "metadata",
    }
)


class TargetBlocked(ValueError):
    """A URL was refused by the admission policy.

    Carries the structured :class:`TargetDecision` so callers can log the
    machine-readable reason code rather than re-parsing a message string.
    """

    def __init__(self, decision: TargetDecision) -> None:
        super().__init__(decision.detail)
        self.decision = decision

    @property
    def reason(self) -> str:
        return self.decision.reason


@dataclass(frozen=True)
class TargetPolicy:
    """The rules in force for one run.

    ``allow_private`` is the explicit operator override that makes
    ``http://localhost:3000`` testable. It deliberately does not cover cloud
    metadata endpoints.
    """

    allow_private: bool = False
    allow_insecure_tls: bool = False
    allowlist: tuple[str, ...] = ()
    resolve_dns: bool = True

    @classmethod
    def from_settings(cls, settings: object) -> TargetPolicy:
        """Build a policy from a :class:`config.Settings`-shaped object."""
        return cls(
            allow_private=bool(getattr(settings, "allow_private_targets", False)),
            allow_insecure_tls=bool(getattr(settings, "allow_insecure_tls", False)),
            allowlist=tuple(getattr(settings, "target_allowlist", ()) or ()),
            resolve_dns=bool(getattr(settings, "target_resolve_dns", True)),
        )


@dataclass(frozen=True)
class TargetDecision:
    """The outcome of evaluating one URL against one policy."""

    url: str
    allowed: bool
    reason: str
    detail: str
    host: str = ""
    scheme: str = ""
    addresses: tuple[str, ...] = ()
    category: str = "unknown"
    overridden: bool = False
    matched_allowlist: str = ""

    def audit(self) -> dict[str, object]:
        """Structured, credential-free record for the decision log.

        The URL is sanitised because a target URL is a routine accidental
        credential channel (``?token=...``, ``https://user:pass@host``).
        """
        return {
            "url": sanitize_url(self.url),
            "allowed": self.allowed,
            "reason": self.reason,
            "detail": self.detail,
            "host": self.host,
            "scheme": self.scheme,
            "addresses": list(self.addresses),
            "category": self.category,
            "overridden": self.overridden,
            "matched_allowlist": self.matched_allowlist,
        }


# --------------------------------------------------------------------------
# Address classification
# --------------------------------------------------------------------------
def classify_address(address: str) -> str:
    """Classify one IP literal into a policy category.

    Returns one of ``metadata``, ``loopback``, ``link-local``, ``private``,
    ``reserved``, ``multicast``, ``unspecified``, ``public`` or ``invalid``.
    Order matters: ``169.254.169.254`` is both link-local and a metadata
    endpoint, and must report as ``metadata`` so the stricter rule applies.
    """
    text = (address or "").strip().strip("[]")
    if not text:
        return "invalid"
    if text.lower() in METADATA_ADDRESSES:
        return "metadata"
    try:
        ip = ipaddress.ip_address(text)
    except ValueError:
        return "invalid"
    # An IPv4-mapped IPv6 address (::ffff:127.0.0.1) must be judged on the
    # embedded IPv4 address, not on the v6 wrapper, which reports as global.
    mapped = getattr(ip, "ipv4_mapped", None)
    if mapped is not None:
        return classify_address(str(mapped))
    if str(ip) in METADATA_ADDRESSES:
        return "metadata"
    if ip.is_unspecified:
        return "unspecified"
    if ip.is_loopback:
        return "loopback"
    if ip.is_link_local:
        return "link-local"
    if ip.is_multicast:
        return "multicast"
    # ``is_reserved`` is checked before ``is_private`` on purpose. Python counts
    # reserved space such as 240.0.0.0/4 as private too, and private is merely
    # overridable while reserved is refused outright. Testing private-first
    # would quietly make reserved space reachable behind ALLOW_PRIVATE_TARGETS.
    if ip.is_reserved:
        return "reserved"
    if ip.is_private:
        return "private"
    return "public"


#: Categories that require the ``allow_private`` override to be reachable.
OVERRIDABLE_CATEGORIES: Final[frozenset[str]] = frozenset(
    {"loopback", "private", "link-local", "unspecified"}
)

#: Categories that are never reachable, with or without an override.
HARD_BLOCKED_CATEGORIES: Final[frozenset[str]] = frozenset(
    {"metadata", "multicast", "reserved", "invalid"}
)


def resolve_host(host: str) -> tuple[str, ...]:
    """Every address ``host`` currently resolves to.

    An IP literal resolves to itself without touching DNS. A name that cannot
    be resolved returns an empty tuple, which the caller treats as a block:
    failing open on an unresolvable name would let a name that resolves only
    inside the target network through the gate.
    """
    text = (host or "").strip().strip("[]")
    if not text:
        return ()
    try:
        ipaddress.ip_address(text)
        return (text,)
    except ValueError:
        pass
    try:
        infos = socket.getaddrinfo(text, None, proto=socket.IPPROTO_TCP)
    except (socket.gaierror, UnicodeError, ValueError) as exc:
        log.info("could not resolve %r: %s", text, type(exc).__name__)
        return ()
    addresses: list[str] = []
    for info in infos:
        sockaddr = info[4]
        if sockaddr and isinstance(sockaddr[0], str) and sockaddr[0] not in addresses:
            addresses.append(sockaddr[0])
    return tuple(addresses)


# --------------------------------------------------------------------------
# Allowlist
# --------------------------------------------------------------------------
def _allowlist_match(host: str, allowlist: Iterable[str]) -> str:
    """Return the allowlist entry matching ``host``, or ``""``.

    Supported entry forms, all case-insensitive:

    * ``example.com``      - exact host match
    * ``*.example.com``    - any subdomain, and the apex itself
    * ``10.0.0.0/8``       - CIDR block, matched against an IP-literal host
    """
    target = (host or "").strip().lower().strip("[]")
    if not target:
        return ""
    for raw in allowlist:
        entry = (raw or "").strip().lower()
        if not entry:
            continue
        if entry.startswith("*."):
            suffix = entry[1:]  # ".example.com"
            if target == entry[2:] or target.endswith(suffix):
                return raw
            continue
        if "/" in entry:
            try:
                network = ipaddress.ip_network(entry, strict=False)
                if ipaddress.ip_address(target) in network:
                    return raw
            except ValueError:
                continue
            continue
        if target == entry:
            return raw
    return ""


# --------------------------------------------------------------------------
# The gate
# --------------------------------------------------------------------------
def evaluate_target(url: str, policy: TargetPolicy | None = None) -> TargetDecision:
    """Decide whether ``url`` may be contacted. Never raises.

    The returned decision is always safe to log: the URL is sanitised and no
    part of the credential surface is echoed.
    """
    rules = policy or TargetPolicy()
    raw = (url or "").strip()
    if not raw:
        return TargetDecision(
            url=raw,
            allowed=False,
            reason="empty-url",
            detail="no target URL was supplied",
        )

    try:
        parts = urlsplit(raw)
    except ValueError:
        return TargetDecision(
            url=raw,
            allowed=False,
            reason="malformed-url",
            detail="the target URL could not be parsed",
        )

    scheme = (parts.scheme or "").lower()
    if scheme not in ALLOWED_SCHEMES:
        return TargetDecision(
            url=raw,
            allowed=False,
            reason="scheme-not-allowed",
            detail=(
                f"scheme {scheme or '<none>'!r} is not testable; "
                "the target must be an http:// or https:// URL"
            ),
            scheme=scheme,
        )

    try:
        host = (parts.hostname or "").strip().lower()
    except ValueError:
        return TargetDecision(
            url=raw,
            allowed=False,
            reason="malformed-url",
            detail="the target URL has an unparsable host",
            scheme=scheme,
        )
    if not host:
        return TargetDecision(
            url=raw,
            allowed=False,
            reason="no-host",
            detail="the target URL must include a host, for example https://example.com",
            scheme=scheme,
        )

    if host in METADATA_HOSTNAMES:
        return TargetDecision(
            url=raw,
            allowed=False,
            reason="metadata-endpoint",
            detail=(
                f"{host!r} is a cloud instance metadata service. It returns credential "
                "material and is never a valid test target, so there is no override."
            ),
            host=host,
            scheme=scheme,
            category="metadata",
        )

    matched = _allowlist_match(host, rules.allowlist) if rules.allowlist else ""
    if rules.allowlist and not matched:
        return TargetDecision(
            url=raw,
            allowed=False,
            reason="not-allowlisted",
            detail=(
                f"{host!r} is not in the operator target allowlist "
                f"({len(rules.allowlist)} entr(y/ies) configured); "
                "add it to TARGET_ALLOWLIST to test it"
            ),
            host=host,
            scheme=scheme,
        )

    if not rules.resolve_dns:
        # Name resolution disabled: classify what we can from the literal alone
        # so an IP-literal target is still judged, and let names through.
        category = classify_address(host)
        if category == "invalid":
            category = "public"
        return _decide_on_categories(
            raw, host, scheme, (), {category}, rules, matched
        )

    addresses = resolve_host(host)
    if not addresses:
        return TargetDecision(
            url=raw,
            allowed=False,
            reason="unresolvable-host",
            detail=(
                f"{host!r} does not resolve to any address from this machine; "
                "refusing to navigate to a host that cannot be classified"
            ),
            host=host,
            scheme=scheme,
        )

    categories = {classify_address(address) for address in addresses}
    return _decide_on_categories(raw, host, scheme, addresses, categories, rules, matched)


def _decide_on_categories(
    url: str,
    host: str,
    scheme: str,
    addresses: tuple[str, ...],
    categories: set[str],
    rules: TargetPolicy,
    matched: str,
) -> TargetDecision:
    """Apply the address-class rules to the categories a host resolved into.

    Every resolved address must pass. A name answering with one public and one
    loopback address is blocked: which one a browser connects to is not ours to
    predict, and that ambiguity is exactly the DNS-rebinding attack.
    """
    hard = sorted(categories & HARD_BLOCKED_CATEGORIES)
    if hard:
        worst = hard[0]
        return TargetDecision(
            url=url,
            allowed=False,
            reason=f"{worst}-address" if worst != "metadata" else "metadata-endpoint",
            detail=(
                (
                    f"{host!r} is a {worst} address. "
                    if not addresses
                    else f"{host!r} resolves to a {worst} address ({', '.join(addresses)}). "
                )
                + (
                    "Cloud metadata services return credential material and are never "
                    "a valid test target, so there is no override."
                    if worst == "metadata"
                    else "This address class is not a testable web application."
                )
            ),
            host=host,
            scheme=scheme,
            addresses=addresses,
            category=worst,
            matched_allowlist=matched,
        )

    needs_override = sorted(categories & OVERRIDABLE_CATEGORIES)
    if needs_override:
        worst = needs_override[0]
        if not rules.allow_private:
            return TargetDecision(
                url=url,
                allowed=False,
                reason=f"{worst}-address",
                detail=(
                    f"{host!r} resolves to a {worst} address "
                    f"({', '.join(addresses) or host}). Local and private targets are "
                    "blocked by default; set ALLOW_PRIVATE_TARGETS=true (or pass "
                    "--allow-private) to test an application on this machine or LAN."
                ),
                host=host,
                scheme=scheme,
                addresses=addresses,
                category=worst,
                matched_allowlist=matched,
            )
        return TargetDecision(
            url=url,
            allowed=True,
            reason="allowed-by-override",
            detail=(
                f"{host!r} is a {worst} target, permitted because "
                "ALLOW_PRIVATE_TARGETS is enabled"
            ),
            host=host,
            scheme=scheme,
            addresses=addresses,
            category=worst,
            overridden=True,
            matched_allowlist=matched,
        )

    return TargetDecision(
        url=url,
        allowed=True,
        reason="ok",
        detail=f"{host!r} resolves to a public address",
        host=host,
        scheme=scheme,
        addresses=addresses,
        category="public",
        matched_allowlist=matched,
    )


def enforce_target(url: str, policy: TargetPolicy | None = None) -> TargetDecision:
    """Evaluate ``url`` and raise :class:`TargetBlocked` when it is refused."""
    decision = evaluate_target(url, policy)
    if not decision.allowed:
        raise TargetBlocked(decision)
    return decision


# --------------------------------------------------------------------------
# Audit logging
# --------------------------------------------------------------------------
def log_decision(decision: TargetDecision, *, context: str = "") -> None:
    """Emit the audit line for one admission decision.

    Blocks and overrides are logged at WARNING because both are security-
    relevant events an operator needs to see; ordinary public targets are
    logged at DEBUG so a normal run stays quiet.
    """
    where = f" [{context}]" if context else ""
    if not decision.allowed:
        log.warning(
            "target BLOCKED%s: %s (%s) host=%s addresses=%s",
            where,
            sanitize_url(decision.url),
            decision.reason,
            decision.host,
            ",".join(decision.addresses) or "-",
        )
    elif decision.overridden:
        log.warning(
            "target ALLOWED BY OVERRIDE%s: %s is %s (ALLOW_PRIVATE_TARGETS=true)",
            where,
            sanitize_url(decision.url),
            decision.category,
        )
    else:
        log.debug("target allowed%s: %s (%s)", where, sanitize_url(decision.url), decision.category)
