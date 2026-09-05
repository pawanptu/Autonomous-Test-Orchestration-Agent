"""Credential-safe authentication against the target application.

Contract
--------
* Credential values are read from :data:`security.SECRET_BOX` at the moment
  they are typed and are never copied into a log line, a return value, an
  exception message, a screenshot or the run state.
* Success or failure is reported as a boolean plus a *safe* explanation. The
  explanation is built from page structure ("an error banner is visible on the
  login page"), never from the submitted values.
* On success the browser context's ``storage_state`` is exported to a
  temporary file so that every later page and every generated test inherits
  the session **without ever seeing the credentials**. That file is deleted
  when the run ends (:meth:`graph.runtime.RunContext.close`).
* On failure the caller emits a high-confidence ``auth blocked`` decision
  event, stops exploring protected pages and continues with whatever is
  publicly reachable.

Token authentication is supported as an ``Authorization: Bearer`` header on the
browser context. That covers the common API-backed SPA case; it is stated as an
assumption in the report rather than silently guessed at, because a target that
expects the token in a cookie or in ``localStorage`` will not accept it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlparse

from logging_setup import get_logger
from security import Credentials, sanitize_url

log = get_logger("aivor.login")

LOGIN_LINK_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\blog\s?in\b", re.I),
    re.compile(r"\bsign\s?in\b", re.I),
    re.compile(r"\bauthenticat", re.I),
    re.compile(r"/login|/signin|/sign-in|/auth|/account/login", re.I),
)

LOGOUT_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\blog\s?out\b", re.I),
    re.compile(r"\bsign\s?out\b", re.I),
    re.compile(r"/logout|/signout|/sign-out", re.I),
)

ERROR_TEXT_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"invalid|incorrect|wrong|failed|not recognis|unauthori[sz]ed|try again", re.I),
)

COMMON_LOGIN_PATHS: tuple[str, ...] = (
    "/login",
    "/signin",
    "/sign-in",
    "/account/login",
    "/users/sign_in",
    "/auth/login",
)


@dataclass
class LoginForm:
    """A detected login form, described structurally (never by value)."""

    username_selector: str | None = None
    password_selector: str | None = None
    submit_selector: str | None = None
    form_index: int = 0
    url: str = ""

    def usable(self) -> bool:
        return bool(self.password_selector and self.username_selector)


@dataclass
class LoginResult:
    """Outcome of an authentication attempt. Safe to log and to serialise."""

    ok: bool
    method: str = "none"
    login_url: str | None = None
    error: str | None = None
    evidence: list[str] = field(default_factory=list)
    storage_state_path: str | None = None

    def as_safe_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "method": self.method,
            "login_url": sanitize_url(self.login_url or ""),
            "error": self.error,
            "evidence": self.evidence[:8],
        }


# --------------------------------------------------------------------------
# Detection
# --------------------------------------------------------------------------
async def detect_login_form(page: Any) -> LoginForm | None:
    """Find a username/password/submit triple on the current page.

    Returns CSS selectors rather than locators so the result is serialisable
    and can be re-resolved after a navigation.
    """
    script = """
    () => {
      const cssFor = (el) => {
        if (el.id) return `#${CSS.escape(el.id)}`;
        if (el.name) return `${el.tagName.toLowerCase()}[name="${CSS.escape(el.name)}"]`;
        const tid = el.getAttribute('data-testid');
        if (tid) return `[data-testid="${CSS.escape(tid)}"]`;
        const type = el.getAttribute('type');
        if (type) return `${el.tagName.toLowerCase()}[type="${CSS.escape(type)}"]`;
        return null;
      };
      const forms = Array.from(document.querySelectorAll('form'));
      const scopes = forms.length ? forms : [document.body];
      for (let i = 0; i < scopes.length; i++) {
        const scope = scopes[i];
        const pw = scope.querySelector('input[type="password"]');
        if (!pw) continue;
        const candidates = Array.from(
          scope.querySelectorAll('input[type="text"], input[type="email"], input:not([type]), input[type="tel"]')
        ).filter(el => (el.type || '').toLowerCase() !== 'hidden');
        const user = candidates[0] || null;
        const submit =
          scope.querySelector('button[type="submit"], input[type="submit"]') ||
          Array.from(scope.querySelectorAll('button')).find(
            b => /log ?in|sign ?in|submit|continue|enter/i.test(b.innerText || '')
          ) || null;
        return {
          username_selector: user ? cssFor(user) : null,
          password_selector: cssFor(pw),
          submit_selector: submit ? cssFor(submit) : null,
          form_index: i,
        };
      }
      return null;
    }
    """
    try:
        data = await page.evaluate(script)
    except Exception as exc:
        log.debug("login form detection failed: %s", exc)
        return None
    if not data or not data.get("password_selector"):
        return None
    form = LoginForm(
        username_selector=data.get("username_selector"),
        password_selector=data.get("password_selector"),
        submit_selector=data.get("submit_selector"),
        form_index=int(data.get("form_index", 0)),
        url=_safe_url(page),
    )
    return form


async def find_login_url(page: Any, base_url: str) -> str | None:
    """Locate the login page from the current page's links, then by convention."""
    try:
        hrefs = await page.evaluate(
            "() => Array.from(document.querySelectorAll('a[href]'))"
            ".map(a => [a.getAttribute('href') || '', (a.innerText || '').trim()])"
            ".slice(0, 300)"
        )
    except Exception:
        hrefs = []

    origin = _origin(base_url)
    for href, text in hrefs or []:
        blob = f"{href} {text}"
        if any(p.search(blob) for p in LOGIN_LINK_PATTERNS) and not any(
            p.search(blob) for p in LOGOUT_PATTERNS
        ):
            absolute = urljoin(base_url, href)
            if _origin(absolute) == origin:
                return absolute

    for path in COMMON_LOGIN_PATHS:
        candidate = urljoin(origin + "/", path.lstrip("/"))
        try:
            response = await page.request.get(candidate, timeout=8000)
            if response.ok:
                return candidate
        except Exception:
            continue
    return None


# --------------------------------------------------------------------------
# Login
# --------------------------------------------------------------------------
async def perform_login(
    page: Any,
    credentials: Credentials,
    *,
    base_url: str,
    action_timeout_ms: int = 8000,
    nav_timeout_ms: int = 25_000,
) -> LoginResult:
    """Authenticate the browser context. Returns a result that is safe to log.

    Strategy order:
      1. A supplied ``login_url``.
      2. A login form already present on the landing page.
      3. A discovered login link, or a conventional ``/login`` path.

    A bearer token, when supplied, is applied by the caller before this call
    (see :func:`apply_token_header`) and this function is skipped entirely
    unless a username/password pair is also present.
    """
    if not credentials.present():
        return LoginResult(ok=False, method="none", error="no credentials supplied")

    target = credentials.login_url or None
    if target:
        try:
            await page.goto(target, timeout=nav_timeout_ms, wait_until="domcontentloaded")
        except Exception as exc:
            return LoginResult(
                ok=False,
                method="form",
                login_url=target,
                error=f"could not open the supplied login_url ({type(exc).__name__})",
            )

    form = await detect_login_form(page)
    if form is None:
        discovered = target or await find_login_url(page, base_url)
        if discovered:
            try:
                await page.goto(discovered, timeout=nav_timeout_ms, wait_until="domcontentloaded")
                form = await detect_login_form(page)
                target = discovered
            except Exception as exc:
                return LoginResult(
                    ok=False,
                    method="form",
                    login_url=discovered,
                    error=f"could not open the discovered login page ({type(exc).__name__})",
                )

    if form is None or not form.usable():
        return LoginResult(
            ok=False,
            method="form",
            login_url=target,
            error="no login form with a password field could be located",
            evidence=["structural scan found no input[type=password] paired with a text input"],
        )

    login_url = _safe_url(page)
    evidence: list[str] = [f"login form detected at {sanitize_url(login_url)}"]

    try:
        if credentials.username:
            await page.locator(form.username_selector).first.fill(
                credentials.username, timeout=action_timeout_ms
            )
            evidence.append("username field populated")
        if credentials.password:
            await page.locator(form.password_selector).first.fill(
                credentials.password, timeout=action_timeout_ms
            )
            evidence.append("password field populated")
    except Exception as exc:
        # The exception message can echo the locator but never the value.
        return LoginResult(
            ok=False,
            method="form",
            login_url=login_url,
            error=f"could not fill the login form ({type(exc).__name__})",
            evidence=evidence,
        )

    try:
        if form.submit_selector:
            await page.locator(form.submit_selector).first.click(timeout=action_timeout_ms)
        else:
            await page.locator(form.password_selector).first.press("Enter")
        evidence.append("credentials submitted")
    except Exception as exc:
        return LoginResult(
            ok=False,
            method="form",
            login_url=login_url,
            error=f"could not submit the login form ({type(exc).__name__})",
            evidence=evidence,
        )

    try:
        await page.wait_for_load_state("networkidle", timeout=min(nav_timeout_ms, 12_000))
    except Exception:
        pass  # networkidle never settles on some SPAs; verification handles it.

    verdict = await verify_authenticated(page, login_url)
    evidence.extend(verdict["evidence"])
    if verdict["ok"]:
        return LoginResult(ok=True, method="form", login_url=login_url, evidence=evidence)
    return LoginResult(
        ok=False,
        method="form",
        login_url=login_url,
        error=verdict["reason"],
        evidence=evidence,
    )


async def verify_authenticated(page: Any, login_url: str) -> dict[str, Any]:
    """Decide whether login succeeded, using page structure only.

    Signals, in order of strength:
      * a password field is still on screen and the URL is unchanged -> failed
      * an error-shaped message is visible                            -> failed
      * a logout affordance appeared                                  -> succeeded
      * the URL moved off the login page                              -> succeeded
    """
    evidence: list[str] = []
    current = _safe_url(page)

    script = """
    () => {
      const visible = (el) => {
        const r = el.getBoundingClientRect();
        return r.width > 0 && r.height > 0;
      };
      const pw = Array.from(document.querySelectorAll('input[type="password"]')).filter(visible);
      const logout = Array.from(document.querySelectorAll('a, button')).some(el =>
        /log ?out|sign ?out/i.test((el.innerText || '') + ' ' + (el.getAttribute('href') || '')));
      const errorNodes = Array.from(document.querySelectorAll(
        '[role="alert"], .error, .alert, .alert-danger, .invalid-feedback, .form-error, .help-block'
      )).filter(visible).map(el => (el.innerText || '').trim().slice(0, 200)).filter(Boolean);
      const bodyText = (document.body ? document.body.innerText : '').slice(0, 4000);
      return { password_visible: pw.length > 0, logout_present: logout,
               errors: errorNodes.slice(0, 5), body_text: bodyText };
    }
    """
    try:
        probe = await page.evaluate(script)
    except Exception as exc:
        return {
            "ok": False,
            "reason": f"could not inspect the page after submit ({type(exc).__name__})",
            "evidence": evidence,
        }

    errors = [e for e in probe.get("errors", []) if any(p.search(e) for p in ERROR_TEXT_PATTERNS)]
    if errors:
        evidence.append(f"error message visible after submit: {errors[0][:120]!r}")
        return {"ok": False, "reason": "the application rejected the credentials", "evidence": evidence}

    if probe.get("logout_present"):
        evidence.append("a logout affordance is present, indicating an authenticated session")
        return {"ok": True, "reason": "", "evidence": evidence}

    if probe.get("password_visible") and _same_page(current, login_url):
        evidence.append("still on the login page with a visible password field")
        return {
            "ok": False,
            "reason": "still on the login page after submitting",
            "evidence": evidence,
        }

    if not _same_page(current, login_url):
        evidence.append(f"navigated away from the login page to {sanitize_url(current)}")
        return {"ok": True, "reason": "", "evidence": evidence}

    evidence.append("no logout affordance and no navigation; treating as not authenticated")
    return {"ok": False, "reason": "could not confirm an authenticated session", "evidence": evidence}


async def apply_token_header(context: Any, token: str) -> None:
    """Attach a bearer token to every request the context makes.

    Assumption, stated in the report: the target accepts
    ``Authorization: Bearer <token>``. Targets that expect a cookie or a
    ``localStorage`` entry will not be authenticated by this and the run will
    correctly report ``login_ok=False``.
    """
    await context.set_extra_http_headers({"Authorization": f"Bearer {token}"})


async def save_storage_state(context: Any, path: str | Path) -> str | None:
    """Export cookies + origin storage so later contexts inherit the session.

    The file contains session material, so it is written outside the reports
    tree, is git-ignored, and is unlinked when the run ends.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        await context.storage_state(path=str(target))
        return str(target)
    except Exception as exc:
        log.warning("could not export storage state: %s", exc)
        return None


# --------------------------------------------------------------------------
def _safe_url(page: Any) -> str:
    try:
        return page.url or ""
    except Exception:  # pragma: no cover
        return ""


def _origin(url: str) -> str:
    parts = urlparse(url)
    return f"{parts.scheme}://{parts.netloc}"


def _same_page(a: str, b: str) -> bool:
    if not a or not b:
        return False
    pa, pb = urlparse(a), urlparse(b)
    return (pa.netloc, pa.path.rstrip("/")) == (pb.netloc, pb.path.rstrip("/"))


def looks_like_logout(href: str, text: str = "") -> bool:
    """True for links that would destroy the session mid-crawl."""
    blob = f"{href} {text}"
    return any(p.search(blob) for p in LOGOUT_PATTERNS)
