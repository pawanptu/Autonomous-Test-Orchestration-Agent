"""Credentials must never be submitted over an unencrypted connection.

The rule: a username, password or bearer token may only travel to a plain
HTTP endpoint when that endpoint is a loopback/private/link-local host the
operator is deliberately pointing at for local development. Anywhere else,
plain HTTP is rejected before a browser ever types a password into a page or
an API request is accepted.
"""

from __future__ import annotations

import pytest

from security import insecure_for_credentials


class TestInsecureForCredentials:
    @pytest.mark.parametrize(
        "url",
        [
            "http://example.com/login",
            "http://app.example.com/",
            "http://8.8.8.8/login",  # public IP, not private
            "http://example.com:8080/account/login",
        ],
    )
    def test_public_plain_http_is_insecure(self, url):
        assert insecure_for_credentials(url) is True

    @pytest.mark.parametrize(
        "url",
        [
            "https://example.com/login",
            "https://app.example.com:8443/",
        ],
    )
    def test_https_is_always_safe(self, url):
        assert insecure_for_credentials(url) is False

    @pytest.mark.parametrize(
        "url",
        [
            "http://localhost:3000/",
            "http://127.0.0.1:8000/login",
            "http://[::1]:3000/",
            "http://192.168.1.20:5173/login",
            "http://10.0.0.5/app",
            "http://myhost.local/login",
        ],
    )
    def test_loopback_and_private_hosts_are_allowed_over_http(self, url):
        assert insecure_for_credentials(url) is False

    def test_empty_url_is_not_flagged(self):
        assert insecure_for_credentials("") is False

    def test_unparsable_url_fails_closed(self):
        assert insecure_for_credentials("http://[bad") is True


class TestRunRequestValidation:
    """Integration check: the API's request model enforces the same rule."""

    def _make(self, **kwargs):
        from api.models import CredentialsIn, RunRequest

        return RunRequest(**kwargs)

    def test_credentials_over_plain_http_public_host_rejected(self):
        from pydantic import ValidationError

        from api.models import CredentialsIn

        with pytest.raises(ValidationError, match="plain http"):
            self._make(
                url="http://example.com/",
                credentials=CredentialsIn(username="qa", password="hunter22"),
            )

    def test_credentials_over_https_accepted(self):
        from api.models import CredentialsIn

        request = self._make(
            url="https://example.com/",
            credentials=CredentialsIn(username="qa", password="hunter22"),
        )
        assert request.credentials.present()

    def test_credentials_over_localhost_http_accepted(self):
        from api.models import CredentialsIn

        request = self._make(
            url="http://localhost:3000/",
            credentials=CredentialsIn(username="qa", password="hunter22"),
        )
        assert request.credentials.present()

    def test_insecure_login_url_rejected_even_when_target_is_https(self):
        from pydantic import ValidationError

        from api.models import CredentialsIn

        with pytest.raises(ValidationError, match="login_url"):
            self._make(
                url="https://example.com/",
                credentials=CredentialsIn(
                    username="qa",
                    password="hunter22",
                    login_url="http://example.com/login",
                ),
            )

    def test_no_credentials_allows_plain_http(self):
        request = self._make(url="http://example.com/")
        assert request.credentials is None
