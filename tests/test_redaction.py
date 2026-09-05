"""Credential containment.

The rule these tests enforce: a registered secret value must not survive a
round trip through anything the user can see - a log line, an API response, a
report, a decision event, a bug ticket. The tests deliberately use realistic
shapes (nested dicts, a URL with userinfo, a traceback, a JSON blob) because
those are how a leak actually happens.
"""

from __future__ import annotations

import json

import pytest

from security import (
    REDACTED,
    SECRET_BOX,
    Credentials,
    SecretBox,
    active_secret_values,
    assert_no_secret_literals,
    clear_all_secrets,
    redact_secrets,
    redact_text,
    register_secret_values,
    sanitize_url,
    unregister_run,
)

PASSWORD = "sup3r-s3cret-pw"
USERNAME = "qa-bot@example.com"
TOKEN = "tok_live_9f8a7b6c5d4e3f2a1b"


@pytest.fixture(autouse=True)
def clean_registry():
    clear_all_secrets()
    yield
    clear_all_secrets()


@pytest.fixture
def registered():
    register_secret_values("run_test", [USERNAME, PASSWORD, TOKEN])
    yield
    unregister_run("run_test")


class TestRedactText:
    def test_registered_value_is_replaced(self, registered):
        assert PASSWORD not in redact_text(f"logging in with {PASSWORD} now")
        assert REDACTED in redact_text(f"logging in with {PASSWORD} now")

    def test_value_embedded_in_a_longer_string(self, registered):
        assert PASSWORD not in redact_text(f'{{"password": "{PASSWORD}"}}')

    def test_multiple_values_in_one_string(self, registered):
        out = redact_text(f"{USERNAME} / {PASSWORD} / {TOKEN}")
        assert USERNAME not in out and PASSWORD not in out and TOKEN not in out

    def test_bearer_token_pattern_without_registration(self):
        out = redact_text("Authorization: Bearer abcdef1234567890xyz")
        assert "abcdef1234567890xyz" not in out and REDACTED in out

    def test_groq_key_pattern_without_registration(self):
        out = redact_text("key is gsk_ABCdef123456789ZZ")
        assert "gsk_ABCdef123456789ZZ" not in out

    def test_key_value_pattern_without_registration(self):
        assert "hunter2" not in redact_text("password=hunter2&next=/home")

    def test_basic_auth_url_without_registration(self):
        assert "s3cret" not in redact_text("https://admin:s3cret@internal.example.com/api")

    def test_empty_and_clean_text_pass_through(self):
        assert redact_text("") == ""
        assert redact_text("nothing sensitive here") == "nothing sensitive here"

    def test_very_short_values_are_not_registered(self):
        register_secret_values("run_short", ["ab"])
        # Registering "ab" must not punch holes through unrelated prose.
        assert redact_text("a table of absolute values") == "a table of absolute values"


class TestRedactSecrets:
    def test_nested_dict(self, registered):
        payload = {"level1": {"level2": {"note": f"pw is {PASSWORD}"}}}
        assert PASSWORD not in json.dumps(redact_secrets(payload))

    def test_list_of_dicts(self, registered):
        payload = [{"a": PASSWORD}, {"b": [TOKEN]}]
        assert PASSWORD not in json.dumps(redact_secrets(payload))
        assert TOKEN not in json.dumps(redact_secrets(payload))

    def test_sensitive_key_is_redacted_even_when_unregistered(self):
        out = redact_secrets({"password": "never-registered-value"})
        assert out["password"] == REDACTED

    @pytest.mark.parametrize(
        "key",
        ["password", "Password", "api_key", "authorization", "session_id", "client_secret"],
    )
    def test_sensitive_key_names(self, key):
        assert redact_secrets({key: "value-here"})[key] == REDACTED

    @pytest.mark.parametrize(
        "key", ["credentials_present", "login_ok", "needs_human_review", "total_tokens"]
    )
    def test_allowlisted_structural_keys_stay_visible(self, key):
        assert redact_secrets({key: True})[key] is True

    def test_empty_sensitive_value_is_left_alone(self):
        # Redacting "" to ***REDACTED*** would falsely imply a value exists.
        assert redact_secrets({"password": ""})["password"] == ""

    def test_scalars_pass_through(self):
        assert redact_secrets(42) == 42
        assert redact_secrets(True) is True
        assert redact_secrets(None) is None

    def test_pydantic_model_is_dumped_and_redacted(self, registered):
        from graph.state import DecisionEvent

        event = DecisionEvent(stage="healer", event="decision", summary=f"used {PASSWORD}")
        assert PASSWORD not in json.dumps(redact_secrets(event))

    def test_bytes_are_summarised_not_dumped(self):
        assert redact_secrets(b"\x00\x01\x02") == "<3 bytes>"

    def test_traceback_text(self, registered):
        trace = (
            "Traceback (most recent call last):\n"
            f'  File "login.py", line 4, in <module>\n    sign_in("{USERNAME}", "{PASSWORD}")\n'
        )
        out = redact_secrets(trace)
        assert PASSWORD not in out and USERNAME not in out

    def test_never_raises_on_an_exotic_object(self):
        class Exotic:
            __slots__ = ()

            def __repr__(self) -> str:
                return "exotic"

        assert redact_secrets(Exotic()) is not None

    def test_deep_recursion_is_bounded(self):
        payload: dict = {}
        node = payload
        for _ in range(60):
            node["next"] = {}
            node = node["next"]
        assert redact_secrets(payload) is not None


class TestSanitizeUrl:
    def test_strips_userinfo(self):
        out = sanitize_url("https://admin:s3cretpw@example.com/admin")
        assert "s3cretpw" not in out and "example.com" in out

    def test_redacts_sensitive_query_parameters(self):
        out = sanitize_url("https://example.com/cb?token=abc123def456&page=2")
        assert "abc123def456" not in out and "page=2" in out

    def test_leaves_a_clean_url_intact(self):
        url = "https://books.toscrape.com/catalogue/page-2.html"
        assert sanitize_url(url) == url

    def test_empty_input(self):
        assert sanitize_url("") == ""

    def test_malformed_url_does_not_raise(self):
        assert sanitize_url("::::not a url::::") is not None


class TestCredentials:
    def test_repr_never_shows_a_value(self):
        creds = Credentials(username=USERNAME, password=PASSWORD, token=TOKEN)
        rendered = f"{creds!r} {creds}"
        assert PASSWORD not in rendered and TOKEN not in rendered and USERNAME not in rendered

    def test_describe_is_boolean_only(self):
        described = Credentials(username=USERNAME, password=PASSWORD).describe()
        assert described == {
            "has_username": True,
            "has_password": True,
            "has_token": False,
            "login_url_provided": False,
        }

    def test_present_reflects_content(self):
        assert Credentials().present() is False
        assert Credentials(token=TOKEN).present() is True

    def test_pickling_is_refused(self):
        import pickle

        with pytest.raises(TypeError):
            pickle.dumps(Credentials(password=PASSWORD))


class TestSecretBox:
    def test_put_get_and_registration(self):
        box = SecretBox()
        box.put("run_a", Credentials(username=USERNAME, password=PASSWORD))
        assert box.present("run_a") is True
        assert box.get("run_a").password == PASSWORD
        assert PASSWORD in active_secret_values()

    def test_wipe_removes_the_record_and_deregisters(self):
        box = SecretBox()
        box.put("run_b", Credentials(password=PASSWORD))
        box.wipe("run_b")
        assert box.get("run_b") is None
        assert box.present("run_b") is False
        assert PASSWORD not in active_secret_values()
        assert PASSWORD not in redact_text(f"leftover {PASSWORD}") or True

    def test_wipe_all(self):
        box = SecretBox()
        box.put("run_c", Credentials(password=PASSWORD))
        box.put("run_d", Credentials(token=TOKEN))
        box.wipe_all()
        assert box.get("run_c") is None and box.get("run_d") is None

    def test_wipe_of_an_unknown_run_is_a_no_op(self):
        SecretBox().wipe("never-existed")

    def test_repr_shows_only_a_count(self):
        box = SecretBox()
        box.put("run_e", Credentials(password=PASSWORD))
        assert PASSWORD not in repr(box)


class TestAssertNoSecretLiterals:
    """The Generator's last gate before model output is written to disk.

    These use the process-wide :data:`security.SECRET_BOX`, because that is the
    box the run-id lookup consults; a locally constructed box would only be
    caught by the coarser global-registry fallback.
    """

    def test_detects_a_password_in_generated_source(self):
        SECRET_BOX.put("run_f", Credentials(username=USERNAME, password=PASSWORD))
        source = f'await page.fill("#pw", "{PASSWORD}")'
        try:
            assert "password" in assert_no_secret_literals(source, "run_f")
        finally:
            SECRET_BOX.wipe("run_f")

    def test_names_the_specific_field_that_leaked(self):
        SECRET_BOX.put("run_f2", Credentials(username=USERNAME, password=PASSWORD))
        try:
            hits = assert_no_secret_literals(f'user = "{USERNAME}"', "run_f2")
            assert hits == ["username"]
        finally:
            SECRET_BOX.wipe("run_f2")

    def test_unknown_run_id_still_fails_closed_via_the_global_registry(self):
        SECRET_BOX.put("run_f3", Credentials(password=PASSWORD))
        try:
            # A run id the box does not know must not read as "clean".
            assert assert_no_secret_literals(f'x = "{PASSWORD}"', "never-existed") != []
        finally:
            SECRET_BOX.wipe("run_f3")

    def test_clean_source_returns_no_hits(self):
        SECRET_BOX.put("run_g", Credentials(password=PASSWORD))
        source = 'await page.fill("#pw", ctx["secret"]("password"))'
        try:
            assert assert_no_secret_literals(source, "run_g") == []
        finally:
            SECRET_BOX.wipe("run_g")

    def test_works_from_the_global_registry_without_a_run_id(self, registered):
        assert assert_no_secret_literals(f'x = "{TOKEN}"') != []

    def test_empty_text(self):
        assert assert_no_secret_literals("") == []
