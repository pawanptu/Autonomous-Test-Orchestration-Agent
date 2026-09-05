"""URL canonicalisation and adaptive crawl prioritisation.

The crawl budget is small - twelve pages by default - so two decisions
determine what the agent is able to test at all: which URLs count as the same
page, and which of the remaining ones is visited first. Getting the first wrong
burns the whole budget on one document reached by twenty tracking links; getting
the second wrong spends it on the press releases and never reaches the login
form.
"""

from __future__ import annotations

import pytest

from browser.crawler import (
    TRACKING_PARAMS,
    canonicalize_url,
    fingerprint_page,
    normalize_url,
    should_follow,
    surface_priority,
)
from graph.state import DiscoveredPage


class TestCanonicalisation:
    def test_fragment_and_trailing_slash_collapse(self):
        canonical = canonicalize_url("https://ex.com/about/")
        assert canonicalize_url("https://ex.com/about") == canonical
        assert canonicalize_url("https://ex.com/about#team") == canonical

    def test_host_is_lowercased(self):
        assert canonicalize_url("https://EX.com/A") == "https://ex.com/A"

    def test_path_case_is_preserved(self):
        """Hosts are case-insensitive; paths are not.

        Lowercasing the path would merge ``/User`` and ``/user``, which are
        different resources on any case-sensitive server.
        """
        assert canonicalize_url("https://ex.com/User") != canonicalize_url("https://ex.com/user")

    @pytest.mark.parametrize("param", sorted(TRACKING_PARAMS)[:12])
    def test_tracking_parameters_are_dropped(self, param):
        assert canonicalize_url(f"https://ex.com/p?{param}=abc") == "https://ex.com/p"

    def test_meaningful_parameters_survive(self):
        """Pagination and filters select genuinely different pages.

        Dropping them would collapse an entire catalogue into its first page and
        leave the agent unable to test pagination at all.
        """
        assert "page=2" in canonicalize_url("https://ex.com/list?page=2")
        assert "q=shoes" in canonicalize_url("https://ex.com/search?q=shoes")

    def test_query_parameter_order_does_not_matter(self):
        assert canonicalize_url("https://ex.com/p?a=1&b=2") == canonicalize_url(
            "https://ex.com/p?b=2&a=1"
        )

    def test_unknown_parameters_are_kept(self):
        """An allowlist of known parameters would stop query-routed apps dead."""
        assert "custom=7" in canonicalize_url("https://ex.com/p?custom=7")

    def test_tracking_stripped_but_content_kept_together(self):
        assert (
            canonicalize_url("https://ex.com/list?utm_source=x&page=3&fbclid=y")
            == "https://ex.com/list?page=3"
        )

    def test_empty_input_is_empty_output(self):
        assert canonicalize_url("") == ""

    def test_normalize_url_is_the_backwards_compatible_alias(self):
        """Persisted memory keys and baselines were written under this name."""
        assert normalize_url("https://ex.com/a/") == canonicalize_url("https://ex.com/a/")


class TestSurfacePrioritisation:
    @pytest.mark.parametrize(
        "path,label",
        [
            ("/login", "authentication"),
            ("/signin", "authentication"),
            ("/register", "registration"),
            ("/checkout/payment", "checkout"),
            ("/cart", "cart"),
            ("/account/settings", "account"),
            ("/search?q=x", "search"),
            ("/products/42", "catalogue"),
            ("/contact", "support"),
            ("/blog/hello", "content"),
        ],
    )
    def test_paths_map_to_surfaces(self, path, label):
        assert surface_priority(path)[1] == label

    @pytest.mark.parametrize(
        "text,label",
        [
            ("Sign in", "authentication"),
            ("Create account", "registration"),
            ("Add to cart", "cart"),
            ("My Account", "account"),
        ],
    )
    def test_link_text_is_scored_for_opaque_spa_routes(self, text, label):
        """A single-page app routes through ``/#/x``; the text is the only signal."""
        assert surface_priority("/#/route", text)[1] == label

    def test_high_value_surfaces_outrank_content(self):
        assert surface_priority("/login")[0] > surface_priority("/blog/post")[0]
        assert surface_priority("/checkout")[0] > surface_priority("/about")[0]

    def test_unrecognised_paths_take_the_neutral_middle_score(self):
        """Unknown is not the same as worthless: it must beat a press release."""
        score, label = surface_priority("/some/unknown/thing")
        assert label == "general"
        assert score > surface_priority("/blog/x")[0]

    def test_ordering_puts_authentication_first(self):
        candidates = ["/blog/a", "/products/1", "/login", "/cart"]
        ordered = sorted(candidates, key=lambda u: -surface_priority(u)[0])
        assert ordered[0] == "/login"
        assert ordered[-1] == "/blog/a"


class TestCrawlBoundaries:
    def test_logout_links_are_never_followed(self):
        """One careless click destroys the authenticated session mid-crawl."""
        assert not should_follow("/logout", "https://ex.com/", same_origin_only=True)
        assert not should_follow(
            "/exit", "https://ex.com/", same_origin_only=True, link_text="Sign out"
        )

    @pytest.mark.parametrize(
        "href", ["mailto:a@b.com", "tel:+123", "javascript:void(0)", "data:text/html,x"]
    )
    def test_non_navigational_schemes_are_skipped(self, href):
        assert not should_follow(href, "https://ex.com/", same_origin_only=True)

    def test_cross_origin_is_skipped_when_same_origin_only(self):
        assert not should_follow("https://other.com/a", "https://ex.com/", same_origin_only=True)
        assert should_follow("https://other.com/a", "https://ex.com/", same_origin_only=False)

    def test_binary_downloads_are_skipped(self):
        assert not should_follow("/manual.pdf", "https://ex.com/", same_origin_only=True)


class TestPageFingerprint:
    @staticmethod
    def _page(**overrides) -> DiscoveredPage:
        defaults = dict(
            url="https://ex.com/login",
            inputs=[{"type": "email", "name": "email", "label": "Email"}],
            buttons=[{"text": "Sign in"}],
            forms=[{"method": "post", "field_count": 2, "has_password": True}],
            headings=["Sign in"],
        )
        defaults.update(overrides)
        return DiscoveredPage(**defaults)

    def test_identical_structure_fingerprints_identically(self):
        assert fingerprint_page(self._page()) == fingerprint_page(self._page())

    def test_a_new_form_field_changes_the_fingerprint(self):
        """A structural change must invalidate any cached locator for the page."""
        changed = self._page(
            inputs=[
                {"type": "email", "name": "email", "label": "Email"},
                {"type": "text", "name": "otp", "label": "One-time code"},
            ]
        )
        assert fingerprint_page(changed) != fingerprint_page(self._page())

    def test_changing_body_text_alone_does_not_change_the_fingerprint(self):
        """A catalogue whose prices moved is the same page for caching purposes.

        Fingerprinting the text would make the cache useless on any page that
        renders live data, which is most of them.
        """
        same = self._page(text_excerpt="prices updated at 14:03")
        assert fingerprint_page(same) == fingerprint_page(self._page())

    def test_fingerprint_is_short_and_stable(self):
        value = fingerprint_page(self._page())
        assert len(value) == 16 and value.isalnum()
