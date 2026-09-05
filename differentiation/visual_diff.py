"""Pixel-level visual regression detection over Playwright screenshots.

A DOM assertion can pass while the page is visually broken: a stylesheet that
404s, a flex container that collapses, a modal that renders off-screen. None of
those move the text the assertions read, so the functional suite stays green.
This module is the second pair of eyes: it compares the viewport frame captured
during a run against a stored reference image and reports what moved.

Baselines are cross-run, not per-run
------------------------------------
A baseline lives under :data:`config.BASELINES_DIR` keyed by *target host +
flow id + viewport*, deliberately **not** by run id. A per-run baseline would
compare a frame against itself and could never detect anything. The first
successful run of a flow at a viewport establishes the reference; every later
run diffs against it. Only a run whose test finished ``PASSED`` or ``HEALED``
may establish one - freezing a broken page as the reference would make the
broken state the new "correct" state and silently hide the regression forever.

Honest limitation
-----------------
A pixel differ has no notion of semantic importance. It cannot tell a shifted
checkout button from a font that hinted one sub-pixel differently because the
run happened on a different machine, a different GPU or a different Chromium
build. Anti-aliasing, font fallback, scrollbar width, animated banners and
timestamps rendered into the page all register as "changed pixels". That is
precisely why :attr:`config.Settings.visual_diff_threshold` is configurable,
why the per-channel tolerance exists, and - most importantly - why visual
findings are reported as their own section of the report rather than being
folded into the functional pass/fail verdict. A visual finding is a prompt for
a human to look at an image, not a build failure.
"""

from __future__ import annotations

import hashlib
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence
from urllib.parse import urlsplit

from config import BASELINES_DIR, Settings, get_settings
from graph.state import RiskLevel, TestFlow, TestResult, TestStatus, VisualFinding
from logging_setup import get_logger
from security import redact_text, sanitize_url

log = get_logger("aivor.visual")

try:  # Pillow is a hard requirement of this feature, not of the whole process.
    from PIL import Image, ImageChops, ImageFilter

    _PIL_IMPORT_ERROR: str | None = None
except Exception as exc:  # pragma: no cover - only hit on a broken install
    Image = ImageChops = ImageFilter = None  # type: ignore[assignment]
    _PIL_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"


# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------
VIEWPORT_FRAME_SUFFIX: str = "__viewport.png"
"""Filename suffix the runner gives the fixed-height frame it captures for every
flow. The full-page frame is unsuitable as a diff subject because its height
changes with the content, which would flag every page as fully changed."""

PAD_BACKGROUND: tuple[int, int, int] = (255, 255, 255)
HIGHLIGHT_COLOUR: tuple[int, int, int] = (255, 45, 60)
HIGHLIGHT_ALPHA: int = 235
DIM_TOWARDS_WHITE: float = 0.55
"""How far the current frame is washed out before the highlight is painted on
top. A dimmed backdrop keeps the layout readable while making the changed
regions unmissable."""

MAX_SLUG_LENGTH: int = 80
_SAFE_SLUG_CHARS: frozenset[str] = frozenset("-_.")


# --------------------------------------------------------------------------
# Path helpers
# --------------------------------------------------------------------------
def _slugify(text: str, fallback: str) -> str:
    """Reduce arbitrary text to a filesystem-safe, Windows-safe token.

    Flow ids and host names both originate outside this process, so they are
    treated as untrusted for path-construction purposes: anything that is not
    alphanumeric or a conservative punctuation character becomes an underscore,
    and the result is length-capped so a pathological id cannot blow past the
    Windows path limit.
    """
    cleaned = "".join(
        ch if (ch.isalnum() and ch.isascii()) or ch in _SAFE_SLUG_CHARS else "_"
        for ch in (text or "")
    ).strip("._")
    if not cleaned:
        return fallback
    return cleaned[:MAX_SLUG_LENGTH]


def _host_slug(target_url: str) -> str:
    """Directory name for one target: its host (and port), never its userinfo.

    ``urlsplit(...).hostname`` is used rather than ``netloc`` because it drops
    any ``user:pass@`` prefix outright, so a credential embedded in the target
    URL can never become a directory name on disk.
    """
    raw = (target_url or "").strip()
    if not raw:
        return "unknown-host"
    candidate = raw if "://" in raw else f"//{raw}"
    try:
        parts = urlsplit(candidate)
        host = parts.hostname or ""
        port = parts.port
    except ValueError:
        # A malformed port or bracketed IPv6 literal: fall back to a sanitised
        # rendering of the URL with any credential material already stripped.
        return _slugify(sanitize_url(raw), "unknown-host")
    # An IPv6 literal is all colons; map them to hyphens first so that "::1" does
    # not slug down to the same token as the unrelated host "1".
    host = host.replace(":", "-")
    label = f"{host}_{port}" if port else host
    return _slugify(label, "unknown-host")


@dataclass(frozen=True)
class BaselineIdentity:
    """Everything that must match for two screenshots to be comparable.

    A baseline is only a valid reference for a capture taken under the same
    conditions. Comparing a staging capture against a production baseline, or a
    Chrome 128 capture against Chrome 131, produces diffs that are real pixel
    differences and entirely uninteresting - which is how visual regression
    suites end up ignored.

    The build identifier is recorded but deliberately *not* part of the key: a
    baseline exists precisely to be compared across builds. Everything else -
    environment, locale, theme, viewport, browser version, masking rules - does
    key the baseline, because a difference in any of them makes the comparison
    meaningless rather than informative.
    """

    environment: str = "default"
    locale: str = "en-US"
    theme: str = "light"
    viewport: str = "1280x900"
    browser: str = "chromium"
    browser_version: str = ""
    mask_signature: str = ""
    build_id: str = ""

    @classmethod
    def from_settings(
        cls,
        settings: Any,
        *,
        viewport: str = "",
        browser_version: str = "",
        theme: str = "light",
    ) -> "BaselineIdentity":
        masks = tuple(getattr(settings, "visual_mask_selectors", ()) or ())
        return cls(
            environment=str(getattr(settings, "visual_environment", "default") or "default"),
            locale=str(getattr(settings, "visual_locale", "en-US") or "en-US"),
            theme=theme,
            viewport=viewport
            or f"{getattr(settings, 'viewport_width', 0)}x{getattr(settings, 'viewport_height', 0)}",
            browser="chromium",
            browser_version=browser_version,
            mask_signature=mask_signature(masks),
            build_id=str(getattr(settings, "visual_build_id", "") or ""),
        )

    def key(self) -> str:
        """Short, stable directory-safe token identifying these conditions."""
        material = "|".join(
            [
                self.environment,
                self.locale,
                self.theme,
                self.viewport,
                self.browser,
                _major_version(self.browser_version),
                self.mask_signature,
            ]
        )
        digest = hashlib.sha256(material.encode("utf-8", "replace")).hexdigest()[:10]
        return f"{_slugify(self.environment, 'env')}-{_slugify(self.theme, 'light')}-{digest}"

    def describe(self) -> dict[str, str]:
        """Provenance recorded alongside the baseline and cited in the report."""
        return {
            "environment": self.environment,
            "locale": self.locale,
            "theme": self.theme,
            "viewport": self.viewport,
            "browser": self.browser,
            "browser_version": self.browser_version,
            "browser_major": _major_version(self.browser_version),
            "mask_signature": self.mask_signature,
            "build_id": self.build_id,
            "key": self.key(),
        }


def _major_version(version: str) -> str:
    """Major browser version only.

    Chromium repaints subtly between patch releases but not in ways that should
    invalidate a baseline; a major version bump routinely does. Keying on the
    major alone keeps baselines useful for longer without making them wrong.
    """
    text = (version or "").strip()
    if not text:
        return ""
    return text.split(".", 1)[0]


def mask_signature(selectors: Sequence[str]) -> str:
    """Stable hash of the masking rules in force.

    Part of baseline identity because a capture with a masked price banner and
    one without are different images by construction; comparing them would
    report the mask itself as a regression.
    """
    if not selectors:
        return "none"
    material = "|".join(sorted(str(s).strip() for s in selectors if str(s).strip()))
    if not material:
        return "none"
    return hashlib.sha256(material.encode("utf-8", "replace")).hexdigest()[:8]


def baseline_path_for(
    target_url: str,
    flow_id: str,
    viewport: str,
    identity: BaselineIdentity | None = None,
) -> Path:
    """Where the cross-run reference image for one flow lives.

    ``reports/baselines/<host>/<identity>/<flow_id>__<viewport>.png``. The
    parent directory is created lazily here so that callers can ask for the path
    (to test whether a baseline exists) without having to know about directory
    creation.

    ``identity`` is optional so that existing callers keep working; when it is
    omitted the legacy flat layout is used, which is what the baselines already
    on disk from previous runs are stored under.
    """
    directory = BASELINES_DIR / _host_slug(target_url)
    if identity is not None:
        directory = directory / identity.key()
    try:
        directory.mkdir(parents=True, exist_ok=True)
    except OSError as exc:  # pragma: no cover - read-only volume / permissions
        log.warning("could not create baseline directory %s: %s", directory, exc)
    name = f"{_slugify(flow_id, 'flow')}__{_slugify(viewport, 'viewport')}.png"
    return directory / name


# --------------------------------------------------------------------------
# Comparison
# --------------------------------------------------------------------------
@dataclass
class DiffResult:
    """The numeric outcome of comparing two frames.

    ``total_pixels == 0`` is the sentinel for "the comparison could not be
    performed"; ``note`` then carries the reason. Callers must not read a
    ``changed_ratio`` of 0.0 as "identical" without checking ``total_pixels``.
    """

    changed_ratio: float
    changed_pixels: int
    total_pixels: int
    size_mismatch: bool
    baseline_size: tuple[int, int] | None
    current_size: tuple[int, int] | None
    note: str = ""

    @property
    def comparable(self) -> bool:
        """True when the two frames were actually diffed."""
        return self.total_pixels > 0

    @property
    def percent(self) -> float:
        """``changed_ratio`` expressed as a percentage, for report prose."""
        return self.changed_ratio * 100.0


def _failed_diff(note: str) -> DiffResult:
    """A DiffResult that says, in its own note, that nothing was compared."""
    return DiffResult(
        changed_ratio=0.0,
        changed_pixels=0,
        total_pixels=0,
        size_mismatch=False,
        baseline_size=None,
        current_size=None,
        note=note,
    )


def _load_rgb(path: str | Path) -> "Image.Image":
    """Open an image as RGB with the file handle closed again immediately.

    Windows keeps a lock on an open file, and the caller may well want to copy
    over or replace this very image a moment later, so the decoded copy is
    detached from the file rather than lazily backed by it.
    """
    with Image.open(Path(path)) as handle:
        return handle.convert("RGB")


def _pad_onto(image: "Image.Image", size: tuple[int, int]) -> "Image.Image":
    """Place ``image`` at the top-left of a white canvas of ``size``."""
    if image.size == size:
        return image
    canvas = Image.new("RGB", size, PAD_BACKGROUND)
    canvas.paste(image, (0, 0))
    return canvas


def _align(
    baseline: "Image.Image", current: "Image.Image"
) -> tuple["Image.Image", "Image.Image", bool, str]:
    """Bring two frames to a common geometry by padding, never by resizing.

    Resizing would rescale a genuine layout shift into an "everything moved by a
    fraction of a pixel" result, which is both a false positive across the whole
    image and a false negative for the actual shift. Padding keeps every
    unchanged region pixel-aligned and confines the consequence of the size
    change to the band that genuinely differs.
    """
    if baseline.size == current.size:
        return baseline, current, False, ""
    width = max(baseline.width, current.width)
    height = max(baseline.height, current.height)
    note = (
        f"The baseline is {baseline.width}x{baseline.height} and the current frame is "
        f"{current.width}x{current.height}; both were padded onto a {width}x{height} white "
        "canvas rather than resized, so the mismatched band is counted as changed instead of "
        "being smeared across the whole image."
    )
    return _pad_onto(baseline, (width, height)), _pad_onto(current, (width, height)), True, note


def _changed_mask(
    baseline: "Image.Image", current: "Image.Image", tolerance: int
) -> tuple["Image.Image", int, int]:
    """Build the boolean change mask and count it, entirely inside Pillow.

    A pixel counts as changed when the largest of its three per-channel absolute
    deltas exceeds ``tolerance``. This is computed with band arithmetic and a
    256-entry lookup table rather than a Python loop: a 1280x900 frame is over a
    million pixels, and iterating those in Python would take longer than the
    browser run that produced them.
    """
    delta = ImageChops.difference(baseline, current)
    red, green, blue = delta.split()
    # ImageChops.lighter is a per-pixel max, so this is max(|dR|, |dG|, |dB|).
    max_delta = ImageChops.lighter(ImageChops.lighter(red, green), blue)

    cutoff = max(0, min(255, int(tolerance)))
    mask = max_delta.point(lambda value: 255 if value > cutoff else 0)

    histogram = mask.histogram()
    changed = int(histogram[255]) if len(histogram) > 255 else 0
    total = int(mask.width) * int(mask.height)
    return mask, changed, total


def compare_images(
    baseline_path: str | Path,
    current_path: str | Path,
    *,
    tolerance: int = 24,
) -> DiffResult:
    """Compare two frames and report the fraction of pixels that moved.

    Never raises. A missing, truncated or undecodable image yields a
    :class:`DiffResult` with ``total_pixels == 0`` and an explanatory ``note``,
    because one unreadable screenshot must not take down a run that otherwise
    produced good evidence.
    """
    if Image is None:  # pragma: no cover - only on a broken install
        return _failed_diff(
            "Pillow is not importable in this environment, so no pixel comparison was "
            f"performed ({_PIL_IMPORT_ERROR})."
        )

    baseline_file = Path(baseline_path)
    current_file = Path(current_path)
    try:
        baseline = _load_rgb(baseline_file)
    except Exception as exc:
        return _failed_diff(
            f"The baseline image could not be read ({type(exc).__name__}), so no comparison "
            "was possible."
        )
    try:
        current = _load_rgb(current_file)
    except Exception as exc:
        return _failed_diff(
            f"The current frame could not be read ({type(exc).__name__}), so no comparison "
            "was possible."
        )

    baseline_size = baseline.size
    current_size = current.size
    try:
        left, right, mismatch, note = _align(baseline, current)
        _, changed, total = _changed_mask(left, right, tolerance)
    except Exception as exc:  # pragma: no cover - decoder or memory failure
        log.warning("pixel comparison failed for %s: %s", current_file.name, exc)
        return _failed_diff(
            f"The pixel comparison failed unexpectedly ({type(exc).__name__}), so this frame "
            "was not evaluated."
        )

    ratio = (changed / total) if total else 0.0
    return DiffResult(
        changed_ratio=round(ratio, 6),
        changed_pixels=changed,
        total_pixels=total,
        size_mismatch=mismatch,
        baseline_size=baseline_size,
        current_size=current_size,
        note=note,
    )


def write_diff_image(
    baseline_path: str | Path,
    current_path: str | Path,
    out_path: str | Path,
    *,
    tolerance: int = 24,
) -> str | None:
    """Render a human-readable diff and return its path, or ``None`` on failure.

    The output is the *current* frame washed out towards white with the changed
    regions painted in saturated red. Showing the current state rather than the
    baseline is deliberate: the reviewer is being asked "is this new rendering
    acceptable?", so the new rendering is what they should be looking at. The
    mask is dilated by one pixel before painting, because a genuine one-pixel
    border change is invisible at report scale otherwise.
    """
    if Image is None:  # pragma: no cover - only on a broken install
        log.warning("cannot render a diff image: Pillow is unavailable (%s)", _PIL_IMPORT_ERROR)
        return None

    target = Path(out_path)
    try:
        baseline = _load_rgb(baseline_path)
        current = _load_rgb(current_path)
        aligned_baseline, aligned_current, _, _ = _align(baseline, current)
        mask, _, _ = _changed_mask(aligned_baseline, aligned_current, tolerance)

        dilated = mask.filter(ImageFilter.MaxFilter(3))
        alpha = dilated.point(lambda value: HIGHLIGHT_ALPHA if value else 0)

        size = aligned_current.size
        greyscale = aligned_current.convert("L").convert("RGB")
        canvas = Image.blend(greyscale, Image.new("RGB", size, PAD_BACKGROUND), DIM_TOWARDS_WHITE)
        canvas.paste(Image.new("RGB", size, HIGHLIGHT_COLOUR), (0, 0), alpha)

        target.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(target, format="PNG")
        return str(target)
    except Exception as exc:
        log.warning("could not write diff image %s: %s", target.name, exc)
        return None


# --------------------------------------------------------------------------
# Run-level analysis
# --------------------------------------------------------------------------
BASELINE_ELIGIBLE_STATUSES: frozenset[TestStatus] = frozenset(
    {TestStatus.PASSED, TestStatus.HEALED}
)
"""Only a green run may become the reference. A failing run's frame is very
often exactly the broken layout we want to catch next time."""


def _viewport_frame_for(result: TestResult) -> Path | None:
    """The stable viewport frame belonging to a result, or ``None``.

    The runner captures ``<flow_id>__viewport.png`` for every flow, but then
    *overwrites* ``screenshot_path`` with the full-page failure frame when the
    test did not pass. The full-page frame has a content-dependent height and is
    useless as a diff subject, so for those results we look for the viewport
    frame sitting next to it. The returned path is not guaranteed to exist; the
    caller checks, so that it can explain a missing frame in the finding.
    """
    raw = (result.screenshot_path or "").strip()
    if not raw:
        return None
    path = Path(raw)
    if path.name.endswith(VIEWPORT_FRAME_SUFFIX):
        return path
    return path.parent / f"{result.flow_id}{VIEWPORT_FRAME_SUFFIX}"


def _latest_per_flow(results: Sequence[TestResult]) -> list[TestResult]:
    """One result per flow id, keeping the most recent occurrence.

    A healed flow can appear twice (original failure, then the re-run). Diffing
    both would double-count the flow and, worse, the second comparison would
    trivially match the baseline the first one had just established. The later
    result is the one that reflects the state the run finished in.
    """
    latest: dict[str, TestResult] = {}
    for result in results:
        latest[result.flow_id] = result
    return list(latest.values())


def _degraded_finding(
    *,
    flow_id: str,
    flow_name: str,
    viewport: str,
    threshold: float,
    note: str,
    risk: RiskLevel | None,
    baseline: Path | None = None,
    current: Path | None = None,
) -> VisualFinding:
    """A finding that records why no comparison happened, never a fake pass."""
    return VisualFinding(
        flow_id=flow_id,
        flow_name=flow_name,
        viewport=viewport,
        baseline_path=str(baseline) if baseline is not None else None,
        current_path=str(current) if current is not None else None,
        diff_path=None,
        changed_ratio=0.0,
        threshold=threshold,
        is_regression=False,
        is_new_baseline=False,
        note=redact_text(note),
        risk=risk,
    )


def analyse(
    *,
    run_directory: Path,
    target_url: str,
    results: Sequence[TestResult],
    flows: Sequence[TestFlow],
    risk_lookup: dict[str, RiskLevel] | None = None,
    settings: Settings | None = None,
) -> list[VisualFinding]:
    """Produce one visual finding per flow that captured a comparable frame.

    Establishes a baseline the first time a flow is seen green at a viewport,
    and diffs against it on every later run. Never raises: a flow whose image is
    missing or corrupt degrades to a finding that says so, because losing one
    screenshot must not cost the run its entire visual section.
    """
    cfg = settings or get_settings()
    lookup = risk_lookup or {}
    viewport = f"{cfg.viewport_width}x{cfg.viewport_height}"
    threshold = float(cfg.visual_diff_threshold)
    tolerance = int(cfg.visual_pixel_tolerance)
    safe_target = sanitize_url(target_url or "")
    # Baselines are keyed by the conditions the capture was taken under, so a
    # staging frame is never diffed against a production reference and a browser
    # major-version bump starts a fresh baseline instead of reporting every page
    # as changed.
    identity = BaselineIdentity.from_settings(cfg, viewport=viewport)
    log.info("visual baseline identity: %s", identity.describe())

    names: dict[str, str] = {flow.id: flow.name for flow in flows}
    visual_dir = Path(run_directory) / "visual"
    findings: list[VisualFinding] = []

    if Image is None:  # pragma: no cover - only on a broken install
        log.error("visual diff disabled: Pillow is unavailable (%s)", _PIL_IMPORT_ERROR)
        return [
            _degraded_finding(
                flow_id=result.flow_id,
                flow_name=names.get(result.flow_id, result.flow_name),
                viewport=viewport,
                threshold=threshold,
                note=(
                    "Pillow could not be imported, so no visual comparison was performed for "
                    f"this run ({_PIL_IMPORT_ERROR})."
                ),
                risk=lookup.get(result.flow_id),
            )
            for result in _latest_per_flow(results)
        ]

    for result in _latest_per_flow(results):
        flow_id = result.flow_id
        flow_name = names.get(flow_id, result.flow_name)
        risk = lookup.get(flow_id)
        try:
            frame = _viewport_frame_for(result)
            if frame is None:
                log.debug("flow %s recorded no screenshot; nothing to diff", flow_id)
                continue

            if not frame.exists():
                findings.append(
                    _degraded_finding(
                        flow_id=flow_id,
                        flow_name=flow_name,
                        viewport=viewport,
                        threshold=threshold,
                        note=(
                            "The stable viewport frame for this flow was not written to disk "
                            "(the capture was skipped or failed), so no visual comparison was "
                            "possible."
                        ),
                        risk=risk,
                    )
                )
                continue

            baseline = baseline_path_for(target_url, flow_id, viewport, identity)

            # ---- no reference yet: establish one, but only from a green run ----
            if not baseline.exists():
                if result.status not in BASELINE_ELIGIBLE_STATUSES:
                    findings.append(
                        _degraded_finding(
                            flow_id=flow_id,
                            flow_name=flow_name,
                            viewport=viewport,
                            threshold=threshold,
                            note=(
                                f"No visual baseline exists for this flow at {viewport} and the "
                                f"test finished with status '{result.status.value}', so this "
                                "frame was not stored as the reference. A baseline is only ever "
                                "established from a passing or healed run, otherwise a broken "
                                "layout would become the definition of correct."
                            ),
                            risk=risk,
                            current=frame,
                        )
                    )
                    continue
                try:
                    shutil.copyfile(frame, baseline)
                except OSError as exc:
                    findings.append(
                        _degraded_finding(
                            flow_id=flow_id,
                            flow_name=flow_name,
                            viewport=viewport,
                            threshold=threshold,
                            note=(
                                "The first frame for this flow could not be stored as the visual "
                                f"baseline ({type(exc).__name__}), so future runs will have "
                                "nothing to compare against until this is resolved."
                            ),
                            risk=risk,
                            current=frame,
                        )
                    )
                    continue

                log.info("established visual baseline for %s at %s", flow_id, viewport)
                findings.append(
                    VisualFinding(
                        flow_id=flow_id,
                        flow_name=flow_name,
                        viewport=viewport,
                        baseline_path=str(baseline),
                        current_path=str(frame),
                        diff_path=None,
                        changed_ratio=0.0,
                        threshold=threshold,
                        is_regression=False,
                        is_new_baseline=True,
                        note=redact_text(
                            f"No visual baseline existed for this flow at {viewport} on "
                            f"{safe_target}, so this frame was stored as the reference for "
                            "future runs. Nothing was compared on this run."
                        ),
                        risk=risk,
                    )
                )
                continue

            # ---- reference exists: diff against it ----------------------------
            diff = compare_images(baseline, frame, tolerance=tolerance)
            if not diff.comparable:
                findings.append(
                    _degraded_finding(
                        flow_id=flow_id,
                        flow_name=flow_name,
                        viewport=viewport,
                        threshold=threshold,
                        note=diff.note or "The frames could not be compared.",
                        risk=risk,
                        baseline=baseline,
                        current=frame,
                    )
                )
                continue

            # A geometry change is a regression in its own right. Relying on the
            # pixel ratio alone would miss it: the white padding band differs from
            # a pale page background by less than the tolerance, so a page that
            # grew or shrank could score 0% changed while demonstrably having
            # moved. The viewport frame is captured at a fixed viewport and the
            # baseline is keyed by that viewport, so a size difference here means
            # the rendered geometry itself changed.
            is_regression = diff.changed_ratio > threshold or diff.size_mismatch
            diff_image: str | None = None
            if diff.changed_pixels > 0 or diff.size_mismatch:
                diff_image = write_diff_image(
                    baseline,
                    frame,
                    visual_dir / f"{_slugify(flow_id, 'flow')}__{viewport}__diff.png",
                    tolerance=tolerance,
                )

            note = (
                f"{diff.percent:.1f}% of pixels changed versus the stored baseline "
                f"(threshold {threshold * 100:.1f}%, per-channel tolerance {tolerance}/255)."
            )
            if diff.size_mismatch:
                note += (
                    " The frame also changed size, which is treated as a regression on its own "
                    "regardless of the pixel ratio."
                )
                if diff.note:
                    note += " " + diff.note
            elif is_regression:
                note += " Reported as a visual regression for human review."
            else:
                note += " Below the reporting threshold, so this is not flagged as a regression."
            if diff_image is None and (diff.changed_pixels > 0 or diff.size_mismatch):
                note += " The annotated diff image could not be written; see the run log."

            if is_regression:
                log.info(
                    "visual regression on %s: %.2f%% changed (threshold %.2f%%)",
                    flow_id,
                    diff.percent,
                    threshold * 100,
                )

            findings.append(
                VisualFinding(
                    flow_id=flow_id,
                    flow_name=flow_name,
                    viewport=viewport,
                    baseline_path=str(baseline),
                    current_path=str(frame),
                    diff_path=diff_image,
                    changed_ratio=diff.changed_ratio,
                    threshold=threshold,
                    is_regression=is_regression,
                    is_new_baseline=False,
                    note=redact_text(note),
                    risk=risk,
                )
            )
        except Exception as exc:  # pragma: no cover - defensive, must not kill the run
            log.warning("visual analysis failed for flow %s: %s", flow_id, exc, exc_info=True)
            findings.append(
                _degraded_finding(
                    flow_id=flow_id,
                    flow_name=flow_name,
                    viewport=viewport,
                    threshold=threshold,
                    note=(
                        f"Visual analysis of this flow failed unexpectedly ({type(exc).__name__}), "
                        "so it was skipped rather than allowed to abort the run."
                    ),
                    risk=risk,
                )
            )

    return findings


def summarise(findings: Sequence[VisualFinding]) -> dict[str, Any]:
    """Headline counters for the report and the ``/status`` payload.

    ``checked`` counts every flow that produced a finding, including the ones
    that only established a baseline or degraded, so the number always reconciles
    with the length of the findings list a reader is looking at.
    """
    ratios = [float(f.changed_ratio) for f in findings]
    return {
        "checked": len(findings),
        "regressions": sum(1 for f in findings if f.is_regression),
        "new_baselines": sum(1 for f in findings if f.is_new_baseline),
        "max_changed_ratio": round(max(ratios), 6) if ratios else 0.0,
    }
