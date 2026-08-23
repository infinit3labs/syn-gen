"""The disclosure documentation exists, is reachable, and has not drifted.

The disclosure summary printed after every `syntab profile` ends by pointing at
docs/disclosure.md. A notice that cites a document is only as good as the
document, so these check the link resolves and that the numbers and flag names
in the prose still match the code. Documentation drift is the normal failure
mode for a file like this.
"""
import re
from pathlib import Path

import pytest

from syntab.cli import profile as profile_cmd
from syntab.disclosure import DisclosureReport
from syntab.profiler import RECOMMENDED_MIN_CELL_COUNT

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "docs" / "disclosure.md"
README = ROOT / "README.md"


@pytest.fixture(scope="module")
def doc_text():
    assert DOC.is_file(), f"{DOC} is missing but the profile notice cites it"
    return DOC.read_text(encoding="utf-8")


def test_the_notice_cites_a_document_that_exists():
    """The rendered notice names docs/disclosure.md; it must be there."""
    text = DisclosureReport().to_text()
    assert "docs/disclosure.md" in text
    assert DOC.is_file()


def test_readme_links_to_the_doc_and_the_link_resolves():
    text = README.read_text(encoding="utf-8")
    assert "docs/disclosure.md" in text
    for target in re.findall(r"\]\(([^)#][^)]*)\)", text):
        assert (ROOT / target).exists(), f"dead README link: {target}"


def test_doc_states_the_posture_explicitly(doc_text):
    """The one sentence this document exists to say."""
    lowered = doc_text.lower()
    assert "derived from your source data" in lowered
    assert "review it before" in lowered
    assert "commit" in lowered and "shar" in lowered
    assert "synthetic data is not automatically anonymous" in lowered


def test_doc_says_what_a_profiled_spec_contains(doc_text):
    for field in ("profile.categorical.values", "profile.numeric.min",
                  "profile.datetime_range", "profile.string_pattern"):
        assert field in doc_text, f"{field} is undocumented"


def test_doc_grounds_itself_in_named_established_practice(doc_text):
    for concept in ("k-anonymity", "Sweeney", "Safe Harbor",
                    "NIST SP 800-122", "generalization", "suppression"):
        assert concept in doc_text, f"{concept} is not cited"


def test_documented_recommended_threshold_matches_the_code(doc_text):
    """Prose and constant must not drift apart."""
    assert f"--min-cell-count {RECOMMENDED_MIN_CELL_COUNT}" in doc_text
    assert f"threshold of {RECOMMENDED_MIN_CELL_COUNT}" in doc_text


def test_doc_explains_the_default_and_its_rationale(doc_text):
    assert "The default is `0`, i.e. off." in doc_text
    assert "deliberate" in doc_text


def test_every_flag_the_doc_mentions_is_a_real_profile_option(doc_text):
    """The drift guard that actually catches things.

    ``secondary_opts`` as well as ``opts``: click keeps the OFF half of a
    ``--x/--no-x`` pair there, so reading ``opts`` alone made the guard reject
    ``--no-condition-on-fds`` -- a real option -- as non-existent. The
    assertion is unchanged; this only completes its idea of what click
    considers a flag.
    """
    real = {opt
            for p in profile_cmd.params
            for opt in (list(getattr(p, "opts", []))
                        + list(getattr(p, "secondary_opts", [])))}
    mentioned = set(re.findall(r"(?<![\w-])--[a-z][a-z0-9-]+", doc_text))
    unknown = mentioned - real
    assert not unknown, f"doc mentions non-existent flag(s): {sorted(unknown)}"


def test_the_two_new_controls_are_both_documented(doc_text):
    for flag in ("--min-cell-count", "--redact-categoricals"):
        assert f"### `{flag}" in doc_text, f"{flag} has no section"


def test_doc_lists_the_residual_risks_the_controls_do_not_cover(doc_text):
    assert "Known limits" in doc_text
    lowered = doc_text.lower()
    for risk in ("id-like", "min/max", "bin edges", "punctuation", "heuristic"):
        assert risk in lowered, f"undocumented residual risk: {risk}"
