"""The documentation is executable.

`README.md`, `docs/architecture.md` and `.claude/rules/measurement.md` state facts about the
live system.
Prose rots silently; an assertion does not. Every fact here that a code change could
invalidate is checked against the running system, so drift surfaces as a failing test naming
the file to edit rather than as a document nobody notices has gone wrong.

Facts that cannot be checked mechanically (a measured noise floor, which devices have been
swept) carry a date and the artifact that produced them, in the documents themselves.

The README was outside this net until it had drifted four ways at once: it claimed 87 tests
against 535, four pathways against five, an import list three names short, and a
`TrainingPipeline(...)` example whose argument order and argument names were both wrong, so
the only worked example in the file raised `TypeError`. Prose about intent can rot quietly;
a signature cannot.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

from bridge.core.phonreps import SPECIAL_TOKENS, load_phoneme_table
from bridge.domain.model.model import PATHWAYS
from bridge.domain.tokenizer import BridgeTokenizer

ROOT = Path(__file__).resolve().parents[1]
ARCHITECTURE = ROOT / "docs" / "architecture.md"
README = ROOT / "README.md"
PROJECT_RULES = ROOT / ".claude" / "rules" / "measurement.md"
DECISIONS = ROOT / "docs" / "decisions"

TABLE = load_phoneme_table()
VOCAB_SIZES = BridgeTokenizer().get_vocabulary_sizes()


def flat(text: str) -> str:
    """Collapse whitespace so a phrase split across a line break still matches."""
    return re.sub(r"\s+", " ", text)


@pytest.fixture(scope="module")
def architecture() -> str:
    return ARCHITECTURE.read_text()


@pytest.fixture(scope="module")
def project_rules() -> str:
    return PROJECT_RULES.read_text()


@pytest.fixture(scope="module")
def readme() -> str:
    return README.read_text()


def test_the_pathway_table_lists_exactly_the_real_pathways(architecture):
    """A new pathway absent from the table is a map that misleads.

    Scoped to the table rather than the whole document: a pathway name occurs in several
    places, so a document-wide substring check passes even after its row is deleted.
    """
    table = architecture.split("| pathway | reads | writes |")[1].split("\n\n")[0]
    listed = set(re.findall(r"^\| `(\w+)` \|", table, re.M))
    assert listed == set(PATHWAYS), (
        f"pathway table lists {sorted(listed)}, model.PATHWAYS is {sorted(PATHWAYS)}"
    )


def test_phoneme_table_dimensions_are_documented(architecture, project_rules):
    """Editing phonreps.csv changes every one of these numbers at once."""
    rows, cols = TABLE.multihot.shape
    base = TABLE.base_dim
    assert f"`({rows}, {cols})`" in architecture, f"table is ({rows}, {cols})"
    assert f"{rows} phonemes by {base} phonetic features" in flat(architecture)
    assert f"{len(SPECIAL_TOKENS)} special-token columns" in flat(architecture)
    for doc in (flat(architecture), flat(project_rules)):
        assert f"0 to {rows - 1}" in doc, f"row space upper bound is {rows - 1}"
        assert f"0 to {cols - 1}" in doc, f"feature space upper bound is {cols - 1}"


def test_recorded_fingerprint_matches_the_table(project_rules):
    """The fingerprint is how a checkpoint notices phonreps.csv moved underneath it."""
    recorded = re.search(r"fingerprint: `([0-9a-f]+)`", project_rules)
    assert recorded, "no fingerprint recorded in .claude/rules/measurement.md"
    assert recorded.group(1) == TABLE.fingerprint, (
        f"phonreps.csv now hashes to {TABLE.fingerprint}; update the project rules and "
        f"expect every checkpoint trained against the old table to warn on load"
    )


def test_documented_target_width_matches_the_vocabulary(architecture):
    """Loss targets are one column narrower than the vocabulary: [PAD] is never predicted."""
    width = VOCAB_SIZES["phonological"] - 1
    assert f"(batch, seq, {width})" in flat(architecture), (
        f"documented target width no longer matches the phonological vocabulary ({width})"
    )


def test_every_path_named_in_the_layout_exists(architecture):
    """A renamed or deleted module leaves the map pointing at nothing."""
    block = architecture.split("## Layout")[1].split("```")[1]
    paths = [
        m.group(1)
        for line in block.splitlines()
        if (m := re.match(r"\s*([\w./-]+/[\w./-]*)", line))
    ]
    assert paths, "layout block parsed to nothing; the test, not the repo, is wrong"
    missing = [p for p in paths if not (ROOT / p).exists()]
    assert not missing, f"architecture.md names paths that do not exist: {missing}"


def test_decision_records_and_the_index_agree(architecture):
    """A record nobody indexes is a record nobody reads."""
    on_disk = {p.name for p in DECISIONS.glob("[0-9]*.md")}
    indexed = set(re.findall(r"\(decisions/([^)]+)\)", architecture))
    assert on_disk == indexed, (
        f"only on disk: {sorted(on_disk - indexed)}; only in the index: {sorted(indexed - on_disk)}"
    )


def test_every_decision_record_declares_a_status():
    """Superseding rather than editing only works if status is actually maintained."""
    for record in sorted(DECISIONS.glob("[0-9]*.md")):
        first_lines = record.read_text().splitlines()[:6]
        assert any(line.startswith("Status:") for line in first_lines), (
            f"{record.name} has no Status line"
        )


def test_known_defects_are_still_open(architecture):
    """A closed issue left in the defect list makes the map read as worse than it is."""
    issues = sorted(set(re.findall(r"\*\*#(\d+)\*\*", architecture)))
    if not issues:
        pytest.skip("no issues listed")
    try:
        out = subprocess.run(
            ["gh", "issue", "list", "--state", "open", "--limit", "200", "--json", "number"],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=ROOT,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pytest.skip("gh unavailable")
    if out.returncode != 0:
        pytest.skip(f"gh failed: {out.stderr[:120]}")
    open_numbers = set(re.findall(r'"number":\s*(\d+)', out.stdout))
    closed = [i for i in issues if i not in open_numbers]
    assert not closed, (
        f"issues {closed} are closed but still listed under Known defects in "
        f"architecture.md; remove them and update anything they invalidated"
    )


# --- README ---------------------------------------------------------------


def test_the_readme_import_list_is_the_public_api(readme):
    """The README used to hand-copy a list of exports that fell three names behind.

    Oracle: a differential against ``bridge.__all__``, which is the surface
    ``tests/test_public_api.py`` pins. The README block is the claim; ``__all__`` is the
    fact.
    """
    import bridge

    block = readme.split("from bridge import (")[1].split(")")[0]
    listed = set(re.findall(r"\b([A-Za-z_][A-Za-z0-9_]*)\b", block))
    exported = set(bridge.__all__)
    assert listed == exported, (
        f"only in README: {sorted(listed - exported)}; "
        f"only in bridge.__all__: {sorted(exported - listed)}"
    )


def test_the_readme_usage_example_matches_the_real_signature(readme):
    """The one worked example in the file must be callable.

    Oracle: ``inspect.signature``. The previous example passed ``dataset`` where
    ``training_config`` goes and a ``wandb_wrapper=`` keyword that no longer existed, so a
    reader following the README got ``TypeError`` on their first attempt.
    """
    import inspect

    from bridge import TrainingPipeline

    parameters = set(inspect.signature(TrainingPipeline.__init__).parameters) - {"self"}
    call = readme.split("pipeline = TrainingPipeline(")[1].split("\n)")[0]
    used = set(re.findall(r"^\s*(\w+)=", call, re.M))
    assert used <= parameters, f"README passes {sorted(used - parameters)}, which do not exist"
    required = {
        name
        for name, parameter in inspect.signature(TrainingPipeline.__init__).parameters.items()
        if name != "self" and parameter.default is inspect.Parameter.empty
    }
    assert required <= used, f"README omits required arguments {sorted(required - used)}"


def test_the_readme_pathway_list_is_complete(readme):
    """A pathway missing from the README is a pathway nobody knows they can use.

    Scoped to the sentence that enumerates them, so deleting a name from the list fails
    even though that name still occurs elsewhere in the file.
    """
    sentence = flat(readme).split("pathways, listed in `bridge.PATHWAYS`")[1].split(". ")[0]
    listed = set(re.findall(r"`(\w+)`", sentence)) - {"bridge.PATHWAYS"}
    assert listed == set(PATHWAYS), (
        f"README names {sorted(listed)}, model.PATHWAYS is {sorted(PATHWAYS)}"
    )


def test_the_readme_id_space_table_matches_the_phoneme_table(readme):
    """Both ranges move together when phonreps.csv is edited."""
    rows, cols = TABLE.multihot.shape
    text = flat(readme)
    assert f"0\u2013{rows - 1}" in text, f"row space upper bound is {rows - 1}"
    assert f"0\u2013{cols - 1}" in text, f"feature space upper bound is {cols - 1}"


def test_every_path_the_readme_links_to_exists(readme):
    """A renamed or deleted file leaves the README linking to nothing.

    Markdown link targets plus the unindented roots of the layout tree. The tree's nested
    entries are relative to their parent and are left to
    ``test_every_path_named_in_the_layout_exists``, which covers the same modules through
    ``docs/architecture.md`` where they are written out in full.
    """
    links = set(re.findall(r"\]\((?!https?:|#)([\w./-]+)\)", readme))
    tree = readme.split("## What's in here")[1].split("```")[1]
    roots = {
        m.group(1) for line in tree.splitlines() if (m := re.match(r"([a-z_]+/[\w./-]*)", line))
    }
    paths = links | roots
    assert paths, "README parsed to no paths at all; the test, not the repo, is wrong"
    missing = sorted(p for p in paths if not (ROOT / p).exists())
    assert not missing, f"README names paths that do not exist: {missing}"


def test_the_readme_dependency_claim_matches_pyproject(readme):
    """The README tells a reader what gets installed. It has to be what gets installed."""
    import tomllib

    declared = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["dependencies"]
    names = {re.match(r"[A-Za-z0-9_.-]+", d).group(0).lower() for d in declared}
    sentence = flat(readme).split("Dependencies are ")[1].split(".")[0]
    claimed = {w.strip().lower() for w in re.split(r",|\band\b", sentence) if w.strip()}
    assert claimed == names, f"README claims {sorted(claimed)}, pyproject declares {sorted(names)}"
