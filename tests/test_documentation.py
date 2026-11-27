"""`README.md`, `docs/architecture.md` and `.claude/rules/measurement.md` state facts about the
live system. Every fact here that a code change could invalidate is checked against the
running system, so drift is a failing test naming the file to edit.

Facts that cannot be checked mechanically (a measured noise floor, which devices have been
swept) carry a date and the artifact that produced them, in the documents themselves.
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
    """Scoped to the table rather than the whole document: a pathway name occurs in several
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


def test_the_documented_tensors_are_the_tensors_of_a_live_batch(architecture):
    """The map lists every tensor of a batch with its rank and dtype, and no other.

    Oracle: a batch built by the live tokenizer. A field that is renamed, added or
    removed, or whose rank or dtype moves, leaves the listing naming something a batch
    does not hold.
    """
    import dataclasses

    from bridge import BridgeBatch

    listed = re.findall(
        r"^(input|target)\.(orth|phon)\.(\w+)\s+\(([^)]*)\)\s+(int64|bool)\b", architecture, re.M
    )
    assert listed, "architecture.md parsed to no tensor listing; the test is wrong"
    documented = {}
    for side, modality, field, shape, dtype in listed:
        axes = [axis.strip() for axis in shape.split(",")]
        documented[f"{side}.{modality}.{field}"] = (
            len(axes),
            dtype,
            int(axes[-1]) if len(axes) == 3 else None,
        )

    tokenizer = BridgeTokenizer()
    batch = BridgeBatch(
        input=tokenizer.encode_input(["cat", "no me"], ["EN", "ES"]),
        target=tokenizer.encode_target(["el gato", "dog"], ["ES", "EN"]),
    )
    live = {}
    for side in ("input", "target"):
        for modality in ("orth", "phon"):
            part = getattr(getattr(batch, side), modality)
            for field in dataclasses.fields(part):
                tensor = getattr(part, field.name)
                live[f"{side}.{modality}.{field.name}"] = (
                    tensor.dim(),
                    str(tensor.dtype).removeprefix("torch."),
                    tensor.shape[-1] if tensor.dim() == 3 else None,
                )
    assert documented == live, (
        f"only documented: {sorted(set(documented) - set(live))}; "
        f"only on a live batch: {sorted(set(live) - set(documented))}; "
        f"differing: {sorted(k for k in set(live) & set(documented) if live[k] != documented[k])}"
    )


def test_the_type_tables_name_every_field_of_every_type(architecture, readme):
    """Oracle: ``dataclasses.fields`` of the exported classes."""
    import dataclasses

    import bridge

    for name, document in (("docs/architecture.md", architecture), ("README.md", readme)):
        table = document.split("| type | holds | read by |")[1].split("\n\n")[0]
        rows = dict(re.findall(r"^\| `(\w+)` \| ([^|]*) \|", table, re.M))
        assert set(rows) == {"Tokens", "Target", "BridgeInput", "BridgeTarget", "BridgeBatch"}, (
            f"{name} tabulates {sorted(rows)}"
        )
        for type_name, holds in rows.items():
            fields = {field.name for field in dataclasses.fields(getattr(bridge, type_name))}
            named = set(re.findall(r"`(\w+)`", holds))
            assert fields <= named, (
                f"{name}: the {type_name} row names {sorted(named)}, the class has {sorted(fields)}"
            )


def test_the_documented_generate_signature_is_the_real_one(architecture):
    """The map writes out how ``Model.generate`` is called, and a caller copies it.

    Oracle: ``inspect.signature``. A parameter that is added, removed, renamed, given a
    default or moved across the keyword-only bar leaves the map showing a call that is
    not the method's.
    """
    import inspect

    from bridge import Model

    parts, keyword_only = [], False
    for name, parameter in inspect.signature(Model.generate).parameters.items():
        if name == "self":
            continue
        if parameter.kind is inspect.Parameter.KEYWORD_ONLY and not keyword_only:
            parts.append("*")
            keyword_only = True
        has_default = parameter.default is not inspect.Parameter.empty
        parts.append(f"{name}={parameter.default!r}" if has_default else name)
    live = f"Model.generate({', '.join(parts)})"

    documented = re.findall(r"`(Model\.generate\([^`]*\*[^`]*\))`", flat(architecture))
    assert documented, "architecture.md parsed to no generate signature; the test is wrong"
    assert set(documented) == {live}, f"the map shows {documented}, the method is {live}"


def test_the_size_tables_give_the_real_defaults(architecture, readme):
    """Oracle: the field defaults of ``ModelConfig``. A default that moves, or a size that
    is added or removed, leaves a table telling a reader the model is another size.
    """
    from bridge import ModelConfig

    live = {
        name: field.default
        for name, field in ModelConfig.model_fields.items()
        if field.annotation is int
    }
    assert "d_embedding" in live, "ModelConfig parsed to no sizes; the test is wrong"
    for name, document in (("docs/architecture.md", architecture), ("README.md", readme)):
        table = document.split("| size | default |")[1].split("\n\n")[0]
        documented = {}
        for names, default in re.findall(r"^\| ([^|]+) \| (\d+) \|", table, re.M):
            for size in re.findall(r"`(\w+)`", names):
                documented[size] = int(default)
        assert documented == live, (
            f"{name}: only documented {sorted(set(documented) - set(live))}; "
            f"only on ModelConfig {sorted(set(live) - set(documented))}; "
            f"differing {sorted(k for k in set(live) & set(documented) if live[k] != documented[k])}"
        )


def test_the_vocabulary_tables_give_the_real_ids(architecture, readme):
    """The map and the README each tabulate the orthographic vocabulary by id range.

    Oracle: a live ``CharacterTokenizer``. An id is a row of a checkpoint's embedding, so
    a table one off sends a reader to the wrong row. A character added to the vocabulary
    and not to both tables fails here too.
    """
    import string

    from bridge import CharacterTokenizer
    from bridge.domain.tokenizer.character_tokenizer import SPANISH_CHARACTERS

    tok = CharacterTokenizer()
    groups = [
        tok.special_tokens,
        tok.language_tokens,
        list(string.printable),
        list(SPANISH_CHARACTERS),
    ]
    assert [entry for group in groups for entry in group] == tok.vocab
    bounds, start = [], 0
    for group in groups:
        bounds.append((start, start + len(group) - 1))
        start += len(group)

    for name, document, heading in (
        ("docs/architecture.md", architecture, "## The orthographic vocabulary"),
        ("README.md", readme, "### Characters"),
    ):
        section = document.split(heading)[1].split("\n#")[0]
        size = re.search(r"holds (\d+) entries", flat(section))
        assert size and int(size.group(1)) == len(tok.vocab), f"{name}: vocabulary size"
        rows = re.findall(r"^\| (\d+) to (\d+) \| (.*) \|$", section, re.M)
        assert [(int(first), int(last)) for first, last, _ in rows] == bounds, name
        specials, languages, printable, spanish = (cell for _, _, cell in rows)
        assert re.findall(r"`(\[\w+\])`", specials) == tok.special_tokens, name
        assert re.findall(r"`([^`]+)`", languages) == tok.language_tokens, name
        assert "`string.printable`" in printable, name
        assert f"the {len(string.printable)} " in printable, name
        assert sorted(ch for ch in spanish if not ch.isascii()) == sorted(SPANISH_CHARACTERS), name


def test_the_recorded_parameter_count_is_a_default_models(project_rules):
    """The project rules say how many parameter tensors and scalars a default model has,
    which is what a differential reports "n of n tensors" against.

    Oracle: a model built from the default ``ModelConfig``.
    """
    from bridge import Model, ModelConfig, VocabSpec

    recorded = re.search(
        r"A default model has (\d+) parameter tensors and (\d+) scalars", flat(project_rules)
    )
    assert recorded, "no parameter count recorded in .claude/rules/measurement.md"
    parameters = list(
        Model(ModelConfig(vocab=VocabSpec.from_tokenizer(BridgeTokenizer()))).parameters()
    )
    live = (len(parameters), sum(parameter.numel() for parameter in parameters))
    assert tuple(map(int, recorded.groups())) == live, (
        f"the rules record {recorded.groups()}, a default model has {live}"
    )


def test_every_path_named_in_the_layout_exists(architecture):
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


def test_the_readme_import_list_is_the_public_api(readme):
    """Oracle: a differential against ``bridge.__all__``, which is the surface
    ``tests/test_public_api.py`` pins.
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
    """Oracle: ``inspect.signature``."""
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


def test_the_readme_multilingual_example_runs(readme):
    """The tokenizer example under "Multilingual support" is run as written.

    Oracle: the interpreter. An argument ``encode_input`` does not take raises, and a
    word missing from the lexicon of the language the example gives it leaves the input
    without phonology, so every input the block assigns must carry both modalities.
    """
    block = readme.split("## Multilingual support")[1].split("```python")[1].split("```")[0]
    assert "tok.encode_input(" in block, "README parsed to no example; the test is wrong"
    namespace: dict = {}
    exec(block, namespace)
    assert namespace["words"].phon is not None and namespace["mixed"].phon is not None
    # The comment under the last call shows what it returns.
    shown = block.split("tok.decode_with_languages(")[1].split("# ")[1].strip()
    ids = namespace["mixed"].orth.ids.tolist()
    assert shown == repr(namespace["tok"].decode_with_languages(ids))


def test_every_readme_python_example_runs_as_written(readme, tmp_path, monkeypatch):
    """Each python block of the README runs alone, top to bottom, as a reader pastes it.

    Oracle: the interpreter, and the results the README itself shows. A bare expression
    followed directly by a comment line is a claim that the comment is its ``repr``, so
    each one is evaluated and compared.

    The usage example reads ``my_pairs.csv``, which is the csv block under "Dataset
    files", so that block is checked as a file that loads and trains. CUDA is hidden, so
    the example's placement line takes its CPU branch whichever machine runs the suite.
    """
    import ast

    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    blocks = re.findall(r"^```python\n(.*?)^```", readme, re.M | re.S)
    data_file = readme.split("### Dataset files")[1].split("```csv\n")[1].split("```")[0]
    assert len(blocks) >= 6, f"README parsed to {len(blocks)} python blocks; the test is wrong"

    shown_results = 0
    for number, block in enumerate(blocks):
        workdir = tmp_path / str(number)
        workdir.mkdir()
        (workdir / "my_pairs.csv").write_text(data_file)
        monkeypatch.chdir(workdir)
        lines = block.splitlines()
        namespace: dict = {}
        for node in ast.parse(block).body:
            source = ast.get_source_segment(block, node)
            if not isinstance(node, ast.Expr):
                exec(compile(ast.Module([node], []), "README.md", "exec"), namespace)
                continue
            value = eval(compile(ast.Expression(node.value), "README.md", "eval"), namespace)
            following = lines[node.end_lineno] if node.end_lineno < len(lines) else ""
            if following.startswith("# "):
                shown_results += 1
                assert following[2:].strip() == repr(value), (
                    f"README shows `{source}` as {following[2:].strip()}, it returns {value!r}"
                )
    assert shown_results >= 7, f"README parsed to {shown_results} shown results; the test is wrong"


def test_the_readme_pathway_list_is_complete(readme):
    """Scoped to the sentence that enumerates them, so deleting a name from the list fails
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
    """Markdown link targets plus the unindented roots of the layout tree. The tree's nested
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
    import tomllib

    declared = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["dependencies"]
    names = {re.match(r"[A-Za-z0-9_.-]+", d).group(0).lower() for d in declared}
    sentence = flat(readme).split("Dependencies are ")[1].split(".")[0]
    claimed = {w.strip().lower() for w in re.split(r",|\band\b", sentence) if w.strip()}
    assert claimed == names, f"README claims {sorted(claimed)}, pyproject declares {sorted(names)}"
