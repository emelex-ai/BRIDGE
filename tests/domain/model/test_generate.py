"""End-to-end generation against a real Model.

Every pathway, exercised through the public :meth:`Model.generate` entry point. The
assertions are structural: which fields a pathway populates, what shapes they carry, that
a deterministic run repeats, and what generation reads. It reads a ``BridgeInput`` and,
on a pathway that writes orthography, the ``orth_start`` the caller names the output
language with. It reads nothing else, and each of those is checked from both sides: what
it must respond to, and what it must not. The numeric values are pinned separately by
``tests/domain/test_phon_baseline_equivalence.py`` and the per-step mechanics by
``tests/domain/model/test_decoder_loops.py``.
"""

import pytest
import torch

from bridge.domain.datamodels import (
    BridgeBatch,
    BridgeInput,
    GenerationOutput,
    ModelConfig,
    VocabSpec,
)
from bridge.domain.model import Model
from bridge.domain.model.model import PATHWAY_IO, PATHWAYS, WRITES_ORTH
from bridge.domain.tokenizer import BridgeTokenizer

TOKENIZER = BridgeTokenizer()
WORDS = ["cat", "dog", "elephant"]
LANGUAGES = ["EN", "EN", "EN"]

# Which GenerationOutput fields each pathway is responsible for filling in.
ORTH_FIELDS = ("orth_tokens", "orth_probs")
PHON_FIELDS = ("phon_tokens", "phon_probs", "phon_vecs")
PRODUCES = {
    "o2p": PHON_FIELDS,
    "p2o": ORTH_FIELDS,
    "p2p": PHON_FIELDS,
    "o2o": ORTH_FIELDS,
    "op2op": ORTH_FIELDS + PHON_FIELDS,
}


@pytest.fixture(scope="module")
def model():
    vocab = VocabSpec.from_tokenizer(TOKENIZER)
    model = Model(ModelConfig(vocab=vocab, d_model=32, nhead=2, seed=11))
    model.eval()
    return model


@pytest.fixture(scope="module")
def inputs():
    inputs = TOKENIZER.encode_input(WORDS, LANGUAGES)
    assert inputs.phon is not None
    return inputs


@pytest.fixture(scope="module")
def start():
    return TOKENIZER.orth_start(LANGUAGES)


@pytest.fixture(scope="module")
def generated(model, inputs, start):
    """Each pathway's greedy output for ``inputs`` and ``start``, generated once."""
    return {
        pathway: model.generate(inputs, pathway, orth_start=start, deterministic=True)
        for pathway in PATHWAYS
    }


def fields(output: GenerationOutput) -> dict:
    return {name: getattr(output, name) for name in ("global_encoding", *ORTH_FIELDS, *PHON_FIELDS)}


def same(a, b) -> bool:
    """Compare nested generation output, whose inner tensors are genuinely ragged."""
    if a is None or b is None:
        return a is None and b is None
    if isinstance(a, torch.Tensor):
        return a.shape == b.shape and torch.equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b, strict=True))


@pytest.mark.parametrize("pathway", sorted(PRODUCES))
def test_each_pathway_fills_exactly_its_own_fields(generated, pathway):
    output = generated[pathway]

    assert isinstance(output, GenerationOutput)
    assert output.global_encoding.shape[0] == len(WORDS)
    for field in PRODUCES[pathway]:
        assert getattr(output, field) is not None, f"{pathway} should produce {field}"
    for field in set(ORTH_FIELDS + PHON_FIELDS) - set(PRODUCES[pathway]):
        assert getattr(output, field) is None, f"{pathway} should not produce {field}"


@pytest.mark.parametrize("pathway", sorted(PRODUCES))
def test_generated_batches_keep_their_alignment(generated, pathway):
    """One entry per input word, in every ragged field a pathway produces."""
    output = generated[pathway]

    for field in PRODUCES[pathway]:
        value = getattr(output, field)
        length = value.shape[0] if isinstance(value, torch.Tensor) else len(value)
        assert length == len(WORDS), f"{pathway}/{field}"


@pytest.mark.parametrize("pathway", sorted(PRODUCES))
def test_deterministic_generation_repeats(model, inputs, start, generated, pathway):
    first = generated[pathway]
    second = model.generate(inputs, pathway, orth_start=start, deterministic=True)
    assert first is not second

    for field in PRODUCES[pathway]:
        assert same(getattr(first, field), getattr(second, field)), f"{pathway}/{field}"


def test_an_unknown_pathway_is_rejected(model, inputs, start):
    with pytest.raises(ValueError, match="Invalid pathway"):
        model.generate(inputs, "nonsense", orth_start=start, deterministic=True)


# --- what generation reads: the input ----------------------------------------


@pytest.mark.parametrize("pathway", PATHWAYS)
def test_generation_responds_to_each_modality_it_reads_and_to_no_other(
    model, inputs, start, generated, pathway
):
    """Oracle: ``PATHWAY_IO``, the table of what each pathway reads.

    The rows of one modality are rolled, so every row carries another row's tokens. The
    output must change exactly when the pathway reads that modality. ``global_encoding``
    is the whole of what a decoder sees of the input, so it is what is compared.
    """
    base = generated[pathway]
    for modality in ("orth", "phon"):
        part = getattr(inputs, modality)
        rolled = type(part)(part.ids.roll(1, dims=0), part.pad_mask.roll(1, dims=0))
        assert not torch.equal(rolled.ids, part.ids), "the roll must change the rows"
        other = {"orth": inputs.orth, "phon": inputs.phon, modality: rolled}
        moved = model.generate(BridgeInput(**other), pathway, orth_start=start, deterministic=True)

        changed = not torch.equal(moved.global_encoding, base.global_encoding)
        assert changed == (modality in PATHWAY_IO[pathway][0]), (pathway, modality)


@pytest.mark.parametrize("pathway", PATHWAYS)
def test_a_target_changes_nothing_generation_returns(model, pathway, tmp_path):
    """Oracle: invariant. Two files hold the same inputs beside different targets, in
    another language and of another length. Generating from a batch of either gives the
    same output, field for field, since generation is handed the batch's input and has no
    way to reach its target.

    Read through the dataset on purpose: a dataset that built its inputs from the target
    columns would pass every repetition test and fail this one.
    """
    from bridge.domain.data import BridgeDataset
    from bridge.domain.datamodels import DatasetConfig

    header = "word_raw,language,target_raw,target_language\n"
    rows = [("no me", "ES"), ("la", "ES"), ("the word", "EN")]
    targets = {
        "same": [f"{text},{lang}" for text, lang in rows],
        "other": ["cat,EN", "the long word,EN", "sin son,ES"],
    }
    batches = {}
    for name, target_cells in targets.items():
        path = tmp_path / f"{name}.csv"
        path.write_text(
            header
            + "".join(f"{t},{g},{cell}\n" for (t, g), cell in zip(rows, target_cells, strict=True))
        )
        dataset = BridgeDataset(DatasetConfig(dataset_filepath=str(path)), tokenizer=TOKENIZER)
        batches[name] = dataset[0 : len(rows)]
    assert not same(batches["same"].target.orth.ids, batches["other"].target.orth.ids)
    assert not same(batches["same"].target.phon.ids, batches["other"].target.phon.ids)

    orth_start = TOKENIZER.orth_start(["ES", "ES", "EN"])
    outputs = [
        fields(model.generate(batch.input, pathway, orth_start=orth_start, deterministic=True))
        for batch in batches.values()
    ]
    assert same(outputs[0], outputs[1])
    for field in PRODUCES[pathway]:
        assert outputs[0][field] is not None


def test_generation_takes_an_input_and_refuses_anything_else(model, inputs, start):
    """Oracle: the rule. A target, or a whole batch, is not what an encoder reads, and
    handing one over raises instead of running on whichever tensors fit."""
    target = TOKENIZER.encode_target(WORDS, LANGUAGES)
    for wrong in (target, BridgeBatch(input=inputs, target=target), None, inputs.orth):
        with pytest.raises(TypeError, match="generate reads a BridgeInput, got"):
            model.generate(wrong, "o2o", orth_start=start, deterministic=True)


# --- a modality the input does not carry --------------------------------------


@pytest.mark.parametrize("pathway", PATHWAYS)
@pytest.mark.parametrize("missing", ["orth", "phon"])
def test_a_pathway_reading_a_modality_the_input_lacks_raises_naming_both(
    model, inputs, start, generated, pathway, missing
):
    """Oracle: ``PATHWAY_IO``. An input carrying one modality runs on the pathways that
    read only that one, and every other pathway refuses it at the boundary, naming the
    pathway and the modality, before any encoder runs."""
    carried = "phon" if missing == "orth" else "orth"
    one_sided = BridgeInput(**{carried: getattr(inputs, carried)})

    if missing in PATHWAY_IO[pathway][0]:
        message = f"pathway '{pathway}' reads {missing}, which this input does not carry"
        with pytest.raises(ValueError, match=message):
            model.generate(one_sided, pathway, orth_start=start, deterministic=True)
    else:
        partial = model.generate(one_sided, pathway, orth_start=start, deterministic=True)
        assert same(fields(generated[pathway]), fields(partial))


def test_a_nonword_is_read_through_its_spelling(model):
    """Oracle: ``PATHWAY_IO``. A nonword has no phonology, so it generates on the two
    pathways that read orthography alone and is refused by name on the other three."""
    nonwords = TOKENIZER.encode_input(["zxqvk", "blorf wug"], ["EN", "EN"])
    assert nonwords.phon is None
    orth_start = TOKENIZER.orth_start(["EN", "EN"])

    assert model.generate(nonwords, "o2p", deterministic=True).phon_tokens is not None
    spelled = model.generate(nonwords, "o2o", orth_start=orth_start, deterministic=True)
    assert spelled.orth_tokens.shape[0] == 2
    for pathway in ("p2o", "p2p", "op2op"):
        with pytest.raises(ValueError, match=f"pathway '{pathway}' reads phon"):
            model.generate(nonwords, pathway, orth_start=orth_start, deterministic=True)


# --- what generation reads: the start of orthography --------------------------


@pytest.mark.parametrize("pathway", WRITES_ORTH)
def test_orth_start_is_the_first_two_orthographic_tokens(model, inputs, pathway):
    """Oracle: analytic. Whatever is asked for comes back as columns 0 and 1, so the
    output language is the caller's choice and the input's language is not consulted:
    the input rows are all English, and each is started in a different language."""
    orth_start = TOKENIZER.orth_start(["ES", "EN", None])
    output = model.generate(inputs, pathway, orth_start=orth_start, deterministic=True)
    assert torch.equal(output.orth_tokens[:, :2], orth_start)
    assert output.orth_tokens[:, :2].tolist() == [[8, 0], [7, 0], [6, 0]]


@pytest.mark.parametrize("pathway", WRITES_ORTH)
def test_changing_orth_start_changes_only_what_orthography_follows(
    model, inputs, generated, pathway
):
    """Oracle: invariant. The start feeds the orthographic decoder and nothing else, so a
    different start leaves the encoding and every phonological field bitwise alike."""
    english = fields(generated[pathway])  # every row started in English
    spanish = fields(
        model.generate(
            inputs, pathway, orth_start=TOKENIZER.orth_start(["ES"] * 3), deterministic=True
        )
    )

    assert not same(english["orth_tokens"][:, 0], spanish["orth_tokens"][:, 0])
    assert not same(english["orth_probs"], spanish["orth_probs"])
    for field in ("global_encoding", *PHON_FIELDS):
        assert same(english[field], spanish[field]), field


@pytest.mark.parametrize(
    ("pathway", "carried"),
    [*((pathway, "both") for pathway in WRITES_ORTH), ("p2o", "phon"), ("o2o", "orth")],
)
def test_a_pathway_that_writes_orthography_requires_orth_start(model, inputs, pathway, carried):
    """Oracle: the rule. Nothing about the output is taken from the input in silence, so
    leaving the start out raises, naming the pathway and how to build one, whatever the
    input carries: both modalities, phonemes alone or a spelling alone. The model is
    left in the mode it was in."""
    given = inputs if carried == "both" else BridgeInput(**{carried: getattr(inputs, carried)})
    model.train()
    try:
        with pytest.raises(ValueError) as raised:
            model.generate(given, pathway, deterministic=True)
        assert model.training
    finally:
        model.eval()
    message = str(raised.value)
    assert f"pathway {pathway!r} writes orthography" in message
    assert "orth_start" in message and "BridgeTokenizer.orth_start" in message


@pytest.mark.parametrize("pathway", sorted(set(PATHWAYS) - set(WRITES_ORTH)))
def test_a_pathway_that_writes_no_orthography_does_not_read_orth_start(
    model, inputs, start, pathway
):
    """Oracle: invariant. With a start, with another, or with none, the output is the same."""
    without = fields(model.generate(inputs, pathway, deterministic=True))
    for orth_start in (start, TOKENIZER.orth_start(["ES"] * 3)):
        with_start = fields(
            model.generate(inputs, pathway, orth_start=orth_start, deterministic=True)
        )
        assert same(without, with_start)


def test_orth_start_and_deterministic_are_keyword_only(model, inputs, start):
    """Oracle: the signature. A positional third argument would be read as one or the
    other depending on the order they were written in."""
    with pytest.raises(TypeError, match="positional"):
        model.generate(inputs, "o2o", start, True)
