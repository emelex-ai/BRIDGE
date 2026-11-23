"""The input reaches the encoders, the target reaches the decoders and the loss, and
neither reaches anything else.

Repetition cannot show this. When the target is the input, both sides of a batch hold the
same strings, so a pipeline that fed the encoder from the target computes the same
numbers and passes every test built on such a batch. Neither batch here is repetition.
In ``PAIRS`` every row's target differs from its input in content, in language and in
length. In ``ROTATED`` every row's target is the next row's input, so the two sides
differ in content and have the widths repetition has: the input one token wider than the
target, in both modalities. A tensor taken from the wrong side only where it happens to
fit is caught there. The rows of each batch differ from one another in length, so each
of its ten tensors is its own.

Two oracles, neither of which reads the pipeline's wiring:

* **Read set.** One tensor of the batch is rolled along the batch dimension, so every row
  holds another row's values. The loss must move exactly when ``READ`` below, written out
  by hand, says the pathway reads that tensor. A tensor wired to the wrong reader moves
  the loss when it should not, and the one it displaced fails to.
* **Hand-wired reference.** ``Model.forward`` is called with tensors picked by hand, per
  pathway, from the tokenizer called on the strings as this file writes them: input
  strings for the encoders, target strings for the decoders. The pipeline is given only
  the dataset loaded from those rows. Its logits, its evaluated loss and metrics, the
  loss its optimizer step reports and the parameters that step leaves must equal the
  reference, bitwise.
"""

import dataclasses
import json
import math

import pytest
import torch

from bridge.application.training.ortho_metrics import calculate_orth_metrics
from bridge.application.training.phon_metrics import calculate_phon_metrics
from bridge.application.training.training_pipeline import TrainingPipeline
from bridge.core.phonreps import load_phoneme_table
from bridge.domain.data import BridgeDataset
from bridge.domain.datamodels import (
    BridgeBatch,
    BridgeInput,
    DatasetConfig,
    ModelConfig,
    TrainingConfig,
    VocabSpec,
)
from bridge.domain.model import Model
from bridge.domain.model.model import PATHWAY_IO, PATHWAYS

# (word_raw, language, target_raw, target_language). No row's target is its input, the
# two sides open in different languages on every row, and the rows differ in length on
# each side, so rolling any tensor of the batch changes it.
PAIRS = [
    ("no", "EN", "no me", "ES"),
    ("the word is gato", "EN EN EN ES", "la", "ES"),
    ("la con", "ES", "the long word", "EN"),
    ("sin", "ES", "no la casa", "EN ES ES"),
    ("me no la", "ES EN ES", "cat", "EN"),
    ("elephant", "EN", "son", "ES"),
]
# Each row's target is the next row's input. Nothing is repetition, and the widths are
# repetition's: an input ends in [EOS] and a target does not, so the input is one token
# wider in both modalities.
ROTATED = [
    (text, languages, *PAIRS[(row + 1) % len(PAIRS)][:2])
    for row, (text, languages, _, _) in enumerate(PAIRS)
]
BATCHES = {"unequal": PAIRS, "one_wider": ROTATED}

TENSORS = (
    "input.orth.ids",
    "input.orth.pad_mask",
    "input.phon.ids",
    "input.phon.pad_mask",
    "target.orth.ids",
    "target.orth.pad_mask",
    "target.orth.labels",
    "target.phon.ids",
    "target.phon.pad_mask",
    "target.phon.labels",
)

# What each pathway reads of a batch, written out by hand: the ids and mask of each
# modality it reads from the input, and the ids, mask and labels of each modality it
# writes from the target.
INPUT_ORTH = {"input.orth.ids", "input.orth.pad_mask"}
INPUT_PHON = {"input.phon.ids", "input.phon.pad_mask"}
TARGET_ORTH = {"target.orth.ids", "target.orth.pad_mask", "target.orth.labels"}
TARGET_PHON = {"target.phon.ids", "target.phon.pad_mask", "target.phon.labels"}
READ = {
    "o2p": INPUT_ORTH | TARGET_PHON,
    "p2o": INPUT_PHON | TARGET_ORTH,
    "o2o": INPUT_ORTH | TARGET_ORTH,
    "p2p": INPUT_PHON | TARGET_PHON,
    "op2op": INPUT_ORTH | INPUT_PHON | TARGET_ORTH | TARGET_PHON,
}
# Labels are read by the loss and never by the model.
LABELS = {"target.orth.labels", "target.phon.labels"}


@pytest.fixture(scope="module")
def datasets(tmp_path_factory, shared_tokenizer):
    """Each batch of ``BATCHES`` as a dataset loaded from a csv of its rows."""
    directory = tmp_path_factory.mktemp("pairs")
    loaded = {}
    for number, (name, rows) in enumerate(BATCHES.items()):
        path = directory / f"{number}.csv"
        path.write_text(
            "word_raw,language,target_raw,target_language\n"
            + "".join(",".join(row) + "\n" for row in rows)
        )
        loaded[name] = BridgeDataset(
            DatasetConfig(dataset_filepath=str(path)), tokenizer=shared_tokenizer
        )
    return loaded


@pytest.fixture(scope="module")
def pairs(datasets):
    return datasets["unequal"]


@pytest.fixture(scope="module")
def make(pairs, tmp_path_factory):
    vocab = VocabSpec.from_tokenizer(pairs.tokenizer)
    artifacts = str(tmp_path_factory.mktemp("artifacts"))

    def build(pathway: str) -> TrainingPipeline:
        return TrainingPipeline(
            model=Model(ModelConfig(vocab=vocab, d_model=32, nhead=2, seed=5)),
            training_config=TrainingConfig(training_pathway=pathway, model_artifacts_dir=artifacts),
        )

    return build


def tensor_of(batch: BridgeBatch, name: str) -> torch.Tensor:
    side, modality, field = name.split(".")
    return getattr(getattr(getattr(batch, side), modality), field)


def with_rolled(batch: BridgeBatch, name: str) -> BridgeBatch:
    """``batch`` with the one tensor ``name`` rolled a row along the batch dimension."""
    side, modality, field = name.split(".")
    part = getattr(getattr(batch, side), modality)
    rolled = getattr(part, field).roll(1, dims=0)
    assert not torch.equal(rolled, getattr(part, field)), f"rolling {name} changes nothing"
    new_side = dataclasses.replace(
        getattr(batch, side), **{modality: dataclasses.replace(part, **{field: rolled})}
    )
    return dataclasses.replace(batch, **{side: new_side})


def differing(left, right) -> int:
    """How many tensors of two aligned runs are not bitwise equal."""
    return sum(not torch.equal(a, b) for a, b in zip(left, right, strict=True))


def loss_and_logits(pipeline: TrainingPipeline, batch: BridgeBatch):
    logits = pipeline.forward(batch)
    return pipeline.compute_loss(logits, batch.target)["loss"], logits


def sides_as_written(tokenizer, rows):
    """The two sides of ``rows``, each from its own two columns as this file writes them.
    Nothing here passes through the dataset, so a row stored or read the wrong way round
    cannot move the reference with it."""
    inputs = tokenizer.encode_input([row[0] for row in rows], [row[1] for row in rows])
    target = tokenizer.encode_target([row[2] for row in rows], [row[3] for row in rows])
    return inputs, target


def hand_wired(model: Model, pathway: str, inputs, target) -> dict[str, torch.Tensor]:
    """``Model.forward`` with its keyword arguments spelled out per pathway."""
    orth_enc = {"orth_enc_input": inputs.orth.ids, "orth_enc_pad_mask": inputs.orth.pad_mask}
    phon_enc = {"phon_enc_input": inputs.phon.ids, "phon_enc_pad_mask": inputs.phon.pad_mask}
    orth_dec = {"orth_dec_input": target.orth.ids, "orth_dec_pad_mask": target.orth.pad_mask}
    phon_dec = {"phon_dec_input": target.phon.ids, "phon_dec_pad_mask": target.phon.pad_mask}
    kwargs = {
        "o2p": orth_enc | phon_dec,
        "p2o": phon_enc | orth_dec,
        "o2o": orth_enc | orth_dec,
        "p2p": phon_enc | phon_dec,
        "op2op": orth_enc | phon_enc | orth_dec | phon_dec,
    }[pathway]
    return model(task=pathway, **kwargs)


def hand_loss(logits: dict[str, torch.Tensor], target, vocab: VocabSpec) -> torch.Tensor:
    """Cross entropy against the target's labels, phonology first, as the pipeline sums it."""
    cross_entropy = torch.nn.functional.cross_entropy
    terms = []
    if "phon" in logits:
        terms.append(
            cross_entropy(logits["phon"], target.phon.labels, ignore_index=vocab.phon_pad_id)
        )
    if "orth" in logits:
        terms.append(
            cross_entropy(logits["orth"], target.orth.labels, ignore_index=vocab.orth_pad_id)
        )
    return torch.stack(terms).sum()


def hand_metrics(logits: dict[str, torch.Tensor], target, vocab: VocabSpec) -> dict:
    """The metric helpers, each pinned analytically in its own file, on logits and labels
    picked by hand."""
    metrics: dict = {}
    if "phon" in logits:
        metrics |= calculate_phon_metrics(
            logits,
            target.phon.labels,
            load_phoneme_table(device=target.phon.labels.device).phonetic_features,
            phon_pad_id=vocab.phon_pad_id,
        )
    if "orth" in logits:
        metrics |= calculate_orth_metrics(
            logits,
            target.orth.labels,
            orth_pad_id=vocab.orth_pad_id,
            orth_language_ids=vocab.orth_language_ids,
        )
    return metrics


# --- the premises ------------------------------------------------------------


def test_the_batch_is_not_repetition(pairs):
    """Every claim below rests on input and target being distinguishable, tensor by tensor."""
    batch = pairs[0 : len(pairs)]
    assert all(e.input != e.target for e in pairs.examples)
    assert all(e.input_languages[0] != e.target_languages[0] for e in pairs.examples)
    assert batch.input.orth.ids.shape[1] != batch.target.orth.ids.shape[1] + 1
    assert batch.input.phon.ids.shape[1] != batch.target.phon.ids.shape[1] + 1
    assert {name for name in TENSORS if tensor_of(batch, name) is not None} == set(TENSORS)


def test_the_rotated_batch_has_the_widths_of_repetition_and_is_not_repetition(datasets):
    """The premise of every test run on ``ROTATED``: a tensor of one side fits where the
    other side's belongs, cut by its last column, and holds other values."""
    rotated = datasets["one_wider"]
    batch = rotated[0 : len(rotated)]
    assert all(e.input != e.target for e in rotated.examples)
    for modality in ("orth", "phon"):
        read, fed = getattr(batch.input, modality), getattr(batch.target, modality)
        assert read.ids.shape[1] == fed.ids.shape[1] + 1
        assert not torch.equal(read.ids[:, :-1], fed.ids)
        assert not torch.equal(read.pad_mask[:, :-1], fed.pad_mask)


def test_the_hand_written_read_table_covers_every_pathway_and_agrees_with_pathway_io():
    """``READ`` is the oracle, so it is checked against the one table the library keeps:
    a pathway added there, or changed, has to be written out here too."""
    assert set(READ) == set(PATHWAYS)
    for pathway, (reads, writes) in PATHWAY_IO.items():
        from_table = {f"input.{m}.{f}" for m in reads for f in ("ids", "pad_mask")}
        from_table |= {f"target.{m}.{f}" for m in writes for f in ("ids", "pad_mask", "labels")}
        assert READ[pathway] == from_table, pathway


# --- read set ----------------------------------------------------------------


@pytest.mark.parametrize("rows", BATCHES)
@pytest.mark.parametrize("pathway", PATHWAYS)
def test_the_loss_moves_exactly_when_a_tensor_the_pathway_reads_is_rolled(
    datasets, make, pathway, rows
):
    """Oracle: ``READ``, hand-written. Every one of the ten tensors is rolled in turn.

    The loss is compared bitwise: a tensor that is not read cannot change it at all. The
    logits are compared too, against the same set without the labels, which says the
    model itself never sees what it is scored against.
    """
    pipeline = make(pathway)
    pipeline.model.eval()
    batch = datasets[rows][0 : len(datasets[rows])]

    moved_loss, moved_logits = set(), set()
    with torch.no_grad():
        base_loss, base_logits = loss_and_logits(pipeline, batch)
        again, _ = loss_and_logits(pipeline, batch)
        assert torch.equal(again, base_loss), (
            "the loss must repeat, or nothing below means anything"
        )
        for name in TENSORS:
            loss, logits = loss_and_logits(pipeline, with_rolled(batch, name))
            if not torch.equal(loss, base_loss):
                moved_loss.add(name)
            if any(not torch.equal(logits[key], base_logits[key]) for key in base_logits):
                moved_logits.add(name)

    assert moved_loss == READ[pathway], (
        f"{pathway}: moved the loss but should not be read {sorted(moved_loss - READ[pathway])}; "
        f"should be read but did not move it {sorted(READ[pathway] - moved_loss)}"
    )
    assert moved_logits == READ[pathway] - LABELS


# --- hand-wired reference ------------------------------------------------------


@pytest.mark.parametrize("rows", BATCHES)
@pytest.mark.parametrize("pathway", PATHWAYS)
def test_the_pipeline_equals_model_forward_wired_by_hand_from_the_rows_own_strings(
    datasets, make, pathway, rows
):
    """Oracle: reference implementation. The encoders get the input strings' tokens and
    the decoders the target strings', chosen by hand here for each pathway.

    The pipeline is given only the dataset. Its logits through ``forward``, and the loss
    and the metrics ``evaluate`` reports, must equal the reference bitwise, so whatever
    stands between a row's strings and the model, the dataset and the pipeline both, has
    each side in its place.
    """
    pipeline, dataset = make(pathway), datasets[rows]
    model, vocab = pipeline.model, pipeline.model.model_config.vocab
    inputs, target = sides_as_written(dataset.tokenizer, BATCHES[rows])

    model.eval()
    with torch.no_grad():
        reference = hand_wired(model, pathway, inputs, target)
        got = pipeline.forward(dataset[0 : len(dataset)])
    assert got.keys() == reference.keys() == set(PATHWAY_IO[pathway][1])
    for key in reference:
        assert got[key].shape == reference[key].shape
        assert torch.equal(got[key], reference[key]), f"{pathway}/{key}"

    evaluated = pipeline.evaluate(dataset, slice(0, len(dataset)))
    assert torch.equal(evaluated["loss"], hand_loss(reference, target, vocab))
    expected = hand_metrics(reference, target, vocab)
    assert expected, "the pathway writes something, so it has metrics"
    assert {key: evaluated[key] for key in expected} == expected
    assert evaluated.keys() - expected.keys() <= {
        "loss",
        "orth_loss",
        "phon_loss",
        "input",
        "target",
    }


@pytest.mark.parametrize("rows", BATCHES)
@pytest.mark.parametrize("pathway", PATHWAYS)
def test_an_optimizer_step_reports_the_loss_of_model_forward_wired_by_hand_and_trains_on_it(
    datasets, make, pathway, rows
):
    """Oracle: reference implementation, in train mode. ``single_step`` is the path that
    trains. A second pipeline, built from the same seed, takes its step by hand:
    ``Model.forward`` on tensors picked by hand, ``hand_loss``, ``backward``, and its
    optimizer's ``step``.

    The loss ``single_step`` reports must equal the reference's, and so must every
    parameter it leaves, bitwise. The loss shows what the step reports. The parameters
    show what it ran ``backward`` on: a step that trains on anything but the loss it
    reports leaves other parameters.

    Dropout draws from the global generator, so each step runs straight after the same
    seed, on a model in the same state.
    """
    stepped, by_hand, dataset = make(pathway), make(pathway), datasets[rows]
    vocab = by_hand.model.model_config.vocab
    inputs, target = sides_as_written(dataset.tokenizer, BATCHES[rows])
    start = [parameter.detach().clone() for parameter in by_hand.model.parameters()]
    assert differing(start, stepped.model.parameters()) == 0, "one seed, one starting point"

    by_hand.model.train()
    torch.manual_seed(0)
    reference = hand_loss(hand_wired(by_hand.model, pathway, inputs, target), target, vocab)
    reference.backward()
    by_hand.optimizer.step()
    assert differing(start, by_hand.model.parameters()) > 0, (
        "the step taken by hand must move parameters, or equal parameters show nothing"
    )

    torch.manual_seed(0)
    reported = stepped.single_step(dataset, slice(0, len(dataset)))["loss"]
    assert torch.equal(reported, reference.detach())
    for (name, got), expected in zip(
        stepped.model.named_parameters(), by_hand.model.parameters(), strict=True
    ):
        assert torch.equal(got, expected), f"{pathway}/{name}"


def test_evaluate_names_the_rows_it_scored(pairs, make):
    """Oracle: the rows as this file writes them. The slice starts past row 0, so the
    dataset's first rows, or the slice moved along by one, are other strings."""
    reported = make("o2p").evaluate(pairs, slice(2, 5), calculate_metrics=False)
    assert json.loads(reported["input"]) == [row[0] for row in PAIRS[2:5]]
    assert json.loads(reported["target"]) == [row[2] for row in PAIRS[2:5]]


@pytest.mark.parametrize("pathway", PATHWAYS)
def test_the_reference_wired_the_wrong_way_round_gives_other_numbers(pairs, make, pathway):
    """The control for the test above: on this batch a swap is visible.

    The same hand wiring, with each side built from the other side's strings, must
    disagree with the pipeline. If it agreed, the batch could not tell the two sides
    apart and the equality above would say nothing about which is which.
    """
    pipeline = make(pathway)
    model, tokenizer, examples = pipeline.model, pairs.tokenizer, pairs.examples
    swapped_inputs = tokenizer.encode_input(
        [e.target for e in examples], [e.target_languages for e in examples]
    )
    swapped_target = tokenizer.encode_target(
        [e.input for e in examples], [e.input_languages for e in examples]
    )

    model.eval()
    with torch.no_grad():
        swapped = hand_wired(model, pathway, swapped_inputs, swapped_target)
        got = pipeline.forward(pairs[0 : len(pairs)])
    for key in got:
        assert got[key].shape != swapped[key].shape or not torch.equal(got[key], swapped[key])


# --- the whole path on a batch that is not repetition ---------------------------


@pytest.mark.parametrize("pathway", PATHWAYS)
def test_a_batch_whose_sides_differ_trains_evaluates_and_generates(pairs, make, pathway):
    """Input and target differ in content, language and length, and every pathway runs.

    Oracles. Training: a step moves parameters and names both sides in what it reports,
    and six steps on the same rows bring the evaluated loss down by a tenth or more
    (measured: to between 0.63 and 0.81 of its start, by pathway), so the steps train.
    That they train against the target is the step reference test's to show, by the
    parameters a step leaves: the loss falls as far with the two sides exchanged.
    Evaluation: the metrics of exactly the modalities the pathway writes. Generation: one
    output row per input row, opening with the start built from the target's first-word
    languages.
    """
    pipeline = make(pathway)
    model, tokenizer, examples = pipeline.model, pairs.tokenizer, pairs.examples
    rows = slice(0, len(pairs))
    reads, writes = PATHWAY_IO[pathway]

    start_loss = float(pipeline.evaluate(pairs, rows, calculate_metrics=False)["loss"])
    before = [parameter.detach().clone() for parameter in model.parameters()]
    first = pipeline.single_step(pairs, rows, calculate_metrics=True)
    assert math.isfinite(float(first["loss"]))
    assert differing(before, model.parameters()) > 0, "an optimizer step must move parameters"
    assert json.loads(first["input"]) == [e.input for e in examples]
    assert json.loads(first["target"]) == [e.target for e in examples]

    for _ in range(5):
        last = pipeline.single_step(pairs, rows)
    end_loss = float(pipeline.evaluate(pairs, rows, calculate_metrics=False)["loss"])
    assert end_loss < 0.9 * start_loss, f"{pathway}: {start_loss:.4f} -> {end_loss:.4f}"
    assert math.isfinite(float(last["loss"]))

    scored = pipeline.evaluate(pairs, rows)
    assert ("phon_feature_accuracy" in scored) == ("phon" in writes)
    assert ("letter_wise_accuracy" in scored) == ("orth" in writes)
    assert all(
        math.isfinite(float(scored[key])) for key in scored if key not in ("input", "target")
    )

    batch = pairs[rows]
    start = tokenizer.orth_start([e.target_languages[0] for e in examples])
    assert torch.equal(start, batch.target.orth.ids[:, :2])
    assert not torch.equal(start, batch.input.orth.ids[:, :2]), "the languages must differ"
    output = model.generate(batch.input, pathway, orth_start=start, deterministic=True)
    assert output.global_encoding.shape[0] == len(pairs)
    if "orth" in writes:
        assert torch.equal(output.orth_tokens[:, :2], start)
    else:
        assert output.orth_tokens is None
    assert (output.phon_tokens is not None) == ("phon" in writes)
    if "phon" in writes:
        assert len(output.phon_tokens) == len(pairs)


# --- the boundary ------------------------------------------------------------


@pytest.mark.parametrize("pathway", PATHWAYS)
@pytest.mark.parametrize("missing", ["orth", "phon"])
def test_forward_refuses_a_pathway_whose_modality_the_input_lacks(pairs, make, pathway, missing):
    """Oracle: ``PATHWAY_IO``. The message names the pathway and the modality, and a
    pathway that does not read the missing modality gives the logits it gives with it."""
    pipeline = make(pathway)
    pipeline.model.eval()
    batch = pairs[0 : len(pairs)]
    carried = "phon" if missing == "orth" else "orth"
    one_sided = dataclasses.replace(
        batch, input=BridgeInput(**{carried: getattr(batch.input, carried)})
    )

    with torch.no_grad():
        if missing in PATHWAY_IO[pathway][0]:
            message = f"pathway '{pathway}' reads {missing}, which this input does not carry"
            with pytest.raises(ValueError, match=message):
                pipeline.forward(one_sided)
        else:
            whole, partial = pipeline.forward(batch), pipeline.forward(one_sided)
            assert all(torch.equal(whole[key], partial[key]) for key in whole)


@pytest.mark.parametrize("pathway", PATHWAYS)
def test_an_input_where_a_target_belongs_is_refused(pairs, make, pathway):
    """Oracle: the rule. An input has no labels, so it cannot be scored against: the loss
    and the metrics raise, naming what is missing, and nothing runs on tensors that
    happen to fit."""
    pipeline = make(pathway)
    pipeline.model.eval()
    batch = pairs[0 : len(pairs)]
    with torch.no_grad():
        logits = pipeline.forward(batch)
        with pytest.raises(AttributeError, match="'Tokens' object has no attribute 'labels'"):
            pipeline.compute_loss(logits, batch.input)
        with pytest.raises(AttributeError, match="'Tokens' object has no attribute 'labels'"):
            pipeline.compute_metrics(logits, batch.input)
    with pytest.raises(TypeError, match="BridgeBatch.target must be a BridgeTarget"):
        BridgeBatch(input=batch.input, target=batch.input)
