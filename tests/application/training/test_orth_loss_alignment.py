"""The orthographic loss must score the positions the decoder actually predicts.

The character tokenizer lays each word out as ``[LANG, BOS, ...chars, EOS, PAD...]`` for
the encoder and ``[LANG, BOS, ...chars, PAD...]`` for the decoder, so the decoder input is
exactly one position shorter than the encoder input. The decoder emits one distribution per
decoder-input position, which means the training target has to be the encoder ids shifted by
a single position: ``enc_input_ids[:, 1:]``.

That alignment is decidable rather than a matter of taste. At inference
``orthography_decoder_loop`` seeds the decoder with a lone ``[BOS]`` and the first token it
samples has to be the first character of the word. Training the ``[BOS]`` position against
anything else teaches the model to skip a character, so the pair at that position is the
whole argument, and ``test_bos_is_trained_to_predict_the_first_character`` pins it.

Dropping two positions instead of one leaves the target a position short of the logits, and
``CrossEntropyLoss`` refuses the batch outright. That is issue #225: ``p2o`` and ``op2op``
raise on every batch and cannot train at all, while ``o2p`` and ``p2p`` are untouched. The
untouched pathways appear here as a control, so a failure in this file can be read as a
statement about the orthographic target rather than about the harness.

Two of these tests carry the alignment claim and the rest do not. Against a wrong but
shape-compatible target such as ``enc_input_ids[:, :-1]``, the pathway test, the width sweep
and the descent test all pass; only ``test_bos_is_trained_to_predict_the_first_character``
and ``test_a_perfect_predictor_scores_one_on_the_orth_metrics`` go red. Read a failure in
either of those as a statement about the alignment, and a failure elsewhere as a statement
that something coarser is wrong.

``test_mismatched_shapes_raise_a_valueerror_naming_both`` is not about #225 at all. It pins
the diagnostic added alongside the fix, and it stays red under a fix that corrects the slice
without adding the check.
"""

import contextlib

import pytest
import torch

from bridge.application.training.ortho_metrics import calculate_orth_metrics
from bridge.application.training.training_pipeline import TrainingPipeline
from bridge.domain.data import BridgeDataset
from bridge.domain.datamodels import (
    DatasetConfig,
    MetricsConfig,
    ModelConfig,
    TrainingConfig,
    VocabSpec,
)
from bridge.domain.model import Model
from bridge.infra.metrics.metrics_logger import STDOutMetricsLogger

DATA_CSV = "tests/domain/model/data/data.csv"

SILENT_METRICS = MetricsConfig(
    batch_metrics=False,
    training_metrics=False,
    validation_metrics=False,
    modes=[],
    filename=None,
)

# Hand-computed from the tokenizer layout for "long". Encoder ids are
# ['--', '[BOS]', 'l', 'o', 'n', 'g', '[EOS]'] and decoder ids are
# ['--', '[BOS]', 'l', 'o', 'n', 'g'], where '--' is the unspecified-language token that
# opens every sequence. Reading the pairs down the column gives the teacher-forcing
# contract: seeing [BOS] the model must produce 'l'.
ALIGNMENT_FOR_LONG = [
    ("--", "[BOS]"),
    ("[BOS]", "l"),
    ("l", "o"),
    ("o", "n"),
    ("n", "g"),
    ("g", "[EOS]"),
]

# Six batches whose decoder widths are 6, 4, 7, 8, 10 and 13. One batch could agree by
# luck; a sweep that never varies the width cannot tell a correct slice from an off-by-one.
BATCH_SLICES = [
    slice(0, 1),
    slice(2, 6),
    slice(8, 16),
    slice(0, 8),
    slice(500, 508),
    slice(100, 104),
]


@pytest.fixture(scope="module")
def dataset():
    """One dataset for the module. Parsing the 7300-word CSV is the slow part."""
    return BridgeDataset(dataset_config=DatasetConfig(dataset_filepath=DATA_CSV))


@pytest.fixture(scope="module")
def artifacts_dir(tmp_path_factory):
    return str(tmp_path_factory.mktemp("model_artifacts"))


@pytest.fixture(scope="module")
def make_pipeline(dataset, artifacts_dir):
    """Build a real pipeline for one pathway. Everything expensive is already built."""

    vocab = VocabSpec.from_tokenizer(dataset.tokenizer)

    def build(pathway: str) -> TrainingPipeline:
        return TrainingPipeline(
            model=Model(ModelConfig(vocab=vocab, d_model=32, nhead=2, seed=5)),
            dataset=dataset,
            training_config=TrainingConfig(
                num_epochs=1,
                training_pathway=pathway,
                model_artifacts_dir=artifacts_dir,
            ),
            metrics_logger=STDOutMetricsLogger(SILENT_METRICS),
        )

    return build


def record_cross_entropy_calls(monkeypatch) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Capture the (logits, target) pairs ``compute_loss`` hands ``CrossEntropyLoss``.

    Instruments the real path instead of re-deriving the slice in the test. Whatever
    ``compute_loss`` chose as the target is what gets inspected, so a test that passes
    here cannot pass by agreeing with itself.
    """
    calls: list[tuple[torch.Tensor, torch.Tensor]] = []
    real_forward = torch.nn.CrossEntropyLoss.forward

    def spy(self, input, target):
        calls.append((input, target))
        return real_forward(self, input, target)

    monkeypatch.setattr(torch.nn.CrossEntropyLoss, "forward", spy)
    return calls


def message_names_shape(message: str, shape) -> bool:
    """True when ``message`` spells out ``shape`` in any of the usual renderings.

    ``torch.Size([8, 8])``, ``[8, 8]`` and ``(8, 8)`` all reduce to the same digits once
    whitespace is dropped, so the check does not pin the wording of the error, only that
    the numbers a reader needs are in it.
    """
    compact = message.replace(" ", "")
    dims = ",".join(str(int(d)) for d in shape)
    return f"[{dims}]" in compact or f"({dims})" in compact


def step_loss(pipeline, dataset, batch_slice) -> float:
    metrics = pipeline.single_step(dataset, batch_slice, calculate_metrics=False)
    return float(metrics["loss"].detach())


@pytest.mark.parametrize("pathway", ["o2p", "p2p", "p2o", "op2op"])
def test_every_pathway_completes_a_training_step(make_pipeline, dataset, pathway):
    """A pathway that raises on every batch cannot train, whatever else is true of it.

    ``o2p`` and ``p2p`` are the control: they never touched the orthographic target and
    pass today. If they fail too, the fixtures are wrong and the other failures here say
    nothing about the alignment.
    """
    pipeline = make_pipeline(pathway)
    metrics = pipeline.single_step(dataset, slice(0, 8), calculate_metrics=True)

    loss = metrics["loss"]
    assert torch.isfinite(loss.detach()).all(), f"{pathway} produced a non-finite loss: {loss}"
    assert loss.detach().ndim == 0, f"{pathway} loss must be a scalar, got shape {loss.shape}"

    if pathway in ("p2o", "op2op"):
        # The orthographic metric slices the same positions as the loss, so it breaks in
        # the same way, and only appears once calculate_metrics is on.
        assert "orth_loss" in metrics
        assert "letter_wise_accuracy" in metrics


def test_bos_is_trained_to_predict_the_first_character(make_pipeline, dataset, monkeypatch):
    """The alignment itself, written out as characters rather than as a slice bound.

    The word is "long" and the expected pairs are hand-computed in ALIGNMENT_FOR_LONG
    above, independently of anything the code does. The load-bearing entry is
    ``('[BOS]', 'l')``: generation starts from a bare ``[BOS]``, so this is the pair that
    decides whether the model's first sampled token is the first letter of the word. The
    rejected alternative trains ``('[BOS]', 'o')`` and skips a letter.
    """
    pipeline = make_pipeline("p2o")
    # Encoded straight from the tokenizer rather than read out of the dataset, which
    # resolves a language token of "EN" for this word. The language token occupies
    # position 0 either way; which one it is has nothing to do with the alignment.
    batch = dataset.tokenizer.encode(["long"])
    orthography, phonology = batch.orthographic, batch.phonological

    logits = pipeline.forward(orthography, phonology)
    calls = record_cross_entropy_calls(monkeypatch)
    # Before the fix the widths disagree and CrossEntropyLoss raises, but the target was
    # already recorded by then, and the alignment it encodes is what this test is about.
    with contextlib.suppress(RuntimeError):
        pipeline.compute_loss(logits, orthography, phonology)

    assert len(calls) == 1, "p2o computes exactly one cross-entropy, the orthographic one"
    _, target = calls[0]

    idx_2_char = dataset.tokenizer.char_tokenizer.idx_2_char
    decoder_tokens = [idx_2_char[i] for i in orthography.dec_input_ids[0].tolist()]
    target_tokens = [idx_2_char[i] for i in target[0].tolist()]

    # strict=False on purpose: when the widths disagree the pairing must still be
    # built, so the failure shows the shifted table rather than a zip error.
    assert list(zip(decoder_tokens, target_tokens, strict=False)) == ALIGNMENT_FOR_LONG
    assert len(target_tokens) == len(decoder_tokens), (
        f"one target per decoder position: {len(target_tokens)} targets "
        f"for {len(decoder_tokens)} decoder inputs"
    )


def test_target_width_matches_logits_width_across_batches(make_pipeline, dataset, monkeypatch):
    """One target position per predicted position, for batches of six different widths.

    A single batch proves nothing here, since an off-by-one target can coincide with the
    logits only by accident of a particular batch. The first assertion is the control: if
    the chosen slices do not actually vary the decoder width, the sweep is decorative.
    """
    pipeline = make_pipeline("p2o")
    batches = [dataset[s] for s in BATCH_SLICES]
    widths = [b.orthographic.dec_input_ids.shape[1] for b in batches]
    assert len(set(widths)) >= 3, f"the sweep must vary the decoder width, got {widths}"

    calls = record_cross_entropy_calls(monkeypatch)
    for batch_slice, batch, width in zip(BATCH_SLICES, batches, widths, strict=True):
        orthography, phonology = batch.orthographic, batch.phonological
        logits = pipeline.forward(orthography, phonology)
        calls.clear()
        with contextlib.suppress(RuntimeError):
            pipeline.compute_loss(logits, orthography, phonology)

        assert len(calls) == 1
        loss_logits, target = calls[0]
        batch_size = orthography.dec_input_ids.shape[0]
        assert tuple(target.shape) == (batch_size, width), (
            f"{batch_slice}: target {tuple(target.shape)} should be "
            f"{(batch_size, width)}, one position per decoder input"
        )
        # CrossEntropyLoss reads (batch, classes, positions).
        assert loss_logits.shape[2] == width


def test_p2o_loss_descends_over_twelve_steps(make_pipeline, dataset):
    """Training moves the loss down, and it is the optimizer steps that move it.

    An aligned loss is only useful if it is learnable, so this runs the real optimizer on
    a fixed batch. The control comes first: in eval mode ``single_step`` takes no step, so
    repeating it on the same batch must return the same number bit for bit. Any descent in
    the second phase is therefore attributable to the twelve updates and not to sampling.
    """
    # No seeding here: `Model.__init__` calls `set_seed(5)`, which reseeds torch globally
    # after any seed set at this point, so a call here would be dead code that reads as if
    # it controlled something.
    pipeline = make_pipeline("p2o")
    batch_slice = slice(0, 8)

    pipeline.model.eval()
    frozen = [step_loss(pipeline, dataset, batch_slice) for _ in range(3)]
    assert frozen[0] == frozen[1] == frozen[2], (
        f"no optimizer step is taken in eval mode, so the loss must not move: {frozen}"
    )

    pipeline.model.train()
    losses = [step_loss(pipeline, dataset, batch_slice) for _ in range(12)]
    assert losses[-1] < losses[0] - 0.5, (
        f"12 AdamW steps on one batch should cut the loss well below its start: "
        f"{losses[0]:.4f} -> {losses[-1]:.4f}"
    )


def test_mismatched_shapes_raise_a_valueerror_naming_both(make_pipeline, dataset):
    """A width disagreement should be reported by name, not by CrossEntropyLoss.

    Issue #225 surfaced as ``RuntimeError: Expected target size [8, 8], got [8, 7]``,
    which says nothing about which tensor is wrong or where it came from. Handing
    ``compute_loss`` logits that are deliberately two positions short is the same class of
    error, and the pipeline should name both shapes itself rather than let the loss
    function speak for it.
    """
    pipeline = make_pipeline("p2o")
    batch = dataset[slice(0, 8)]
    orthography, phonology = batch.orthographic, batch.phonological
    logits = pipeline.forward(orthography, phonology)

    truncated = logits["orth"][:, :, :-2]
    expected_target_shape = orthography.enc_input_ids[:, 1:].shape

    with pytest.raises(ValueError) as excinfo:
        pipeline.compute_loss({"orth": truncated}, orthography, phonology)

    message = str(excinfo.value)
    assert message_names_shape(message, expected_target_shape), (
        f"the error should name the target shape {tuple(expected_target_shape)}: {message}"
    )
    assert message_names_shape(message, truncated.shape), (
        f"the error should name the logits shape {tuple(truncated.shape)}: {message}"
    )


def test_a_perfect_predictor_scores_one_on_the_orth_metrics(dataset):
    """The metric has to score the positions the loss trains, or it reports on nothing.

    Logits built as one-hot over ``enc_input_ids[:, 1:]`` are a predictor that is right at
    every position under the alignment above, so letter and word accuracy are analytically
    1.0. Corrupting the first character of one word out of four is the control: with 22
    non-padding positions in this batch, letter accuracy must fall to exactly 21/22 and
    word accuracy to exactly 3/4. A metric reading a different set of positions cannot
    produce both numbers.
    """
    tokenizer = dataset.tokenizer
    orthography = dataset[slice(0, 4)].orthographic
    vocabulary_size = tokenizer.char_tokenizer.get_vocabulary_size()
    target = orthography.enc_input_ids[:, 1:]

    perfect = torch.nn.functional.one_hot(target, num_classes=vocabulary_size)
    perfect = perfect.permute(0, 2, 1).float()

    scores = calculate_orth_metrics(
        {"orth": perfect}, orthography, orth_pad_id=tokenizer.orth_pad_id
    )
    assert scores["letter_wise_accuracy"] == pytest.approx(1.0)
    assert scores["word_wise_accuracy"] == pytest.approx(1.0)

    valid_positions = int((target != tokenizer.orth_pad_id).sum())
    assert valid_positions == 22, "the analytic control below is computed for this batch"

    corrupted = perfect.clone()
    first_character = target[0, 1].item()
    corrupted[0, :, 1] = 0.0
    corrupted[0, (first_character + 1) % vocabulary_size, 1] = 1.0

    scores = calculate_orth_metrics(
        {"orth": corrupted}, orthography, orth_pad_id=tokenizer.orth_pad_id
    )
    assert scores["letter_wise_accuracy"] == pytest.approx(21 / 22)
    assert scores["word_wise_accuracy"] == pytest.approx(0.75)
