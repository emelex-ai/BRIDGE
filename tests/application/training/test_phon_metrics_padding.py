"""Phonological metrics must mask padding with the phonological pad id, not with 2.

Every one of the eight phonological metrics builds its valid mask as ``phon_true != 2``.
Two is the *orthographic* pad id. Phonological targets are multi-hot rows whose only
values are 0, 1 and 35, so the comparison is a tautology: it keeps every position, and
padded rows are scored as though they were real phonemes.

The two id spaces are the trap. Row space (0 to 90) says which phoneme; feature space
(0 to 35) says which phonetic feature, and ``phon_targets`` lives in feature space where
[PAD] is 35. A literal that belongs to neither space silently disables the mask instead
of raising, which is why nothing has caught it.

These tests assert the fixed contract: ``calculate_phon_metrics`` takes a required
``phon_pad_id``, mirroring ``calculate_orth_metrics``'s ``orth_pad_id``, and the pipeline
hands it the vocabulary's own phonological pad id. Passing 2 reproduces today's
behaviour, so each comparison below runs the same code twice and only the pad id changes.

The oracle is analytic throughout. The predictions are constructed to match the targets
exactly at every real position, so a mask that keeps only real positions must report a
perfect score, and every departure from perfect is padding leaking in. The control is a
batch of equal-length words: with no padding to find, the two pad ids must agree bitwise.
"""

import inspect
import math
from types import SimpleNamespace

import pytest
import torch

from bridge.application.training import training_pipeline as pipeline_module
from bridge.application.training.phon_metrics import calculate_phon_metrics
from bridge.application.training.training_pipeline import TrainingPipeline
from bridge.domain.tokenizer import BridgeTokenizer
from tests.vocab import PHONEME_TABLE, TEST_VOCAB

ORTH_PAD_ID = 2
PHON_PAD_ID = 35

# "a" is one phoneme, "cat" three, "elephant" eight, so the batch pads to 8 positions
# and 10 of its 24 rows are pure padding.
PADDED_WORDS = ["a", "cat", "elephant"]
REAL_ROWS = 14
TOTAL_ROWS = 24

# Width of a target row in feature columns. Asserted against the real tensor below.
FEATURE_WIDTH = 35

# Three three-phoneme words. Nothing is padded, so the pad id cannot matter.
EVEN_WORDS = ["cat", "dog", "pig"]

METRIC_NAMES = (
    "phon_cosine_similarity",
    "phon_euclidean_distance",
    "phon_feature_accuracy",
    "phon_phoneme_wise_accuracy",
    "phon_word_accuracy",
    "closest_phoneme_l1_accuracy",
    "closest_phoneme_l2_accuracy",
    "closest_phoneme_cosine_accuracy",
)


@pytest.fixture(scope="module")
def tokenizer():
    return BridgeTokenizer()


@pytest.fixture(scope="module")
def phon_reps():
    """What ``TrainingPipeline`` passes: the (86, 31) base feature matrix."""
    return PHONEME_TABLE.phonetic_features


def perfect_batch(tokenizer, words, corrupt=None):
    """Encode ``words`` and fabricate logits that are exactly right at every real row.

    ``phon_pred`` is ``argmax(logits["phon"], dim=1)``, so stacking ``1 - p`` under ``p``
    makes the prediction whatever ``p`` is. Setting ``p = (targets == 1)`` reproduces
    every real row bit for bit and puts zeros on the padded rows, which hold 35. A
    prediction can therefore never match a padded position, which is what makes the
    expected numbers below closed form rather than measured.

    ``corrupt`` is an index whose predicted row is inverted, to make a batch that is
    imperfect on real positions too.
    """
    phonology = tokenizer.encode(words).phonological
    targets = phonology.phon_targets
    prediction = (targets == 1).long()
    if corrupt is not None:
        prediction[corrupt] = 1 - prediction[corrupt]
    logits = {"phon": torch.stack([1 - prediction, prediction], dim=1).float()}
    assert torch.equal(torch.argmax(logits["phon"], dim=1), prediction)
    return logits, phonology


def test_padding_exists_and_the_orthographic_pad_id_cannot_see_it(tokenizer):
    """The premise the whole defect rests on, checked against the real tokenizer.

    Both halves matter. The mask built from the real pad id keeps strictly less than
    every position, so padding is genuinely there to be masked. The value 2 never occurs
    anywhere in the targets, so the mask in use could not have found that padding under
    any batch, and its passing today says nothing.
    """
    targets = tokenizer.encode(PADDED_WORDS).phonological.phon_targets

    assert TEST_VOCAB.phon_pad_id == PHON_PAD_ID
    assert TEST_VOCAB.orth_pad_id == ORTH_PAD_ID
    assert sorted(torch.unique(targets).tolist()) == [0, 1, PHON_PAD_ID]

    # A padded position is a whole row of the pad feature id.
    pad_rows = (targets == PHON_PAD_ID).all(dim=-1)
    assert pad_rows.sum(dim=-1).tolist() == [6, 4, 0]
    assert int((~pad_rows).sum()) == REAL_ROWS
    assert targets.shape == (len(PADDED_WORDS), TOTAL_ROWS // len(PADDED_WORDS), FEATURE_WIDTH)

    # The mask actually in use keeps everything.
    assert bool((targets != ORTH_PAD_ID).all())
    # The correct mask keeps only the real rows.
    kept = (targets != PHON_PAD_ID).float().mean().item()
    assert kept == pytest.approx(REAL_ROWS / TOTAL_ROWS)
    assert kept < 1.0


def test_calculate_phon_metrics_requires_a_phon_pad_id(tokenizer, phon_reps):
    """The pad id is not an optional courtesy, it is part of the contract.

    ``calculate_orth_metrics`` already takes ``orth_pad_id`` as a required argument, so
    an orthographic caller cannot forget it. The phonological side must be symmetric:
    a call that omits the pad id has not said which id space its targets are in, and
    guessing is exactly how the literal 2 got there.
    """
    logits, phonology = perfect_batch(tokenizer, PADDED_WORDS)

    parameters = inspect.signature(calculate_phon_metrics).parameters
    assert "phon_pad_id" in parameters
    assert parameters["phon_pad_id"].default is inspect.Parameter.empty

    with pytest.raises(TypeError):
        calculate_phon_metrics(logits, phonology, phon_reps)


def test_every_phon_metric_moves_when_the_mask_uses_the_real_pad_id(tokenizer, phon_reps):
    """All eight metrics are wrong under padding, not just one or two.

    The prediction is exact on every real row, so a mask that keeps only real rows must
    score perfectly: accuracy 1.0 and euclidean distance 0. The numbers under pad id 2
    are equally closed form. Ten of the 24 rows are padding, and a padded target row is
    35 in all 35 features while a prediction can only be 0 or 1, so every padded row is
    counted wrong: the feature, phoneme-wise, cosine and closest-phoneme accuracies all
    collapse to 14/24, word accuracy to 1/3 because only "elephant" escapes padding, and
    the euclidean distance becomes ten rows of ||35 - 0|| over 35 columns, averaged over
    all 24 rows.
    """
    logits, phonology = perfect_batch(tokenizer, PADDED_WORDS)

    fixed = calculate_phon_metrics(logits, phonology, phon_reps, phon_pad_id=PHON_PAD_ID)
    current = calculate_phon_metrics(logits, phonology, phon_reps, phon_pad_id=ORTH_PAD_ID)

    assert set(fixed) == set(METRIC_NAMES)
    for name in METRIC_NAMES:
        assert float(fixed[name]) != float(current[name]), f"{name} did not move"

    kept = REAL_ROWS / TOTAL_ROWS
    # A padded row is 35 in all 35 columns and its prediction is 0 in all of them.
    pad_row_distance = PHON_PAD_ID * math.sqrt(FEATURE_WIDTH)

    assert float(fixed["phon_cosine_similarity"]) == pytest.approx(1.0)
    assert float(current["phon_cosine_similarity"]) == pytest.approx(kept)

    assert float(fixed["phon_euclidean_distance"]) == pytest.approx(0.0, abs=1e-4)
    assert float(current["phon_euclidean_distance"]) == pytest.approx(
        (TOTAL_ROWS - REAL_ROWS) * pad_row_distance / TOTAL_ROWS, rel=1e-5
    )

    assert float(fixed["phon_feature_accuracy"]) == pytest.approx(1.0)
    assert float(current["phon_feature_accuracy"]) == pytest.approx(kept)

    assert float(fixed["phon_phoneme_wise_accuracy"]) == pytest.approx(1.0)
    assert float(current["phon_phoneme_wise_accuracy"]) == pytest.approx(kept)

    # Only "elephant" fills its row, so it is the one word scored correctly under the
    # broken mask.
    assert float(fixed["phon_word_accuracy"]) == pytest.approx(1.0)
    assert float(current["phon_word_accuracy"]) == pytest.approx(1 / 3)

    # The closest-phoneme metrics compare argmins against the phoneme table. On real rows
    # prediction and target are identical, so they must agree. On a padded row the two
    # argmins are free to coincide by chance, so the claim there is a bound, not a value.
    for name in (
        "closest_phoneme_l1_accuracy",
        "closest_phoneme_l2_accuracy",
        "closest_phoneme_cosine_accuracy",
    ):
        assert float(fixed[name]) == pytest.approx(1.0), name
        # float32 accumulation, hence the epsilon on the lower bound.
        assert kept - 1e-6 <= float(current[name]) < 1.0, name


def test_control_a_batch_with_no_padding_is_blind_to_the_pad_id(tokenizer, phon_reps):
    """The control that proves the test above measures padding and nothing else.

    Three words of three phonemes each pad to nothing, so both masks keep every position
    and the two metric dicts must be identical. If these ever diverged, the differences
    in the padded case would be evidence of some other disagreement rather than of
    padding being scored.
    """
    logits, phonology = perfect_batch(tokenizer, EVEN_WORDS)
    targets = phonology.phon_targets
    assert not bool((targets == PHON_PAD_ID).any()), "control batch must carry no padding"

    fixed = calculate_phon_metrics(logits, phonology, phon_reps, phon_pad_id=PHON_PAD_ID)
    current = calculate_phon_metrics(logits, phonology, phon_reps, phon_pad_id=ORTH_PAD_ID)

    for name in METRIC_NAMES:
        assert float(fixed[name]) == float(current[name]), f"{name} moved without padding"


def test_scoring_padded_positions_biases_feature_accuracy_downward(tokenizer, phon_reps):
    """Which way the bias runs, measured rather than assumed.

    A padded target row is 35 everywhere and a prediction is 0 or 1, so no padded feature
    can ever be counted correct. Both figures share a numerator, the count of correct
    real features, and differ only in the denominator: real positions versus all
    positions. The corrected accuracy is therefore the larger by exactly the ratio of the
    two counts, and today's reported number is an understatement.

    The batch here is deliberately imperfect on real positions too, one whole row of the
    prediction inverted, so the corrected figure is 13/14 rather than a trivial 1.0.
    """
    logits, phonology = perfect_batch(tokenizer, PADDED_WORDS, corrupt=(1, 0))

    over_real = float(
        calculate_phon_metrics(logits, phonology, phon_reps, phon_pad_id=PHON_PAD_ID)[
            "phon_feature_accuracy"
        ]
    )
    over_all = float(
        calculate_phon_metrics(logits, phonology, phon_reps, phon_pad_id=ORTH_PAD_ID)[
            "phon_feature_accuracy"
        ]
    )

    correct_features = (REAL_ROWS - 1) * FEATURE_WIDTH
    assert over_real == pytest.approx(correct_features / (REAL_ROWS * FEATURE_WIDTH))
    assert over_all == pytest.approx(correct_features / (TOTAL_ROWS * FEATURE_WIDTH))

    assert over_real > over_all
    assert over_all == pytest.approx(over_real * REAL_ROWS / TOTAL_ROWS, rel=1e-5)


def test_compute_metrics_passes_the_vocabularys_phon_pad_id(phon_reps, monkeypatch):
    """The wiring, not just the helper.

    A correct helper called with the orthographic pad id is still broken, so the pad id
    ``TrainingPipeline.compute_metrics`` chooses is itself the thing under test. It must
    be the vocabulary's phonological pad id, which the phoneme table fixes at 35, and it
    must not be the orthographic one. The method reads no other state on an "o2p"
    pathway, so a bare stub carries everything it touches.
    """
    seen = {}

    def recorder(logits, phonology, phon_reps, phon_pad_id):
        seen["phon_pad_id"] = phon_pad_id
        return {}

    monkeypatch.setattr(pipeline_module, "calculate_phon_metrics", recorder)

    stub = SimpleNamespace(
        model=SimpleNamespace(model_config=SimpleNamespace(vocab=TEST_VOCAB)),
        training_config=SimpleNamespace(training_pathway="o2p"),
        phon_reps=phon_reps,
    )
    stub.compute_metrics = TrainingPipeline.compute_metrics.__get__(stub)
    stub.compute_metrics({}, None, None)

    assert seen["phon_pad_id"] == TEST_VOCAB.phon_pad_id == PHON_PAD_ID
    assert seen["phon_pad_id"] != TEST_VOCAB.orth_pad_id
