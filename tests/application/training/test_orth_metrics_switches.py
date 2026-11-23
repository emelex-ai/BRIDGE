"""Orthographic accuracy is reported twice: every token, and letters only.

A mixed-language row carries a language token wherever the language changes, the switch,
and the decoder is trained to emit it. ``letter_wise_accuracy`` and ``word_wise_accuracy``
count the switch like any other scored token, so one missed switch costs a letter and
fails its row. The ``_without_switches`` pair leaves out exactly the positions whose target
is a language token, so it measures spelling alone.

Oracle: analytic. The batch is small enough to count by hand, the predictor is built
one-hot from the target so it is right everywhere, and each test then makes it predict a
chosen wrong token at one chosen position:

    row 0  "no me"    ES EN   targets  [BOS] n o _ EN m e [EOS]      8 scored, 1 switch
    row 1  "la casa"  ES      targets  [BOS] l a _ c a s a [EOS]     9 scored, 0 switches
    row 2  "no me"    EN ES   targets  [BOS] n o _ ES m e [EOS]      8 scored, 1 switch

25 scored positions, of which 2 are switches and 23 are not.
"""

import pytest
import torch

from bridge.application.training.ortho_metrics import calculate_orth_metrics
from bridge.application.training.training_pipeline import TrainingPipeline
from bridge.domain.datamodels import ModelConfig, TrainingConfig, VocabSpec
from bridge.domain.model import Model

TEXTS = ["no me", "la casa", "no me"]
LANGUAGES = ["ES EN", "ES", "EN ES"]
SWITCH_COLUMN = 4  # in the target: [BOS] n o _ <switch>
LETTER_COLUMN = 5  # the "m" after it, and the "a" of "casa" in row 1


@pytest.fixture(scope="module")
def orthography(shared_tokenizer):
    return shared_tokenizer.encode(TEXTS, languages=LANGUAGES).orthographic


def one_hot_logits(target: torch.Tensor, vocabulary_size: int) -> torch.Tensor:
    """Logits ``(batch, classes, positions)`` whose argmax is ``target`` everywhere."""
    return torch.nn.functional.one_hot(target, num_classes=vocabulary_size).permute(0, 2, 1).float()


def wrong_at(logits: torch.Tensor, row: int, column: int, predicted: int) -> torch.Tensor:
    """A copy of ``logits`` that predicts class ``predicted`` at one position."""
    out = logits.clone()
    assert int(out[row, :, column].argmax()) != predicted, "the prediction must change"
    out[row, :, column] = 0.0
    out[row, predicted, column] = 1.0
    return out


@pytest.fixture(scope="module")
def char_id(shared_tokenizer):
    return shared_tokenizer.char_tokenizer.char_2_idx


@pytest.fixture(scope="module")
def score(shared_tokenizer, orthography):
    def run(logits: torch.Tensor) -> dict[str, float]:
        return calculate_orth_metrics(
            {"orth": logits},
            orthography,
            orth_pad_id=shared_tokenizer.orth_pad_id,
            orth_language_ids=shared_tokenizer.orth_language_ids,
        )

    return run


@pytest.fixture(scope="module")
def perfect(shared_tokenizer, orthography):
    target = orthography.enc_input_ids[:, 1:]
    return one_hot_logits(target, shared_tokenizer.char_tokenizer.get_vocabulary_size())


def test_the_batch_is_the_one_the_arithmetic_below_assumes(shared_tokenizer, orthography):
    """The premise, checked: 25 scored targets, switches at column 4 of rows 0 and 2."""
    target = orthography.enc_input_ids[:, 1:]
    idx_2_char = shared_tokenizer.char_tokenizer.idx_2_char
    assert int((target != shared_tokenizer.orth_pad_id).sum()) == 25
    assert [idx_2_char[int(i)] for i in target[:, SWITCH_COLUMN]] == ["EN", "c", "ES"]
    assert [idx_2_char[int(i)] for i in target[:, LETTER_COLUMN]] == ["m", "a", "m"]


def test_a_perfect_predictor_scores_one_on_all_four(score, perfect):
    assert score(perfect) == {
        "letter_wise_accuracy": 1.0,
        "word_wise_accuracy": 1.0,
        "letter_wise_accuracy_without_switches": 1.0,
        "word_wise_accuracy_without_switches": 1.0,
    }


@pytest.mark.parametrize("predicted", ["ES", "m", "[EOS]"])
def test_a_missed_switch_costs_the_plain_figures_and_not_the_letters_only_ones(
    score, perfect, char_id, predicted
):
    """Only the switch of row 0 is wrong: every letter of every row is still right.

    Counting every token: 24 of 25 positions, 2 of 3 rows. Letters only: the wrong
    position is not counted, so 23 of 23 and 3 of 3. What was predicted in its place does
    not matter, the other language, the next letter or the end of the row, because the
    position is left out for what its target is, not for what the model said.
    """
    scores = score(wrong_at(perfect, 0, SWITCH_COLUMN, char_id[predicted]))
    assert scores["letter_wise_accuracy"] == pytest.approx(24 / 25)
    assert scores["word_wise_accuracy"] == pytest.approx(2 / 3)
    assert scores["letter_wise_accuracy_without_switches"] == 1.0
    assert scores["word_wise_accuracy_without_switches"] == 1.0


@pytest.mark.parametrize("predicted", ["x", "EN", "ES"])
def test_a_missed_letter_costs_both_pairs(score, perfect, char_id, predicted):
    """The control: the letter after the switch is wrong instead, and both pairs see it.

    Counting every token: 24 of 25, 2 of 3. Letters only: 22 of 23, 2 of 3. Predicting a
    language token where a letter belongs is a wrong letter like any other. Without this
    the test above would pass on a metric that ignored row 0 altogether.
    """
    scores = score(wrong_at(perfect, 0, LETTER_COLUMN, char_id[predicted]))
    assert scores["letter_wise_accuracy"] == pytest.approx(24 / 25)
    assert scores["word_wise_accuracy"] == pytest.approx(2 / 3)
    assert scores["letter_wise_accuracy_without_switches"] == pytest.approx(22 / 23)
    assert scores["word_wise_accuracy_without_switches"] == pytest.approx(2 / 3)


def test_rows_with_no_switch_score_the_same_both_ways(shared_tokenizer, char_id):
    """Oracle: invariant. With no mid-sequence language token the two pairs are equal.

    Exactly equal, not approximately: they count the same positions. Checked on
    predictions that are right in some places and wrong in others, two of the wrong ones
    being language tokens the model had no call to emit, so the equality is not the
    trivial 1.0 == 1.0. The rows open with three different language tokens.
    """
    orthography = shared_tokenizer.encode(
        ["no me", "la casa", "long", "the word"], languages=["ES", "EN", None, "EN EN"]
    ).orthographic
    target = orthography.enc_input_ids[:, 1:]
    logits = one_hot_logits(target, shared_tokenizer.char_tokenizer.get_vocabulary_size())
    for row, column, predicted in ((0, 1, "x"), (0, 3, "ES"), (2, 0, "EN"), (3, 5, "q")):
        logits = wrong_at(logits, row, column, char_id[predicted])

    scores = calculate_orth_metrics(
        {"orth": logits},
        orthography,
        orth_pad_id=shared_tokenizer.orth_pad_id,
        orth_language_ids=shared_tokenizer.orth_language_ids,
    )
    assert 0.0 < scores["letter_wise_accuracy"] < 1.0
    assert 0.0 < scores["word_wise_accuracy"] < 1.0
    assert scores["letter_wise_accuracy_without_switches"] == scores["letter_wise_accuracy"]
    assert scores["word_wise_accuracy_without_switches"] == scores["word_wise_accuracy"]


def test_the_pipeline_scores_switches_with_the_vocabularys_language_ids(
    shared_tokenizer, orthography, perfect, char_id, tmp_path
):
    """The language ids reach the metric from ``VocabSpec``, not from a default.

    Oracle: the same analytic figures as above, through ``TrainingPipeline.compute_metrics``
    on a model built from ``VocabSpec.from_tokenizer``. If the ids did not arrive, the
    letters-only pair would equal the plain pair, 24/25 and 2/3.
    """
    vocab = VocabSpec.from_tokenizer(shared_tokenizer)
    idx_2_char = shared_tokenizer.char_tokenizer.idx_2_char
    assert [idx_2_char[i] for i in vocab.orth_language_ids] == ["--", "EN", "ES"]

    pipeline = TrainingPipeline(
        model=Model(ModelConfig(vocab=vocab, d_model=32, nhead=2, seed=5)),
        training_config=TrainingConfig(training_pathway="p2o", model_artifacts_dir=str(tmp_path)),
    )
    logits = {"orth": wrong_at(perfect, 0, SWITCH_COLUMN, char_id["m"])}
    scores = pipeline.compute_metrics(logits, orthography, None)

    assert scores["letter_wise_accuracy"] == pytest.approx(24 / 25)
    assert scores["word_wise_accuracy"] == pytest.approx(2 / 3)
    assert scores["letter_wise_accuracy_without_switches"] == 1.0
    assert scores["word_wise_accuracy_without_switches"] == 1.0


def test_a_vocabulary_with_no_language_tokens_scores_both_pairs_alike(
    shared_tokenizer, orthography, perfect, char_id
):
    """Oracle: analytic. With no language ids nothing is a switch, so both pairs count all
    25 positions: 24 of 25 and 2 of 3, twice."""
    scores = calculate_orth_metrics(
        {"orth": wrong_at(perfect, 0, SWITCH_COLUMN, char_id["m"])},
        orthography,
        orth_pad_id=shared_tokenizer.orth_pad_id,
        orth_language_ids=(),
    )
    assert scores["letter_wise_accuracy"] == pytest.approx(24 / 25)
    assert scores["letter_wise_accuracy_without_switches"] == scores["letter_wise_accuracy"]
    assert scores["word_wise_accuracy"] == pytest.approx(2 / 3)
    assert scores["word_wise_accuracy_without_switches"] == scores["word_wise_accuracy"]


def test_a_spec_that_does_not_name_its_language_tokens_reports_the_plain_pair_only(
    shared_tokenizer, orthography, perfect, char_id, tmp_path
):
    """A hand-written ``VocabSpec`` may leave ``orth_language_ids`` out, and then it is unknown.

    Oracle: analytic, 24 of 25 and 2 of 3 for the plain pair. Unknown is not the same as
    empty: an empty default would report a letters-only pair equal to the plain one on
    these rows, which hold two switches.
    """
    fields = VocabSpec.from_tokenizer(shared_tokenizer).model_dump()
    del fields["orth_language_ids"]
    pipeline = TrainingPipeline(
        model=Model(ModelConfig(vocab=VocabSpec(**fields), d_model=32, nhead=2, seed=5)),
        training_config=TrainingConfig(training_pathway="p2o", model_artifacts_dir=str(tmp_path)),
    )
    logits = {"orth": wrong_at(perfect, 0, SWITCH_COLUMN, char_id["m"])}
    assert pipeline.compute_metrics(logits, orthography, None) == {
        "letter_wise_accuracy": pytest.approx(24 / 25),
        "word_wise_accuracy": pytest.approx(2 / 3),
    }


def test_a_pickled_spec_without_the_attribute_still_scores(
    shared_tokenizer, orthography, perfect, char_id, tmp_path
):
    """A model built from a checkpoint's own pickled config computes metrics.

    Pickle restores a ``VocabSpec``'s ``__dict__`` verbatim and validates nothing, so a
    spec whose pickle does not hold ``orth_language_ids`` has no such attribute, default
    or not. That state is reproduced here by deleting the entry, and the first assertion
    checks the premise: reading the field raises. The pipeline must then report the plain
    pair, 24 of 25 and 2 of 3, and leave the letters-only pair out.
    """
    pipeline = TrainingPipeline(
        model=Model(
            ModelConfig(
                vocab=VocabSpec.from_tokenizer(shared_tokenizer), d_model=32, nhead=2, seed=5
            )
        ),
        training_config=TrainingConfig(training_pathway="p2o", model_artifacts_dir=str(tmp_path)),
    )
    vocab = pipeline.model.model_config.vocab
    del vocab.__dict__["orth_language_ids"]
    with pytest.raises(AttributeError):
        _ = vocab.orth_language_ids

    logits = {"orth": wrong_at(perfect, 0, SWITCH_COLUMN, char_id["m"])}
    assert pipeline.compute_metrics(logits, orthography, None) == {
        "letter_wise_accuracy": pytest.approx(24 / 25),
        "word_wise_accuracy": pytest.approx(2 / 3),
    }
