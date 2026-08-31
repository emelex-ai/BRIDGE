"""Every epoch must train on a freshly shuffled order (issue #224, defect 3).

``BridgeDataset.shuffle(cutoff)`` exists and is tested, but nothing ever calls it:
``TrainingPipeline`` walks ``self.train_slices`` in file order, every epoch, for every
run. A lexicon sorted alphabetically therefore trains on all the "a" words first and
all the "z" words last, in the same order, forever, which correlates the gradient with
the alphabet and makes the last batch of an epoch the one the weights remember.

The fix shuffles the **train** partition only, at the top of each epoch, seeded from
``TrainingConfig.seed`` and the epoch number. Four properties make that safe, and each
gets a test here:

* the validation tail is never touched, or validation scores stop being comparable
  between epochs and the whole reason to hold data out is gone;
* the train partition really is permuted, and permuted differently each epoch;
* nothing is dropped or duplicated, and each word keeps its language tag, since
  ``words`` and ``languages`` are parallel lists and the tokenizer reads both;
* ``shuffle_each_epoch=False`` restores the old behaviour exactly, which is the
  control proving these tests watch shuffling and not some other perturbation.

The order is read from the batches the training loop actually pulls, by wrapping
``single_step``, rather than inferred from ``dataset.words`` afterwards. A shuffle
applied after training instead of before would leave the same final state and is
invisible to the weaker observation.

The dataset here is 48 hand-listed words, 30 English and 18 Spanish, written to a csv
under ``tmp_path``: the repo's own ``data.csv`` is 7300 words and single-language, so it
is both too slow to run epochs over and unable to catch a words/languages misalignment.
"""

import pytest

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
from bridge.domain.tokenizer import BridgeTokenizer
from bridge.infra.metrics.metrics_logger import STDOutMetricsLogger

# All 30 are keys of bridge/core/pronunciation_lexicons/en.json (uppercased there).
ENGLISH_WORDS = [
    "able",
    "about",
    "above",
    "after",
    "again",
    "along",
    "apart",
    "baker",
    "candle",
    "dinner",
    "easy",
    "fable",
    "garden",
    "harbor",
    "island",
    "jacket",
    "kettle",
    "ladder",
    "magnet",
    "napkin",
    "orange",
    "pencil",
    "quiet",
    "rabbit",
    "saddle",
    "table",
    "under",
    "valley",
    "window",
    "yellow",
]
# All 18 are keys of es.json and none of them is a key of en.json, so no word in this
# dataset is a cross-language homograph, which ``__getitem__`` would reject.
SPANISH_WORDS = [
    "que",
    "por",
    "una",
    "del",
    "los",
    "con",
    "para",
    "pero",
    "bien",
    "las",
    "eso",
    "como",
    "esto",
    "todo",
    "muy",
    "esta",
    "vamos",
    "ahora",
]

FILE_ORDER = [(w, "EN") for w in ENGLISH_WORDS] + [(w, "ES") for w in SPANISH_WORDS]
TRAIN_TEST_SPLIT = 0.8
# The same cut ``create_data_slices`` computes: int(48 * 0.8) == 38.
CUTPOINT = int(len(FILE_ORDER) * TRAIN_TEST_SPLIT)
TRAIN_HEAD = FILE_ORDER[:CUTPOINT]
VAL_TAIL = FILE_ORDER[CUTPOINT:]

BATCH_SIZE_TRAIN = 32
SEED = 1234
EPOCHS = 3


@pytest.fixture(scope="module")
def tokenizer():
    """One tokenizer for every dataset built here: constructing one parses the
    pronunciation lexicons, which is the expensive part of building a dataset."""
    return BridgeTokenizer()


@pytest.fixture(scope="module")
def words_csv(tmp_path_factory):
    path = tmp_path_factory.mktemp("shuffling") / "words.csv"
    rows = ["word_raw,language"] + [f"{word},{lang}" for word, lang in FILE_ORDER]
    path.write_text("\n".join(rows) + "\n")
    return str(path)


@pytest.fixture(scope="module")
def artifacts_dir(tmp_path_factory):
    return str(tmp_path_factory.mktemp("artifacts"))


def build_pipeline(words_csv, tokenizer, artifacts_dir, num_epochs=EPOCHS, **overrides):
    """A real pipeline over the 48-word csv. Batches are large enough that an epoch is
    two training steps and one validation step, and ``save_every`` is out of reach so
    no checkpoint is written."""
    dataset = BridgeDataset(DatasetConfig(dataset_filepath=words_csv), tokenizer=tokenizer)
    assert [(w, lang) for w, lang in zip(dataset.words, dataset.languages, strict=True)] == (
        FILE_ORDER
    ), "the csv did not load in the order it was written"

    model = Model(
        ModelConfig(vocab=VocabSpec.from_tokenizer(dataset.tokenizer), d_model=16, nhead=2, seed=5)
    )
    training_config = TrainingConfig(
        num_epochs=num_epochs,
        training_pathway="o2p",
        train_test_split=TRAIN_TEST_SPLIT,
        batch_size_train=BATCH_SIZE_TRAIN,
        batch_size_val=32,
        save_every=10_000,
        model_artifacts_dir=artifacts_dir,
        **overrides,
    )
    metrics_config = MetricsConfig(
        batch_metrics=False,
        training_metrics=False,
        validation_metrics=False,
        modes=[],
        filename=None,
    )
    return TrainingPipeline(
        model=model,
        dataset=dataset,
        training_config=training_config,
        metrics_logger=STDOutMetricsLogger(metrics_config),
    )


def run_and_record(pipeline):
    """Run the whole train/val loop, returning the (word, language) pairs each epoch
    actually trained and validated on, in the order the loop pulled them.

    ``single_step`` is wrapped rather than reading ``dataset.words`` at the end of an
    epoch, so a shuffle applied after training rather than before it would be caught.
    Training and validation steps are told apart by ``model.training``, which
    ``validate_single_epoch`` clears.
    """
    original = pipeline.single_step
    batch_train: list[tuple[str, str]] = []
    batch_val: list[tuple[str, str]] = []

    def spy(dataset, batch_slice, calculate_metrics=False):
        seen = batch_train if pipeline.model.training else batch_val
        seen.extend(zip(dataset.words[batch_slice], dataset.languages[batch_slice], strict=True))
        return original(dataset, batch_slice, calculate_metrics)

    pipeline.single_step = spy

    train_orders, val_orders = [], []
    for _ in pipeline.run_train_val_loop("shuffling-test"):
        train_orders.append(list(batch_train))
        val_orders.append(list(batch_val))
        batch_train.clear()
        batch_val.clear()
    return train_orders, val_orders


@pytest.fixture(scope="module")
def shuffled_run(words_csv, tokenizer, artifacts_dir):
    """One seeded three-epoch run, shared by the tests that only read its order log.
    Driving epochs is the slow part, so it is done once."""
    pipeline = build_pipeline(words_csv, tokenizer, artifacts_dir, seed=SEED)
    return run_and_record(pipeline)


def fixed_points(order):
    """Positions where a word did not move relative to the file order."""
    return sum(1 for got, want in zip(order, TRAIN_HEAD, strict=True) if got == want)


def batches_of(order):
    """Split a training order into the batches the loop will actually pull from it."""
    return [
        order[start : start + BATCH_SIZE_TRAIN] for start in range(0, len(order), BATCH_SIZE_TRAIN)
    ]


# --- the flags the fix introduces ------------------------------------------


def test_shuffling_is_configurable_and_on_by_default():
    """Defaulting to False would leave the defect in place for every existing config,
    which is the whole point of the default being True. ``seed`` defaults to None so a
    run that does not ask for reproducibility still gets a different order each time.

    This is a drift guard over the schema, not a behavioural test: a fix that declared both
    fields and read neither would pass it. What the flags actually do is covered by
    ``test_shuffling_can_be_turned_off`` and ``test_the_seed_reproduces_the_order``.
    """
    fields = TrainingConfig.model_fields
    assert "shuffle_each_epoch" in fields, "no flag, so nothing can turn shuffling off"
    assert "seed" in fields, "no seed, so a shuffled run cannot be reproduced"
    assert fields["shuffle_each_epoch"].get_default() is True
    assert fields["seed"].get_default() is None


# --- what must not move ----------------------------------------------------


def test_validation_words_are_never_reordered(shuffled_run):
    """Validation exists to be compared across epochs, and that comparison is only
    meaningful if the held-out words arrive in the same order every time. The control
    is in the same test: the train partition must have moved, or this assertion passes
    for the uninteresting reason that nothing was shuffled at all.
    """
    train_orders, val_orders = shuffled_run

    assert any(order != TRAIN_HEAD for order in train_orders), (
        "no epoch reordered the training data, so the validation tail being unchanged says nothing"
    )
    for epoch, order in enumerate(val_orders):
        assert order == VAL_TAIL, f"epoch {epoch} reordered the validation partition"


# --- what must move --------------------------------------------------------


def test_every_epoch_trains_on_the_whole_partition_in_a_new_order(shuffled_run):
    """The partition arrives complete, and not in file order.

    A uniform permutation of 38 distinct words leaves the order unchanged with probability
    1/38!, about 2e-45, so an unchanged order means no shuffle happened.

    Counting words that stayed put is deliberately NOT asserted here. It cannot see either
    of the two wrong fixes that actually occur, which is what
    ``test_words_change_which_batch_they_are_in`` is for.
    """
    train_orders, _ = shuffled_run
    for epoch, order in enumerate(train_orders):
        assert len(order) == CUTPOINT, f"epoch {epoch} trained on {len(order)} words"
        assert order != TRAIN_HEAD, f"epoch {epoch} trained in file order"


def test_words_change_which_batch_they_are_in(shuffled_run):
    """Words cross batch boundaries, which is what "shuffle the partition" actually means.

    This is the discriminating test in the file, and it exists because a fixed-point
    threshold is not. Two wrong fixes were planted and measured against the rest of this
    file, and both passed all eight tests:

      * ``dataset.shuffle(batch_size_train)`` instead of ``dataset.shuffle(cutpoint)``,
        which pins the last 6 training words to the end of every epoch;
      * shuffling within each batch, which reorders everything while leaving every batch
        holding exactly the words it started with.

    Both leave few words in their original position, so both pass a fixed-point bound. What
    separates them from a real permutation is membership: which words share a gradient step.
    The trailing batch is 6 of 38 words, so under a real permutation its membership repeats
    the file order's with probability 1/C(38, 6) = 1/2760681. The leading batch is the
    complement of it, so the same bound applies.
    """
    train_orders, _ = shuffled_run
    from_file = batches_of(TRAIN_HEAD)
    assert len(from_file) >= 2, "one batch cannot show a word crossing a boundary"

    for epoch, order in enumerate(train_orders):
        seen = batches_of(order)
        for index, (got, want) in enumerate(zip(seen, from_file, strict=True)):
            assert set(got) != set(want), (
                f"epoch {epoch}: batch {index} holds exactly the {len(want)} words it holds "
                "in file order, so nothing crossed a batch boundary. A shuffle confined to "
                "one batch, or applied within each batch, looks like this."
            )
        moved = sum(
            1
            for position, word in enumerate(order)
            if TRAIN_HEAD.index(word) // BATCH_SIZE_TRAIN != position // BATCH_SIZE_TRAIN
        )
        assert moved >= 5, (
            f"epoch {epoch}: only {moved} of {CUTPOINT} words changed batch, which is too "
            "few for a permutation of the whole partition"
        )


def test_successive_epochs_differ_from_each_other(shuffled_run):
    """Reshuffling to the same order every epoch is the defect wearing a disguise: the
    order would differ from the file but not between epochs. Two independent
    permutations of 38 words coincide with probability 1/38!, so any repeat here is the
    seed failing to advance with the epoch.
    """
    train_orders, _ = shuffled_run
    for earlier in range(len(train_orders)):
        for later in range(earlier + 1, len(train_orders)):
            assert train_orders[earlier] != train_orders[later], (
                f"epochs {earlier} and {later} trained in the same order"
            )


# --- what must be conserved ------------------------------------------------


def test_shuffling_drops_and_duplicates_nothing(shuffled_run):
    """A permutation conserves the multiset. Sorting both sides compares content while
    ignoring the order, which is the part that is allowed to change.
    """
    train_orders, val_orders = shuffled_run
    for epoch, (train, val) in enumerate(zip(train_orders, val_orders, strict=True)):
        assert sorted(train) == sorted(TRAIN_HEAD), f"epoch {epoch} changed the train words"
        assert sorted(train + val) == sorted(FILE_ORDER), f"epoch {epoch} changed the dataset"


def test_words_keep_their_language_tags(shuffled_run):
    """``words`` and ``languages`` are parallel lists, and the language decides which
    pronunciation lexicon a word is looked up in. Permuting one and not the other would
    silently encode "ahora" as English. The oracle is the csv this module wrote: every
    English word is in ENGLISH_WORDS and every Spanish word is in SPANISH_WORDS.
    """
    train_orders, _ = shuffled_run
    expected = dict(FILE_ORDER)
    for epoch, order in enumerate(train_orders):
        for word, language in order:
            assert language == expected[word], (
                f"epoch {epoch} trained on '{word}' tagged {language}, not {expected[word]}"
            )
        assert {lang for _, lang in order} == {"EN", "ES"}, (
            f"epoch {epoch} lost a language entirely"
        )


# --- reproducibility and the control ---------------------------------------


def test_the_seed_reproduces_the_order(words_csv, tokenizer, artifacts_dir):
    """Paired comparison over the same csv: identical seeds must give an identical
    first-epoch order, and a different seed must give a different one. Without the
    first half a run cannot be reproduced from its config; without the second half
    ``seed`` could be ignored and the test would still pass.
    """

    def first_epoch(seed):
        pipeline = build_pipeline(words_csv, tokenizer, artifacts_dir, num_epochs=1, seed=seed)
        train_orders, _ = run_and_record(pipeline)
        return train_orders[0]

    same_a, same_b, different = first_epoch(SEED), first_epoch(SEED), first_epoch(SEED + 77)

    assert same_a == same_b, "the same seed gave two different orders"
    assert same_a != different, "a different seed gave the same order, so seed is ignored"


def test_shuffling_can_be_turned_off(words_csv, tokenizer, artifacts_dir):
    """The control. Same harness, same dataset, one flag flipped: the loop must walk
    the file order in every epoch. If this fails while the tests above pass, the order
    log is measuring something other than shuffling.
    """
    pipeline = build_pipeline(
        words_csv, tokenizer, artifacts_dir, num_epochs=2, seed=SEED, shuffle_each_epoch=False
    )
    train_orders, val_orders = run_and_record(pipeline)

    for epoch, (train, val) in enumerate(zip(train_orders, val_orders, strict=True)):
        assert train == TRAIN_HEAD, f"epoch {epoch} reordered the data with shuffling off"
        assert val == VAL_TAIL, f"epoch {epoch} reordered the validation data"
