"""The top-level ``bridge`` package must be enough to do the package's main job.

``bridge/__init__.py`` is the supported import surface. Everything under
``bridge.application`` and ``bridge.domain`` is internal layout that a reorganisation is
free to move. That promise only holds if the exported names are *sufficient*: if
assembling the primary object, a ``TrainingPipeline``, forces a consumer to reach into an
internal module, then moving that module breaks code that had no supported alternative.

The assembly test below is the real assertion. The rest guard the export list itself
against the two ways it drifts: a name listed but not imported, and a name quietly
deleted.
"""

import math

import bridge

# The names downstream code already relies on. A superset check rather than equality:
# adding an export is not a breaking change and must not fail the suite, while removing
# one is, and does.
ESTABLISHED_EXPORTS = frozenset(
    {
        "BridgeDataset",
        "BridgeEncoding",
        "BridgeTokenizer",
        "DatasetConfig",
        "EncodingComponent",
        "GenerationOutput",
        "Model",
        "ModelConfig",
        "TrainingConfig",
        "TrainingEvent",
        "TrainingPhase",
        "TrainingPipeline",
        "VocabSpec",
    }
)

DATA = "tests/domain/model/data/data.csv"


def loss_of(metrics):
    """The step's loss as a plain float."""
    value = metrics["loss"]
    return float(value.detach()) if hasattr(value, "detach") else float(value)


def test_a_working_pipeline_assembles_from_the_top_level_package_alone(tmp_path):
    """Assemble and run a real training step using only ``from bridge import ...``.

    This is the test that states the promise. Every import here is from the top-level
    package, so the test fails at exactly the point where the public API runs out, and
    passes only when a consumer could genuinely write this code. It runs one
    ``single_step`` rather than only constructing the pipeline, because a public API that
    builds an object which cannot compute anything is not sufficient either.

    Oracle: an invariant with an analytic anchor. The ``o2p`` loss is cross entropy over
    two classes per phonetic feature column, so an untrained model scores near chance,
    ``log 2`` or about 0.693 nats. The assertion stays at finiteness and sign rather than
    pinning a number, because the claim under test is that the pipeline computes at all,
    and a broken forward pass gives ``nan``, ``inf`` or exactly zero.
    """
    from bridge import (
        BridgeDataset,
        DatasetConfig,
        Model,
        ModelConfig,
        TrainingConfig,
        TrainingPipeline,
        VocabSpec,
    )

    dataset = BridgeDataset(dataset_config=DatasetConfig(dataset_filepath=DATA))
    model = Model(
        ModelConfig(vocab=VocabSpec.from_tokenizer(dataset.tokenizer), d_model=32, nhead=2, seed=5)
    )
    pipeline = TrainingPipeline(
        model=model,
        training_config=TrainingConfig(
            num_epochs=1, training_pathway="o2p", model_artifacts_dir=str(tmp_path)
        ),
        dataset=dataset,
    )

    metrics = pipeline.single_step(dataset, slice(0, 8), calculate_metrics=False)

    loss = loss_of(metrics)
    assert math.isfinite(loss), f"loss is {loss}, so the step did not really run"
    assert loss > 0.0


def test_the_two_phoneme_id_spaces_are_convertible_through_the_public_api():
    """``docs/architecture.md`` names confusing row space for feature space as the defect
    that yields silently wrong output rather than an error, and names
    ``PhonemeTable.features_of`` / ``row_of`` as the remedy. A remedy a consumer cannot
    import is not one.

    Oracle: a round trip. ``row_of`` maps a phoneme to its table row; ``features_of``
    gives that row's active feature columns, which must be non-empty and inside feature
    space for a phoneme that carries features.
    """
    from bridge import load_phoneme_table

    table = load_phoneme_table()
    row = table.row_of("AE")
    assert 0 <= row < table.num_rows

    features = table.features_of(row)
    assert features.numel() > 0, "AE carries phonetic features, so its row cannot be empty"
    assert int(features.max()) < table.vocab_size


def test_the_pathways_are_enumerable_through_the_public_api():
    """``Model.generate`` takes a pathway; a consumer needs a supported way to list them.

    Oracle: a differential against the model module's own tuple, which is what
    ``generate`` validates against.
    """
    from bridge.domain.model.model import PATHWAYS as internal

    assert set(bridge.PATHWAYS) == set(internal)
    assert len(bridge.PATHWAYS) == 5


def test_every_listed_export_resolves():
    """A name in ``__all__`` that was never imported is invisible until a user hits it.

    ``from bridge import X`` and ``from bridge import *`` both fail on such a name, with a
    message that points at the user's code rather than at the package. The listing is the
    claim; attribute access is the check.

    Oracle: the invariant that ``__all__`` names attributes of the module it belongs to,
    which is what the language itself assumes when expanding a star import.
    """
    missing = [name for name in bridge.__all__ if not hasattr(bridge, name)]
    assert missing == [], f"listed in bridge.__all__ but not defined on the package: {missing}"


def test_no_established_export_has_been_dropped():
    """The reverse drift: a name removed from ``__all__`` breaks importers silently here
    and loudly for them.

    Oracle: a pinned baseline of the names already published. Superset rather than
    equality, so growing the API is free and shrinking it is not.
    """
    exported = set(bridge.__all__)
    dropped = sorted(ESTABLISHED_EXPORTS - exported)
    assert dropped == [], f"no longer exported from bridge: {dropped}"
