"""The top-level ``bridge`` package must be enough to do the package's main job.

``bridge/__init__.py`` is the supported import surface. Everything under
``bridge.application``, ``bridge.domain`` and ``bridge.infra`` is internal layout that a
reorganisation is free to move. That promise only holds if the exported names are
*sufficient*: if assembling the primary object, a ``TrainingPipeline``, forces a consumer
to reach into ``bridge.infra.metrics.metrics_logger``, then moving that module breaks code
that had no supported alternative.

``TrainingPipeline.__init__`` requires a ``MetricsLogger``, and ``metrics_logger_factory``
is the only way to build one from a ``MetricsConfig``. ``MetricsConfig`` is exported and
the factory is not, so the public API stops one step short of a usable pipeline. The
assembly test below is the real assertion; the rest guard the export list itself against
the two ways it drifts, a name listed but not imported, and a name quietly deleted.
"""

import math

import bridge

# The names downstream code already relies on. A superset check rather than equality:
# adding an export is not a breaking change and must not fail the suite, while removing
# one is, and does. `metrics_logger_factory` is deliberately absent here, since its
# absence is the defect and has its own test.
ESTABLISHED_EXPORTS = frozenset(
    {
        "BridgeDataset",
        "BridgeEncoding",
        "BridgeTokenizer",
        "DatasetConfig",
        "EncodingComponent",
        "GenerationOutput",
        "MetricsConfig",
        "Model",
        "ModelConfig",
        "TrainingConfig",
        "TrainingPipeline",
        "VocabSpec",
    }
)

DATA = "tests/domain/model/data/data.csv"


def loss_of(metrics):
    """The step's loss as a plain float, detached so reading it is side-effect free."""
    value = metrics["loss"]
    return float(value.detach()) if hasattr(value, "detach") else float(value)


def test_metrics_logger_factory_is_part_of_the_public_api():
    """The one name the assembly below needs and cannot get.

    Oracle: the invariant that a package's ``__all__`` must cover the arguments its own
    exported constructors require. ``MetricsConfig`` and ``TrainingPipeline`` are both
    exported, and no exported name turns the former into the ``MetricsLogger`` the latter
    demands.
    """
    from bridge import metrics_logger_factory

    assert callable(metrics_logger_factory)
    assert "metrics_logger_factory" in bridge.__all__, (
        f"importable but unlisted, so `from bridge import *` misses it; "
        f"__all__ is {sorted(bridge.__all__)}"
    )


def test_a_working_pipeline_assembles_from_the_top_level_package_alone(tmp_path):
    """Assemble and run a real training step using only ``from bridge import ...``.

    This is the test that states the promise. Every import here is from the top-level
    package, so the test fails at exactly the point where the public API runs out, and
    passes only when a consumer could genuinely write this code. It runs one
    ``single_step`` rather than only constructing the pipeline, because a public API that
    builds an object which cannot compute anything is not sufficient either.

    Oracle: an invariant with an analytic anchor. The ``o2p`` loss is cross entropy over
    two classes per phonetic feature column, so an untrained model scores near chance,
    ``log 2`` or about 0.693 nats; the measured value with ``seed=5`` is 0.819. The
    assertion stays at finiteness and sign rather than pinning that number, because the
    claim under test is that the pipeline computes at all, and a mock or a broken forward
    pass gives ``nan``, ``inf`` or exactly zero. The paired control test below rules out
    the alternative explanation that this recipe is simply wrong.
    """
    from bridge import (
        BridgeDataset,
        DatasetConfig,
        MetricsConfig,
        Model,
        ModelConfig,
        TrainingConfig,
        TrainingPipeline,
        VocabSpec,
        metrics_logger_factory,
    )

    dataset = BridgeDataset(dataset_config=DatasetConfig(dataset_filepath=DATA))
    model = Model(
        ModelConfig(vocab=VocabSpec.from_tokenizer(dataset.tokenizer), d_model=32, nhead=2, seed=5)
    )
    training_config = TrainingConfig(
        num_epochs=1, training_pathway="o2p", model_artifacts_dir=str(tmp_path)
    )
    metrics_config = MetricsConfig(
        batch_metrics=False,
        training_metrics=False,
        validation_metrics=False,
        modes=[],
        filename=None,
    )
    pipeline = TrainingPipeline(
        model=model,
        dataset=dataset,
        training_config=training_config,
        metrics_logger=metrics_logger_factory(metrics_config, training_config),
    )

    metrics = pipeline.single_step(dataset, slice(0, 8), calculate_metrics=False)

    loss = loss_of(metrics)
    assert math.isfinite(loss), f"loss is {loss}, so the step did not really run"
    assert loss > 0.0


def test_control_the_same_pipeline_assembles_from_the_internal_module_paths(tmp_path):
    """Control for the test above, and the reason its failure means what it claims.

    Identical construction, identical step, one difference: the metrics logger comes from
    ``bridge.infra.metrics.metrics_logger`` instead of from ``bridge``. If this passes
    while the public-API version fails, the recipe is sound and the missing export is the
    whole difference. If both fail, the test file is broken and neither result says
    anything about the export list.
    """
    from bridge import (
        BridgeDataset,
        DatasetConfig,
        MetricsConfig,
        Model,
        ModelConfig,
        TrainingConfig,
        TrainingPipeline,
        VocabSpec,
    )
    from bridge.infra.metrics.metrics_logger import metrics_logger_factory

    dataset = BridgeDataset(dataset_config=DatasetConfig(dataset_filepath=DATA))
    model = Model(
        ModelConfig(vocab=VocabSpec.from_tokenizer(dataset.tokenizer), d_model=32, nhead=2, seed=5)
    )
    training_config = TrainingConfig(
        num_epochs=1, training_pathway="o2p", model_artifacts_dir=str(tmp_path)
    )
    metrics_config = MetricsConfig(
        batch_metrics=False,
        training_metrics=False,
        validation_metrics=False,
        modes=[],
        filename=None,
    )
    pipeline = TrainingPipeline(
        model=model,
        dataset=dataset,
        training_config=training_config,
        metrics_logger=metrics_logger_factory(metrics_config, training_config),
    )

    metrics = pipeline.single_step(dataset, slice(0, 8), calculate_metrics=False)

    loss = loss_of(metrics)
    assert math.isfinite(loss), f"loss is {loss}, so the control step did not really run"
    assert loss > 0.0


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

    Oracle: a pinned baseline of the names already published, captured from the export
    list at the commit where this test was written. Superset rather than equality, so
    growing the API is free and shrinking it is not.
    """
    exported = set(bridge.__all__)
    dropped = sorted(ESTABLISHED_EXPORTS - exported)
    assert dropped == [], f"no longer exported from bridge: {dropped}"
