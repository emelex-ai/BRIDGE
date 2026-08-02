"""Pins ``TrainingPipeline._create_sub_slices``.

``single_step``'s accumulated-gradient path loops over these sub-slices and carries the
final iteration's logits/components out to ``compute_metrics``. That makes two
properties load-bearing:

* the sub-slices must **tile the batch exactly** — a gap silently drops training data
  from the accumulated loss, an overlap double-counts it;
* an empty batch slice yields **no** sub-slices, so the loop body never runs and the
  loop-carried values stay ``None``. ``single_step`` raises explicitly for that case.

The method touches no instance state, so it is exercised on a bare instance rather than
standing up a full pipeline.
"""

import pytest

from bridge.application.training.training_pipeline import TrainingPipeline


@pytest.fixture(scope="module")
def pipeline():
    """A TrainingPipeline shell — `_create_sub_slices` reads no attributes."""
    return object.__new__(TrainingPipeline)


def flatten(sub_slices):
    return [i for s in sub_slices for i in range(s.start, s.stop)]


@pytest.mark.parametrize(
    ("batch_slice", "num_chunks", "expected"),
    [
        (slice(0, 8), 2, [slice(0, 4), slice(4, 8)]),
        (slice(0, 8), 4, [slice(0, 2), slice(2, 4), slice(4, 6), slice(6, 8)]),
        (slice(0, 8), 1, [slice(0, 8)]),
        (slice(10, 18), 2, [slice(10, 14), slice(14, 18)]),
        # Not evenly divisible: 7 // 2 == 3, so chunks are 3, 3, 1.
        (slice(0, 7), 2, [slice(0, 3), slice(3, 6), slice(6, 7)]),
        # More chunks than items: chunk_size floors to 1.
        (slice(0, 3), 10, [slice(0, 1), slice(1, 2), slice(2, 3)]),
        (slice(5, 6), 3, [slice(5, 6)]),
    ],
)
def test_sub_slices(pipeline, batch_slice, num_chunks, expected):
    assert pipeline._create_sub_slices(batch_slice, num_chunks) == expected


@pytest.mark.parametrize("size", [1, 2, 3, 5, 8, 16, 33])
@pytest.mark.parametrize("num_chunks", [1, 2, 3, 4, 8])
def test_sub_slices_tile_the_batch_exactly(pipeline, size, num_chunks):
    """No gaps and no overlaps: every index appears exactly once, in order."""
    start = 7
    batch_slice = slice(start, start + size)
    sub_slices = pipeline._create_sub_slices(batch_slice, num_chunks)

    assert flatten(sub_slices) == list(range(start, start + size))
    assert all(s.stop > s.start for s in sub_slices), "no empty sub-slice"


def test_empty_batch_slice_yields_no_sub_slices(pipeline):
    """The precondition behind ``single_step``'s explicit empty-batch error.

    With zero sub-slices the accumulation loop never executes, so the logits and
    encoding components it carries forward are never bound.
    """
    assert pipeline._create_sub_slices(slice(4, 4), 2) == []


def test_sub_slices_never_exceed_the_batch(pipeline):
    for num_chunks in (1, 3, 7):
        for s in pipeline._create_sub_slices(slice(2, 11), num_chunks):
            assert 2 <= s.start < s.stop <= 11
