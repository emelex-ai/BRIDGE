# 0005. Phonological metrics take the pad id as an argument, and every historical value is void

Status: Accepted
Date: 2026-08-30

## Context

Every metric in `bridge/application/training/phon_metrics.py` masked padding with the literal
`2`, at six sites. `2` is the **orthographic** pad id. The phonological one is `35`, and the
two id spaces are described in `docs/architecture.md`: nothing but naming separates them, and
confusing them produces silently wrong output rather than an error.

Phonological targets take values in `{0, 1, 35}`. The value `2` never occurs, so `!= 2` was a
tautology and the mask selected every position, padded ones included. Measured on
`["a", "cat", "elephant"]`: `!= 2` keeps 100.0% of positions while 41.7% of them are padding.

The loss was never affected. It passes `ignore_index=vocab.phon_pad_id` to `CrossEntropyLoss`,
which is correct. Models trained correctly. Only the numbers describing them were wrong, which
is exactly why this survived: nothing looked broken.

## The size of it

Measured by loading the real `phon_metrics.py` source twice, once verbatim and once with the
sentinel replaced, and running both on identical inputs. The instrument was validated first on
a batch of words with equal phoneme counts, where no padding exists and the two builds must
agree: zero differences across all eight metrics.

Against the committed fixtures `tests/application/training/data/phon_{true,pred}.pt`, which
are **54.2% padding**:

| metric | reported | corrected | ratio |
|---|---|---|---|
| `phon_euclidean_distance` | 112.257 | **0.234** | 0.002 |
| `closest_phoneme_l2_accuracy` | 0.3894 | **0.8504** | 2.18 |
| `closest_phoneme_cosine_accuracy` | 0.4483 | **0.9738** | 2.17 |
| `phon_cosine_similarity` | 0.5262 | **0.9487** | 1.80 |

Identity cases, a tensor compared against itself, are unmoved to 1e-6, which is what confirms
the substitution touched only the sentinel.

Both filed descriptions of the direction were half right. Accuracy metrics were biased
**downward**, because padded rows are entirely `35` while predictions there are `0`/`1`, so
padding scored as wrong. Similarity and distance metrics were biased **upward**, because
padded rows are near-identical to each other. Issue #227 predicted only the first; issue #224
predicted only the second.

Script: `metric_delta_227.py` and `golden_phon_metrics.py`, run at `825293e`, CPU,
torch 2.12.0+cu130.

## Decision

`calculate_phon_metrics` takes `phon_pad_id: int` as a required argument, mirroring
`calculate_orth_metrics`, which already takes `orth_pad_id`. Each of the six helpers that
builds a mask takes it too. `TrainingPipeline.compute_metrics` passes
`self.model.model_config.vocab.phon_pad_id`.

Required rather than defaulted. A default would let a caller silently reintroduce the defect,
and the value is already on the config every caller holds.

## Consequences

**Every phonological metric BRIDGE has reported to date is void.** Not merely imprecise:
`phon_euclidean_distance` was off by a factor of roughly 480, and the accuracies by roughly
2.2. The size of the error tracks how much padding each batch happened to carry, so it varies
between runs with identical model quality. Metric rows recorded before this change cannot be
compared with rows recorded after, and cannot be corrected after the fact because the padding
fraction per batch was never recorded.

The pinned values in `tests/application/training/test_phon_metrics.py` moved accordingly. The
fixtures themselves are untouched; only the expectations changed, and both the old and new
numbers are recorded above so the move is auditable rather than a silently regenerated
baseline.

The masks remain elementwise while the reshape that follows them assumes whole rows are
selected. That holds because a padded target row is entirely `35` and a real one contains no
`35`, verified on the fixtures: the fraction of elements equal to `35` and the fraction of
all-`35` rows are both 0.542067. A future target encoding that broke that property would
produce a reshape error rather than a wrong number, which is the acceptable failure.
