# 0009. A finished sequence emits padding, and only the accepted sequence is masked

Status: Accepted
Date: 2026-09-14

## Context

Both decoder loops run until every sequence in the batch has emitted `[EOS]`. A dense batch
cannot stop for one row while others are still decoding, so that part is not optional. What
was optional, and wrong, was keeping what the finished rows went on sampling.

`CharacterTokenizer.decode` strips `[EOS]` rather than stopping at it, so that content reached
the caller inside the decoded word. Measured on an untrained model: a five-word batch left 32
non-padding tokens after a terminating `[EOS]`, and four of the five decoded to something
longer than the model produced.

The failure had the worst available shape. A word decoded correctly on its own and incorrectly
beside a longer word, so it was right in exactly the small case someone debugs with, and the
error was silent and batch-dependent.

The phonological loop had the same defect, and it took a deliberate setup to see: greedy
decoding on an untrained model terminates every item on the same step, and a lockstep batch has
no post-termination positions at all. A batch of eight with seven items finishing at step 0 and
one at step 4 left 28 positions of sampled features after their own `[EOS]`.

## Decision

A row that has emitted `[EOS]` emits padding from then on. This is the standard batched
generation guard: `next = where(finished, pad, next)`, applied **before** `finished` is updated
for that step, so the terminating `[EOS]` is itself kept and only what follows becomes padding.

Two places, `orthography_decoder_loop` and `phonology_decoder_loop`. The phonological padding
is the one-hot at `phon_pad_id`, which is the encoding `phono_sample` already produces for an
all-off vector, not a second convention.

### What is deliberately not masked

**`decode` is untouched.** It already strips padding, so a correctly padded row decodes
correctly, and the truncation belongs where the sequence is produced rather than where it is
rendered. Truncating inside `decode` would also change behaviour for a caller passing rows
built some other way, whose content after an `[EOS]` might be theirs to keep.

**`phon_vecs` and `phon_probs` keep every step.** The reason is sharper than "they are the raw
output". A suppressed position and a genuinely all-off one are both exactly `[PAD]` in
`phon_tokens` and cannot be told apart there; `phon_vecs` is the only place the difference
survives, and masking it would erase it. `phono_sample`'s docstring already turns on that
distinction, noting that an all-off vector is *reported* as all-off but *decodes* to `[PAD]`.

Unmasked is not unchanged, and an earlier draft of this record got that wrong. Suppressing a
finished row's contribution changes what the decoder reads back for that row, so `phon_probs`
and `phon_vecs` values from two steps past a row's `[EOS]` onward differ from an unsuppressed
run, measured at 293 of 293 such elements. Those positions are discarded either way. The claim
being made is that they are not masked, not that they are identical.

`orth_probs` was already cut per item by the `kept` counter, so orthography reaches a clean end
by a different route, and that asymmetry between the two modalities is pre-existing and
untouched here.

## Evidence

**The content before `[EOS]` is unchanged.** A differential between `main` and this change over
120 configurations, 2 seeds by 6 batches by 5 pathways by 2 sampling modes:

| invariant | result |
|---|---|
| orthographic rows, content up to `[EOS]` unchanged | **288 / 288** |
| phonological rows, content up to `[EOS]` unchanged | **288 / 288** |
| `global_encoding` unchanged | 120 / 120 |
| `orth_probs` lengths unchanged | 72 / 72 |

and the effect is present rather than the probe being vacuous: 27 orthographic and 111
phonological row tails changed, and every one of them is now padding.

**The contract holds over every schedule, not just the ones anyone thought of.** Termination is
scripted through the samplers so the batch finishes exactly where the sweep says, and the
expected rows are computed in plain Python from the contract rather than from the model. Every
assignment of `{step 0, step 1, step 2, never}` to each row, for batch sizes 1 to 4, plus a set
of deliberately awkward larger schedules:

| sweep | this change | `main` |
|---|---|---|
| orthographic, 346 schedules | **346 matched** | 20 matched, 326 violations |
| phonological, 344 schedules | **344 matched** | 18 matched, 326 violations |

The controls in that sweep are what make it readable: a row that never terminates must contain
no padding at all, which separates "pads after its own termination" from "pads whenever the
loop ends", and a lockstep batch must gain no padding at all.

**It reaches the user.** A model trained to memorise eight words of deliberately ragged length,
then generated from as one batch. Both arms load the *same* checkpoint file, so the weights are
bitwise identical and the only variable is which `model.py` is imported:

| | this change | pre-fix |
|---|---|---|
| words decoded correctly from a mixed batch, seed A | **8 / 8** | 1 / 8 |
| words decoded correctly from a mixed batch, seed B | **8 / 8** | 2 / 8 |
| rows that decode differently alone than in a batch | **0 / 8** | 7 / 8 and 6 / 8 |

That last row is the symptom the issue calls the worst kind: right in the small case someone
debugs with, wrong in the batch they ship.

**It does not change control flow.** Step counts per `generate` call are identical between the
two trees in 300 of 300 CPU configurations and 200 of 200 CUDA configurations, exact integers,
and 102 of those 300 are configurations where the guard actually rewrote a cell rather than
being a no-op. A finished row's substituted feed never reaches an unfinished row: 501 of 501
never-terminating rows are bitwise identical including their probability histories.

**On the GPU it behaves as it does on CPU.** 0 non-padding positions out of 7816 orthographic
and 3077 phonological post-`[EOS]` positions on an RTX 5080, against a pre-fix control showing
7778 and 3077 violations on the same configurations. CPU and CUDA generation from identical
weights agree exactly, padding tails included.

**It costs one operation per decoding step.** Quoted as an exact op count rather than a time,
because wall clock is not reportable on this host: the same-arm interquartile width was 66 ms
against a sub-millisecond effect. Counting aten dispatches per `generate` call, against the
same call on `main`:

| call | with the guard | without | added |
|---|---|---|---|
| `p2o`, 10 decoding steps | 1508 | 1487 | **+21**, 1.4% |
| `o2p`, 2 decoding steps | 227 | 221 | **+6**, 2.7% |
| `p2o`, batch of 1 | 340 | 335 | +5 |

One `torch.where` per step per loop, plus one constant allocation per call. Both pad constants
are loop invariant and hoisted out; an earlier revision rebuilt them per step, and a cost
measurement taken against that revision is not a measurement of what ships.

Scripts: `vv1_differential.py`, `vv2_exhaustive_schedule.py`, `vv3_phon_schedule.py`,
`vv4_cuda.py`, and six independent falsification sweeps.

## Consequences

`GenerationOutput.orth_tokens` and `phon_tokens` now end cleanly: everything after a row's
terminator is padding, so a consumer can find where a row's real output ends, which was not
previously possible for either.

The end marker is `[EOS]`, not the first `[PAD]`. `[PAD]` is an ordinary token the model can
sample mid-word, and does, at a rate that is a property of the model rather than a constant:
one sweep counted 23 orthographic and 37 phonological occurrences before a terminator across 40
configurations, and another found 81% of terminated orthographic rows carrying one at a
different configuration. Both counts are identical with and without this guard. So "scan to the
first padding" is the wrong way to read these tensors and "scan to the first `[EOS]`" is the
right one.

**`tests/fixtures/phon_baseline.pt` cannot regress-test this.** Measured rather than assumed: no
orthographic item in that fixture ever emits `[EOS]`, on any pathway, and every phonological
item terminates at position 1 in lockstep. There are zero post-termination positions in it, so
it is blind to this whole class of change by construction. It passed unchanged, and that fact
carries no information. Coverage comes from
`tests/domain/model/test_generation_stops_at_eos.py` instead.

Issue #163 asks for `[BOS]` to be stripped from the generate output, and decision 0008 left the
leading `[LANG, BOS]` prefix in place. Both are about what `orth_tokens` should contain and are
still open; this record only settles what happens after the sequence ends.
