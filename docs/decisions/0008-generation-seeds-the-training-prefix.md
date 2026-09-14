# 0008. Orthographic generation is seeded with the prefix training uses

Status: Accepted
Date: 2026-09-14

## Context

`CharacterTokenizer` lays every orthographic decoder sequence out as
`[LANG, BOS, ...chars, PAD...]`, so during teacher forcing `[BOS]` sits at position 1 and
position 0 holds a language token. `Model._generate` seeded the orthographic decoder loop
with a `(batch, 1)` tensor holding `[BOS]` alone at `position_offset=0`.

The decoder was therefore asked to continue from a state it never saw in training, and the
language token it had been conditioned on was absent entirely.

Phonology does not share the problem, and that asymmetry is what makes the fix obvious rather
than a matter of taste. Measured by instrumenting both real decoder loops:

| modality | `[BOS]` position in training | `[BOS]` position in generation | aligned |
|---|---|---|---|
| orthography | 1 | 0 | no |
| phonology | 0 | 0 | yes |

The phoneme tokenizer emits `[BOS, ...phonemes, PAD...]` with no language slot, so its
generation seed was already correct. Orthography was the odd one out.

Script: `asymmetry.py`.

## What the evidence did not say

Issue #228 attributed a visible artifact, `'longl'` for **long**, to the position mismatch.
That attribution was wrong, and reading the raw token ids rather than the decoded strings is
what showed it:

```
long -> ['[BOS]', '[BOS]', 'l', 'o', 'n', 'g', '[EOS]', '[EOS]', '[EOS]', 'l', '[EOS]']
```

The model emitted `[EOS]` in the right place. The trailing `l` is post-`[EOS]` sampling: the
batched loop runs until *every* sequence has finished, and `CharacterTokenizer.decode` strips
`[EOS]` rather than stopping at it. That is a separate defect, tracked separately, and it
affects every layout equally. Scoring the arms below therefore truncates at the first `[EOS]`,
which is what separates "generated the wrong word" from "kept going after saying it was done".

## The arms

Four layouts, each trained to memorise a closed 8-word list on `p2o`, 600 full-batch steps at
`d_model=128`, then generated from greedily. All four ran through one shared decoder loop so
only the seeding differed.

| arm | seed | exact, truncated at `[EOS]` |
|---|---|---|
| shipped | `[BOS]` at position 0 | 5/8 |
| **A** | **`[LANG, BOS]` at positions 0,1** | **8/8** |
| B | `[BOS]` at position 1 | 2/8 |
| C | drop `[LANG]` from the decoder input, matching phonology | 8/8 |
| shuffled target (control) | n/a | 0/8 |

The shuffled control scoring 0/8 is what makes the rest readable. B is worse than doing
nothing: fixing the position without supplying the language leaves the decoder at a position
whose context it never saw.

Script: `arms_228.py`.

## Choosing between A and C

A and C both reproduce the list perfectly, so memorisation cannot separate them. C is
tempting: it makes orthography structurally identical to phonology, which `docs/architecture.md`
calls the load-bearing structural fact.

The separating question is whether the decoder's language token does any work. For `p2o` the
encoder sees phonology only, which carries no language, so the decoder is the only place a
target language could enter. The lexicons contain **58 phoneme sequences shared by English and
Spanish with different spellings**: `T AO1 S` is *toss* in English and *tos* in Spanish.

Trained on those 58 pairs and generated twice, once per language seed:

| arm | EN correct | ES correct | combined | same output for both seeds | final loss |
|---|---|---|---|---|---|
| **A** | 53/58 | 57/58 | **110/116** | 2/58 | **0.0065** |
| C | 37/58 | 27/58 | 64/116 | 58/58 | 0.1683 |

C's 58/58 identical outputs is the control: it has no language input, so anything less would
mean the harness was not holding the phonology fixed. C also cannot fit the training data at
all, loss stalling at 0.168 against 0.0065, because one input maps to two answers it has no
way to tell apart.

Script: `steering_228.py`.

## Decision

`Model.generate` seeds the orthographic decoder with `encodings.orthographic.dec_input_ids[:, :2]`:
the same two positions the tokenizer puts in front of every word, taken off the encoding rather
than rebuilt in the model. The language is therefore whatever the caller asked for through
`BridgeTokenizer.encode(..., language_map=...)`, and needs no new argument.

Three things follow, and each is load-bearing:

- `orthography_decoder_loop` sizes its causal mask from the prefix actually decoded,
  `generated_orth_embeddings.shape[1]`, rather than from the step index. The two were equal
  only while the seed was one token; a two-token seed asked a 1x1 mask to cover two positions
  and raised `shape '[1, 1, 2, 2]' is invalid for input of size 1`.
- The seeded probability rows are built from the seeded tokens by `scatter_`, one per position,
  rather than being a single hardcoded row certain of `[BOS]`. A fixed row would leave the
  probability history one short of the token history.
- The orthographic placeholder component, built for phonology-only encodings, becomes
  `[--, BOS]` instead of a single column of zeros. Zero is `[BOS]`, so the old placeholder read
  as a sequence already under way, and it was too narrow to supply the prefix. `p2o` is the one
  pathway that emits orthography without consuming any, so without this the model would need a
  special case for it. `--` is the honest default: a phonology-only encoding does not say what
  language to spell in.

Confirmed on the shipped path rather than on the probe's loop: trained through the real
`TrainingPipeline` and generated through the real `Model.generate`, the memorised list comes
back 8/8, opening with `['EN', '[BOS]']`.

## Consequences

`GenerationOutput.orth_tokens` now opens with the language token. Decoded strings are
unaffected, because `CharacterTokenizer.decode` already strips language tokens along with the
other non-content tokens. Issue #163 asks for `[BOS]` to be stripped from the generate output
and is the right place to decide what that tensor should contain.

The recorded `orth_tokens` and `orth_probs` in `tests/fixtures/phon_baseline.pt` describe the
old seeding and are superseded. **The fixture was not regenerated.** The effect was measured
and bounded instead:

| pathway | `global_encoding` | `orth_tokens` | phonological |
|---|---|---|---|
| o2p | same | n/a | same |
| p2o | same | **moved** | n/a |
| p2p | same | n/a | same |
| op2op | same | **moved** | same |
| o2o | same | **moved** | n/a |

`op2op` is the load-bearing row: it generates both modalities from one encoding, and only its
orthographic half moved. Shapes are unchanged throughout. The equality assertions for
orthographic generation are reduced to shape, and the new behaviour is pinned against analytic
oracles in `tests/domain/model/test_orth_generation_prefix.py` rather than against a
re-recording.

The loop's step budget is `max_orth_seq_len - prefix_len`, so a longer seed buys fewer
generated positions and the total still cannot outrun the position table.
