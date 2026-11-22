# 0014. A config that validates builds a model that runs every pathway

Status: Accepted
Date: 2026-10-09

## Context

`ModelConfig` carried one size rule, a pydantic field validator on `d_model` that looked
`nhead` up in `info.data` and raised if it did not divide. It rejected nothing. Measured at
`32b66a4`, pydantic 2.13.4, torch 2.12.0+cu130, CPU, by `sys.settrace` on the validator
itself (`scratchpad/probe_nhead.py`):

| config | validator called | `nhead` in `info.data` | result |
|---|---|---|---|
| `d_model=32, nhead=2` | once | no | validates |
| `d_model=10, nhead=4` | once | no | validates |
| `nhead=3`, `d_model` left at 64 | never | | validates |

Two separate causes. Pydantic validates fields in declaration order and `d_model` is
declared before `nhead`, so the lookup always came back empty. And field validators do not
run on defaults, so the commonest config, one that sets only `nhead`, never reached it.

`Model(ModelConfig(d_model=10, nhead=4))` then raised
`AssertionError: embed_dim must be divisible by num_heads` at
`torch/nn/modules/activation.py:1193`, four calls below the line that built the model and
naming neither field the researcher typed.

No other size was bounded. What each unbounded value did, same conditions, with
validation bypassed by `model_construct`, one `single_step` over `slice(0, 8)` and one
`generate` per pathway (`scratchpad/probe_degenerate.py`, `d_model=16`, seed 5):

| value | builds | what happens next |
|---|---|---|
| `nhead=0`, `d_model=0`, or negative | no | `ValueError` or `RuntimeError` from torch |
| any layer count negative | yes | the stack has 0 layers, as if 0 had been written |
| an encoder or decoder count of 0 | yes | `IndexError` from torch on the 3 pathways that use that stack; the other 2 run |
| `num_mixing_enc_layers=0` | yes | `IndexError` on all 5 pathways |
| `d_embedding=0` | yes | trains to a finite loss on all 5; `generate` raises |

Reading torch suggested a decoder with no layers would run as an identity, because
`TransformerDecoder.forward` loops over its layers. It does not run:
`transformer.py:638` reads `self.layers[0]` before the loop, as `:448` does for the
encoder. The measurement overturned the reading.

`d_embedding=0` is the silent one. The global memory has shape `(batch, 0, d_model)`, so
the decoder has nothing to attend to. Two `o2p` forwards differing only in the
orthographic input gave logits that were bitwise identical, against a maximum difference
of 5.498e-01 at `d_embedding=1` and 4.884e-01 at 2, which is the control showing the
comparison can see a difference (`scratchpad/probe_d_embedding.py`). The model trains and
reports a loss while reading nothing.

## Decision

**Every size in `ModelConfig` has a floor of one, and `nhead` must divide `d_model`.** The
five layer counts, `d_model`, `nhead` and `d_embedding` are declared `ge=1`. Divisibility
is a `model_validator(mode="after")`, which runs once every field is set, defaults
included, and only if each field passed its own floor.

The contract this buys is the title: a config that validates builds a model on which all
five pathways train and generate. `docs/architecture.md` already promised that of the
model; now the config cannot describe a model that breaks it.

The divisibility error is written for the person who typed the config. It names both
fields with their values, says why the rule exists, and offers the nearest values that
would work on each side:

```
d_model=10 is not divisible by nhead=4. Each attention head gets d_model / nhead
dimensions, so that must be a whole number. Nearest fixes: keep d_model=10 and set nhead
to 2 or 5, or keep nhead=4 and set d_model to 8 or 12.
```

## The alternative not taken

A floor of zero on the four modality stacks would let a researcher who trains only `p2p`
write `num_orth_enc_layers=0` and skip parameters they never use. That works today for
the two pathways that avoid the stack. It was rejected because the other three then fail
with `IndexError: index 0 is out of range` from inside torch, and nothing in a
`ModelConfig` says which pathways the model is for; that lives in `TrainingConfig`, and
generation takes a pathway per call.

Raising a floor later breaks configs. Lowering one breaks nothing. If pathway-specific
models become wanted, the change is to lower these four floors and reject the affected
pathways at the boundary `docs/decisions/0011` established, in the same change.

## Consequences

A mistake in a size fails when the config is built, before any data is loaded or any
model allocated, and the error names the field.

Checkpoints are unaffected. `save_checkpoint` pickles the `ModelConfig` object and pickle
restores state without validating, so a checkpoint holding a size these floors reject
still loads. Measured by round trip through `torch.save` and `torch.load` on a config
built with `model_construct` (`scratchpad/probe_after.py`).

The floors are compared against torch rather than restated in a test.
`tests/domain/datamodels/test_model_config.py` asserts that the config and
`nn.MultiheadAttention` agree on every `(d_model, nhead)` pair in a grid of 405, so a
torch release that changes what it accepts fails that test. The same file builds a model
with every size at its floor and runs all five pathways. Each assertion was seen to fail
first: all 29 tests against the dead validator, then five planted defects one at a time
(a floor too low, a floor too high, a wrong width offered, a head count offered that is
not the nearest, the divisibility rule removed).

Not covered: an odd `nhead` validates and runs, but building the model prints a torch
`UserWarning` three times, once per encoder stack, saying it is declining a nested-tensor
fast path. Measured at `nhead` 1 and 3 against none at 2 and 4. That is torch describing
inference speed, not a wrong result, and silencing it is a separate change.
