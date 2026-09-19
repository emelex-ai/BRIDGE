"""What a training run emits, one record at a time.

The shared vocabulary for a run. `TrainingPipeline.train_steps` emits the `train` ones; a
caller's own loop emits the rest, since the library runs no loop of its own. That is the
point of keeping all four phases here rather than only the one the library produces: a
downstream logger can consume any BRIDGE run without inventing its own record type.

See docs/decisions/0013-the-caller-owns-the-loop-in-fact.md, which supersedes the half of
0006 that shipped the loop alongside the step.
"""

from dataclasses import dataclass, field
from typing import Literal

import torch

# Per-step results carry the JSON-encoded `word` alongside loss tensors and scalars;
# per-epoch aggregates carry tensors and floats only.
type EventMetrics = dict[str, torch.Tensor | float | str]

TrainingPhase = Literal["train", "validation", "test", "epoch"]


@dataclass(frozen=True, slots=True)
class TrainingEvent:
    """One moment in a run: a training step, a validation pass, or an epoch summary.

    A frozen dataclass rather than a pydantic model, unlike the configs and
    `GenerationOutput`. Those sit on validation boundaries where a caller supplies the
    values; this is a carrier the pipeline fills in itself, one per optimizer step, so
    per-field validation would cost something on the hot path and defend against nothing.

    Attributes:
        phase: Which kind of moment this is.

            ``train``       one optimizer step, emitted by `train_steps`. ``step`` is its
                            index within the epoch.
            ``validation``  a validation pass. Emitted by the caller's loop.
            ``test``        a held-out test pass. Emitted by the caller's loop.
            ``epoch``       an epoch summary. Emitted by the caller's loop, which is also
                            what decides what an epoch aggregate means.
        epoch: Zero-based epoch index, counting from `start_epoch` on a resumed run.
        step: Zero-based step index within the epoch, for ``train`` only.
        metrics: The metrics for this moment. `train_steps` fills these from
            `single_step`, unprefixed. A caller aggregating across steps chooses its own
            key convention for the rows it emits.
    """

    phase: TrainingPhase
    epoch: int
    metrics: EventMetrics = field(default_factory=dict)
    step: int | None = None

    def __post_init__(self) -> None:
        if (self.step is None) == (self.phase == "train"):
            raise ValueError(
                f"a {self.phase!r} event has step={self.step!r}: `train` events carry a "
                "step index and every other phase covers a whole epoch, so must not"
            )
