"""What a training run emits, one record at a time.

`TrainingPipeline.run_train_val_loop` yields these. A caller reads the stream and decides
what to do with each: log it, checkpoint on it, stop early on it. The pipeline no longer
decides any of that, which is why the record has to say enough for the caller to tell one
kind of moment from another.

See docs/decisions/0006-the-caller-owns-the-training-loop.md.
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

            ``train``       one optimizer step. ``step`` is its index within the epoch.
            ``validation``  the validation partition, once per epoch. ``step`` is None.
            ``test``        the held-out test set, once per epoch, when one is configured.
            ``epoch``       the epoch summary, always last for its epoch. Its metrics are
                            the aggregate the loop used to yield before it yielded per
                            step, so a caller that only wants epoch rows filters on this.
        epoch: Zero-based epoch index, counting from `start_epoch` on a resumed run.
        step: Zero-based step index within the epoch, for ``train`` only.
        metrics: The metrics for this moment. Keys are prefixed by phase on the aggregate
            rows (``train_``, ``valid_``, ``test_``) and unprefixed on ``train`` steps.
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
