"""The tensors of a training example, each named for its one reader.

An example is an input and a target, two different things. An encoder reads the input:
:class:`Tokens`, ids and a pad mask. A decoder is fed the target and scored against it:
:class:`Target`, ids, a pad mask and one label per position. :class:`BridgeInput` and
:class:`BridgeTarget` hold the two modalities of one side, and a :class:`BridgeBatch` is one
of each. The two sides may hold the same strings, which is repetition, and nothing treats
that case differently.

Both modalities have the same shape: ``(batch, sequence)`` integer ids. Orthographic ids
index the character vocabulary; phonological ids index
:class:`~bridge.core.phonreps.PhonemeTable` rows (*which phoneme*, not which feature).

Every class is frozen and validates when it is built, so a tensor of the wrong shape, or an
input put where a target belongs, fails where it was assembled.
"""

from dataclasses import dataclass
from typing import Any

import torch


def pad_rows(
    rows: list[list[int]], pad: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """``rows`` padded with ``pad`` to one width as a tensor, and its pad mask."""
    width = max(len(row) for row in rows)
    ids = torch.tensor(
        [[*row, *[pad] * (width - len(row))] for row in rows], dtype=torch.long, device=device
    )
    return ids, ids == pad


def _check_ids_and_mask(owner: str, ids: torch.Tensor, pad_mask: torch.Tensor) -> None:
    """``ids`` are non-negative ``(batch, sequence)`` integers and ``pad_mask`` is a boolean
    tensor covering exactly those positions, on the same device."""
    for name, tensor in (("ids", ids), ("pad_mask", pad_mask)):
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f"{owner} {name} must be a torch.Tensor, got {type(tensor).__name__}")
        if tensor.dim() != 2:
            raise ValueError(
                f"{owner} {name} must be 2-dimensional (batch, sequence), got shape "
                f"{tuple(tensor.shape)}"
            )
    if ids.dtype not in (torch.long, torch.int):
        raise ValueError(f"{owner} ids must have dtype torch.long or torch.int, got {ids.dtype}")
    if pad_mask.dtype != torch.bool:
        raise ValueError(f"{owner} pad_mask must have dtype torch.bool, got {pad_mask.dtype}")
    if ids.shape != pad_mask.shape:
        raise ValueError(
            f"Shape mismatch: {owner} ids are {tuple(ids.shape)} but pad_mask is "
            f"{tuple(pad_mask.shape)}"
        )
    # Placement before content: the range check below reads the data.
    if pad_mask.device != ids.device:
        raise ValueError(
            f"Device mismatch: {owner} ids on {ids.device}, pad_mask on {pad_mask.device}"
        )
    if torch.any(ids < 0):
        raise ValueError(f"{owner} ids cannot contain negative indices")


@dataclass(frozen=True, slots=True)
class Tokens:
    """What an encoder reads: one modality of the input strings.

    ``ids`` is ``(batch, sequence)`` int64 and ``pad_mask`` is the same shape, ``True``
    where padding.
    """

    ids: torch.Tensor
    pad_mask: torch.Tensor

    def __post_init__(self) -> None:
        _check_ids_and_mask("Tokens", self.ids, self.pad_mask)

    def to(self, device: torch.device) -> "Tokens":
        """These tokens on ``device``."""
        return Tokens(self.ids.to(device), self.pad_mask.to(device))


@dataclass(frozen=True, slots=True)
class Target:
    """What a decoder is fed and scored against: one modality of the target strings.

    ``ids`` and ``pad_mask`` are what the decoder is fed. ``labels[:, i]`` is what it must
    emit after reading ``ids[:, : i + 1]``, so there is one label per decoder position.
    Orthographic labels are ``(batch, sequence)`` character ids. Phonological labels are
    ``(batch, sequence, features)``, one class per phonetic feature, in feature space.
    """

    ids: torch.Tensor
    pad_mask: torch.Tensor
    labels: torch.Tensor

    def __post_init__(self) -> None:
        _check_ids_and_mask("Target", self.ids, self.pad_mask)
        if not isinstance(self.labels, torch.Tensor):
            raise ValueError(
                f"Target labels must be a torch.Tensor, got {type(self.labels).__name__}"
            )
        # The loss takes int64 class indices and no other integer type.
        if self.labels.dtype != torch.long:
            raise ValueError(f"Target labels must have dtype torch.long, got {self.labels.dtype}")
        if self.labels.shape[:2] != self.ids.shape:
            raise ValueError(
                f"Shape mismatch: Target labels are {tuple(self.labels.shape)}, which is not "
                f"one label per decoder position; ids are {tuple(self.ids.shape)}"
            )
        if self.labels.device != self.ids.device:
            raise ValueError(
                f"Device mismatch: Target ids on {self.ids.device}, labels on {self.labels.device}"
            )

    def to(self, device: torch.device) -> "Target":
        """This target on ``device``."""
        return Target(self.ids.to(device), self.pad_mask.to(device), self.labels.to(device))


def _check_modalities(owner: str, kind: type, parts: dict[str, Any]) -> None:
    """Every modality in ``parts`` is a ``kind``, and they agree on batch size and device."""
    for modality, part in parts.items():
        if not isinstance(part, kind):
            raise TypeError(
                f"{owner}.{modality} must be a {kind.__name__}, got {type(part).__name__}"
            )
    sizes = {modality: part.ids.size(0) for modality, part in parts.items()}
    if len(set(sizes.values())) > 1:
        raise ValueError(f"Batch size mismatch in {owner}: {sizes}")
    devices = {modality: str(part.ids.device) for modality, part in parts.items()}
    if len(set(devices.values())) > 1:
        raise ValueError(f"Device mismatch in {owner}: {devices}")


@dataclass(frozen=True, slots=True, kw_only=True)
class BridgeInput:
    """What the encoders read: the input strings in each modality.

    A modality the input does not carry is ``None``. A nonword has no phonology, and
    phonemes given without a spelling have no orthography. A pathway reads only its own
    modalities, and :meth:`read` is how it asks for one.

    Each modality is passed by name, ``BridgeInput(phon=...)``: both are ``Tokens``, so
    the name is all that says which one a tensor is.
    """

    orth: Tokens | None = None
    phon: Tokens | None = None

    def __post_init__(self) -> None:
        if not self._carried():
            raise ValueError("A BridgeInput carries at least one modality; both are None")
        _check_modalities("BridgeInput", Tokens, self._carried())

    def _carried(self) -> dict[str, Tokens]:
        """The modalities this input carries."""
        both = (("orth", self.orth), ("phon", self.phon))
        return {modality: tokens for modality, tokens in both if tokens is not None}

    def read(self, modality: str, pathway: str) -> Tokens:
        """The ``modality`` of this input, for ``pathway`` to read. Raises when the input
        does not carry it."""
        tokens: Tokens | None = getattr(self, modality)
        if tokens is None:
            raise ValueError(
                f"pathway {pathway!r} reads {modality}, which this input does not carry"
            )
        return tokens

    @property
    def device(self) -> torch.device:
        """The device every tensor of this input lives on."""
        return next(iter(self._carried().values())).ids.device

    def to(self, device: torch.device) -> "BridgeInput":
        """This input on ``device``."""
        return BridgeInput(**{m: tokens.to(device) for m, tokens in self._carried().items()})

    def __len__(self) -> int:
        """The batch size."""
        return next(iter(self._carried().values())).ids.size(0)


@dataclass(frozen=True, slots=True)
class BridgeTarget:
    """What the decoders are fed and scored against: the target strings in each modality."""

    orth: Target
    phon: Target

    def __post_init__(self) -> None:
        _check_modalities("BridgeTarget", Target, {"orth": self.orth, "phon": self.phon})
        # Also what tells the two modalities apart, so swapped ones cannot pass.
        for modality, target, rank in (("orth", self.orth, 2), ("phon", self.phon, 3)):
            if target.labels.dim() != rank:
                raise ValueError(
                    f"BridgeTarget.{modality} labels must be {rank}-dimensional, got shape "
                    f"{tuple(target.labels.shape)}"
                )

    @property
    def device(self) -> torch.device:
        """The device every tensor of this target lives on."""
        return self.orth.ids.device

    def to(self, device: torch.device) -> "BridgeTarget":
        """This target on ``device``."""
        return BridgeTarget(orth=self.orth.to(device), phon=self.phon.to(device))

    def __len__(self) -> int:
        """The batch size."""
        return self.orth.ids.size(0)


@dataclass(frozen=True, slots=True)
class BridgeBatch:
    """Training examples: what the model reads, and what it is taught to write."""

    input: BridgeInput
    target: BridgeTarget

    def __post_init__(self) -> None:
        for name, side, kind in (
            ("input", self.input, BridgeInput),
            ("target", self.target, BridgeTarget),
        ):
            if not isinstance(side, kind):
                raise TypeError(
                    f"BridgeBatch.{name} must be a {kind.__name__}, got {type(side).__name__}"
                )
        if len(self.input) != len(self.target):
            raise ValueError(
                f"Batch size mismatch in BridgeBatch: {len(self.input)} inputs, "
                f"{len(self.target)} targets"
            )
        if self.input.device != self.target.device:
            raise ValueError(
                f"Device mismatch in BridgeBatch: input on {self.input.device}, target on "
                f"{self.target.device}"
            )

    def to(self, device: torch.device) -> "BridgeBatch":
        """This batch on ``device``."""
        return BridgeBatch(input=self.input.to(device), target=self.target.to(device))

    def __len__(self) -> int:
        """The batch size."""
        return len(self.input)
