from collections.abc import Callable
from typing import Annotated

import torch
from pydantic import BaseModel, ConfigDict, Field, model_validator


def validate_global_encoding(v: torch.Tensor) -> torch.Tensor:
    """Validates the global encoding tensor has correct shape and properties."""
    if not isinstance(v, torch.Tensor):
        raise ValueError("global_encoding must be a torch.Tensor")
    if v.dim() != 3:
        raise ValueError("global_encoding must be 3-dimensional (batch × embedding × model)")
    if v.size(1) <= 0:
        raise ValueError("embedding dimension must be positive")
    if v.size(2) <= 0:
        raise ValueError("model dimension must be positive")
    return v


def validate_nested_tensors(
    v: list[list[torch.Tensor]] | None,
    name: str,
    element_check: Callable[[torch.Tensor, str], None] | None = None,
) -> list[list[torch.Tensor]] | None:
    """Validate a ``list[list[Tensor]]`` of per-step, per-batch 1-D tensors.

    The structural walk is plain Python and costs nothing. ``element_check`` is numeric,
    so it runs **once** over every row stacked together rather than per tensor: at batch
    x steps the per-tensor form is thousands of whole-tensor reductions, each a device
    sync, and it dominated ``generate()`` on CUDA. The trade is that a numeric failure
    names the field rather than the offending ``[batch][step]``.
    """
    if v is None:
        return None

    if not isinstance(v, list) or not all(isinstance(x, list) for x in v):
        raise ValueError(f"{name} must be a list of lists of tensors")

    stackable: tuple[int, torch.device] | None = None
    for batch_idx, sequence in enumerate(v):
        for step_idx, tensor in enumerate(sequence):
            label = f"{name}[{batch_idx}][{step_idx}]"
            if not isinstance(tensor, torch.Tensor):
                raise ValueError(f"{label} must be a tensor")
            if tensor.dim() != 1:
                raise ValueError(f"{label} must be 1-dimensional")
            if element_check is None:
                # No numeric check means no stack, and nothing to agree on. `phon_tokens`
                # is this case, and its rows are genuinely ragged: they hold each
                # position's active feature indices, and phonemes differ in how many they
                # have.
                continue
            # Width and device are tensor metadata, so agreeing on them here is free, and
            # it keeps a mismatch inside pydantic. The stack below would otherwise raise a
            # bare RuntimeError straight past the validator.
            if stackable is None:
                stackable = (tensor.shape[0], tensor.device)
            elif tensor.shape[0] != stackable[0]:
                raise ValueError(f"{label} has length {tensor.shape[0]}, expected {stackable[0]}")
            elif tensor.device != stackable[1]:
                raise ValueError(f"{label} is on device {tensor.device}, expected {stackable[1]}")

    if stackable is not None and element_check is not None:
        element_check(torch.stack([tensor for sequence in v for tensor in sequence]), name)

    return v


def check_is_distribution(rows: torch.Tensor, name: str) -> None:
    """Every row is a normalized probability distribution: sums to 1, entries in [0, 1]."""
    sums = rows.sum(-1)
    if not torch.allclose(sums, torch.ones_like(sums), atol=1e-5):
        raise ValueError(f"{name} probabilities must sum to 1")
    if torch.any(rows < 0) or torch.any(rows > 1):
        raise ValueError(f"{name} probabilities must be between 0 and 1")


def check_in_unit_range(rows: torch.Tensor, name: str) -> None:
    """Independent per-feature probabilities: entries in [0, 1], no sum constraint."""
    if torch.any(rows < 0) or torch.any(rows > 1):
        raise ValueError(f"{name} probabilities must be between 0 and 1")


def check_is_binary(rows: torch.Tensor, name: str) -> None:
    """Binary feature vectors: every entry is exactly 0 or 1."""
    if not torch.all((rows == 0) | (rows == 1)):
        raise ValueError(f"{name} must contain only binary values")


def validate_orthographic_tokens(v: torch.Tensor | None) -> torch.Tensor | None:
    """Validates orthographic token tensor has correct shape and properties."""
    if v is None:
        return None

    if not isinstance(v, torch.Tensor):
        raise ValueError("orth_tokens must be a torch.Tensor")
    if v.dim() != 2:
        raise ValueError("orth_tokens must be 2-dimensional (batch × sequence)")
    if v.dtype not in [torch.long, torch.int]:
        raise ValueError("orth_tokens must have dtype torch.long or torch.int")
    if torch.any(v < 0):
        raise ValueError("orth_tokens cannot contain negative indices")

    return v


class GenerationOutput(BaseModel):
    """
    Unified output format for all generation pathways.

    This model provides a consistent structure for generation outputs across different
    pathways (o2p, p2o, op2op, ...), making analysis and experimentation easier through
    a predictable interface.

    Attributes:
        global_encoding: Tensor of shape (batch_size, d_embedding, d_model) containing
            the global representation used for generation.
        orth_probs: Nested list of tensors containing probability distributions for
            each orthographic generation step. None for non-orthographic pathways.
        orth_tokens: Tensor of generated orthographic token indices. None for
            non-orthographic pathways.
        phon_probs: Nested list of tensors containing probability distributions for
            each phonological feature. None for non-phonological pathways.
        phon_vecs: Nested list of binary tensors representing generated phonological
            feature vectors. None for non-phonological pathways.
        phon_tokens: Nested list of tensors containing active feature indices for
            each generated phoneme. None for non-phonological pathways.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    global_encoding: Annotated[torch.Tensor, Field(validate_default=True)]
    orth_probs: list[list[torch.Tensor]] | None = None
    orth_tokens: torch.Tensor | None = None
    phon_probs: list[list[torch.Tensor]] | None = None
    phon_vecs: list[list[torch.Tensor]] | None = None
    # One entry longer than phon_probs/phon_vecs: it includes the [BOS] seed position,
    # which was never sampled and so has no probability or vector of its own.
    phon_tokens: list[list[torch.Tensor]] | None = None

    @model_validator(mode="after")
    def validate_structure(self) -> "GenerationOutput":
        # Validate individual components. Each validator returns its input unchanged.
        validate_global_encoding(self.global_encoding)
        validate_orthographic_tokens(self.orth_tokens)
        # Orthographic steps are a softmax over the vocabulary; phonological steps are
        # independent per-feature probabilities, so only the former must sum to 1.
        validate_nested_tensors(self.orth_probs, "orth_probs", check_is_distribution)
        validate_nested_tensors(self.phon_probs, "phon_probs", check_in_unit_range)
        validate_nested_tensors(self.phon_vecs, "phon_vecs", check_is_binary)
        validate_nested_tensors(self.phon_tokens, "phon_tokens")

        # Cross-component validation
        batch_size = self.global_encoding.size(0)

        # Validate batch size consistency for orthographic components
        if self.orth_probs is not None and len(self.orth_probs) != batch_size:
            raise ValueError("orth_probs batch size mismatch")
        if self.orth_tokens is not None and self.orth_tokens.size(0) != batch_size:
            raise ValueError("orth_tokens batch size mismatch")

        # Validate batch size consistency for phonological components
        for name, field in [
            ("phon_probs", self.phon_probs),
            ("phon_vecs", self.phon_vecs),
            ("phon_tokens", self.phon_tokens),
        ]:
            if field is not None and len(field) != batch_size:
                raise ValueError(f"{name} batch size mismatch")

        # Validate orthographic component consistency
        if (self.orth_probs is None) != (self.orth_tokens is None):
            raise ValueError(
                "orth_probs and orth_tokens must either both be present or both be None"
            )

        # Validate phonological component consistency
        phon_fields = [self.phon_probs, self.phon_vecs, self.phon_tokens]
        if any(f is not None for f in phon_fields) and not all(f is not None for f in phon_fields):
            raise ValueError(
                "All phonological components (probs, vecs, tokens) must be present if any are"
            )

        # Validate pathway consistency
        has_orth = self.orth_tokens is not None
        has_phon = self.phon_tokens is not None
        if not (has_orth or has_phon):
            raise ValueError("At least one modality (orthographic or phonological) must be present")

        # Validate device consistency
        devices = {self.global_encoding.device}
        if self.orth_tokens is not None:
            devices.add(self.orth_tokens.device)
        if self.orth_probs is not None:
            devices.update(prob.device for probs in self.orth_probs for prob in probs)
        for field in [self.phon_probs, self.phon_vecs, self.phon_tokens]:
            if field is not None:
                devices.update(t.device for seq in field for t in seq)
        if len(devices) > 1:
            raise ValueError("All tensors must be on the same device")

        return self
