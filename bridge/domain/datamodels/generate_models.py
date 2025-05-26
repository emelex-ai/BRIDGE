from typing import Annotated
from pydantic import BaseModel, Field, model_validator
import torch


def validate_global_encoding(v: torch.Tensor) -> torch.Tensor:
    """Validates the global encoding tensor has correct shape and properties."""
    if not isinstance(v, torch.Tensor):
        raise ValueError("global_encoding must be a torch.Tensor")
    if v.dim() != 3:
        raise ValueError(
            "global_encoding must be 3-dimensional (batch × embedding × model)"
        )
    if v.size(1) <= 0:
        raise ValueError("embedding dimension must be positive")
    if v.size(2) <= 0:
        raise ValueError("model dimension must be positive")
    return v


def validate_probability_list(
    v: list[list[torch.Tensor]] | None, name: str
) -> list[list[torch.Tensor]] | None:
    """Validates nested probability tensor lists have correct structure and properties."""
    if v is None:
        return None

    if not isinstance(v, list) or not all(isinstance(x, list) for x in v):
        raise ValueError(f"{name} must be a list of lists of tensors")

    # Validate each probability tensor
    batch_size = len(v)
    for batch_idx, sequence in enumerate(v):
        for step_idx, prob_tensor in enumerate(sequence):
            if not isinstance(prob_tensor, torch.Tensor):
                raise ValueError(f"{name}[{batch_idx}][{step_idx}] must be a tensor")
            if prob_tensor.dim() != 1:
                raise ValueError(
                    f"{name}[{batch_idx}][{step_idx}] must be 1-dimensional"
                )
            if name == "orth_probs" and not torch.isclose(
                prob_tensor.sum(), torch.tensor(1.0), atol=1e-5
            ):
                raise ValueError(
                    f"{name}[{batch_idx}][{step_idx}] probabilities must sum to 1, got {prob_tensor.sum()}"
                )
            # Probabilities should be between 0 and 1
            if torch.any(prob_tensor < 0) or torch.any(prob_tensor > 1):
                raise ValueError(
                    f"{name}[{batch_idx}][{step_idx}] probabilities must be between 0 and 1"
                )

    return v


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


def validate_phonological_vectors(
    v: list[list[torch.Tensor]] | None, name: str
) -> list[list[torch.Tensor]] | None:
    """Validates phonological vector lists have correct structure and properties."""
    if v is None:
        return None

    if not isinstance(v, list) or not all(isinstance(x, list) for x in v):
        raise ValueError(f"{name} must be a list of lists of tensors")

    batch_size = len(v)
    for batch_idx, sequence in enumerate(v):
        for step_idx, vector in enumerate(sequence):
            if not isinstance(vector, torch.Tensor):
                raise ValueError(f"{name}[{batch_idx}][{step_idx}] must be a tensor")
            if vector.dim() != 1:
                raise ValueError(
                    f"{name}[{batch_idx}][{step_idx}] must be 1-dimensional"
                )

            # For phon_vecs specifically, validate binary values
            if name == "phon_vecs":
                if not torch.all((vector == 0) | (vector == 1)):
                    raise ValueError(
                        f"phon_vecs[{batch_idx}][{step_idx}] must contain only binary values"
                    )

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

    global_encoding: Annotated[torch.Tensor, Field(validate_default=True)]
    orth_probs: list[list[torch.Tensor]] | None = None
    orth_tokens: torch.Tensor | None = None
    phon_probs: list[list[torch.Tensor]] | None = None
    phon_vecs: list[list[torch.Tensor]] | None = None
    phon_tokens: list[list[torch.Tensor]] | None = None

    @model_validator(mode="after")
    def validate_structure(self) -> "GenerationOutput":
        # Validate individual components
        self.global_encoding = validate_global_encoding(self.global_encoding)
        self.orth_probs = validate_probability_list(self.orth_probs, "orth_probs")
        self.orth_tokens = validate_orthographic_tokens(self.orth_tokens)
        self.phon_probs = validate_probability_list(self.phon_probs, "phon_probs")
        self.phon_vecs = validate_phonological_vectors(self.phon_vecs, "phon_vecs")
        self.phon_tokens = validate_phonological_vectors(
            self.phon_tokens, "phon_tokens"
        )

        # Cross-component validation
        batch_size = self.global_encoding.size(0)

        # Validate batch size consistency for orthographic components
        if self.orth_probs is not None:
            if len(self.orth_probs) != batch_size:
                raise ValueError("orth_probs batch size mismatch")
        if self.orth_tokens is not None:
            if self.orth_tokens.size(0) != batch_size:
                raise ValueError("orth_tokens batch size mismatch")

        # Validate batch size consistency for phonological components
        for field in [self.phon_probs, self.phon_vecs, self.phon_tokens]:
            if field is not None:
                if len(field) != batch_size:
                    raise ValueError(f"{field} batch size mismatch")

        # Validate orthographic component consistency
        if (self.orth_probs is None) != (self.orth_tokens is None):
            raise ValueError(
                "orth_probs and orth_tokens must either both be present or both be None"
            )

        # Validate phonological component consistency
        phon_fields = [self.phon_probs, self.phon_vecs, self.phon_tokens]
        if any(f is not None for f in phon_fields) and not all(
            f is not None for f in phon_fields
        ):
            raise ValueError(
                "All phonological components (probs, vecs, tokens) must be present if any are"
            )

        # Validate pathway consistency
        has_orth = self.orth_tokens is not None
        has_phon = self.phon_tokens is not None
        if not (has_orth or has_phon):
            raise ValueError(
                "At least one modality (orthographic or phonological) must be present"
            )

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

    class Config:
        arbitrary_types_allowed = True
