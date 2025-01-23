from typing import Annotated
from pydantic import BaseModel, Field, model_validator
import torch


def validate_tensor(v: torch.Tensor, name: str) -> torch.Tensor:
    """Validates a tensor exists and has correct basic properties."""
    if not isinstance(v, torch.Tensor):
        raise ValueError(f"{name} must be a torch.Tensor")
    if v.dim() != 2:
        raise ValueError(f"{name} must be 2-dimensional (batch × sequence)")
    return v


def validate_bool_tensor(v: torch.Tensor, name: str) -> torch.Tensor:
    """Additional validation for boolean mask tensors."""
    v = validate_tensor(v, name)
    if v.dtype != torch.bool:
        raise ValueError(f"{name} must have dtype torch.bool")
    return v


def validate_id_tensor(v: torch.Tensor, name: str) -> torch.Tensor:
    """Additional validation for token ID tensors."""
    v = validate_tensor(v, name)
    if v.dtype not in [torch.long, torch.int]:
        raise ValueError(f"{name} must have dtype torch.long or torch.int")
    if torch.any(v < 0):
        raise ValueError(f"{name} cannot contain negative indices")
    return v


class OrthographicEncoding(BaseModel):
    """Encodes orthographic (character-level) inputs and masks."""

    enc_input_ids: Annotated[torch.Tensor, Field(validate_default=True)]
    enc_pad_mask: Annotated[torch.Tensor, Field(validate_default=True)]
    dec_input_ids: Annotated[torch.Tensor, Field(validate_default=True)]
    dec_pad_mask: Annotated[torch.Tensor, Field(validate_default=True)]

    @model_validator(mode="after")
    def validate_tensors(self) -> "OrthographicEncoding":
        # Validate individual tensor properties
        self.enc_input_ids = validate_id_tensor(self.enc_input_ids, "enc_input_ids")
        self.dec_input_ids = validate_id_tensor(self.dec_input_ids, "dec_input_ids")
        self.enc_pad_mask = validate_bool_tensor(self.enc_pad_mask, "enc_pad_mask")
        self.dec_pad_mask = validate_bool_tensor(self.dec_pad_mask, "dec_pad_mask")

        # Validate shape consistency
        if self.enc_input_ids.shape != self.enc_pad_mask.shape:
            raise ValueError("enc_input_ids and enc_pad_mask must have same shape")
        if self.dec_input_ids.shape != self.dec_pad_mask.shape:
            raise ValueError("dec_input_ids and dec_pad_mask must have same shape")

        # Validate device consistency
        devices = {
            t.device
            for t in [
                self.enc_input_ids,
                self.enc_pad_mask,
                self.dec_input_ids,
                self.dec_pad_mask,
            ]
        }
        if len(devices) > 1:
            raise ValueError("All tensors must be on the same device")

        return self

    class Config:
        arbitrary_types_allowed = True


class PhonologicalEncoding(BaseModel):
    """Encodes phonological (feature-level) inputs and masks."""

    enc_input_ids: list[list[torch.Tensor]]
    enc_pad_mask: Annotated[torch.Tensor, Field(validate_default=True)]
    dec_input_ids: list[list[torch.Tensor]]
    dec_pad_mask: Annotated[torch.Tensor, Field(validate_default=True)]
    targets: Annotated[torch.Tensor, Field(validate_default=True)]

    @model_validator(mode="after")
    def validate_structure(self) -> "PhonologicalEncoding":
        # Validate masks
        self.enc_pad_mask = validate_bool_tensor(self.enc_pad_mask, "enc_pad_mask")
        self.dec_pad_mask = validate_bool_tensor(self.dec_pad_mask, "dec_pad_mask")

        # Validate targets tensor
        if not isinstance(self.targets, torch.Tensor):
            raise ValueError("targets must be a torch.Tensor")
        if self.targets.dim() != 3:  # batch × sequence × features
            raise ValueError("targets must be 3-dimensional")

        # Validate input structure
        if not isinstance(self.enc_input_ids, list) or not all(
            isinstance(x, list) for x in self.enc_input_ids
        ):
            raise ValueError("enc_input_ids must be a list of lists of tensors")
        if not isinstance(self.dec_input_ids, list) or not all(
            isinstance(x, list) for x in self.dec_input_ids
        ):
            raise ValueError("dec_input_ids must be a list of lists of tensors")

        # Validate all inner elements are tensors
        for batch in self.enc_input_ids + self.dec_input_ids:
            if not all(isinstance(t, torch.Tensor) for t in batch):
                raise ValueError("All elements must be torch.Tensor")

        # Validate batch sizes match
        batch_size = len(self.enc_input_ids)
        if (
            len(self.dec_input_ids) != batch_size
            or self.enc_pad_mask.size(0) != batch_size
            or self.dec_pad_mask.size(0) != batch_size
            or self.targets.size(0) != batch_size
        ):
            raise ValueError("Batch sizes must match across all components")

        # Validate device consistency
        devices = {
            self.enc_pad_mask.device,
            self.dec_pad_mask.device,
            self.targets.device,
        }
        for batch in self.enc_input_ids + self.dec_input_ids:
            for tensor in batch:
                devices.add(tensor.device)
        if len(devices) > 1:
            raise ValueError("All tensors must be on the same device")

        return self

    class Config:
        arbitrary_types_allowed = True


class BridgeEncoding(BaseModel):
    """Container for both orthographic and phonological encodings."""

    orthographic: OrthographicEncoding
    phonological: PhonologicalEncoding

    @model_validator(mode="after")
    def validate_consistency(self) -> "BridgeEncoding":
        # Validate batch sizes match between orthographic and phonological
        orth_batch_size = self.orthographic.enc_input_ids.size(0)
        phon_batch_size = len(self.phonological.enc_input_ids)

        if orth_batch_size != phon_batch_size:
            raise ValueError(
                f"Batch size mismatch: orthographic ({orth_batch_size}) != "
                f"phonological ({phon_batch_size})"
            )

        # Validate devices match
        orth_device = self.orthographic.enc_input_ids.device
        phon_device = (
            self.phonological.enc_pad_mask.device
        )  # Use any tensor from phonological

        if orth_device != phon_device:
            raise ValueError(
                f"Device mismatch: orthographic ({orth_device}) != "
                f"phonological ({phon_device})"
            )

        return self

    class Config:
        arbitrary_types_allowed = True
