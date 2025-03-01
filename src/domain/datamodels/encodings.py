"""
BridgeEncoding: A high-performance data structure for managing orthographic and phonological encodings.
Uses slots and frozen dataclasses for optimal memory usage and access speed.
"""

from dataclasses import dataclass, field
from typing import Any, Union, Optional, List
import torch
import functools


@dataclass(frozen=True, slots=True)
class EncodingComponent:
    """A component of a BridgeEncoding, representing either orthographic or phonological data."""
    enc_input_ids: Any  # Tensor for orth, list of lists of tensors for phon
    enc_pad_mask: torch.Tensor
    dec_input_ids: Any  # Tensor for orth, list of lists of tensors for phon
    dec_pad_mask: torch.Tensor
    targets: Optional[torch.Tensor] = None  # Only used for phonological data


@dataclass(frozen=True, slots=True)
class BridgeEncoding:
    """
    Unified container for orthographic and phonological encodings.

    This class is immutable (frozen) and uses slots for better memory usage
    and faster attribute access. All tensor operations maintain device consistency.

    Attributes:
        orthographic: EncodingComponent containing orthographic encodings
        phonological: Optional[EncodingComponent] containing phonological encodings
        device: torch.device - Device all tensors reside on
    """
    orthographic: EncodingComponent
    phonological: Optional[EncodingComponent] = None
    device: torch.device = field(default=torch.device("cpu"))

    # Legacy property accessors for backwards compatibility
    @property
    def orth_enc_ids(self) -> torch.Tensor:
        return self.orthographic.enc_input_ids

    @property
    def orth_enc_mask(self) -> torch.Tensor:
        return self.orthographic.enc_pad_mask

    @property
    def orth_dec_ids(self) -> torch.Tensor:
        return self.orthographic.dec_input_ids

    @property
    def orth_dec_mask(self) -> torch.Tensor:
        return self.orthographic.dec_pad_mask

    @property
    def phon_enc_ids(self) -> List[List[torch.Tensor]]:
        if self.phonological is None:
            raise AttributeError("Phonological component is not available")
        return self.phonological.enc_input_ids

    @property
    def phon_enc_mask(self) -> torch.Tensor:
        if self.phonological is None:
            raise AttributeError("Phonological component is not available")
        return self.phonological.enc_pad_mask

    @property
    def phon_dec_ids(self) -> List[List[torch.Tensor]]:
        if self.phonological is None:
            raise AttributeError("Phonological component is not available")
        return self.phonological.dec_input_ids

    @property
    def phon_dec_mask(self) -> torch.Tensor:
        if self.phonological is None:
            raise AttributeError("Phonological component is not available")
        return self.phonological.dec_pad_mask

    @property
    def phon_targets(self) -> torch.Tensor:
        if self.phonological is None or self.phonological.targets is None:
            raise AttributeError("Phonological targets are not available")
        return self.phonological.targets

    def __post_init__(self):
        """Validate components and set device."""
        # Use object.__setattr__ since the class is frozen
        if self.orthographic is not None and hasattr(self.orthographic.enc_input_ids, 'device'):
            device = self.orthographic.enc_input_ids.device
            object.__setattr__(self, "device", device)
        
        if self.orthographic is not None and self.phonological is not None:
            self._validate_full_encoding()
        elif self.orthographic is not None:
            self._validate_orthographic_only()
        elif self.phonological is not None:
            self._validate_phonological_only()
        else:
            raise ValueError("At least one encoding component must be provided")

    def _validate_orthographic_only(self):
        """Validate orthographic encoding component."""
        # Validate orthographic tensors are valid
        self._validate_orthographic_component(self.orthographic)

    def _validate_phonological_only(self):
        """Validate phonological encoding component."""
        # Validate phonological tensors are valid
        self._validate_phonological_component(self.phonological)

    def _validate_full_encoding(self):
        """Validate both encoding components and ensure consistency."""
        # Validate individual components
        self._validate_orthographic_component(self.orthographic)
        self._validate_phonological_component(self.phonological)
        
        # Validate batch sizes match
        orth_batch_size = self.orthographic.enc_input_ids.size(0)
        phon_batch_size = len(self.phonological.enc_input_ids)
        
        if orth_batch_size != phon_batch_size:
            raise ValueError(
                f"Batch size mismatch: orthographic component has {orth_batch_size} samples, "
                f"phonological component has {phon_batch_size} samples"
            )
        
        # Validate devices match
        if self.orthographic.enc_input_ids.device != self.device:
            raise ValueError(
                f"Device mismatch: orthographic component on {self.orthographic.enc_input_ids.device}, "
                f"expected {self.device}"
            )
        
        # Check phonological device consistency
        for batch in self.phonological.enc_input_ids:
            for tensor in batch:
                if tensor.device != self.device:
                    raise ValueError(
                        f"Device mismatch: phonological tensor on {tensor.device}, "
                        f"expected {self.device}"
                    )

    def _validate_orthographic_component(self, component: EncodingComponent):
        """Validate orthographic component tensors."""
        # Validate orthographic tensors
        for name, tensor in [
            ("enc_input_ids", component.enc_input_ids),
            ("dec_input_ids", component.dec_input_ids),
        ]:
            if not isinstance(tensor, torch.Tensor):
                raise ValueError(f"Orthographic {name} must be a torch.Tensor")
            if tensor.dim() != 2:
                raise ValueError(f"Orthographic {name} must be 2-dimensional (batch × sequence)")
            if tensor.dtype not in [torch.long, torch.int]:
                raise ValueError(f"Orthographic {name} must have dtype torch.long or torch.int")
            if torch.any(tensor < 0):
                raise ValueError(f"Orthographic {name} cannot contain negative indices")

        # Validate padding masks
        for name, tensor in [
            ("enc_pad_mask", component.enc_pad_mask),
            ("dec_pad_mask", component.dec_pad_mask),
        ]:
            if not isinstance(tensor, torch.Tensor):
                raise ValueError(f"Orthographic {name} must be a torch.Tensor")
            if tensor.dim() != 2:
                raise ValueError(f"Orthographic {name} must be 2-dimensional")
            if tensor.dtype != torch.bool:
                raise ValueError(f"Orthographic {name} must have dtype torch.bool")
                
        # Validate batch consistency
        batch_size = component.enc_input_ids.size(0)
        tensors_to_check = [
            ("enc_pad_mask", component.enc_pad_mask),
            ("dec_input_ids", component.dec_input_ids),
            ("dec_pad_mask", component.dec_pad_mask),
        ]
        
        for name, tensor in tensors_to_check:
            if tensor.size(0) != batch_size:
                raise ValueError(
                    f"Batch size mismatch: orthographic {name} has size {tensor.size(0)}, "
                    f"expected {batch_size}"
                )

    def _validate_phonological_component(self, component: EncodingComponent):
        """Validate phonological component tensors."""
        # Validate phonological feature tensors
        for name, tensor_list in [
            ("enc_input_ids", component.enc_input_ids),
            ("dec_input_ids", component.dec_input_ids),
        ]:
            if not isinstance(tensor_list, list):
                raise ValueError(f"Phonological {name} must be a list of lists of tensors")
            if not all(isinstance(batch, list) for batch in tensor_list):
                raise ValueError(f"Each batch in phonological {name} must be a list")
            if not all(
                isinstance(t, torch.Tensor) for batch in tensor_list for t in batch
            ):
                raise ValueError(
                    f"All elements in phonological {name} must be torch.Tensor"
                )

        # Validate padding masks
        for name, tensor in [
            ("enc_pad_mask", component.enc_pad_mask),
            ("dec_pad_mask", component.dec_pad_mask),
        ]:
            if not isinstance(tensor, torch.Tensor):
                raise ValueError(f"Phonological {name} must be a torch.Tensor")
            if tensor.dim() != 2:
                raise ValueError(f"Phonological {name} must be 2-dimensional")
            if tensor.dtype != torch.bool:
                raise ValueError(f"Phonological {name} must have dtype torch.bool")

        # Validate targets tensor if present
        if component.targets is not None:
            if not isinstance(component.targets, torch.Tensor):
                raise ValueError("Phonological targets must be a torch.Tensor")
            if component.targets.dim() != 3:
                raise ValueError(
                    "Phonological targets must be 3-dimensional (batch × sequence × features)"
                )
                
        # Validate batch consistency
        batch_size = len(component.enc_input_ids)
        tensors_to_check = [
            ("enc_pad_mask", component.enc_pad_mask),
            ("dec_pad_mask", component.dec_pad_mask),
        ]
        
        for name, tensor in tensors_to_check:
            if tensor.size(0) != batch_size:
                raise ValueError(
                    f"Batch size mismatch: phonological {name} has size {tensor.size(0)}, "
                    f"expected {batch_size}"
                )
                
        if len(component.dec_input_ids) != batch_size:
            raise ValueError(
                f"Batch size mismatch: phonological dec_input_ids has {len(component.dec_input_ids)} batches, "
                f"expected {batch_size}"
            )
            
        if component.targets is not None and component.targets.size(0) != batch_size:
            raise ValueError(
                f"Batch size mismatch: phonological targets has size {component.targets.size(0)}, "
                f"expected {batch_size}"
            )

    @classmethod
    def from_dict(
        cls, data: dict[str, Any], device: Optional[torch.device] = None
    ) -> "BridgeEncoding":
        """
        Create a BridgeEncoding instance from a dictionary representation.

        Args:
            data: Dictionary containing orthographic and/or phonological components
            device: Optional device to place tensors on

        Returns:
            BridgeEncoding instance
        """
        if device is None:
            device = torch.device("cpu")

        # Initialize components
        orthographic = None
        phonological = None
        
        # Process orthographic data if present
        if "orthographic" in data:
            orth_data = data["orthographic"]
            orthographic = EncodingComponent(
                enc_input_ids=orth_data["enc_input_ids"].to(device),
                enc_pad_mask=orth_data["enc_pad_mask"].to(device),
                dec_input_ids=orth_data["dec_input_ids"].to(device),
                dec_pad_mask=orth_data["dec_pad_mask"].to(device)
            )
            
        # Process phonological data if present
        if "phonological" in data:
            phon_data = data["phonological"]
            phonological = EncodingComponent(
                enc_input_ids=[
                    [t.to(device) for t in batch] for batch in phon_data["enc_input_ids"]
                ],
                enc_pad_mask=phon_data["enc_pad_mask"].to(device),
                dec_input_ids=[
                    [t.to(device) for t in batch] for batch in phon_data["dec_input_ids"]
                ],
                dec_pad_mask=phon_data["dec_pad_mask"].to(device),
                targets=phon_data.get("targets", None)
            )
            
            if "targets" in phon_data:
                phonological.targets = phon_data["targets"].to(device)

        return cls(
            orthographic=orthographic,
            phonological=phonological,
            device=device,
        )

    def to_dict(self) -> dict[str, Any]:
        """
        Convert to dictionary format for compatibility with existing code.

        Returns:
            Dictionary containing orthographic and/or phonological components
        """
        result = {}
        
        if self.orthographic is not None:
            result["orthographic"] = {
                "enc_input_ids": self.orthographic.enc_input_ids,
                "enc_pad_mask": self.orthographic.enc_pad_mask,
                "dec_input_ids": self.orthographic.dec_input_ids,
                "dec_pad_mask": self.orthographic.dec_pad_mask,
            }
            
        if self.phonological is not None:
            result["phonological"] = {
                "enc_input_ids": self.phonological.enc_input_ids,
                "enc_pad_mask": self.phonological.enc_pad_mask,
                "dec_input_ids": self.phonological.dec_input_ids,
                "dec_pad_mask": self.phonological.dec_pad_mask,
            }
            
            if self.phonological.targets is not None:
                result["phonological"]["targets"] = self.phonological.targets
                
        return result

    def to(self, device: torch.device) -> "BridgeEncoding":
        """
        Move all tensors to the specified device.

        Args:
            device: Target device

        Returns:
            New BridgeEncoding instance with all tensors on target device
        """
        orthographic = None
        phonological = None
        
        if self.orthographic is not None:
            orthographic = EncodingComponent(
                enc_input_ids=self.orthographic.enc_input_ids.to(device),
                enc_pad_mask=self.orthographic.enc_pad_mask.to(device),
                dec_input_ids=self.orthographic.dec_input_ids.to(device),
                dec_pad_mask=self.orthographic.dec_pad_mask.to(device)
            )
            
        if self.phonological is not None:
            phonological = EncodingComponent(
                enc_input_ids=[[t.to(device) for t in batch] for batch in self.phonological.enc_input_ids],
                enc_pad_mask=self.phonological.enc_pad_mask.to(device),
                dec_input_ids=[[t.to(device) for t in batch] for batch in self.phonological.dec_input_ids],
                dec_pad_mask=self.phonological.dec_pad_mask.to(device),
                targets=self.phonological.targets.to(device) if self.phonological.targets is not None else None
            )
            
        return BridgeEncoding(
            orthographic=orthographic,
            phonological=phonological,
            device=device,
        )

    def __getitem__(self, idx: Union[int, slice]) -> dict[str, Any]:
        """
        Get a batch slice of the encoding.

        Args:
            idx: Integer index or slice

        Returns:
            Dictionary containing orthographic and/or phonological components for the slice
        """
        result = {}
        
        if self.orthographic is not None:
            if isinstance(idx, int):
                result["orthographic"] = {
                    "enc_input_ids": self.orthographic.enc_input_ids[idx : idx + 1],
                    "enc_pad_mask": self.orthographic.enc_pad_mask[idx : idx + 1],
                    "dec_input_ids": self.orthographic.dec_input_ids[idx : idx + 1],
                    "dec_pad_mask": self.orthographic.dec_pad_mask[idx : idx + 1],
                }
            elif isinstance(idx, slice):
                result["orthographic"] = {
                    "enc_input_ids": self.orthographic.enc_input_ids[idx],
                    "enc_pad_mask": self.orthographic.enc_pad_mask[idx],
                    "dec_input_ids": self.orthographic.dec_input_ids[idx],
                    "dec_pad_mask": self.orthographic.dec_pad_mask[idx],
                }
            else:
                raise TypeError("Index must be int or slice")
                
        if self.phonological is not None:
            if isinstance(idx, int):
                result["phonological"] = {
                    "enc_input_ids": self.phonological.enc_input_ids[idx : idx + 1],
                    "enc_pad_mask": self.phonological.enc_pad_mask[idx : idx + 1],
                    "dec_input_ids": self.phonological.dec_input_ids[idx : idx + 1],
                    "dec_pad_mask": self.phonological.dec_pad_mask[idx : idx + 1],
                }
                if self.phonological.targets is not None:
                    result["phonological"]["targets"] = self.phonological.targets[idx : idx + 1]
            elif isinstance(idx, slice):
                result["phonological"] = {
                    "enc_input_ids": self.phonological.enc_input_ids[idx],
                    "enc_pad_mask": self.phonological.enc_pad_mask[idx],
                    "dec_input_ids": self.phonological.dec_input_ids[idx],
                    "dec_pad_mask": self.phonological.dec_pad_mask[idx],
                }
                if self.phonological.targets is not None:
                    result["phonological"]["targets"] = self.phonological.targets[idx]
            else:
                raise TypeError("Index must be int or slice")
                
        return result

    def __len__(self) -> int:
        """Return the batch size."""
        if self.orthographic is not None:
            return self.orthographic.enc_input_ids.size(0)
        elif self.phonological is not None:
            return len(self.phonological.enc_input_ids)
        else:
            return 0
