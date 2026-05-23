"""Shared loader for the phonological feature representation table.

The phonreps CSV (`bridge/core/phonreps.csv`) maps each ARPAbet phoneme to a
binary feature vector. Multiple parts of the codebase need access to this:

* `PhonemeTokenizer` uses it to build the feature-vector encoding.
* `TrainingPipeline` / `phon_metrics` use it for closest-phoneme distance
  computations during evaluation.

Previously the tensor was owned by `PhonemeTokenizer` and other consumers
reached for it via `dataset.tokenizer.phoneme_tokenizer.phonreps_array` — a
three-level Demeter violation that also coupled training metrics to the
existence of a tokenizer instance. This module breaks that coupling: phonreps
is configuration-of-the-feature-scheme, not state-of-any-tokenizer.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import pandas as pd
import torch

from bridge.utils import get_project_root

_PHONREPS_CSV_RELATIVE = "bridge/core/phonreps.csv"


@dataclass(frozen=True)
class PhonReps:
    """Materialized view of the phonological feature table.

    Frozen so consumers can stash a reference without worrying about mutation.
    """

    dataframe: pd.DataFrame
    array: torch.Tensor
    index: dict[str, int]

    @property
    def base_dim(self) -> int:
        """Number of phonological features (columns of the feature table)."""
        return len(self.dataframe.columns)


def load_phonreps(device: torch.device | str = "cpu") -> PhonReps:
    """Load phonreps.csv into a `PhonReps` view.

    Args:
        device: where to place the feature tensor (CPU by default; pass a CUDA
            device for direct on-device metric computation).
    """
    csv_path = os.path.join(get_project_root(), _PHONREPS_CSV_RELATIVE)
    dataframe = pd.read_csv(csv_path)
    dataframe.set_index("phone", inplace=True)
    array = torch.tensor(dataframe.values, dtype=torch.float, device=device)
    index = {p: i for i, p in enumerate(dataframe.index)}
    return PhonReps(dataframe=dataframe, array=array, index=index)


def load_phonreps_array(device: torch.device | str = "cpu") -> torch.Tensor:
    """Convenience: load only the feature tensor (skip building a DataFrame view).

    Use this from callers that just need the tensor for distance metrics.
    """
    return load_phonreps(device=device).array
