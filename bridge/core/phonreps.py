"""The phonological feature scheme: which phonetic features each phoneme carries.

`bridge/core/phonreps.csv` maps each phoneme to a binary feature vector. The names are
ARPAbet's, and the Spanish lexicon's own for the seven sounds ARPAbet lacks. Both the
tokenizer (to encode words) and the model (to embed phonemes, and to score closest-phoneme
metrics) need it, and neither owns it. It is configuration of the feature scheme, not
state of any tokenizer.
"""

from __future__ import annotations

import functools
import hashlib
import os
from dataclasses import dataclass

import pandas as pd
import torch

from bridge.utils import get_project_root

_PHONREPS_CSV_RELATIVE = "bridge/core/phonreps.csv"

# Order is part of the row-id contract. See PhonemeTable.
SPECIAL_TOKENS: tuple[str, ...] = ("[BOS]", "[EOS]", "[UNK]", "[SPC]", "[PAD]")

# The special tokens hold rows 86 to 90, after the first 86 rows of phonreps.csv and before
# every later one, so a phoneme appended to the csv takes a new row id and moves none.
# Row ids are held outside this module: in batches a caller keeps, and as literals in tests.
_FIRST_SPECIAL_ROW = 86

# `[PAD]` occupies the final feature column, and three places rely on it: the tokenizer
# drops that column to build loss targets, the decoder head is sized to match, and
# `phono_sample` widens a sampled vector by one column to reach it.
assert SPECIAL_TOKENS[-1] == "[PAD]", "[PAD] must remain the last special token"


def row_normalize(multihot: torch.Tensor) -> torch.Tensor:
    """Scale each row of a non-negative tensor to sum to 1, so ``x @ W`` averages.

    An all-zero row stays zero instead of becoming NaN, which both callers need:
    ``phonreps.csv`` contains one featureless phoneme (``'_'``), and generation can sample
    an all-off feature vector. Only that case is special-cased, so a row of fractional
    weights still normalizes rather than being left scaled down.
    """
    totals = multihot.sum(-1, keepdim=True)
    return multihot / totals.where(totals > 0, torch.ones_like(totals))


@dataclass(frozen=True, eq=False)
class PhonemeTable:
    """Every phoneme as a multi-hot row over the phonological feature vocabulary.

    BRIDGE uses two distinct integer spaces for phonology, and confusing them is the
    easiest way to break this code:

    * **row space** (``0 .. num_rows-1``): *which phoneme*. Model inputs index this.
    * **feature space** (``0 .. vocab_size-1``): *which phonetic feature*. Model outputs,
      sampling, and ``VocabSpec.phon_*_id`` live here.

    ``multihot[r, f]`` is 1 when phoneme ``r`` has feature ``f``, the bridge between them.

    Row order is the contract that lets the tokenizer and the model agree without talking
    to each other: the first 86 rows of ``phonreps.csv``, then :data:`SPECIAL_TOKENS`, then
    the rest of the csv. Defined here and nowhere else.

    A phoneme's embedding is the mean of its active features' embeddings, which is linear
    in the embedding weight, so ``row_normalize(multihot) @ W`` yields *every* phoneme's
    embedding in one small matrix multiply, and embedding a batch becomes a table lookup.
    See ``docs/decisions/0002-phoneme-row-ids-replace-ragged-feature-lists.md``.
    """

    multihot: torch.Tensor
    row_index: dict[str, int]
    base_dim: int

    @property
    def num_rows(self) -> int:
        """Number of phonemes, including special tokens (row-space size)."""
        return self.multihot.shape[0]

    @property
    def vocab_size(self) -> int:
        """Number of phonological features, including special tokens (feature-space size)."""
        return self.multihot.shape[1]

    @property
    def fingerprint(self) -> str:
        """Stable digest of the feature scheme, for checkpoint provenance.

        Covers both the matrix and the row order, so reordering rows changes it even if
        the set of phonemes does not.
        """
        digest = hashlib.sha256()
        digest.update(",".join(self.row_index).encode())
        digest.update(self.multihot.detach().to("cpu", torch.uint8).numpy().tobytes())
        return digest.hexdigest()[:16]

    @functools.cached_property
    def phonetic_features(self) -> torch.Tensor:
        """The real phonemes' feature vectors, without the special-token rows or columns.

        The ``(phonemes, base_dim)`` block that closest-phoneme distance metrics compare
        against. Row ``i`` is ``phonemes[i]``, which is not table row ``i`` past the special
        tokens. Shared between callers, so treat it as read-only.
        """
        rows = [self.row_index[phoneme] for phoneme in self.phonemes]
        return self.multihot[rows, : self.base_dim]

    @property
    def phonemes(self) -> list[str]:
        """The real phonemes, in table order, excluding special tokens."""
        return [name for name in self.row_index if name not in SPECIAL_TOKENS]

    def row_of(self, phoneme: str) -> int:
        """Row id for ``phoneme``, falling back to ``[UNK]`` for unknown phonemes."""
        return self.row_index.get(phoneme, self.row_index["[UNK]"])

    def feature_of(self, special_token: str) -> int:
        """Feature-space id of a special token: its dedicated column after ``base_dim``."""
        return self.base_dim + SPECIAL_TOKENS.index(special_token)

    def features_of(self, row: int) -> torch.Tensor:
        """Active feature indices of one row, the inverse of the multi-hot encoding."""
        return torch.nonzero(self.multihot[row]).flatten()


def load_phoneme_table(device: torch.device | str = "cpu") -> PhonemeTable:
    """Build the phoneme -> feature multi-hot table from ``phonreps.csv``.

    Special tokens occupy one dedicated feature each, appended after the phonetic
    features, matching ``PhonemeTokenizer.special_token_dims``.

    Cached per device: the CSV is immutable configuration and every tokenizer and model
    wants the same table. The returned table is therefore shared, so treat it as read-only.
    The key is normalized first, or `"cpu"` and `torch.device("cpu")` would each build one.
    """
    return _load_phoneme_table(torch.device(device))


@functools.cache
def _load_phoneme_table(device: torch.device) -> PhonemeTable:
    dataframe = pd.read_csv(os.path.join(get_project_root(), _PHONREPS_CSV_RELATIVE))
    dataframe.set_index("phone", inplace=True)
    features = torch.tensor(dataframe.values, dtype=torch.float, device=device)

    phonemes = list(dataframe.index)
    names = [
        *phonemes[:_FIRST_SPECIAL_ROW],
        *SPECIAL_TOKENS,
        *phonemes[_FIRST_SPECIAL_ROW:],
    ]
    row_index = {name: row for row, name in enumerate(names)}

    base_dim = features.shape[1]
    multihot = torch.zeros(
        (len(names), base_dim + len(SPECIAL_TOKENS)), dtype=torch.float, device=device
    )
    multihot[[row_index[phoneme] for phoneme in phonemes], :base_dim] = features
    for offset, token in enumerate(SPECIAL_TOKENS):
        multihot[row_index[token], base_dim + offset] = 1.0

    return PhonemeTable(multihot=multihot, row_index=row_index, base_dim=base_dim)
