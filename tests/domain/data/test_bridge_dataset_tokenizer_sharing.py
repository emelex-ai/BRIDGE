"""Paired datasets must be able to share one tokenizer (issue #221, related finding 6).

Constructing a ``BridgeTokenizer`` parses the pronunciation lexicons: ~1 s and ~81 MB
retained. A run usually holds a second ``BridgeDataset`` for its test split, so without
injection that cost is paid twice.
"""

import torch

from bridge.domain.data import BridgeDataset
from bridge.domain.datamodels import DatasetConfig
from bridge.domain.tokenizer.bridge_tokenizer import BridgeTokenizer

DATA = "tests/domain/model/data/data.csv"


def test_injected_tokenizer_is_reused_not_rebuilt():
    tokenizer = BridgeTokenizer()
    dataset = BridgeDataset(DatasetConfig(dataset_filepath=DATA), tokenizer=tokenizer)
    assert dataset.tokenizer is tokenizer


def test_injection_does_not_change_encodings():
    """A shared tokenizer must produce byte-identical output to a private one."""
    config = DatasetConfig(dataset_filepath=DATA)
    private = BridgeDataset(config)
    shared = BridgeDataset(config, tokenizer=BridgeTokenizer())

    assert private.examples[:8] == shared.examples[:8]
    a, b = private[slice(0, 8)], shared[slice(0, 8)]

    assert torch.equal(a.input.orth.ids, b.input.orth.ids)
    assert torch.equal(a.target.orth.ids, b.target.orth.ids)
    assert torch.equal(a.target.orth.labels, b.target.orth.labels)
    assert torch.equal(a.input.phon.pad_mask, b.input.phon.pad_mask)
    assert torch.equal(a.target.phon.labels, b.target.phon.labels)
