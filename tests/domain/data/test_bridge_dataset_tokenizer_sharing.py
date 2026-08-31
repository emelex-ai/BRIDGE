"""Paired datasets must be able to share one tokenizer (issue #221, related finding 6).

Constructing a ``BridgeTokenizer`` parses the pronunciation lexicons: ~1 s and ~81 MB
retained. ``TrainingPipeline`` builds a second ``BridgeDataset`` for the test split, so
without injection that cost, and the memory the periodic ``gc.collect()`` then has to
walk, was paid twice.
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

    assert private.words[:8] == shared.words[:8]
    a, b = private[slice(0, 8)], shared[slice(0, 8)]

    assert torch.equal(a.orthographic.enc_input_ids, b.orthographic.enc_input_ids)
    assert torch.equal(a.orthographic.dec_input_ids, b.orthographic.dec_input_ids)
    assert torch.equal(a.phonological.enc_pad_mask, b.phonological.enc_pad_mask)
    assert torch.equal(a.phonological.phon_targets, b.phonological.phon_targets)
