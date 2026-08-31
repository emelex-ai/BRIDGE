"""Phonological hot-path benchmarks (issue #221).

Measures the four things the refactor is meant to move, through the *public* API, so the
same script runs unchanged before and after:

    embed_phon_tokens   the double loop being replaced
    encode              PhonemeTokenizer.encode, incl. the per-position `targets` write
    train step          end-to-end o2p, so the isolated wins can be put in context
    generate            o2p, deterministic

The reference double-loop implementation is carried here permanently so the speedup stays
measurable after the loop is gone from the model.

    BENCH_DEV=cpu  uv run python benchmarks/bench_phon.py
    BENCH_DEV=cuda uv run python benchmarks/bench_phon.py
    BENCH_SIZES=32,256 BENCH_DEV=cpu uv run python benchmarks/bench_phon.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

torch.set_num_threads(1)

from bridge.utils.device_manager import DeviceManager, device_manager  # noqa: E402

# Resolve through DeviceManager so an unavailable device falls back to CPU rather than
# failing deep inside torch, then read back what was actually selected.
device_manager._device = DeviceManager(device=os.environ.get("BENCH_DEV", "cpu")).device
DEV = device_manager.device

from bridge.domain.data import BridgeDataset  # noqa: E402
from bridge.domain.datamodels import DatasetConfig, ModelConfig, VocabSpec  # noqa: E402
from bridge.domain.model import Model  # noqa: E402

DATA = "tests/domain/model/data/data.csv"
SIZES = [int(s) for s in os.environ.get("BENCH_SIZES", "32,256,1024").split(",")]


def sync() -> None:
    # Compare the resolved type, not the requested string: "cuda:0" is a valid request
    # that would never match "cuda", leaving every CUDA timing unsynchronized.
    if DEV.type == "cuda":
        torch.cuda.synchronize()


def bench(fn, n: int = 30, warm: int = 5) -> float:
    """Mean wall-clock ms per call."""
    for _ in range(warm):
        fn()
    sync()
    start = time.perf_counter()
    for _ in range(n):
        fn()
    sync()
    return (time.perf_counter() - start) / n * 1000


def ragged_features(ids, tokenizer) -> list[list[torch.Tensor]]:
    """``(B, L)`` phoneme row ids -> ragged per-position feature-index tensors.

    The shape the reference implementation below was written against.
    """
    table = tokenizer.phoneme_tokenizer.phoneme_table
    return [[table.features_of(int(r)) for r in row] for row in ids]


def reference_embed(model: Model, ragged) -> torch.Tensor:
    """The pre-refactor double loop, preserved as the speedup baseline."""
    batch_size = len(ragged)
    seq_len = len(ragged[0])
    out = torch.zeros((batch_size, seq_len, model.model_config.d_model), device=model.device)
    for b, row in enumerate(ragged):
        for i, feats in enumerate(row):
            out[b, i, :] = model.phonology_embedding(feats).mean(axis=0)
    return out + model.phon_position_embedding.weight[None, :seq_len]


def measure(dataset, tokenizer, model, optimizer, vocab, size: int) -> dict[str, float]:
    """Time every stage at one batch size.

    A function rather than a loop body so the closures below bind their operands as
    arguments instead of capturing loop variables.
    """
    words = dataset.words[:size]
    if len(words) < size:
        words = (dataset.words * (size // len(dataset.words) + 1))[:size]
    batch_slice = slice(0, size)

    encoding = dataset[batch_slice]
    phon = encoding.phonological
    ragged = ragged_features(phon.dec_input_ids, tokenizer)

    def train_step() -> None:
        optimizer.zero_grad()
        batch = dataset[batch_slice]
        logits = model(
            task="o2p",
            orth_enc_input=batch.orthographic.enc_input_ids,
            orth_enc_pad_mask=batch.orthographic.enc_pad_mask,
            phon_dec_input=batch.phonological.dec_input_ids,
            phon_dec_pad_mask=batch.phonological.dec_pad_mask,
        )
        loss = torch.nn.CrossEntropyLoss(ignore_index=vocab.phon_pad_id)(
            logits["phon"], batch.phonological.phon_targets
        )
        loss.backward()
        optimizer.step()

    return {
        "ref": bench(lambda: reference_embed(model, ragged), n=10, warm=2),
        "now": bench(lambda: model.embed_phon_tokens(phon.dec_input_ids)),
        "encode": bench(lambda: tokenizer.encode(words), n=10, warm=2),
        "step": bench(train_step, n=10, warm=3),
        "generate": bench(lambda: model.generate(encoding, "o2p", deterministic=True), n=5, warm=2),
    }


def main() -> None:
    dataset = BridgeDataset(DatasetConfig(dataset_filepath=DATA))
    tokenizer = dataset.tokenizer
    vocab = VocabSpec.from_tokenizer(tokenizer)
    model = Model(ModelConfig(vocab=vocab, seed=1)).to(device_manager.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    print(f"device={DEV}  d_model={model.model_config.d_model}  torch={torch.__version__}")
    print()
    print(
        f"{'B':>6} | {'embed ref':>10} {'embed now':>10} {'x':>7} | "
        f"{'encode':>9} | {'step':>9} | {'generate':>9}"
    )
    print("-" * 78)

    for size in SIZES:
        row = measure(dataset, tokenizer, model, optimizer, vocab, size)
        print(
            f"{size:>6} | {row['ref']:9.3f}m {row['now']:9.3f}m "
            f"{row['ref'] / row['now']:6.1f}x | {row['encode']:8.3f}m | "
            f"{row['step']:8.3f}m | {row['generate']:8.3f}m"
        )

    print()
    print("embed ref = pre-refactor Python double loop, kept as the fixed comparison point.")
    print("All times are ms/call, single-threaded CPU math.")


if __name__ == "__main__":
    main()
