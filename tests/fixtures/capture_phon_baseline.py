"""Record a behavioural fingerprint of the phonological path, for refactor equivalence.

Run this ONCE on the pre-refactor commit; commit the resulting ``phon_baseline.pt``.
``tests/domain/test_phon_baseline_equivalence.py`` then asserts every recorded value at
every phase of the issue #221 refactor.

    uv run python tests/fixtures/capture_phon_baseline.py

The refactor replaces the ragged ``list[list[Tensor]]`` phonological representation with a
``(B, L)`` tensor of phoneme row ids. Anything recorded here that names the *representation*
would be worthless afterwards, so ragged structures are canonicalised to
``list[list[tuple[int, ...]]]``: "which features does each position carry", sorted. That
question has the same answer before and after, which is exactly what must be proven.

Everything runs on CPU (``device_manager`` defaults to CPU) so the fixture is portable and
CI-comparable. CUDA reduction order differs; see the spec's tolerance discussion.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bridge.domain.datamodels import ModelConfig, VocabSpec  # noqa: E402
from bridge.domain.model import Model  # noqa: E402
from bridge.domain.tokenizer.bridge_tokenizer import BridgeTokenizer  # noqa: E402

FIXTURE = Path(__file__).with_name("phon_baseline.pt")

SEED = 11
D_MODEL = 32
NHEAD = 2

# Chosen to exercise: single-phoneme words, long words, a multi-word phrase (which routes
# through [SPC]), and a repeated word (so the phoneme table's dedup is covered).
WORDS = [
    "a",
    "at",
    "or",
    "long",
    "pencil",
    "computer",
    "elephant",
    "hello world",
    "at",
]


def canon_ragged(ragged) -> list[list[tuple[int, ...]]]:
    """Ragged feature tensors -> sorted index tuples.

    Representation-independent: it records *which features* each position carries, not how
    they are stored, so the assertion survives the ragged -> row-id change.
    """
    return [[tuple(sorted(int(v) for v in t.reshape(-1).tolist())) for t in row] for row in ragged]


def stack_nested(nested) -> torch.Tensor:
    """list[list[Tensor]] with uniform inner length -> a single stacked tensor."""
    return torch.stack([torch.stack(row) for row in nested])


def build() -> dict:
    tokenizer = BridgeTokenizer()
    encoding = tokenizer.encode(WORDS)
    if encoding is None:
        raise SystemExit("Encoding failed: a word is missing from the pronunciation lexicon.")

    vocab = VocabSpec.from_tokenizer(tokenizer)
    model = Model(ModelConfig(vocab=vocab, d_model=D_MODEL, nhead=NHEAD, seed=SEED))
    model.eval()

    orth, phon = encoding.orthographic, encoding.phonological
    out: dict = {
        "words": WORDS,
        "seed": SEED,
        "d_model": D_MODEL,
        "nhead": NHEAD,
        "vocab": vocab.model_dump(),
    }

    # ---- tokenizer -------------------------------------------------------------
    out["tok"] = {
        "orth_enc_input_ids": orth.enc_input_ids.clone(),
        "orth_dec_input_ids": orth.dec_input_ids.clone(),
        "orth_enc_pad_mask": orth.enc_pad_mask.clone(),
        "orth_dec_pad_mask": orth.dec_pad_mask.clone(),
        "phon_enc_pad_mask": phon.enc_pad_mask.clone(),
        "phon_dec_pad_mask": phon.dec_pad_mask.clone(),
        "phon_targets": phon.phon_targets.clone(),
        "phon_enc_features": canon_ragged(phon.enc_input_ids),
        "phon_dec_features": canon_ragged(phon.dec_input_ids),
    }

    # The orthography-only path builds a placeholder phonological component; the refactor
    # rewrites that builder, so pin it too.
    placeholder = tokenizer.encode(WORDS, modality_filter="orthography")
    assert placeholder is not None
    ph = placeholder.phonological
    out["placeholder_phon"] = {
        "enc_features": canon_ragged(ph.enc_input_ids),
        "dec_features": canon_ragged(ph.dec_input_ids),
        "enc_pad_mask": ph.enc_pad_mask.clone(),
        "dec_pad_mask": ph.dec_pad_mask.clone(),
        "targets": ph.phon_targets.clone(),
    }

    # ---- embedding + gradients -------------------------------------------------
    embed_out = model.embed_phon_tokens(phon.dec_input_ids)
    out["embed"] = {"dec": embed_out.detach().clone()}
    out["embed"]["enc"] = model.embed_phon_tokens(phon.enc_input_ids).detach().clone()

    model.zero_grad(set_to_none=True)
    model.embed_phon_tokens(phon.dec_input_ids).sum().backward()
    out["grads"] = {
        "phonology_embedding": model.phonology_embedding.weight.grad.clone(),
        "phon_position_embedding": model.phon_position_embedding.weight.grad.clone(),
    }
    model.zero_grad(set_to_none=True)

    # ---- forward logits, every trainable pathway --------------------------------
    kwargs = {
        "o2p": {
            "orth_enc_input": orth.enc_input_ids,
            "orth_enc_pad_mask": orth.enc_pad_mask,
            "phon_dec_input": phon.dec_input_ids,
            "phon_dec_pad_mask": phon.dec_pad_mask,
        },
        "p2o": {
            "phon_enc_input": phon.enc_input_ids,
            "phon_enc_pad_mask": phon.enc_pad_mask,
            "orth_dec_input": orth.dec_input_ids,
            "orth_dec_pad_mask": orth.dec_pad_mask,
        },
        "p2p": {
            "phon_enc_input": phon.enc_input_ids,
            "phon_enc_pad_mask": phon.enc_pad_mask,
            "phon_dec_input": phon.dec_input_ids,
            "phon_dec_pad_mask": phon.dec_pad_mask,
        },
        "op2op": {
            "orth_enc_input": orth.enc_input_ids,
            "orth_enc_pad_mask": orth.enc_pad_mask,
            "orth_dec_input": orth.dec_input_ids,
            "orth_dec_pad_mask": orth.dec_pad_mask,
            "phon_enc_input": phon.enc_input_ids,
            "phon_enc_pad_mask": phon.enc_pad_mask,
            "phon_dec_input": phon.dec_input_ids,
            "phon_dec_pad_mask": phon.dec_pad_mask,
        },
    }
    logits: dict[str, dict[str, torch.Tensor]] = {}
    with torch.no_grad():
        for pathway, kw in kwargs.items():
            logits[pathway] = {k: v.clone() for k, v in model(task=pathway, **kw).items()}
    out["logits"] = logits

    # ---- deterministic generation ----------------------------------------------
    gen: dict[str, dict] = {}
    for pathway in ("o2p", "p2o", "p2p", "op2op", "o2o"):
        result = model.generate(encoding, pathway, deterministic=True)
        entry: dict = {"global_encoding": result.global_encoding.clone()}
        if result.orth_tokens is not None:
            entry["orth_tokens"] = result.orth_tokens.clone()
            entry["orth_probs"] = stack_nested(result.orth_probs)
        if result.phon_tokens is not None:
            entry["phon_tokens"] = canon_ragged(result.phon_tokens)
            entry["phon_probs"] = stack_nested(result.phon_probs)
            entry["phon_vecs"] = stack_nested(result.phon_vecs)
        gen[pathway] = entry
    out["generate"] = gen

    return out


if __name__ == "__main__":
    fixture = build()
    torch.save(fixture, FIXTURE)
    print(f"wrote {FIXTURE}")
    print(f"  words            : {len(WORDS)}")
    print(
        f"  phon enc shape   : {len(fixture['tok']['phon_enc_features'])} x "
        f"{len(fixture['tok']['phon_enc_features'][0])}"
    )
    print(f"  targets shape    : {tuple(fixture['tok']['phon_targets'].shape)}")
    print(f"  pathways (logits): {sorted(fixture['logits'])}")
    print(f"  pathways (gen)   : {sorted(fixture['generate'])}")
