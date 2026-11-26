"""`BRIDGE-Models.drawio` is checked against the running model, not against a description of it.

The diagram is served on the model's website, one page per pathway. Each page draws every
block and fades the ones its pathway does not use, so every page makes a claim of the form
"this pathway runs that module". Those claims are checked here against what the model
actually runs: forward hooks on every top-level submodule record which ones fire during
the training forward of each pathway, and a block must be opaque exactly when its module
fired. A refactor that changes what a pathway runs therefore fails a test naming the page
and the block, rather than leaving a public figure quietly wrong.

Every page also claims which side of a training example each block reads. The boxes on the
left are the input and the boxes on the right are the target, and they hold different words.
An encoder is drawn reading the input. A decoder is drawn fed the target, through a
tokenization and embedding block of its own. Those claims are checked against hooks on the
encoders and decoders, run on the words the figure draws: the input and the target differ
in length there, so a captured tensor can be the embedding of only one of them. A figure
that feeds a decoder from the input's embedding cube fails.

Every page also claims where position enters. The two groups of blocks that hold the
encoders and the decoders name it beside their self-attention, and no other label names it.
That is checked against a measurement of which stages read the order of the sequence they
are handed: each stage is run again on the tensors the forward gave it, with the sequence
reversed. A figure whose embedding blocks say "+ Position" fails, as does one whose
encoders do not name it.

What a test cannot check, and what is therefore left to review: whether a label's wording
is the clearest true description, and layout.
"""

from __future__ import annotations

import csv
import html
import re
import types
import xml.etree.ElementTree as ET
from functools import cache
from pathlib import Path

import pytest
import torch

from bridge.application.training.training_pipeline import TrainingPipeline
from bridge.core.phonreps import SPECIAL_TOKENS, load_phoneme_table
from bridge.domain.datamodels import BridgeBatch, ModelConfig
from bridge.domain.datamodels.pathways import PATHWAY_IO, PATHWAYS
from bridge.domain.datamodels.vocab_spec import VocabSpec
from bridge.domain.model import Model
from bridge.domain.model.layers import Attention
from bridge.domain.tokenizer import BridgeTokenizer

ROOT = Path(__file__).resolve().parents[1]
DIAGRAM = ROOT / "BRIDGE-Models.drawio"

# Every pathway gets exactly one page and every page is a pathway.
PAGE_PATHWAY = {
    "OP2OP Model": "op2op",
    "O2P Model": "o2p",
    "O2O Model": "o2o",
    "P2O Model": "p2o",
    "P2P Model": "p2p",
}

# Diagram label -> the top-level `Model` submodule whose firing decides whether the block
# is drawn opaque. Several blocks share a module: an encoded cube exists exactly when its
# encoder ran, and every block in the global-embedding stage exists exactly when the mixer
# ran. Both cross-attention blocks and their outputs run together, only inside `embed_op`.
BLOCK_MODULE = {
    "Orthography Encoder": "orthography_encoder",
    "Phonology Encoder": "phonology_encoder",
    "Encoded Orthography": "orthography_encoder",
    "Encoded Phonology": "phonology_encoder",
    "Phonetically Re-encoded Orthography": "gp_multihead_attention",
    "Orthographically Re-encoded Phonology": "pg_multihead_attention",
    "Concatenated output with GE": "gp_multihead_attention",
    "Transformer Mixer Encoder": "transformer_mixer",
    "Mixed Encoding": "transformer_mixer",
    "Encoded GE": "transformer_mixer",
    "Orthography Decoder": "orthography_decoder",
    "Phonology Decoder": "phonology_decoder",
}
CROSS_ATTENTION_PREFIX = "Multihead Cross-Attention"

# Blocks on one modality's input lane, which runs from the left-hand box through that
# modality's tokenizer and embedding cube to its encoder. A pathway uses the lane exactly
# when it reads the modality. What it writes takes the other lane, from the right-hand box
# through a DECODER_EMBEDDING block to the decoder.
MODALITY_BLOCK = {
    "Orthographic Tokenizer": "orth",
    "Phoneme Tokenizer": "phon",
    "Orthographic Sequence Length": "orth",
    "Phonological Sequence Length": "phon",
}
DECODER_EMBEDDING = "Tokenization + Embedding"

# The two labels that say where position enters, each on the group that holds the blocks it
# describes. No other label on a page names position.
POSITION_GROUPS = {
    "Multihead Self-Attention (rotary position) Encoders": (
        "Orthography Encoder",
        "Phonology Encoder",
    ),
    "Masked Self-Attention (rotary position) + Cross-Attention Decoders": (
        "Orthography Decoder",
        "Phonology Decoder",
    ),
}

# Kind of stage -> whether the figure draws position entering there. The two labels above
# put it beside self-attention, and the decoders' one ahead of "+ Cross-Attention".
DRAWN_WITH_POSITION = {
    "embedding": False,
    "encoder self-attention": True,
    "cross-attention": False,
    "mixer": False,
    "decoder self-attention": True,
    "decoder cross-attention": False,
}

# The blocks that are fed a token sequence, and so read one side of the example.
TOKEN_READERS = (
    "Orthography Encoder",
    "Phonology Encoder",
    "Orthography Decoder",
    "Phonology Decoder",
)

# The words on the figure's cards. Each input differs from its target, in length too, so a
# decoder fed from the input is a different drawing and a different tensor.
INPUT_WORDS = ["cats", "apples", "rivers"]
TARGET_WORDS = ["cat", "apple", "river"]


def plain(value: str | None) -> str:
    """A cell's label as a reader sees it."""
    text = re.sub(r"<[^>]+>", " ", value or "")
    return re.sub(r"\s+", " ", html.unescape(text).replace("\xa0", " ")).strip()


def style_of(cell: ET.Element) -> dict[str, str]:
    parts = (cell.get("style") or "").split(";")
    return dict(p.split("=", 1) for p in parts if "=" in p)


def faded(cell: ET.Element) -> bool:
    """The diagram's convention for "not used on this page": opacity or textOpacity below 100."""
    style = style_of(cell)
    return any(float(style.get(key, "100")) < 100 for key in ("opacity", "textOpacity"))


class Page:
    def __init__(self, diagram: ET.Element) -> None:
        self.name = diagram.get("name", "")
        cells = list(diagram.iter("mxCell"))
        self.cells = {c.get("id"): c for c in cells}
        self.vertices = [c for c in cells if c.get("vertex") == "1"]
        self.edges = [c for c in cells if c.get("edge") == "1"]
        self._origin: dict[str, tuple[float, float]] = {}
        self._frame: tuple[float, float] | None = None

    def origin(self, cid: str | None) -> tuple[float, float]:
        """Absolute top-left of a vertex, summing its parents' offsets."""
        if cid is None or cid not in self.cells or self.cells[cid].get("vertex") != "1":
            return (0.0, 0.0)
        if cid not in self._origin:
            cell = self.cells[cid]
            geometry = cell.find("mxGeometry")
            px, py = self.origin(cell.get("parent"))
            x = float(geometry.get("x", 0)) if geometry is not None else 0.0
            y = float(geometry.get("y", 0)) if geometry is not None else 0.0
            self._origin[cid] = (px + x, py + y)
        return self._origin[cid]

    def labelled(self, label: str) -> list[ET.Element]:
        return [c for c in self.vertices if plain(c.get("value")) == label]

    def frame_origin(self) -> tuple[float, float]:
        """Top-left of the page's border rectangle: its widest vertex that is not a group."""
        if self._frame is None:
            frame = max(
                (c for c in self.vertices if not (c.get("style") or "").startswith("group")),
                key=lambda c: float(c.find("mxGeometry").get("width", 0)),
            )
            self._frame = self.origin(frame.get("id"))
        return self._frame

    def box(self, cid: str) -> tuple[float, float, float, float]:
        """Frame-relative (x, y, width, height) as stored, so independent of where draw.io
        happens to keep the page on its canvas."""
        x, y = self.origin(cid)
        fx, fy = self.frame_origin()
        geometry = self.cells[cid].find("mxGeometry")
        return (x - fx, y - fy, float(geometry.get("width", 0)), float(geometry.get("height", 0)))

    def visual_box(self, cid: str) -> tuple[float, float, float, float]:
        """`box` as drawn: a cell rotated by a quarter turn keeps its centre and swaps sides."""
        x, y, w, h = self.box(cid)
        if style_of(self.cells[cid]).get("rotation") in ("90", "-90", "270", "-270"):
            return (x + w / 2 - h / 2, y + h / 2 - w / 2, h, w)
        return (x, y, w, h)

    def frame_mid_x(self) -> float:
        """Horizontal centre of the page, separating input boxes from output boxes."""
        widest = max(self.vertices, key=lambda c: float(c.find("mxGeometry").get("width", 0)))
        x, _ = self.origin(widest.get("id"))
        return x + float(widest.find("mxGeometry").get("width")) / 2


@cache
def load_pages(path: Path = DIAGRAM) -> dict[str, Page]:
    root = ET.parse(path).getroot()
    return {d.get("name", ""): Page(d) for d in root.iter("diagram")}


def word_box(page: Page, cell: ET.Element) -> tuple[str, str] | None:
    """The side of the example and the modality a box of words holds, or None for any other
    cell. The boxes left of the page's centre are the input and those right of it the target."""
    label = plain(cell.get("value"))
    if label not in ("Orthography", "Phonology", "Phonology (features)"):
        return None
    side = "input" if page.origin(cell.get("id"))[0] < page.frame_mid_x() else "target"
    return (side, "orth" if label == "Orthography" else "phon")


def drawn_sources(page: Page, cid: str) -> set[tuple[str, str]]:
    """The boxes of words a block's tokens are drawn as coming from.

    Follows the arrows into the block backwards and stops at a box of words. It goes no
    further back than a module's output, since what a module emits is not the words."""
    found: set[tuple[str, str]] = set()
    seen, stack = {cid}, [cid]
    while stack:
        here = stack.pop()
        for edge in page.edges:
            source = page.cells.get(edge.get("source"))
            if edge.get("target") != here or source is None or source.get("id") in seen:
                continue
            seen.add(source.get("id"))
            box = word_box(page, source)
            if box is not None:
                found.add(box)
            elif plain(source.get("value")) not in BLOCK_MODULE:
                stack.append(source.get("id"))
    return found


@cache
def fired_modules() -> dict[str, frozenset[str]]:
    """Top-level submodules that fire in each pathway's training forward, measured.

    Runs the real `TrainingPipeline.forward` body, which builds the pathway's keyword
    arguments from `PATHWAY_INPUTS`, against a stub carrying only the two attributes it
    reads. A small seeded model is enough: which modules run does not depend on sizes.
    """
    tokenizer = BridgeTokenizer()
    config = ModelConfig(vocab=VocabSpec.from_tokenizer(tokenizer), d_model=16, nhead=2, seed=0)
    model = Model(config)
    batch = figure_batch(tokenizer)
    fired: set[str] = set()
    for name, module in model.named_children():
        module.register_forward_pre_hook(lambda _m, _a, name=name: fired.add(name))
    result = {}
    with torch.no_grad():
        for pathway in PATHWAYS:
            fired.clear()
            stub = types.SimpleNamespace(
                model=model, training_config=types.SimpleNamespace(training_pathway=pathway)
            )
            TrainingPipeline.forward(stub, batch)
            result[pathway] = frozenset(fired)
    return result


def figure_batch(tokenizer: BridgeTokenizer) -> BridgeBatch:
    """The words the figure draws, as a batch: its input cards and its target cards."""
    return BridgeBatch(
        input=tokenizer.encode_input(INPUT_WORDS), target=tokenizer.encode_target(TARGET_WORDS)
    )


def pronunciations(words: list[str]) -> list[str]:
    """Each word's phonemes as the live tokenizer encodes them, specials left out."""
    by_row = {row: phoneme for phoneme, row in load_phoneme_table().row_index.items()}
    rows = BridgeTokenizer().encode_input(words).phon.ids.tolist()
    return [" ".join(p for r in row if (p := by_row[r]) not in SPECIAL_TOKENS) for row in rows]


def small_model(d_embedding: int = 1) -> tuple[Model, BridgeTokenizer]:
    tokenizer = BridgeTokenizer()
    vocab = VocabSpec.from_tokenizer(tokenizer)
    config = ModelConfig(vocab=vocab, d_model=16, nhead=2, d_embedding=d_embedding, seed=0)
    return Model(config).eval(), tokenizer


@cache
def decoder_calls_causal() -> dict[str, bool]:
    """Whether every self-attention of each decoder, over every pathway's training forward
    and generation, was allowed no key ahead of its query, measured. A decoder that never
    ran is absent."""
    model, tokenizer = small_model()
    batch = figure_batch(tokenizer)
    start = tokenizer.orth_start([None] * len(batch))
    seen: dict[str, bool] = {}

    def record(name: str):
        def hook(_m, args):
            allowed = args[2]
            ahead = torch.triu(torch.ones(allowed.shape[-2:], dtype=torch.bool), 1)
            seen[name] = seen.get(name, True) and not bool((allowed & ahead).any())

        return hook

    for name in ("orthography_decoder", "phonology_decoder"):
        for layer in getattr(model, name).layers:
            layer.self_attn.register_forward_pre_hook(record(name))
    with torch.no_grad():
        for pathway in PATHWAYS:
            stub = types.SimpleNamespace(
                model=model, training_config=types.SimpleNamespace(training_pathway=pathway)
            )
            TrainingPipeline.forward(stub, batch)
            model.generate(
                batch.input, pathway, max_new_tokens=8, orth_start=start, deterministic=True
            )
    return seen


@cache
def global_embedding_added_back() -> bool:
    """Whether the global encoding is the mixer output's first `d_embedding` rows plus the
    global embedding itself, bitwise, for all three encoders, measured at `d_embedding=2`.

    Asserts its own control: the same rows without the add must differ, or the comparison
    could not tell the two apart.
    """
    model, tokenizer = small_model(d_embedding=2)
    inputs = figure_batch(tokenizer).input
    o, p = inputs.orth, inputs.phon
    captured: dict[str, torch.Tensor] = {}
    model.transformer_mixer.register_forward_hook(lambda _m, _a, out: captured.update(mix=out))
    encoders = (
        lambda: model.embed_o(o.ids, o.pad_mask),
        lambda: model.embed_p(p.ids, p.pad_mask),
        lambda: model.embed_op(o.ids, o.pad_mask, p.ids, p.pad_mask),
    )
    holds = True
    with torch.no_grad():
        for encode in encoders:
            out = encode()
            rows = captured["mix"][:, :2]
            assert not torch.equal(out, rows), "control failed: rows alone already equal the output"
            holds &= torch.equal(out, rows + model.global_embedding.expand(rows.shape[0], -1, -1))
    return holds


@cache
def fed_sides() -> dict[str, frozenset[str]]:
    """Which side of the batch each encoder and decoder is fed, over every pathway's
    training forward, measured.

    A forward pre-hook on each captures the embedded sequence it receives, which is compared
    bitwise with the embedding of the input's ids and of the target's ids. The figure's
    words give the two different lengths, asserted here as the control, so a captured
    tensor can match only one of them.
    """
    model, tokenizer = small_model()
    batch = figure_batch(tokenizer)
    embed = {"orth": model.embed_orth_tokens, "phon": model.embed_phon_tokens}
    with torch.no_grad():
        embedded = {
            (side, modality): embed[modality](getattr(getattr(batch, side), modality).ids)
            for side in ("input", "target")
            for modality in embed
        }
    for modality in embed:
        assert embedded["input", modality].shape != embedded["target", modality].shape, (
            "control failed: the input and the target embed to the same shape"
        )
    seen: dict[str, set[str]] = {}

    def record(name: str, modality: str):
        def hook(_m, args, kwargs):
            fed = kwargs["tgt"] if "tgt" in kwargs else args[0]
            sides = [
                side
                for side in ("input", "target")
                if (e := embedded[side, modality]).shape == fed.shape and torch.equal(e, fed)
            ]
            seen.setdefault(name, set()).update(sides or ["neither"])

        return hook

    for label in TOKEN_READERS:
        modality = "orth" if label.startswith("Orthography") else "phon"
        module = getattr(model, BLOCK_MODULE[label])
        module.register_forward_pre_hook(record(BLOCK_MODULE[label], modality), with_kwargs=True)
    with torch.no_grad():
        for pathway in PATHWAYS:
            stub = types.SimpleNamespace(
                model=model, training_config=types.SimpleNamespace(training_pathway=pathway)
            )
            TrainingPipeline.forward(stub, batch)
    return {name: frozenset(sides) for name, sides in seen.items()}


@cache
def reversal_deviations() -> dict[str, tuple[str, float]]:
    """Each stage of the model, its kind, and how far its output moves when the sequence it
    is handed is reversed, measured.

    One op2op training forward runs every stage. It runs on the figure's first cards alone,
    so nothing is padding and reversing a sequence moves only real tokens. A hook on every
    attention and on the mixer keeps what the forward handed it, and each is then run again
    on those tensors with the sequence reversed. An attention has its keys and values
    reversed, and its mask with them, which changes nothing unless it reads where a key
    sits. The mixer and the embedders have their rows reversed, and must give the same
    rows back, reversed.

    A deviation is the largest difference over the largest magnitude of the reference. A
    stage that reads no order moves by a few float32 epsilons at most, the cost of adding in
    another order, and an embedder by nothing at all.
    """
    model, tokenizer = small_model(d_embedding=2)
    batch = BridgeBatch(
        input=tokenizer.encode_input(INPUT_WORDS[:1]),
        target=tokenizer.encode_target(TARGET_WORDS[:1]),
    )
    sides = (batch.input.orth, batch.input.phon, batch.target.orth, batch.target.phon)
    assert not any(side.pad_mask.any() for side in sides), "control failed: a row is padded"
    kinds = {"transformer_mixer": "mixer"}
    kinds |= dict.fromkeys(("gp_multihead_attention", "pg_multihead_attention"), "cross-attention")
    for name, module in model.named_modules():
        if isinstance(module, Attention):
            stack = "encoder" if "encoder" in name else "decoder"
            kinds[name] = f"{stack} {'self' if name.endswith('self_attn') else 'cross'}-attention"
    handed: dict[str, tuple[tuple, dict]] = {}
    for name in kinds:
        model.get_submodule(name).register_forward_pre_hook(
            lambda _m, args, kwargs, name=name: handed.update({name: (args, kwargs)}),
            with_kwargs=True,
        )
    deviation: dict[str, float] = {}

    def compare(name: str, reference: torch.Tensor, reversed_: torch.Tensor) -> None:
        deviation[name] = float((reference - reversed_).abs().max() / reference.abs().max())

    with torch.no_grad():
        stub = types.SimpleNamespace(
            model=model, training_config=types.SimpleNamespace(training_pathway="op2op")
        )
        TrainingPipeline.forward(stub, batch)
        assert set(handed) == set(kinds), "control failed: a stage did not run"
        for name, (args, kwargs) in list(handed.items()):
            module = model.get_submodule(name)
            if isinstance(module, Attention):
                query, memory, allowed, *positions = args
                mask = None if allowed is None else allowed.flip(-1)
                compare(name, module(*args), module(query, memory.flip(1), mask, *positions))
            elif kinds[name] == "cross-attention":
                query, key, value = args
                turned = module(query, key.flip(1), value.flip(1), **kwargs)[0]
                compare(name, module(*args, **kwargs)[0], turned)
            else:
                (rows,) = args
                compare(name, module(rows, **kwargs), module(rows.flip(1), **kwargs).flip(1))
        for name, modality in (("embed_orth_tokens", "orth"), ("embed_phon_tokens", "phon")):
            ids = getattr(batch.input, modality).ids
            assert not torch.equal(ids, ids.flip(1)), "control failed: the row is a palindrome"
            kinds[name] = "embedding"
            compare(name, getattr(model, name)(ids), getattr(model, name)(ids.flip(1)).flip(1))
    return {name: (kinds[name], deviation[name]) for name in kinds}


def expected_blocks(page: Page, pathway: str) -> dict[str, tuple[ET.Element, bool]]:
    """Every block on the page this module can judge, keyed by a readable name, with whether
    the pathway uses it."""
    fired = fired_modules()[pathway]
    reads, writes = PATHWAY_IO[pathway]  # type: ignore[index]
    out: dict[str, tuple[ET.Element, bool]] = {}
    for cell in page.vertices:
        label = plain(cell.get("value"))
        key = f"{label!r} ({cell.get('id')})"
        if label in BLOCK_MODULE:
            out[key] = (cell, BLOCK_MODULE[label] in fired)
        elif label.startswith(CROSS_ATTENTION_PREFIX):
            out[key] = (cell, "gp_multihead_attention" in fired)
        elif label in MODALITY_BLOCK:
            out[key] = (cell, MODALITY_BLOCK[label] in reads)
        elif label == DECODER_EMBEDDING:
            # Used exactly when the decoder it feeds runs.
            feeds = {
                plain(page.cells[e.get("target")].get("value"))
                for e in page.edges
                if e.get("source") == cell.get("id") and e.get("target") in page.cells
            }
            out[key] = (cell, any(BLOCK_MODULE.get(name) in fired for name in feeds))
        elif (box := word_box(page, cell)) is not None:
            side, modality = box
            out[f"{side} {key}"] = (cell, modality in (reads if side == "input" else writes))
    return out


# Each check takes the file to read, so it can be run against any copy of the diagram.


def page_set_problems(path: Path = DIAGRAM) -> list[str]:
    pages = load_pages(path)
    problems = []
    if set(pages) != set(PAGE_PATHWAY):
        problems.append(f"pages {sorted(pages)} != expected {sorted(PAGE_PATHWAY)}")
    if sorted(PAGE_PATHWAY.values()) != sorted(PATHWAYS):
        problems.append(
            f"PAGE_PATHWAY covers {sorted(PAGE_PATHWAY.values())}, PATHWAYS is {PATHWAYS}"
        )
    for name, page in pages.items():
        pathway = PAGE_PATHWAY.get(name)
        if pathway and not page.labelled(f"{pathway.upper()} Pathway"):
            problems.append(f"page {name!r} has no title reading {pathway.upper()!r} Pathway")
    return problems


def opacity_problems(path: Path = DIAGRAM) -> list[str]:
    problems = []
    for name, page in load_pages(path).items():
        pathway = PAGE_PATHWAY.get(name)
        if pathway is None:  # reported by page_set_problems
            continue
        blocks = expected_blocks(page, pathway)
        if len(blocks) < 20:
            problems.append(f"{name}: only {len(blocks)} blocks recognised; labels changed?")
        for key, (cell, used) in blocks.items():
            if used == faded(cell):
                state = "faded" if faded(cell) else "opaque"
                problems.append(
                    f"{name}: {key} is drawn {state} but {pathway} {'uses' if used else 'does not use'} it"
                )
    return problems


def mixer_input_problems(path: Path = DIAGRAM) -> list[str]:
    """The block feeding the mixer on a single-modality pathway names the modality it holds.

    `embed_o` and `embed_p` prepend the global embedding to whichever encoding the pathway
    reads; `embed_op` concatenates both, which page 1 draws separately.
    """
    problems = []
    for name, page in load_pages(path).items():
        if name not in PAGE_PATHWAY:  # reported by page_set_problems
            continue
        reads = PATHWAY_IO[PAGE_PATHWAY[name]][0]  # type: ignore[index]
        found = {
            plain(c.get("value"))
            for c in page.vertices
            if re.fullmatch(r"Encoded \w+ with GE", plain(c.get("value")))
        }
        want = {"Encoded Orthography with GE"} if reads == {"orth"} else set()
        want |= {"Encoded Phonology with GE"} if reads == {"phon"} else set()
        if found != want:
            problems.append(f"{name}: mixer-input labels {sorted(found)}, expected {sorted(want)}")
    return problems


def edge_problems(path: Path = DIAGRAM) -> list[str]:
    """Every arrow is attached at both ends, animates exactly when it is drawn active, and is
    drawn active exactly when both blocks it joins are. Every "+" sums at least two inputs
    and feeds something, and no "+" feeds a decoder: the decoders' causal mask goes to
    self-attention, and the decoder input reaches the first layer unchanged
    (measured bitwise), so a sum drawn into a decoder claims an addition that never happens."""
    decoders = {"Orthography Decoder", "Phonology Decoder"}
    problems = []
    for name, page in load_pages(path).items():
        for edge in page.edges:
            eid = edge.get("id")
            src, tgt = page.cells.get(edge.get("source")), page.cells.get(edge.get("target"))
            if src is None or tgt is None:
                problems.append(f"{name}: edge {eid} is not attached at both ends")
                continue
            active = not faded(edge)
            animated = style_of(edge).get("flowAnimation") == "1"
            if active != animated:
                problems.append(f"{name}: edge {eid} active={active} but animated={animated}")
            if active != (not faded(src) and not faded(tgt)):
                problems.append(
                    f"{name}: edge {eid} active={active} joins {plain(src.get('value'))!r} "
                    f"(faded={faded(src)}) to {plain(tgt.get('value'))!r} (faded={faded(tgt)})"
                )
        for cell in page.vertices:
            if plain(cell.get("value")) != "+":
                continue
            cid = cell.get("id")
            n_in = sum(e.get("target") == cid for e in page.edges)
            n_out = sum(e.get("source") == cid for e in page.edges)
            if n_in < 2 or n_out < 1:
                problems.append(f"{name}: '+' {cid} has {n_in} inputs and {n_out} outputs")
            fed = [
                plain(page.cells[e.get("target")].get("value"))
                for e in page.edges
                if e.get("source") == cid and e.get("target") in page.cells
            ]
            if decoders & set(fed):
                problems.append(f"{name}: '+' {cid} feeds {sorted(decoders & set(fed))}")
    return problems


def axis_label_problems(path: Path = DIAGRAM) -> list[str]:
    """Each embedding cube's top face names the axis the embedded tensor actually has there.

    Measured: the embedders map (batch, L) ids to (batch, L, d_model) for two lengths, so
    the axis is sequence length, never a vocabulary.
    """
    tokenizer = BridgeTokenizer()
    model = Model(ModelConfig(vocab=VocabSpec.from_tokenizer(tokenizer), d_model=16, nhead=2))
    for length in (3, 7):
        ids = torch.zeros((2, length), dtype=torch.long)
        assert model.embed_orth_tokens(ids).shape[1] == length
        assert model.embed_phon_tokens(ids).shape[1] == length
    # A cube's top-face label is "<Modality> <two words>", optionally prefixed "Max"; the
    # tokenizer labels have one word after the modality and so are not matched.
    axis = re.compile(r"(?:Max )?(Orthographic|Phonological) \w+ \w+")
    problems = []
    for name, page in load_pages(path).items():
        labels = [plain(c.get("value")) for c in page.vertices]
        for modality in ("Orthographic", "Phonological"):
            found = [lab for lab in labels if (m := axis.fullmatch(lab)) and m[1] == modality]
            if found != [f"{modality} Sequence Length"]:
                problems.append(
                    f"{name}: {modality} cube axis labels {found}; the axis is the batch's padded sequence length"
                )
    return problems


def layout_problems(path: Path = DIAGRAM) -> list[str]:
    """A labelled block that appears on more than one page sits at the same place on each, so
    switching pages on the website does not make it jump. Compared frame-relative, within half
    a unit, against the first page that has it; a label used several times on a page (the
    word cards, the "+" circles) is compared as a sorted list of positions."""
    pages = load_pages(path)
    positions: dict[str, dict[str, list[tuple[float, ...]]]] = {}
    for name, page in pages.items():
        for cell in page.vertices:
            label = plain(cell.get("value"))
            if label and not label.endswith(" Pathway"):
                positions.setdefault(label, {}).setdefault(name, []).append(
                    page.box(cell.get("id"))
                )
    problems = []
    compared = 0
    for label, per_page in positions.items():
        if len(per_page) < 2:
            continue
        (first, reference), *others = per_page.items()
        for name, boxes in others:
            compared += 1
            if len(boxes) != len(reference):
                problems.append(f"{label!r}: {len(reference)} on {first}, {len(boxes)} on {name}")
                continue
            for a, b in zip(sorted(reference), sorted(boxes), strict=True):
                if max(abs(u - v) for u, v in zip(a, b, strict=True)) > 0.5:
                    problems.append(
                        f"{label!r} sits at {tuple(round(v, 1) for v in b)} on {name} but "
                        f"{tuple(round(v, 1) for v in a)} on {first}"
                    )
    if compared < 100:
        problems.append(f"only {compared} block comparisons made; labels changed?")
    return problems


def mask_problems(path: Path = DIAGRAM) -> list[str]:
    """Every page draws an "Autoregressive Mask" arrowing straight into each decoder, faded
    exactly when that decoder is, because every decoder call is causal (measured)."""
    causal = decoder_calls_causal()
    problems = [
        f"{name} let a query read a key ahead of it" for name, ok in causal.items() if not ok
    ]
    for name, page in load_pages(path).items():
        for decoder_label in ("Orthography Decoder", "Phonology Decoder"):
            for decoder in page.labelled(decoder_label):
                masks = [
                    page.cells[e.get("source")]
                    for e in page.edges
                    if e.get("target") == decoder.get("id")
                    and plain(page.cells.get(e.get("source"), decoder).get("value"))
                    == "Autoregressive Mask"
                ]
                if len(masks) != 1:
                    problems.append(f"{name}: {decoder_label} is fed by {len(masks)} masks")
                elif faded(masks[0]) != faded(decoder):
                    problems.append(
                        f"{name}: {decoder_label}'s mask faded={faded(masks[0])}, "
                        f"decoder faded={faded(decoder)}"
                    )
    return problems


def global_residual_problems(path: Path = DIAGRAM) -> list[str]:
    """Every page adds the global embedding back to the rows plucked off the mixer output: a
    "+" fed by "Global Embedding (Learnable)" and by an arrow labelled "Extract GE", whose
    output is the block labelled "Encoded GE". The code does exactly that (measured)."""
    problems = [] if global_embedding_added_back() else ["the code no longer adds GE back"]
    for name, page in load_pages(path).items():
        (ge,) = page.labelled("Global Embedding (Learnable)")
        (encoded_text,) = page.labelled("Encoded GE")
        ex, ey, ew, eh = page.visual_box(encoded_text.get("id"))
        sums = [
            c
            for c in page.labelled("+")
            if any(
                e.get("source") == ge.get("id") and e.get("target") == c.get("id")
                for e in page.edges
            )
        ]
        if len(sums) != 1:
            problems.append(f"{name}: {len(sums)} '+' circles take the Global Embedding")
            continue
        cid = sums[0].get("id")
        inputs = [e for e in page.edges if e.get("target") == cid]
        extract = [
            e
            for e in inputs
            if any(
                v.get("parent") == e.get("id") and plain(v.get("value")) == "Extract GE"
                for v in page.vertices
            )
        ]
        outputs = [e.get("target") for e in page.edges if e.get("source") == cid]
        holds_encoded = [
            t
            for t in outputs
            if (b := page.visual_box(t))[0] <= ex
            and ex + ew <= b[0] + b[2] + 1
            and b[1] <= ey
            and ey + eh <= b[1] + b[3] + 1
        ]
        if len(inputs) != 2 or len(extract) != 1 or len(holds_encoded) != 1:
            problems.append(
                f"{name}: GE '+' {cid} has {len(inputs)} inputs, {len(extract)} labelled "
                f"Extract GE, {len(holds_encoded)} outputs into Encoded GE"
            )
    return problems


@cache
def cat_targets() -> tuple[list[str], list[str], list[list[int]]]:
    """Feature names in output-column order, the phonemes of the first target word, "cat",
    plus [EOS], and the decoder's training targets for them, all from the live feature
    table and tokenizer."""
    header = next(csv.reader(open(ROOT / "bridge" / "core" / "phonreps.csv")))[1:]
    names = [h.lstrip("#") for h in header] + list(SPECIAL_TOKENS[:-1])  # [PAD] is not predicted
    table = load_phoneme_table()
    by_row = {row: phoneme for phoneme, row in table.row_index.items()}
    phon = BridgeTokenizer().encode_target(TARGET_WORDS[:1]).phon
    ids = [int(r) for r in phon.ids[0]]
    phonemes = [by_row[r] for r in ids if by_row[r] not in SPECIAL_TOKENS] + ["[EOS]"]
    targets = phon.labels[0, : len(phonemes)].tolist()
    assert all(len(t) == len(names) for t in targets), "feature names and target width disagree"
    return names, phonemes, targets


def output_table_problems(path: Path = DIAGRAM) -> list[str]:
    """The output box's feature table states the real decoder targets for "cat".

    Columns are its phonemes plus [EOS]; rows are exactly the features on in any of them, in
    table order; every 0/1 is the training target; the remainder is counted; and the table
    fades with its box. An edit to phonreps.csv that changes any of this fails here."""
    names, phonemes, targets = cat_targets()

    def short(name: str) -> str:
        return name.replace("[EOS]", "EOS")

    want_rows = [n for i, n in enumerate(names) if any(t[i] for t in targets)]
    problems = []
    for name, page in load_pages(path).items():
        (box,) = page.labelled("Phonology (features)")
        cells = [c for c in page.vertices if c.get("parent") == box.get("parent") and c is not box]
        texts = {c.get("id"): plain(c.get("value")) for c in cells if plain(c.get("value"))}

        def centre(cid: str, page: Page = page) -> tuple[float, float]:
            x, y, w, h = page.box(cid)
            return (x + w / 2, y + h / 2)

        heads = [c for c, t in texts.items() if t in {short(p) for p in phonemes}]
        top = min((centre(c)[1] for c in heads), default=0)
        heads = sorted(
            (c for c in heads if abs(centre(c)[1] - top) < 1), key=lambda c: centre(c)[0]
        )
        left = centre(heads[0])[0] if heads else 0
        rows = sorted(
            (c for c, t in texts.items() if t in {short(n) for n in names} and centre(c)[0] < left),
            key=lambda c: centre(c)[1],
        )
        if [texts[c] for c in heads] != [short(p) for p in phonemes]:
            problems.append(f"{name}: columns {[texts[c] for c in heads]}, want {phonemes}")
            continue
        if [texts[c] for c in rows] != [short(n) for n in want_rows]:
            problems.append(f"{name}: rows {[texts[c] for c in rows]}, want {want_rows}")
            continue
        for r, row in enumerate(rows):
            for k, head in enumerate(heads):
                digit = [
                    texts[c]
                    for c in texts
                    if texts[c] in ("0", "1")
                    and abs(centre(c)[1] - centre(row)[1]) < 1
                    and abs(centre(c)[0] - centre(head)[0]) < 3
                ]
                want = str(targets[k][names.index(want_rows[r])])
                if digit != [want]:
                    problems.append(
                        f"{name}: {want_rows[r]} at {phonemes[k]} reads {digit}, want {want}"
                    )
        rest = len(names) - len(want_rows)
        if not any(f"+ {rest} more features" in t for t in texts.values()):
            problems.append(f"{name}: no '+ {rest} more features' line")
        wrong_fade = [c for c in texts if faded(page.cells[c]) != faded(box)]
        if wrong_fade:
            problems.append(f"{name}: {len(wrong_fade)} table cells fade differently from the box")
    return problems


def feed_problems(path: Path = DIAGRAM) -> list[str]:
    """Every encoder is drawn reading the input box of its modality and every decoder is
    drawn fed from the target box of its modality, which is what the model does (measured).

    A decoder's tokens reach it through exactly one DECODER_EMBEDDING block: what it is fed
    is the embedding of the target's ids, not the ids. The check is made on every page,
    faded blocks included, because a faded block is drawn with the same arrows."""
    sides = fed_sides()
    problems = []
    for name, page in load_pages(path).items():
        for label in TOKEN_READERS:
            modality = "orth" if label.startswith("Orthography") else "phon"
            want = {(side, modality) for side in sides[BLOCK_MODULE[label]]}
            for block in page.labelled(label):
                drawn = drawn_sources(page, block.get("id"))
                if drawn != want:
                    problems.append(
                        f"{name}: {label} is drawn fed from {sorted(drawn)}, "
                        f"but the model feeds it {sorted(want)}"
                    )
                embeddings = [
                    e
                    for e in page.edges
                    if e.get("target") == block.get("id")
                    and plain(page.cells.get(e.get("source"), block).get("value"))
                    == DECODER_EMBEDDING
                ]
                if len(embeddings) != (1 if label.endswith("Decoder") else 0):
                    problems.append(
                        f"{name}: {label} is fed by {len(embeddings)} {DECODER_EMBEDDING!r} blocks"
                    )
    return problems


def word_problems(path: Path = DIAGRAM) -> list[str]:
    """The cards are the batch the measurements here run on. The input boxes hold
    INPUT_WORDS and their pronunciations, the target box holds TARGET_WORDS, no input word
    is its own target, and a header names each side."""
    want = {
        ("input", "orth"): INPUT_WORDS,
        ("input", "phon"): pronunciations(INPUT_WORDS),
        ("target", "orth"): TARGET_WORDS,
    }
    problems = [
        f"{word!r} is both an input and its own target, so the two sides look alike"
        for word, target in zip(INPUT_WORDS, TARGET_WORDS, strict=True)
        if word == target
    ]
    for name, page in load_pages(path).items():
        mid = page.frame_mid_x()
        for box in page.vertices:
            key = word_box(page, box)
            if key not in want:
                continue
            cards = sorted(
                (c for c in page.vertices if c.get("parent") == box.get("parent") and c is not box),
                key=lambda c, page=page: page.visual_box(c.get("id"))[0],
            )
            drawn = [plain(c.get("value")) for c in cards]
            if drawn != want[key]:
                problems.append(
                    f"{name}: the {key[0]} {key[1]} box shows {drawn}, want {want[key]}"
                )
        for header, on_the_left in (("Input", True), ("Target", False)):
            cells = page.labelled(header)
            if len(cells) != 1 or (page.origin(cells[0].get("id"))[0] < mid) != on_the_left:
                side = "left" if on_the_left else "right"
                problems.append(f"{name}: no single {header!r} header on the {side}")
    return problems


def position_problems(path: Path = DIAGRAM) -> list[str]:
    """Each page says where position enters, and says it nowhere else.

    Position is a rotation inside the self-attention of the encoders and of the decoders.
    An embedding carries none, and the mixer and every cross-attention read none (measured).
    So each POSITION_GROUPS label sits once on a page, on the group holding its blocks, and
    no other label, on a block or on an arrow, names position.

    A stage reads order when reversing its sequence moves its output by more than 1e-3 of
    its scale, and reads none when that stays under 1e-5. A deviation between the two is
    neither, and is reported instead of being called one or the other."""
    problems = []
    for stage, (kind, deviation) in reversal_deviations().items():
        reads = deviation > 1e-3
        if 1e-5 <= deviation <= 1e-3:
            problems.append(f"{stage} moves by {deviation:.1e} when reversed: noise or order?")
        elif reads != DRAWN_WITH_POSITION[kind]:
            problems.append(
                f"{stage} {'reads' if reads else 'does not read'} the order of its sequence, "
                f"but the figure draws the {kind} {'without' if reads else 'with'} position"
            )
    for name, page in load_pages(path).items():
        for label, blocks in POSITION_GROUPS.items():
            groups = page.labelled(label)
            held = {
                plain(c.get("value"))
                for group in groups
                for c in page.vertices
                if c.get("parent") == group.get("parent")
            }
            if len(groups) != 1 or not set(blocks) <= held:
                problems.append(
                    f"{name}: {len(groups)} groups labelled {label!r}, holding "
                    f"{sorted(held & set(blocks))} of {list(blocks)}"
                )
        for cell in page.cells.values():
            label = plain(cell.get("value"))
            if "position" in label.lower() and label not in POSITION_GROUPS:
                problems.append(
                    f"{name}: {label!r} names position, which enters only in self-attention"
                )
    return problems


@pytest.mark.parametrize(
    "check",
    [
        page_set_problems,
        opacity_problems,
        mixer_input_problems,
        edge_problems,
        axis_label_problems,
        layout_problems,
        mask_problems,
        global_residual_problems,
        output_table_problems,
        feed_problems,
        word_problems,
        position_problems,
    ],
    ids=lambda f: f.__name__,
)
def test_the_diagram_matches_the_model(check):
    problems = check()
    assert not problems, "BRIDGE-Models.drawio disagrees with the code:\n" + "\n".join(problems)
