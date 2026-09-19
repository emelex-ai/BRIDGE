"""What each pathway reads and what it writes.

The single definition. Every "which pathways use orthography" question in the codebase is
a query against this table rather than its own literal tuple, which is how two of them
came to be written one line apart in different orders, one gating the loss and the other
the metrics. See ``docs/decisions/0011``.

It lives in ``datamodels`` rather than in ``model.py`` because it is data about modality
structure that both the model and ``TrainingConfig`` need. With it in ``model.py``,
``TrainingConfig`` importing ``Pathway`` closed a cycle
(``datamodels/__init__`` -> ``training_config`` -> ``model`` -> ``datamodels``) that
happened to resolve only because of the order of the names in ``model.py``'s import line;
adding one more name to it broke ``import bridge`` outright.
"""

from typing import Literal

Pathway = Literal["o2p", "p2o", "op2op", "p2p", "o2o"]
PATHWAYS: tuple[Pathway, ...] = ("o2p", "p2o", "op2op", "p2p", "o2o")

MODALITIES: tuple[str, ...] = ("orth", "phon")

# pathway -> (modalities read, modalities written)
PATHWAY_IO: dict[Pathway, tuple[frozenset[str], frozenset[str]]] = {
    "o2p": (frozenset({"orth"}), frozenset({"phon"})),
    "p2o": (frozenset({"phon"}), frozenset({"orth"})),
    "o2o": (frozenset({"orth"}), frozenset({"orth"})),
    "p2p": (frozenset({"phon"}), frozenset({"phon"})),
    "op2op": (frozenset({"orth", "phon"}), frozenset({"orth", "phon"})),
}

READS_ORTH = tuple(p for p in PATHWAYS if "orth" in PATHWAY_IO[p][0])
READS_PHON = tuple(p for p in PATHWAYS if "phon" in PATHWAY_IO[p][0])
# Pathways whose second half is orthography, so they run the orthographic decoder loop.
WRITES_ORTH = tuple(p for p in PATHWAYS if "orth" in PATHWAY_IO[p][1])
WRITES_PHON = tuple(p for p in PATHWAYS if "phon" in PATHWAY_IO[p][1])

# The encoder/decoder tensors a training forward needs, derived rather than restated: a
# modality that is read supplies encoder input, one that is written supplies decoder input.
PATHWAY_INPUTS: dict[Pathway, tuple[tuple[str, str], ...]] = {
    pathway: tuple(
        [(m, "enc") for m in MODALITIES if m in reads]
        + [(m, "dec") for m in MODALITIES if m in writes]
    )
    for pathway, (reads, writes) in PATHWAY_IO.items()
}
