"""What each pathway reads and what it writes.

The single definition. Every "which pathways use orthography" question in the codebase is
a query against this table, not a literal tuple of its own. See ``docs/decisions/0011``.

It lives in ``datamodels``, not in ``model.py``, because ``TrainingConfig`` needs
``Pathway`` too. Importing it from ``model.py`` closes a cycle:
``datamodels/__init__`` -> ``training_config`` -> ``model`` -> ``datamodels``.
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
WRITES_ORTH = tuple(p for p in PATHWAYS if "orth" in PATHWAY_IO[p][1])
WRITES_PHON = tuple(p for p in PATHWAYS if "phon" in PATHWAY_IO[p][1])

# The tensors a training forward needs: a modality that is read supplies encoder input,
# and one that is written supplies decoder input.
PATHWAY_INPUTS: dict[Pathway, tuple[tuple[str, str], ...]] = {
    pathway: tuple(
        [(m, "enc") for m in MODALITIES if m in reads]
        + [(m, "dec") for m in MODALITIES if m in writes]
    )
    for pathway, (reads, writes) in PATHWAY_IO.items()
}
