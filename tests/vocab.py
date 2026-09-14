"""A ``VocabSpec`` consistent with the real phonological feature table.

Model tests used to hand-write ``phon_vocab_size=34`` with invented special-token ids.
Since issue #221 the model derives phoneme embeddings from ``phonreps.csv``, so the
phonological half of a ``VocabSpec`` is no longer free: ``Model.__init__`` rejects a spec
that disagrees with the table. The orthographic half still is, and stays hand-written.

Import the constant rather than a fixture: several tests reference it at module level
inside ``parametrize`` decorators, which run before fixtures exist.
"""

from bridge.core.phonreps import load_phoneme_table
from bridge.domain.datamodels import VocabSpec

PHONEME_TABLE = load_phoneme_table()

TEST_VOCAB = VocabSpec(
    orth_vocab_size=49,
    phon_vocab_size=PHONEME_TABLE.vocab_size,
    orth_pad_id=2,
    orth_bos_id=0,
    orth_eos_id=1,
    phon_pad_id=PHONEME_TABLE.feature_of("[PAD]"),
    phon_bos_id=PHONEME_TABLE.feature_of("[BOS]"),
    phon_eos_id=PHONEME_TABLE.feature_of("[EOS]"),
    phon_table_fingerprint=PHONEME_TABLE.fingerprint,
)
