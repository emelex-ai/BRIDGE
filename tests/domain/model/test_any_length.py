"""A sequence of any length runs, and it means the same thing however it is run.

Position is a rotation inside self-attention, computed for whatever length arrives. Nothing
in the model is indexed by position, so nothing bounds a sequence. That is one claim. The
others are what any way of holding position has to satisfy, and they are where one goes
wrong in silence: the model is run in several ways that must agree, and each way hands
positions to the decoder by a different route.

==================================  ==========================================  =========
the two ways that must agree        what a disagreement would be                oracle
==================================  ==========================================  =========
every position moved, or not        attention reading a slot, not a distance    analytic
what each stack hands the           a stack reading no position, or one that    analytic
rotation, and 0 to length - 1       stops counting, skips or runs backwards
eval mode, and train mode with      position handled differently where          invariant
dropout off                         training and evaluation part
generating a step at a time, and    the decoder placing a token differently     invariant
decoding the result in one pass     in the loop than under teacher forcing
a row in a padded batch, and the    padding read as content, or a position      invariant
row alone                           counted from the wrong end
==================================  ==========================================  =========

The last claim is the plain one: rows past sixty tokens train, evaluate and generate on
every pathway.

Floats are compared by ``deviation``, the largest difference over the largest magnitude of
the reference, and probabilities by their largest absolute difference. Every integer
decision is compared exactly: an argmax, a feature being on, a generated id. Measured
2026-11-23 on CPU, torch 2.12.0, one thread, on this file's model and rows:

====================================  =======  ==============================
check                                 honest   the control that must move
====================================  =======  ==============================
every position moved by 17            3.4e-07  some of them moved: 6.5e-02
eval against train, dropout off       4.5e-07  dropout left on: 4.1e-01
a step at a time against one pass     1.2e-07  none in the test; see below
a padded batch against a row alone    3.3e-07  none in the test; see below
====================================  =======  ==============================

``NOISE`` is 1e-5, over twenty times the largest honest figure. The last two rows have no
control inside the test, because the only thing that moves them is a defect. Each defect
was planted in the code, where it fails these tests, and measured on a trained model of
``d_model=64``: a decoder that places tokens differently in the loop moved the third by
7.7e-01, and encoders that read their padding moved the fourth by 4.1e-01.

**Generated ids are compared exactly, which is safe here by measurement.** Batching moves
the encoding by 7e-07, and an argmax within that of a tie could flip for a reason that has
nothing to do with padding. On this model, these rows and this cap the smallest
orthographic argmax margin is 2.6e-04 and the smallest ``|P(ON) - 0.5|`` is 2.6e-04, over
all five pathways, about 360 times the perturbation. The seed was picked for that out of
forty. If a change to the rows or the model makes the id comparison fail by one token,
suspect that margin first.
"""

import copy
import math

import pytest
import torch

from bridge.application.training import TrainingPipeline
from bridge.domain.data import BridgeDataset
from bridge.domain.datamodels import (
    BridgeBatch,
    DatasetConfig,
    ModelConfig,
    TrainingConfig,
    VocabSpec,
)
from bridge.domain.datamodels.pathways import PATHWAY_INPUTS, PATHWAY_IO, PATHWAYS
from bridge.domain.model import Model, layers
from tests.conftest import batch_tensors

# Four rows of unequal length, so the batch is padded, and the longest is past 30 tokens.
ROWS = ["a", "the cat sat", "elephant", "the dog and the cat sat on the long pencil"]
# What each row is taught to write: another string, of another length.
TARGETS = ["the long cat", "a", "the dog sat on the elephant and the pencil", "cat"]
LANGUAGES = ["EN"] * len(ROWS)
CAP = 12  # the most tokens a generated row writes after its start
NOISE = 1e-5
SEEN = 1e-3  # what the control of a check must exceed

# Two examples whose input and target are each past sixty tokens in both modalities.
LONG = [
    (
        "the elephant and the long pencil sat on the computer and the dog sat on it",
        "a computer and a pencil and a long elephant sat on the dog and the cat",
    ),
    (
        "the long dog and the elephant sat on a pencil and the computer sat on a cat",
        "a cat and a dog sat on the long computer and the elephant sat on a pencil",
    ),
]


def deviation(reference: torch.Tensor, other: torch.Tensor) -> float:
    """The largest difference between two tensors, over the reference's largest magnitude."""
    assert reference.shape == other.shape
    return float((reference - other).abs().max() / reference.abs().max())


@pytest.fixture(scope="module")
def vocab(shared_tokenizer):
    return VocabSpec.from_tokenizer(shared_tokenizer)


@pytest.fixture(scope="module")
def model(vocab):
    return Model(ModelConfig(vocab=vocab, d_model=32, nhead=2, seed=17)).eval()


@pytest.fixture(scope="module")
def batch(shared_tokenizer):
    batch = BridgeBatch(
        shared_tokenizer.encode_input(ROWS, LANGUAGES),
        shared_tokenizer.encode_target(TARGETS, LANGUAGES),
    )
    for modality in ("orth", "phon"):
        tokens = getattr(batch.input, modality)
        assert tokens.ids.shape[1] > 30, "control: the longest row must be past 30 tokens"
        assert bool(tokens.pad_mask.any()), "control: the batch must be padded"
    return batch


@pytest.fixture(scope="module")
def start(shared_tokenizer):
    return shared_tokenizer.orth_start(LANGUAGES)


def alone(shared_tokenizer, row: int) -> BridgeBatch:
    """Row ``row`` of the batch as a batch of its own, with no padding."""
    return BridgeBatch(
        shared_tokenizer.encode_input([ROWS[row]], ["EN"]),
        shared_tokenizer.encode_target([TARGETS[row]], ["EN"]),
    )


def logits(model: Model, pathway: str, batch: BridgeBatch, train: bool = False) -> dict:
    """Teacher-forced logits of ``pathway``: the encoders read the input and the decoders
    are fed the target, as ``TrainingPipeline.forward`` wires them."""
    kwargs = {}
    for modality, side in PATHWAY_INPUTS[pathway]:
        part = getattr(batch.input if side == "enc" else batch.target, modality)
        kwargs[f"{modality}_{side}_input"] = part.ids
        kwargs[f"{modality}_{side}_pad_mask"] = part.pad_mask
    model.train(train)
    try:
        with torch.no_grad():
            return model(task=pathway, **kwargs)
    finally:
        model.eval()


def generate(model: Model, pathway: str, inputs, start: torch.Tensor):
    return model.generate(inputs, pathway, max_new_tokens=CAP, orth_start=start, deterministic=True)


def phon_features(result, row: int) -> list[tuple[int, ...]]:
    return [tuple(sorted(t.tolist())) for t in result.phon_tokens[row]]


@pytest.mark.parametrize("pathway", PATHWAYS)
def test_moving_every_position_changes_no_logit(monkeypatch, model, batch, pathway):
    """Oracle: analytic. Attention reads the distance between two tokens, so adding 17 to
    every position, in both encoders and both decoders at once, moves no logit past float
    noise and changes no argmax. The mixer and cross-attention have no position to move.

    The control moves only the positions from 3 on, three places further, which changes
    the distance across that point and must change every output.
    """
    real = layers.rotate
    base = logits(model, pathway, batch)

    monkeypatch.setattr(layers, "rotate", lambda x, positions: real(x, positions + 17))
    moved = logits(model, pathway, batch)
    for key, reference in base.items():
        assert deviation(reference, moved[key]) < NOISE, key
        assert torch.equal(reference.argmax(1), moved[key].argmax(1)), key

    monkeypatch.setattr(
        layers,
        "rotate",
        lambda x, positions: real(x, torch.where(positions >= 3, positions + 3, positions)),
    )
    split = logits(model, pathway, batch)
    for key, reference in base.items():
        assert deviation(reference, split[key]) > SEEN, f"control: {key} did not move"


@pytest.mark.parametrize("pathway", PATHWAYS)
def test_every_self_attention_turns_its_queries_and_keys_at_positions_zero_to_length(
    monkeypatch, shared_tokenizer, vocab, pathway
):
    """Oracle: analytic, exact. ``rotate`` is the one place position enters, so what it is
    handed says where position is read. On a teacher-forced forward, every layer of every
    stack the pathway runs turns two tensors, its queries and its keys, each
    ``(batch, nhead, length, head_dim)``, at positions ``0, 1, ..., length - 1``. Nothing
    else is turned: not the values, not cross-attention, not the mixer.

    The four stacks have 1, 2, 3 and 4 layers, so a stack that reads no position, or
    reads it in some of its layers, cannot hide behind another. The control comes first:
    every sequence is past 30 tokens, so positions that stop counting at some length are
    seen as well.
    """
    d_model, nhead = 32, 4
    stack_layers = {("orth", "enc"): 1, ("phon", "enc"): 2, ("orth", "dec"): 3, ("phon", "dec"): 4}
    model = Model(
        ModelConfig(
            vocab=vocab,
            d_model=d_model,
            nhead=nhead,
            num_orth_enc_layers=1,
            num_phon_enc_layers=2,
            num_orth_dec_layers=3,
            num_phon_dec_layers=4,
            seed=1,
        )
    )
    languages = ["EN"] * len(LONG)
    batch = BridgeBatch(
        shared_tokenizer.encode_input([given for given, _ in LONG], languages),
        shared_tokenizer.encode_target([taught for _, taught in LONG], languages),
    )
    turned = []
    real = layers.rotate

    def recording(x, positions):
        # One entry per tensor turned, however many of them a call stacks together.
        each = (tuple(x.shape[-4:]), positions.tolist())
        turned.extend([each] * math.prod(x.shape[:-4]))
        return real(x, positions)

    monkeypatch.setattr(layers, "rotate", recording)
    expected = []
    for modality, side in PATHWAY_INPUTS[pathway]:
        length = getattr(batch.input if side == "enc" else batch.target, modality).ids.shape[1]
        assert length > 30, f"control: the {modality} {side} sequence is only {length} long"
        each = ((len(LONG), nhead, length, d_model // nhead), list(range(length)))
        expected += [each] * 2 * stack_layers[modality, side]

    logits(model, pathway, batch)

    assert sorted(turned) == sorted(expected)


@pytest.mark.parametrize("pathway", PATHWAYS)
def test_eval_and_train_agree_once_dropout_is_off(model, batch, pathway):
    """Oracle: invariant. With every dropout at zero the two modes are one function, so
    what a model is trained on is what it is evaluated on. Only torch's own mixer takes a
    different code path between them, which is float noise."""
    quiet = copy.deepcopy(model)
    for module in quiet.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0
        if isinstance(module, torch.nn.MultiheadAttention | layers.Attention):
            module.dropout = 0.0

    evaluated = logits(quiet, pathway, batch)
    trained = logits(quiet, pathway, batch, train=True)

    for key, reference in trained.items():
        assert deviation(reference, evaluated[key]) < NOISE, key
        assert torch.equal(reference.argmax(1), evaluated[key].argmax(1)), key

    noisy = logits(model, pathway, batch, train=True)
    for key, reference in trained.items():
        assert deviation(reference, noisy[key]) > SEEN, "control: dropout changes nothing"


@pytest.mark.parametrize("pathway", PATHWAYS)
def test_generating_step_by_step_equals_decoding_the_result_at_once(model, batch, start, pathway):
    """Oracle: invariant. Each distribution the loop sampled from, while the sequence was
    still growing, equals the one a single teacher-forced pass over the finished sequence
    gives at that position. So a token sits where training would have put it, at every
    length the loop passes through.

    The pass is given a padding mask and the loop is not, which are the two ways the
    decoder is called, and they must place tokens alike.
    """
    result = generate(model, pathway, batch.input, start)
    memory = result.global_encoding
    compared = 0
    with torch.no_grad():
        if "orth" in PATHWAY_IO[pathway][1]:
            tokens = result.orth_tokens
            at_once = torch.softmax(
                model._decode_orth(memory, tokens, torch.zeros_like(tokens, dtype=torch.bool)),
                dim=1,
            )
            for row, history in enumerate(result.orth_probs):
                # Position 1 + k of the pass predicts the token the loop wrote at 2 + k.
                for k, step in enumerate(history[2:]):
                    assert float((step - at_once[row, :, 1 + k]).abs().max()) < NOISE
                    assert int(step.argmax()) == int(at_once[row, :, 1 + k].argmax())
                    compared += 1
        if "phon" in PATHWAY_IO[pathway][1]:
            length = len(result.phon_tokens[0])
            multihot = torch.zeros(
                (len(ROWS), length, model.phonological_vocabulary_size), dtype=torch.long
            )
            for row, positions in enumerate(result.phon_tokens):
                for position, features in enumerate(positions):
                    multihot[row, position, features] = 1
            decoded = model.phonology_decoder(
                tgt=model.embed_phon_vectors(multihot),
                memory=memory,
                tgt_key_padding_mask=torch.zeros((len(ROWS), length), dtype=torch.bool),
            )
            on = torch.softmax(
                model.linear_phonology_decoder(decoded).view(len(ROWS), length, 2, -1), dim=2
            )[:, :, 1]
            for row, history in enumerate(result.phon_probs):
                for k, step in enumerate(history):
                    assert float((step - on[row, k]).abs().max()) < NOISE
                    assert torch.equal(step > 0.5, on[row, k] > 0.5)
                    compared += 1
    assert compared >= len(ROWS), "control: nothing was generated, so nothing was compared"


@pytest.mark.parametrize("pathway", PATHWAYS)
def test_a_padded_batch_scores_each_row_as_the_row_alone(model, shared_tokenizer, batch, pathway):
    """Oracle: invariant. A row's logits at its own positions are what the row gets with
    no other row beside it and no padding after it, to float noise, with the same argmax
    everywhere. The rows differ in length on both sides, so three of four are padded."""
    batched = logits(model, pathway, batch)
    for row in range(len(ROWS)):
        for key, single in logits(model, pathway, alone(shared_tokenizer, row)).items():
            width = single.shape[-1] if key == "orth" else single.shape[2]
            beside = batched[key][row : row + 1, :, :width]
            assert deviation(single, beside) < NOISE, (key, row)
            assert torch.equal(single.argmax(1), beside.argmax(1)), (key, row)


@pytest.mark.parametrize("pathway", PATHWAYS)
def test_a_padded_batch_generates_each_row_as_the_row_alone(
    model, shared_tokenizer, vocab, batch, start, pathway
):
    """Oracle: invariant, on exact ids. Each row generates, up to and including its
    ``[EOS]``, the tokens it generates alone. See the module docstring for why an exact
    comparison is safe on this model."""
    batched = generate(model, pathway, batch.input, start)
    for row in range(len(ROWS)):
        single = generate(model, pathway, alone(shared_tokenizer, row).input, start[row : row + 1])
        if "orth" in PATHWAY_IO[pathway][1]:
            ours, theirs = single.orth_tokens[0].tolist(), batched.orth_tokens[row].tolist()
            eos = vocab.orth_eos_id
            end = ours.index(eos) + 1 if eos in ours else len(ours)
            assert theirs[:end] == ours[:end], (pathway, ROWS[row])
            assert len(ours) > 2
        if "phon" in PATHWAY_IO[pathway][1]:
            ours, theirs = phon_features(single, 0), phon_features(batched, row)
            ends = [i for i, features in enumerate(ours) if vocab.phon_eos_id in features]
            end = ends[0] + 1 if ends else len(ours)
            assert theirs[:end] == ours[:end], (pathway, ROWS[row])
            assert len(ours) > 1


@pytest.fixture(scope="module")
def long_dataset(tmp_path_factory, shared_tokenizer):
    path = tmp_path_factory.mktemp("long") / "long.csv"
    path.write_text(
        "word_raw,language,target_raw,target_language\n"
        + "".join(f"{given},EN,{taught},EN\n" for given, taught in LONG)
    )
    return BridgeDataset(DatasetConfig(dataset_filepath=str(path)), tokenizer=shared_tokenizer)


@pytest.mark.parametrize("pathway", PATHWAYS)
def test_rows_past_sixty_tokens_train_evaluate_and_generate(
    monkeypatch, long_dataset, shared_tokenizer, vocab, pathway
):
    """Oracle: invariant. Any length runs, on every pathway. The control comes first:
    every tensor of the batch, input and target, is at least 60 tokens wide.

    Training is one optimizer step with a finite loss that moves the weights. Evaluation
    is finite losses and metrics. Generation writes 60 tokens after the start: the
    samplers are scripted never to emit ``[EOS]``, so the decoder is run at every length
    up to 62, where an untrained model might stop early by chance.
    """
    rows = slice(0, len(LONG))
    batch = long_dataset[rows]
    for name, tensor in batch_tensors(batch).items():
        assert tensor.shape[1] >= 60, f"control: {name} is only {tensor.shape[1]} wide"

    model = Model(ModelConfig(vocab=vocab, d_model=16, nhead=2, seed=5))
    pipeline = TrainingPipeline(model, TrainingConfig(training_pathway=pathway))
    before = [parameter.detach().clone() for parameter in model.parameters()]

    stepped = pipeline.single_step(long_dataset, rows, calculate_metrics=True)
    assert math.isfinite(float(stepped["loss"]))
    assert any(
        not torch.equal(old, new) for old, new in zip(before, model.parameters(), strict=True)
    )

    scored = pipeline.evaluate(long_dataset, rows)
    numbers = {key: float(value) for key, value in scored.items() if key not in ("input", "target")}
    assert all(math.isfinite(value) for value in numbers.values()), numbers

    letter = shared_tokenizer.char_tokenizer.char_2_idx["z"]
    monkeypatch.setattr(
        model, "ortho_sample", lambda probs, deterministic: torch.full((len(LONG), 1), letter)
    )

    def one_feature(probs, deterministic):
        embedding_input = torch.zeros((len(LONG), vocab.phon_vocab_size), dtype=torch.long)
        embedding_input[:, 0] = 1
        return embedding_input[:, :-1], embedding_input

    monkeypatch.setattr(model, "phono_sample", one_feature)
    written = model.generate(
        batch.input,
        pathway,
        max_new_tokens=60,
        orth_start=shared_tokenizer.orth_start(["EN"] * len(LONG)),
        deterministic=True,
    )
    if "orth" in PATHWAY_IO[pathway][1]:
        assert written.orth_tokens.shape == (len(LONG), 62)
    if "phon" in PATHWAY_IO[pathway][1]:
        assert [len(row) for row in written.phon_tokens] == [61] * len(LONG)
