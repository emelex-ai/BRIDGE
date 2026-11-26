"""Position is a rotation inside self-attention, and nothing else about a block is new.

``bridge/domain/model/layers.py`` owns the blocks of the two encoders and the two decoders,
because the rotation sits between the query and key projections and the dot product, where
torch's stock layers give no access. Three claims are pinned here, each against its own
oracle.

* **Reference implementation.** With the rotation replaced by the identity, an owned stack
  is torch's stock stack: the same output on the same weights, which load by name. In
  training that includes the dropout, draw for draw. Seeded alike, a fresh owned stack
  also starts from the weights a fresh stock stack starts from.
* **Analytic, on the stacks.** A rotated dot product depends on how far apart two positions
  are and not on where they are. So moving every position by the same amount changes
  nothing, and moving only some of them does. The same holds of one attention handed the
  positions directly. The values are not rotated. Cross-attention is not rotated either,
  so a decoder reads the rows of its memory as a set.
* **Analytic, on** ``rotate`` **itself.** The angle of each pair, worked out by hand.

``rotate`` is the one place position enters, and it is a module-level function. A test
turns the rotation off, or moves every position, by replacing it through ``monkeypatch``.

Floats are compared by ``deviation``: the largest difference over the largest magnitude of
the reference. Measured 2026-11-23 on CPU, torch 2.12.0, one thread, at this file's sizes:

=====================================  ============  ==============================
check                                  honest        the control that must move
=====================================  ============  ==============================
rotation off against the stock stack   0.0, bitwise  rotation on: 2.9e-02
the same against torch's fast path     3.4e-07       rotation on: 2.9e-02
the same in training, dropout on       3.7e-07       another seed: 9.2e-01
every position moved by 1 or 17        4.9e-07       half of them moved: 5.3e-02
every position moved by 1000           1.6e-06       half of them moved: 5.3e-02
one attention, positions 17 on         5.4e-07       twice as far apart: 7.4e-01
memory rows permuted                   4.3e-07       one row replaced: 5.5e-01
=====================================  ============  ==============================

``NOISE`` is 1e-5, twenty times the float noise and over a thousand times under the
smallest control. A move of 1000 gets ``FAR``, 1e-4: the angle ``position * rate`` is held
in float32, whose rounding grows with the position, and at ``d_model=64`` the same move
measured 1.2e-05. With the angle in float64 it measured 1e-14, so that figure is rounding
and not a dependence on where a token sits. The training row is the worst of five seeds,
measured 2026-11-24. In float64 it measured 7e-16, so it is rounding too. The row of one
attention is the worst of five seeds as well, with the smallest of their controls.
"""

import functools
import math

import pytest
import torch
import torch.nn as nn

from bridge.domain.model import layers
from bridge.domain.model.layers import Decoder, Encoder, rotate

D_MODEL, NHEAD, LAYERS = 32, 2, 2
BATCH, LENGTH, MEMORY = 5, 14, 6
NOISE = 1e-5
FAR = 1e-4
# What a control must exceed to count as seen. The smallest measured is 2.9e-02.
SEEN = 1e-3


def deviation(reference: torch.Tensor, other: torch.Tensor) -> float:
    """The largest difference between two tensors, over the reference's largest magnitude."""
    assert reference.shape == other.shape
    reference, other = reference.detach(), other.detach()
    return float((reference - other).abs().max() / reference.abs().max())


def randomized(module: nn.Module) -> nn.Module:
    """``module`` in eval mode with every parameter redrawn, so that biases and norms are
    not at their neutral values and the layers of a stack differ from one another."""
    for parameter in module.parameters():
        nn.init.normal_(parameter, std=0.3)
    return module.eval()


@pytest.fixture
def sample():
    """A batch with two padded rows, a memory for the decoder, and the causal mask."""
    torch.manual_seed(0)
    x = torch.randn(BATCH, LENGTH, D_MODEL)
    memory = torch.randn(BATCH, MEMORY, D_MODEL)
    pad = torch.zeros(BATCH, LENGTH, dtype=torch.bool)
    pad[1, 9:] = True
    pad[3, 5:] = True
    causal = torch.triu(torch.ones(LENGTH, LENGTH, dtype=torch.bool), 1)
    return x, memory, pad, causal


@pytest.fixture
def stacks(sample):
    """An owned encoder and decoder with redrawn weights, and a function running both."""
    x, memory, pad, _ = sample
    torch.manual_seed(1)
    encoder = randomized(Encoder(D_MODEL, NHEAD, LAYERS))
    decoder = randomized(Decoder(D_MODEL, NHEAD, LAYERS))

    def run(memory: torch.Tensor = memory) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            return (
                encoder(x, src_key_padding_mask=pad),
                decoder(x, memory, tgt_key_padding_mask=pad),
            )

    return run


def moved(by) -> object:
    """``rotate`` with every position passed through ``by`` first."""
    real = layers.rotate
    return lambda x, positions: real(x, by(positions))


def stock_and_owned() -> tuple[nn.Module, nn.Module, Encoder, Decoder]:
    """Torch's stock encoder and decoder with redrawn weights, and an owned encoder and
    decoder holding the same weights, loaded by name with ``strict=True``. All in eval mode.
    """
    torch.manual_seed(1)
    stock_layer = nn.TransformerEncoderLayer(D_MODEL, NHEAD, 4 * D_MODEL, batch_first=True)
    stock_encoder = randomized(nn.TransformerEncoder(stock_layer, LAYERS))
    stock_layer = nn.TransformerDecoderLayer(D_MODEL, NHEAD, 4 * D_MODEL, batch_first=True)
    stock_decoder = randomized(nn.TransformerDecoder(stock_layer, LAYERS))
    encoder, decoder = (
        Encoder(D_MODEL, NHEAD, LAYERS).eval(),
        Decoder(D_MODEL, NHEAD, LAYERS).eval(),
    )
    encoder.load_state_dict(stock_encoder.state_dict(), strict=True)
    decoder.load_state_dict(stock_decoder.state_dict(), strict=True)
    return stock_encoder, stock_decoder, encoder, decoder


@pytest.mark.parametrize("grad", [True, False], ids=["ordinary_path", "fast_path"])
def test_an_owned_stack_with_the_rotation_off_is_torchs_stock_stack(monkeypatch, sample, grad):
    """Oracle: reference implementation, ``nn.TransformerEncoder`` and
    ``nn.TransformerDecoder`` on the same weights.

    The weights are the stock stack's own ``state_dict``, loaded with ``strict=True``, so
    this also pins that an owned stack names and shapes its parameters as torch does.
    With grad off torch's encoder takes its fast path, which is what a stock layer runs in
    evaluation; it writes zeros at padded positions, so the real positions are compared.

    The control runs first: with the rotation on, the owned stacks are not the stock ones.
    """
    x, memory, pad, causal = sample
    stock_encoder, stock_decoder, encoder, decoder = stock_and_owned()

    def pairs() -> list[tuple[torch.Tensor, torch.Tensor]]:
        with torch.set_grad_enabled(grad):
            return [
                (
                    stock_encoder(x, src_key_padding_mask=pad)[~pad],
                    encoder(x, src_key_padding_mask=pad)[~pad],
                ),
                (
                    stock_decoder(x, memory, tgt_mask=causal, tgt_key_padding_mask=pad)[~pad],
                    decoder(x, memory, tgt_key_padding_mask=pad)[~pad],
                ),
            ]

    for stock, owned in pairs():
        assert deviation(stock, owned) > SEEN, "control: the rotation changes nothing"

    monkeypatch.setattr(layers, "rotate", lambda x, positions: x)
    for stock, owned in pairs():
        assert deviation(stock, owned) < NOISE


def test_in_training_an_owned_stack_drops_out_as_torchs_stock_stack_does(monkeypatch, sample):
    """Oracle: reference implementation, the stock stacks in train mode with their dropout
    on. Seeded alike, an owned stack makes the same draws: the same number, at the same
    rate, in the same order, applied in the same places. A dropout missing, added or moved
    shifts every draw after it.

    Torch's attention hands its output back as a view of a ``(length, batch, d_model)``
    tensor, and dropout lays its draws out in memory order. The owned attention's output
    is given that layout here, with the same values, so that the draw each element
    receives can be compared and not only how many are made.

    The control: the stock stacks at another seed differ, so dropout is on and is seen.
    """
    x, memory, pad, causal = sample
    stacks = stock_and_owned()
    for stack in stacks:
        stack.train()
    stock_encoder, stock_decoder, encoder, decoder = stacks
    monkeypatch.setattr(layers, "rotate", lambda x, positions: x)
    forward = layers.Attention.forward

    def in_torchs_layout(self, *args, **kwargs):
        return forward(self, *args, **kwargs).transpose(0, 1).contiguous().transpose(0, 1)

    monkeypatch.setattr(layers.Attention, "forward", in_torchs_layout)

    def run(an_encoder, a_decoder, seed: int) -> list[torch.Tensor]:
        torch.manual_seed(seed)
        encoded = an_encoder(x, src_key_padding_mask=pad)[~pad]
        torch.manual_seed(seed)
        decoded = a_decoder(x, memory, tgt_key_padding_mask=pad)[~pad]
        return [encoded, decoded]

    # The stock decoder is told to be causal. The owned one always is.
    stock_causal = functools.partial(stock_decoder, tgt_mask=causal)
    stock = run(stock_encoder, stock_causal, seed=0)
    for reference, owned in zip(stock, run(encoder, decoder, seed=0), strict=True):
        assert deviation(reference, owned) < NOISE

    for reference, other in zip(stock, run(stock_encoder, stock_causal, seed=1), strict=True):
        assert deviation(reference, other) > SEEN, "control: dropout changes nothing"


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_a_fresh_owned_stack_starts_from_the_weights_a_fresh_stock_stack_would(seed):
    """Oracle: reference implementation, torch's stock stacks as they are built. Seeded
    alike, an owned stack holds what a stock stack holds before either is trained, tensor
    for tensor and bitwise: the same names in the same order, the same initial
    distributions drawn in the same order, and one draw shared by the layers of a stack.

    The control: a stock stack at another seed holds other weights.
    """

    def fresh(build, at: int) -> dict[str, torch.Tensor]:
        torch.manual_seed(at)
        return build().state_dict()

    def stock_encoder() -> nn.Module:
        layer = nn.TransformerEncoderLayer(D_MODEL, NHEAD, 4 * D_MODEL, batch_first=True)
        return nn.TransformerEncoder(layer, LAYERS)

    def stock_decoder() -> nn.Module:
        layer = nn.TransformerDecoderLayer(D_MODEL, NHEAD, 4 * D_MODEL, batch_first=True)
        return nn.TransformerDecoder(layer, LAYERS)

    for stock, owned in (
        (stock_encoder, lambda: Encoder(D_MODEL, NHEAD, LAYERS)),
        (stock_decoder, lambda: Decoder(D_MODEL, NHEAD, LAYERS)),
    ):
        reference, ours = fresh(stock, seed), fresh(owned, seed)
        assert list(ours) == list(reference)
        differing = [name for name in reference if not torch.equal(ours[name], reference[name])]
        assert differing == []
        other = fresh(stock, seed + 100)
        assert any(not torch.equal(other[name], reference[name]) for name in reference), "control"


@pytest.mark.parametrize(("shift", "limit"), [(1, NOISE), (17, NOISE), (1000, FAR)])
def test_moving_every_position_by_the_same_amount_changes_nothing(
    monkeypatch, stacks, shift, limit
):
    """Oracle: analytic. Turning a query and a key by angles proportional to their
    positions leaves their dot product a function of the difference of the positions, so
    a constant added to all of them cancels. That holds through a whole encoder and a
    whole decoder, padding and causal mask included."""
    encoder_out, decoder_out = stacks()
    monkeypatch.setattr(layers, "rotate", moved(lambda positions: positions + shift))
    encoder_moved, decoder_moved = stacks()

    assert deviation(encoder_out, encoder_moved) < limit
    assert deviation(decoder_out, decoder_moved) < limit


def test_moving_half_the_positions_changes_the_output(monkeypatch, stacks):
    """The other side of the invariant, and the control that a move is seen at all: the
    second half of the sequence three places further on changes the distance between the
    two halves, and so both outputs."""
    encoder_out, decoder_out = stacks()
    half = LENGTH // 2
    monkeypatch.setattr(layers, "rotate", moved(lambda p: torch.where(p >= half, p + 3, p)))
    encoder_moved, decoder_moved = stacks()

    assert deviation(encoder_out, encoder_moved) > SEEN
    assert deviation(decoder_out, decoder_moved) > SEEN


def test_a_decoder_reads_the_rows_of_its_memory_as_a_set(stacks, sample):
    """Oracle: analytic. Cross-attention is not rotated, so no row of the memory has a
    position, and putting the rows in another order changes nothing a decoder writes.

    The control: one row replaced by other values does change it, so the memory is read.
    """
    _, memory, _, _ = sample
    _, decoder_out = stacks()

    _, permuted = stacks(memory[:, torch.tensor([3, 0, 5, 1, 4, 2])])
    assert deviation(decoder_out, permuted) < NOISE

    replaced = memory.clone()
    replaced[:, 0] = torch.randn(BATCH, D_MODEL)
    _, changed = stacks(replaced)
    assert deviation(decoder_out, changed) > SEEN


def test_self_attention_scores_how_far_apart_two_positions_are():
    """Oracle: analytic, on ``Attention`` itself with nothing replaced. A query and a key
    are turned by their own positions, in the same direction and at the same rate, so
    their product keeps only the difference. Handed positions 17 further on,
    self-attention therefore writes what it wrote. A key turned the other way, or at
    another rate than its query, would leave a sum or a multiple in the product, and the
    17 would show.

    The control: handed positions twice as far apart, it writes something else.
    """
    torch.manual_seed(0)
    attention = randomized(layers.Attention(D_MODEL, NHEAD))
    x = torch.randn(BATCH, LENGTH, D_MODEL)
    positions = torch.arange(LENGTH)

    with torch.no_grad():
        here = attention(x, x, None, positions)
        further = attention(x, x, None, positions + 17)
        spread = attention(x, x, None, 2 * positions)

    assert deviation(here, further) < NOISE
    assert deviation(here, spread) > SEEN


def test_rotate_turns_each_pair_by_its_own_rate_times_the_position():
    """Oracle: analytic, by hand. Four coordinates are two pairs. The first turns one
    radian per position and the second ``10000 ** -0.5``, a hundredth of that. The vector
    ``(1, 0, 1, 0)`` at position ``p`` therefore becomes
    ``(cos p, sin p, cos 0.01p, sin 0.01p)``, and at position 0 it is unchanged."""
    positions = torch.tensor([0, 1, 2, 50])
    x = torch.tensor([1.0, 0.0, 1.0, 0.0]).expand(1, 1, 4, 4)

    turned = rotate(x, positions)[0, 0]

    expected = torch.tensor(
        [[math.cos(p), math.sin(p), math.cos(0.01 * p), math.sin(0.01 * p)] for p in (0, 1, 2, 50)]
    )
    assert torch.allclose(turned, expected, atol=1e-6)
    assert torch.equal(turned[0], x[0, 0, 0])


def test_a_rotated_dot_product_depends_only_on_the_distance():
    """Oracle: analytic. A query at ``i`` against a key at ``j`` scores the same wherever
    the pair sits, as long as ``i - j`` is the same. Three placements of a distance of
    three agree, and a distance of four does not."""
    torch.manual_seed(0)
    q, k = torch.randn(1, 1, 1, 8), torch.randn(1, 1, 1, 8)

    def score(i: int, j: int) -> float:
        return float((rotate(q, torch.tensor([i])) * rotate(k, torch.tensor([j]))).sum())

    three_apart = [score(3, 0), score(13, 10), score(503, 500)]
    assert max(three_apart) - min(three_apart) < 1e-4 * abs(three_apart[0])
    assert abs(score(4, 0) - three_apart[0]) > 1e-2 * abs(three_apart[0])


def test_rotation_keeps_the_length_of_every_vector_and_its_dtype():
    """A rotation moves a vector without stretching it, in whatever precision it is in."""
    torch.manual_seed(0)
    x = torch.randn(2, 3, 7, 8)
    turned = rotate(x, torch.arange(7))
    assert torch.allclose(turned.norm(dim=-1), x.norm(dim=-1), rtol=1e-5)
    for dtype in (torch.float16, torch.float64):
        assert rotate(x.to(dtype), torch.arange(7)).dtype == dtype


@pytest.mark.parametrize("head_dim", [3, 5])
def test_an_odd_head_dimension_keeps_its_last_coordinate(head_dim):
    """Oracle: analytic, by hand. Coordinates turn in pairs, so an odd count leaves one
    over. It is the last, and it passes through unchanged. The ``n`` pairs before it turn
    as a head of ``2n`` would turn them: pair ``i`` by ``position * 10000 ** (-i / n)``.
    A pair ``(1, 1)`` turned by ``a`` is ``(cos a - sin a, sin a + cos a)``."""
    pairs = head_dim // 2
    places = (0, 1, 2, 50)
    x = torch.ones(1, 1, len(places), head_dim)

    turned = rotate(x, torch.tensor(places))

    expected = torch.ones(len(places), head_dim)
    for row, place in enumerate(places):
        for i in range(pairs):
            a = place * 10000 ** (-i / pairs)
            expected[row, 2 * i] = math.cos(a) - math.sin(a)
            expected[row, 2 * i + 1] = math.sin(a) + math.cos(a)
    assert torch.allclose(turned[0, 0], expected, atol=1e-6)
    assert torch.equal(turned[..., -1], x[..., -1])


def test_a_head_of_one_dimension_is_not_turned_at_all():
    """One coordinate is no pair, so ``rotate`` hands it back as it is at every position.
    A model of such heads could not read order, which is why ``ModelConfig`` refuses one."""
    torch.manual_seed(0)
    x = torch.randn(2, 2, 6, 1)
    assert torch.equal(rotate(x, torch.arange(6)), x)
