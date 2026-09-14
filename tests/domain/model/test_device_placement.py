"""A model must live where it says it lives (issue #229).

``Model.__init__`` used to store ``self.device = device_manager.device`` and then create
exactly one of its 169 parameters there. The other 168 took torch's default. A model built
while the manager pointed at a GPU therefore reported ``cuda:0`` while almost all of it sat
on the CPU, and generation failed with a device mismatch until the caller separately called
``.to()``.

``.to()`` did not update the stored attribute either, so moving a model somewhere other than
``device_manager.device`` left it building every causal mask and generation buffer on the
device it used to be on. That is the same failure as #223 from the opposite direction.

Two properties are asserted here and they are different claims. *Consistency*: every
parameter and buffer is on one device and it is the reported one. *Authority*: ``.to()``
decides, so the report follows the module rather than a snapshot taken at construction.

Most of this runs without a GPU. ``meta`` is a real torch device that needs no hardware and
is not the default, so it exercises the placement branch on any host; without it the whole
file would pass on CI against a model that places nothing, since everything would trivially
be on the CPU already.
"""

import pytest
import torch

from bridge.domain.datamodels import ModelConfig, VocabSpec
from bridge.domain.model import Model
from bridge.domain.tokenizer import BridgeTokenizer
from bridge.utils import device_manager
from tests.vocab import TEST_VOCAB

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")


def build(d_model=32, nhead=2, seed=5):
    return Model(ModelConfig(vocab=TEST_VOCAB, d_model=d_model, nhead=nhead, seed=seed))


def placement(model):
    """Every distinct device the model's tensors are on, parameters and buffers alike."""
    return sorted({str(t.device) for t in [*model.parameters(), *model.buffers()]})


@pytest.mark.parametrize("requested", ["meta", "cpu"])
def test_a_built_model_is_entirely_on_the_requested_device(requested):
    """Consistency. One device, and it is the one that was asked for.

    ``meta`` carries this test on a host with no GPU. It is not torch's default, so a model
    that placed only some of itself would show two devices here, which is exactly what the
    defect looked like: ``['cpu', 'cuda:0']`` against a claimed ``cuda:0``.
    """
    device_manager.set_device(requested)
    model = build()

    assert placement(model) == [requested], (
        f"model spans {placement(model)} after asking for {requested}"
    )
    assert str(model.device) == requested


def test_the_whole_module_moves_and_the_report_follows():
    """Authority. ``.to()`` decides where the model is, and ``device`` reports it.

    The stored attribute could not do this: it was a snapshot of ``device_manager.device``
    and ``nn.Module.to`` knows nothing about it.
    """
    device_manager.set_device("cpu")
    model = build()
    assert str(model.device) == "cpu"

    model.to("meta")

    assert placement(model) == ["meta"]
    assert str(model.device) == "meta", "device must follow the module, not the manager"
    assert str(device_manager.device) == "cpu", "the manager itself must not have moved"


def test_device_is_derived_and_cannot_be_set_out_of_sync():
    """A writable attribute is how the report and the module drifted apart.

    Assigning to it must fail rather than silently produce a model that lies about itself.
    """
    model = build()
    with pytest.raises(AttributeError):
        model.device = torch.device("meta")


def test_parameters_are_counted_so_a_new_submodule_cannot_be_missed():
    """The placement assertions above are only meaningful over everything the model owns.

    169 parameters and 1 buffer at this configuration. The count is pinned so that a
    submodule added later, which would be the way this defect returns, shows up here as a
    changed number rather than slipping past a set comparison that never saw it.
    """
    device_manager.set_device("cpu")
    model = build()

    assert sum(1 for _ in model.parameters()) == 169
    assert sum(1 for _ in model.buffers()) == 1
    assert [name for name, _ in model.named_buffers()] == ["phon_feature_matrix"]


@needs_cuda
def test_a_cuda_model_needs_no_explicit_to_from_the_caller():
    """The reported symptom, end to end.

    Before the fix this raised `Expected all tensors to be on the same device, but got index
    is on cuda:0, different from other tensors on cpu` on every pathway, because the weights
    were on the CPU while the runtime tensors followed the reported device.
    """
    device_manager.set_device("cuda")
    tokenizer = BridgeTokenizer()
    encoding = tokenizer.encode(["cat", "dog"])
    assert encoding is not None
    model = build_from(tokenizer)

    assert placement(model) == [str(device_manager.device)]

    for pathway in ("o2p", "op2op", "p2o", "p2p", "o2o"):
        output = model.generate(encoding, pathway, deterministic=True)
        assert output.global_encoding.device.type == "cuda"


def build_from(tokenizer):
    return Model(
        ModelConfig(vocab=VocabSpec.from_tokenizer(tokenizer), d_model=32, nhead=2, seed=5)
    )


@needs_cuda
def test_the_same_seed_gives_the_same_weights_on_cpu_and_cuda():
    """Initialisation stopped depending on where the model will run.

    Drawing on the default device and then moving means one generator produces the weights,
    so a seed means the same thing everywhere. Previously ``global_embedding`` alone was
    drawn with ``device=cuda`` and so came from the CUDA generator, making that one
    parameter differ between a CPU run and a GPU run of the same seed.

    Bitwise, not approximate: a seeded generator is exactly reproducible.
    """
    device_manager.set_device("cpu")
    on_cpu = dict(build().named_parameters())
    device_manager.set_device("cuda")
    on_cuda = dict(build().named_parameters())

    assert set(on_cpu) == set(on_cuda)
    differing = [
        name
        for name, tensor in on_cpu.items()
        if not torch.equal(tensor.detach().cpu(), on_cuda[name].detach().cpu())
    ]
    assert not differing, f"{len(differing)} parameters differ across devices: {differing[:5]}"
