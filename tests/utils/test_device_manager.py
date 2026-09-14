"""``DeviceManager`` must name the device torch actually allocates on (issue #223).

``torch.device("cuda")`` and ``torch.device("cuda:0")`` are different objects and compare
unequal, yet every tensor torch allocates for ``device="cuda"`` comes back carrying the
index. A manager that stores the request verbatim therefore holds a device that no tensor
in the process will ever match, and ``Model._validate_device`` compares with ``!=``, so it
rejects input that is on exactly the right GPU.

The damage looked pathway-specific rather than device-specific because only three of the
five pathways run a device check at all: ``o2p`` and ``op2op`` never call
``_validate_device``, so they generated happily on an indexless device while ``p2o``,
``p2p`` and ``o2o`` raised. Those two are the control here.

The second half of the issue is that there is no supported way to point the process at a
different device: no setter, no environment variable, only assignment to a private
attribute. ``set_device`` and ``BRIDGE_DEVICE`` are that supported way.
"""

import importlib.util
import logging
import os

import pytest
import torch

import bridge.domain.model.model as model_module
from bridge.domain.datamodels import ModelConfig
from bridge.domain.model import Model
from bridge.domain.tokenizer import BridgeTokenizer
from bridge.utils import DeviceManager, get_project_root
from tests.vocab import TEST_VOCAB

LOGGER = "bridge.utils.device_manager"
WORDS = ["cat", "dog"]

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")


@needs_cuda
def test_requesting_cuda_names_the_device_torch_allocates_on():
    """The invariant: the manager's device is the device tensors actually land on.

    Oracle is torch itself. ``torch.zeros(device="cuda")`` reports where the allocation
    went, and that reading is independent of anything ``DeviceManager`` computed. The
    manager exists to describe the process device, so any device it reports that a real
    allocation does not match is a wrong answer, whatever it prints.

    The control is the bare ``torch.device("cuda")``: it must come out unequal to the same
    allocation, which is what proves the comparison can tell the two apart and that the
    index is the whole difference.
    """
    allocated = torch.zeros(2, 2, device="cuda").device
    manager = DeviceManager("cuda")

    assert torch.device("cuda") != allocated, "control: an indexless cuda never matches"
    assert manager.device.index is not None, f"no index resolved: {manager.device}"
    assert manager.device == allocated, f"{manager.device} is not where torch allocated"


@needs_cuda
def test_every_pathway_generates_on_a_cuda_device_requested_without_an_index(monkeypatch):
    """End to end: asking for "cuda" must not make three of the five pathways unusable.

    ``Model`` reads ``device_manager.device`` once, in ``__init__``, so the manager has to
    be in place before the model is built. Everything then really runs on the GPU: the
    parameters, the encoding and the decode loops.

    ``o2p`` and ``op2op`` are the control. They reach the same decoders over the same
    tensors but skip ``_validate_device``, so they generate today and must keep generating.
    If they were to fail too, the failure would be something about CUDA generally rather
    than about the device comparison, and this test would not be evidence for #223.
    """
    manager = DeviceManager("cuda")
    monkeypatch.setattr(model_module, "device_manager", manager)

    model = Model(ModelConfig(vocab=TEST_VOCAB, d_model=32, nhead=2, seed=5))
    model.to(manager.device)
    encoding = BridgeTokenizer().encode(WORDS)
    assert encoding is not None
    encoding = encoding.to(manager.device)

    assert next(model.parameters()).device.type == "cuda"
    assert encoding.phonological.enc_pad_mask.device.type == "cuda"

    outcomes = {}
    for pathway in ("o2p", "op2op", "p2o", "p2p", "o2o"):
        try:
            output = model.generate(encoding, pathway, deterministic=True)
        except Exception as exc:
            outcomes[pathway] = f"{type(exc).__name__}: {exc}"
        else:
            outcomes[pathway] = None
            assert output.global_encoding.shape[0] == len(WORDS)

    for control in ("o2p", "op2op"):
        assert outcomes[control] is None, f"control {control} failed: {outcomes[control]}"
    for pathway in ("p2o", "p2p", "o2o"):
        assert outcomes[pathway] is None, f"{pathway}: {outcomes[pathway]}"


def test_requesting_cpu_names_the_device_torch_allocates_on():
    """The same invariant as the CUDA case, on the host every machine has.

    CPU has no index to lose, so this passes with or without the fix. It is here to show
    that the equality being asserted above is the ordinary one and holds for a device the
    manager already gets right.
    """
    manager = DeviceManager("cpu")

    assert manager.device == torch.zeros(2, 2).device


def test_cuda_on_a_host_without_it_falls_back_to_cpu_and_says_so(monkeypatch, caplog):
    """Falling back to CPU, and doing it without ever asking for a device index.

    ``torch.cuda.current_device()`` raises when there is no CUDA runtime, so resolving the
    index has to sit behind the availability check rather than in front of it. Booby
    trapping ``current_device`` is what makes this test see the ordering: patching
    availability alone would pass whichever order the two calls were written in.
    """
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        torch.cuda,
        "current_device",
        lambda: (_ for _ in ()).throw(AssertionError("asked for an index with no CUDA runtime")),
    )

    with caplog.at_level(logging.WARNING, logger=LOGGER):
        manager = DeviceManager("cuda")

    assert manager.device == torch.device("cpu")
    assert "CUDA" in caplog.text


def test_the_index_is_resolved_without_needing_real_hardware(monkeypatch):
    """The #223 defect itself, on a host with no GPU.

    Every other test that can see the missing index is skipped without CUDA, which means
    the file goes entirely green on a CPU-only host against a version that still stores
    ``cuda`` unresolved. Faking both halves of the availability check exercises the
    resolution branch directly, so the defect cannot hide behind a skip.

    Index 3 rather than 0, so a manager that hardcoded ``cuda:0`` fails here.
    """
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 3)

    resolved = DeviceManager("cuda").device

    assert resolved.index is not None, f"the index was left unresolved: {resolved}"
    assert resolved == torch.device("cuda", 3)
    # An explicit index is never overridden by the current device.
    assert DeviceManager("cuda:1").device == torch.device("cuda", 1)


@pytest.mark.parametrize("requested", ["meta", "cpu", "cuda", "cuda:1", None])
def test_set_device_agrees_with_the_constructor(monkeypatch, requested):
    """The setter must land where the constructor would, for every kind of request.

    Asserting only that a setter sets is a test with no oracle: it would pass against an
    implementation that stored the argument verbatim and skipped the CUDA resolution and
    the availability fallback entirely. The oracle here is the constructor, which is the
    one path already covered by the tests above, so the two cannot drift apart unnoticed.

    CUDA is faked so the cuda cases run on any host, and index 3 is used so a setter that
    hardcodes an index fails.
    """
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 3)

    manager = DeviceManager("cpu")
    assert hasattr(manager, "set_device"), "DeviceManager has no supported device setter"

    returned = manager.set_device(requested)

    expected = DeviceManager(requested).device
    assert manager.device == expected
    assert returned == expected
    # "cpu" and None both legitimately resolve to cpu, so only the rest can show that a
    # setter which dropped its argument would have been caught.
    if requested not in ("cpu", None):
        assert manager.device != torch.device("cpu"), "an ignored argument would leave cpu"


def import_device_manager_module():
    """Execute ``bridge/utils/device_manager.py`` as a fresh, unregistered module.

    This runs the module's top-level code, which is where the process singleton is built,
    without touching ``sys.modules``, so the copy every other module already holds is left
    alone. It covers the construction the import performs; it does not and cannot cover the
    one real import that happened before the test session started.
    """
    path = os.path.join(get_project_root(), "bridge/utils/device_manager.py")
    spec = importlib.util.spec_from_file_location("device_manager_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_bridge_device_selects_the_process_device(monkeypatch):
    """The environment variable is the only knob available before anything is constructed.

    Tokenizers, datasets, the pipeline and the model each capture ``device_manager.device``
    in their own ``__init__``, so a device chosen after import reaches none of them. It has
    to be settable before the first import runs.
    """
    monkeypatch.setenv("BRIDGE_DEVICE", "meta")

    assert import_device_manager_module().device_manager.device == torch.device("meta")


def test_the_process_device_defaults_to_cpu(monkeypatch):
    """Control for the test above: with no variable set the singleton is still CPU.

    Without this, a manager that was hardcoded to ``meta`` would pass the env-var test.
    """
    monkeypatch.delenv("BRIDGE_DEVICE", raising=False)

    assert import_device_manager_module().device_manager.device == torch.device("cpu")
