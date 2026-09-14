"""Session-wide guards for the two pieces of global state BRIDGE tests can disturb.

Both exist because a test that leaks here fails a *different* test, which is the hardest
kind of failure to read.
"""

import os

# Popped at import, before `bridge` is imported below, and deliberately not in a fixture.
# `BRIDGE_DEVICE` selects the process device when `bridge.utils.device_manager` is first
# imported, which happens while pytest is collecting. A fixture runs long after that, so
# it would be reading an already-built singleton and could only claim to help. Pytest
# imports the rootdir conftest before any test module, so this is early enough.
# Measured: with BRIDGE_DEVICE=meta exported, popping here gives a clean suite where the
# fixture form gave 19 failed, 38 errors.
_BRIDGE_DEVICE = os.environ.pop("BRIDGE_DEVICE", None)

import pytest  # noqa: E402

from bridge.utils import device_manager  # noqa: E402


@pytest.fixture(autouse=True)
def restore_process_device():
    """Put ``device_manager`` back after any test that repoints it.

    Every BRIDGE object captures ``device_manager.device`` in its own ``__init__``, so a
    test that leaves the singleton on CUDA silently moves every model built after it. The
    device tests repoint it deliberately; this makes that safe rather than relying on each
    one to clean up.
    """
    before = device_manager.device
    yield
    device_manager._device = before
