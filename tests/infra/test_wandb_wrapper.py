"""A missing W&B credential must disable logging, not abort the run (issue #224, defect 6).

``WandbWrapper`` is a logging wrapper: it is a side channel, never the thing being computed.
Reading the credential with a bare subscript makes an unset ``WANDB_API_KEY`` raise a
``KeyError`` out of the constructor, which kills a training run for a reason the message does
not explain and does not suggest the remedy. The contract asserted here is that an absent or
placeholder-empty key turns the wrapper off with a warning naming the variable, and that a key
that is actually present is passed through to ``wandb.login`` unchanged.

Every test also pins the surrounding behaviour that must not move: login is attempted exactly
when a key is present and the wrapper is enabled, and never otherwise.
"""

import logging

import pytest
import wandb

from bridge.infra.clients.wandb.wandb_wrapper import WandbWrapper

LOGGER = "bridge.infra.clients.wandb.wandb_wrapper"

# An unguessable marker, so a recorded key can only have come from the environment we set.
REAL_KEY = "k3y-9f2c7ab4-do-not-invent"


@pytest.fixture(autouse=True)
def fresh_singleton():
    """``WandbWrapper`` extends ``Singleton``, which caches the object in the class attribute
    ``_instance``. ``__new__`` hands back the cached object while ``__init__`` runs again, so
    without this reset the wrapper built by one test is the wrapper the next test gets, and
    any attribute ``__init__`` does not reassign survives with it. Measured: a second
    construction returns the same object and carries the first one's extra attributes, and a
    constructor that raised part way through still leaves ``_instance`` populated. Reset on
    both sides so neither ordering within this file nor the rest of the suite is affected.
    """
    WandbWrapper._instance = None
    yield
    WandbWrapper._instance = None


@pytest.fixture
def wandb_spy(monkeypatch):
    """Records what the wrapper asks of ``wandb`` and guarantees no test reaches the network."""
    calls: dict[str, list] = {"login": [], "init": [], "log": []}

    def fake_login(*args, **kwargs):
        calls["login"].append(kwargs.get("key", args[0] if args else None))

    def fake_init(*args, **kwargs):
        calls["init"].append(kwargs)
        return object()

    def fake_log(*args, **kwargs):
        calls["log"].append((args, kwargs))

    monkeypatch.setattr(wandb, "login", fake_login)
    monkeypatch.setattr(wandb, "init", fake_init)
    monkeypatch.setattr(wandb, "log", fake_log)
    return calls


def test_an_unset_key_disables_the_wrapper_instead_of_raising(monkeypatch, caplog, wandb_spy):
    """The defect: this construction raises KeyError('WANDB_API_KEY') out of a logging wrapper."""
    monkeypatch.delenv("WANDB_API_KEY", raising=False)

    with caplog.at_level(logging.WARNING, logger=LOGGER):
        wrapper = WandbWrapper(project_name="probe", is_enabled=True)

    assert wrapper.is_enabled is False
    assert "WANDB_API_KEY" in caplog.text
    assert wandb_spy["login"] == []


def test_an_empty_key_counts_as_absent(monkeypatch, caplog, wandb_spy):
    """An empty value is what people set to get past a crash on a bare subscript, so it means
    "no credential", not "log in with the empty string". The control is the second half: the
    same construction with a real key must come out the other way, which is what shows the
    assertion is about emptiness and not about the code path always disabling itself.
    """
    monkeypatch.setenv("WANDB_API_KEY", "")
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        empty = WandbWrapper(project_name="probe", is_enabled=True)

    assert empty.is_enabled is False
    assert "WANDB_API_KEY" in caplog.text
    assert wandb_spy["login"] == []

    WandbWrapper._instance = None
    caplog.clear()

    monkeypatch.setenv("WANDB_API_KEY", REAL_KEY)
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        present = WandbWrapper(project_name="probe", is_enabled=True)

    assert present.is_enabled is True
    assert "WANDB_API_KEY" not in caplog.text
    assert wandb_spy["login"] == [REAL_KEY]


def test_a_present_key_is_passed_through_unchanged(monkeypatch, wandb_spy):
    """The oracle is the argument value: the key handed to ``wandb.login`` must be exactly the
    string in the environment, so a future rewrite cannot quietly log in with something else.
    """
    monkeypatch.setenv("WANDB_API_KEY", REAL_KEY)

    wrapper = WandbWrapper(project_name="probe", is_enabled=True)

    assert wrapper.is_enabled is True
    assert wandb_spy["login"] == [REAL_KEY]


def test_disabling_the_wrapper_skips_login_even_with_a_key_present(monkeypatch, wandb_spy):
    """The control for the spy itself. A spy that cannot observe a non-call cannot report one,
    so this asserts an empty record under ``is_enabled=False`` and then, with the only change
    being that flag, a populated record. The two halves must disagree.
    """
    monkeypatch.setenv("WANDB_API_KEY", REAL_KEY)

    disabled = WandbWrapper(project_name="probe", is_enabled=False)
    assert disabled.is_enabled is False
    assert wandb_spy["login"] == []

    WandbWrapper._instance = None
    WandbWrapper(project_name="probe", is_enabled=True)
    assert wandb_spy["login"] == [REAL_KEY]


def test_a_disabled_wrapper_degrades_instead_of_raising(monkeypatch, wandb_spy):
    """Once the missing key has turned the wrapper off, the caller keeps calling it. Both
    guarded methods must return quietly without touching ``wandb``. ``log_metrics`` carries the
    control: on an enabled wrapper with no run it raises RuntimeError, so the silence below is
    the guard firing rather than the method being inert.
    """
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    wrapper = WandbWrapper(project_name="probe", is_enabled=True)
    assert wrapper.is_enabled is False

    assert wrapper.start_run(run_name="run") is None
    assert wrapper.log_metrics({"loss": 1.0}, step=0) is None
    assert wrapper.run is None
    assert wandb_spy["init"] == []
    assert wandb_spy["log"] == []

    WandbWrapper._instance = None
    monkeypatch.setenv("WANDB_API_KEY", REAL_KEY)
    enabled = WandbWrapper(project_name="probe", is_enabled=True)
    with pytest.raises(RuntimeError):
        enabled.log_metrics({"loss": 1.0}, step=0)
