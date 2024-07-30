from addict import Dict as AttrDict
import pytest
from src.main import create_config, load_config, read_args, validate_config

def assert_required_args_in_config(config, required_args):
    for arg in required_args:
        assert arg in config

def test_load_config_default(required_args):
    config = load_config()

    assert isinstance(config, AttrDict)
    assert_required_args_in_config(config, required_args)

def test_loading_config_file(required_args):
    filename = "config.yaml"
    config = load_config(filename)
    assert isinstance(config, AttrDict)
    assert_required_args_in_config(config, required_args)

def test_create_config_default(required_args):
    config = create_config(args = [])

    assert isinstance(config, AttrDict)
    assert_required_args_in_config(config, required_args)

def test_create_config_with_conf(required_args):
    config = load_config()
    config = create_config(config)
    assert_required_args_in_config(config, required_args)

    assert isinstance(config, AttrDict)
    print(config)

def test_arguments_from_cli(monkeypatch):
    """Test whether arguments from the command line are set up correctly."""
    monkeypatch.setattr("sys.argv", ["-m" "script_name", "--config", "config.yaml"])
    args = read_args()
    assert args.config == "config.yaml"

def test_cli_missing_parameters(monkeypatch):
    """Test whether arguments from the command line are set up correctly."""
    monkeypatch.setattr("sys.argv", ["-m", "script_name"])
    with pytest.raises(TypeError, match=r'missing 1 required positional argument'):
        validate_config()
    pass
