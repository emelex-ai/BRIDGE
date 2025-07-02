# Installing uv

`uv` is a fast Python package installer and dependency resolver. Here are installation instructions for different operating systems.

## Quick Install (Recommended)

### macOS and Linux

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Windows (PowerShell)

```powershell
powershell -c "irm https://astral.sh/uv/install.sh | iex"
```

## Alternative Installation Methods

### Using pip

If you already have Python and pip installed:

```bash
pip install uv
```

### Using pipx (Recommended for Python tools)

```bash
pipx install uv
```

### Using Homebrew (macOS)

```bash
brew install uv
```

### Using winget (Windows)

```bash
winget install --id=astral-sh.uv  -e
```

### Using scoop (Windows)

```bash
scoop install uv
```

### Using conda/mamba

```bash
conda install -c conda-forge uv
# or
mamba install -c conda-forge uv
```

### Using Docker

```bash
docker run --rm -it ghcr.io/astral-sh/uv:latest
```

## Platform-Specific Notes

### macOS
- The installer script automatically adds uv to your PATH
- You may need to restart your terminal or run `source ~/.zshrc` (or `~/.bash_profile`)

### Windows
- The PowerShell installer adds uv to your PATH automatically
- You may need to restart your terminal for PATH changes to take effect
- For older Windows versions, you might need to install the Microsoft Visual C++ Redistributable

### Linux
- The installer script works on most Linux distributions
- On some minimal distributions, you might need to install `curl` first: `sudo apt install curl` or `sudo yum install curl`

## Verification

After installation, verify that uv is working:

```bash
uv --version
```

You should see output like:
```
uv 0.1.x (...)
```

## Getting Started

Once installed, you can start using uv:

```bash
# Create a new project
uv init my-project
cd my-project

# Add dependencies
uv add requests

# Install all dependencies
uv sync

# Run Python with the project environment
uv run python script.py
```

## Set up for the BRIDGE project
```bash
bash generate_uv.sh
```
produces `pyproject.toml`, `uv.lock`, .venv

The final step is to activate the project:
```bash
source .venv/bin/activate
```

## Troubleshooting

**Permission denied errors:**
- Try running the installer with `sudo` (not recommended) or use a user-level installation method like `pipx`

**Command not found:**
- Restart your terminal
- Check if the installation directory is in your PATH
- Try logging out and back in

**Network issues:**
- If behind a corporate firewall, you may need to configure proxy settings
- Try using the pip installation method instead

For more detailed information, visit the [official uv documentation](https://docs.astral.sh/uv/).
