# Project Setup

This project uses a Python virtual environment in the repository root and a .env file derived from .env.template.

## Prerequisites

- Python 3.11 installed
- (Windows) PowerShell
- (Linux/macOS) Bash shell

## One-time environment setup

From the project root (this folder), run **one** of the following:

### Windows (PowerShell)

```powershell
pwsh -File setup/setup.ps1
```

### Linux/macOS (Bash)

```bash
bash setup/setup.sh
```

Both scripts:
- Ensure they run from the project root
- Create .env from .env.template if .env does not already exist
- Remove any existing .venv and __pycache__ folders
- Create a new Python 3.11 virtual environment in .venv
- Install dependencies from requirements.txt

## Activating the virtual environment

After running the setup script:

- **Windows (PowerShell):**
  ```powershell
  .\.venv\Scripts\Activate.ps1
  ```

- **Linux/macOS (Bash):**
  ```bash
  source .venv/bin/activate
  ```

## OpenVLA server (7B model)

> <span style="color:#d9534f"><strong>Warning:</strong></span> The OpenVLA server in [openvla_server.ipynb](openvla_server.ipynb) loads the 7B-parameter `openvla/openvla-7b` model. Downloading and running this locally on low-end machines is likely to crash or exhaust memory.

Recommended setup:
- Use a hosted notebook environment (e.g. Google Colab) with a GPU runtime similar to a T4 or better.
- Upload or open [openvla_server.ipynb](openvla_server.ipynb) there and run all cells with GPU enabled.
- The notebook uses `pyngrok` to expose a public URL and prints a line like:
  ```
  [config.py] VLA_API_URL = "https://<your-ngrok-subdomain>.ngrok-free.dev"
  ```
- Copy that URL into the `VLA_API_URL` setting in [config.py](config.py) (or your `.env`, if you externalise it) so the simulation can call the remote VLA server instead of a local instance.

This avoids downloading the full 7B model on your local machine while keeping the rest of the project runnable from your PC.
