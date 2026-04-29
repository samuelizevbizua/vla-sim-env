# Project Setup

This project uses a Python virtual environment in the repository root and a .env file derived from .env.template.

## Prerequisites

- Python 3.11 installed
- (Linux/macOS) Bash shell

## One-time environment setup

```bash
bash setup/setup.sh
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

## SmolVLA server (LeRobot)

> <span style="color:#d9534f"><strong>Warning:</strong></span> The SmolVLA server in [smolvla_server.ipynb](smolvla_server.ipynb) loads the LeRobot `smolvla` model. Downloading and running this locally on low-end machines may cause performance issues.

Recommended setup:
- Use a hosted notebook environment (e.g. Google Colab) with a GPU runtime similar to a T4 or better.
- Upload or open [smolvla_server.ipynb](smolvla_server.ipynb) there and run all cells with GPU enabled.
- The notebook uses `pyngrok` to expose a public URL and prints a line like:
  ```
  [config.py] VLA_API_URL = "https://<your-ngrok-subdomain>.ngrok-free.dev"
  ```
- Copy that URL into the `VLA_API_URL` setting in [config.py](config.py) (or your `.env`, if you externalise it) so the simulation can call the remote SmolVLA server instead of a local instance.

This avoids downloading the model on your local machine while keeping the rest of the project runnable from your PC.

RUNPOD Template for current working pod - https://console.runpod.io/deploy?template=juh5pg634w&ref=jr9ndf5k
Training done on rtx l40s