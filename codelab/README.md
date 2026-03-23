# Codelab

```bash
# init codelab
$ uv pip install torch torchvision torchaudio --torch-backend=auto
$ uv run python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU')"
```