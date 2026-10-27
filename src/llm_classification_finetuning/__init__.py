"""LLM Classification Finetuning package."""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load project settings before libraries read their environment at import time.
load_dotenv(dotenv_path=Path.cwd() / ".env", override=False)

# Set the allocator default before PyTorch imports; honor its legacy CUDA alias.
if "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
	os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

# Hugging Face reads download settings when its libraries are first imported.
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
if os.environ["HF_ENDPOINT"].rstrip("/") == "https://hf-mirror.com":
	# Use HTTP downloads for the mirror to avoid Xet CAS authentication errors.
	os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
