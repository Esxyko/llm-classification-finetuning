"""LLM Classification Finetuning package."""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load project settings before Hugging Face freezes its environment at import time.
load_dotenv(dotenv_path=Path.cwd() / ".env", override=False)

# Hugging Face reads download settings when its libraries are first imported.
os.environ.setdefault("HF_ENDPOINT", "https://huggingface.co")
if os.environ["HF_ENDPOINT"].rstrip("/") == "https://hf-mirror.com":
	# Use HTTP downloads for the mirror to avoid Xet CAS authentication errors.
	os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
