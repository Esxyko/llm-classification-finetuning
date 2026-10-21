"""LLM Classification Finetuning package."""

import os

# Hugging Face reads download settings when its libraries are first imported.
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
if os.environ["HF_ENDPOINT"].rstrip("/") == "https://hf-mirror.com":
	# Use HTTP downloads for the mirror to avoid Xet CAS authentication errors.
	os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
