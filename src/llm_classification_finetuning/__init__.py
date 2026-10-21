"""LLM Classification Finetuning package."""

import os

# Hugging Face reads the endpoint when its libraries are first imported.
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
