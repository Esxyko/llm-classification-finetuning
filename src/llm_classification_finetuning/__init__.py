"""LLM Classification Finetuning package."""

from pathlib import Path

from dotenv import load_dotenv

# Load project settings before Hugging Face freezes its environment at import time.
load_dotenv(dotenv_path=Path.cwd() / ".env", override=False)
