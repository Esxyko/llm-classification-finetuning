import torch
import transformers.integrations.sdpa_attention as sdpa

"""
Apply the T4 SDPA compatibility patch inside run_kaggle_submit.py, after any from __future__ imports and before project imports, model loading, or CLI execution.

Set sdpa.use_gqa_in_sdpa to always return False, enable memory-efficient SDPA, and disable math SDPA. This makes Transformers expand K/V heads through repeat_kv, allowing compatible calls to use memory-efficient attention on T4 with PyTorch 2.11.

If the notebook generates this script, insert the patch into the generated Python source, then regenerate and relaunch it. Applying the patch only in the notebook process will not affect the submission subprocess.

Keep FP16, batch size 1, and the existing maximum sequence length unchanged initially. The workaround still needs verification in Kaggle.
"""
sdpa.use_gqa_in_sdpa = lambda *args, **kwargs: False

torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(False)
