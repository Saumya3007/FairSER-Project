import os
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfApi, login


load_dotenv() 

HF_TOKEN = os.getenv("HF_TOKEN")
HF_REPO_ID = os.getenv("HF_REPO_ID") 

if HF_TOKEN is None or HF_REPO_ID is None:
    raise RuntimeError(
        "HF_TOKEN and HF_REPO_ID must be set in your .env file."
    )

login(token=HF_TOKEN)

api = HfApi()


print(f"[HF] Using repo: {HF_REPO_ID}")
api.create_repo(
    repo_id=HF_REPO_ID,
    token=HF_TOKEN,
    repo_type="model",
    private=False,
    exist_ok=True,  
)


BASE_DIR = Path(__file__).resolve().parent
CKPT_DIR = BASE_DIR / "results" / "checkpoints"
files_to_upload = {
    "checkpoints/head_best.pt": CKPT_DIR / "head_best.pt",
    "checkpoints/lora_best.pt": CKPT_DIR / "lora_best.pt",
    "checkpoints/clues_lora_best.pt": CKPT_DIR / "clues_lora_best.pt",
    "checkpoints/full_best.pt": CKPT_DIR / "full_best.pt",
}

existing_files = {
    path_in_repo: local_path
    for path_in_repo, local_path in files_to_upload.items()
    if local_path.exists()
}

if not existing_files:
    raise FileNotFoundError(
        f"No checkpoints found under {CKPT_DIR}. "
    )

print("[HF] Files that will be uploaded:")
for path_in_repo, local_path in existing_files.items():
    print(f"  - {local_path}  ->  {path_in_repo}")


for path_in_repo, local_path in existing_files.items():
    print(f"[HF] Uploading {local_path} -> {path_in_repo}")
    api.upload_file(
        path_or_fileobj=str(local_path),
        path_in_repo=path_in_repo,
        repo_id=HF_REPO_ID,
        repo_type="model",
        token=HF_TOKEN,
    )


readme_path = BASE_DIR / "README_hf.md"

model_card = f"""---
language:
- en
- hi
tags:
- audio
- speech
- emotion-recognition
- fairness
- robustness
- explainability
license: mit
datasets:
- iemocap
- iitkgp
base_model: facebook/wav2vec2-base
library_name: pytorch
pipeline_tag: audio-classification
---

# FairHindiSER: Fair and Robust Speech Emotion Recognition

This repository contains the checkpoints for **FairHindiSER**, a speech emotion
recognition model trained on a 50/50 mix of Hindi (IITKGP) and English (IEMOCAP)
emotional speech. The model predicts four emotions: *angry, happy, neutral, sad*.

The backbone is [`facebook/wav2vec2-base`](https://huggingface.co/facebook/wav2vec2-base),
adapted with:

- A FairSER MLP head on top of pooled wav2vec2 features.
- LoRA adapters on the top transformer layers.
- CLUES-style contrastive debiasing to reduce gaps across language/gender groups.
- Gradual full unfreezing of the encoder with Optuna-tuned learning rates.

## Checkpoints in this repo

- `checkpoints/head_best.pt` — head-only fine-tuning (backbone frozen).
- `checkpoints/lora_best.pt` — LoRA fine-tuning with focal loss and class weights.
- `checkpoints/clues_lora_best.pt` — LoRA + CLUES contrastive debiasing.
- `checkpoints/full_best.pt` — final model with gradual full unfreezing.

You can load the final model like this:

```python
import torch
from transformers import Wav2Vec2Model
from fairhindiser import FairSERModel  # your model class, if you publish it as a pip package

ckpt = torch.load("checkpoints/full_best.pt", map_location="cpu")
model = FairSERModel()
model.load_state_dict(ckpt)
model.eval()
```

(Adapt the import path to your own project structure.)

## Intended use

- Research on cross-lingual and fair speech emotion recognition.
- Analysis of robustness and calibration under common audio corruptions
  (noise, speed, pitch perturbations).
- As a backbone for downstream SER systems in multilingual / accented settings.

Not intended for high-stakes decisions or medical/psychological diagnosis.

## Training data

- **Hindi:** IITKGP Hindi corpus (naturalistic, acted dialogues).
- **English:** IEMOCAP 4-class subset (angry, happy/excited, neutral, sad).

All audio was resampled to 16 kHz mono and normalized. The combined dataset
contains 3200 Hindi + 3200 English clips, stratified into train/val/test splits.

## Evaluation

We report metrics on the held-out test set and AudioTrust-style axes:

- **Per-class F1** and confusion matrix.
- **Group F1** by language, gender and accent.
- **Robustness** under additive noise, speed and pitch perturbations.
- **Calibration & privacy proxies** from confidence distributions.

(See the associated paper / report for full numbers and plots.)
"""

readme_path.write_text(model_card.strip() + "\n", encoding="utf-8")

print(f"[HF] Uploading model card -> README.md")
api.upload_file(
    path_or_fileobj=str(readme_path),
    path_in_repo="README.md",  
    repo_id=HF_REPO_ID,
    repo_type="model",
    token=HF_TOKEN,
)

print("[HF] Done. Check your model at:")
print(f"     https://huggingface.co/{HF_REPO_ID}")