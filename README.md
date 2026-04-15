# Singlish Scam Classifier

This repository packages a notebook-first NLP project for classifying Singlish messages as scam or non-scam. The main deliverable is a fine-tuned SingBERT classifier, with a lightweight inference script and a small set of supporting experiments around Singlish persona fine-tuning.

The repo is intentionally structured for public presentation: the classifier workflow is the primary story, the notebook order is explicit, and local-only assets such as full datasets and model weights stay out of git.

## What's in This Repo

The main deliverable is the Singlish scam classifier workflow:

- `01_dataset_processing.ipynb` prepares the canonical merged training dataset.
- `02_classifier_training.ipynb` fine-tunes and evaluates the SingBERT classifier.
- `03_inference_demo.ipynb` shows notebook-based inference against a saved local model.
- `inference.py` provides a simple script entrypoint for running the saved classifier.

The repo also includes optional persona fine-tuning experiments under `experiments/persona_training/`.

## Repo Layout

```text
.
├── 01_dataset_processing.ipynb
├── 02_classifier_training.ipynb
├── 03_inference_demo.ipynb
├── README.md
├── LICENSE
├── requirements.txt
├── inference.py
├── experiments/
│   └── persona_training/
│       ├── ah_beng_persona_training.ipynb
│       ├── nsf_persona_training.ipynb
│       └── xmm_persona_training.ipynb
└── samples/
    ├── classifier_sample.csv
    └── persona_sample.json
```

## Quick Start

Install the core classifier dependencies:

```bash
pip install -r requirements.txt
```

Run the inference script from the repo root:

```bash
python inference.py
```

This expects a local `model/` directory containing the saved classifier weights and tokenizer files.

## Notebook Order

Run the core notebooks in this order:

1. `01_dataset_processing.ipynb`
2. `02_classifier_training.ipynb`
3. `03_inference_demo.ipynb`

## Optional Experiments

The notebooks in `experiments/persona_training/` are secondary experiments for fine-tuning Singlish persona-style assistants such as Ah Beng, NSF, and XMM variants.

These notebooks are more GPU-heavy and install additional packages inline, including `unsloth`, `trl`, `peft`, and related training dependencies. They are kept separate from the main classifier workflow so the public repo stays focused.

## What Is Not Committed

The following assets are intentionally kept out of git:

- `datasets/` with the full local training data
- `model/` with local model weights and tokenizer artifacts
- generated outputs such as `artifacts/`
- large local bundles such as `model.zip`

Small schema-compatible examples are provided in `samples/` so the data layout is still clear without publishing the full assets.
