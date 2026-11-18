# AML-Homework1

Pretraining a 0.2B Transformer from scratch on TinyStories and studying scaling laws.

**Xizhi Xiao · College of AI, Tsinghua University**

[Report](report/main.pdf) · [Model](aml/model.py) · [Experiment configurations](experiments/) · [Results](report/full_validation_all.json)

## Results

The main model has 218.97M parameters and was trained for 4,000 updates on 1.049B tokens.

| Full-validation metric | Value |
|---|---:|
| Cross-entropy | 1.11692 |
| Perplexity | 3.05544 |
| Bits per byte | 0.42039 |

Thirty scaling experiments explore fixed-token and fixed-compute budgets. A power law fitted to five smaller models predicts the main model's loss with **2.08% relative error**.

## Setup

Use Python 3.10+ and install PyTorch for your CUDA environment, then run:

```bash
pip install -r requirements.txt
python -m unittest discover -s tests -v
```

Run all commands from the repository root. Main-model training uses a single CUDA GPU with BF16.

## Data and training

Download `TinyStories-train.txt` from [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories), revision `f54c09fd23315a6f9c86f9dc80f725de7d8f9c64`, into `raw/`.

```bash
python -m aml.data --input raw/TinyStories-train.txt \
  --input-format tinystories --tokenizer flexitok/bpe_fw_edu_16000 \
  --output data/tinystories

python -m aml.train --config configs/main.json \
  --data data/tinystories --output runs/main

python -m aml.evaluate --checkpoint runs/main/best.pt \
  --data data/tinystories --output runs/main/final_metrics.json --precision bf16

python -m aml.generate --checkpoint runs/main/best.pt \
  --data data/tinystories --output runs/main/samples.json
```

The custom byte-level BPE implementation and its trained 1,024-token vocabulary are in [`aml/bpe.py`](aml/bpe.py) and [`artifacts/custom_bpe/`](artifacts/custom_bpe/). Model training uses the course reference tokenizer with a 16,000-token vocabulary.

## Scaling experiments

Configurations are provided in `experiments/path1_pilot/`, `path1_full/`, and `path1_refinement/`. Some runs are shared between plans. Use `python -m aml.scaling --help` for planning, execution, and collection commands.

Reproduce the report's fits and figures from saved results without training:

```bash
python scripts/analyze_report_complete.py
```

Datasets and model checkpoints are distributed separately. The repository includes source code, configurations, tests, tokenizer artifacts, experiment results, and the report.

## AI assistance

AI tools assisted with implementation, debugging, experiment execution, and report editing. The author directed the study, made decisions on experimental scope, and reviewed and revised the analysis and presentation.
