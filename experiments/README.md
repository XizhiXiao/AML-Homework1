# Scaling Experiments

Thirty completed runs study how validation loss changes with model size, training tokens, and compute. The main model is held out for extrapolation.

[Report](../report/main.pdf) · [Full-validation results](../report/full_validation_all.json) · [Analysis](../report/final_analysis/analysis.json)

## Experiment plans

| Plan | Design | Runs |
|---|---|---:|
| [Pilot](path1_pilot/plan.json) | Five model sizes at 67.11M tokens; two IsoFLOP budgets | 15 |
| [Full](path1_full/plan.json) | Five model sizes at 1.049B tokens; three IsoFLOP budgets | 20 |
| [Refinement](path1_refinement/plan.json) | Additional sizes near the low-loss regions | 5 |

Pilot and full share ten IsoFLOP runs, giving **30 unique runs** across all plans. Each directory contains `plan.json`, individual configurations, and `budget.csv`.

The base grid uses approximately 5M, 11M, 25M, 50M, and 99M non-embedding parameters. Refinements add 1.78M, 2.67M, and 16.63M models at selected budgets. Compute is estimated as `C = 6ND`, with budgets of `1e16`, `3e16`, and `1e17` FLOPs.

All runs share the tokenizer, corpus split, and 1,024-token context. Scaling models use 32,768 tokens per update, a peak learning rate of `1.4142e-4`, and independent warmup/cosine schedules.

## Run experiments

Run commands from the repository root. Preview a plan without starting training:

```bash
python -m aml.scaling run \
  --plan experiments/path1_full/plan.json \
  --data data/tinystories --runs runs/scaling --gpu 1
```

Add `--execute` to train, or `--only RUN_ID` to select runs. The runner skips completed runs and resumes existing checkpoints. Reuse the same output directory across plans to avoid duplicate training.

The runner checks GPU UUIDs against the original course/lab host profiles and refuses busy GPUs. For another machine, configure the device checks for your authorized hardware before execution. Individual configurations can also be run with `python -m aml.train`.

## Analyze results

Reproduce the report's full-validation fits and figures from the included results:

```bash
python scripts/analyze_report_complete.py
```

Outputs in `report/final_analysis/` include the fixed-data power law, held-out predictions, IsoFLOP curves, compute envelope, and all training trajectories.

For newly trained runs, collect and analyze their monitoring metrics:

```bash
python -m aml.scaling collect \
  --plan experiments/path1_full/plan.json \
  --runs runs/scaling --main runs/main \
  --output runs/scaling/full_results.json

python -m aml.scaling analyze \
  --results runs/scaling/full_results.json \
  --output runs/scaling/full_analysis
```

Training curves use the fixed 128-document validation subset. The report's final fits use all 21,275 validation documents. Fixed-data fits match the main model's training-token budget; compute-envelope points are the best configurations tested at each budget.
