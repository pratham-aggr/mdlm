# Activation Steering in LLaDA

Study of activation steering effects on LLaDA-8B-Instruct, a masked diffusion language model.
Measures accuracy and entropy cost of steering across 9 conditions on GSM8K math problems.

## Setup

```bash
conda env create -f environment.yaml
conda activate llada
export HF_TOKEN=your_token_here
```

## Quickstart

```bash
# 1. Prepare data
python data/prepare_gsm8k.py
python data/prepare_steering_prompts.py

# 2. Extract steering vector (layer 23, last prompt token)
CUDA_VISIBLE_DEVICES=0 python scripts/extract_vector.py

# 3. Run all conditions
bash scripts/run_all_conditions.sh 0

# 4. Analyze results
python analysis/entropy_analysis.py
```

## Structure

```
src/
  config.py       central hyperparameters and paths
  model.py        load_model() — LLaDA-8B-Instruct on CUDA
  steering.py     extract_steering_vector(), apply_steering()
  generate.py     generate_with_measurements() — denoising loop with hooks
  conditions.py   CONDITIONS dict (A–I)
  measure.py      answer extraction, failure mode classification, entropy helpers

scripts/
  extract_vector.py        extract and save steering vector
  run_condition.py         run one condition, save results.jsonl
  run_all_conditions.sh    run all conditions sequentially on one GPU
  validate_vector.py       quick qualitative check: A vs B vs I on 3 prompts
  find_best_layer.py       sweep layers 0-31, measure refusal rate per layer

data/
  gsm8k_200.jsonl          200 GSM8K test problems (id, question, answer)
  harmful_prompts.json     30 harmful prompts for steering vector extraction
  harmless_prompts.json    30 harmless prompts for steering vector extraction

results/
  steering_vector.pt       saved steering vector (d_model=4096)
  condition_*/results.jsonl per-problem results for each condition
  summary_for_pi.md        experiment summary with accuracy table and figures

analysis/
  entropy_analysis.py      entropy curve plots and statistical tests
  figures/                 saved figures
```

## Conditions

| Key | Description |
|-----|-------------|
| A | Baseline — no steering |
| B | Full steering — all layers, all positions, all steps |
| C | Prompt tokens only |
| D | Response tokens only |
| E | First 20% of denoising steps |
| F | Last 20% of denoising steps |
| G | Mid-to-late layers (16-31) |
| H | Early layers (0-7) |
| I | Prompt only + last 20% steps (proposed fix) |

## Results

Baseline accuracy (A): 64.0%. Full steering (B): 58.5% (−5.5 pp).
Condition I recovers to 63.5% by restricting steering to prompt tokens during the final denoising steps.
See `results/summary_for_pi.md` for the full accuracy table and entropy analysis.
