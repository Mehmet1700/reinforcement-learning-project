# RL Group Project: Sepsis Treatment Optimisation

**Master in Data Science & Advanced Analytics — Reinforcement Learning**
Deadline: June 5, 2026

## Group Members

| Name | Student ID |
|------|------------|
| Mehmet Karaca  | 20250344 |
| Luis   | ... |
| Veronica  | ... |
| Ana Macedo  | 20250405 |
| Miguel  | ... |


## Project Overview

Clinical reinforcement learning project using the **ICU-Sepsis-v2** environment, built on real MIMIC-III ICU data. The goal is to learn optimal treatment policies for sepsis patients (vasopressor dosing × IV fluid levels).

| Configuration | Observation Space | Methods | 
|---------------|-------------------|---------|
| **Config A** | Discrete (716 states) | DP, Q-Learning, SARSA |
| **Config B** | Continuous (47-dim) | DQN, PPO |
| **Extension** | TBD | TBD |

## Random Baselines (must beat these)

| Config | Mean Return | Survival Rate |
|--------|-------------|---------------|
| Config A | 0.5809 | 68.3% |
| Config B | 0.5624 | 65.6% |

## Setup

```bash
uv sync
source .venv/bin/activate      # Windows: .venv\Scripts\activate
```

> Requires [uv](https://docs.astral.sh/uv/getting-started/installation/). Alternatively: `pip install -r requirements.txt`

> **Important**: Always launch Jupyter from the **project root** so that the `envs` package is importable.

```bash
# From project root:
jupyter lab
```

## Project Structure

```
envs/          Provided environment code — do not modify
agents/        Algorithm implementations
  config_a/    Tabular RL (Q-Learning, SARSA)
  config_b/    Deep RL (DQN, PPO)
  extension/   Creative extension
utils/         Shared evaluation and plotting helpers
notebooks/     One notebook per configuration
results/       Saved models and training curves (excluded from git)
plots/         Saved figures
report/        Final PDF report
```

## Notebooks

| Notebook | Purpose |
|----------|---------|
| `00_environment_exploration.ipynb` | Starter code, env setup, random baselines |
| `01_config_a_tabular.ipynb` | Q-Learning and SARSA on Config A |
| `02_config_b_deep_rl.ipynb` | DQN and PPO on Config B |
| `03_extension.ipynb` | Creative extension |

## Reproducibility

All experiments use `SEED = 42`. The shared evaluation function `utils/evaluation.py:eval_agent()` must be used for all reported results.

## Evaluation Metrics

- Mean return per episode
- Survival rate (%)
- Convergence speed
- Config A vs Config B comparison
