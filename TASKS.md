# Project Tasks

Deadline: **June 5, 2026**

## Setup

- [ ] Add all group members as GitHub collaborators (`gh repo invite <username>`)
- [x] Fill in the group members table in `README.md`
- [ ] Each member: clone repo, create a virtual environment, run `pip install -r requirements.txt`
- [ ] Verify setup: run `notebooks/00_environment_exploration.ipynb` end-to-end without errors

---

## Config A — Tabular RL

Work in `agents/config_a/` and `notebooks/01_config_a_tabular.ipynb`.

- [ ] Implement Q-Learning training loop (`agents/config_a/q_learning.py`)
- [ ] Tune Q-Learning hyperparameters (learning rate, epsilon decay, number of episodes)
- [ ] Implement SARSA training loop (`agents/config_a/sarsa.py`)
- [ ] Tune SARSA hyperparameters
- [ ] Run Policy Iteration / Value Iteration (`agents/config_a/dynamic_programming.py`) — gives the **optimal policy** as an upper-bound benchmark
- [ ] Implement Q-Learning training loop (`agents/config_a/q_learning.py`)
- [ ] Tune Q-Learning hyperparameters (learning rate, epsilon decay, number of episodes)
- [ ] Implement SARSA training loop (`agents/config_a/sarsa.py`)
- [ ] Tune SARSA hyperparameters
- [ ] Evaluate all agents using `utils/evaluation.eval_agent()` with `n_eval_episodes=1000, seed=42`
- [ ] Plot training curves (smoothed return over episodes)
- [ ] Compare DP (optimal) vs Q-Learning vs SARSA vs random baseline (survival rate, mean return)
- [ ] Analyse convergence: how many episodes until Q-Learning / SARSA approach the DP optimum?

---

## Config B — Deep RL

Work in `agents/config_b/` and `notebooks/02_config_b_deep_rl.ipynb`.

- [ ] Train DQN agent (`agents/config_b/dqn.py`) — tune architecture and hyperparameters
- [ ] Train PPO agent (`agents/config_b/ppo.py`) — tune architecture and hyperparameters
- [ ] Evaluate both agents using `utils/evaluation.eval_agent()` with `n_eval_episodes=1000, seed=42`
- [ ] Plot training curves
- [ ] Compare DQN vs PPO vs random baseline
- [ ] Analyse impact of clinical failure wrappers (noisy obs, missing obs, acute events)
- [ ] Save trained models to `results/config_b/` for reproducibility

---

## Creative Extension

Work in `agents/extension/` and `notebooks/03_extension.ipynb`.

- [ ] Decide on the extension idea and document the motivation in `notebooks/03_extension.ipynb`
- [ ] Implement the extension
- [ ] Evaluate using the same `eval_agent()` protocol
- [ ] Compare against Config A / Config B baselines
- [ ] Write the justification section for the report

**Extension ideas:**
- Reward shaping using clinical domain knowledge (penalise high vasopressor doses)
- Offline / batch RL on trajectories logged by the random baseline
- Ensemble policy with uncertainty quantification
- Interpretability: SHAP or saliency maps on Config B policy
- Robust RL: explicitly train against one of the clinical failure wrappers in isolation

---

## Report (max 15 pages, excluding references)

- [ ] Introduction: clinical problem, why RL, overall approach
- [ ] Methodology: algorithm descriptions and hyperparameter justification
- [ ] Evaluation: training curves, results table, convergence analysis
- [ ] Config A vs Config B comparison section
- [ ] Creative extension section
- [ ] Conclusion: findings, limitations, future improvements
- [ ] Proofread and trim to 15 pages

---

## Code Quality & Reproducibility

- [ ] Ensure all notebooks run end-to-end without errors (restart kernel → run all)
- [ ] Fix random seeds everywhere (`SEED = 42` via `utils/evaluation.py`)
- [ ] Add setup instructions to `README.md` if anything changes
- [ ] Remove debug prints and commented-out dead code before submission
- [ ] Make sure `results/` model files are saved so results can be reproduced without retraining

---

## Grading Breakdown (for reference)

| Area | Weight |
|------|--------|
| Algorithm choice & justification | 25% |
| Evaluation & comparative analysis | 30% |
| Visualisations & interpretability | 15% |
| Creative extension | 10% |
| Code quality & reproducibility | 10% |
| Report quality & presentation | 10% |
