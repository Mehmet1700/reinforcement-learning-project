"""
Parallel hyperparameter search utilities for Config B.

Trial functions are defined at module level (not inside the notebook) so they
can be pickled by joblib and sent to worker processes on macOS, which uses the
'spawn' multiprocessing start method by default (Python 3.8+).

Usage (in notebook):
    from joblib import Parallel, delayed
    from utils.search import run_dqn_trial, run_ppo_trial

    results = Parallel(n_jobs=8, backend='loky', verbose=10)(
        delayed(run_dqn_trial)(i, params, ...) for i, params in enumerate(trial_params)
    )
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any

import torch

# ── Activation function registry ─────────────────────────────────────────────
# Stored as strings in JSON / dicts, converted to torch classes before training.

ACTIVATION_MAP: dict[str, type] = {
    "ReLU": torch.nn.ReLU,
    "Tanh": torch.nn.Tanh,
    "ELU":  torch.nn.ELU,
}


def _to_json_safe(params: dict) -> dict:
    """Convert a params dict to JSON-serialisable form.

    Handles:
      - torch.nn classes   → their __name__ string  (e.g. torch.nn.Tanh → 'Tanh')
      - numpy scalars      → plain Python float/int
      - lists (net_arch)   → plain Python list
    """
    out: dict = {}
    for k, v in params.items():
        if isinstance(v, type):          # e.g. torch.nn.ReLU
            out[k] = v.__name__
        elif hasattr(v, "item"):         # numpy scalar
            out[k] = v.item()
        else:
            out[k] = v
    return out


def _from_json(params: dict) -> dict:
    """Restore activation_fn from its string name, and net_arch as plain list."""
    out = dict(params)
    if "activation_fn" in out and isinstance(out["activation_fn"], str):
        name = out["activation_fn"]
        if name not in ACTIVATION_MAP:
            raise ValueError(
                f"Unknown activation_fn '{name}'. "
                f"Valid options: {list(ACTIVATION_MAP)}"
            )
        out["activation_fn"] = ACTIVATION_MAP[name]
    if "net_arch" in out:
        out["net_arch"] = [int(x) for x in out["net_arch"]]
    return out


# ── DQN trial ─────────────────────────────────────────────────────────────────

def run_dqn_trial(
    trial_idx: int,
    params: dict,          # JSON-safe: activation_fn as string, net_arch as list
    save_dir: str,
    total_timesteps: int,
    eval_episodes: int,
    seed: int,
    project_root: str,
) -> dict | None:
    """
    Run one DQN hyperparameter trial. Safe to call from a joblib worker.

    Returns a result dict, or None if the trial fails (so the rest of the
    batch continues rather than the whole job crashing).
    """
    # Worker processes start fresh — add project root so imports work.
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    try:
        from envs.wrappers import make_clinical_env  # noqa: PLC0415
        from agents.config_b.dqn import (            # noqa: PLC0415
            DQNConfig, train_dqn, load_dqn, get_policy,
        )
        from utils.evaluation import eval_agent       # noqa: PLC0415

        zip_path = os.path.join(save_dir, f"dqn_grid_{trial_idx}.zip")
        cfg_path = os.path.join(save_dir, f"dqn_grid_{trial_idx}_config.json")

        if os.path.exists(zip_path) and os.path.exists(cfg_path):
            with open(cfg_path) as fh:
                saved = json.load(fh)
            resolved = _from_json(saved)
            model = load_dqn(zip_path.replace(".zip", ""))
            print(f"[DQN {trial_idx:>3}] loaded from cache", flush=True)
        else:
            resolved = _from_json(params)
            cfg = DQNConfig(total_timesteps=total_timesteps, seed=seed, **resolved)
            save_path = os.path.join(save_dir, f"dqn_grid_{trial_idx}")
            model, _ = train_dqn(make_clinical_env, cfg, save_path=save_path)
            with open(cfg_path, "w") as fh:
                json.dump(_to_json_safe(resolved), fh)
            print(f"[DQN {trial_idx:>3}] trained & saved", flush=True)

        res = eval_agent(
            get_policy(model), make_clinical_env,
            n_eval_episodes=eval_episodes, seed=seed,
        )
        result = {
            "trial": trial_idx,
            **_to_json_safe(resolved),
            "survival_rate": res["survival_rate"],
            "mean_return":   res["mean_return"],
        }
        print(
            f"[DQN {trial_idx:>3}] "
            f"lr={resolved.get('learning_rate'):.0e}  "
            f"arch={resolved.get('net_arch')}  "
            f"act={params.get('activation_fn')}  "
            f"→ survival={res['survival_rate']:.1%}",
            flush=True,
        )
        return result

    except Exception as exc:
        print(f"[DQN {trial_idx:>3}] FAILED: {exc}", flush=True)
        return None


# ── PPO trial ─────────────────────────────────────────────────────────────────

def run_ppo_trial(
    trial_idx: int,
    params: dict,
    save_dir: str,
    total_timesteps: int,
    eval_episodes: int,
    seed: int,
    project_root: str,
) -> dict | None:
    """
    Run one PPO hyperparameter trial. Safe to call from a joblib worker.

    Returns a result dict, or None if the trial fails.
    """
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    try:
        from envs.wrappers import make_clinical_env  # noqa: PLC0415
        from agents.config_b.ppo import (            # noqa: PLC0415
            PPOConfig, train_ppo, load_ppo, get_policy,
        )
        from utils.evaluation import eval_agent       # noqa: PLC0415

        zip_path = os.path.join(save_dir, f"ppo_grid_{trial_idx}.zip")
        cfg_path = os.path.join(save_dir, f"ppo_grid_{trial_idx}_config.json")

        if os.path.exists(zip_path) and os.path.exists(cfg_path):
            with open(cfg_path) as fh:
                saved = json.load(fh)
            resolved = _from_json(saved)
            model = load_ppo(zip_path.replace(".zip", ""))
            print(f"[PPO {trial_idx:>3}] loaded from cache", flush=True)
        else:
            resolved = _from_json(params)
            cfg = PPOConfig(total_timesteps=total_timesteps, seed=seed, **resolved)
            save_path = os.path.join(save_dir, f"ppo_grid_{trial_idx}")
            model, _ = train_ppo(make_clinical_env, cfg, save_path=save_path)
            with open(cfg_path, "w") as fh:
                json.dump(_to_json_safe(resolved), fh)
            print(f"[PPO {trial_idx:>3}] trained & saved", flush=True)

        res = eval_agent(
            get_policy(model), make_clinical_env,
            n_eval_episodes=eval_episodes, seed=seed,
        )
        result = {
            "trial": trial_idx,
            **_to_json_safe(resolved),
            "survival_rate": res["survival_rate"],
            "mean_return":   res["mean_return"],
        }
        print(
            f"[PPO {trial_idx:>3}] "
            f"lr={resolved.get('learning_rate'):.0e}  "
            f"n_steps={resolved.get('n_steps')}  "
            f"arch={resolved.get('net_arch')}  "
            f"act={params.get('activation_fn')}  "
            f"→ survival={res['survival_rate']:.1%}",
            flush=True,
        )
        return result

    except Exception as exc:
        print(f"[PPO {trial_idx:>3}] FAILED: {exc}", flush=True)
        return None


# ── Q-Learning trial (Config A) ───────────────────────────────────────────────

def run_ql_trial(
    trial_idx: int,
    params: dict,
    save_dir: str,
    n_episodes: int,
    eval_episodes: int,
    seed: int,
    project_root: str,
) -> dict | None:
    """
    Run one Q-Learning hyperparameter trial. Safe to call from a joblib worker.

    Q-table cached as .npy; config cached as .json.
    Returns a result dict, or None if the trial fails.
    """
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    try:
        import numpy as np                                        # noqa: PLC0415
        from envs.env_setup import make_sepsis_env, N_STATES, N_ACTIONS  # noqa: PLC0415
        from agents.config_a.q_learning import QLearningAgent, QLearningConfig  # noqa: PLC0415
        from utils.evaluation import eval_agent                   # noqa: PLC0415

        os.makedirs(save_dir, exist_ok=True)
        npy_path = os.path.join(save_dir, f"ql_trial_{trial_idx}.npy")
        cfg_path = os.path.join(save_dir, f"ql_trial_{trial_idx}_config.json")

        cfg = QLearningConfig(n_episodes=n_episodes, **params)
        agent = QLearningAgent(N_STATES, N_ACTIONS, cfg)

        if os.path.exists(npy_path) and os.path.exists(cfg_path):
            agent.Q = np.load(npy_path)
            print(f"[QL  {trial_idx:>3}] loaded from cache", flush=True)
        else:
            # Reuse single env instance (reset each episode) — much faster than
            # creating a new env per episode as the original notebook did.
            env = make_sepsis_env(verbose=False)
            for ep in range(n_episodes):
                # seed once so each trial's training run is reproducible
                obs, _ = env.reset(seed=seed if ep == 0 else None)
                done = False
                while not done:
                    a = agent.select_action(obs, training=True)
                    next_obs, r, term, trunc, _ = env.step(a)
                    done = term or trunc
                    agent.update(obs, a, r, next_obs, done)
                    obs = next_obs
            np.save(npy_path, agent.Q)
            with open(cfg_path, "w") as fh:
                json.dump(params, fh)
            print(f"[QL  {trial_idx:>3}] trained & saved", flush=True)

        res = eval_agent(
            agent.get_policy(), make_sepsis_env,
            n_eval_episodes=eval_episodes, seed=seed,
        )
        result = {
            "trial":         trial_idx,
            **params,
            "survival_rate": res["survival_rate"],
            "mean_return":   res["mean_return"],
        }
        print(
            f"[QL  {trial_idx:>3}] "
            f"alpha={params.get('alpha')}  "
            f"eps_decay={params.get('epsilon_decay')}  "
            f"eps_end={params.get('epsilon_end')}  "
            f"→ survival={res['survival_rate']:.1%}",
            flush=True,
        )
        return result

    except Exception as exc:
        print(f"[QL  {trial_idx:>3}] FAILED: {exc}", flush=True)
        return None


# ── SARSA trial (Config A) ────────────────────────────────────────────────────

def run_sarsa_trial(
    trial_idx: int,
    params: dict,
    save_dir: str,
    n_episodes: int,
    eval_episodes: int,
    seed: int,
    project_root: str,
) -> dict | None:
    """
    Run one SARSA hyperparameter trial. Safe to call from a joblib worker.

    Q-table cached as .npy; config cached as .json.
    Returns a result dict, or None if the trial fails.
    """
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    try:
        import numpy as np                                        # noqa: PLC0415
        from envs.env_setup import make_sepsis_env, N_STATES, N_ACTIONS  # noqa: PLC0415
        from agents.config_a.sarsa import SARSAAgent, SARSAConfig  # noqa: PLC0415
        from utils.evaluation import eval_agent                   # noqa: PLC0415

        os.makedirs(save_dir, exist_ok=True)
        npy_path = os.path.join(save_dir, f"sarsa_trial_{trial_idx}.npy")
        cfg_path = os.path.join(save_dir, f"sarsa_trial_{trial_idx}_config.json")

        cfg = SARSAConfig(n_episodes=n_episodes, **params)
        agent = SARSAAgent(N_STATES, N_ACTIONS, cfg)

        if os.path.exists(npy_path) and os.path.exists(cfg_path):
            agent.Q = np.load(npy_path)
            print(f"[SARSA {trial_idx:>3}] loaded from cache", flush=True)
        else:
            agent.train(make_sepsis_env, seed=seed)
            np.save(npy_path, agent.Q)
            with open(cfg_path, "w") as fh:
                json.dump(params, fh)
            print(f"[SARSA {trial_idx:>3}] trained & saved", flush=True)

        res = eval_agent(
            agent.get_policy(), make_sepsis_env,
            n_eval_episodes=eval_episodes, seed=seed,
        )
        result = {
            "trial":         trial_idx,
            **params,
            "survival_rate": res["survival_rate"],
            "mean_return":   res["mean_return"],
        }
        print(
            f"[SARSA {trial_idx:>3}] "
            f"alpha={params.get('alpha')}  "
            f"gamma={params.get('gamma')}  "
            f"eps_decay={params.get('epsilon_decay')}  "
            f"→ survival={res['survival_rate']:.1%}",
            flush=True,
        )
        return result

    except Exception as exc:
        print(f"[SARSA {trial_idx:>3}] FAILED: {exc}", flush=True)
        return None
