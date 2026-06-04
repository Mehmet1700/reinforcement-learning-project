"""
Worker function for parallel policy evaluation across environments.
Must live in a real module (not a notebook cell) so multiprocessing
spawn can pickle and import it on macOS / Python 3.13.
"""

from __future__ import annotations
import os, sys
import numpy as np


def eval_worker(args: tuple) -> np.ndarray:
    """
    Evaluate a policy for n_episodes in one environment and return
    a binary survival array (1 = survived, 0 = died).

    args = (model_type, model_path, env_type, n_episodes, seed, project_root)
    """
    model_type, model_path, env_type, n_episodes, seed, project_root = args

    # Make sure the project is importable in the fresh worker process
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    os.chdir(project_root)

    from agents.config_b.dqn import load_dqn
    from agents.config_b.ppo import load_ppo
    from envs.wrappers import make_clinical_env

    model = load_dqn(model_path) if model_type == "dqn" else load_ppo(model_path)
    policy_fn = lambda obs: int(
        model.predict(np.array(obs, dtype=np.float32), deterministic=True)[0]
    )

    if env_type == "clean":
        env_factory = lambda: make_clinical_env(
            malfunction_prob=0.0, missing_prob=0.0, event_prob=0.0
        )
    else:
        env_factory = make_clinical_env

    env = env_factory()
    outcomes: list[int] = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        done, total = False, 0.0
        while not done:
            obs, r, term, trunc, _ = env.step(policy_fn(obs))
            total += r
            done = term or trunc
        outcomes.append(1 if total > 0 else 0)
    env.close()
    return np.array(outcomes, dtype=float)
