"""
Fixed evaluation protocol. Do not change eval_agent() once results are being reported —
all numbers in the report must come from this function with n_eval_episodes=1000, seed=42.
"""

import numpy as np
from typing import Callable


def eval_agent(
    policy_fn: Callable,
    env_factory: Callable,
    n_eval_episodes: int = 1000,
    seed: int = 42,
) -> dict:
    """
    Evaluate a policy for n_eval_episodes episodes.

    Args:
        policy_fn: callable(obs) -> action. Works for both discrete (int obs)
                   and continuous (np.ndarray obs) environments.
        env_factory: zero-argument callable returning a fresh env,
                     e.g. make_sepsis_env or make_clinical_env.
        n_eval_episodes: number of evaluation episodes.
        seed: master seed for reproducibility.

    Returns:
        dict with keys:
            mean_return (float), std_return (float), survival_rate (float),
            mean_ep_length (float), returns (np.ndarray)
    """
    returns = []
    ep_lengths = []
    survivals = []

    env = env_factory()

    for ep in range(n_eval_episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        total_reward = 0.0
        steps = 0

        while not done:
            action = policy_fn(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            steps += 1

        returns.append(total_reward)
        ep_lengths.append(steps)
        # survival: episode ended with positive final reward
        survivals.append(1 if total_reward > 0 else 0)

    env.close()

    returns = np.array(returns)
    return {
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "survival_rate": float(np.mean(survivals)),
        "mean_ep_length": float(np.mean(ep_lengths)),
        "returns": returns,
    }


def print_results(label: str, results: dict) -> None:
    print(f"=== {label} ===")
    print(f"  Mean return  : {results['mean_return']:.4f} ± {results['std_return']:.4f}")
    print(f"  Survival rate: {results['survival_rate']:.1%}")
    print(f"  Mean ep len  : {results['mean_ep_length']:.1f}")
