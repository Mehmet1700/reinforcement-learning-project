"""
DQN agent for Config B (47-dim continuous observation sepsis environment).
Uses stable-baselines3 under the hood.
"""

from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
import numpy as np


def make_dqn_agent(env_factory, total_timesteps: int = 200_000, seed: int = 42):
    """
    Train a DQN agent.

    Args:
        env_factory: zero-argument callable returning a fresh gymnasium env.
        total_timesteps: total environment steps to train for.
        seed: random seed.

    Returns:
        Trained SB3 DQN model.
    """
    env = env_factory()

    model = DQN(
        policy="MlpPolicy",
        env=env,
        learning_rate=1e-4,
        buffer_size=50_000,
        learning_starts=1_000,
        batch_size=64,
        gamma=1.0,
        target_update_interval=500,
        exploration_fraction=0.2,
        exploration_final_eps=0.05,
        verbose=1,
        seed=seed,
    )

    model.learn(total_timesteps=total_timesteps)
    return model


def get_policy(model) -> callable:
    """Wrap a trained SB3 model into a policy callable for eval_agent()."""
    def policy_fn(obs):
        action, _ = model.predict(obs, deterministic=True)
        return int(action)
    return policy_fn
