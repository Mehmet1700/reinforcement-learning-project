"""
PPO agent for Config B (47-dim continuous observation sepsis environment).
Uses stable-baselines3 under the hood.
"""

from stable_baselines3 import PPO
import numpy as np


def make_ppo_agent(env_factory, total_timesteps: int = 200_000, seed: int = 42):
    """
    Train a PPO agent.

    Args:
        env_factory: zero-argument callable returning a fresh gymnasium env.
        total_timesteps: total environment steps to train for.
        seed: random seed.

    Returns:
        Trained SB3 PPO model.
    """
    env = env_factory()

    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=1.0,
        gae_lambda=0.95,
        clip_range=0.2,
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
