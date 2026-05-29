"""
DQN agent for Config B — 47-dim continuous observation sepsis environment.

Implementation notes
--------------------
- Uses Stable-Baselines3 DQN with a custom MLP policy.
- gamma=1.0 matches the environment convention (undiscounted returns).
- exploration_fraction=0.3 dedicates 30% of training to epsilon decay,
  giving the agent enough time to explore all 25 actions before exploiting.
- target_update_interval=500 keeps the target network stable relative to
  the short episode length (~10 steps).
- learning_starts=2_000 fills the replay buffer before the first gradient
  step, ensuring diverse early batches.
- gradient_steps=1 performs one gradient update per env step, keeping
  learning stable and avoiding overfit to recent transitions.
- RewardTrackingCallback mirrors the one in ppo.py so training curves are
  directly comparable between the two algorithms.
- train_dqn accepts an optional callback so the notebook can pass a
  CheckpointCallback (utils/callbacks.py) without duplicating model setup.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

SEED = 42


# ── Hyperparameter config ────────────────────────────────────────────────────

@dataclass
class DQNConfig:
    # Training
    total_timesteps: int = 100_000
    seed: int = SEED

    # DQN core
    learning_rate: float = 1e-4
    buffer_size: int = 50_000       # replay buffer capacity
    learning_starts: int = 2_000    # steps before first gradient update
    batch_size: int = 64
    gamma: float = 1.0              # undiscounted (match env)
    train_freq: int = 4             # update every N env steps
    gradient_steps: int = 1         # gradient updates per env step
    target_update_interval: int = 500  # steps between target net syncs

    # Exploration (epsilon-greedy)
    exploration_fraction: float = 0.3   # fraction of training for eps decay
    exploration_final_eps: float = 0.05

    # Network architecture
    net_arch: list = field(default_factory=lambda: [64, 64])
    activation_fn: type = None   # None → SB3 default (ReLU); pass e.g. torch.nn.Tanh

    def __post_init__(self):
        # CSV/JSON loading can silently convert ints to floats — enforce types
        self.total_timesteps         = int(self.total_timesteps)
        self.seed                    = int(self.seed)
        self.buffer_size             = int(self.buffer_size)
        self.learning_starts         = int(self.learning_starts)
        self.batch_size              = int(self.batch_size)
        self.train_freq              = int(self.train_freq)
        self.gradient_steps          = int(self.gradient_steps)
        self.target_update_interval  = int(self.target_update_interval)


# ── Callback: track episode rewards during training ──────────────────────────

class RewardTrackingCallback(BaseCallback):
    """
    Stores mean episode return after every completed rollout.
    Mirrors the PPO callback for directly comparable training curves.
    """

    def __init__(self):
        super().__init__()
        self.episode_returns: list[float] = []
        self.episode_survivals: list[int] = []
        self.episode_timesteps: list[int] = []
        self._window: list[float] = []
        self.rollout_means: list[float] = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" in info:
                ret = float(info["episode"]["r"])
                self.episode_returns.append(ret)
                self.episode_survivals.append(1 if ret > 0 else 0)
                self.episode_timesteps.append(self.num_timesteps)
                self._window.append(ret)
                if len(self._window) >= 100:
                    self.rollout_means.append(float(np.mean(self._window)))
                    self._window = []
        return True


# ── Training ─────────────────────────────────────────────────────────────────

def train_dqn(
    env_factory: Callable,
    cfg: DQNConfig = None,
    save_path: str = "results/config_b/dqn_model",
    callback: BaseCallback = None,
) -> tuple[DQN, RewardTrackingCallback]:
    """
    Train a DQN agent on the clinical sepsis environment.

    Args:
        env_factory : zero-argument callable returning a fresh gymnasium env.
        cfg         : DQNConfig (uses defaults if None).
        save_path   : where to save the trained model (no extension needed).
        callback    : optional SB3 callback; defaults to RewardTrackingCallback.
                      Pass a CheckpointCallback (utils/callbacks.py) to enable
                      mid-training evaluation and best-checkpoint selection.

    Returns:
        (model, callback) — trained SB3 DQN model and the callback used.
    """
    cfg = cfg or DQNConfig()

    env = Monitor(env_factory())

    model = DQN(
        policy="MlpPolicy",
        env=env,
        learning_rate=cfg.learning_rate,
        buffer_size=cfg.buffer_size,
        learning_starts=cfg.learning_starts,
        batch_size=cfg.batch_size,
        gamma=cfg.gamma,
        train_freq=cfg.train_freq,
        gradient_steps=cfg.gradient_steps,
        target_update_interval=cfg.target_update_interval,
        exploration_fraction=cfg.exploration_fraction,
        exploration_final_eps=cfg.exploration_final_eps,
        policy_kwargs={
            "net_arch":      cfg.net_arch,
            **({"activation_fn": cfg.activation_fn} if cfg.activation_fn is not None else {}),
        },
        verbose=0,
        seed=cfg.seed,
    )

    cb = callback if callback is not None else RewardTrackingCallback()
    model.learn(total_timesteps=cfg.total_timesteps, callback=cb)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.save(save_path)
    print(f"Model saved to {save_path}.zip")

    return model, cb


# ── Load ─────────────────────────────────────────────────────────────────────

def load_dqn(path: str = "results/config_b/dqn_model") -> DQN:
    """Load a previously saved DQN model."""
    return DQN.load(path)


# ── Policy wrapper ────────────────────────────────────────────────────────────

def get_policy(model: DQN) -> Callable:
    """Wrap a trained SB3 DQN model into a policy callable for eval_agent()."""
    def policy_fn(obs):
        action, _ = model.predict(obs, deterministic=True)
        return int(action)
    return policy_fn
