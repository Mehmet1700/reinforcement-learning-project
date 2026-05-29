"""
PPO agent for Config B — 47-dim continuous observation sepsis environment.

Implementation notes
--------------------
- Uses Stable-Baselines3 PPO with a custom MLP policy.
- gamma=1.0 matches the environment convention (undiscounted returns).
- ent_coef=0.01 adds a small entropy bonus to encourage exploration across
  25 actions on what is otherwise a short-horizon (~10 steps) episode.
- n_steps=1024 means ~100 complete episodes per update, giving a stable
  gradient estimate before clipping.
- RewardTrackingCallback logs episode returns during training so we can
  plot a proper learning curve without relying on SB3's verbose output.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor

SEED = 42


# ── Hyperparameter config ────────────────────────────────────────────────────

@dataclass
class PPOConfig:
    # Training
    total_timesteps: int = 100_000
    seed: int = SEED

    # PPO core
    learning_rate: float = 3e-4
    n_steps: int = 1024        # env steps collected before each update
    batch_size: int = 64       # minibatch size for gradient updates
    n_epochs: int = 10         # passes over each collected batch
    gamma: float = 1.0         # undiscounted (match env)
    gae_lambda: float = 0.95   # GAE smoothing factor
    clip_range: float = 0.2    # PPO clip epsilon
    ent_coef: float = 0.01     # entropy bonus — encourages exploration
    vf_coef: float = 0.5       # value function loss weight
    max_grad_norm: float = 0.5 # gradient clipping

    # Network architecture
    net_arch: list = field(default_factory=lambda: [64, 64])
    activation_fn: type = None   # None → SB3 default (Tanh); pass e.g. torch.nn.ReLU

    def __post_init__(self):
        # CSV/JSON loading can silently convert ints to floats — enforce types
        self.total_timesteps = int(self.total_timesteps)
        self.seed            = int(self.seed)
        self.n_steps         = int(self.n_steps)
        self.batch_size      = int(self.batch_size)
        self.n_epochs        = int(self.n_epochs)


# ── Callback: track episode rewards during training ──────────────────────────

class RewardTrackingCallback(BaseCallback):
    """
    Stores mean episode return after every completed rollout.
    Use .episode_returns for the raw per-episode values and
    .rollout_means for a smoother curve (mean over each rollout batch).
    """

    def __init__(self):
        super().__init__()
        self.episode_returns: list[float] = []
        self.episode_survivals: list[int] = []
        self.episode_timesteps: list[int] = []
        self.rollout_means: list[float] = []
        self._current_rollout: list[float] = []

    def _on_step(self) -> bool:
        # SB3 populates infos with 'episode' dict when an episode ends
        for info in self.locals.get("infos", []):
            if "episode" in info:
                ret = float(info["episode"]["r"])
                self.episode_returns.append(ret)
                self.episode_survivals.append(1 if ret > 0 else 0)
                self.episode_timesteps.append(self.num_timesteps)
                self._current_rollout.append(ret)
        return True

    def _on_rollout_end(self) -> None:
        if self._current_rollout:
            self.rollout_means.append(float(np.mean(self._current_rollout)))
            self._current_rollout = []


# ── Training ─────────────────────────────────────────────────────────────────

def train_ppo(
    env_factory: Callable,
    cfg: PPOConfig = None,
    save_path: str = "results/config_b/ppo_model",
    callback: BaseCallback = None,
) -> tuple[PPO, RewardTrackingCallback]:
    """
    Train a PPO agent on the clinical sepsis environment.

    Args:
        env_factory : zero-argument callable returning a fresh gymnasium env.
        cfg         : PPOConfig (uses defaults if None).
        save_path   : where to save the trained model (no extension needed).
        callback    : optional SB3 callback; defaults to RewardTrackingCallback.
                      Pass a CheckpointCallback (utils/callbacks.py) to enable
                      mid-training evaluation and best-checkpoint selection.

    Returns:
        (model, callback) — trained SB3 PPO model and the callback used.
    """
    cfg = cfg or PPOConfig()

    # Wrap env in Monitor so SB3 gets episode stats
    env = Monitor(env_factory())

    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=cfg.learning_rate,
        n_steps=cfg.n_steps,
        batch_size=cfg.batch_size,
        n_epochs=cfg.n_epochs,
        gamma=cfg.gamma,
        gae_lambda=cfg.gae_lambda,
        clip_range=cfg.clip_range,
        ent_coef=cfg.ent_coef,
        vf_coef=cfg.vf_coef,
        max_grad_norm=cfg.max_grad_norm,
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

def load_ppo(path: str = "results/config_b/ppo_model") -> PPO:
    """Load a previously saved PPO model."""
    return PPO.load(path)


# ── Policy wrapper ────────────────────────────────────────────────────────────

def get_policy(model: PPO) -> Callable:
    """Wrap a trained SB3 PPO model into a policy callable for eval_agent()."""
    def policy_fn(obs):
        action, _ = model.predict(obs, deterministic=True)
        return int(action)
    return policy_fn
