"""
Shared SB3 callbacks for Config B training cells.

CheckpointCallback extends the basic reward tracking with periodic policy
evaluation, saving the best-performing checkpoint seen during training.
This protects against the common DQN/PPO instability where the final model
is worse than an intermediate one.

Usage (in notebook):
    from utils.callbacks import CheckpointCallback

    callback = CheckpointCallback(
        eval_env_factory=make_clinical_env,
        eval_freq=10_000,
        n_eval_episodes=100,
        seed=SEED,
    )
    model, callback = train_dqn(make_clinical_env, cfg=cfg,
                                save_path=path, callback=callback)

    # Restore best checkpoint before saving / evaluating
    if callback.best_params is not None:
        model.set_parameters(callback.best_params)
"""

from __future__ import annotations

import copy
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class CheckpointCallback(BaseCallback):
    """
    Combined reward-tracking + periodic checkpoint evaluation callback.

    Stores the same fields as RewardTrackingCallback so all downstream
    training-curve cells work without modification:
        .episode_returns, .episode_survivals, .episode_timesteps, .rollout_means

    Additionally, every eval_freq environment steps it runs n_eval_episodes
    deterministic evaluation episodes, records the survival rate and saves
    a deep copy of the model parameters if this is the best result so far.

    After training, restore the best checkpoint with:
        model.set_parameters(callback.best_params)

    Attributes:
        eval_steps          : list of timesteps at which checkpoints were taken.
        eval_survival_rates : corresponding survival rates (0–1).
        best_survival       : best survival rate seen during training.
        best_step           : timestep of the best checkpoint.
        best_params         : model parameter dict of the best checkpoint.
    """

    def __init__(
        self,
        eval_env_factory,
        eval_freq: int = 10_000,
        n_eval_episodes: int = 100,
        seed: int = 42,
    ):
        super().__init__()
        self.eval_env_factory   = eval_env_factory
        self.eval_freq          = eval_freq
        self.n_eval_episodes    = n_eval_episodes
        self.seed               = seed

        # ── Reward tracking (compatible with RewardTrackingCallback) ──────────
        self.episode_returns:    list[float] = []
        self.episode_survivals:  list[int]   = []
        self.episode_timesteps:  list[int]   = []
        self.rollout_means:      list[float] = []
        self._rollout_buf:       list[float] = []

        # ── Checkpoint evaluation ─────────────────────────────────────────────
        self.eval_steps:          list[int]   = []
        self.eval_survival_rates: list[float] = []
        self.best_survival:       float       = -np.inf
        self.best_step:           int | None  = None
        self.best_params:         dict | None = None

    # ── Per-step hook ─────────────────────────────────────────────────────────

    def _on_step(self) -> bool:
        # Collect completed episode stats (same logic as RewardTrackingCallback)
        for info in self.locals.get("infos", []):
            if "episode" in info:
                ret = float(info["episode"]["r"])
                self.episode_returns.append(ret)
                self.episode_survivals.append(1 if ret > 0 else 0)
                self.episode_timesteps.append(self.num_timesteps)
                self._rollout_buf.append(ret)

        # Run checkpoint evaluation at the configured interval
        if self.num_timesteps % self.eval_freq == 0:
            surv_rate = self._evaluate()
            self.eval_steps.append(self.num_timesteps)
            self.eval_survival_rates.append(surv_rate)

            if surv_rate > self.best_survival:
                self.best_survival = surv_rate
                self.best_step     = self.num_timesteps
                self.best_params   = copy.deepcopy(self.model.get_parameters())

        return True

    def _on_rollout_end(self) -> None:
        # Flush the rollout buffer (matches PPO RewardTrackingCallback)
        if self._rollout_buf:
            self.rollout_means.append(float(np.mean(self._rollout_buf)))
            self._rollout_buf = []

    # ── Internal evaluation ───────────────────────────────────────────────────

    def _evaluate(self) -> float:
        """Run n_eval_episodes deterministic episodes and return survival rate."""
        env = self.eval_env_factory()
        survivals = []

        for ep in range(self.n_eval_episodes):
            obs, _ = env.reset(seed=self.seed + ep)
            done   = False
            total  = 0.0

            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, r, term, trunc, _ = env.step(int(action))
                total += r
                done   = term or trunc

            survivals.append(1 if total > 0 else 0)

        env.close()
        return float(np.mean(survivals))
