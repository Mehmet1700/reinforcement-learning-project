"""Q-Learning agent for Config A (discrete 716-state sepsis MDP)."""

from dataclasses import dataclass
import numpy as np
import time

SEED = 42


@dataclass
class QLearningConfig:
    alpha: float = 0.1
    gamma: float = 1.0
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: int = 50_000  # steps over which epsilon decays linearly (after exploration phase)
    n_episodes: int = 10_000
    exploration_episodes: int = 2_000  # episodes of pure exploration before any decay


class QLearningAgent:
    def __init__(self, n_states: int, n_actions: int, cfg: QLearningConfig = None):
        self.cfg = cfg or QLearningConfig()
        self.Q = np.zeros((n_states, n_actions))
        self._step = 0
        self._episode = 0  # tracks episodes for exploration phase gating
        self._rng = np.random.default_rng(SEED)

    @property
    def epsilon(self) -> float:
        cfg = self.cfg
        if self._episode < cfg.exploration_episodes:
            return cfg.epsilon_start  # flat exploration phase
        frac = min(self._step / cfg.epsilon_decay, 1.0)
        return cfg.epsilon_start + frac * (cfg.epsilon_end - cfg.epsilon_start)

    def select_action(self, state: int, training: bool = True) -> int:
        if training and self._rng.random() < self.epsilon:
            return int(self._rng.integers(self.Q.shape[1]))
        return int(np.argmax(self.Q[state]))

    def update(self, s: int, a: int, r: float, s_next: int, done: bool) -> None:
        cfg = self.cfg
        target = r if done else r + cfg.gamma * np.max(self.Q[s_next])
        self.Q[s, a] += cfg.alpha * (target - self.Q[s, a])
        self._step += 1

    def train(self, env_factory, verbose: bool = True, verbose_interval: int = 1000) -> list:
        returns = []
        start_time = time.time()

        for episode in range(self.cfg.n_episodes):
            self._episode = episode  # update before select_action is called
            if episode == self.cfg.exploration_episodes:
                self._step = 0  # reset decay clock so decay starts fresh at episode 2000

            env = env_factory()
            obs, _ = env.reset()
            done = False
            total = 0.0

            while not done:
                a = self.select_action(obs, training=True)
                next_obs, r, terminated, truncated, _ = env.step(a)
                done = terminated or truncated
                self.update(obs, a, r, next_obs, done)
                obs = next_obs
                total += r

            returns.append(total)

            if verbose and (episode + 1) % verbose_interval == 0:
                elapsed = time.time() - start_time
                avg_return = np.mean(returns[-verbose_interval:])
                episodes_per_sec = (episode + 1) / elapsed
                eta_remaining = (self.cfg.n_episodes - episode - 1) / episodes_per_sec if episodes_per_sec > 0 else 0

                print(f"Episode {episode + 1}/{self.cfg.n_episodes} | "
                      f"Avg Return (last {verbose_interval}): {avg_return:.3f} | "
                      f"Epsilon: {self.epsilon:.4f} | "
                      f"Elapsed: {elapsed:.1f}s | ETA: {eta_remaining:.1f}s")

        if verbose:
            total_time = time.time() - start_time
            print(f"\n✓ Training complete! Total time: {total_time:.1f}s")

        return returns

    def get_policy(self):
        return lambda obs: int(np.argmax(self.Q[obs]))