"""Q-Learning agent for Config A (discrete 716-state sepsis MDP)."""

from dataclasses import dataclass
import numpy as np

SEED = 42


@dataclass
class QLearningConfig:
    alpha: float = 0.1
    gamma: float = 1.0
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: int = 50_000  # steps over which epsilon decays linearly
    n_episodes: int = 10_000


class QLearningAgent:
    def __init__(self, n_states: int, n_actions: int, cfg: QLearningConfig = None):
        self.cfg = cfg or QLearningConfig()
        self.Q = np.zeros((n_states, n_actions))
        self._step = 0
        self._rng = np.random.default_rng(SEED)

    @property
    def epsilon(self) -> float:
        cfg = self.cfg
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

    def train(self, env_factory) -> list:
        """Train for cfg.n_episodes episodes. Returns list of episode returns."""
        returns = []
        for _ in range(self.cfg.n_episodes):
            obs, _ = env_factory().reset()
            done = False
            total = 0.0
            while not done:
                a = self.select_action(obs, training=True)
                next_obs, r, terminated, truncated, _ = env_factory().step(a)
                done = terminated or truncated
                self.update(obs, a, r, next_obs, done)
                obs = next_obs
                total += r
            returns.append(total)
        return returns

    def get_policy(self):
        """Return greedy policy callable for use with utils.evaluation.eval_agent()."""
        return lambda obs: int(np.argmax(self.Q[obs]))
