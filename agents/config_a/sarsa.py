"""SARSA (on-policy TD) agent for Config A (discrete 716-state sepsis MDP)."""

from dataclasses import dataclass
import numpy as np
from typing import Union

SEED = 42


@dataclass
class SARSAConfig:
    alpha: float = 0.1
    gamma: float = 0.9
    epsilon_start: float = 0.9
    epsilon_end: float = 0.05
    epsilon_decay: int = 5_000
    n_episodes: int = 10_000

    alpha_decay: str = 'constant'                # Options: 'constant' or 'linear'
    q_init: Union[float, str] = 0.0              # Options: 0.0, 1.0, or 'random'

    alpha_decay_steps: int = 5_000  # steps over which alpha decays linearly
    alpha_floor: float = 0.001       # minimum alpha after full decay

class SARSAAgent:
    
    def __init__(self, n_states: int, n_actions: int, cfg: SARSAConfig = None):
        self.cfg = cfg or SARSAConfig()
        self._step = 0
        self._rng = np.random.default_rng(SEED)
        
        # Initialize Q-table based on grid search parameters
        if self.cfg.q_init == 'random':
            # Small random uniform initialization between 0.0 and 0.1
            self.Q = self._rng.uniform(0.0, 0.1, size=(n_states, n_actions))
        else:
            # Constant initialization (e.g., pessimistic 0.0 or optimistic 1.0)
            self.Q = np.full((n_states, n_actions), float(self.cfg.q_init))



    @property
    def epsilon(self) -> float:
        cfg = self.cfg
        frac = min(self._step / cfg.epsilon_decay, 1.0)
        return cfg.epsilon_start + frac * (cfg.epsilon_end - cfg.epsilon_start)
    
   
    @property
    def current_alpha(self) -> float:
        """Calculates current alpha based on the selected decay strategy."""
        if self.cfg.alpha_decay == 'linear':
            # Linearly decay alpha down to a custom floor (Y) over a custom number of steps (X)
            frac = min(self._step / self.cfg.alpha_decay_steps, 1.0)
            return self.cfg.alpha + frac * (self.cfg.alpha_floor - self.cfg.alpha)
            
        return self.cfg.alpha

    def select_action(self, state: int, training: bool = True) -> int:
        if training and self._rng.random() < self.epsilon:
            return int(self._rng.integers(self.Q.shape[1]))
        return int(np.argmax(self.Q[state]))

    def update(self, s: int, a: int, r: float, s_next: int, a_next: int, done: bool) -> None:
        cfg = self.cfg
        target = r if done else r + cfg.gamma * self.Q[s_next, a_next]
        # Applies the dynamically determined learning rate
        self.Q[s, a] += self.current_alpha * (target - self.Q[s, a])
        self._step += 1

    def train(self, env_factory, seed: int = SEED) -> list:
        """Train for cfg.n_episodes episodes. Returns list of episode returns.

        The environment is seeded once at the start so the whole training run is
        reproducible; subsequent resets advance the same deterministic RNG stream.
        """
        returns = []
        env = env_factory()
        for ep in range(self.cfg.n_episodes):
            obs, _ = env.reset(seed=seed if ep == 0 else None)
            a = self.select_action(obs, training=True)
            done = False
            total = 0.0
            while not done:
                next_obs, r, terminated, truncated, _ = env.step(a)
                done = terminated or truncated
                a_next = self.select_action(next_obs, training=True)
                self.update(obs, a, r, next_obs, a_next, done)
                obs, a = next_obs, a_next
                total += r
            returns.append(total)
        return returns

    def get_policy(self):
        """Return greedy policy callable for use with utils.evaluation.eval_agent()."""
        return lambda obs: int(np.argmax(self.Q[obs]))

