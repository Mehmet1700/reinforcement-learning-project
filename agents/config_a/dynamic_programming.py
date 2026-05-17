"""
Dynamic Programming for Config A (discrete 716-state sepsis MDP).

Policy Iteration and Value Iteration using the full MDP model (P, R)
accessible via the environment. This gives the optimal policy, which
serves as an upper-bound benchmark for Q-Learning and SARSA.

Usage:
    from envs.env_setup import make_sepsis_env
    env = make_sepsis_env()
    P = env.unwrapped.P   # transition matrix [n_states, n_actions, n_states]
    R = env.unwrapped.R   # reward matrix     [n_states, n_actions]
"""

import numpy as np


def policy_iteration(P: np.ndarray, R: np.ndarray, gamma: float = 1.0, tol: float = 1e-6) -> tuple:
    """
    Exact Policy Iteration on a tabular MDP.

    Args:
        P: transition tensor of shape (n_states, n_actions, n_states).
        R: reward matrix of shape (n_states, n_actions).
        gamma: discount factor (1.0 to match env convention).
        tol: convergence threshold for policy evaluation.

    Returns:
        (policy, V) — greedy policy array of shape (n_states,) and
        value function array of shape (n_states,).
    """
    n_states, n_actions, _ = P.shape
    policy = np.zeros(n_states, dtype=int)

    while True:
        # Policy evaluation: solve V = R_pi + gamma * P_pi @ V
        V = _evaluate_policy(policy, P, R, gamma, tol)

        # Policy improvement
        Q = R + gamma * np.einsum("san,n->sa", P, V)
        new_policy = np.argmax(Q, axis=1)

        if np.all(new_policy == policy):
            break
        policy = new_policy

    return policy, V


def value_iteration(P: np.ndarray, R: np.ndarray, gamma: float = 1.0, tol: float = 1e-6) -> tuple:
    """
    Value Iteration on a tabular MDP.

    Args:
        P: transition tensor of shape (n_states, n_actions, n_states).
        R: reward matrix of shape (n_states, n_actions).
        gamma: discount factor.
        tol: convergence threshold (max Bellman residual).

    Returns:
        (policy, V) — greedy policy array and value function array.
    """
    n_states, n_actions, _ = P.shape
    V = np.zeros(n_states)

    while True:
        Q = R + gamma * np.einsum("san,n->sa", P, V)
        V_new = np.max(Q, axis=1)
        if np.max(np.abs(V_new - V)) < tol:
            break
        V = V_new

    policy = np.argmax(Q, axis=1)
    return policy, V


def _evaluate_policy(
    policy: np.ndarray, P: np.ndarray, R: np.ndarray, gamma: float, tol: float
) -> np.ndarray:
    """Iterative policy evaluation until convergence."""
    n_states = len(policy)
    V = np.zeros(n_states)
    while True:
        r_pi = R[np.arange(n_states), policy]
        p_pi = P[np.arange(n_states), policy, :]
        V_new = r_pi + gamma * p_pi @ V
        if np.max(np.abs(V_new - V)) < tol:
            return V_new
        V = V_new


def get_policy(policy_array: np.ndarray):
    """Wrap a policy array into a callable for use with utils.evaluation.eval_agent()."""
    return lambda obs: int(policy_array[obs])
