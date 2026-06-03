"""Dynamic Programming agents for Config A (discrete 716-state sepsis MDP).

Provides model-based Policy Iteration and Value Iteration. Both algorithms
require the full MDP transition matrix P(s'|s,a) and reward matrix R(s,a,s'),
which are available in Config A via the ICU-Sepsis-v2 environment internals.
"""

import numpy as np


def compute_expected_reward(P: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Compute expected reward R_exp[s,a] = sum_{s'} P(s'|s,a) * R(s,a,s').

    Args:
        P: Transition tensor of shape (S, A, S').
        R: Reward tensor of shape (S, A, S').

    Returns:
        R_exp: Expected reward matrix of shape (S, A).
    """
    return np.sum(P * R, axis=2)


def compute_value_function(
    policy: np.ndarray,
    P: np.ndarray,
    R_expected: np.ndarray,
    gamma: float = 1.0,
    tol: float = 1e-6,
    max_iter: int = 10_000,
) -> np.ndarray:
    """Evaluate a fixed policy via iterative Bellman expectation backup.

    Args:
        policy:     Integer array of shape (S,) mapping each state to an action.
        P:          Transition tensor (S, A, S').
        R_expected: Expected reward matrix (S, A) from compute_expected_reward.
        gamma:      Discount factor.
        tol:        Convergence threshold on max|ΔV|.
        max_iter:   Maximum number of iterations.

    Returns:
        V: Value vector of shape (S,).
    """
    n_states = P.shape[0]
    V = np.zeros(n_states)
    for _ in range(max_iter):
        V_new = np.array([
            np.sum(P[s, policy[s]] * (R_expected[s, policy[s]] + gamma * V))
            for s in range(n_states)
        ])
        if np.max(np.abs(V_new - V)) < tol:
            break
        V = V_new
    return V


def extract_policy(
    V: np.ndarray,
    P: np.ndarray,
    R_expected: np.ndarray,
    gamma: float = 1.0,
) -> np.ndarray:
    """Extract the greedy policy from a value function.

    Args:
        V:          Value vector of shape (S,).
        P:          Transition tensor (S, A, S').
        R_expected: Expected reward matrix (S, A).
        gamma:      Discount factor.

    Returns:
        policy: Integer array of shape (S,) with the greedy action per state.
    """
    n_states, n_actions = R_expected.shape
    Q = np.array([
        [np.sum(P[s, a] * (R_expected[s, a] + gamma * V)) for a in range(n_actions)]
        for s in range(n_states)
    ])
    return np.argmax(Q, axis=1)


def policy_iteration(
    P: np.ndarray,
    R: np.ndarray,
    gamma: float = 1.0,
    tol: float = 1e-6,
    max_iter: int = 1_000,
) -> tuple:
    """Run Policy Iteration to convergence.

    Alternates between full policy evaluation (Bellman expectation) and greedy
    policy improvement until the policy stops changing.

    Args:
        P:       Transition tensor (S, A, S').
        R:       Reward tensor (S, A, S').
        gamma:   Discount factor.
        tol:     Convergence threshold for inner policy evaluation.
        max_iter: Maximum number of outer PI iterations.

    Returns:
        policy: Optimal policy array (S,).
        V:      Value function under the optimal policy (S,).
        deltas: List of policy-change counts per iteration (for convergence plots).
    """
    R_exp = compute_expected_reward(P, R)
    n_states = P.shape[0]
    policy = np.zeros(n_states, dtype=int)
    deltas = []
    for i in range(max_iter):
        V = compute_value_function(policy, P, R_exp, gamma, tol)
        new_policy = extract_policy(V, P, R_exp, gamma)
        delta = int(np.sum(new_policy != policy))
        deltas.append(delta)
        if np.array_equal(new_policy, policy):
            print(f"Policy Iteration converged after {i + 1} iteration(s).")
            break
        policy = new_policy
    return policy, V, deltas


def value_iteration(
    P: np.ndarray,
    R: np.ndarray,
    gamma: float = 1.0,
    tol: float = 1e-6,
    max_iter: int = 10_000,
) -> tuple:
    """Run Value Iteration to convergence.

    Applies the Bellman optimality operator at every state until max|ΔV| < tol.

    Args:
        P:       Transition tensor (S, A, S').
        R:       Reward tensor (S, A, S').
        gamma:   Discount factor.
        tol:     Convergence threshold on max|ΔV|.
        max_iter: Maximum number of sweeps.

    Returns:
        policy: Greedy policy derived from the converged value function (S,).
        V:      Converged value function (S,).
        deltas: List of max|ΔV| per sweep (for convergence plots).
    """
    R_exp = compute_expected_reward(P, R)
    n_states, n_actions = R_exp.shape
    V = np.zeros(n_states)
    Q = np.zeros((n_states, n_actions))
    deltas = []
    for i in range(max_iter):
        Q = np.array([
            [np.sum(P[s, a] * (R_exp[s, a] + gamma * V)) for a in range(n_actions)]
            for s in range(n_states)
        ])
        V_new = np.max(Q, axis=1)
        delta = float(np.max(np.abs(V_new - V)))
        deltas.append(delta)
        V = V_new
        if delta < tol:
            print(f"Value Iteration converged after {i + 1} iteration(s).")
            break
    policy = np.argmax(Q, axis=1)
    return policy, V, deltas


def get_dp_policy(policy_array: np.ndarray):
    """Wrap a DP policy array as a callable for use with eval_agent().

    Args:
        policy_array: Integer array of shape (S,) mapping states to actions.

    Returns:
        Callable: obs -> action (int).
    """
    return lambda obs: int(policy_array[obs])
