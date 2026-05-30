

**CAPA/PRESENTATION**

# **Table of Contents**

[**1\. Introduction	3**](#introduction)

[**2\. Methodology	3**](#methodology)

[2.1 Configuration A: Discrete State Space	3](#2.1-configuration-a:-discrete-state-space-\(0.80%\))

[2.1.1 Baseline (Random Search)	3](#2.1.1-baseline-\(random-search\))

2.1.2 Dynamic programming	3

[2.1.3 TD Learning	3](#2.1.3-td-learning)

[2.1.4 Q- Learning	3](#2.1.4-q--learning)

[2.1.5 SARSA	3](#2.1.5-sarsa)

[2.1.6 Extension	3](#2.1.6-extension)

[2.2 Configuration B: Continuous Observation Space	3](#heading=h.rcfz1ujem8gb)

[2.2.1 Baseline (Random Search)	3](#heading=h.hbrn0jop4dk8)

[2.2.2 DQN	3](#heading=h.i5wpe7rw1sl6)

[2.2.3 PPO	3](#heading=h.pfcdqxnftr23)

[2.2.4 Extension	3](#heading=h.98kyr5r13tu8)

[2.3 Extension	3](#2.3-extension)

[**3\. Evaluation	4**](#evaluation)

[3.1 Configuration A: Discrete State Space	4](#3.1-configuration-a:-discrete-state-space)

[2.2 Configuration B: Continuous Observation Space	4](#3.2-configuration-b:-continuous-observation-space)

[**4\. Conclusion	4**](#conclusion)

[**5\. References	5**](#references)

[**6\. Annex	6**](#annex)

# 

# 

1. # Introduction {#introduction}

(Explain the clinical problem, why RL is appropriate, and your overall approach.)

TEXTTTTTTTTTTTTTT

2. # Methodology  {#methodology}

(Explain Description of each algorithm (policy or off policy, model based or free, function approximation or not, bootstrap, ect), why you chose it (pros, con), and key hyperparameters.)

TEXTTTTTTTTTTTTTT

## 2.1 Configuration A: Discrete State Space (0.80%) {#2.1-configuration-a:-discrete-state-space-(0.80%)}

TEXTTTTTTTTTTTTTT

### 2.1.1 Baseline (Random Search) {#2.1.1-baseline-(random-search)}

TEXTTTTTTTTTTTTTT

### 2.1.2 Dynamic programming (DP)

Dynamic Programming (DP) is a model-based method that computes the optimal policy directly from the known transition matrix P(s'|s,a) and reward matrix R(s,a,s'), without requiring environment interaction during training. This makes it well-suited to ICU-Sepsis, where full MDP dynamics are available and unnecessary exploration raises patient safety concerns.

Two algorithms were implemented: Policy Iteration, which alternates between full policy evaluation (Bellman expectation equation) and greedy policy improvement until the policy stabilises and Value Iteration, which combines both steps into a single Bellman optimality update per iteration. Both used a convergence threshold of 1e-6 and were tested across γ ∈ {0.90, 0.95, 0.99, 1.0}, with γ \= 1.0 adopted as the final setting following the ICU-Sepsis paper convention.

Bellman expectation equation (Policy Iteration):  V(s) \= Σ\_{s'} P(s'|s,a) · \[R(s,a) \+ γ · V(s')\] 

Bellman optimality equation (Value Iteration): V(s) \= max\_a Σ\_{s'} P(s'|s,a) · \[R(s,a) \+ γ · V(s')\]

### 2.1.3 TD Learning {#2.1.3-td-learning}

TEXTTTTTTTTTTTTTT

### 2.1.4 Q- Learning {#2.1.4-q--learning}

TEXTTTTTTTTTTTTTT

### 2.1.5 SARSA {#2.1.5-sarsa}

TEXTTTTTTTTTTTTTT

### 2.1.6 Extension {#2.1.6-extension}

If selected

## 2.2 Configuration B: Continuous Observation Space

Configuration B exposes the agent to the raw 47-dimensional physiological feature vector from the MIMIC-III database: SOFA score, heart rate, lactate, blood pressure, creatinine, and 42 additional clinical variables. Because the observation space is now continuous, the agent never encounters the exact same state twice. Tabular methods such as Q-Learning and SARSA are no longer applicable — a table indexed by state would require infinitely many entries. Function approximation via neural networks becomes necessary.

We selected **Deep Q-Network (DQN)** and **Proximal Policy Optimisation (PPO)** as our two algorithms. This pairing is deliberate: the two algorithms represent fundamentally different RL paradigms, making their comparison informative beyond raw performance numbers. Both are implemented using Stable-Baselines3 with custom MLP policies and trained with `gamma = 1.0`, matching the environment convention of undiscounted episodic returns.

2.2.1 Deep Q-Network (DQN)

**DQN** is an off-policy, value-based method. It approximates the action-value function Q(s, a) with a neural network and trains on mini-batches drawn from an experience replay buffer. Two mechanisms address the instability that arises when neural networks are used as function approximators: (i) the replay buffer breaks temporal correlations in the training data, and (ii) a periodically synchronised target network provides stable regression targets. DQN is a natural extension of Q-Learning to the continuous observation setting and is well-suited to the discrete 25-action space of ICU-Sepsis-v2.

The network architecture uses two fully-connected layers of 64 units with ReLU activations (Kaiming uniform initialisation — the PyTorch default and the appropriate choice for ReLU networks). Key hyperparameter choices:

| Parameter | Value | Rationale |
| ----- | ----- | ----- |
| `learning_rate` | 1 × 10⁻⁴ | Conservative; avoids oscillation on short (≈10 step) episodes |
| `buffer_size` | 25,000 | Sufficient replay diversity without excessive memory for episode length |
| `exploration_fraction` | 0.25 | 25k steps for ε-decay; ensures all 25 actions are explored before exploitation |
| `exploration_final_eps` | 0.10 | Retains 10% exploration throughout; clinical diversity in actions is valuable |
| `target_update_interval` | 500 | Keeps target network stable relative to episode length |
| `gradient_steps` | 1 | One gradient update per environment step; prevents overfitting to recent transitions |
| `learning_starts` | 2,000 | Fills replay buffer with diverse transitions before the first update |

2.2.2 Proximal Policy Optimisation (PPO)

**PPO** is an on-policy, policy-gradient method. Rather than learning a value function and deriving a greedy policy, PPO directly parameterises the policy π(a|s) and optimises it via a clipped surrogate objective. The clipping prevents the policy from moving too far from the current iterate in a single update, which is the key stability improvement over vanilla policy gradient methods. PPO is the standard deep RL baseline for environments with discrete actions and moderate episode lengths, and its on-policy nature provides a useful contrast to DQN's replay-based training.

The network uses two layers of 128 units with ReLU activations. PPO's on-policy nature requires larger rollouts to reduce gradient variance, and the actor-critic architecture benefits from a wider network than DQN.

| Parameter | Value | Rationale |
| ----- | ----- | ----- |
| `learning_rate` | 1 × 10⁻⁴ | Matched to DQN for fair comparison; stable for small batches |
| `n_steps` | 1,024 | ≈100 complete episodes per rollout; stable gradient estimate before clipping |
| `n_epochs` | 20 | Multiple passes over each rollout; maximises sample efficiency |
| `ent_coef` | 0.005 | Small entropy bonus encourages exploration across 25 actions |
| `clip_range` | 0.3 | Slightly relaxed from the default 0.2; benefits short-horizon episodes |
| `net_arch` | \[128, 128\] | Larger network needed for policy \+ value function heads |

2.2.3 Hyperparameter Search

Choosing hyperparameters by intuition alone risks leaving performance on the table. We conducted a two-stage search to identify the best configuration for each algorithm.

**Stage 1 — Exhaustive grid search.** We defined a multi-dimensional grid over the most impactful hyperparameters (486 DQN configurations, 648 PPO configurations) and trained each combination for 100,000 steps. The grid covered learning rate, exploration schedule, network architecture, activation function, buffer size, and exploration final epsilon (DQN), or learning rate, rollout length, entropy coefficient, architecture, activation, epochs, and clip range (PPO). All grid trials were run with fixed seed 42 and evaluated over 200 episodes.

**Stage 2 — Bayesian optimisation with Optuna.** Grid search fixes two dimensions that were later highlighted as important in the course Lab notebooks: `target_update_interval` for DQN and the `[128, 64]` asymmetric architecture for both algorithms. We used Optuna's Tree-structured Parzen Estimator (TPE) to run 20 additional trials per algorithm, each at the full 100k-step budget, covering these new dimensions. To enable parallel execution while preserving the sequential feedback loop that TPE requires, we used the ask-and-tell pattern with joblib batches of 4 workers. The grid-search best configuration was enqueued as trial 0 in both studies, ensuring a fair head-to-head comparison at identical budget.

2.2.4 Best-Checkpoint Selection

Training curves for DQN in deep RL settings frequently exhibit non-monotonic behaviour: the agent may reach a strong policy mid-training that subsequently degrades due to replay buffer shifts or exploration noise. To guard against saving an inferior final-step model, we implemented a `CheckpointCallback` that evaluates the current policy over 100 deterministic episodes every 10,000 steps and records the survival rate at each checkpoint. At the end of training, the best-performing snapshot is restored before the model is saved. The full training curve and checkpoint evaluation results are presented in Section 3.2.

## 2.3 Extension {#2.3-extension}

If selected

3. # Evaluation  {#evaluation}

(training curves, results table, convergence analysis, algorithms comparison and Config A vs Config B comparison)

(include **Visualizations**: Meaningful plots to represent learning and performance.)

TEXTTTTTTTTTTTTTT

## 3.1 Configuration A: Discrete State Space {#3.1-configuration-a:-discrete-state-space}

### 

### 3.1.1 Dynamic Programming

Both Policy Iteration and Value Iteration successfully converged to the optimal policy for the ICU-Sepsis MDP. Policy Iteration converged in 4 iterations, while Value Iteration required 135 iterations, which are expected, since Policy Iteration converges in fewer outer iterations and Value Iteration performs partial updates and requires more sweeps to propagate values accurately across all states.

Despite the difference in convergence speed, both algorithms produced identical final policies. The evaluation results over 1,000 episodes with γ \= 1.0 are 83.2% of survival rate, which represents a substantial improvement over a random treatment policy (73.4%). This demonstrates that even a relatively simple model-based approach can learn clinically meaningful treatment strategies when the full MDP dynamics are available.

The grid search over discount factors revealed that results were stable across γ values, with γ \= 1.0 performing consistently well. This aligns with the ICU-Sepsis paper convention, where no temporal discounting is applied, all time steps within a patient episode are treated equally, which is appropriate in a clinical setting where the outcome (survival or death) matters regardless of when it occurs.

A key limitation of DP in this context is its dependence on a known model. In real clinical practice, the true transition probabilities are never fully known, they are estimated from historical data. This means DP results represent an upper bound on what model-based methods can achieve given perfect knowledge, and model-free approaches explored in the following sections must close this gap through experience.

## 3.2 Configuration B: Continuous Observation Space {#3.2-configuration-b:-continuous-observation-space}

### 2.2.3 Training and Best-Checkpoint Selection

Training curves for DQN in RL settings frequently exhibit non-monotonic behaviour: the agent may reach a good policy mid-training that subsequently degrades due to replay buffer shifts or exploration noise. To guard against this, we implemented a `CheckpointCallback` that evaluates the current policy over 100 determiniy\<stic episodes every 10,000 steps and saves the best-performing weight snapshot. At the end of training, the best checkpoint is restored before the model is saved.

**DQN** reached its best checkpoint at step 20,000 with **75.0% clinical survival**, then oscillated between 61–72% for the remainder of training (Figure 1). The final-step policy achieved only 70%. Without checkpoint selection, we would have saved the inferior final policy.

**PPO** exhibited more stable training — survival rates grew gradually from 67% to a peak of **72.0% at step 70,000** — consistent with PPO's on-policy stability. The final evaluation also reached 70%, confirming less regression than DQN.

The clinical environment survival rates are somewhat lower than what the clean-environment numbers suggest (≈78% DQN, ≈79% PPO on clean env). This gap is explained by the three clinical wrappers: `EpisodicNoisyObsEnv` adds Gaussian noise to observations, `EpisodicMissingObsEnv` randomly zeros out feature dimensions, and `AcuteEventEnv` introduces a 1% per-step forced-death probability that is independent of the policy. This last wrapper alone forces approximately 9% of episodes to end in death regardless of agent behaviour, creating a hard ceiling of ≈91% on clinical survival. The gap between random baseline (67.2%) and our agents (72–75%) therefore represents genuine policy quality.

### 2.2.4 Robustness Analysis

#### **Multi-seed evaluation**

A single seed can produce misleadingly good or bad results due to lucky initialisations or exploration trajectories. We retrained both algorithms with seeds 0, 1, and 42, and evaluated each over 500 episodes. DQN shows a standard deviation of **±2.9 percentage points** across seeds; PPO is more stable at **±1.8 pp**. This is consistent with DQN's higher sensitivity to early exploration and replay buffer composition.

#### **Weight initialisation**

We investigated whether the default weight initialisation strategy is optimal by comparing three methods applied at construction time: Kaiming Uniform (PyTorch default, designed for ReLU), Orthogonal (SB3 default for policy gradient, preserves gradient norms), and Xavier Normal (designed for sigmoid/tanh activations).

Each algorithm's own default is its best, and swapping them hurts:

* **DQN \+ Kaiming: 78.2% clean** vs Orthogonal 73.6% (−4.6 pp) and Xavier 73.4% (−4.8 pp). Kaiming scales weights by √(2/fan\_in), maintaining activation variance through ReLU layers.  
* **PPO \+ Orthogonal: 79.2% clean** vs Kaiming 75.0% (−4.2 pp) and Xavier 73.2% (−6.0 pp). Orthogonal initialisation preserves gradient norms, which is especially beneficial for the policy gradient updates PPO uses.

These results confirm that the default choices embedded in the libraries reflect genuine algorithmic considerations, not arbitrary convention.

### 2.2.5 Policy Analysis

To connect the learned policies back to the clinical context, we examined three diagnostics.

**Action distribution.** Both agents concentrate their choices on a subset of the 25 available dose combinations. DQN's distribution is more peaked (higher exploitation), consistent with its greedy Q-value policy. PPO's distribution is more spread (higher entropy), reflecting its explicit entropy bonus. Clinically, neither agent administers extreme interventions uniformly: the most common actions correspond to moderate vasopressor and IV fluid doses, which is consistent with standard ICU guidelines.

**Policy entropy (PPO).** Per-state entropy is highest early in episodes, when patient state is uncertain, and decreases as the episode progresses and the agent becomes more confident in the appropriate treatment. This mirrors clinical practice: early ICU decision-making involves more uncertainty than later-stage management of a stable or deteriorating patient.

**Critic value trajectories (PPO).** We collected critic value estimates V(s) along surviving and dying patient trajectories. Surviving trajectories show consistently higher estimated values from mid-episode onwards, suggesting the critic correctly identifies early indicators of patient stability. Dying trajectories (outside the 1% forced-death events) show a gradual decline in critic values, indicating the agent recognises deterioration before the terminal step.

### 2.2.6 Comparative Summary

| Metric | DQN | PPO |
| ----- | ----- | ----- |
| Best clinical survival (eval) | 75.0% | 72.0% |
| Clean-env survival | ≈78% | ≈79% |
| Multi-seed std (clinical) | ±2.9 pp | ±1.8 pp |
| Training stability | Low (oscillates) | High (monotonic rise) |
| Best checkpoint step | 20,000 / 100,000 | 70,000 / 100,000 |
| Best weight init | Kaiming (ReLU default) | Orthogonal (PPO default) |

DQN reaches a stronger peak earlier but is less stable — checkpoint selection is critical. PPO is more consistent but converges more slowly. On the clean environment (without wrappers), performance is nearly identical (78% vs 79%), suggesting both algorithms have learned approximately the same underlying policy. The differences seen in the clinical environment are largely attributable to how each algorithm responds to observation noise and missing features introduced by the wrappers.

## 3.3 Extension

If selected

4. # Conclusion {#conclusion}

(Summary of findings, limitations and potential improvements.)

(EXTRA):  
3\. Creative extension • Design and implement an extension that meaningfully builds on  or goes beyond what was done in Config A and Config B. This could relate to the reward function, the learning algorithm, the environment settings, **interpretability**, or any other aspect of the problem you find interesting. • Justify  your  choice  in  the  report.  Explain  what  motivated  it,  what  you expected  to  find,  what  you  actually  found,  and  what  it  adds  to  your understanding of the problem.  
Suggestions:

## **Explainable AI (XAI) & Clinical Interpretability**

In medicine, a "black box" algorithm will not be trusted by doctors. This extension focuses on understanding why your agent chooses specific doses of vasopressors or IV fluids based on the patient's physiological state.

Example 1: Action-Value Heatmaps (Config A) Map out the learned Q-table by clustering the 716 states into clinical categories (e.g., "High Heart Rate \+ Low Blood Pressure"). Visualize which actions (fluid/vasopressor combinations) the agent prefers for each cluster to see if it matches standard medical guidelines.

Example 2: SHAP / LIME Feature Importance (Config B) Apply SHAP (SHapley Additive exPlanations) to your Deep RL neural network. This will allow you to plot which of the 47 physiological variables (like lactate or SOFA score) are driving the agent's decisions to increase or decrease medication in real-time.

**Advanced Reward Function Engineering**

The default reward is sparse and clinical-outcome heavy: $+1$ for survival, $0$ for death, while aiming to minimize unnecessary treatment. You can design a more sophisticated reward function to reshape how the agent learns.  

* Example 1: **Penalty for "Volatile" Clinical Changes** Add a step-wise penalty for sudden, drastic changes in medication (e.g., jumping from zero vasopressors to maximum dose in a single step), as wild swings in blood pressure are dangerous for real patients.    
* Example 2: **Intermediate Physiological Reward Shaping** Instead of waiting until the end of the episode to give a reward, introduce minor intermediate rewards or penalties based on whether the patient's continuous state variables (like blood pressure or heart rate) are moving closer to or further away from safe, normal clinical ranges.    
* Example 3: **Multi-Objective Reinforcement Learning (MORL)**Explicitly separate the reward into two conflicting objectives: $R\_{survival}$ and $R\_{toxicity}$ (cost of high drug dosage). Implement a Pareto-front analysis to show how the agent's behavior changes when you prioritize patient survival vs. minimizing drug side-effects.

## **Off-Policy Policy Evaluation (OPE)**

Since your environment is built from the real MIMIC-III clinical database, you can evaluate how your trained RL agent compares to the actual human doctors who treated those patients, without deploying the model.

* Example 1: **Per-State Clinician vs. Agent Discrepancy Analysis**  
  Identify states where your trained agent's chosen action completely disagrees with the historical clinician's action. Plot these states to analyze whether the agent is uncovering "hidden" optimal treatments or making mistakes.  
* Example 2: **Historical Trajectory Backtesting** Take historical patient data from the MIMIC-III database where patients unfortunately died under human care. Feed those exact patient states into your trained agent step-by-step and document whether your agent would have recommended a different treatment path.

## **Safe Exploration & Constraint-Based RL**

In healthcare, random exploration (like $\\epsilon$-greedy) can "kill" a simulated patient. This extension focuses on introducing safety boundaries to the agent's learning process.

* Example 1: **"Do No Harm" Action Masking**  
  Create a hard-coded clinical safety filter layer over your RL agent. If a patient's blood pressure is already dangerously high, mask out (forbid) the actions that allow the agent to administer more vasopressors, forcing it to choose from safe actions only.  
* Example 2: **Clinician-Guided Exploration**  
  Instead of starting with completely random actions, initialize your agent's policy using Behavioral Cloning (Imitation Learning) on a basic rule-based clinician policy. Let the RL agent explore and optimize only within a safe deviation window from human clinician behavior.  
* Example 3: **Reward-Constrained Optimization** Implement a simple constrained RL framework where the agent is strictly penalized if the simulated patient's SOFA score exceeds a critical threshold for more than $X$ consecutive steps, training the agent to be risk-averse.

5. # References {#references}

TEXTTTTTTTTTTTTTT

# 

6. # Annex {#annex}

TEXTTTTTTTTTTTTTT