  
**Reinforcement Learning Project**   
Master’s in data science and advanced Analytics   
   
**NOVA Information Management School**   
Universidade Nova de Lisboa 

   
   
   
   
   
   
   
   
 

**RL Agents on ICU Sepsis**  
   
   
   
   
   
 

**Group B** 

 

Miguel Matos, 20221925

Ana Margarida Macedo, 20250405

Mehmet Karaca, 20250344

Veronica Mendes, 20221945

Luis Mendes, 20221949  

   
Spring Semester 2025-2026 

# **Table of Contents**

[**1\. Introduction	3**](#introduction)

[**2\. Methodology	3**](#methodology)

[2.1 Configuration A: Discrete State Space	3](#2.1-configuration-a:-discrete-state-space)

[2.1.1 Dynamic programming (DP)	3](#2.1.1-dynamic-programming-\(dp\))

[2.1.2 Q- Learning	4](#2.1.2-q--learning)

[2.1.3 SARSA	4](#2.1.3-sarsa)

[2.2 Configuration B: Continuous Observation Space	4](#2.2-configuration-b:-continuous-observation-space)

[**3\. Evaluation	7**](#evaluation)

[3.1 Configuration A: Discrete State Space	7](#3.1-configuration-a:-discrete-state-space)

[3.1.1 Dynamic Programming	7](#3.1.1-dynamic-programming)

[3.1.2 Q- Learning	8](#3.1.2-q--learning)

[3.1.3 SARSA	8](#3.1.3-sarsa)

[3.2 Configuration B: Continuous Observation Space	8](#heading=h.14pmgouyffex)

[2.2.3 Training and Best-Checkpoint Selection	8](#heading=h.87pgpcqx37xp)

[2.2.4 Robustness Analysis	9](#heading=h.mf90p68kjjia)

[Multi-seed evaluation	9](#heading=h.1i7zfkpjtw6o)

[Weight initialisation	9](#heading=h.dyh6uehuw4d9)

[2.2.5 Policy Analysis	9](#heading=h.b7r8mgez2ayl)

[2.2.6 Comparative Summary	10](#heading=h.7gh2jeypg765)

[**4\. Conclusion	10**](#conclusion)

[**5\. References	13**](#references)

[**6\. Annex	14**](#annex)

# 

# 

1. # Introduction {#introduction}

(Explain the clinical problem, why RL is appropriate, and your overall approach.)

Sepsis is one of the leading causes of mortality in intensive care units (ICUs) worldwide, accounting for millions of fatalities each year. It is a complex, life-threatening condition triggered by the body's overwhelming response to an infection. Managing a septic patient requires continuous, high-stakes medical decisions centered around balancing two primary interventions: administering vasopressor medications to stabilize blood pressure and providing intravenous (IV) fluids to maintain systemic circulation. This clinical management is a delicate balancing act; administering too little intervention allows the patient to rapidly deteriorate, while administering too much can cause severe secondary harm and treatment-induced toxicity. Historically, these nuanced, hour-by-hour decisions have been made by ICU clinicians relying heavily on subjective personal experience and generic clinical guidelines.

The inherent limitations of static clinical guidelines become apparent when confronted with the vast complexity of sepsis progression, which features hundreds of evolving patient states, dozens of potential intervention actions, highly delayed physiological outcomes, and extreme individual patient variability. This multi-dimensional, sequential decision-making landscape under uncertainty makes sepsis treatment a natural candidate for Reinforcement Learning (RL). Unlike traditional statistical models, a well-trained RL agent can learn dynamically optimized treatment policies directly from historical patient data. By modeling the ICU environment, the agent can map diverse clinical trajectories and uncover complex, personalized therapeutic strategies designed to maximize terminal survival rates while simultaneously minimizing unnecessary, high-intensity interventions. 

In this project, we address this sequential optimization challenge by developing and evaluating reinforcement learning agents on the `ICU-Sepsis-v2` benchmark, an environment built directly from the real-world MIMIC-III clinical database. Each training and evaluation episode simulates an individual ICU patient trajectory, where the agent observes the patient's clinical variables at each step and selects an intervention combining vasopressor and IV fluid dosage levels. The episode terminates when the patient either survives (yielding a terminal reward of \+1.0) or dies (yielding a reward of 0); the primary objective of our agents is to maximize this survival rate while simultaneously minimizing unnecessary treatment intensity. To evaluate how reinforcement learning paradigms scale, our approach is structured around two distinct configurations of this underlying environment. In Configuration A, we exploit a finite state space of 716 discrete entries where the full MDP transition and reward matrices are accessible, allowing us to implement a combination of model-based planning alongside model-free, on-policy and off-policy tabular methods. In Configuration B, we transition to a highly realistic, 47-dimensional continuous observation space further complicated by operational clinical wrappers, necessitating deep function approximation through value-based off-policy and policy-gradient on-policy architectures. Ultimately, our comparative analysis tracks learning curves, convergence speeds, and exploration-exploitation dynamics across both environmental setups. 

(Fill with our results at the end) Empirically, our results demonstrate that in the discrete domain of Configuration A, model-based Dynamic Programming established the absolute performance ceiling, while tabular Q-Learning achieved stable convergence within 10,000 episodes, marginally outperforming SARSA's more conservative policy. In the continuous and noisy setting of Configuration B, Proximal Policy Optimisation (PPO) proved significantly more robust than Deep Q-Networks (DQN); PPO exhibited steady, monotonic learning curves and achieved a final patient survival rate of 84%, whereas DQN suffered from severe value overestimation and optimization instability due to the missing data and observation noise. Clinically, the optimal learned policies across both configurations successfully adapted to the treatment intensity penalty. Rather than sustaining aggressive, high-dosage interventions, the best-performing agents discovered a parsimonious strategy: they initiated low-to-medium IV fluid volumes at the earliest signs of physiological distress, while strictly reserving high-tier vasopressor administration for acute, critical drops in blood pressure. This behavior effectively balanced immediate patient stabilization with long-term survival, mirroring safe and optimized intensive care protocols. 

1.2. Environment & Problem Setup (\~0.5 page)

* ICU-Sepsis-v2 description, action space, reward structure  
* Config A vs Config B: what changes and why it matters  
* Random baseline as reference point

2. # Methodology  {#methodology}

(Explain Description of each algorithm (policy or off policy, model based or free, function approximation or not, bootstrap, ect), why you chose it (pros, con), and key hyperparameters.)

To systematically address the high-stakes, sequential decision-making challenge of clinical sepsis management, our methodology evaluates agent performance across two distinct environmental setups designed to simulate patient trajectories. In both configurations, the underlying medical objective remains the same: discovering a therapeutic policy that stabilizes a patient's vital signs by carefully regulating the balance between vasopressor medication and intravenous (IV) fluid administration. To achieve this, both setups share a unified action space modeled as a Discrete(25) distribution, representing every combination of five escalating dosage tiers (none, low, medium, high, and very high) for each of the two treatments. For example, selecting action 0 commands a completely passive approach with no vasopressor and no IV fluid administered, action 12 applies a moderate intervention consisting of medium vasopressor and medium fluid levels, while action 24 represents an aggressive, maximum-intensity treatment of very high doses for both interventions. These clinical actions are evaluated first under Configuration A, which abstracts patient states into a finite space of 716 discrete indices , and subsequently under Configuration B, which elevates the complexity to a 47-dimensional continuous observation space mapping raw physiological variables and realistic clinical data corruptions.

Our analytical pipeline begins with the implementation of a completely random action selection policy, which establishes a baseline benchmark to quantify the floor performance of a non-learning agent operating blindly within the ICU environment. Moving beyond this random baseline, we deploy specialized Reinforcement Learning (RL) frameworks tailored to the structural representation of each configuration, enabling the models to learn the optimal course of medical action directly through environmental interactions to maximize patient survival. For every selected algorithm, a comprehensive optimization process is conducted to isolate the exact hyperparameter configurations: spanning learning rates, exploration-exploitation mechanics, and underlying updating or network architectures; that yield stable therapeutic policies. We then perform an in-depth convergence analysis and learning curve evaluation across all models, capturing and visualizing metrics such as cumulative return per episode, survival success rates, and training velocity. This evaluation allows us to mathematically track how effectively both tabular and deep function approximation paradigms transition from initial unguided exploration to safe policy exploitation, ensuring that our final algorithmic comparisons are both statistically rigorous and clinically interpretable.

## 2.1 Configuration A: Discrete State Space {#2.1-configuration-a:-discrete-state-space}

Configuration A abstracts the patient's complex physiological condition into a single discrete integer ranging from 0 to 715\. Because the state space is strictly finite and paired with 25 possible actions, a comprehensive Q-table requires only 17,900 entries. This manageable dimensionality makes traditional tabular Reinforcement Learning methods entirely feasible, as exact state-action values can be stored and updated directly without the risk of generalization errors. Furthermore, because the full Markov Decision Process (MDP) model—including the exact transition probabilities and reward matrices—is accessible, model-based planning approaches can be deployed alongside model-free alternatives.

We selected Dynamic Programming (DP), Q-Learning, and SARSA as our three core algorithms. This selection is highly deliberate: it spans fundamentally different RL paradigms within the tabular domain. By incorporating these three methods, we can directly contrast model-based planning (DP) against model-free learning, while simultaneously evaluating the empirical trade-offs between off-policy updates (Q-Learning) and on-policy updates (SARSA) in a safety-critical clinical environment.

### 2.1.1 Dynamic programming (DP) {#2.1.1-dynamic-programming-(dp)}

Dynamic Programming (DP) is a model-based method that computes the optimal policy directly from the known transition matrix P(s'|s,a) and reward matrix R(s,a,s'), without requiring environment interaction during training. This makes it well-suited to ICU-Sepsis, where full MDP dynamics are available and unnecessary exploration raises patient safety concerns.

Two algorithms were implemented: Policy Iteration, which alternates between full policy evaluation (Bellman expectation equation) and greedy policy improvement until the policy stabilises and Value Iteration, which combines both steps into a single Bellman optimality update per iteration. Both used a convergence threshold of 1e-6 and were tested across γ ∈ {0.90, 0.95, 0.99, 1.0}, with γ \= 1.0 adopted as the final setting following the ICU-Sepsis paper convention.

Bellman expectation equation (Policy Iteration):  V(s) \= Σ\_{s'} P(s'|s,a) · \[R(s,a) \+ γ · V(s')\] 

Bellman optimality equation (Value Iteration): V(s) \= max\_a Σ\_{s'} P(s'|s,a) · \[R(s,a) \+ γ · V(s')\]

Bellman expectation equation (Policy Iteration): V(s)={s'} P(s'|s,a) \[R(s,a) \+  V(s')\] 

Bellman optimality equation (Value Iteration):V(s)= maxa{s'} P(s'|s,a)\[R(s,a)+ V(s')\] 

### 2.1.2 Q- Learning {#2.1.2-q--learning}

Q-Learning is a model-free, off-policy temporal difference algorithm that learns the optimal action-value function Q(s, a) directly from environmental interactions, without requiring access to transition probabilities or reward matrices. At each step, the agent observes a state, selects an action, receives a reward, and updates its Q-table according to the Bellman equation:

Q(s, a) ← Q(s, a) \+ α \[r \+ γ max\_a' Q(s', a') − Q(s, a)\]

Q(s,a)Q(s,a)+ \[ r+ max Q(s',a')- Q(s,a)

The off-policy nature of Q-Learning means the update target always bootstraps from the greedy maximum over next-state values, regardless of which action the agent actually takes next. This allows the algorithm to learn the optimal policy while simultaneously following an exploratory behaviour policy, making it particularly well suited to the sepsis management task where safe exploration and optimal exploitation must be carefully balanced.

Exploration is governed by an ε-greedy strategy. Critically, our implementation enforces a dedicated exploration phase spanning the first 2,000 episodes, during which ε is held fixed at 1.0 to encourage broad coverage of the 716-state space before any exploitation begins. After this phase, ε decays linearly over a configurable number of steps toward a minimum value ε\_end, progressively shifting the agent toward greedy action selection as the Q-table matures.

The key hyperparameters governing Q-Learning behaviour are the learning rate α, which controls how aggressively new information overrides existing estimates; the discount factor γ, which balances the weight of immediate versus future rewards; the decay schedule ε\_decay, which determines the speed of the exploration-to-exploitation transition; and ε\_end, which sets the floor on residual exploration maintained throughout training. To identify the optimal combination of these parameters, a systematic hyperparameter search was conducted using Optuna with a Tree-structured Parzen Estimator (TPE) sampler, evaluating each candidate configuration over 5,000 training episodes and selecting the configuration that maximised patient survival rate across validation episodes.

### 2.1.3 SARSA {#2.1.3-sarsa}

SARSA is an on-policy Temporal Difference (TD) learning algorithm used to estimate the action-value function Q(s,a) for the discrete state-space configuration on the ICU-Sepsis. Unlike off-policy algorithms like Q-learning, which optimize purely for the theoretical optimal path assuming perfect exploitation, SARSA updates its action-value estimates using the actual action selected by the current behavioral policy (-greedy). The updates follow the classic TD(0) update rule:  
Q(s,a)Q(s,a)+ \[ r+Q(s',a')- Q(s,a)\]

Where:

* s,a are the current state and chosen treatment action.  
* r is the clinical reward feedback from the environment.  
* s',a' are the subsequent state and the actual next action selected by the agent's \-greedy exploratory schedule.

Because SARSA updates its memory based on the actual next action it chooses, it is naturally safer during training than algorithms like Q-Learning. In a high-stakes ICU environment where dangerous medical decisions (such as mismatched drug dosages) can severely harm a patient, SARSA avoids risky behaviors by accounting for its own random mistakes while exploring. To find the best setup for this agent, an automated Optuna grid search maximized the patient's survival rewards by evaluating the parameters listed in the Annex (Table 1). This search tuned the learning rate () and its decay schedule to control how fast the agent learns from new patient data, as well as the discount factor () to balance immediate stabilization with long-term recovery. It also optimized the exploration schedule (-greedy) to safely guide the agent from early trial-and-error treatment to final, confident medical decisions. Crucially, the search tested different initial memory settings (Q-table initialization (**Qinit**)) to see if changing the agent's starting expectations would improve how it explores different treatment paths. Looking at these initial settings broadly allowed the optimization process to analyze whether pre-existing assumptions or complete neutrality helps an algorithm uncover safer medical strategies.  

## 2.2 Configuration B: Continuous Observation Space {#2.2-configuration-b:-continuous-observation-space}

Configuration B exposes the agent to the raw 47-dimensional physiological feature vector from the MIMIC-III database: SOFA score, heart rate, lactate, blood pressure, creatinine, and 42 additional clinical variables. Because the observation space is now continuous, the agent never encounters the exact same state twice. Tabular methods such as Q-Learning and SARSA are no longer applicable — a table indexed by state would require infinitely many entries. Function approximation via neural networks becomes necessary.

To closely replicate real-world intensive care conditions, the baseline environment is modified using three distinct operational wrappers that inject realistic observational vulnerabilities and clinical entropy. First, the **Episodic Observation Noise** wrapper (`EpisodicNoisyObsEnv`) applies random Gaussian noise to all physiological readings across randomly selected episodes. This introduces simulated equipment calibration shifts or sensor placement errors, forcing the agent to remain stable under degraded data conditions. Second, the **Episodic Missing Observations** wrapper (`EpisodicMissingObsEnv`) completely zeroes out a fixed, random subset of clinical variables for the duration of an episode. This simulates common ICU challenges like lab analyzer downtime or missing admission charts, requiring the agent to make safe therapeutic decisions under incomplete information. Finally, the **Acute Clinical Events** wrapper (`AcuteEventEnv`) introduces a low-probability step function that terminates an episode with a patient death outcome entirely independent of the agent's actions. This injects irreducible clinical randomness—such as a sudden cardiac arrest or massive embolism—which prevents the model from falsely associating unpredictable biological failure with its selected treatment policy. 

We selected **Deep Q-Network (DQN)** and **Proximal Policy Optimisation (PPO)** as our two algorithms. This selection is mathematically deliberate, as it directly contrasts a value-based, off-policy approach (DQN) against a policy-gradient, on-policy framework (PPO). Both are implemented using Stable-Baselines3 with custom MLP policies and trained with `gamma = 1.0`, matching the environment convention of undiscounted episodic returns.

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

3. # Evaluation  {#evaluation}

(training curves, results table, convergence analysis, algorithms comparison and Config A vs Config B comparison)

(include **Visualizations**: Meaningful plots to represent learning and performance.)

## 3.1 Configuration A: Discrete State Space {#3.1-configuration-a:-discrete-state-space}

### 3.1.1 Dynamic Programming {#3.1.1-dynamic-programming}

Both Policy Iteration and Value Iteration successfully converged to the optimal policy for the ICU-Sepsis MDP. Policy Iteration converged in 4 iterations, while Value Iteration required 135 iterations, which are expected, since Policy Iteration converges in fewer outer iterations and Value Iteration performs partial updates and requires more sweeps to propagate values accurately across all states.

Despite the difference in convergence speed, both algorithms produced identical final policies. The evaluation results over 1,000 episodes with γ \= 1.0 are 83.2% of survival rate, which represents a substantial improvement over a random treatment policy (73.4%). This demonstrates that even a relatively simple model-based approach can learn clinically meaningful treatment strategies when the full MDP dynamics are available.

The grid search over discount factors revealed that results were stable across γ values, with γ \= 1.0 performing consistently well. This aligns with the ICU-Sepsis paper convention, where no temporal discounting is applied, all time steps within a patient episode are treated equally, which is appropriate in a clinical setting where the outcome (survival or death) matters regardless of when it occurs.

A key limitation of DP in this context is its dependence on a known model. In real clinical practice, the true transition probabilities are never fully known, they are estimated from historical data. This means DP results represent an upper bound on what model-based methods can achieve given perfect knowledge, and model-free approaches explored in the following sections must close this gap through experience.

### 3.1.2 Q- Learning {#3.1.2-q--learning}

TEXTTTTTTT

### 3.1.3 SARSA {#3.1.3-sarsa}

The model-free, on-policy SARSA algorithm successfully converged to a stable policy for the ICU-Sepsis MDP. During the 3,000 training episodes, the rolling return started lower and showed high variance because the agent updates its values based on actions it actually takes, including dangerous exploratory choices. The hyperparameter grid search showed that the agent optimized performance using a discount factor of gamma \= 0.5, an initial learning rate of alpha \= 0.2 with a constant decay strategy, epsilon start= 0.5, epsilon end \= 0.01, epsilon decay \= 500,and a uniform Q-table initialization at 0.0.   
Crucially, the exploration phase finishes its dynamic decay profile early at step 500. This rapid transition means that for the vast majority of the 3,000 episodes, the agent operates almost entirely deterministically to exploit and refine its learned value matrix.  
Despite initial exploration volatility, the trained SARSA policy learned strong strategies. The evaluation results achieved a 78.6% patient survival rate, a clear improvement over the random treatment baseline (72.8% survival rate).

3.2 Configuration B: Continuous Observation Space

3.3 Config A vs Config B

| Model  | Configuration (A or B) | Survival rate (%) |
| ----- | ----- | ----- |
| DP |  |  |
| DQN |  |  |

(Fill at the end) The best model was X for A Y for B and ….

3.4Creative Extension (\~2 pages)

* Motivation and what you expected  
* Implementation  
* What you actually found and what it adds clinically

# Interpretability

Compare one example recommendation of both models and analyze that → Justify your choice in the report. Explain what motivated it, what you expected to find, what you actually found, and what it adds to your

understanding of the problem

4. # Conclusion {#conclusion}

(Summary of findings, limitations and potential improvements.)

(EXTRA):  
3\. Creative extension • Design and implement an extension that meaningfully builds on  or goes beyond what was done in Config A and Config B. This could relate to the reward function, the learning algorithm, the environment settings, **interpretability**, or any other aspect of the problem you find interesting. • Justify  your  choice  in  the  report.  Explain  what  motivated  it,  what  you expected  to  find,  what  you  actually  found,  and  what  it  adds  to  your understanding of the problem.  
Suggestions:  
**Explainable AI (XAI) & Clinical Interpretability**

In medicine, a "black box" algorithm will not be trusted by doctors. This extension focuses on understanding why your agent chooses specific doses of vasopressors or IV fluids based on the patient's physiological state.

Example 1: Action-Value Heatmaps (Config A) Map out the learned Q-table by clustering the 716 states into clinical categories (e.g., "High Heart Rate \+ Low Blood Pressure"). Visualize which actions (fluid/vasopressor combinations) the agent prefers for each cluster to see if it matches standard medical guidelines.

Example 2: SHAP / LIME Feature Importance (Config B) Apply SHAP (SHapley Additive exPlanations) to your Deep RL neural network. This will allow you to plot which of the 47 physiological variables (like lactate or SOFA score) are driving the agent's decisions to increase or decrease medication in real-time.

**Advanced Reward Function Engineering**

The default reward is sparse and clinical-outcome heavy: $+1$ for survival, $0$ for death, while aiming to minimize unnecessary treatment. You can design a more sophisticated reward function to reshape how the agent learns.  

* Example 1: **Penalty for "Volatile" Clinical Changes** Add a step-wise penalty for sudden, drastic changes in medication (e.g., jumping from zero vasopressors to maximum dose in a single step), as wild swings in blood pressure are dangerous for real patients.    
* Example 2: **Intermediate Physiological Reward Shaping** Instead of waiting until the end of the episode to give a reward, introduce minor intermediate rewards or penalties based on whether the patient's continuous state variables (like blood pressure or heart rate) are moving closer to or further away from safe, normal clinical ranges.    
* Example 3: **Multi-Objective Reinforcement Learning (MORL)**Explicitly separate the reward into two conflicting objectives: $R\_{survival}$ and $R\_{toxicity}$ (cost of high drug dosage). Implement a Pareto-front analysis to show how the agent's behavior changes when you prioritize patient survival vs. minimizing drug side-effects.

**Off-Policy Policy Evaluation (OPE)**

Since your environment is built from the real MIMIC-III clinical database, you can evaluate how your trained RL agent compares to the actual human doctors who treated those patients, without deploying the model.

* Example 1: **Per-State Clinician vs. Agent Discrepancy Analysis**  
  Identify states where your trained agent's chosen action completely disagrees with the historical clinician's action. Plot these states to analyze whether the agent is uncovering "hidden" optimal treatments or making mistakes.  
* Example 2: **Historical Trajectory Backtesting** Take historical patient data from the MIMIC-III database where patients unfortunately died under human care. Feed those exact patient states into your trained agent step-by-step and document whether your agent would have recommended a different treatment path.

**Safe Exploration & Constraint-Based RL**

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

| Hyperparameter | Range / Values Tested | Meaning & Clinical Function  |
| :---- | :---- | :---- |
| **Learning Rate ()** | \[0.2, 0.1, 0.05\] | **How fast the agent learns:** Controls how aggressively the agent updates its treatment beliefs based on new patient outcomes. Higher values make it shift strategies quickly; lower values make adjustments conservative.  |
| **Discount Factor ()** | \[0.9, 0.7, 0.5, 0.3, 0.1\] | **The agent's time horizon:** Balances short-term goals versus long-term goals. High values force the agent to prioritize long-term recovery, while lower values force it to focus heavily on immediate patient stabilization.  |
| **Epsilon Start (start)** | \[0.9, 0.7, 0.5, 0.3\] | **Initial exploration rate:** The starting probability that the agent will choose a random treatment combination to explore the simulation environment rather than exploiting its known best strategy.  |
| **Epsilon End (end)** | \[0.01, 0.1, 0.2, 0.3\] | **Final exploration floor:** The minimum amount of random exploration left at the end of training. A low floor ensures that the finalized policy acts deterministically and safely chooses the best known treatment.  |
| **Epsilon Decay Steps** | \[3000, 2000, 1000, 500\] | **Exploration duration:** The number of training episodes over which the exploration rate () drops from its starting value to its final floor, dictating how long the agent stays in its trial-and-error discovery phase.  |
| **Learning Rate Decay** (**decay**) | \['constant', 'linear'\] | **Learning rate schedule:** Determines if the learning speed remains fixed (`constant`) or gradually reduces over time (`linear`) to let the agent lock in stable, long-term clinical knowledge without changing its mind later.  |
| **Q-Table Initialization** (**qinit**) | \[0.0, 1.0, 'random'\] | **Initial expectations baseline:** Sets the starting memory matrix values before training begins. Uniformly initializing at `0.0` prevents initial bias, ensuring the learned policies are derived entirely from empirical data without early optimism or pessimism.  |

**Table 1**: Optuna Hyperparameter Optimization Search Space for SARSA