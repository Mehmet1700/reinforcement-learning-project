# Reinforcement Learning Project

## Overview

Welcome to your Reinforcement Learning (RL) course project! You will work in groups of between 4 to 5 students. The project deadline has been set to the 5th of June. Following the submission, each group will present and defend their project. The format and structure of the presentation are at the discretion of each group, you are free to organise it however best communicates your work and findings.

## Introduction

Sepsis is one of the leading causes of death in intensive care units worldwide, accounting for millions of fatalities each year. It is a life-threatening condition triggered by the body's response to infection, and its treatment requires continuous, high-stakes decisions: how much vasopressor medication to administer to stabilise blood pressure, and how much intravenous fluid to give to maintain circulation. These two interventions must be carefully balanced, too little and the patient deteriorate; too much and the treatment itself causes harm. Traditionally, these decisions are made by ICU clinicians based on experience and clinical guidelines. However, the complexity of the problem, hundreds of patient states, dozens of possible actions, delayed outcomes, and high individual variability, makes it a natural candidate for Reinforcement Learning. A well-trained RL agent could learn treatment policies directly from patient data, potentially uncovering strategies that improve survival rates while reducing unnecessary intervention.

In this project you will develop and evaluate RL agents on the **ICU-Sepsis-v2** benchmark, a well-established environment built from the real MIMIC-III clinical database. Each episode simulates one ICU patient trajectory. At every step the agent observes the patient's state and selects a combination of vasopressor and IV fluid dose. The episode ends when the patient either survives (reward +1) or dies (reward 0). The agent's objective is to maximise the survival rate while minimising unnecessary treatment intensity.

The project is structured around two configurations of the same underlying environment, designed to explore the transition from classical tabular RL to modern deep RL as the observation space grows in complexity.

## Configurations

### 1. Configuration A: Discrete State Space

The agent observes a discrete integer representing the patient's clinical state (0–715). With a finite state space of 716 states and 25 possible actions, a Q-table has only 17,900 entries, making tabular RL methods entirely feasible. The full MDP model (transition probabilities and reward matrix) is also accessible, enabling model-based approaches such as Policy Iteration.

### 2. Configuration B: Continuous Observation Space

The agent now receives a 47-dimensional continuous feature vector containing the actual normalised physiological measurements used in the original Komorowski et al. (2018) AI Clinician paper: SOFA score, heart rate, lactate, blood pressure, creatinine, and 42 other clinical variables. Because the observation space is continuous, the agent never sees the exact same state twice, tabular methods are no longer applicable and function approximation via neural networks becomes necessary.

## Project Requirements

### 1. Algorithms

- Implement two different RL algorithms (e.g., DP, Monte Carlo, Q-Learning, SARSA, DDPG, PPO, etc.) in each configuration.
- You may choose algorithms seen in class or explore new ones.

### 2. Evaluation and Analysis

- Train your agents and record performance metrics, e.g.:
  - Total reward per episode/Return.
  - Convergence speed.
  - Exploration vs. exploitation balance.
- Provide visualizations of learning progress (e.g., reward curves, success rates).
- Perform a comparative analysis of both algorithms and environments.
- Justify all the decisions you did in your report.

### 3. Creative Extension

- Design and implement an extension that meaningfully builds on or goes beyond what was done in Config A and Config B. This could relate to the reward function, the learning algorithm, the environment settings, interpretability, or any other aspect of the problem you find interesting.
- Justify your choice in the report. Explain what motivated it, what you expected to find, what you actually found, and what it adds to your understanding of the problem.

## Project Objectives

- Choose appropriate RL algorithms for different observation and action spaces and argue their relevance.
- Understand and explain the trade-off between exploration and exploitation in a real-world setting.
- Analyse learning progress through meaningful visualisations and performance metrics.
- Interpret agent behaviour and connect it back to the clinical context.
- Understand the limitations of tabular methods and when deep RL becomes necessary.

## Deliverables

### 1. Code Implementation

A Jupyter notebook (or one per configuration) capable of replicating your reported results. The notebook must run end-to-end without errors, use the fixed evaluation functions for all results, and include clear setup instructions.

### 2. Project Report (Max 15 pages, excluding references & annexes)

- **Introduction**: The clinical problem, why RL is appropriate, and your overall approach.
- **Methodology**: Description of each algorithm, why you chose it, and key hyperparameters.
- **Evaluation**: training curves, results table, convergence analysis, algorithms comparison and Config A vs Config B comparison.
- **Visualizations**: Meaningful plots to represent learning and performance.
- **Conclusion**: Summary of findings, limitations and potential improvements.

## Project Evaluation Criteria

The evaluation focuses on the quality of your analysis and argumentation rather than raw performance numbers. The grading is distributed as follows:

| Criterion | Weight |
|---|---|
| Algorithm choice and justification | 25% |
| Evaluation and comparative analysis | 30% |
| Visualisations and interpretability | 15% |
| Creative extension | 10% |
| Code quality and reproducibility | 10% |
| Quality of presentation and report | 10% |

- **Algorithm choice and justification (25%)** — Are the chosen algorithms appropriate for each configuration? Is the reasoning behind each choice clearly argued in the report?
- **Evaluation and comparative analysis (30%)** — Quality of training curves, convergence analysis, final results table, and the comparison between algorithms and between Config A and Config B.
- **Visualisations and interpretability (15%)** — Clarity and relevance of plots, ability to connect the agent's behaviour back to the clinical context.
- **Creative extension (10%)** — Originality of the proposed extension, quality of the implementation, and depth of the analysis and justification in the report.
- **Code quality and reproducibility (10%)** — Notebook runs end-to-end without errors, seeds are fixed, results are reproducible, and setup instructions are clear.
- **Quality of presentation and report (10%)** — Clarity of writing, logical structure, appropriate use of figures and tables, and overall readability of the report. Results and conclusions are communicated in a precise and concise manner.

## Submission Guidelines

Submit a compressed folder in Moodle containing:

- Group members in a clear manner.
- All code files and dependencies.
- A well-structured project report in PDF format.
- Any additional notes or documentation.

---

*NOVA Information Management School*  
*Instituto Superior de Estatística e Gestão de Informação*  
*Universidade NOVA de Lisboa*