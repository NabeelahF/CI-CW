# Multi-Objective Battery Energy Storage Optimisation under Solar Uncertainty

This project implements a hybrid **Machine Learning + Evolutionary Optimisation framework** to optimise **Battery Energy Storage System (BESS)** scheduling under uncertain solar generation and Time-of-Use (ToU) electricity pricing.

The system combines **solar power forecasting**, **Genetic Algorithms (GA)**, **NSGA-II**, and an **adaptive Reinforcement Learning (Q-Learning) controller** to balance financial profit against battery degradation.

---

## Project Overview

With the increasing adoption of residential solar energy, efficient energy arbitrage has become crucial. Homeowners must decide **when to charge, discharge, or idle batteries** while accounting for:
- Uncertain solar generation
- Dynamic electricity tariffs
- Physical battery constraints
- Long-term battery degradation

This project formulates the problem as a **24-hour Battery Scheduling Problem (BSP)** and solves it using **evolutionary computation**, enhanced with **adaptive mutation control**.

---

## Key Features

-  **Battery Scheduling Optimisation** (24-hour horizon)
-  **Solar Power Forecasting** using an MLP Regressor
-  **Genetic Algorithm (Single-Objective)** for profit maximisation
-  **NSGA-II (Multi-Objective)** for profit vs battery wear trade-offs
-  **Reinforcement Learning (Q-Learning)** for adaptive mutation control
-  **Physics-based simulation** ensuring feasible battery behaviour
-  **Pareto front analysis** for decision-making insights

---

## System Architecture

The system consists of two main pipelines:

1. **ML Forecasting Pipeline**
   - Predicts hourly solar generation using weather features
2. **Optimisation Engine**
   - GA / NSGA-II for scheduling
   - Optional RL controller dynamically adjusts mutation rates

Two modes are supported:
- **Baseline Mode**: Static evolutionary parameters
- **Adaptive Mode**: RL-tuned mutation rate

---

## Technologies & Methods

- **Python**
- **Scikit-learn** (MLP Regressor)
- **Genetic Algorithms**
- **NSGA-II**
- **Reinforcement Learning (Q-Learning)**
- **Physics-based constraint handling**
