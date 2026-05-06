# ExpGA-duopoly

An investigation into the evolutionary stability of pricing heuristics in duopoly markets under varying Signal-to-Noise Ratio (SNR) regimes. This repository contains the implementation of the **Exponentially Weighted Genetic Algorithm (ExpGA)**, a population-based heuristic designed to study algorithmic convergence in non-stationary environments.

## Getting Started

### Prerequisites
*   Python 3.8+
*   Dependencies listed in `requirements.txt` (numpy, pandas, matplotlib, seaborn)

### Installation
1. Clone the repository:
   ```bash
   git clone [https://github.com/your-username/ExpGA-duopoly.git](https://github.com/your-username/ExpGA-duopoly.git)
   cd ExpGA-duopoly
    ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Running the Simulation
The main script executes 50 independent trials for each noise regime ($\mu \in \{0.05, 0.25, 0.5\}$). Each trial runs for 100,000 periods.

    ```bash
    python ExpGA_duo.py
    ```

The simulation takes ~25 minutes to run.

Upon completion, the script generates:

`expga_results.csv`: Raw data containing average price, correlation, and regret for every run.
`price_convergence_violin.png`: A violin plot visualizing the price distribution across SNR levels.

## Methodology

The ExpGA agent utilizes several key mechanisms to navigate the duopoly market:
- Temporal Memory: Fitness is updated via an Exponentially Weighted Moving Average (EWMA) with $\gamma = 0.1$.

- Selection: Fitness-proportional (Roulette Wheel) selection, mirroring stochastic choice rules.

- Crossover: Convex recombination ($P_c = 0.8$) with an argmin mapping back to the discrete price grid $\mathcal{P} = \{0.1, \dots, 1.0\}$.

- Mutation: A stochastic disruptor ($m = 0.05$) to prevent premature convergence.

## Authors
Prashanth Krishnan, Alexandyr Card, and Thinh Tu

Developed for Multi-Agent Systems and Machine Learning, Spring 2026