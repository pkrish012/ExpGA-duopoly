import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import argparse, os

class ExpGA_Agent:
    def __init__(self, price_grid, n_pop=20, gamma=0.1, mutation_rate=0.05, crossover_rate=0.8):
        self.price_grid = price_grid
        self.n_pop = n_pop
        self.gamma = gamma
        self.m_rate = mutation_rate
        self.c_rate = crossover_rate
        
        # Initialization: Uniform random population
        self.population = np.random.choice(self.price_grid, size=self.n_pop)
        
        # Fitness Initialization: Uniformly 1
        self.fitness = {p: 1.0 for p in self.price_grid}

    def get_price(self):
        # Randomly select a price from current population to play this period
        return np.random.choice(self.population)

    def update_fitness(self, price, profit):
        # EWMA Fitness Update
        self.fitness[price] = (1 - self.gamma) * self.fitness[price] + self.gamma * profit

    def evolve(self):
        new_pop = []
        
        # 1. Selection (Roulette Wheel) - ONLY FROM CURRENT POPULATION
        # Extract fitness for the prices currently in the population
        fits = np.array([self.fitness[p] for p in self.population])
        fits = np.maximum(fits, 1e-6)
        probs = fits / fits.sum()
        
        # 2. Generate New Population
        for _ in range(self.n_pop // 2):
            # Selection happens from self.population indices
            idx_a = np.random.choice(len(self.population), p=probs)
            idx_b = np.random.choice(len(self.population), p=probs)
            parent_a = self.population[idx_a]
            parent_b = self.population[idx_b]
            
            # Crossover
            if np.random.rand() < self.c_rate:
                lam = np.random.rand()
                # Convex crossover: child = lam*a + (1-lam)*b
                child_1 = lam * parent_a + (1 - lam) * parent_b
                child_2 = (1 - lam) * parent_a + lam * parent_b
                
                # Map back to nearest discrete grid point
                child_1 = self.price_grid[np.abs(self.price_grid - child_1).argmin()]
                child_2 = self.price_grid[np.abs(self.price_grid - child_2).argmin()]
            else:
                child_1, child_2 = parent_a, parent_b # Cloning
            
            # 3. Mutation
            for c in (child_1, child_2):
                if np.random.rand() < self.m_rate:
                    c = np.random.choice(self.price_grid)
                new_pop.append(c)
        
        self.population = np.array(new_pop)

def deterministic_profit(p_i, p_j, alpha=4.1, beta=-4.74):
    """Noiseless logit profit used for BR and regret accounting."""
    exp_i = np.exp(alpha + beta * p_i)
    exp_j = np.exp(alpha + beta * p_j)
    denom = 1 + exp_i + exp_j
    return (p_i * exp_i) / denom


def best_response_profit(opponent_price, price_grid, alpha=4.1, beta=-4.74):
    profits = []
    for p in price_grid:
        profits.append(deterministic_profit(p, opponent_price, alpha=alpha, beta=beta))

    return max(profits)

def expected_profit(p_i, p_j, alpha=4.1, beta=-4.74):
    return deterministic_profit(p_i, p_j, alpha=alpha, beta=beta)

def market_demand(p1, p2, mu, alpha=4.1, beta=-4.74):
    """
    Realized profit for learning updates.
    mu is the standard deviation of idiosyncratic profit shocks.
    """
    base_profit1 = deterministic_profit(p1, p2, alpha=alpha, beta=beta)
    base_profit2 = deterministic_profit(p2, p1, alpha=alpha, beta=beta)

    profit1 = base_profit1 + np.random.normal(0.0, mu)
    profit2 = base_profit2 + np.random.normal(0.0, mu)

    return profit1, profit2


def run_simulation(seed=11242011):
    np.random.seed(seed)
    noise_levels = [0.05, 0.25, 0.5]
    n_runs = 50
    total_periods = 100000
    epoch_len = 100
    price_grid = np.linspace(0.1, 1.0, 10)
    alpha = 4.1
    beta = -4.74
    
    all_results = []
    
    for mu in noise_levels:
        print(f"Running simulation for mu = {mu}...")
        for run in range(n_runs):
            # Initialize two independent agents
            agent1 = ExpGA_Agent(price_grid)
            agent2 = ExpGA_Agent(price_grid)
            
            prices1, prices2 = [], []
            regrets1, regrets2 = [], []
            for t in range(total_periods):
                p1, p2 = agent1.get_price(), agent2.get_price()
                prof1, prof2 = market_demand(p1, p2, mu, alpha=alpha, beta=beta)

                br1 = best_response_profit(p2, price_grid, alpha=alpha, beta=beta)
                br2 = best_response_profit(p1, price_grid, alpha=alpha, beta=beta)

                regret1 = br1 - expected_profit(p1, p2, alpha=alpha, beta=beta)
                regret2 = br2 - expected_profit(p2, p1, alpha=alpha, beta=beta)

                agent1.update_fitness(p1, prof1)
                agent2.update_fitness(p2, prof2)
                
                # Evolve every 100 periods
                if t % epoch_len == 0 and t > 0:
                    agent1.evolve()
                    agent2.evolve()
                
                # Record prices for the final 10,000 periods
                if t >= 90000:
                    prices1.append(p1)
                    prices2.append(p2)
                    regrets1.append(regret1)
                    regrets2.append(regret2)
            
            # Calculate metrics for this run
            avg_p = (np.median(prices1) + np.median(prices2)) / 2
            corr = np.corrcoef(prices1, prices2)[0, 1]
            if np.isnan(corr):
                corr = 0.0
            avg_regret = (np.mean(regrets1) + np.mean(regrets2)) / 2

            all_results.append({
                'mu': mu,
                'run': run,
                'avg_price': avg_p,
                'correlation': corr,
                'avg_regret': avg_regret
            })

    return pd.DataFrame(all_results)

def plot_expga_results(csv_path="expga_results.csv"):
    df = pd.read_csv(csv_path)
    
    # Set aesthetics for an academic paper
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))
    
    # Create the violin plot
    ax = sns.violinplot(
        x='mu', 
        y='avg_price', 
        data=df, 
        inner="quartile", 
        palette="muted",
        cut=0
    )
    
    # Overlay individual runs to show density and outliers
    sns.stripplot(
        x='mu', 
        y='avg_price', 
        data=df, 
        color="black", 
        alpha=0.3, 
        size=4
    )
    
    # Add reference lines for Nash Equilibrium and Monopoly Price
    plt.axhline(y=0.4, color='r', linestyle='--', label='Competitive Nash ($P^D$)')
    plt.axhline(y=0.8, color='g', linestyle='--', label='Monopoly Price ($P^M$)')
    
    # Labels and Title
    plt.title('Convergence Price Distribution across Noise Regimes (50 Runs/$\mu$)', fontsize=14)
    plt.xlabel('Noise Level ($\mu$) / Inherent Signal-to-Noise Ratio', fontsize=12)
    plt.ylabel('Average Convergence Price', fontsize=12)
    plt.legend(loc='upper left')
    
    plt.tight_layout()
    plt.savefig("price_convergence_violin.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ExpGA Duopoly Simulation and Plotting")
    parser.add_argument("--plot-only", type=str, help="Path to existing csv to plot without running simulation")
    args = parser.parse_args()

    if args.plot_only:
        if os.path.exists(args.plot_only):
            print(f"Plotting results from {args.plot_only}...")
            plot_expga_results(args.plot_only)
        else:
            print(f"Error: File {args.plot_only} not found.")
    else:
        # Default behavior: Run simulation and then plot
        print("Starting full simulation...")
        df_results = run_simulation()
        df_results.to_csv("expga_results.csv", index=False)
        print("Simulation Complete. Data saved to expga_results.csv")
        plot_expga_results("expga_results.csv")