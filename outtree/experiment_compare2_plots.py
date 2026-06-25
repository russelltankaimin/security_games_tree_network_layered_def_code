import numpy as np
import random
import os
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from generator import generate_random_out_tree
from generator import print_tree_hierarchical
from utils.print_tree_topology import save_tree_topology_figure
from gittins import compute_exact_gittins_indices
from experiment_compare1 import simulate_attack, random_policy, myopic_policy, gittins_policy, dfs_policy


# (Assume the classes Node, PiecewiseLinearFunction, LinearFunction 
# and the functions generate_random_out_tree, compute_exact_gittins_indices, 
# and the 4 policy functions are already defined above this in your script).

# ==========================================
# 1. Data Collection Functions
# ==========================================
def collect_macro_data(root, iterations=500):
    """Runs Monte Carlo simulations and tracks cumulative moving averages."""
    policies = [
        ("Gittins", gittins_policy, 'blue'),
        ("Myopic", myopic_policy, 'orange'),
        ("DFS", dfs_policy, 'green'),
        ("Random", random_policy, 'red')
    ]
    
    macro_data = {name: {'win_rates': [], 'avg_times': [], 'all_times': [], 'color': color} 
                  for name, _, color in policies}
    
    for name, policy_func, _ in policies:
        wins = 0
        total_time = 0.0
        
        for i in range(1, iterations + 1):
            is_success, time_spent = simulate_attack(root, policy_func, verbose=False)
            if is_success:
                wins += 1
                total_time += time_spent
                macro_data[name]['all_times'].append(time_spent)
                
            macro_data[name]['win_rates'].append((wins / i) * 100)
            avg_time = (total_time / wins) if wins > 0 else np.nan
            macro_data[name]['avg_times'].append(avg_time)
            
    return macro_data

def collect_micro_trace(root, policy_func):
    """Runs a single simulation and meticulously logs step-by-step metrics."""
    frontier = [root]
    attempt_state = {root.node_id: 0}
    
    trace = {
        'steps': [],
        'targeted_index': [],
        'frontier_size': [],
        'wasted_time': []
    }
    
    step = 1
    wasted = 0.0
    
    while frontier:
        target = policy_func(frontier, attempt_state)
        k = attempt_state[target.node_id]
        current_index = target.gittins.get(k, 0.0) if hasattr(target, 'gittins') and target.gittins else 0.0
        
        trace['steps'].append(step)
        trace['targeted_index'].append(current_index)
        trace['frontier_size'].append(len(frontier))
        trace['wasted_time'].append(wasted)
        
        roll = random.random()
        if roll < target.p[k]:
            if target.is_leaf:
                break # Success
            frontier.remove(target)
            for child in target.children:
                frontier.append(child)
                attempt_state[child.node_id] = 0
        else:
            wasted += target.l
            attempt_state[target.node_id] += 1
            if attempt_state[target.node_id] == target.q:
                frontier.remove(target)
        step += 1
        
    return trace

# ==========================================
# 2. Plotting Dashboard
# ==========================================
def generate_academic_figures(macro_data, gittins_trace, dfs_trace):
    """Renders a 3x2 grid of publication-ready charts."""
    
    # Use a clean, academic style
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(3, 2, figsize=(15, 16))
    fig.suptitle("Performance Analysis of Adaptive Attack Policies in Out-Tree Networks", fontsize=18, fontweight='bold', y=0.98)
    
    # --- Plot 1: Convergence of Expected Time ---
    ax = axes[0, 0]
    for name, data in macro_data.items():
        ax.plot(data['avg_times'], label=name, color=data['color'], alpha=0.8, linewidth=2)
    ax.set_title("1. Expected Time-to-Compromise (Convergence)", fontsize=12, fontweight='bold')
    ax.set_xlabel("Monte Carlo Iterations")
    ax.set_ylabel("Cumulative Average Time")
    ax.legend()

    # --- Plot 2: Convergence of Win Rate ---
    ax = axes[0, 1]
    for name, data in macro_data.items():
        ax.plot(data['win_rates'], label=name, color=data['color'], alpha=0.8, linewidth=2)
    ax.set_title("2. Attacker Success Rate (Avoidance of Lockouts)", fontsize=12, fontweight='bold')
    ax.set_xlabel("Monte Carlo Iterations")
    ax.set_ylabel("Win Rate (%)")
    ax.set_ylim(0, 105)
    ax.legend()

    # --- Plot 3: Empirical Density of Attack Durations ---
    ax = axes[1, 0]
    for name, data in macro_data.items():
        times = data['all_times']
        if len(times) > 1:
            kde = gaussian_kde(times)
            x_range = np.linspace(min(times), max(times), 200)
            ax.fill_between(x_range, kde(x_range), alpha=0.3, label=name, color=data['color'])
            ax.plot(x_range, kde(x_range), color=data['color'], linewidth=1.5)
    ax.set_title("3. Density Distribution of Breach Times", fontsize=12, fontweight='bold')
    ax.set_xlabel("Time to Breach")
    ax.set_ylabel("Density")
    ax.legend()

    # --- Plot 4: The Abandonment Threshold (Gittins Trace) ---
    ax = axes[1, 1]
    ax.step(gittins_trace['steps'], gittins_trace['targeted_index'], where='mid', color='blue', linewidth=2, marker='o', markersize=4)
    ax.set_title("4. The Abandonment Threshold (Gittins Single Run Trace)", fontsize=12, fontweight='bold')
    ax.set_xlabel("Decision Step")
    ax.set_ylabel("Targeted Gittins Index $g(e, k)$")
    
    # --- Plot 5: Evolution of the Attack Surface ---
    ax = axes[2, 0]
    ax.plot(gittins_trace['steps'], gittins_trace['frontier_size'], label='Gittins (Breadth/Optimal)', color='blue', linewidth=2)
    ax.plot(dfs_trace['steps'], dfs_trace['frontier_size'], label='DFS (Deep Dive)', color='green', linewidth=2, linestyle='--')
    ax.set_title("5. Active Frontier Expansion", fontsize=12, fontweight='bold')
    ax.set_xlabel("Decision Step")
    ax.set_ylabel("Number of Exposed Controls")
    ax.legend()

    # --- Plot 6: Cumulative Wasted Effort ---
    ax = axes[2, 1]
    ax.fill_between(gittins_trace['steps'], gittins_trace['wasted_time'], color='blue', alpha=0.2, label='Gittins Wasted Time')
    ax.plot(gittins_trace['steps'], gittins_trace['wasted_time'], color='blue', linewidth=2)
    
    ax.fill_between(dfs_trace['steps'], dfs_trace['wasted_time'], color='green', alpha=0.2, label='DFS Wasted Time')
    ax.plot(dfs_trace['steps'], dfs_trace['wasted_time'], color='green', linewidth=2, linestyle='--')
    ax.set_title("6. Accumulation of Sunk Costs", fontsize=12, fontweight='bold')
    ax.set_xlabel("Decision Step")
    ax.set_ylabel("Cumulative Time Wasted on Failed Paths")
    ax.legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    print("Displaying academic figures...")
    plt.show()

def generate_individual_figures(macro_data, gittins_trace, dfs_trace, file_format='pdf'):
    """
    Generates and saves the 6 academic figures individually as high-res files.
    Ideal for direct \includegraphics{} imports in LaTeX.
    """
    # Use a clean, academic style
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Standardised figure size for individual publication charts
    fig_size = (8, 6)
    
    print(f"\nSaving individual figures as .{file_format} files...")

    for fig_num in range(1, 7):
        plt.figure(figsize=fig_size)
        
        # ---------------------------------------------------------
        # FIGURE 1: Convergence of Expected Time
        # ---------------------------------------------------------
        if fig_num == 1:
            for name, data in macro_data.items():
                plt.plot(data['avg_times'], label=name, color=data['color'], alpha=0.8, linewidth=2)
            plt.title("Expected Time-to-Compromise (Convergence)", fontsize=14, fontweight='bold')
            plt.xlabel("Monte Carlo Iterations", fontsize=12)
            plt.ylabel("Cumulative Average Time", fontsize=12)
            plt.legend()
            filename = f"Figure_1_Expected_Time.{file_format}"
            
        # ---------------------------------------------------------
        # FIGURE 2: Attacker Success Rate (Win Rate)
        # ---------------------------------------------------------
        elif fig_num == 2:
            for name, data in macro_data.items():
                plt.plot(data['win_rates'], label=name, color=data['color'], alpha=0.8, linewidth=2)
            plt.title("Attacker Success Rate (Avoidance of Lockouts)", fontsize=14, fontweight='bold')
            plt.xlabel("Monte Carlo Iterations", fontsize=12)
            plt.ylabel("Win Rate (%)", fontsize=12)
            plt.ylim(0, 105)
            plt.legend()
            filename = f"Figure_2_Win_Rate.{file_format}"
            
        # ---------------------------------------------------------
        # FIGURE 3: Empirical Density of Attack Durations
        # ---------------------------------------------------------
        elif fig_num == 3:
            for name, data in macro_data.items():
                times = data['all_times']
                if len(times) > 1:
                    kde = gaussian_kde(times)
                    x_range = np.linspace(min(times), max(times), 200)
                    plt.fill_between(x_range, kde(x_range), alpha=0.3, label=name, color=data['color'])
                    plt.plot(x_range, kde(x_range), color=data['color'], linewidth=1.5)
            plt.title("Density Distribution of Breach Times", fontsize=14, fontweight='bold')
            plt.xlabel("Time to Breach", fontsize=12)
            plt.ylabel("Density", fontsize=12)
            plt.legend()
            filename = f"Figure_3_Density_Distribution.{file_format}"
            
        # ---------------------------------------------------------
        # FIGURE 4: The Abandonment Threshold (Gittins Trace)
        # ---------------------------------------------------------
        elif fig_num == 4:
            plt.step(gittins_trace['steps'], gittins_trace['targeted_index'], 
                     where='mid', color='blue', linewidth=2, marker='o', markersize=5)
            plt.title("The Abandonment Threshold (Optimal Lateral Switching)", fontsize=14, fontweight='bold')
            plt.xlabel("Decision Step", fontsize=12)
            plt.ylabel("Targeted Gittins Index", fontsize=12)
            filename = f"Figure_4_Abandonment_Threshold.{file_format}"
            
        # ---------------------------------------------------------
        # FIGURE 5: Evolution of the Attack Surface
        # ---------------------------------------------------------
        elif fig_num == 5:
            plt.plot(gittins_trace['steps'], gittins_trace['frontier_size'], 
                     label='Gittins (Optimal)', color='blue', linewidth=2)
            plt.plot(dfs_trace['steps'], dfs_trace['frontier_size'], 
                     label='DFS (Deep Dive)', color='green', linewidth=2, linestyle='--')
            plt.title("Active Frontier Expansion", fontsize=14, fontweight='bold')
            plt.xlabel("Decision Step", fontsize=12)
            plt.ylabel("Number of Exposed Controls", fontsize=12)
            plt.legend()
            filename = f"Figure_5_Frontier_Expansion.{file_format}"
            
        # ---------------------------------------------------------
        # FIGURE 6: Cumulative Wasted Effort (Sunk Costs)
        # ---------------------------------------------------------
        elif fig_num == 6:
            plt.fill_between(gittins_trace['steps'], gittins_trace['wasted_time'], 
                             color='blue', alpha=0.2, label='Gittins Wasted Time')
            plt.plot(gittins_trace['steps'], gittins_trace['wasted_time'], color='blue', linewidth=2)
            
            plt.fill_between(dfs_trace['steps'], dfs_trace['wasted_time'], 
                             color='green', alpha=0.2, label='DFS Wasted Time')
            plt.plot(dfs_trace['steps'], dfs_trace['wasted_time'], color='green', linewidth=2, linestyle='--')
            
            plt.title("Accumulation of Sunk Costs", fontsize=14, fontweight='bold')
            plt.xlabel("Decision Step", fontsize=12)
            plt.ylabel("Cumulative Time Wasted on Failed Paths", fontsize=12)
            plt.legend()
            filename = f"Figure_6_Sunk_Costs.{file_format}"

        # Ensure the layout is tight so labels aren't cut off in the PDF
        plt.tight_layout()
        
        # Save the figure to the current directory
        plt.savefig(filename, format=file_format, bbox_inches='tight', dpi=300)
        print(f"  ✓ Saved: {filename}")
        
        # Critically important: Close the figure to free memory before the next loop iteration
        plt.close()

    print("All figures successfully generated and saved!")



    """
    Generates and saves the 6 academic figures individually as high-res files.
    Ideal for direct \includegraphics{} imports in LaTeX.
    """
    # Use a clean, academic style
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Standardised figure size for individual publication charts
    fig_size = (8, 6)
    
    print(f"\nSaving individual figures as .{file_format} files...")

    for fig_num in range(1, 7):
        plt.figure(figsize=fig_size)
        
        # ---------------------------------------------------------
        # FIGURE 1: Convergence of Expected Time
        # ---------------------------------------------------------
        if fig_num == 1:
            for name, data in macro_data.items():
                plt.plot(data['avg_times'], label=name, color=data['color'], alpha=0.8, linewidth=2)
            plt.title("Expected Time-to-Compromise (Convergence)", fontsize=14, fontweight='bold')
            plt.xlabel("Monte Carlo Iterations", fontsize=12)
            plt.ylabel("Cumulative Average Time", fontsize=12)
            plt.legend()
            filename = f"Figure_1_Expected_Time.{file_format}"
            
        # ---------------------------------------------------------
        # FIGURE 2: Attacker Success Rate (Win Rate)
        # ---------------------------------------------------------
        elif fig_num == 2:
            for name, data in macro_data.items():
                plt.plot(data['win_rates'], label=name, color=data['color'], alpha=0.8, linewidth=2)
            plt.title("Attacker Success Rate (Avoidance of Lockouts)", fontsize=14, fontweight='bold')
            plt.xlabel("Monte Carlo Iterations", fontsize=12)
            plt.ylabel("Win Rate (%)", fontsize=12)
            plt.ylim(0, 105)
            plt.legend()
            filename = f"Figure_2_Win_Rate.{file_format}"
            
        # ---------------------------------------------------------
        # FIGURE 3: Empirical Density of Attack Durations
        # ---------------------------------------------------------
        elif fig_num == 3:
            for name, data in macro_data.items():
                times = data['all_times']
                if len(times) > 1:
                    kde = gaussian_kde(times)
                    x_range = np.linspace(min(times), max(times), 200)
                    plt.fill_between(x_range, kde(x_range), alpha=0.3, label=name, color=data['color'])
                    plt.plot(x_range, kde(x_range), color=data['color'], linewidth=1.5)
            plt.title("Density Distribution of Breach Times", fontsize=14, fontweight='bold')
            plt.xlabel("Time to Breach", fontsize=12)
            plt.ylabel("Density", fontsize=12)
            plt.legend()
            filename = f"Figure_3_Density_Distribution.{file_format}"
            
        # ---------------------------------------------------------
        # FIGURE 4: The Abandonment Threshold (Gittins Trace)
        # ---------------------------------------------------------
        elif fig_num == 4:
            plt.step(gittins_trace['steps'], gittins_trace['targeted_index'], 
                     where='mid', color='blue', linewidth=2, marker='o', markersize=5)
            plt.title("The Abandonment Threshold (Optimal Lateral Switching)", fontsize=14, fontweight='bold')
            plt.xlabel("Decision Step", fontsize=12)
            plt.ylabel("Targeted Gittins Index", fontsize=12)
            filename = f"Figure_4_Abandonment_Threshold.{file_format}"
            
        # ---------------------------------------------------------
        # FIGURE 5: Evolution of the Attack Surface
        # ---------------------------------------------------------
        elif fig_num == 5:
            plt.plot(gittins_trace['steps'], gittins_trace['frontier_size'], 
                     label='Gittins (Optimal)', color='blue', linewidth=2)
            plt.plot(dfs_trace['steps'], dfs_trace['frontier_size'], 
                     label='DFS (Deep Dive)', color='green', linewidth=2, linestyle='--')
            plt.title("Active Frontier Expansion", fontsize=14, fontweight='bold')
            plt.xlabel("Decision Step", fontsize=12)
            plt.ylabel("Number of Exposed Controls", fontsize=12)
            plt.legend()
            filename = f"Figure_5_Frontier_Expansion.{file_format}"
            
        # ---------------------------------------------------------
        # FIGURE 6: Cumulative Wasted Effort (Sunk Costs)
        # ---------------------------------------------------------
        elif fig_num == 6:
            plt.fill_between(gittins_trace['steps'], gittins_trace['wasted_time'], 
                             color='blue', alpha=0.2, label='Gittins Wasted Time')
            plt.plot(gittins_trace['steps'], gittins_trace['wasted_time'], color='blue', linewidth=2)
            
            plt.fill_between(dfs_trace['steps'], dfs_trace['wasted_time'], 
                             color='green', alpha=0.2, label='DFS Wasted Time')
            plt.plot(dfs_trace['steps'], dfs_trace['wasted_time'], color='green', linewidth=2, linestyle='--')
            
            plt.title("Accumulation of Sunk Costs", fontsize=14, fontweight='bold')
            plt.xlabel("Decision Step", fontsize=12)
            plt.ylabel("Cumulative Time Wasted on Failed Paths", fontsize=12)
            plt.legend()
            filename = f"Figure_6_Sunk_Costs.{file_format}"

        # Ensure the layout is tight so labels aren't cut off in the PDF
        plt.tight_layout()
        
        # Save the figure to the current directory
        plt.savefig(filename, format=file_format, bbox_inches='tight', dpi=300)
        print(f"  ✓ Saved: {filename}")
        
        # Critically important: Close the figure to free memory before the next loop iteration
        plt.close()

    print("All figures successfully generated and saved!")

def generate_individual_figures_new(macro_data, gittins_trace, dfs_trace, seed, file_format='pdf'):
    """
    Generates and saves the 6 academic figures into a seed-specific directory.
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    fig_size = (8, 6)
    
    # 1. Create the seed directory
    dir_name = f"outtree/results/images/seed_{seed}"
    os.makedirs(dir_name, exist_ok=True)
    
    print(f"\nSaving individual figures to directory: ./{dir_name}/")

    for fig_num in range(1, 7):
        plt.figure(figsize=fig_size)
        suffix = ""
        
        # ---------------------------------------------------------
        # FIGURE 1: Convergence of Expected Time
        # ---------------------------------------------------------
        if fig_num == 1:
            for name, data in macro_data.items():
                plt.plot(data['avg_times'], label=name, color=data['color'], alpha=0.8, linewidth=2)
            plt.title("Expected Time-to-Compromise (Convergence)", fontsize=14, fontweight='bold')
            plt.xlabel("Monte Carlo Iterations", fontsize=12)
            plt.ylabel("Cumulative Average Time", fontsize=12)
            plt.legend()
            suffix = "Expected_Time"
            
        # ---------------------------------------------------------
        # FIGURE 2: Attacker Success Rate (Win Rate)
        # ---------------------------------------------------------
        elif fig_num == 2:
            for name, data in macro_data.items():
                plt.plot(data['win_rates'], label=name, color=data['color'], alpha=0.8, linewidth=2)
            plt.title("Attacker Success Rate (Avoidance of Lockouts)", fontsize=14, fontweight='bold')
            plt.xlabel("Monte Carlo Iterations", fontsize=12)
            plt.ylabel("Win Rate (%)", fontsize=12)
            plt.ylim(0, 105)
            plt.legend()
            suffix = "Win_Rate"
            
        # ---------------------------------------------------------
        # FIGURE 3: Empirical Density of Attack Durations
        # ---------------------------------------------------------
        elif fig_num == 3:
            for name, data in macro_data.items():
                times = data['all_times']
                if len(times) > 1:
                    kde = gaussian_kde(times)
                    x_range = np.linspace(min(times), max(times), 200)
                    plt.fill_between(x_range, kde(x_range), alpha=0.3, label=name, color=data['color'])
                    plt.plot(x_range, kde(x_range), color=data['color'], linewidth=1.5)
            plt.title("Density Distribution of Breach Times", fontsize=14, fontweight='bold')
            plt.xlabel("Time to Breach", fontsize=12)
            plt.ylabel("Density", fontsize=12)
            plt.legend()
            suffix = "Density_Distribution"
            
        # ---------------------------------------------------------
        # FIGURE 4: The Abandonment Threshold (Gittins Trace)
        # ---------------------------------------------------------
        elif fig_num == 4:
            plt.step(gittins_trace['steps'], gittins_trace['targeted_index'], 
                     where='mid', color='blue', linewidth=2, marker='o', markersize=5)
            plt.title("The Abandonment Threshold (Optimal Lateral Switching)", fontsize=14, fontweight='bold')
            plt.xlabel("Decision Step", fontsize=12)
            plt.ylabel("Targeted Gittins Index", fontsize=12)
            suffix = "Abandonment_Threshold"
            
        # ---------------------------------------------------------
        # FIGURE 5: Evolution of the Attack Surface
        # ---------------------------------------------------------
        elif fig_num == 5:
            plt.plot(gittins_trace['steps'], gittins_trace['frontier_size'], 
                     label='Gittins (Optimal)', color='blue', linewidth=2)
            plt.plot(dfs_trace['steps'], dfs_trace['frontier_size'], 
                     label='DFS (Deep Dive)', color='green', linewidth=2, linestyle='--')
            plt.title("Active Frontier Expansion", fontsize=14, fontweight='bold')
            plt.xlabel("Decision Step", fontsize=12)
            plt.ylabel("Number of Exposed Controls", fontsize=12)
            plt.legend()
            suffix = "Frontier_Expansion"
            
        # ---------------------------------------------------------
        # FIGURE 6: Cumulative Wasted Effort (Sunk Costs)
        # ---------------------------------------------------------
        elif fig_num == 6:
            plt.fill_between(gittins_trace['steps'], gittins_trace['wasted_time'], 
                             color='blue', alpha=0.2, label='Gittins Wasted Time')
            plt.plot(gittins_trace['steps'], gittins_trace['wasted_time'], color='blue', linewidth=2)
            
            plt.fill_between(dfs_trace['steps'], dfs_trace['wasted_time'], 
                             color='green', alpha=0.2, label='DFS Wasted Time')
            plt.plot(dfs_trace['steps'], dfs_trace['wasted_time'], color='green', linewidth=2, linestyle='--')
            
            plt.title("Accumulation of Sunk Costs", fontsize=14, fontweight='bold')
            plt.xlabel("Decision Step", fontsize=12)
            plt.ylabel("Cumulative Time Wasted on Failed Paths", fontsize=12)
            plt.legend()
            suffix = "Sunk_Costs"

        plt.tight_layout()
        
        # 2. Dynamically construct the file path
        filename = os.path.join(dir_name, f"Figure_{fig_num}_{seed}_{suffix}.{file_format}")
        
        # 3. Save and close
        plt.savefig(filename, format=file_format, bbox_inches='tight', dpi=300)
        print(f"  ✓ Saved: {filename}")
        plt.close()

    print("\nAll figures successfully generated and saved!")

# ==========================================
# 3. Execution Block
# ==========================================
if __name__ == "__main__":
    # 1. Generate or define the seed
    # You can change this to a hardcoded number (e.g., experiment_seed = 42) 
    # if you find a specific tree topology you really like for the paper.
    experiment_seed = random.randint(10000, 99999)
    
    # 2. Lock the random engines to ensure reproducibility
    random.seed(experiment_seed)
    np.random.seed(experiment_seed)
    
    print(f"==================================================")
    print(f" INITIALIZING EXPERIMENT WITH SEED: {experiment_seed}")
    print(f"==================================================")
    
    print("\nGenerating tree and computing exact algebra...")
    # Because the random engine is seeded, this will generate the exact 
    # same tree every time you use this specific seed.
    network_root = generate_random_out_tree(max_depth=5, branching_factor=3)
    compute_exact_gittins_indices(network_root, lam=0.1)
    
    print("Running Macro Monte Carlo Simulations (takes a few seconds)...")
    macro_data = collect_macro_data(network_root, iterations=8000)
    
    print("Running Micro Traces...")
    gittins_trace = collect_micro_trace(network_root, gittins_policy)
    dfs_trace = collect_micro_trace(network_root, dfs_policy)
    
    # 3. Generate the figures and pass the seed down
    generate_individual_figures_new(macro_data, gittins_trace, dfs_trace, seed=experiment_seed, file_format='pdf')

    # 4. Print Tree Topology Into A File
    save_tree_topology_figure(root=network_root, seed=experiment_seed)