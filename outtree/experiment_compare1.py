import random
from generator import generate_random_out_tree
from generator import print_tree_hierarchical
from gittins import compute_exact_gittins_indices

def gittins_policy(frontier, attempt_state):
    """Optimal: Selects the node with the highest exact Gittins index."""
    return max(frontier, key=lambda n: n.gittins[attempt_state[n.node_id]])

def myopic_policy(frontier, attempt_state):
    """Greedy: Maximises immediate success probability per unit of time."""
    return max(frontier, key=lambda n: n.p[attempt_state[n.node_id]] / n.l)

def random_policy(frontier, attempt_state):
    """Baseline: Selects a node uniformly at random."""
    return random.choice(frontier)

def dfs_policy(frontier, attempt_state):
    """
    Depth-First Search: Always picks the most recently discovered node.
    Because we append new children to the end of the frontier list, 
    selecting frontier[-1] naturally implements a strict DFS stack.
    """
    return frontier[-1]


def simulate_attack(root, policy_func, verbose=True):
    """
    Simulates the attack using a specific policy function.
    Returns: (is_successful: bool, time_spent: float)
    """
    # Initialize the frontier and tracking states
    frontier = [root]
    attempt_state = {root.node_id: 0}
    total_time = 0.0
    step = 1
    
    if verbose:
        print("\n==================================================")
        print(f" STARTING SIMULATION [{policy_func.__name__}]")
        print("==================================================")
    
    while frontier:
        # 1. Target Selection via Policy
        target = policy_func(frontier, attempt_state)
        k = attempt_state[target.node_id]
        
        if verbose:
            print(f"Step {step} | Attacking {target.node_id} (Attempt {k+1}/{target.q}) | Cost: {target.l:.2f}, Prob: {target.p[k]:.3f}")
        
        # 2. Advance time and roll outcome
        total_time += target.l
        roll = random.random()
        
        if roll < target.p[k]:
            # --- SUCCESS ---
            if verbose: print("  -> SUCCESS!")
            if target.is_leaf:
                return True, total_time
            
            # Remove compromised node, expand children to frontier
            frontier.remove(target)
            for child in target.children:
                frontier.append(child)
                attempt_state[child.node_id] = 0 
                
        else:
            # --- FAILURE ---
            if verbose: print("  -> FAILURE.")
            attempt_state[target.node_id] += 1
            
            # Check for permanent lockout
            if attempt_state[target.node_id] == target.q:
                if verbose: print(f"  -> LOCKOUT: {target.node_id} permanently sealed.")
                frontier.remove(target)
                
        step += 1
        
    # Attack failed (all paths locked out)
    if verbose: print("  -> DEFENDER WINS. All paths locked out.")
    return False, total_time

def evaluate_policies(max_depth=4, branching_factor=2, iterations=1000):
    print(f"\nGenerating Random Out-Tree (Depth={max_depth}, Branching={branching_factor})...")
    network_root = generate_random_out_tree(max_depth, branching_factor)
    
    print("Computing Exact Gittins Indices for the network...")
    compute_exact_gittins_indices(network_root, lam=0.1)
    
    policies = [
        ("Gittins (Optimal)", gittins_policy),
        ("Myopic (Greedy)", myopic_policy),
        ("Depth-First (DFS)", dfs_policy),
        ("Random Walk", random_policy)
    ]
    
    print(f"\nRunning {iterations} Monte Carlo simulations per policy...")
    print(f"{'Policy':<20} | {'Win Rate':<10} | {'Avg Time (When Successful)':<25}")
    print("-" * 60)
    
    results = {}
    
    for name, policy_func in policies:
        success_count = 0
        total_success_time = 0.0
        
        for _ in range(iterations):
            # Run silently for bulk evaluation
            is_success, time_spent = simulate_attack(network_root, policy_func, verbose=False)
            
            if is_success:
                success_count += 1
                total_success_time += time_spent
                
        win_rate = (success_count / iterations) * 100
        avg_time = (total_success_time / success_count) if success_count > 0 else float('inf')
        
        results[name] = {"Win Rate": win_rate, "Avg Time": avg_time}
        
        print(f"{name:<20} | {win_rate:>8.1f}% | {avg_time:>20.2f} units")
        
    return results

# ==========================================
# Run the Evaluation
# ==========================================
if __name__ == "__main__":
    # You can still print/visualize the tree if you want:
    # network_root = generate_random_out_tree(max_depth=3, branching_factor=2)
    # compute_exact_gittins_indices(network_root, lam=0.1)
    # visualize_tree(network_root)
    
    # Run the Monte Carlo empirical comparison
    evaluate_policies(max_depth=10, branching_factor=7, iterations=90000)