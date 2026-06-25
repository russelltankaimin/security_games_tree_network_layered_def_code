from generator import generate_random_out_tree
from generator import print_tree_hierarchical
from gittins import compute_exact_gittins_indices
import random

def simulate_attack(root):
    """
    Simulates an adaptive attacker navigating the out-tree using the optimal 
    Gittins index policy. Logs the frontier, choices, and stochastic outcomes.
    """
    # Initialize the frontier with the root node and track attempt states
    frontier = [root]
    attempt_state = {root.node_id: 0}
    
    print("\n==================================================")
    print("      STARTING ADAPTIVE ATTACK SIMULATION         ")
    print("==================================================")
    
    step = 1
    total_time = 0.0
    
    while frontier:
        print(f"\n--- Step {step} ---")
        
        # Display the current frontier and the respective Gittins indices at their current state k
        frontier_info = ", ".join([
            f"{n.node_id} (g={n.gittins[attempt_state[n.node_id]]:.4f})" 
            for n in frontier
        ])
        print(f"Current Frontier : [{frontier_info}]")
        
        # 1. Greedy Target Selection
        # The attacker selects the node in the frontier with the absolute highest current Gittins index
        target = max(frontier, key=lambda n: n.gittins[attempt_state[n.node_id]])
        k = attempt_state[target.node_id]
        
        print(f"Action           : Attacking {target.node_id} (Attempt {k+1} of {target.q})")
        print(f"Stats            : Time Cost = {target.l:.2f}, Success Prob = {target.p[k]:.3f}")
        
        # Advance the simulation time clock
        total_time += target.l
        
        # 2. Simulate the Stochastic Outcome
        roll = random.random()
        if roll < target.p[k]:
            # --- SUCCESS ---
            print(f"Outcome          : SUCCESS! (Rolled {roll:.3f} < {target.p[k]:.3f})")
            
            if target.is_leaf:
                print("\n==================================================")
                print(f" TARGET COMPROMISED! Attacker wins.")
                print(f" Total time elapsed: {total_time:.2f} units.")
                print("==================================================")
                return True
            else:
                # Remove the compromised node from the frontier
                frontier.remove(target)
                
                # Expose its children and initialize their attempt states to 0
                unlocked_ids = []
                for child in target.children:
                    frontier.append(child)
                    attempt_state[child.node_id] = 0 
                    unlocked_ids.append(child.node_id)
                
                print(f"Update           : {target.node_id} bypassed. Unlocked children: {unlocked_ids}")
                
        else:
            # --- FAILURE ---
            print(f"Outcome          : FAILURE. (Rolled {roll:.3f} >= {target.p[k]:.3f})")
            
            # Increment the attempt state
            attempt_state[target.node_id] += 1
            new_k = attempt_state[target.node_id]
            
            # Check for permanent lockout
            if new_k == target.q:
                print(f"Update           : LOCKOUT. Maximum attempts reached. {target.node_id} is permanently sealed.")
                frontier.remove(target)
            else:
                # The node remains on the frontier, but its index will degrade for the next turn
                new_gittins = target.gittins[new_k]
                print(f"Update           : {target.node_id} state degrades to k={new_k}. New Gittins index: {new_gittins:.4f}")
                
        step += 1
        
    # If the loop exits because the frontier is empty, the defender wins
    print("\n==================================================")
    print(" ATTACK DEFEATED! All available paths have been locked out.")
    print(f" Total time wasted by attacker: {total_time:.2f} units.")
    print("==================================================")
    return False

# ==========================================
# Execution
# ==========================================

if __name__ == "__main__":
    network_root = generate_random_out_tree(max_depth=3, branching_factor=2)
    compute_exact_gittins_indices(network_root, lam=0.1)
    print_tree_hierarchical(network_root)
    
    # Run the simulation!
    simulate_attack(network_root)