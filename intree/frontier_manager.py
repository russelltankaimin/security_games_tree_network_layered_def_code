import random
import numpy as np
from index_calc import compute_alpha


def get_path_to_root(start_node):
    """Helper function to find all nodes on the path from a node to the target."""
    path = set()
    curr = start_node
    while curr is not None:
        path.add(curr.node_id)
        curr = curr.successor
    return path

def simulate_single_game(network, discount_rate=0.05):
    """
    Simulates a single attack run using the local target index policy.
    Returns (success_boolean, total_time_spent, discounted_reward)
    """
    # Initialize the frontier with all 'leaf' nodes (nodes with no incoming branches)
    # The frontier maps the Node object to its current attempt count 'k'
    frontier = {}
    for node_id, node in network.items():
        if not node.incoming_branches and not node.is_target:
            frontier[node] = 0
            
    total_time = 0.0
    
    while frontier:
        # 1. Evaluate the active frontier using the Oracle
        best_node = None
        max_alpha = -1.0
        
        for node, k in frontier.items():
            # compute_alpha is the Oracle function we defined in the previous step
            alpha_val = compute_alpha(node, k, discount_rate)
            if alpha_val > max_alpha:
                max_alpha = alpha_val
                best_node = node
                
        # If the highest alpha is 0 (or all nodes are locked out/unviable), the attacker quits
        if max_alpha <= 0.0 or best_node is None:
            break
            
        # 2. Execute the attack on the chosen node
        current_k = frontier[best_node]
        p_success = best_node.p_v(current_k)
        
        # Advance time
        total_time += best_node.time_cost
        
        # Roll the dice for stochastic success
        if random.random() < p_success:
            # --- SUCCESS STATE ---
            if best_node.successor.is_target:
                # Target reached! Game over.
                discounted_reward = np.exp(-discount_rate * total_time)
                return True, total_time, discounted_reward
            else:
                next_node = best_node.successor
                
                # --- OR-LOGIC PRUNING ---
                # We reached next_node. Any node currently in the frontier that exists 
                # solely to unlock next_node (or paths through it) is now a sunk cost.
                nodes_to_remove = []
                for f_node in frontier.keys():
                    # If next_node is on f_node's path to the root, f_node is obsolete
                    if next_node.node_id in get_path_to_root(f_node):
                        nodes_to_remove.append(f_node)
                        
                for obs_node in nodes_to_remove:
                    del frontier[obs_node]
                
                # Add the newly unlocked node to the frontier at fresh state k=0
                frontier[next_node] = 0
                
        else:
            # --- FAILURE STATE ---
            # Increment the attempt count
            frontier[best_node] += 1
            
            # If the node hits its lockout limit, remove it from the active frontier
            if frontier[best_node] >= best_node.lockout_limit:
                del frontier[best_node]
                
    # If the loop exits without returning True, the attacker failed or gave up
    return False, total_time, 0.0