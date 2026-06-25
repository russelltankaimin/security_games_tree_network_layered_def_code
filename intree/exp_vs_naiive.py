import math
import random

# ==========================================
# 1. THE ENVIRONMENT (DATA STRUCTURE)
# ==========================================

class InTreeNode:
    def __init__(self, node_id, time_cost, lockout_limit, initial_prob, decay_rate, is_target=False):
        self.node_id = node_id
        self.time_cost = time_cost
        self.lockout_limit = lockout_limit
        self.is_target = is_target
        
        # Pre-compute decaying probabilities: p_v(k)
        self.success_probs = [initial_prob * (decay_rate ** k) for k in range(lockout_limit)]
        
        self.successor = None
        self.incoming_branches = []

    def p_v(self, k):
        if k < self.lockout_limit:
            return self.success_probs[k]
        return 0.0

    def set_successor(self, target_node):
        if self.is_target:
            raise ValueError("Target cannot have a successor.")
        self.successor = target_node
        target_node.incoming_branches.append(self)
        
    def __repr__(self):
        status = "TARGET" if self.is_target else f"-> {self.successor.node_id}"
        return f"Node({self.node_id} | {status} | q={self.lockout_limit})"


def generate_random_in_tree(num_nodes):
    nodes = {}
    # Target Node (phi)
    nodes[0] = InTreeNode(0, 1.0, 1, 1.0, 1.0, is_target=True)
    
    # Internal & Perimeter Controls
    for i in range(1, num_nodes):
        l_v = round(random.uniform(0.5, 3.0), 2)
        q_v = random.randint(2, 5)
        init_p = round(random.uniform(0.3, 0.8), 2)
        decay = round(random.uniform(0.7, 0.95), 2)
        
        new_node = InTreeNode(i, l_v, q_v, init_p, decay)
        
        # Enforce in-tree geometry
        successor_id = random.randint(0, i - 1)
        new_node.set_successor(nodes[successor_id])
        nodes[i] = new_node
        
    return nodes


# ==========================================
# 2. THE MATHEMATICAL ORACLE
# ==========================================

def evaluate_path_surplus(start_node, start_k, gamma, discount_rate):
    """Computes B_gamma(v, k) working backward from the target."""
    path = []
    curr = start_node
    while not curr.is_target:
        path.append(curr)
        curr = curr.successor
        
    b_plus = 1.0 - gamma 
    
    for i in range(len(path) - 1, -1, -1):
        v = path[i]
        beta_v = math.exp(-discount_rate * v.time_cost)
        
        b_current_array = [0.0] * v.lockout_limit
        b_minus = 0.0 
        
        for j in range(v.lockout_limit - 1, -1, -1):
            p_v_j = v.p_v(j)
            expected_future = p_v_j * b_plus + (1 - p_v_j) * b_minus
            b_val = -gamma * (1 - beta_v) + beta_v * expected_future
            b_val = max(0.0, b_val)
            b_current_array[j] = b_val
            b_minus = b_val
            
        b_plus = b_current_array[0]
        
        if i == 0: 
            return b_current_array[start_k]
            
    return 0.0

def compute_alpha(node, k, discount_rate):
    """Finds alpha(v, k) using Bisection Search."""
    if k >= node.lockout_limit:
        return 0.0
        
    low, high, alpha = 0.0, 1.0, 0.0
    
    while (high - low) > 1e-5:
        mid = (low + high) / 2.0
        surplus = evaluate_path_surplus(node, k, mid, discount_rate)
        if surplus > 0:
            alpha = mid
            low = mid
        else:
            high = mid
            
    return alpha


# ==========================================
# 3. THE GAME LOOP & FRONTIER MANAGER
# ==========================================

def get_path_to_root(start_node):
    path = set()
    curr = start_node
    while curr is not None:
        path.add(curr.node_id)
        curr = curr.successor
    return path

def simulate_single_game(network, policy="index", discount_rate=0.05):
    """Simulates one attack using either 'index' or 'naive' policy."""
    frontier = {}
    for node_id, node in network.items():
        if not node.incoming_branches and not node.is_target:
            frontier[node] = 0
            
    total_time = 0.0
    
    while frontier:
        best_node = None
        
        if policy == "index":
            max_alpha = -1.0
            for node, k in frontier.items():
                alpha_val = compute_alpha(node, k, discount_rate)
                if alpha_val > max_alpha:
                    max_alpha = alpha_val
                    best_node = node
            if max_alpha <= 0.0 or best_node is None:
                break
                
        elif policy == "naive":
            best_node = random.choice(list(frontier.keys()))
            
        current_k = frontier[best_node]
        p_success = best_node.p_v(current_k)
        total_time += best_node.time_cost
        
        if random.random() < p_success:
            # --- SUCCESS STATE ---
            if best_node.successor.is_target:
                return True, total_time, math.exp(-discount_rate * total_time)
            else:
                next_node = best_node.successor
                # --- OR-LOGIC PRUNING ---
                nodes_to_remove = [f_node for f_node in frontier.keys() 
                                   if next_node.node_id in get_path_to_root(f_node)]
                for obs_node in nodes_to_remove:
                    del frontier[obs_node]
                frontier[next_node] = 0
        else:
            # --- FAILURE STATE ---
            frontier[best_node] += 1
            if frontier[best_node] >= best_node.lockout_limit:
                del frontier[best_node]
                
    return False, total_time, 0.0


# ==========================================
# 4. THE MONTE CARLO ENGINE
# ==========================================

def run_monte_carlo(num_nodes=15, num_sims=5000, discount_rate=0.05):
    print(f"Generating In-Tree Network with {num_nodes} nodes...")
    network = generate_random_in_tree(num_nodes)
    
    policies = ["naive", "index"]
    results = {}
    
    for policy in policies:
        print(f"\nRunning {num_sims} simulations for '{policy.upper()}' policy...")
        success_count = 0
        total_rewards = 0.0
        
        for _ in range(num_sims):
            is_success, _, reward = simulate_single_game(network, policy, discount_rate)
            if is_success:
                success_count += 1
                total_rewards += reward
                
        emp_success_rate = success_count / num_sims
        exp_reward = total_rewards / num_sims
        results[policy] = {"success_rate": emp_success_rate, "reward": exp_reward}
        
        print(f"[{policy.upper()}] Success Rate: {emp_success_rate * 100:.2f}% | Expected Reward: {exp_reward:.4f}")

    # Calculate Improvement
    if results['naive']['reward'] > 0:
        improvement = ((results['index']['reward'] - results['naive']['reward']) / results['naive']['reward']) * 100
        print(f"\n=> The Index Policy improved expected reward by {improvement:.2f}% over the Naive Baseline.")

if __name__ == "__main__":
    run_monte_carlo(num_nodes=20, num_sims=3000, discount_rate=0.05)