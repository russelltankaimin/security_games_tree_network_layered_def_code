import math
import random
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. THE ENVIRONMENT
# ==========================================
class InTreeNode:
    def __init__(self, node_id, time_cost, lockout_limit, initial_prob, decay_rate, is_target=False):
        self.node_id = node_id
        self.time_cost = time_cost
        self.lockout_limit = lockout_limit
        self.is_target = is_target
        self.success_probs = [initial_prob * (decay_rate ** k) for k in range(lockout_limit)]
        self.successor = None
        self.incoming_branches = []

    def p_v(self, k):
        if k < self.lockout_limit: return self.success_probs[k]
        return 0.0

    def set_successor(self, target_node):
        if self.is_target: raise ValueError("Target cannot have a successor.")
        self.successor = target_node
        target_node.incoming_branches.append(self)

def generate_wide_in_tree(total_nodes):
    """Generates an in-tree and measures its frontier size."""
    nodes = {}
    nodes[0] = InTreeNode(0, 1.0, 1, 1.0, 1.0, is_target=True)
    
    for i in range(1, total_nodes):
        l_v = round(random.uniform(0.5, 3.0), 2)
        q_v = random.randint(2, 5)
        init_p = round(random.uniform(0.3, 0.8), 2)
        decay = round(random.uniform(0.7, 0.95), 2)
        new_node = InTreeNode(i, l_v, q_v, init_p, decay)
        
        # Connect to a random existing node
        successor_id = random.randint(0, i - 1)
        new_node.set_successor(nodes[successor_id])
        nodes[i] = new_node
        
    # Calculate initial frontier size
    frontier_size = sum(1 for n in nodes.values() if not n.incoming_branches and not n.is_target)
    return nodes, frontier_size

# ==========================================
# 2. THE MATHEMATICAL ORACLE
# ==========================================
def evaluate_path_surplus(start_node, start_k, gamma, discount_rate):
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
            b_val = max(0.0, -gamma * (1 - beta_v) + beta_v * expected_future)
            b_current_array[j] = b_val
            b_minus = b_val
        b_plus = b_current_array[0]
        if i == 0: return b_current_array[start_k]
    return 0.0

def compute_alpha(node, k, discount_rate):
    if k >= node.lockout_limit: return 0.0
    low, high, alpha = 0.0, 1.0, 0.0
    while (high - low) > 1e-2: # Very low precision for speed on large networks
        mid = (low + high) / 2.0
        if evaluate_path_surplus(node, k, mid, discount_rate) > 0:
            alpha = mid
            low = mid
        else:
            high = mid
    return alpha

# ==========================================
# 3. GAME LOOP
# ==========================================
def get_path_to_root(start_node):
    path = set()
    curr = start_node
    while curr is not None:
        path.add(curr.node_id)
        curr = curr.successor
    return path

def simulate_single_game(network, policy="index", discount_rate=0.05):
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
            if max_alpha <= 0.0 or best_node is None: break
        elif policy == "naive":
            best_node = random.choice(list(frontier.keys()))
            
        current_k = frontier[best_node]
        p_success = best_node.p_v(current_k)
        total_time += best_node.time_cost
        
        if random.random() < p_success:
            if best_node.successor.is_target:
                return True, total_time, math.exp(-discount_rate * total_time)
            else:
                next_node = best_node.successor
                nodes_to_remove = [f_node for f_node in frontier.keys() if next_node.node_id in get_path_to_root(f_node)]
                for obs_node in nodes_to_remove: del frontier[obs_node]
                frontier[next_node] = 0
        else:
            frontier[best_node] += 1
            if frontier[best_node] >= best_node.lockout_limit:
                del frontier[best_node]
    return False, total_time, 0.0

# ==========================================
# 4. EXPERIMENT & PLOTTING
# ==========================================
node_counts = [20, 40, 60, 80, 100, 150] # Push to much larger networks
num_sims = 400
discount_rate = 0.05

actual_frontiers = []
index_success = []
index_reward = []
naive_success = []
naive_reward = []

print("Running massive frontier simulations...")

for num_nodes in node_counts:
    random.seed(100 + num_nodes)
    network, frontier_size = generate_wide_in_tree(num_nodes)
    actual_frontiers.append(frontier_size)
    print(f"Network: {num_nodes} nodes | Initial Frontier Size: {frontier_size} paths")
    
    for policy, s_list, r_list in [("index", index_success, index_reward), ("naive", naive_success, naive_reward)]:
        successes = 0
        rewards = 0.0
        random.seed(200 + num_nodes + (1 if policy=="index" else 2))
        for _ in range(num_sims):
            is_success, _, rew = simulate_single_game(network, policy, discount_rate)
            if is_success:
                successes += 1
                rewards += rew
        s_list.append(successes / num_sims)
        r_list.append(rewards / num_sims)

# PLOTTING AGAINST FRONTIER SIZE
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(actual_frontiers, index_success, marker='o', label='Local Target Index', color='#1f77b4', linewidth=2)
ax1.plot(actual_frontiers, naive_success, marker='x', label='Naive Baseline', color='#d62728', linestyle='--', linewidth=2)
ax1.set_title('Empirical Success Rate vs. Initial Frontier Size')
ax1.set_xlabel('Number of Parallel Perimeter Paths (Frontier Size)')
ax1.set_ylabel('Success Rate')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(actual_frontiers, index_reward, marker='o', label='Local Target Index', color='#1f77b4', linewidth=2)
ax2.plot(actual_frontiers, naive_reward, marker='x', label='Naive Baseline', color='#d62728', linestyle='--', linewidth=2)
ax2.set_title('Expected Discounted Reward vs. Initial Frontier Size')
ax2.set_xlabel('Number of Parallel Perimeter Paths (Frontier Size)')
ax2.set_ylabel('Expected Reward (E[e^(-ρt)])')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('massive_frontier_comparison.png', dpi=300)
print("Simulation complete. Plot saved.")