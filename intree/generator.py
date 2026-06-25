import numpy as np
import random

class InTreeNode:
    """
    Represents a single control node (v) in a Layered Defences In-Tree.
    """
    def __init__(self, node_id, time_cost, lockout_limit, initial_prob, decay_rate, is_target=False):
        self.node_id = node_id
        self.time_cost = time_cost          # l_v
        self.lockout_limit = lockout_limit  # q_v
        self.is_target = is_target          # True if this is the final target phi
        
        # Pre-compute decaying success probabilities for each attempt k
        # Ensures p_v(k) strictly drops as k increases
        self.success_probs = [initial_prob * (decay_rate ** k) for k in range(lockout_limit)]
        
        self.successor = None               # p(v): The unique next node closer to the target
        self.incoming_branches = []         # Sibling paths merging into this node (OR-logic)

    def p_v(self, k):
        """
        Returns the probability of success at attempt k.
        Returns 0.0 if the node has reached its lockout limit (k >= q_v).
        """
        if k < self.lockout_limit:
            return self.success_probs[k]
        return 0.0

    def set_successor(self, target_node):
        """
        Establishes the structural geometry. Sets this node's unique 
        successor and registers this node as an incoming branch for the target.
        """
        if self.is_target:
            raise ValueError("The final target cannot have a successor.")
        
        self.successor = target_node
        target_node.incoming_branches.append(self)
        
    def __repr__(self):
        status = "TARGET" if self.is_target else f"-> {self.successor.node_id}"
        return f"Node({self.node_id} | {status} | q={self.lockout_limit})"
    
def generate_random_in_tree(num_nodes):
    """
    Generates a random Layered Defences In-Tree.
    Node 0 is always the target (phi).
    """
    if num_nodes < 1:
        raise ValueError("Tree must have at least 1 node (the target).")

    nodes = {}
    
    # Initialize the Target Node (phi)
    # Time cost and lockout limit for the target can be arbitrary as success ends the game
    nodes[0] = InTreeNode(
        node_id=0, 
        time_cost=1.0, 
        lockout_limit=1, 
        initial_prob=1.0, 
        decay_rate=1.0, 
        is_target=True
    )
    
    # Generate the perimeter and internal controls
    for i in range(1, num_nodes):
        # Randomize control parameters
        l_v = round(random.uniform(0.5, 5.0), 2)
        q_v = random.randint(2, 5)
        init_p = round(random.uniform(0.3, 0.9), 2)
        decay = round(random.uniform(0.7, 0.95), 2)
        
        new_node = InTreeNode(i, l_v, q_v, init_p, decay)
        
        # Enforce in-tree geometry: randomly pick one existing node to be the successor.
        # Because we only pick from already generated nodes (0 to i-1), 
        # it naturally builds inward towards Node 0 without cycles.
        successor_id = random.randint(0, i - 1)
        new_node.set_successor(nodes[successor_id])
        
        nodes[i] = new_node
        
    return nodes

# --- Example Initialization ---
if __name__ == "__main__":
    # Generate an in-tree with 1 target and 9 defensive controls
    in_tree_network = generate_random_in_tree(10)
    
    print("Network Topology:")
    for node_id, node in in_tree_network.items():
        print(node)