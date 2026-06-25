import random

# ==========================================
# 1. Data Structure
# ==========================================
class Node:
    def __init__(self, node_id, is_leaf=False, parent=None):
        self.node_id = node_id
        self.is_leaf = is_leaf
        self.parent = parent
        self.children = []
        
        # Lockout threshold (q)
        self.q = random.randint(3, 5) 
        
        # Strictly decaying success probabilities p_e(k)
        self.p = sorted([random.uniform(0.1, 0.9) for _ in range(self.q)], reverse=True)
        
        # Duration of the attack l(e)
        self.l = random.uniform(1.0, 4.0) 
        
        # Dictionaries to store the computed functions and indices
        self.phi = {}       # Maps state k -> PiecewiseLinearFunction phi_k(m)
        self.gittins = {}   # Maps state k -> scalar Gittins index g(e, k)

# ==========================================
# 2. Tree Generation
# ==========================================
def generate_random_out_tree(max_depth, branching_factor, current_depth=0, parent_node=None, node_id_counter=None):
    """
    Recursively generates a random out-tree.
    """
    if node_id_counter is None:
        node_id_counter = [0]
        
    node_id = f"Node_{node_id_counter[0]}"
    node_id_counter[0] += 1
    
    # Base case: max depth reached, force it to be a leaf target
    if current_depth == max_depth:
        return Node(node_id, is_leaf=True, parent=parent_node)
    
    node = Node(node_id, is_leaf=False, parent=parent_node)
    
    # Randomly determine number of children for this branch
    num_children = random.randint(1, branching_factor)
    for _ in range(num_children):
        # Pass the current node as the parent for the next level
        child = generate_random_out_tree(
            max_depth, branching_factor, current_depth + 1, parent_node=node, node_id_counter=node_id_counter
        )
        node.children.append(child)
        
    return node

# ==========================================
# 3. Hierarchical Printing
# ==========================================
def print_tree_hierarchical(node, depth=0):
    """
    Prints the tree structure and node information using indentation
    to represent the hierarchy (Depth-First Search).
    """
    # Create an indentation string based on the current depth
    indent = "    " * depth
    
    # Extract Parent ID (Out-trees have at most 1 parent)
    parent_id = node.parent.node_id if node.parent else "None (Root)"
    
    # Extract Children IDs
    children_ids = [child.node_id for child in node.children] if node.children else ["None (Leaf)"]
    
    # Format the probabilities for easy reading
    probs_str = ", ".join([f"k={k}: {p:.2f}" for k, p in enumerate(node.p)])
    
    # Print the block of information for this node
    print(f"{indent}■ {node.node_id}")
    print(f"{indent}  ├─ Parent:   {parent_id}")
    print(f"{indent}  ├─ Children: {', '.join(children_ids)}")
    print(f"{indent}  ├─ Cost (l): {node.l:.2f}")
    print(f"{indent}  ├─ Lockout:  q = {node.q}")
    
    # Conditionally print the Gittins indices if they exist
    if node.gittins:
        print(f"{indent}  ├─ Probs:    [{probs_str}]")
        
        # Sort the dictionary items so states k=0, 1, 2... are printed in order
        gittins_str = ", ".join([f"k={k}: {g:.4f}" for k, g in sorted(node.gittins.items())])
        print(f"{indent}  └─ Gittins:  [{gittins_str}]")
    else:
        # If gittins is empty, cap off the tree structure at the probabilities
        print(f"{indent}  └─ Probs:    [{probs_str}]")

    print("") # Empty line for spacing
    
    # Recursively print all children, increasing the depth (and indentation) by 1
    for child in node.children:
        print_tree_hierarchical(child, depth + 1)

# # ==========================================
# # 4. Main Execution
# # ==========================================
# if __name__ == "__main__":
#     print("Generating random out-tree topology...\n")
#     # Generate a tree with a maximum depth of 2 and up to 2 branches per node
#     # (Kept small so the console output is easy to read)
#     network_root = generate_random_out_tree(max_depth=2, branching_factor=2)
    
#     print("--- Hierarchical Tree Information ---")
#     # Print the tree starting from the root at depth 0
#     print_tree_hierarchical(network_root)