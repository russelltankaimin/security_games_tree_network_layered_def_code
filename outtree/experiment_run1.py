from generator import print_tree_hierarchical
from generator import generate_random_out_tree
from gittins import compute_exact_gittins_indices

root = generate_random_out_tree(max_depth=5, branching_factor=3)
print_tree_hierarchical(root)
compute_exact_gittins_indices(root=root, lam=0.8)
print_tree_hierarchical(root)