"""
This experiment:
1. Generates a random tree (in the future it will read off a file)
2. Compute the Gittins Index
"""
from utils.piecewise_linear_function import PiecewiseLinearFunction
from utils.linear_function import LinearFunction
from generator import print_tree_hierarchical

import numpy as np

def compute_exact_gittins_indices(root, lam: float, max_m: float = 1.0):
    """
    Recursively computes the exact Gittins index for each node in the out-tree
    using the provided PiecewiseLinearFunction algebra.
    
    root: The root Node of the out-tree.
    lam: The continuous-time discount rate (lambda).
    max_m: The maximum possible reward/retirement value (typically 1.0).
    """
    
    # The base retirement function f(m) = m
    m_func = PiecewiseLinearFunction([max_m], [LinearFunction(1.0, 0.0)])
    
    def evaluate_node(node):
        # 1. Post-order traversal: compute for all children subtrees first
        for child in node.children:
            evaluate_node(child)
            
        discount = np.exp(-lam * node.l)
        
        node.phi = {}
        node.gittins = {}
        
        # Base case: Lockout state q
        # At lockout, success probability is 0, so phi(m) = m
        node.phi[node.q] = m_func
        node.gittins[node.q] = 0.0
        
        # 2. Evaluate the set value function of the child branches: Phi_Ch(m)
        if not node.is_leaf and len(node.children) > 0:
            # Get the derivative of phi_0 for the first child
            prod_deriv = node.children[0].phi[0].derivative()
            
            # Multiply with derivatives of all parallel children
            for i in range(1, len(node.children)):
                # Uses your overloaded __mul__ for piecewise constant functions
                prod_deriv = prod_deriv * node.children[i].phi[0].derivative()
            
            # Integrate: F(m) = int_0^m prod_deriv(x) dx
            F_m = prod_deriv.integrate_piecewise_constant()
            
            # F_max = int_0^max_m prod_deriv(x) dx
            F_max = prod_deriv.integrate_zero_to_x(max_m)
            
            # Phi_Ch(m) = 1.0 - int_m^max_m prod_deriv(x) dx 
            # Which algebraically is: 1.0 - (F_max - F(m)) = F(m) + (1.0 - F_max)
            Phi_Ch = F_m.add_const(1.0 - F_max)
        else:
            # For a leaf target, success yields a terminal reward of 1.0
            Phi_Ch = PiecewiseLinearFunction([max_m], [LinearFunction(0.0, 1.0)])
            
        # 3. Backward induction for attempt states k = q-1 down to 0
        for k in range(node.q - 1, -1, -1):
            
            # Expected value = discount * [p * Phi_Ch + (1 - p) * phi_{k+1}]
            expected_success = Phi_Ch.mult_by_const(node.p[k])
            expected_failure = node.phi[k+1].mult_by_const(1.0 - node.p[k])
            
            # Uses your overloaded __add__ 
            expected_val = (expected_success + expected_failure).mult_by_const(discount)
            
            # phi_k(m) = max(m, expected_val(m))
            phi_k = expected_val.max_with_linear()
            
            # CRITICAL: Clean the function to merge collinear segments. 
            # Without this, the changepoint array grows exponentially!
            phi_k.clean() 
            
            node.phi[k] = phi_k
            
            # The Gittins index is the minimum m where phi(m) >= m
            node.gittins[k] = phi_k.compute_fixed_point()

    # Initiate the recursive evaluation from the root
    evaluate_node(root)
    print("Exact Gittins indices successfully computed using symbolic piecewise functions.")

