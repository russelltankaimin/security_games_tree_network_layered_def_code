import numpy as np

def evaluate_path_surplus(start_node, start_k, gamma, discount_rate):
    """
    Computes B_gamma(v, k) by working backward from the target down to the start_node.
    """
    # 1. Extract the unique linear path from start_node to the target
    path = []
    curr = start_node
    while not curr.is_target:
        path.append(curr)
        curr = curr.successor
        
    # 2. Base condition at the target
    # If p(v) = phi, B_gamma(T) = 1 - gamma
    b_plus = 1.0 - gamma 
    
    # 3. Backward induction from the node closest to the target, down to start_node
    for i in range(len(path) - 1, -1, -1):
        v = path[i]
        beta_v = np.exp(-discount_rate * v.time_cost)
        
        # Array to hold B_gamma(v, j) for this specific node across all its attempt states
        b_current_array = [0.0] * v.lockout_limit
        
        # B_gamma(D) = 0 (Lockout boundary condition)
        b_minus = 0.0 
        
        # Iterate backwards through attempt states: from q_v-1 down to 0
        for j in range(v.lockout_limit - 1, -1, -1):
            p_v_j = v.p_v(j)
            
            # Expected surplus of future state
            expected_future = p_v_j * b_plus + (1 - p_v_j) * b_minus
            
            # The surplus recursion formula
            b_val = -gamma * (1 - beta_v) + beta_v * expected_future
            
            # Max operator ensures we drop to 0 if it's not worth attacking
            b_val = max(0.0, b_val)
            b_current_array[j] = b_val
            
            # This state's value becomes the b_minus for the preceding attempt (j-1)
            b_minus = b_val
            
        # The value of arriving at this node fresh (k=0) becomes the b_plus 
        # for the NEXT node physically further down the chain.
        b_plus = b_current_array[0]
        
        # If we have fully evaluated down to our target start_node, return the specific state value
        if i == 0: 
            return b_current_array[start_k]
            
    return 0.0

def compute_alpha(node, k, discount_rate=0.05, tol=1e-5):
    """
    Finds the local target index alpha(v, k) using Bisection Search.
    alpha(v,k) is the supremum gamma in [0,1] such that B_gamma(v, k) > 0.
    """
    # If the node is already locked out, the index is 0
    if k >= node.lockout_limit:
        return 0.0
        
    low = 0.0
    high = 1.0
    alpha = 0.0
    
    # Bisection search to find the tipping point
    while (high - low) > tol:
        mid = (low + high) / 2.0
        surplus = evaluate_path_surplus(node, k, mid, discount_rate)
        
        if surplus > 0:
            # mid is still a viable outside option, true alpha is higher
            alpha = mid
            low = mid
        else:
            # mid is too high, the attacker wouldn't attack. True alpha is lower
            high = mid
            
    return alpha