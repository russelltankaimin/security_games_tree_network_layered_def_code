import numpy as np
import copy
from typing import List
import itertools

# Hacky.

def gen_succ_probs_exp_backoff(maximum_backoff: int, base_prob: float, backoff_ratio: float):
    succ_probs = []
    for i in range(maximum_backoff):
        succ_probs.append(base_prob * (backoff_ratio ** i))
    return succ_probs

def gen_succ_probs_manual(probs: List[float]):
    return probs.copy()

def gen_succ_probs_beta(maximum_length: int, alpha_orig: float, beta_orig: float):
    succ_probs = []
    for i in range(maximum_length):
        succ_probs.append(alpha_orig / (alpha_orig + beta_orig + i))
    return succ_probs
    
def gen_succ_probs_constant(maximum_length: int, prob: float):
    return [prob] * maximum_length

def brute_force_mdp_solve(serieses: List[List[List[float]]], discount_factor: float, lengths: List[List[int]]):
    '''
    Brute force MDP solver. 
    '''
    num_arms = len(serieses)
    state_space_per_arm = []
    for i in range(num_arms):
        state_space_per_arm.append([])
        for j in range(len(serieses[i])):
            for k in range(len(serieses[i][j])):
                state_space_per_arm[i].append((j, k))
            state_space_per_arm[i].append((j, 'e'))
    
    state_space = list(itertools.product(*state_space_per_arm))
    state_space.append('END')

    state_to_id = {}
    for id, S in enumerate(state_space):
        state_to_id[S] = id

    V = [1.0] * len(state_space)
    for iter in range(1000):
        V_new = [0.0] * len(state_space)
        
        # Check if reach end
        for s in range(len(state_space)):
            S = state_space[s]
            if S == 'END':
                V_new[s] =1.0
                continue    
            
            # No control has not reached end
            V_new_pos = []
            for a in range(num_arms):
                Q = 0.0
                j,k = S[a]
                if k == 'e':
                    continue
            
                # Compute expected value
                # possible transiitons
                store = 0.0
                # 1. Fail
                prob = 1 - serieses[a][j][k]
                new_state = list(copy.deepcopy(S))
                if k == len(serieses[a][j]) - 1:
                    new_state[a] = (j,'e')
                elif k < len(serieses[a][j]) - 1:
                    new_state[a] = (j,k+1)
                else:
                    assert False
                new_state = tuple(new_state)
                new_s = state_to_id[new_state]
                Q += V[new_s] * np.exp(-discount_factor * lengths[a][j]) * prob

                # 2. Pass
                prob = serieses[a][j][k]
                new_state = list(copy.deepcopy(S))
                if j == len(serieses[a]) - 1:
                    new_state = 'END'
                elif j < len(serieses[a]) - 1:
                    new_state[a] = (j+1,0)
                    new_state = tuple(new_state)
                else:
                    assert False
                new_s = state_to_id[new_state]
                
                Q += V[new_s] * np.exp(-discount_factor * lengths[a][j]) * prob
                
                V_new_pos.append((Q, a))
            V_new_pos.sort(reverse=True)
            if len(V_new_pos) == 0:
                # Should be case where all 'e'
                V_new[s] = 1.0
            else:
                V_new[s] = V_new_pos[0][0]
        V = V_new
        print(V[0])

    print(V[0])

def evaluate_gittins_brute_force(serieses, discount_factor, lengths, gits):
    '''
    Brute force MDP solver. 
    '''
    num_arms = len(serieses)
    state_space_per_arm = []
    for i in range(num_arms):
        state_space_per_arm.append([])
        for j in range(len(serieses[i])):
            for k in range(len(serieses[i][j])):
                state_space_per_arm[i].append((j, k))
            state_space_per_arm[i].append((j, 'e'))
    
    state_space = list(itertools.product(*state_space_per_arm))
    state_space.append('END')

    state_to_id = {}
    for id, S in enumerate(state_space):
        state_to_id[S] = id

    V = [1.0] * len(state_space)
    for iter in range(1000):
        V_new = [0.0] * len(state_space)
        
        # Check if reach end
        for s in range(len(state_space)):
            S = state_space[s]
            if S == 'END':
                V_new[s] =1.0
                continue    
            
            # No control has not reached end
            # Get best action
            opts = []
            for a in range(num_arms):
                j,k = S[a]
                if k == 'e':
                    opts.append((0.0, a))
                else:
                    opts.append((gits[a][j][k], a))

            opts.sort(reverse=True)
            best_a = opts[0][1]

            for a in [best_a]:
                q = 0.0
                j,k = S[a]
                if k == 'e':
                    continue
            
                # Compute expected value
                # possible transiitons
                store = 0.0
                # 1. Fail
                prob = 1 - serieses[a][j][k]
                new_state = list(copy.deepcopy(S))
                if k == len(serieses[a][j]) - 1:
                    new_state[a] = (j,'e')
                elif k < len(serieses[a][j]) - 1:
                    new_state[a] = (j,k+1)
                else:
                    assert False
                new_state = tuple(new_state)
                new_s = state_to_id[new_state]
                q += V[new_s] * np.exp(-discount_factor * lengths[a][j]) * prob

                # 2. Pass
                prob = serieses[a][j][k]
                new_state = list(copy.deepcopy(S))
                if j == len(serieses[a]) - 1:
                    new_state = 'END'
                elif j < len(serieses[a]) - 1:
                    new_state[a] = (j+1,0)
                    new_state = tuple(new_state)
                else:
                    assert False
                new_s = state_to_id[new_state]
                
                q += V[new_s] * np.exp(-discount_factor * lengths[a][j]) * prob
                
                # V_new_pos.append((q, a))
            # V_new_pos.sort(reverse=True)
            # if len(V_new_pos) == 0:
                # Should be case where all 'e'
            #     V_new[s] = 1.0
            # else:
            V_new[s] = q

        V = V_new
        print(V[0])

    print(V[0])


def compute_special_gittens_OLD(series: List[List[float]], discount_factor: float, lengths: List[int], verbose=False):
    '''
    Must be decreasing 
    '''

    # TODO: verify if decreasing

    d = len(series)
    status = [0] * d
    disc_rew = [0.0] * d
    disc_time = [1.0] * d

    component_id_order = []
    gittens_index = [[] for _ in range(d)]
    
    while (True):
        # preprocess time
        cum_rew = [1.0]* d
        cum_time = [1.0] * d
        for i in reversed(range(d)):
            if i == d - 1:
                cum_rew[i] = disc_rew[i]
                cum_time[i] = disc_time[i]
            else:
                cum_rew[i] = cum_rew[i+1] * disc_rew[i]
                cum_time[i] = cum_time[i+1] * disc_time[i]

        # Find best to choose:
        ratios = [0.0] * d
        rs = [0.0] * d
        ts = [0.0] * d
        done = True
        for i in range(d):
            # Check if reach end of subchain
            if status[i] == len(series[i]):
                continue

            if i == d - 1:
                r = series[i][status[i]] * np.exp(-discount_factor * lengths[i]) 
                # t = (1-series[i][status[i]]) * np.exp(-discount_factor * lengths[i]) + \
                #     series[i][status[i]] * np.exp(-discount_factor * lengths[i])
                t = np.exp(-discount_factor * lengths[i])
            else:
                r = series[i][status[i]] * np.exp(-discount_factor * lengths[i]) * cum_rew[i+1]
                t = (1-series[i][status[i]]) * np.exp(-discount_factor * lengths[i]) + \
                    series[i][status[i]] * np.exp(-discount_factor * lengths[i]) * cum_time[i+1]

            # if t == 0: 
            #     print(i, cum_time[i+1])
            #     assert False
            
            rs[i] = r
            ts[i] = t

            denom = 1/discount_factor * (1 - np.exp(-discount_factor * t * lengths[i]))
            ratios[i] = r/denom

            done = False
        
        if done:
             break

        best_component_id = np.argmax(ratios)
        # assert ratios[best_component_id] > 0.0
        gittens_index[best_component_id].append(ratios[best_component_id])
        component_id_order.append(best_component_id)

        # Update times and rews
        if status[best_component_id] == 0: # This part is hacky. Consider original trick of setting time to 1.0 at first.
            residual = 1.0
            disc_time[best_component_id] = 1.0
        else:
            residual = disc_time[best_component_id] - disc_rew[best_component_id]
        disc_rew[best_component_id] += residual * \
            series[best_component_id][status[best_component_id]] * \
            np.exp(-discount_factor * lengths[best_component_id])
        disc_time[best_component_id] -= residual
        disc_time[best_component_id] += residual * np.exp(-discount_factor * lengths[best_component_id])

        # Set new status
        status[best_component_id] += 1

    return component_id_order, gittens_index

def compute_special_gittens(series: List[List[float]], discount_factor: float, lengths: List[int], verbose=False):
    '''
    Compute gittins index for a special case where each arm is a series system where controls have decreasing probabilities of success.
    '''
    # TODO: verify if decreasing

    d = len(series)
    status = [0] * d
    # disc_rew = [0.0] * d
    # disc_time = [0.0] * d

    component_id_order = []
    gittens_index = [[] for _ in range(d)]
    
    # Preprocess consecutive failure probabilities
    # F[i][j] = P(fail_{i,0} * fail_{i,1} * ... * fail_{i,j-1})
    F = []
    for i in range(d):
        F.append([1.0])
        for j in range(len(series[i])):
            F[-1].append(F[-1][-1] * (1 - series[i][j]))
            # disc_time[i] += series[i][j] * np.exp(-discount_factor * lengths[i])

    prod_rew = [0.0] * d
    rew_from_ctrl_start = [0.0] * d # This only gives the expected exp(-lambda * T) to reach the end of THIS control. 
    time_from_ctrl_start = [1.0] * d # This is exponential of 0. Will contain the time to succeed *all the way, past future components* from the start of the control
    iter = 0
    prev_added = float('inf')
    while (True):
        # preprocess time
        # cum_rew = [1.0]* d
        # cum_time = [1.0] * d
        # for i in reversed(range(d)):
        #     if i == d - 1:
        #         cum_rew[i] = disc_rew[i]
        #         # cum_time[i] = disc_time[i]
        #     else:
        #         cum_rew[i] = cum_rew[i+1] * disc_rew[i]
        #         # cum_time[i] = cum_time[i+1] * disc_time[i]

        # Find best to choose:
        for i in reversed(range(d)):
            if i == d - 1:
                prod_rew[i] = rew_from_ctrl_start[i]
            else:
                prod_rew[i] = rew_from_ctrl_start[i] * prod_rew[i+1]

        # Search for best
        done = True
        ratios = [-1.0] * d
        for i in range(d):
            # Check if reach end of subchain
            if status[i] == len(series[i]):
                continue
            
            if lengths[i] == 0:
                # If length is 0, then this is really really good.
                ratios[i] = float('inf')
                done = False
                continue

            if i == d - 1:
                r = series[i][status[i]] * np.exp(-discount_factor * lengths[i])
                t = np.exp(-discount_factor * lengths[i])
            else:
                r = series[i][status[i]] * np.exp(-discount_factor * lengths[i]) * prod_rew[i+1]
                t = np.exp(-discount_factor * lengths[i]) * (1 - series[i][status[i]] + series[i][status[i]] * time_from_ctrl_start[i+1]) 

            denom = (1.0/discount_factor) * (1 - t)
            ratios[i] = r/denom
            done = False
            assert denom >= 0.0, denom

        if done:
             break
        
        best_component_id = np.argmax(ratios)
        assert ratios[best_component_id] > 0.0, f'{best_component_id} {ratios[best_component_id]}'
        assert ratios[best_component_id] <= prev_added + 1e-5, f'{best_component_id} {ratios[best_component_id]} {prev_added}'
        gittens_index[best_component_id].append(ratios[best_component_id])
        component_id_order.append(best_component_id)
        
        # Update for time to succeed 
        k = status[best_component_id]
        rew_from_ctrl_start[best_component_id] += F[best_component_id][k] * series[best_component_id][k] * np.exp(-discount_factor * (k+1) * lengths[best_component_id])
        # Update time from ctrl start
        
        # We are using k+1 here for the total length because k has not yet been incremented. In contrast to paper
        if best_component_id < d - 1:
            time_from_ctrl_start[best_component_id] = F[best_component_id][k+1] * np.exp(-discount_factor * (k + 1) * lengths[best_component_id]) + \
                rew_from_ctrl_start[best_component_id] * time_from_ctrl_start[best_component_id + 1]
        else:
            time_from_ctrl_start[best_component_id] = F[best_component_id][k+1] * np.exp(-discount_factor * (k + 1) * lengths[best_component_id]) + \
                rew_from_ctrl_start[best_component_id]

        for component_id in reversed(range(best_component_id)):
            k = status[component_id]
            time_from_ctrl_start[component_id] = F[component_id][k] * np.exp(-discount_factor * k * lengths[component_id]) + \
                rew_from_ctrl_start[component_id] * time_from_ctrl_start[component_id + 1]

        status[best_component_id] += 1
        iter += 1
    return component_id_order, gittens_index

def compute_greedy_index(series: List[List[float]], discount_factor: float, lengths: List[int], verbose=False):
    """
    Used for abalations and baselien comparisons.
    """
    W = []
    for i in range(len(series)):
        W.append([])
        for j in range(len(series[i])):
            W[-1].append(series[i][j] * np.exp(-discount_factor * lengths[i]))
    
    return None, W
    
def compute_exact_payoffs_from_gittins_policy(
                          serieses: List[List[List[float]]], 
                          discount_factor: float, 
                          lengths: List[List[int]],
                          indices: List[List[List[int]]]
                          ):
    '''
    Compute performance of some index policy. Using dynamic programming and not sampling
    Note: gittins contains the gittins index typically
    But it can also contain any index policy --- it doesn't have to be gittins.
    '''
    
    # STEP 1: Preprocessing. Sort all indices. 
    # TODO: Favor the ones with lower arm index for tiebreaks
    # and if there is still a tie, favor the earlier control index.
    to_sort = []
    for i in range(len(indices)):
        for j in range(len(indices[i])):
            for k in range(len(indices[i][j])):
                index = indices[i][j][k]
                # index, -i, -j, -k # TODO: This is the tiebreak order
                to_sort.append((index, -k, -i, i, j, k)) # tiebreak by favoring lower number of trials
    to_sort.sort(reverse=True)
    
    # Extract the indices from gittins and vice versa
    location_to_rank = copy.deepcopy(indices)
    for index, _, _, i, j, k in to_sort:
        location_to_rank[i][j][k] = index
    rank_to_location = []
    rank_to_index = []
    for rank, (index, _, _, i, j, k) in enumerate(to_sort):
        rank_to_location.append((i, j, k))
        rank_to_index.append(index)

    # STEP 2: Preprocess product probabilities that we keep using
    # 1 ---- fail, fail, fail, ..., fail, pass [k times, *including* the final pass]
    #   ---- we want to comptute for each control the product of the sequenceo failures
    # 2 ---- fail, fail, fail, ..., fail [k times in total]

    # Compute the product of the sequence of failures for each control
    # P_{ij}[k] = fail_{i,j,0} * fail_{i,j,1} * ... * fail_{i,j,k-1}
    # P[i][j][0] = 1.0
    # P[i][j][k] = fail_{i,j,0} * ... fail{i,j,k-1}
    prod_fail = []
    
    for i in range(len(serieses)):
        prod_fail.append([])
        for j in range(len(serieses[i])):
            prod_fail[-1].append([1.0])
            for k in range(len(serieses[i][j])):
                prod_fail[-1][-1].append(prod_fail[-1][-1][-1] * (1 - serieses[i][j][k]))
    
    xi = copy.deepcopy(indices)
    xi_bar = copy.deepcopy(indices)
    for i in range(len(serieses)):
        for j in range(len(serieses[i])):
            xi[i][j][0] = 0.0
            xi_bar[i][j][0] = 1.0 # * np.exp(-discount_factor * lengths[i][j])

            # Add in one dummy thing. Quantity shouldn't matter.
            xi[i][j].append(0.0)
            xi_bar[i][j].append(1.0)

    # P's contain the Big-Xis. These are initialized to 1.0, since xi_bar[i][j][0] = 1.0.
    P = []
    for i in range(len(serieses)):
        P.append(1.0)
    
    # PProd is the product of all P[i], i.e., the product of all the big-Xi's at any iteration.
    PProd = 1.0

    # STEP 3: Evaluate by increasing the stage bit by bit
    # Compute xi's for each control
    C = []
    for i in range(len(serieses)):
        C.append([])
        for j in range(len(serieses[i])):
            C[-1].append(0)

    # This is the cumulative discounted mass that is pushed over, i.e., the sum of all the A^{(w)}'s.
    total_mass_over = 0.0

    for rank, (i,j,k) in enumerate(rank_to_location):
        # print(i,j)
        # print(serieses[i][j])
        # print(indices[i][j])
        assert rank_to_index[rank] == indices[i][j][C[i][j]], f'{i,j,k} {rank_to_index[rank]} {indices[i][j][C[i][j]]}' # must be indeed right rank
        assert k == C[i][j] # Could be tied by inf
        
        # print(i,j,k)
        # print(prod_fail[i][j][k-1])
        ## Update xi and xi_bar
        xi[i][j][k+1] = xi[i][j][k] + \
            prod_fail[i][j][k] * serieses[i][j][k] * np.exp(-discount_factor * (k+1) * lengths[i][j])
        xi_bar[i][j][k+1] = xi_bar[i][j][k] * np.exp(-discount_factor * lengths[i][j]) * (1.0 - serieses[i][j][k])
        
        ## Update C
        C[i][j] += 1

        ##  Update all P's and PProd
        PProd /= P[i] # ---> Remove the effect of old P[i]
        # Compute new P[i]
        # TODO: can do in constant time, currently O(m). Optimize by caching cumprod of xi[i][j] over j = 0...
        # But we stick to the paper's version to be consistent.
        # We do this division and multiplication to PProd to avoid recomputing the entire product every time.
        tmp = 1.0
        P[i] = 0.0
        for j_ in range(len(serieses[i])):
            P[i] += tmp * xi_bar[i][j_][C[i][j_]]
            tmp *= xi[i][j_][C[i][j_]]
        PProd *= P[i] # ---> Add the effect of new P[i]

        # Get discounted mass that is pushed over
        mass_over = PProd
        mass_over /= P[i] # This is the first product term in the A^{(w)}'s
        # TODO: can do in constant time again by maintaining cumprod over xis
        for j_ in range(len(serieses[i])):
            if j != j_:
                mass_over *= xi[i][j_][C[i][j_]]
            else:
                mass_over *= xi_bar[i][j][k] * np.exp(-discount_factor * lengths[i][j]) * serieses[i][j][k]
        total_mass_over += mass_over

    return total_mass_over


def compute_exact_gradients_from_gittins_policy(
                          serieses: List[List[List[float]]], 
                          discount_factor: float, 
                          lengths: List[List[int]],
                          indices: List[List[List[int]]]
                          ):
    '''
    HEAVILY ADAPTED from the evaluation itself.

    Compute performance of some index policy. Using dynamic programming and not sampling
    Note: gittins contains the gittins index typically
    But it can also contain any index policy --- it doesn't have to be gittins.
    '''
    
    """
    for i in range(len(serieses)):
        print('---------')
        for j in range(len(serieses[i])):
            print(indices[i][j])
    print('=================================')
    for i in range(len(serieses)):
        print('---------')
        for j in range(len(serieses[i])):
            print(lengths[i][j])
    """
            
    # STEP 1: Preprocessing. Sort all indices. 
    # TODO: Favor the ones with lower arm index for tiebreaks
    # and if there is still a tie, favor the earlier control index.
    to_sort = []
    for i in range(len(indices)):
        for j in range(len(indices[i])):
            for k in range(len(indices[i][j])):
                index = indices[i][j][k]
                # index, -i, -j, -k # TODO: This is the tiebreak order
                to_sort.append((index, -k, -i, i, j, k)) # tiebreak in order of earlier in control better.
    to_sort.sort(reverse=True)
    
    # Extract the indices from gittins and vice versa
    location_to_rank = copy.deepcopy(indices)
    for index, _, _, i, j, k in to_sort:
        location_to_rank[i][j][k] = index
    rank_to_location = []
    rank_to_index = []
    for rank, (index, _, _, i, j, k) in enumerate(to_sort):
        rank_to_location.append((i, j, k))
        rank_to_index.append(index)

    # STEP 2: Preprocess product probabilities that we keep using
    # 1 ---- fail, fail, fail, ..., fail, pass [k times, *including* the final pass]
    #   ---- we want to comptute for each control the product of the sequenceo failures
    # 2 ---- fail, fail, fail, ..., fail [k times in total]

    # Compute the product of the sequence of failures for each control
    # P_{ij}[k] = fail_{i,j,0} * fail_{i,j,1} * ... * fail_{i,j,k-1}
    # P[i][j][0] = 1.0
    # P[i][j][k] = fail_{i,j,0} * ... fail{i,j,k-1}
    prod_fail = []
    
    for i in range(len(serieses)):
        prod_fail.append([])
        for j in range(len(serieses[i])):
            prod_fail[-1].append([1.0])
            for k in range(len(serieses[i][j])):
                prod_fail[-1][-1].append(prod_fail[-1][-1][-1] * (1 - serieses[i][j][k]))
    
    xi = copy.deepcopy(indices)
    xi_bar = copy.deepcopy(indices)
    xi_d = copy.deepcopy(indices)
    xi_bar_d = copy.deepcopy(indices)
    for i in range(len(serieses)):
        for j in range(len(serieses[i])):
            xi[i][j][0] = 0.0
            xi_bar[i][j][0] = 1.0 # * np.exp(-discount_factor * lengths[i][j])
            xi_d[i][j][0] = 0.0
            xi_bar_d[i][j][0] = 0.0 # No lengths involved in the product

            # Add in one dummy thing. Quantity shouldn't matter.
            xi[i][j].append(0.0)
            xi_bar[i][j].append(1.0)
            xi_d[i][j].append(0.0)
            xi_bar_d[i][j].append(0.0)

    P = []
    P_d = [] # Derivatives for each j. So P_d[i][j] is the derivative of P[i] with respect to length[i,j]
    for i in range(len(serieses)):
        P.append(1.0)
        P_d.append([]) # Gradient with respect to each j
        for j in range(len(serieses[i])):
            P_d[-1].append(0.0) # Gradient is 0 when no stages have passed
    PProd = 1.0

    # STEP 3: Evaluate by increasing the stage bit by bit
    # Compute xi's for each control
    C = []
    for i in range(len(serieses)):
        C.append([])
        for j in range(len(serieses[i])):
            C[-1].append(0)

    total_mass_over = 0.0

    # grad_mass_over[i][j] is the gradient of total_mass_over with respect to lengths[i][j]
    grad_mass_over = []
    for i in range(len(serieses)):
        grad_mass_over.append([])
        for j in range(len(serieses[i])):
            grad_mass_over[-1].append(0.0)

    for rank, (i,j,k) in enumerate(rank_to_location):
        assert rank_to_index[rank] == indices[i][j][C[i][j]], f'{i,j,k} {rank_to_index[rank]} {indices[i][j][C[i][j]]}' # must be indeed right rank
        assert k == C[i][j], f'{i,j}, {k} {C[i][j]}' # Could be tied by inf
        
        ## Update xi and xi_bar
        xi[i][j][k+1] = xi[i][j][k] + \
            prod_fail[i][j][k] * serieses[i][j][k] * np.exp(-discount_factor * (k+1) * lengths[i][j])
        xi_bar[i][j][k+1] = xi_bar[i][j][k] * np.exp(-discount_factor * lengths[i][j]) * (1.0 - serieses[i][j][k])
        
        # Same as xi but with an extra factor multiplied to the term to be added.
        xi_d[i][j][k+1] = xi_d[i][j][k] + \
            (k + 1) * prod_fail[i][j][k] * serieses[i][j][k] * np.exp(-discount_factor * (k+1) * lengths[i][j])
        
        # Same as xi_bar but with an extra factor multiplied to the term to be added.
        # The factor is because the total length of continual failure has increased by 1.
        xi_bar_d[i][j][k+1] = xi_bar[i][j][k+1] * (k + 1)

        ## Update C
        C[i][j] += 1

        ##  Update all P's and PProd
        PProd /= P[i]
        # Compute new P[i]
        # TODO: can do in constant time, currently O(m). Optimize by caching cumprod of xi[i][j] over j = 0...
        tmp = 1.0
        P[i] = 0.0
        for j_ in range(len(serieses[i])):
            P[i] += tmp * xi_bar[i][j_][C[i][j_]]
            tmp *= xi[i][j_][C[i][j_]]

        PProd *= P[i]

        # Compute new P_d[i]
        tmp_d = [1.0] * len(serieses[i]) # This is the same as tmp but one for each gradient
        P_d[i] = [0.0] * len(serieses[i]) 
        
        # Update P_d # TODO: optimize
        for j_d in range(len(serieses[i])):
            for j_ in range(len(serieses[i])):
                if j_ > j_d:
                    P_d[i][j_d] += tmp_d[j_d] * xi_bar[i][j_][C[i][j_]]
                elif j_ == j_d:
                    P_d[i][j_d] += tmp_d[j_d] * xi_bar_d[i][j_][C[i][j_]]
                
                if j_ != j_d:
                    tmp_d[j_d] *= xi[i][j_][C[i][j_]] # By default.
                else:
                    tmp_d[j_d] *= xi_d[i][j_][C[i][j_]] # The derivative term

        # Get discounted mass that is pushed over <---- old from evaluation
        mass_over = PProd
        mass_over /= P[i]
        # TODO: can do in constant time again by maintaining cumprod over xis
        for j_ in range(len(serieses[i])):
            if j != j_:
                mass_over *= xi[i][j_][C[i][j_]]
            else:
                mass_over *= xi_bar[i][j][k] * np.exp(-discount_factor * lengths[i][j]) * serieses[i][j][k]
        total_mass_over += mass_over

        # Get gradient of discounted mass that is pushed over
        for i_d in range(len(serieses)):
            if i_d != i: # Easy case, no special updates
                for j_d in range(len(serieses[i_d])):
                    grad_mass_over[i_d][j_d] += mass_over / P[i_d] * P_d[i_d][j_d]
            elif i_d == i:
                # First preprocess product of all xis in i_d = i
                xi_prod = 1.0
                for j_ in range(len(serieses[i_d])):
                    xi_prod *= xi[i_d][j_][C[i_d][j_]]

                # Now compute gradient terms # TODO: FIX!!!!!
                for j_d in range(len(serieses[i_d])):
                    if j_d != j: # Slighly harder case
                        after_basic_mod = xi_prod / xi[i_d][j][C[i_d][j]] * xi_bar[i][j][k] * np.exp(-discount_factor * lengths[i_d][j]) * serieses[i_d][j][k]
                        
                        if xi[i_d][j_d][C[i_d][j_d]] == 0: # Handle special case where denom is 0, avoid 0/0
                            after_derivative_mod = 0.0
                        else:
                            after_derivative_mod = after_basic_mod / xi[i_d][j_d][C[i_d][j_d]] * xi_d[i_d][j_d][C[i_d][j_d]]

                        grad_mass_over[i_d][j_d] += PProd / P[i_d] * after_derivative_mod
                    else: # Very special case where gradient is also the selected vertex. Need to compensate
                        grad_mass_over[i_d][j_d] += PProd / P[i_d] * xi_prod / xi[i_d][j][C[i_d][j]] * xi_bar[i][j][k] * np.exp(-discount_factor * lengths[i_d][j]) * serieses[i_d][j][k] * (k + 1)
                        # print(PProd / P[i_d] * xi_prod / xi[i_d][j][C[i_d][j]] * xi_bar[i][j][k] * np.exp(-discount_factor * lengths[i_d][j]) * serieses[i_d][j][k] * (k + 1))
    # Correction for constant factors
    for i in range(len(serieses)):
        for j in range(len(serieses[i])):
            grad_mass_over[i][j] *= -discount_factor

    return grad_mass_over

def simulate_payoff_gittens(serieses: List[List[List[float]]], 
                            discount_factor: float, 
                            lengths: List[List[int]],
                            gittins: List[List[List[float]]]):
    # Note: gittins contains the gittins index typically
    # But it can also contain any index policy --- it doesn't have to be gittins.


    n = len(serieses)
    status = [(0, 0)] * n

    ttime = 0.0    
    C = []
    # Setup contribution
    for i in range(n):
        d = len(serieses[i])
        C.append([])
        for j in range(d):
            C[-1].append(0)

    while True:
        A = []
        stuck = True
        # For each arm, check if stuck, if so, continue
        for i in range(n):
            if status[i][1] >= len(serieses[i][status[i][0]]):
                # Stuck already.
                A.append(-1)
                continue
            
            stuck = False
            A.append(gittins[i][status[i][0]][status[i][1]])

        # Special case that all arms are stuck
        if stuck == True:
            C = []
            for i in range(n):
                d = len(serieses[i])
                C.append([])
                for j in range(d):
                    C[-1].append(0)
            # print('---')
            return 0.0, C

        # Get best arm index
        best_arm_id = np.argmax(A)
        # print(best_arm_id)
        # assert gittins[best_arm_id][status[best_arm_id][0]][status[best_arm_id][1]] >= 0.0
        l =  lengths[best_arm_id][status[best_arm_id][0]]
        ttime += l
        C[best_arm_id][status[best_arm_id][0]] += 1
        
        # Get next state
        p_succ = serieses[best_arm_id][status[best_arm_id][0]][status[best_arm_id][1]]
        succ = np.random.choice([True, False], p=[p_succ, 1-p_succ])
        if not succ:
            status[best_arm_id] = (status[best_arm_id][0], status[best_arm_id][1] + 1)
        else:
            status[best_arm_id] = (status[best_arm_id][0] + 1, 0)

        # Reached crown jewel
        if status[best_arm_id][0] == len(serieses[best_arm_id]):
            disc_reward = np.exp(-discount_factor * ttime)

            grads = copy.deepcopy(C)
            for i in range(n):
                for j in range(len(serieses[i])):
                        grads[i][j] *= -discount_factor * np.exp(-discount_factor * ttime)

            # print('------------')
            return disc_reward, grads
        
    
"""
order, gi = compute_special_gittens([[0.5, 0.4, 0.3], [0.6, 0.5, 0.4], [0.7, 0.6, 0.5]], 0.9, [1, 1, 1])
order, gi = compute_special_gittens([[0.9, 0.9, 0.9], [0.6, 0.5, 0.4], [0.7, 0.6, 0.01]], 0.9, [1, 1, 1])
print(order)
print(gi)

print('-------------')
print('Running Gradient Descent')
print('-------------')
"""

def convert_lengths_np_to_lists(L, S):
    j = 0
    C = []
    n = len(S)
    for i in range(n):
        C.append([])
        # for component_id in range(len(serieses[i])):
        for component_id in range(len(S[i])):
            C[-1].append(L[j])
            j += 1
    return C

def convert_lengths_lists_to_np(grad_lists, S):
    size_lengths= sum([len(x) for x in S])
    n = len(S)
    ret = np.zeros(size_lengths)
    j = 0
    for i in range(n):
        for component_id in range(len(S[i])):
            ret[j] = grad_lists[i][component_id]
            j += 1
    return ret

def solve(serieses, discount_factor, batch_size, learning_rate, num_iterations = 2000, use_greedy_indices=False):
    size_lengths= sum([len(x) for x in serieses])
    current_lengths = np.ones(size_lengths)
    current_lengths/= size_lengths
    for it in range(num_iterations):
        lengths_lists = convert_lengths_np_to_lists(current_lengths, serieses)

        if use_greedy_indices:
            # Compute greedy indices
            gittens = []
            for k in range(len(serieses)):
                _ , gi = compute_greedy_index(serieses[k], discount_factor, lengths_lists[k])
                gittens.append(gi)
        else:
            # Compute gittins indices and hence optimal policy
            gittens = []
            for k in range(len(serieses)):
                order, gi = compute_special_gittens(serieses[k], discount_factor, lengths_lists[k], verbose=True)
                gittens.append(gi)

        # Get batch gradients
        payoffs = []
        grads = []
        grads_numpy = np.zeros(size_lengths) 
        for j in range(batch_size):
            payoff, grad = simulate_payoff_gittens(serieses, discount_factor, lengths_lists, gittens)
            grad_numpy = convert_lengths_lists_to_np(grad, serieses)
            payoffs.append(payoff)
            grads.append(grad)
            grads_numpy += grad_numpy

        grads_numpy /= batch_size
        print(it, np.mean(payoffs))

        current_lengths *= np.exp(-learning_rate * grads_numpy)
        current_lengths /= np.sum(current_lengths)

        print(current_lengths)
   

    return convert_lengths_np_to_lists(current_lengths, serieses), payoffs

def solve_RM(serieses, discount_factor, batch_size, num_iterations = 2000, use_greedy_indices=False, use_exact_gradient=False):
    size_lengths= sum([len(x) for x in serieses])

    accum_lengths = np.zeros(size_lengths)
    payoffs_store = np.zeros(size_lengths)
    recieved_rewards = 0.0


    min_it_before_accum = 10
    times_accum = 0
    for it in range(num_iterations):
        regret = - payoffs_store + recieved_rewards

        if np.all(regret <= 0):
            current_lengths = np.ones(size_lengths)
            current_lengths/= size_lengths
        else:
            current_lengths = regret 
            current_lengths[current_lengths <= 0] = 0
            current_lengths /= np.sum(current_lengths)

        if it >= min_it_before_accum:
            accum_lengths += current_lengths
            times_accum += 1

        lengths_lists = convert_lengths_np_to_lists(current_lengths, serieses)

        if use_greedy_indices:
            # Compute greedy indices
            gittens = []
            for k in range(len(serieses)):
                _ , gi = compute_greedy_index(serieses[k], discount_factor, lengths_lists[k])
                gittens.append(gi)
        else:
            # Compute gittins indices and hence optimal policy
            gittens = []
            for k in range(len(serieses)):
                order, gi = compute_special_gittens(serieses[k], discount_factor, lengths_lists[k], verbose=True)
                gittens.append(gi)

        if use_exact_gradient == False:
            # Get batch gradients
            payoffs = []
            grads = []
            grads_numpy = np.zeros(size_lengths) 
            for j in range(batch_size):
                payoff, grad = simulate_payoff_gittens(serieses, discount_factor, lengths_lists, gittens)
                grad_numpy = convert_lengths_lists_to_np(grad, serieses)
                payoffs.append(payoff)
                grads.append(grad)
                grads_numpy += grad_numpy

            grads_numpy /= batch_size
        else:

            
            grad = compute_exact_gradients_from_gittins_policy(serieses, discount_factor, lengths_lists, gittens)        
            grads_numpy = convert_lengths_lists_to_np(grad, serieses)
            payoff = compute_exact_payoffs_from_gittins_policy(serieses, discount_factor, lengths_lists, gittens)
            payoffs = [payoff]

           
        # print('GRAD')
        # print(grads_numpy)
        
        print(it, np.mean(payoffs))

        # print(np.mean(payoffs))
        # print(np.var(payoffs))
        # print('grad', grads_numpy)
        
        recieved_rewards += np.dot(grads_numpy, current_lengths)
        payoffs_store += np.array(grads_numpy)

        # current_lengths *= np.exp(-learning_rate * grads_numpy)
        # current_lengths /= np.sum(current_lengths)

        
        # print('gittins', [g[0][0] for g in gittens])

    return convert_lengths_np_to_lists(accum_lengths / times_accum, serieses), payoffs

    return convert_lengths_np_to_lists(accum_lengths / num_iterations, serieses), payoffs

    return convert_lengths_np_to_lists(current_lengths, serieses), payoffs

if __name__ == '__main__':


    ### TO SHOW WHY ADAPTIVE IS IMPT

    serieses = [[gen_succ_probs_exp_backoff(1, 1.0, 0.001), gen_succ_probs_exp_backoff(1, 0.001, 0.001)]] * 5 + [[gen_succ_probs_exp_backoff(1, 0.2,0.001 )]]
    print(serieses)
    size_lengths= sum([len(x) for x in serieses])
    n  = len(serieses)
    discount_factor = 5
    batch_size = 1000
    learning_rate = 1.0

    gits = []
    # Optimal we compute
    bloop = [0.00398015, 0.00398015, 0.00398188, 0.00398188, 0.00398192, 0.00398192,
        0.0039882,  0.0039882,  0.00398312, 0.00398312, 0.96016946]


    bloop_lists = convert_lengths_np_to_lists(bloop)
    for i in range(n):
        order, git= compute_special_gittens(serieses[i], discount_factor, bloop_lists[i])
        gits.append(git)
    payoffs= []
    for k in range(100000):
        payoff, _ = simulate_payoff_gittens(serieses, discount_factor, bloop_lists, gits)
        payoffs.append(payoff)
    print(np.mean(payoffs))
    