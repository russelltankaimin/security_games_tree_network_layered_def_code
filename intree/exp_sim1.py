from frontier_manager import simulate_single_game

def run_monte_carlo(network, num_simulations=10000, discount_rate=0.05):
    """
    Runs the game loop thousands of times to extract statistically significant metrics.
    """
    success_count = 0
    total_rewards = 0.0
    successful_times = []
    
    print(f"Running {num_simulations} Monte Carlo simulations...")
    
    for _ in range(num_simulations):
        is_success, time_spent, reward = simulate_single_game(network, discount_rate)
        
        if is_success:
            success_count += 1
            total_rewards += reward
            successful_times.append(time_spent)
            
    # Calculate aggregate metrics
    empirical_success_rate = success_count / num_simulations
    expected_discounted_reward = total_rewards / num_simulations
    
    if success_count > 0:
        avg_time_to_breach = sum(successful_times) / success_count
    else:
        avg_time_to_breach = float('inf')
        
    print("\n--- Simulation Results ---")
    print(f"Empirical Success Rate:     {empirical_success_rate * 100:.2f}%")
    print(f"Expected Discounted Reward: {expected_discounted_reward:.4f}")
    print(f"Avg Time (on success):      {avg_time_to_breach:.2f} units")
    
    return empirical_success_rate, expected_discounted_reward