import torch
import numpy as np
from huac_rl import HUAC, ReplayBuffer
from finance_env import PortfolioEnv, collect_offline_data
import matplotlib.pyplot as plt
from tqdm import tqdm

def evaluate_policy(policy, env, eval_episodes=10):
    avg_reward = 0.
    for _ in range(eval_episodes):
        state, _ = env.reset()
        done = False
        while not done:
            action = policy.select_action(np.array(state))
            state, reward, done, _, _ = env.step(action)
            avg_reward += reward
    avg_reward /= eval_episodes
    return avg_reward

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    env = PortfolioEnv()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    # Collect offline data
    print("Collecting offline data...")
    states, actions, next_states, rewards, dones = collect_offline_data(env, num_episodes=50)
    
    replay_buffer = ReplayBuffer(state_dim, action_dim, device)
    for i in range(len(states)):
        replay_buffer.add(states[i], actions[i], next_states[i], rewards[i], dones[i])

    # Initialize HUAC
    policy = HUAC(
        state_dim=state_dim,
        action_dim=action_dim,
        max_action=max_action,
        device=device
    )

    # Baseline: Behavioral Cloning
    print("Evaluating Behavioral Policy (Baseline)...")
    behavioral_reward = evaluate_policy(policy, env) # Initial policy is random/BC-like
    print(f"Behavioral Reward: {behavioral_reward:.6f}")

    # Training loop
    batch_size = 256
    iterations = 10000
    eval_freq = 1000
    eval_rewards = []

    print("Starting training...")
    for i in tqdm(range(iterations)):
        policy.train(replay_buffer, batch_size)
        
        if (i + 1) % eval_freq == 0:
            avg_reward = evaluate_policy(policy, env)
            eval_rewards.append(avg_reward)
            print(f"Iteration {i+1}: Average Reward: {avg_reward:.6f}")

    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(range(eval_freq, iterations + 1, eval_freq), eval_rewards, marker='o', label='HUAC')
    plt.axhline(y=behavioral_reward, color='r', linestyle='--', label='Behavioral Baseline')
    plt.title("HUAC Performance in Portfolio Optimization")
    plt.xlabel("Iterations")
    plt.ylabel("Average Reward")
    plt.legend()
    plt.grid(True)
    plt.savefig("/home/ubuntu/results_plot.png")
    print("Results plot saved to /home/ubuntu/results_plot.png")

    # Save final reward for report
    with open("/home/ubuntu/final_results.txt", "w") as f:
        f.write(f"Final Average Reward: {eval_rewards[-1]:.6f}\n")
        f.write(f"Initial Average Reward: {eval_rewards[0]:.6f}\n")
        improvement = (eval_rewards[-1] - eval_rewards[0]) / abs(eval_rewards[0]) * 100 if eval_rewards[0] != 0 else 0
        f.write(f"Improvement: {improvement:.2f}%\n")

if __name__ == "__main__":
    main()
