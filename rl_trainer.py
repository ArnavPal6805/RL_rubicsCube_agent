import torch
import torch.optim as optim
from torch.distributions import Categorical
from ml_agent import load_trained_model
from rubiks_env import RubiksEnv

def train_rl_agent(episodes=1000, gamma=0.99, lr=5e-6):
    # 1. Initialize Custom Environment
    env = RubiksEnv()
    
    # 2. Load the Pre-Trained Behavioral Cloning Weights
    model = load_trained_model("rubiks_cfop_weights.pth")
    model.train()
    
    # Lower learning rate for stable fine-tuning
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Running baseline tracker to stabilize policy gradients
    running_reward = None
    
    print("\nStarting Stabilized Phase 3: RL Fine-Tuning with Baseline...")
    
    for episode in range(episodes):
        state, _ = env.reset()
        log_probs = []
        rewards = []
        
        terminated = False
        truncated = False
        
        while not (terminated or truncated):
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            
            logits = model(state_tensor)
            probs = torch.softmax(logits, dim=1)
            
            m = Categorical(probs)
            action = m.sample()
            
            next_state, reward, terminated, truncated, _ = env.step(action.item())
            
            log_probs.append(m.log_prob(action))
            rewards.append(reward)
            
            state = next_state
            
        # Compute discounted returns
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + gamma * R
            returns.insert(0, R)
            
        returns = torch.tensor(returns, dtype=torch.float32)
        
        # Track running average reward for baseline subtraction
        total_ep_reward = sum(rewards)
        if running_reward is None:
            running_reward = total_ep_reward
        else:
            running_reward = 0.05 * total_ep_reward + 0.95 * running_reward
            
        # Baseline Subtraction (reduces gradient variance)
        advantage = returns - running_reward
        if len(advantage) > 1 and advantage.std() > 0:
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-9)
        
        # Calculate Policy Gradient Loss with baseline advantage
        policy_loss = []
        for log_prob, adv in zip(log_probs, advantage):
            policy_loss.append(-log_prob * adv)
            
        optimizer.zero_grad()
        if len(policy_loss) > 0:
            policy_loss = torch.cat(policy_loss).sum()
            policy_loss.backward()
            # Gradient clipping to prevent exploding loss spikes
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        else:
            policy_loss = torch.tensor(0.0)
        
        if (episode + 1) % 50 == 0:
            print(f"RL Episode {episode + 1}/{episodes} | Total Reward: {total_ep_reward:.2f} | Running Avg: {running_reward:.2f} | Loss: {policy_loss.item():.4f}")

    torch.save(model.state_dict(), "rubiks_rl_weights.pth")
    print("\nStabilized RL Training Complete! Saved as rubiks_rl_weights.pth")

if __name__ == "__main__":
    train_rl_agent()