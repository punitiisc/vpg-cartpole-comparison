import gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

env = gym.make("CartPole-v1")
gamma = 0.99
lr = 1e-2
writer = SummaryWriter("runs/single_traj")

class PolicyNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(4, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.fc(x)

policy = PolicyNetwork()
optimizer = optim.Adam(policy.parameters(), lr=lr)

def compute_returns(rewards, gamma=0.99):
    R = 0
    returns = []
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
    return returns

reward_list = []

for episode in range(500):
    state = env.reset()[0]
    log_probs, rewards = [], []
    done = False

    while not done:
        state_tensor = torch.FloatTensor(state)
        probs = policy(state_tensor)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_probs.append(dist.log_prob(action))

        state, reward, done, _, _ = env.step(action.item())
        rewards.append(reward)

    returns = torch.FloatTensor(compute_returns(rewards, gamma))

    # Normalize returns (optional)
    returns = (returns - returns.mean()) / (returns.std() + 1e-9)

    loss = -torch.stack(log_probs) * returns
    loss = loss.sum()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    total_reward = sum(rewards)
    reward_list.append(total_reward)
    writer.add_scalar("Reward", total_reward, episode)

writer.close()

# Plotting
plt.plot(reward_list)
plt.title("Single-Trajectory VPG Rewards")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.grid()
plt.savefig("reward_single_traj.png")
plt.show()

