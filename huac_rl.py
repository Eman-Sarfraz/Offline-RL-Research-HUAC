import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import copy

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Tanh()
        )
        self.max_action = max_action

    def forward(self, state):
        return self.max_action * self.net(state)

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, state, action):
        return self.net(torch.cat([state, action], 1))

class Value(nn.Module):
    def __init__(self, state_dim):
        super(Value, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, state):
        return self.net(state)

class HUAC:
    def __init__(
        self,
        state_dim,
        action_dim,
        max_action,
        device,
        num_critics=5,
        gamma=0.99,
        tau=0.005,
        beta=0.1,
        alpha_expectile=0.7,
        lambda_min=1.0,
        lambda_max=10.0,
        kappa=1.0,
        uncertainty_threshold=0.5,
        lr=3e-4
    ):
        self.device = device
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)

        self.critics = nn.ModuleList([Critic(state_dim, action_dim).to(device) for _ in range(num_critics)])
        self.critic_targets = copy.deepcopy(self.critics)
        self.critic_optimizers = [optim.Adam(c.parameters(), lr=lr) for c in self.critics]

        self.value = Value(state_dim).to(device)
        self.value_optimizer = optim.Adam(self.value.parameters(), lr=lr)

        self.max_action = max_action
        self.gamma = gamma
        self.tau = tau
        self.beta = beta
        self.alpha_expectile = alpha_expectile
        self.lambda_min = lambda_min
        self.lambda_max = lambda_max
        self.kappa = kappa
        self.uncertainty_threshold = uncertainty_threshold
        self.num_critics = num_critics

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        return self.actor(state).cpu().data.numpy().flatten()

    def expectile_loss(self, diff, alpha):
        weight = torch.where(diff < 0, 1 - alpha, alpha)
        return weight * (diff**2)

    def train(self, replay_buffer, batch_size=256):
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

        # --- Update Value Function (IQL style) ---
        with torch.no_grad():
            q_values = torch.stack([c(state, action) for c in self.critics])
            min_q = q_values.min(dim=0)[0]
        
        v = self.value(state)
        value_loss = self.expectile_loss(min_q - v, self.alpha_expectile).mean()
        
        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

        # --- Update Critic Ensemble ---
        with torch.no_grad():
            target_q = reward + not_done * self.gamma * self.value(next_state)

        # Calculate adaptive regularization coefficient lambda(s, a)
        with torch.no_grad():
            current_q_values_for_lambda = torch.stack([c(state, action) for c in self.critics])
            current_variance = current_q_values_for_lambda.var(dim=0)
            lambd = self.lambda_min + (self.lambda_max - self.lambda_min) * torch.sigmoid(
                self.kappa * (current_variance - self.uncertainty_threshold)
            )

        for i, critic in enumerate(self.critics):
            current_q = critic(state, action)
            td_loss = F.mse_loss(current_q, target_q)

            # Conservative Regularization
            random_actions = torch.FloatTensor(batch_size, action.shape[1]).uniform_(-self.max_action, self.max_action).to(self.device)
            q_ood = critic(state, random_actions)
            conservative_reg = q_ood.mean() - current_q.mean()

            critic_loss = td_loss + lambd.mean() * conservative_reg + self.beta * value_loss.detach()

            self.critic_optimizers[i].zero_grad()
            critic_loss.backward()
            self.critic_optimizers[i].step()

        # --- Update Actor (Advantage Weighted Regression style) ---
        with torch.no_grad():
            adv = min_q - v
            exp_adv = torch.exp(adv * 3.0).clamp(max=100.0)
        
        pi = self.actor(state)
        actor_loss = (exp_adv * F.mse_loss(pi, action, reduction='none')).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # --- Update Target Networks ---
        for i in range(self.num_critics):
            for param, target_param in zip(self.critics[i].parameters(), self.critic_targets[i].parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

class ReplayBuffer:
    def __init__(self, state_dim, action_dim, device, max_size=int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        self.device = device

        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))

    def add(self, state, action, next_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device)
        )
