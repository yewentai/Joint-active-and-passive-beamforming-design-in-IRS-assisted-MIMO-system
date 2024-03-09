import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from collections import deque
from torch.autograd import Variable


class Actor(nn.Module):
    def __init__(self, input_size, hidden_size_1, hidden_size_2, output_size):
        super(Actor, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size_1)
        self.linear2 = nn.Linear(hidden_size_1, hidden_size_2)
        self.linear3 = nn.Linear(hidden_size_2, output_size)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = torch.tanh(self.linear3(x))
        return x


class Critic(nn.Module):
    def __init__(self, input_size, hidden_size_1, hidden_size_2, output_size):
        super(Critic, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size_1)
        self.linear2 = nn.Linear(hidden_size_1, hidden_size_2)
        self.linear3 = nn.Linear(hidden_size_2, output_size)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=2).squeeze()
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = torch.tanh(self.linear3(x))
        return x


class Replay:
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)

    def push(self, state, action, reward, next_state):
        experience = (state, action, np.array([reward]), next_state)
        self.buffer.append(experience)

    def sample(self, batch_size):
        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []

        batch = random.sample(self.buffer, batch_size)

        for experience in batch:
            state, action, reward, next_state = experience
            state_batch.append(state)
            action_batch.append(action)
            reward_batch.append(reward)
            next_state_batch.append(next_state)

        return state_batch, action_batch, reward_batch, next_state_batch

    def __len__(self):
        return len(self.buffer)


class DDPGagent:
    def __init__(
        self,
        num_actions,
        num_states,
        actor_learning_rate=1e-4,
        critic_learning_rate=1e-3,
        disc_fact=0.99,
        tau=1e-2,
        max_memory_size=50000,
    ):
        # Params
        self.num_states = num_states
        self.num_actions = num_actions
        self.disc_fact = disc_fact
        self.tau = tau
        self.hidden_size_1 = 300
        self.hidden_size_2 = 200

        # Networks
        self.actor_eval = Actor(
            self.num_states, self.hidden_size_1, self.hidden_size_2, self.num_actions
        )
        self.actor_target = Actor(
            self.num_states, self.hidden_size_1, self.hidden_size_2, self.num_actions
        )
        self.critic_eval = Critic(
            self.num_actions+self.num_states, self.hidden_size_1, self.hidden_size_2, 1
        )
        self.critic_target = Critic(
            self.num_actions+self.num_states, self.hidden_size_1, self.hidden_size_2, 1
        )

        for target_param, param in zip(
            self.actor_target.parameters(), self.actor_eval.parameters()
        ):
            target_param.data.copy_(param.data)

        for target_param, param in zip(
            self.critic_target.parameters(), self.critic_eval.parameters()
        ):
            target_param.data.copy_(param.data)

        # Training
        self.memory = Replay(max_memory_size)
        self.critic_criterion = nn.MSELoss()
        self.actor_optimizer = optim.Adam(
            self.actor_eval.parameters(), lr=actor_learning_rate
        )
        self.critic_optimizer = optim.Adam(
            self.critic_eval.parameters(), lr=critic_learning_rate
        )

    def get_action(self, state):
        state = Variable(torch.from_numpy(state).float().unsqueeze(0))
        # print(state.shape)
        action = np.pi*self.actor_eval.forward(state)
        # print(action.shape)
        action = action.detach().numpy()[0, 0]

        return action.reshape((1, self.num_actions))

    def update(self, batch_size):
        states, actions, rewards, next_states = self.memory.sample(batch_size)
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)

        # Critic loss
        Qvals = self.critic_eval.forward(states, actions)
        next_actions = self.actor_target.forward(next_states)
        next_Q = self.critic_target.forward(next_states, next_actions.detach())
        Qprime = rewards.clone()
        for i in range(len(rewards)):
            Qprime[i] = rewards[i] + self.disc_fact * next_Q[i]
        critic_loss = self.critic_criterion(Qvals, Qprime)

        # Actor loss
        policy_loss = -self.critic_eval.forward(
            states, self.actor_eval.forward(states)
        ).mean()

        # update networks
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # update target networks
        for target_param, param in zip(
            self.actor_target.parameters(), self.actor_eval.parameters()
        ):
            target_param.data.copy_(
                param.data * self.tau + target_param.data * (1.0 - self.tau)
            )

        for target_param, param in zip(
            self.critic_target.parameters(), self.critic_eval.parameters()
        ):
            target_param.data.copy_(
                param.data * self.tau + target_param.data * (1.0 - self.tau)
            )


class OUNoise(object):
    def __init__(
        self,
        action_dim,
        mu=0.0,
        theta=0.15,
        max_sigma=0.3,
        min_sigma=0.3,
        decay_period=100000,
    ):
        self.mu = mu
        self.theta = theta
        self.sigma = max_sigma
        self.max_sigma = max_sigma
        self.min_sigma = min_sigma
        self.decay_period = decay_period
        self.low = 0
        self.high = 2 * np.pi
        self.action_dim = action_dim
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu

    def evolve_state(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * \
            np.random.randn(self.action_dim)
        self.state = x + dx
        return self.state

    def get_action(self, action, t=0):
        ou_state = self.evolve_state()
        self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(
            1.0, t / self.decay_period
        )
        return np.clip(action + ou_state, self.low, self.high)
