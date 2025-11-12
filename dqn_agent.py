import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from DQN import DQN
from replay_memory import ReplayMemory, Transition

class DQNAgent:
    def __init__(self, state_dim, action_dim, hparams, device=None):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = hparams.get('gamma', 0.99)
        self.epsilon_start = hparams.get('epsilon_start', 0.9)
        self.epsilon_end = hparams.get('epsilon_end', 0.01)
        self.epsilon_decay = hparams.get('epsilon_decay', 2500)
        self.lr = hparams.get('learning_rate', 1e-3)
        self.batch_size = hparams.get('batch_size', 128)
        self.tau = hparams.get('tau', 0.005)
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.policy_net = DQN(state_dim, action_dim).to(self.device)
        self.target_net = DQN(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = AdamW(self.policy_net.parameters(), lr=self.lr, amsgrad=True)
        self.memory = ReplayMemory(hparams.get('memory_size', 10000))
        self.steps_done = 0

    def eps_threshold(self):
        return self.epsilon_end + (self.epsilon_start - self.epsilon_end) * np.exp(-1.0 * self.steps_done / self.epsilon_decay)

    def select_action(self, state, epsilon_unused=None):
        eps = self.eps_threshold()
        self.steps_done += 1
        if random.random() < eps:
            return random.randrange(self.action_dim)
        with torch.no_grad():
            s = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            q = self.policy_net(s)
            return int(q.argmax(dim=1).item())

    def store_transition(self, state, action, reward, next_state, done):
        ns = None if done else next_state
        self.memory.push(state, action, ns, reward)

    def soft_update(self):
        for tp, pp in zip(self.target_net.parameters(), self.policy_net.parameters()):
            tp.data.copy_(self.tau * pp.data + (1.0 - self.tau) * tp.data)

    def can_optimize(self):
        return len(self.memory) >= self.batch_size

    def train_step(self):
        if not self.can_optimize():
            return None
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.as_tensor(np.stack([s for s in batch.next_state if s is not None]), dtype=torch.float32, device=self.device) if non_final_mask.any() else None
        state_batch = torch.as_tensor(np.stack(batch.state), dtype=torch.float32, device=self.device)
        action_batch = torch.as_tensor(batch.action, dtype=torch.int64, device=self.device).unsqueeze(1)
        reward_batch = torch.as_tensor(batch.reward, dtype=torch.float32, device=self.device).unsqueeze(1)
        q_values = self.policy_net(state_batch).gather(1, action_batch)
        next_q = torch.zeros((self.batch_size, 1), device=self.device)
        if non_final_mask.any():
            next_vals = self.target_net(non_final_next_states).max(dim=1, keepdim=True)[0]
            next_q[non_final_mask] = next_vals
        target = reward_batch + self.gamma * next_q
        loss = F.smooth_l1_loss(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100.0)
        self.optimizer.step()
        return float(loss.item())

    def update_target(self):
        self.soft_update()
