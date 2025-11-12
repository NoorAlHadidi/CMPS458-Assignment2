import numpy as np
import torch
import torch.nn.functional as F
from dqn_agent import DQNAgent
from replay_memory import Transition

class DDQNAgent(DQNAgent):
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
            next_actions = self.policy_net(non_final_next_states).argmax(dim=1, keepdim=True)
            target_vals = self.target_net(non_final_next_states).gather(1, next_actions)
            next_q[non_final_mask] = target_vals
        target = reward_batch + self.gamma * next_q
        loss = F.smooth_l1_loss(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100.0)
        self.optimizer.step()
        return float(loss.item())
