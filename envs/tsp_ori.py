import numpy as np
import torch
from tensordict import TensorDict


class TspEnv:

    def __init__(self, n_nodes, m_min, m_max, batch_size, device='cpu'):
        self.device = device

        self.n_nodes = n_nodes
        self.m_min = m_min
        self.m_max = m_max
        self.batch_size = batch_size

        self.locs = None
        self.agents = None
        self.actions = []

        # step-wise state
        self.first_node = None
        self.current_node = None

        # action masks
        self.action_masks = None

        self.agent_used = None
        self.prev_dist = None
        self.prev_node = None
        self.max_dist = None

        self.now_step = 0

    def _problem_rand_init(self):
        self.locs = torch.rand((self.batch_size, self.n_nodes, 2), device=self.device)
        self.agents = torch.randint(self.m_min, self.m_max + 1, (self.batch_size, ), device=self.device)
        self.agents[:] = self.agents[0]
        self.first_node = torch.zeros((self.batch_size, ), dtype=torch.int64).to(self.device)
        self.current_node = torch.zeros((self.batch_size, ), dtype=torch.int64).to(self.device)
        self.action_masks = torch.zeros((self.batch_size, self.n_nodes), dtype=torch.bool).to(self.device)
        self.agent_used = torch.zeros((self.batch_size, ), device=self.device, dtype=torch.long)
        self.prev_node = torch.zeros((self.batch_size, ), device=self.device, dtype=torch.long)
        self.prev_dist = torch.zeros((self.batch_size, ), device=self.device, dtype=torch.float32)
        self.max_dist = torch.zeros((self.batch_size, ), device=self.device, dtype=torch.float32)

    def reset(self, td=None):
        self.actions.clear()
        # reset batch_size
        if td is not None:
            self.batch_size = td.batch_size[0]
        
        self._problem_rand_init()
        if td is None:
            td = TensorDict({
                'locs': self.locs,
                'agents': self.agents
            }, batch_size=(self.batch_size, ), device=self.device)
        else:
            self.locs = td['locs']
            # self.agents = td['agents']

        td['first_node'] = self.first_node
        td['current_node'] = self.current_node
        td['mask'] = self.action_masks

        self.now_step = 0
        return td

    def step(self, td):
        self.now_step += 1

        self.actions.append(td['action'].clone())
        batch_axis = torch.arange(self.batch_size, device=self.device)
        action = td['action'].clone()

        assert torch.all(self.action_masks[batch_axis, action] == 0)

        self.action_masks[batch_axis, action] = 1
        self.action_masks[:, 0] = 0
        self.agent_used[action == 0] += 1
        self.action_masks[self.agent_used == self.agents - 1, 0] = 1

        assert action.shape == (self.locs.shape[0], )
        prev_loc = self.locs[torch.arange(self.locs.size(0)), self.prev_node]
        now_loc = self.locs[torch.arange(self.locs.size(0)), action]  # shape: (batch_size, 2)
        distance = torch.norm(prev_loc - now_loc, dim=1)  # shape: (batch_size,)
        self.prev_dist += distance
        self.max_dist[action == 0] = torch.maximum(self.max_dist[action == 0], self.prev_dist[action == 0])
        self.prev_dist[action == 0] = 0.
        self.prev_node = action.clone()


        if self.is_done():
            prev_loc = self.locs[torch.arange(self.locs.size(0)), self.prev_node]
            now_loc = self.locs[:, 0, :]
            distance = torch.norm(prev_loc - now_loc, dim=1)
            self.prev_dist += distance
            self.max_dist = torch.maximum(self.max_dist, self.prev_dist)

        td['current_node'] = action
        td['mask'] = self.action_masks
        return td

    def is_done(self):
        return torch.all(self.action_masks == 1)
    
    def get_reward(self):
        assert self.is_done()
        return -self.max_dist.clone()
        # acts = torch.stack(self.actions).transpose(0, 1).contiguous()
        # num_agents = self.agents.cpu().numpy()
        # locs_np = self.locs.cpu().float().numpy()
        # acts_np = acts.contiguous().cpu().numpy().astype(np.int32)

        # split_res = get_minmax_length(locs_np, acts_np, num_agents)
        # return -torch.from_numpy(split_res).to(self.device)
