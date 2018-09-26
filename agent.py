"""
Create agents for communication games
"""
import copy

import numpy as np
import torch
from torch import optim
from torch.nn.utils import clip_grad_norm

from utils.dotdic import DotDic
from modules.dru import DRU

class CNetAgent:
	def __init__(self, opt, game, model, target, index):
		self.opt = opt
		self.game = game
		self.model = model
		self.model_target = target
		self.episodes_seen = 0
		self.dru = DRU(opt.game_comm_sigma, opt.model_comm_narrow)
		self.id = index
		self.optimizer = optim.RMSprop(params=model.get_params(), lr=opt.learningrate, momentum=opt.momentum)

	def reset(self):
		self.model.reset_params()
		self.model_target.reset_params()
		self.episodes_seen = 0

	def _eps_flip(self, eps):
		# Sample Bernoulli with P(True) = eps
		return np.random.rand(self.opt.bs) < eps

	def _random_choice(self, items):
		return torch.from_numpy(np.random.choice(items, 1)).item()

	def select_action_and_comm(self, step, q, eps=0, target=False, train_mode=False):
		# eps-Greedy action selector
		if not train_mode:
			eps = 0
		opt = self.opt
		action_range, comm_range = self.game.get_action_range(step, self.id)
		action = torch.zeros(opt.bs, dtype=torch.long)
		action_value = torch.zeros(opt.bs)
		comm_dtype = opt.model_dial and torch.float or torch.long
		comm_dtype = torch.float
		comm_action = torch.zeros(opt.bs).int()
		comm_vector = torch.zeros(opt.bs, opt.game_comm_bits)
		comm_value = None
		if not opt.model_dial:
			comm_value = torch.zeros(opt.bs)

		should_select_random = self._eps_flip(eps)

		# Get action
		for b in range(opt.bs):
			q_a_range = range(0, opt.game_action_space)
			if action_range[b, 1].item() > 0:
				a_range = range(action_range[b, 0].item(), action_range[b, 1].item() + 1)
				if should_select_random[b]:
					action[b] = self._random_choice(a_range)
					action_value[b] = q[b, action[b]]
				else:
					action_value[b], action[b] = q[b, a_range].max(0)
				action[b] = action[b] + 1
			elif target:
				action_value[b], _ = q[b, q_a_range].max(0)

			q_c_range = range(opt.game_action_space, opt.game_action_space_total)
			if comm_range[b, 1].item() > 0:
				c_range = range(comm_range[b, 0].item(), comm_range[b, 1].item() + 1)
				if not opt.model_dial:
					if should_select_random[b]:
						comm_action[b] = self._random_choice(c_range)
						comm_value[b] = q[b, comm_action[b]]
						comm_action[b] = comm_action[b] - opt.game_action_space
					else:
						comm_value[b], comm_action[b] = q[b, c_range].max(0)
					comm_vector[b][comm_action[b]] = 1
					comm_action[b] = comm_action[b] + 1
				elif opt.model_dial:
					comm_vector[b] = self.dru.forward(q[b, q_c_range], train_mode=train_mode) # apply DRU
			elif not opt.model_dial and target:
				comm_value[b], _ = q[b, q_c_range].max(0)

		return (action, action_value), (comm_vector, comm_action, comm_value)

	def forward(t, *inputs):
		hidden, q = self.model_t[t].forward(*inputs)
		return hidden, q

	def episode_loss(self, episode):
		opt = self.opt
		total_loss = torch.zeros(opt.bs)
		for b in range(self.opt.bs):
			b_steps = episode.steps[b].item()
			for step in range(b_steps):
				record = episode.step_records[step]
				for i in range(self.opt.game_nagents):
					td_action = 0
					td_comm = 0
					r_t = record.r_t[b][i]
					q_a_t = record.q_a_t[b][i] 
					q_comm_t = 0

					if record.a_t[b][i].item() > 0:
						if record.terminal[b]:
							td_action = r_t - q_a_t
						else:
							next_record = episode.step_records[step + 1]
							q_next_max = next_record.q_a_max_t[b][i]
							if not opt.model_dial and opt.model_avg_q:
								q_next_max = (q_next_max + next_record.q_comm_max_t[b][i])/2.0
							td_action = r_t + opt.gamma * q_next_max - q_a_t

					if not opt.model_dial and record.a_comm_t[b][i].item() > 0:
						q_comm_t = record.q_comm_t[b][i]
						if record.terminal[b]:
							td_comm = r_t - q_comm_t
						else:
							next_record = episode.step_records[step + 1]
							q_next_max = next_record.q_comm_max_t[b][i]
							if opt.model_avg_q: 
								q_next_max = (q_next_max + next_record.q_a_max_t[b][i])/2.0
							td_comm = r_t + opt.gamma * q_next_max - q_comm_t

					loss_t = (td_action) ** 2 + (td_comm) ** 2
					total_loss[b] = total_loss[b] + loss_t
		loss = total_loss.sum()
		loss = loss/(self.opt.bs * self.opt.game_nagents)
		return loss

	def learn_from_episode(self, episode):
		self.optimizer.zero_grad()
		loss = self.episode_loss(episode)
		loss.backward(retain_graph=True)
		clip_grad_norm(parameters=self.model.get_params(), max_norm=10)
		self.optimizer.step()

		self.episodes_seen = self.episodes_seen + 1
		if self.episodes_seen % self.opt.step_target == 0:
			self.model_target.load_state_dict(self.model.state_dict())


