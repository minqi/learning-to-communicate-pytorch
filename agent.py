"""
Create agents for communication games
"""
import copy
import numpy as np
import torch
from utils.dotdic import DotDic
from modules.dru import DRU

class CNetAgent:
	def __init__(self, opt, game, model, index):
		self.opt = opt
		self.game = game
		self.model = model
		self.model_target = copy.deepcopy(model)
		self.dru = DRU(opt.game_comm_sigma, opt.model_comm_narrow)
		# self.unroll_length = opt.nsteps + 1
		# self.unroll_model()

		self.id = index

		self.hidden_t = []
		self.hidden_target_t = []
		self.hidden_t.append(torch.zeros(opt.bs, opt.model_rnn_size))
		self.hidden_target_t.append(torch.zeros(opt.bs, opt.model_rnn_size))

	def unroll_model(self, model):
		self.model_t = []
		self.model_target_t = []
		model_target = copy.deepcopy(model)
		for i in range(self.unroll_length):
			self.model_t.append(model)
			self.mode_target_t.append(model_target)

	def _eps_flip(self, eps):
		# Sample Bernoulli with P(True) = eps
		return np.random.rand(self.opt.bs) < eps

	def _random_choice(self, items):
		return torch.from_numpy(np.random.choice(items, 1)).item()

	def select_action_and_comm(self, step, q, eps=0, train_mode=False):
		# eps-Greedy action selector
		opt = self.opt
		action_range, comm_range = self.game.get_action_range(step, self.id)
		action = torch.zeros(opt.bs, dtype=torch.long)
		action_value = torch.zeros(opt.bs)
		comm_dtype = opt.model_dial and torch.float or torch.long
		comm_action = torch.zeros(opt.bs)
		comm_vector = torch.zeros(opt.bs, opt.game_comm_bits)
		comm_value = None
		if not opt.model_dial:
			comm_value = torch.zeros(opt.bs)

		should_select_random = self._eps_flip(opt.eps)

		# Get action
		for b in range(opt.bs):
			if action_range[b, 1].item() > 0:
				a_range = range(action_range[b, 0].item()-1, action_range[b, 1].item())
				if should_select_random[b]:
					# Select random action
					action[b] = self._random_choice(a_range)
					action_value[b] = q[b, action[b]]
				else:
					action_value[b], action[b] = q[b, a_range].max(0)
				action[b] = action[b] + 1

			if comm_range[b, 1].item() > comm_range[b, 0].item():
				c_range = range(comm_range[b, 0].item(), comm_range[b, 1].item())
				if not opt.model_dial and comm_range[b, 1].item() > 0:
					if should_select_random[b]:
						# Select random comm
						comm_action[b] = self._random_choice(c_range)
						comm_value[b] = q[b, comm_action[b]]
					else:		
						comm_value[b], comm_action[b] = q[b, c_range].max(0)
					comm_action[b] = comm_action[b] + 1
					comm_vector[b, comm_action[b]] = 1
				else:
					comm_vector[b] = self.dru.forward(q[b, c_range], train_mode=train_mode) # apply DRU
			
		return (action, action_value), (comm_vector, comm_action, comm_value)


	def forward(t, *inputs):
		hidden, q = self.model_t[t].forward(*inputs)
		return hidden, q

