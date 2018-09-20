"""
Create agents for communication games
"""
import copy
import numpy as np
import torch
from utils.dotdic import DotDic

class DQRNNAgent:
	def __init__(self, opt, game, model, index):
		self.game = game
		self.model = model
		self.eps = opt.eps
		# self.unroll_length = opt.nsteps + 1
		# self.unroll_model()

		self.id = index

		self.hidden_t = []
		self.hidden_target_t = []
		self.hidden_t.append(torch.zeros(opt.bs, opt.model_rnn_size))
		self.hidden_target_t.append(torch.zeros(opt.bs, opt.model_rnn_size))
		# self.state.append(torch.zeros(
		# 	opt.bs, opt.model_rnn_states, opt.model_rnn_size))
		# self.state_target.append(torch.zeros(
		# 	opt.bs, opt.model_rnn_states, opt.model_rnn_size))
		# self.q_next_max = torch.zeros(opt.bs, opt.episode_steps)
		# self.q_comm_next_max = torch.zeros(opt.bs, opt.episode_steps)

	def unroll_model(self, model):
		self.model_t = []
		self.model_target_t = []
		model_target = copy.deepcopy(model)
		for i in range(self.unroll_length):
			self.model_t.append(model)
			self.mode_target_t.append(model_target)

	def select_action(self, q, action_range, eps=0):
		# eps-Greedy action selector:
		# Select max action with probability 1 - eps,
		# and random action with probability eps
		if torch.random().item < eps:
			pass

		_, max_a = torch.max(q, 1)
		return max_a

	def select_comm(self):
		# eps-Greedy message selector for RIAL agent:
		# Select max 
		pass

	def forward(t, *inputs):
		hidden, q = self.model_t[t].forward(*inputs)
		return hidden, q

