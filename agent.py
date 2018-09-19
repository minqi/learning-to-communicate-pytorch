"""
Create agents for communication games
"""
import copy
import numpy as np
import torch
from utils.dotdic import DotDic

class DQRNNAgent:
	def __init__(self, opt, model, index):
		self.unroll_length = opt.nsteps + 1
		self.unroll_model()

		self.id = torch.Tensor(opt.bs).fill_(index)

		self.input = []
		self.input_target = []
		self.state = []
		self.state_target = []
		self.state.append(torch.zeros(
			opt.bs, opt.model_rnn_states, opt.model_rnn_size))
		self.state_target.append(torch.zeros(
			opt.bs, opt.model_rnn_states, opt.model_rnn_size))
		self.q_next_max = []
		self.q_comm_next_max = []

	def unroll_model(self, model):
		self.model_t = []
		self.model_target_t = []
		model_target = copy.deepcopy(model)
		for i in range(self.unroll_length):
			self.model_t.append(model)
			self.mode_target_t.append(model_target)
