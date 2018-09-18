"""
Create agents for communication games
"""
import copy
import numpy as np
import torch
from utils.dotdic import DotDic

class DQAgent:
	def __init__(self, opt, model, index):
		self.model = model
		self.model_target = copy.deepcopy(model)

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