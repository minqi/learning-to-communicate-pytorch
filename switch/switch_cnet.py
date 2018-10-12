"""

DRQN-based agent that learns to communicate with other agents to play 
the Switch game.

"""
import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable


class SwitchCNet(nn.Module):

	def __init__(self, opt):
		super(SwitchCNet, self).__init__()

		self.opt = opt
		self.comm_size = opt.game_comm_bits
		self.init_param_range = (-0.08, 0.08)

		# Set up inputs
		self.agent_lookup = nn.Embedding(opt.game_nagents, opt.model_rnn_size)
		self.state_lookup = nn.Embedding(2, opt.model_rnn_size)

		# Action aware
		self.prev_message_lookup = None
		if opt.model_action_aware:
			if opt.model_dial:
				self.prev_action_lookup = nn.Embedding(opt.game_action_space_total, opt.model_rnn_size)
			else:
				self.prev_action_lookup = nn.Embedding(opt.game_action_space + 1, opt.model_rnn_size)
				self.prev_message_lookup = nn.Embedding(opt.game_comm_bits + 1, opt.model_rnn_size)

		# Communication enabled
		if opt.comm_enabled:
			self.messages_mlp = nn.Sequential()
			if opt.model_bn:
				self.messages_mlp.add_module('batchnorm1', nn.BatchNorm1d(self.comm_size))
			self.messages_mlp.add_module('linear1', nn.Linear(self.comm_size, opt.model_rnn_size))
			if opt.model_comm_narrow:
				self.messages_mlp.add_module('relu1', nn.ReLU(inplace=True))

		# Set up RNN
		dropout_rate = opt.model_rnn_dropout_rate or 0
		self.rnn = nn.GRU(input_size=opt.model_rnn_size, hidden_size=opt.model_rnn_size, 
			num_layers=opt.model_rnn_layers, dropout=dropout_rate, batch_first=True)

		# Set up outputs
		self.outputs = nn.Sequential()
		if dropout_rate > 0:
			self.outputs.add_module('dropout1', nn.Dropout(dropout_rate))
		self.outputs.add_module('linear1', nn.Linear(opt.model_rnn_size, opt.model_rnn_size))
		if opt.model_bn:
			self.outputs.add_module('batchnorm1', nn.BatchNorm1d(opt.model_rnn_size))
		self.outputs.add_module('relu1', nn.ReLU(inplace=True))
		self.outputs.add_module('linear2', nn.Linear(opt.model_rnn_size, opt.game_action_space_total))

	def get_params(self):
		return list(self.parameters())

	def reset_parameters(self):
		opt = self.opt
		self.messages_mlp.linear1.reset_parameters()
		self.rnn.reset_parameters()
		self.agent_lookup.reset_parameters()
		self.state_lookup.reset_parameters()
		self.prev_action_lookup.reset_parameters()
		if self.prev_message_lookup:
			self.prev_message_lookup.reset_parameters()
		if opt.comm_enabled and opt.model_dial:
			self.messages_mlp.batchnorm1.reset_parameters()
		self.outputs.linear1.reset_parameters()
		self.outputs.linear2.reset_parameters()
		for p in self.rnn.parameters():
			p.data.uniform_(*self.init_param_range)


	def forward(self, s_t, messages, hidden, prev_action, agent_index):
		opt = self.opt

		s_t = Variable(s_t)
		hidden = Variable(hidden)
		prev_message = None
		if opt.model_dial:
			if opt.model_action_aware:
				prev_action = Variable(prev_action)
		else:
			if opt.model_action_aware:
				prev_action, prev_message = prev_action
				prev_action = Variable(prev_action)
				prev_message = Variable(prev_message)
			messages = Variable(messages)
		agent_index = Variable(agent_index)

		z_a, z_o, z_u, z_m = [0]*4
		z_a = self.agent_lookup(agent_index)
		z_o = self.state_lookup(s_t)
		if opt.model_action_aware:
			z_u = self.prev_action_lookup(prev_action)
			if prev_message is not None:
				z_u += self.prev_message_lookup(prev_message)
		z_m = self.messages_mlp(messages.view(-1, self.comm_size))

		z = z_a + z_o + z_u + z_m
		z = z.unsqueeze(1)

		rnn_out, h_out = self.rnn(z, hidden)
		outputs = self.outputs(rnn_out[:, -1, :].squeeze())

		return h_out, outputs
