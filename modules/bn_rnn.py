import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class RecurrentBatchNorm(nn.Module):
	def __init__(self, input_size, gamma, beta, max_t):
		pass


class GRUCell(nn.Module):
	def __init__(self, input, hidden, w_ih, w_hh, b_ih=None, b_hh=None, dropout_rate=0.0, use_bn=True):
		pass


class LSTMCell(nn.Module):

	def __init__(self, input_size, hidden_size, w_ih=None, w_hh=None, b=None, dropout_rate=0.0, use_bn=True):
		super(LSTM, self).__init__()
		self.input_size = input_size
		self.hidden_size = hidden_size
		self.w_ih = w_ih
		self.w_hh = w_hh
		self.b_ih = b_ih
		self.b_hh = b_hh
		self.dropout_rate = dropout_rate
		self.use_bn = use_bn
		self.reset_params()

	def reset_params(self):
		all_gate_size = 4*self.hidden_size
		if not self.w_ih:
			self.w_ih = init.orthogonal(torch.Tensor(all_gate_size, self.input_size))
		if not self.w_hh:
			self.w_hh = init.orthogonal(torch.Tensor(all_gate_size, self.input_size))
		if not self.b:
			self.b = torch.zeros(all_gate_size)

	def forward(self, x, hidden, t=0):
		c, h = hidden
		gates = F.linear(x, self.w_ih, 0) + F.linear(h, self.w_hh, 0) + self.b

		forgetgate, ingate, outgate, cellgate = gates.chunk(4, 1)

		forgetgate = torch.sigmoid(forgetgate)
		ingate = torch.sigmoid(ingate)
		outgate = torch.sigmoid(outgate)
		cellgate = torch.tanh(cellgate)

		c_next = (forgetgate * c) + (ingate * cellgate)
		h_next = outgate * torch.tanh(c_next)

		return c_next, h_next

class RNN(nn.Module):

	def __init__(self, mode, input_size, hidden_size, num_layers=1, dropout_rate=0, use_bn=True):
		super(StackedRNN, self).__init__()

		self.input_size = input_size
		self.hidden_size = hidden_size

		if mode == 'gru':
			cell = GRUCell
		elif mode == 'lstm':
			cell = LSTMCell
		else:
			raise Exception('Unknown mode: {}'.format(mode))

		self.layers = []
		for i in xrange(num_layers):
			layers.append(cell(input_size, hidden_size, dropout_rate=dropout_rate, use_bn=use_bn))

	def reset_params(self):
		for layer in self.layers:
			layer.reset_params()

	def cell_forward(self, cell, x, length, hidden)
		pass

	def forward(self, x, hidden=None, t=0):
		batch_size, max_t = x.size()
		if not hidden:
			hidden = Variable(torch.zero(batch_size, self.hidden_size))

		c = []
		h = []
		layer_out = None
		for layer in self.layers:
			layer_out, (layer_c, layer_h) = self.cell_forward()





