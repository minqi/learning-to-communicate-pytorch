import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable


class RecurrentBatchNorm(nn.Module):
	def __init__(self, input_size, gamma, beta, max_t):
		pass


class GRUCell(nn.Module):
	def __init__(self, input, hidden, w_ih, w_hh, b_ih=None, b_hh=None, dropout_rate=0.0, use_bn=True):
		pass


class LSTMCell(nn.Module):

	def __init__(self, input_size, hidden_size, w_ih=None, w_hh=None, b=None, dropout_rate=0.0, use_bn=True):
		super(LSTMCell, self).__init__()
		self.input_size = input_size
		self.hidden_size = hidden_size
		self.w_ih = w_ih
		self.w_hh = w_hh
		self.b = b
		self.dropout_rate = dropout_rate
		self.use_bn = use_bn
		self.reset_params()

	def reset_params(self):
		all_gate_size = 4*self.hidden_size
		if not self.w_ih:
			self.w_ih = nn.Parameter(
				nn.init.orthogonal_(torch.Tensor(all_gate_size, self.input_size)))
		if not self.w_hh:
			self.w_hh = nn.Parameter(
				nn.init.orthogonal_(torch.Tensor(all_gate_size, self.hidden_size)))
		if not self.b:
			self.b = nn.Parameter(torch.zeros(all_gate_size))

	def forward(self, x, hidden, t=0):
		c, h = hidden
		batch_bias = self.b.expand(x.size(0), *self.b.size())

		gates = F.linear(x, self.w_ih) + F.linear(h, self.w_hh) + batch_bias
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
		super(RNN, self).__init__()

		self.input_size = input_size
		self.hidden_size = hidden_size

		if mode == 'gru':
			cell = GRUCell
		elif mode == 'lstm':
			cell = LSTMCell
		else:
			raise Exception('Unknown mode: {}'.format(mode))

		self.layers = []
		for i in range(num_layers):
			layer = cell(input_size, hidden_size, dropout_rate=dropout_rate, use_bn=use_bn)
			self.layers.append(layer)
			setattr(self, 'layer_{}'.format(i), layer) # so torch will discover params

	def reset_params(self):
		for layer in self.layers:
			layer.reset_params()

	@staticmethod
	def layer_forward(layer, x, lengths, hidden):
		max_t = x.size(1)
		out = []
		for t in range(max_t):
			c_next, h_next = layer(x[:,t], hidden, t)
			mask = (t < lengths).float().unsqueeze(1).expand_as(h_next)
			c_next = c_next*mask + hidden[0]*(1 - mask)
			h_next = h_next*mask + hidden[1]*(1 - mask)
			hidden_next = (c_next, h_next)
			out.append(h_next)
			hidden = hidden_next
		out = torch.stack(out, 1)
		return out, hidden

	def forward(self, x, lengths=None, hidden=None, t=0):
		batch_size, max_t, _ = x.size()

		if not lengths:
			lengths = Variable(torch.LongTensor([max_t] * batch_size))
		if not hidden:
			hidden = Variable(torch.zero(batch_size, self.hidden_size))
			hidden = (hidden, hidden)

		c = []
		h = []
		layer_out = None
		for i, layer in enumerate(self.layers):
			layer_out, (layer_c, layer_h) = RNN.layer_forward(layer, x, lengths, hidden)
			c.append(layer_c)
			h.append(layer_h)
			x = layer_out

		c = torch.stack(c, 0)
		h = torch.stack(h, 0)

		return layer_out, (c, h)

