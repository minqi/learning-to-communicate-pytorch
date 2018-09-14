import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable


class RecurrentBatchNorm(nn.Module):
	"""
	Learns weight and bias separately for each time step
	"""
	def __init__(self, input_size, max_t, eps=1e-5, momentum=.1):
		super(RecurrentBatchNorm, self).__init__()
		self.max_t = max_t

		self.weight = nn.Parameter(torch.FloatTensor(input_size))
		self.bias = nn.Parameter(torch.zeros(input_size))
		self.eps = eps
		self.momentum = momentum
		self.train = True

		self.mean_t = [] 
		self.var_t = []
		for i in range(self.max_t):
			self.register_buffer('mean_{}'.format(i), torch.zeros(input_size))
			self.register_buffer('var_{}'.format(i), torch.ones(input_size))

		self.reset_params()

	def reset_params(self):
		self.weight.data.fill_(0.1)
		self.bias.data.zero_()
		for i in range(self.max_t):
			mean_t = getattr(self, 'mean_{}'.format(i))
			mean_t.zero_()
			var_t = getattr(self, 'var_{}'.format(i))
			var_t.fill_(1)

	def forward(self, x, t):
		if t >= self.max_t:
			t = self.max_t - 1

		mean_t = getattr(self, 'mean_{}'.format(t))
		var_t = getattr(self, 'var_{}'.format(t))

		# if t == 1:
			# print(self.weight)

		return F.batch_norm(
			input=x, running_mean=mean_t, running_var=var_t,
			weight=self.weight, bias=self.bias,
			eps=self.eps, momentum=self.momentum,
			training=self.training)


class GRUCell(nn.Module):
	def __init__(self, input, hidden, w_ih, w_hh, b_ih=None, b_hh=None, dropout_rate=0.0, use_bn=True):
		pass


class LSTMCell(nn.Module):

	def __init__(
		self, input_size, hidden_size, w_ih=None, w_hh=None, b=None, dropout_rate=0.0, 
		use_bn=True, max_t=0):
		super(LSTMCell, self).__init__()
		self.input_size = input_size
		self.hidden_size = hidden_size
		self.w_ih = w_ih
		self.w_hh = w_hh
		self.b = b
		self.dropout_rate = dropout_rate

		self.use_bn = use_bn
		if self.use_bn:
			self.bn_ih = RecurrentBatchNorm(input_size=4*hidden_size, max_t=max_t)
			self.bn_hh = RecurrentBatchNorm(input_size=4*hidden_size, max_t=max_t)
			self.bn_c = RecurrentBatchNorm(input_size=hidden_size, max_t=max_t)

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

		if self.use_bn:
			self.bn_ih.reset_params()
			self.bn_hh.reset_params()
			self.bn_c.reset_params()

	def forward(self, x, hidden, t=0):
		c, h = hidden
		batch_bias = self.b.expand(x.size(0), *self.b.size())
		ih = F.linear(x, self.w_ih)
		hh = F.linear(h, self.w_hh)
		if self.use_bn:
			ih = self.bn_ih(ih, t)
			hh = self.bn_hh(hh, t)
		gates = ih + hh + batch_bias
		forgetgate, ingate, outgate, cellgate = gates.chunk(4, 1)

		forgetgate = torch.sigmoid(forgetgate)
		ingate = torch.sigmoid(ingate)
		outgate = torch.sigmoid(outgate)
		cellgate = torch.tanh(cellgate)

		c_next = (forgetgate * c) + (ingate * cellgate)
		if self.use_bn:
			c_next = self.bn_c(c_next, t)

		h_next = outgate * torch.tanh(c_next)

		return c_next, h_next


class RNN(nn.Module):

	def __init__(
		self, mode, input_size, hidden_size, num_layers=1, dropout_rate=0, 
		use_bn=True, max_t=0):
		super(RNN, self).__init__()

		self.input_size = input_size
		self.hidden_size = hidden_size
		self.dropout_layer = nn.Dropout(dropout_rate)

		if mode == 'gru':
			cell = GRUCell
		elif mode == 'lstm':
			cell = LSTMCell
		else:
			raise Exception('Unknown mode: {}'.format(mode))

		self.layers = []
		for i in range(num_layers):
			layer_input_size = i == 0 and input_size or hidden_size
			layer = cell(
				layer_input_size, hidden_size, dropout_rate=dropout_rate, 
				use_bn=use_bn, max_t=max_t)
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

	def forward(self, x, lengths=None, hidden=None):
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
			x = self.dropout_layer(layer_out)
			# import pdb; pdb.set_trace()

		c = torch.stack(c, 0)
		h = torch.stack(h, 0)

		return layer_out, (c, h)

