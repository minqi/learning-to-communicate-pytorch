import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable


class RecurrentBatchNorm(nn.Module):
	"""
	Learns weight and bias separately for each time step
	"""
	def __init__(self, input_size, max_t, eps=1e-5, momentum=0.1):
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
			mean_t.zero_().detach()
			var_t = getattr(self, 'var_{}'.format(i))
			var_t.fill_(1).detach()

	def forward(self, x, t):
		if t >= self.max_t:
			t = self.max_t - 1

		mean_t = getattr(self, 'mean_{}'.format(t))
		var_t = getattr(self, 'var_{}'.format(t))

		return F.batch_norm(
			input=x, running_mean=mean_t, running_var=var_t,
			weight=self.weight, bias=self.bias,
			eps=self.eps, momentum=self.momentum,
			training=self.training)


class GRUCell(nn.Module):

	def __init__(
		self, input_size, hidden_size, w_ih=None, w_hh=None, b_ih=None, b_hh=None, 
		use_bn=True, bn_max_t=0):
		super(GRUCell, self).__init__()
		self.input_size = input_size
		self.hidden_size = hidden_size
		self.w_ih = w_ih
		self.w_hh = w_hh
		self.b_ih = b_ih
		self.b_hh = b_hh

		self.use_bn = use_bn
		if self.use_bn:
			self.bn_ih = RecurrentBatchNorm(input_size=3*hidden_size, max_t=bn_max_t)
			self.bn_hh = RecurrentBatchNorm(input_size=3*hidden_size, max_t=bn_max_t)
			self.bn_new = RecurrentBatchNorm(input_size=hidden_size, max_t=bn_max_t)

		self.reset_params()

	def reset_params(self):
		all_gate_size = 3*self.hidden_size
		if self.w_ih is None:
			self.w_ih = nn.Parameter(
				nn.init.orthogonal_(torch.Tensor(all_gate_size, self.input_size)))
		if self.w_hh is None:
			self.w_hh = nn.Parameter(
				nn.init.orthogonal_(torch.Tensor(all_gate_size, self.hidden_size)))
		if self.b_ih is None:
			self.b_ih = nn.Parameter(torch.zeros(all_gate_size))
		if self.b_hh is None:
			self.b_hh = nn.Parameter(torch.zeros(all_gate_size))

		if self.use_bn:
			self.bn_ih.reset_params()
			self.bn_hh.reset_params()
			self.bn_new.reset_params()

	def forward(self, x, hidden, t=0):
		gi = F.linear(x, self.w_ih, self.b_ih)
		gh = F.linear(hidden, self.w_hh, self.b_hh)
		if self.use_bn:
			gi = self.bn_ih(gi, t)
			gh = self.bn_hh(gh, t)
		i_r, i_i, i_n = gi.chunk(3, 1)
		h_r, h_i, h_n = gh.chunk(3, 1)

		resetgate = torch.sigmoid(i_r + h_r)
		inputgate = torch.sigmoid(i_i + h_i)
		newgate_input = i_n + resetgate*h_n
		if self.use_bn:
			newgate_input = self.bn_new(newgate_input, t)

		newgate = torch.tanh(newgate_input)
		h_y = newgate + inputgate * (hidden - newgate)
		
		return h_y

	def forward_sequence(self, x, lengths, hidden):
		max_length = x.size(1)
		out = []
		for t in range(max_length):
			h_next = self.forward(x[:,t], hidden, t)
			mask = (t < lengths).float().unsqueeze(1).expand_as(h_next)
			h_next = h_next*mask + hidden * (1 - mask)
			out.append(h_next)
			hidden = h_next
		out = torch.stack(out, 1)
		return out, hidden


class LSTMCell(nn.Module):

	def __init__(
		self, input_size, hidden_size, w_ih=None, w_hh=None, b=None, 
		use_bn=True, bn_max_t=0):
		super(LSTMCell, self).__init__()
		self.input_size = input_size
		self.hidden_size = hidden_size
		self.w_ih = w_ih
		self.w_hh = w_hh
		self.b = b

		self.use_bn = use_bn
		if self.use_bn:
			self.bn_ih = RecurrentBatchNorm(input_size=4*hidden_size, max_t=bn_max_t)
			self.bn_hh = RecurrentBatchNorm(input_size=4*hidden_size, max_t=bn_max_t)
			self.bn_c = RecurrentBatchNorm(input_size=hidden_size, max_t=bn_max_t)

		self.reset_params()

	def reset_params(self):
		all_gate_size = 4*self.hidden_size
		if self.w_ih is None:
			self.w_ih = nn.Parameter(
				nn.init.orthogonal_(torch.Tensor(all_gate_size, self.input_size)))
		if self.w_hh is None:
			self.w_hh = nn.Parameter(
				nn.init.orthogonal_(torch.Tensor(all_gate_size, self.hidden_size)))
		if self.b is None:
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

	def forward_sequence(self, x, lengths, hidden):
		max_length = x.size(1)
		out = []
		for t in range(max_length):
			c_next, h_next = self.forward(x[:,t], hidden, t)
			mask = (t < lengths).float().unsqueeze(1).expand_as(h_next)
			c_next = c_next*mask + hidden[0]*(1 - mask)
			h_next = h_next*mask + hidden[1]*(1 - mask)
			hidden_next = (c_next, h_next)
			out.append(h_next)
			hidden = hidden_next
		out = torch.stack(out, 1)
		return out, hidden

class RNN(nn.Module):

	def __init__(
		self, mode, input_size, hidden_size, num_layers=1, dropout_rate=0, 
		use_bn=True, bn_max_t=0):
		super(RNN, self).__init__()

		self.mode = mode
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
				layer_input_size, hidden_size, 
				use_bn=use_bn, bn_max_t=bn_max_t)
			self.layers.append(layer)
			setattr(self, 'layer_{}'.format(i), layer) # so torch will discover params

	def reset_params(self):
		for layer in self.layers:
			layer.reset_params()

	def layer_forward(layer, x, lengths, hidden):
		return layer.forward_sequence(x, lengths, hidden)

	def forward(self, x, lengths=None, hidden=None):
		batch_size, max_length, _ = x.size()

		if lengths is None:
			lengths = Variable(torch.LongTensor([max_length] * batch_size))
		if hidden is None:
			hidden = Variable(torch.zero(batch_size, self.hidden_size))
			if self.mode == 'lstm':
				hidden = (hidden, hidden)

		c = []
		h = []
		layer_out = None
		for i, layer in enumerate(self.layers):
			if self.mode == 'lstm':
				layer_out, (layer_c, layer_h) = RNN.layer_forward(layer, x, lengths, hidden)
				c.append(layer_c)
				h.append(layer_h)
			elif self.mode == 'gru':
				layer_out, hidden = RNN.layer_forward(layer, x, lengths, hidden)
				h.append(hidden)
			x = self.dropout_layer(layer_out)

		if self.mode == 'lstm':
			c = torch.stack(c, 0)
			h = torch.stack(h, 0)
			hidden_out = (c, h)
		elif self.mode == 'gru':
			hidden_out = torch.stack(h, 0)

		return layer_out, hidden_out

