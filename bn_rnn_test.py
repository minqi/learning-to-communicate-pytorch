import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm
import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt

from modules import bn_rnn

HIDDEN_SIZE = 100
BATCH_SIZE = 128
MODE = 'gru'

class NN:
	def __init__(self, mode='lstm'):
		self.mode = mode
		self.rnn = bn_rnn.RNN(
			mode=self.mode, input_size=1, hidden_size=HIDDEN_SIZE, 
			num_layers=1, use_bn=True, bn_max_t=16, dropout_rate=0)
		self.rnn.train = True
		self.linear = nn.Linear(HIDDEN_SIZE, 1)
		self.params = list(self.rnn.parameters()) + list(self.linear.parameters())

	def predict(self, x, hidden):
		hidden_out, _ = self.rnn(x=x, hidden=hidden)
		linear_out = self.linear(hidden_out[:, -1, :])

		return linear_out

	def _compute_loss_accuracy(self, model, loss_fn, data, label):
		h0 = Variable(data.data.new(data.size(0), HIDDEN_SIZE).normal_(0, 0.1))
		c0 = Variable(data.data.new(data.size(0), HIDDEN_SIZE).normal_(0, 0.1))
		hidden = h0
		if self.mode == 'lstm':
			hidden = (h0, c0)

		pred = model.predict(data, hidden)
		loss = loss_fn(input=pred, target=label)

		return loss, pred

	def train(self, x, y):
		loss_fn = nn.MSELoss()
		optimizer = optim.RMSprop(params=model.params, lr=5e-3, momentum=0.9)

		num_epochs = 100
		pred = None
		num_batches = len(x) % BATCH_SIZE + 1
		for e in range(num_epochs):
			perm = torch.randperm(x.size()[0])
			x_perm = x.index_select(0, perm)
			y_perm = y.index_select(0, perm)

			for b in range(num_batches):
				optimizer.zero_grad()
				start_i = b
				end_i = (b + 1) * BATCH_SIZE
				train_loss, pred = self._compute_loss_accuracy(
					model, loss_fn, x_perm[start_i:end_i, :], y_perm[start_i:end_i, :])
				train_loss.backward()
				clip_grad_norm(parameters=model.params, max_norm=1)
				optimizer.step()

				print(e, b, train_loss.item())

if __name__ == '__main__':
	model = NN(mode=MODE)

	REPEAT_COUNT = 32
	x = (np.pi/8)*torch.FloatTensor(range(8*8)).view(-1, 4, 1).contiguous()
	x = x.repeat(REPEAT_COUNT, 1, 1)
	y = np.sin(x[:, x.size()[1] - 1])
	y = y + torch.FloatTensor(y.size()).normal_(0, .5)

	train_x = x.clone()
	train_y = y.clone()

	train_x = Variable(train_x)
	train_y = Variable(train_y)

	model.train(train_x, train_y)

	h0 = Variable(train_x.data.new(train_x.size(0), HIDDEN_SIZE).normal_(0, 0.1))
	c0 = Variable(train_x.data.new(train_x.size(0), HIDDEN_SIZE).normal_(0, 0.1))
	hidden = h0
	if MODE == 'lstm':
		hidden = (h0, c0)

	pred = model.predict(x, hidden)

	y_mean = y.view(REPEAT_COUNT, -1).mean(0).squeeze().numpy()
	pred_mean = pred.detach().view(REPEAT_COUNT, -1).mean(0).squeeze().numpy()

	plt.plot(np.array(range(len(y_mean))), y_mean)
	plt.plot(np.array(range(len(pred_mean))), pred_mean)
	plt.show()

