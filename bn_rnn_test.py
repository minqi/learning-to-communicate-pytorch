import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm
import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt

from modules import bn_rnn

HIDDEN_SIZE = 8
BATCH_SIZE = 16

class NN:
	def __init__(self):
		self.rnn = bn_rnn.RNN(
			mode='lstm', input_size=1, hidden_size=HIDDEN_SIZE, 
			num_layers=2, use_bn=True, max_t=32, dropout_rate=0.0)
		self.rnn.train = True
		self.linear = nn.Linear(HIDDEN_SIZE, 1)
		self.params = list(self.rnn.parameters()) + list(self.linear.parameters())

	def predict(self, x, hidden):
		hidden_out, _ = self.rnn(x=x, hidden=hidden)
		linear_out = self.linear(hidden_out)

		return linear_out

	def _compute_loss_accuracy(self, model, loss_fn, data, label):
		h0 = Variable(data.data.new(data.size(0), HIDDEN_SIZE).normal_(0, 0.1))
		c0 = Variable(data.data.new(data.size(0), HIDDEN_SIZE).normal_(0, 0.1))
		hidden = (h0, c0)

		pred = model.predict(data, hidden)
		loss = loss_fn(input=pred, target=label)

		return loss, pred

	def train(self, x, y):
		loss_fn = nn.MSELoss()
		optimizer = optim.RMSprop(params=model.params, lr=1e-3, momentum=0.9)

		num_epochs = 500
		pred = None
		num_batches = len(x) % BATCH_SIZE + 1
		for e in range(num_epochs):
			for b in range(num_batches):
				optimizer.zero_grad()
				start_i = b
				end_i = (b + 1) * BATCH_SIZE
				train_loss, pred = self._compute_loss_accuracy(
					model, loss_fn, x[start_i:end_i, :], y[start_i:end_i, :])
				train_loss.backward()
				clip_grad_norm(parameters=model.params, max_norm=1)
				optimizer.step()

				print(e, b, train_loss.item())

		# plt.plot(np.array(range(len(x[0]))), pred[0].detach().numpy())
		# plt.plot(np.array(range(len(x[0]))), torch.mean(y, 0).numpy())
		# plt.show()


if __name__ == '__main__':
	model = NN()

	x = (np.pi/8)*torch.FloatTensor(range(8*8)).expand(1, -1).transpose(1, 0)
	# x = x.view(-1, 4, 1)
	# import pdb; pdb.set_trace()
	x = [x] * 128
	x = torch.stack(x, 0)
	y = np.sin(x) + torch.FloatTensor(x.size()).normal_(0, .4)

	train_x = x.clone()
	train_y = y.clone()
	for i in range(x.size()[0]):
		# perm = torch.randperm(len(x[0]))
		perm = torch.LongTensor(range(len(x[0])))
		train_x[i] = x[i][perm]
		train_y[i] = y[i][perm]

	train_x = Variable(train_x)
	train_y = Variable(train_y)

	model.train(train_x, train_y)

	h0 = Variable(train_x.data.new(train_x.size(0), HIDDEN_SIZE).normal_(0, 0.1))
	c0 = Variable(train_x.data.new(train_x.size(0), HIDDEN_SIZE).normal_(0, 0.1))
	hidden = (h0, c0)
	pred = model.predict(x, hidden)
	plt.plot(np.array(range(len(x[0]))), torch.mean(y, 0).numpy())
	plt.plot(np.array(range(len(x[0]))), torch.mean(pred.detach(), 0).numpy())
	plt.show()


