import torch
from torch import nn, optim
from torch.autograd import Variable
import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt

from modules import bn_rnn

HIDDEN_SIZE = 8

class NN:
	def __init__(self):
		self.rnn = bn_rnn.RNN(mode='lstm', input_size=1, hidden_size=HIDDEN_SIZE)
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

		num_epochs = 100
		pred = None
		for e in range(num_epochs):
			for t in range(len(x)):
				optimizer.zero_grad()
				train_loss, pred = self._compute_loss_accuracy(model, loss_fn, x, y)
				train_loss.backward()
				optimizer.step()

				print(e, t, train_loss.item())

		plt.plot(np.array(range(len(x[0]))), pred[0].detach().numpy())
		plt.plot(np.array(range(len(x[0]))), y[0].numpy())
		plt.show()


if __name__ == '__main__':
	model = NN()

	x = (np.pi/8)*torch.FloatTensor(range(2*8)).expand(1, -1).transpose(1, 0)
	x = [x, x, x]
	x = torch.stack(x, 0)
	y = np.sin(x)

	x = Variable(x)
	y = Variable(y)

	model.train(x, y)


