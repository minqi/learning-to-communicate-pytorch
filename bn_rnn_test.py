import torch
from torch import nn, optim
from torch.autograd import Variable
import numpy as np

from modules import bn_rnn

HIDDEN_SIZE = 8

def compute_loss_accuracy(model, loss_fn, data, label):
	h0 = Variable(data.data.new(data.size(0), HIDDEN_SIZE).normal_(0, 0.1))
	c0 = Variable(data.data.new(data.size(0), HIDDEN_SIZE).normal_(0, 0.1))
	hidden = (h0, c0)

	_, (h, _) = model(x=data, hidden=hidden)
	loss = loss_fn(input=h, target=label)

	return loss

if __name__ == '__main__':
	model = bn_rnn.RNN(mode='lstm', input_size=1, hidden_size=HIDDEN_SIZE)

	x = (np.pi/8)*torch.FloatTensor(range(2*8)).expand(1, -1)
	y = np.sin(x).expand(1, -1)

	loss_fn = nn.MSELoss()
	model.train(True)
	params = list(model.parameters())
	optimizer = optim.RMSprop(params=params, lr=1e-3, momentum=0.9)

	num_epochs = 100
	for e in range(num_epochs):
		for t in range(len(x)):
			model.train(True)
			model.zero_grad()
			train_loss = compute_loss_accuracy(model, loss_fn, x, y)
			train_loss.backward()
			optimizer.step()

			print(e, t, train_loss)



