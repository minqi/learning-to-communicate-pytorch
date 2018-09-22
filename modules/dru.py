import numpy as np
import torch

class DRU:
	def __init__(self, sigma, comm_narrow=True):
		self.sigma = sigma
		self.comm_narrow = comm_narrow

	def regularize(self, m):	
		m_reg = m + torch.randn(m.size()) * self.sigma
		if self.comm_narrow:
			m_reg = torch.sigmoid(m_reg)
		else:
			m_reg = torch.softmax(m_reg, 0)
		return m_reg

	def discretize(self, m):
		return torch.sigmoid((m.gt(0.5).float() - 0.5) * self.sigma * 20)

	def forward(self, m, train_mode):
		if train_mode:
			return self.regularize(m)
		else:
			return self.discretize(m)
			