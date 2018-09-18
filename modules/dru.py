import numpy as np
import torch

class DRU:
	def __init__(self, opt):
		self.opt = opt

	def regularize(m):	
		m_reg = m + torch.randn(m.size()) * self.opt.game_comm_sigma
		if self.opt.model_comm_narrow:
			m_reg = torch.sigmoid(m_reg)
		else:
			m_reg = torch.softmax(m_reg, 0)
		return m_reg

	def discretize(m):
		return torch.sigmoid((m.gt(0.5) - 0.5) * self.opt.game_comm_sigma * 20)