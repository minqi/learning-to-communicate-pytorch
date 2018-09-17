import numpy as np
import torch

from modules.dru import DRU


class Arena:
	def __init__(self, opt, game):
		self.opt = opt
		self.game = game

	def run_episode(self, *agents):
		pass

	def train(self, *agents):
		# run episodes

		# backward pass through each agent to learn from experience
		pass