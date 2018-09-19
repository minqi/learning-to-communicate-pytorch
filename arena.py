import numpy as np
import torch

from utils.dotdic import DotDic
from modules.dru import DRU


class Arena:
	def __init__(self, opt, game):
		self.opt = opt
		self.game = game

	def create_episode(self):
		opt = self.opt
		episode = DotDic({})
		episode.steps = torch.zeros(opt.bs)
		episode.ended = torch.zeros(opt.bs)
		episode.r = torch.zeros(opt.bs, opt.game_nagents)
		episode.comm_per = torch.zeros(opt.bs)
		episode.comm_count = 0
		episode.non_comm_count = 0
		episode.step_records = []

		episode.d_err = torch.zeros(opt.bs, opt.game_action_space_total)
		episode.td_err = torch.zeros(opt.bs)
		episode.td_comm_err = torch.zeros(opt.bs)

		return episode

	def create_step_record(self, s_t):
		opt = self.opt
		record = DotDic({})
		record.state_t = state_t
		record.terminal = torch.zeros(opt.bs)

		record.a_t = torch.zeros(opt.bs, opt.game_nagents)
		if opt.model_dial:
			record.a_comm_t = torch.zeros(opt.bs, opt.game_nagents)

		# Initialize comm channel
		if opt.game_comm_bits > 0 and opt.game_nagents > 1:
			record.comm = torch.zeros(opt.bs, opt.game_nagents, opt.game_comm_bits)
			if opt.model_dial and opt.model_target
				record.comm_target = record.comm.clone()
		record.d_comm = torch.zeros(opt.bs, opt.game_nagents, opt.game_comm_bits)

	def select_action(self):
		pass

	def run_episode(self, *agents, train_mode=False):
		opt = self.opt
		game = self.game
		game.reset()

		num_steps = train_mode and opt.nsteps + 1 or opt.nsteps
		step = 0
		episode = self.create_episode()
		episode[step] = self.create_step_record(s_t=game.get_state()) 
		while step < num_steps and episode.ended.sum() < opt.bs:

			# Iterate agents
			for i in opt.game_nagents:
				pass

			step += 1
			episode[step] = self.create_step_record(s_t=game.get_state())


	def train(self, *agents):
		# run episodes

		# backward pass through each agent to learn from experience
		pass
