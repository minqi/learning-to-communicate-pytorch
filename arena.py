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

		return episode

	def create_step_record(self):
		opt = self.opt
		record = DotDic({})
		record.state_t = 0 
		record.terminal = torch.zeros(opt.bs)

		# Track actions at time t per agent
		record.a_t = torch.zeros(opt.bs, opt.game_nagents)
		if not opt.model_dial:
			record.a_comm_t = torch.zeros(opt.bs, opt.game_nagents)

		# Track messages sent at time t per agent
		if opt.comm_enabled:
			record.comm = torch.zeros(opt.bs, opt.game_nagents, opt.game_comm_bits)
			if opt.model_dial and opt.model_target:
				record.comm_target = record.comm.clone()
		record.d_comm = torch.zeros(opt.bs, opt.game_nagents, opt.game_comm_bits)

		# Track q_t and q_t_max per agent
		record.q_a = torch.zeros(opt.bs, opt.game_nagents)
		record.q_a_max = torch.zeros(opt.bs, opt.game_nagents)

		return record

	def run_episode(self, agents, train_mode=False):
		opt = self.opt
		game = self.game
		game.reset()

		step = 0
		episode = self.create_episode()
		s_t = game.get_state()
		episode.step_records.append(self.create_step_record())
		episode.step_records[-1].s_t = s_t 
		while step < opt.episode_steps and episode.ended.sum() < opt.bs:
			episode.step_records.append(self.create_step_record())

			for i in range(1, opt.game_nagents + 1):
				# Get received messages per agent per batch
				comm = None
				if opt.comm_enabled:
					comm = episode.step_records[step].comm.clone()
					comm_limited = self.game.get_comm_limited(step, i)
					if comm_limited is not None:
						comm_lim = torch.zeros(opt.bs, 1, opt.game_comm_bits)
						for b in range(opt.bs):
							comm_lim[b] = comm[b][comm_limited[b]]
						comm = comm_lim
					else:
						comm[:, i].zero_()

				# Get prev action per batch
				prev_action = None
				if opt.model_action_aware:
					prev_action = torch.zeros(opt.bs, dtype=torch.long)
					if step > 1:
						prev_action = episode.step_records[step - 1].a_t[:, i]
					if not opt.model_dial:
						prev_message = torch.zeros(opt.bs, dtype=torch.long)
						if step > 1:
							prev_message = episode_step_records[step - 1].a_comm_t[:, i]
						prev_action = (prev_action, prev_message)

				# Batch agent index for input into model
				batch_agent_index = torch.zeros(opt.bs, dtype=torch.long).fill_(i)

				agent_inputs = [
					s_t[:, i],
					comm,
					agents[i].hidden_t[step], # Hidden state
					prev_action,
					batch_agent_index
				]

				# Compute model ouput (Q function + message bits)
				hidden_t, q_t = agents[i].model(*agent_inputs)

				import pdb; pdb.set_trace()
				# agents[i].hidden_t[step + 1] = hidden_t
				
				# Choose next action


				# Choose next comm


				# Choose next action and comm for target network


			# update game state to next step


			# save rewards, terminal states, etc

			# episode[step] = self.create_step_record(s_t=game.get_state())


	def train(self, *agents):
		for e in range(opt.nepisodes):
			# run episode

			# backprop Q values

			# backprop message gradients

			# update parameters

			pass
