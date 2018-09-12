"""

Switch game

This class manages the state of the Switch game among multiple agents.

Actions:
1 = On
2 = Off
3 = Tell
4 = Nothing

"""
import numpy as np
import torch

from utils.dotdic import DotDic 

class SwitchGame:

	def __init__(self, opt):
		self.game_actions = DotDic({
			'ON': 1,
			'OFF': 2,
			'TELL': 3,
			'NOTHING': 4,
		})

		self.game_states = DotDic({
			'OUTSIDE': 0,
			'INSIDE': 1,
		})

		self.opt = opt

		# Set game defaults
		opt_game_default = DotDic({
			'game_action_space': 2,
			'game_reward_shift': 0,
			'game_comm_bits': 0,
			'game_comm_sigma': 2
		})
		for k in opt_game_default:
			if k not in self.opt:
				self.opt[k] = opt_game_default[k]

		self.opt.nsteps = 4 * self.opt.game_nagents - 6

		self.reward_all_live = 1
		self.reward_all_die = -1 

		self.reset()

	def reset(self):
		# Step count
		self.step_count = 0

		# Rewards
		self.reward = torch.zeros(self.opt.bs, self.opt.game_nagents)

		# Who has been in the room?
		self.has_been = torch.zeros(self.opt.bs, self.opt.nsteps, self.opt.game_nagents)

		# Terminal state
		self.terminal = torch.zeros(self.opt.bs)

		# Active agent
		self.active_agent = torch.zeros(self.opt.bs, self.opt.nsteps)
		for b in xrange(self.opt.bs):
			for step in xrange(self.opt.nsteps):
				agent_id = 1 + np.random.randint(self.opt.game_nagents)
				self.active_agent[b][step] = agent_id
				self.has_been[b][step][agent_id] = True

		return self

	def get_action_range(self, step, agent):
		action_range = np.zeros((self.opt.bs, 2))
		if self.opt.model_dial:
			bound = self.opt.game_action_space
			for b in xrange(self.opt.bs):
				if self.active_agent[b][step] == agent:
					action_range[b] = np.array([1, self.opt.game_action_space])

			return action_range
		else:
			comm_range = np.zeros((self.opt.bs, 2))
			for b in xrange(self.opt.bs):
				if self.active_agent[b][step] == agent:
					action_range[b] = np.array([1, self.opt.game_action_space])
					comm_range[b] = np.array(
						[self.opt.game_action_space, self.opt.game_action_space_total])

			return action_range, comm_range

	def get_comm_limited(self, step, i):
		if self.opt.game_comm_limited:
			comm_lim = np.zeros(self.opt.bs)
			for b in xrange(self.opt.bs):
				if step > 0 and i == self.active_agent[b][step]:
					comm_lim[b] = self.active_agent[b][step - 1]
			return comm_lim
		return None

	def get_reward(self, a_t):
		# Return reward for action a_t by active agent
		for b in xrange(self.opt.bs):
			active_agent = self.active_agent[b][self.step_counter]
			if a_t[b][active_agent] == self.game_actions.TELL and not self.terminal[b]:
				has_been = self.has_been[b][:self.step_counter].sum(0).gt(0).sum().item()
				if has_been == self.opt.game_nagents:
					self.reward[b] = self.reward_all_live
				else:
					self.reward[b] = self.reward_all_die
				self.terminal[b] = True
			elif self.step_counter == self.opt.nsteps and not self.terminal[b]:
				self.terminal[b] = True

		return self.reward.clone(), self.terminal.clone()

	def step(self, a_t):
		reward, terminal = self.get_reward(a_t)
		self.step_counter += 1

		return reward, terminal

	def get_state(self):
		state = torch.zeros(self.opt.game_nagents, self.opt.bs)

		# Get the state of the game
		for a in xrange(self.opt.game_nagents):
			for b in xrange(self.opt.bs):
				if self.active_agent[b][self.step_counter] == agent:
					state[a][b] = self.game_states.INSIDE

		return state
