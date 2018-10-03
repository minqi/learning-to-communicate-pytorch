"""
Switch game

This class manages the state of the Switch game among multiple agents.

RIAL Actions:

1 = Nothing
2 = Tell
3 = On
4 = Off
"""

import numpy as np
import torch

from utils.dotdic import DotDic 

class SwitchGame:

	def __init__(self, opt):
		self.game_actions = DotDic({
			'NOTHING': 1,
			'TELL': 2
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
			'game_comm_bits': 1,
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
		self.terminal = torch.zeros(self.opt.bs, dtype=torch.long)

		# Active agent
		self.active_agent = torch.zeros(self.opt.bs, self.opt.nsteps, dtype=torch.long) # 1-indexed agents
		for b in range(self.opt.bs):
			for step in range(self.opt.nsteps):
				agent_id = 1 + np.random.randint(self.opt.game_nagents)
				self.active_agent[b][step] = agent_id
				self.has_been[b][step][agent_id - 1] = 1

		return self

	def get_action_range(self, step, agent_id):
		"""
		Return 1-indexed indices into Q vector for valid actions and communications (so 0 represents no-op)
		"""
		opt = self.opt
		action_dtype = torch.long
		action_range = torch.zeros((self.opt.bs, 2), dtype=action_dtype)
		comm_range = torch.zeros((self.opt.bs, 2), dtype=action_dtype)
		for b in range(self.opt.bs): 
			if self.active_agent[b][step] == agent_id:
				action_range[b] = torch.tensor([1, opt.game_action_space], dtype=action_dtype)
				comm_range[b] = torch.tensor(
					[opt.game_action_space + 1, opt.game_action_space_total], dtype=action_dtype)
			else:
				action_range[b] = torch.tensor([1, 1], dtype=action_dtype)

		return action_range, comm_range

	def get_comm_limited(self, step, agent_id):
		if self.opt.game_comm_limited:
			comm_lim = torch.zeros(self.opt.bs, dtype=torch.long)
			for b in range(self.opt.bs):
				if step > 0 and agent_id == self.active_agent[b][step]:
					comm_lim[b] = self.active_agent[b][step - 1]
			return comm_lim
		return None

	def get_reward(self, a_t):
		# Return reward for action a_t by active agent
		for b in range(self.opt.bs):
			active_agent_idx = self.active_agent[b][self.step_count].item() - 1
			if a_t[b][active_agent_idx].item() == self.game_actions.TELL and not self.terminal[b].item():
				has_been = self.has_been[b][:self.step_count + 1].sum(0).gt(0).sum(0).item()
				if has_been == self.opt.game_nagents:
					self.reward[b] = self.reward_all_live
				else:
					self.reward[b] = self.reward_all_die
				self.terminal[b] = 1
			elif self.step_count == self.opt.nsteps - 1 and not self.terminal[b]:
				self.terminal[b] = 1

		return self.reward.clone(), self.terminal.clone()

	def step(self, a_t):
		reward, terminal = self.get_reward(a_t)
		self.step_count += 1

		return reward, terminal

	def get_state(self):
		state = torch.zeros(self.opt.bs, self.opt.game_nagents, dtype=torch.long)

		# Get the state of the game
		for b in range(self.opt.bs):
			for a in range(1, self.opt.game_nagents + 1):
				if self.active_agent[b][self.step_count] == a:
					state[b][a - 1] = self.game_states.INSIDE

		return state

	def god_strategy_reward(self, steps):
		reward = torch.zeros(self.opt.bs)
		for b in range(self.opt.bs):
			has_been = self.has_been[b][:self.opt.nsteps].sum(0).gt(0).sum().item()
			if has_been == self.opt.game_nagents:
				reward[b] = self.reward_all_live

		return reward

	def naive_strategy_reward(self):
		pass

	def get_stats(self, steps):
		stats = DotDic({})
		stats.god_reward = self.god_strategy_reward(steps)
		return stats

	def describe_game(self, b=0):
		print('has been:', self.has_been[b])
		print('num has been:', self.has_been[b].sum(0).gt(0).sum().item())
		print('active agents: ', self.active_agent[b])
		print('reward:', self.reward[b])

