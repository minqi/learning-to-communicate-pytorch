import copy

from utils.dotdic import DotDic
from arena import Arena
from agent import CNetAgent
from switch.switch_game import SwitchGame
from switch.switch_cnet import SwitchCNet

"""
Play communication games
"""

# configure opts for Switch game with 3 DIAL agents
opt = DotDic({
	'game': 'switch',
	'game_nagents': 3,
	'game_action_space': 2,
	'game_comm_limited': True,
	'game_comm_bits': 1,
	'game_comm_sigma': 2,
	'nsteps': 6,
	'gamma': 1,
	'model_dial': False,
	'model_target': True,
	'model_bn': True,
	'model_know_share': True,
	'model_action_aware': True,
	'model_rnn': 'gru',
	'model_rnn_size': 128,
	'model_rnn_dropout_rate': 0,
	'bs': 32,
	'learningrate': 2e-4,
	'momentum': 0.95,
	'eps': 0.05,
	'nepisodes': 3000,
	'step': 100,
	'step_test': 10,
	'step_target': 100,
	'cuda': 0
})

def init_action_and_comm_bits(opt):
	opt.comm_enabled = opt.game_comm_bits > 0 and opt.game_nagents > 1

	opt.model_comm_narrow = opt.model_dial
	if opt.model_rnn == 'lstm':
		opt.model_rnn_states = 2*opt.model_rnn_layers
	elif opt.model_rnn == 'gru':
		opt.model_rnn_states = opt.model_rnn_layers

	if not opt.model_comm_narrow and opt.game_comm_bits > 0:
		opt.game_comm_bits = 2 ** opt.game_comm_bits

	if opt.comm_enabled:
		opt.game_action_space_total = opt.game_action_space + opt.game_comm_bits
	else:
		opt.game_action_space_total = opt.game_action_space

	return opt

def init_opt(opt):
	if not opt.model_rnn_layers:
		opt.model_rnn_layers = 2
	if not opt.model_avg_q:
		opt.model_avg_q = True
	opt = init_action_and_comm_bits(opt)
	return opt

def create_game(opt):
	game_name = opt.game.lower()
	if game_name == 'switch':
		return SwitchGame(opt)
	else:
		raise Exception('Unknown game: {}'.format(game_name))

def create_cnet(opt):
	game_name = opt.game.lower()
	if game_name == 'switch':
		return SwitchCNet(opt)
	else:
		raise Exception('Unknown game: {}'.format(game_name))

def create_agents(opt, game):
	agents = [None] # 1-index agents
	cnet = create_cnet(opt)
	cnet_target = copy.deepcopy(cnet)
	for i in range(1, opt.game_nagents + 1):
		agents.append(CNetAgent(opt, game=game, model=cnet, target=cnet_target, index=i))
		if not opt.model_know_share:
			cnet = create_cnet(opt)
			cnet_target = copy.deepcopy(cnet)
	return agents

def main(opt):
	# Initialize action and comm bit settings
	opt = init_opt(opt)

	game = create_game(opt)
	agents = create_agents(opt, game)
	arena = Arena(opt, game)

	arena.train(agents)

	# Report model statistics

if __name__ == '__main__':
	main(opt)

