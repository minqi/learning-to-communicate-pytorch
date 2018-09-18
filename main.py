from utils.dotdic import DotDic
from arena import Arena
from dqagent import DQAgent
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
	'model_dial': True,
	'model_target': True,
	'model_bn': True,
	'model_know_share': True,
	'model_action_aware': True,
	'model_rnn': 'gru',
	'model_rnn_size': 128,
	'model_rnn_dropout_rate': 0,
	'bs': 32,
	'learningrate': 5e-4,
	'nepisodes': 5000,
	'step': 100,
	'step_test': 10,
	'step_target': 100,
	'train_mode': True,
	'cuda': 0
})

def init_action_and_comm_bits(opt):
	opt.model_comm_narrow = opt.model_dial
	if opt.model_rnn == 'lstm':
		opt.model_rnn_states = 2*opt.model_rnn_layers
	elif opt.model_rnn == 'gru':
		opt.model_rnn_states = opt.model_rnn_layers

	if not opt.model_comm_narrow and opt.game_comm_bits > 0:
		opt.game_comm_bits = 2 ** opt.game_comm_bits

	if opt.game_comm_bits > 0 and opt.game_nagents > 1:
		opt.game_action_space_total = opt.game_action_space + opt.game_comm_bits
	else:
		opt.game_action_space_total = opt.game_action_space

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

def main(opt):
	# Initialize action and comm bit settings
	opt = init_action_and_comm_bits(opt)

	# Create game
	game = create_game(opt)

	# Create agents
	agents = []
	for a in range(opt.game_nagents):
		cnet = create_cnet(opt)
		agents.append(DQAgent(opt, model=cnet, index=a))

	# Create arena
	arena = Arena(opt, game)

	# Iterate episodes
	for e in range(opt.game_nepisodes):
		arena.run_episode(agents, train_mode=opt.train_mode)

	# Report model statistics

if __name__ == '__main__':
	main(opt)

