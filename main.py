from utils.dotdic import DotDic

"""

Play communication games

"""
# configure opts for Switch game with 3 DIAL agents
opt = DotDic({
	'game': 'Switch',
	'game_nagents': 3,
	'game_action_space': 2,
	'game_comm_limited': True,
	'game_comm_bits': 1,
	'game_comm_sigma': 2,
	'nsteps': 6,
	'gamma': 1,
	'model_dial': True,
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
	'cuda': 0
})

# Create agents

# Set-up aux vectors

# Iterate episodes

# Report model statistics
