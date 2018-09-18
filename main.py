from utils.dotdic import DotDic
from switch.switch_cnet import SwitchCNet

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

def init_action_and_comm_bits(opt):
	if not opt.model_comm_narrow and opt.game_comm_bits > 0:
		opt.game_comm_bits = 2 ** opt.game_comm_bits

	if opt.game_comm_bits > 0 and opt.game_nagents > 1:
		opt.game_action_space_total = opt.game_action_space + opt.game_comm_bits
	else:
		opt.game_action_space_total = opt.game_action_space

def main(opt):
	init_action_and_comm_bits(opt)

	# Create agents
	agent = SwitchCNet(opt)
	

	# Iterate episodes

	# Report model statistics

if __name__ == '__main__':
	main(opt)

