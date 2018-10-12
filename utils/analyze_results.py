import argparse, glob, fnmatch, os, csv, json, re
from pathlib import Path

import numpy as np
import pandas
from scipy.interpolate import interp1d
import matplotlib as mpl
import matplotlib.style
mpl.use('TkAgg')
mpl.style.use('seaborn')
import matplotlib.pyplot as plt

def file_index_key(f):
	pattern = r'\d+$'
	key_match = re.findall(pattern, Path(f).stem)
	if len(key_match):
		return int(key_match[0])
	return f

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-p', '--prefix', type=str, nargs='+', default=[''], help='average all files starting with prefix')
	parser.add_argument('-r', '--results_path', type=str, nargs='+', default=[''], help='path to results directory')
	parser.add_argument('-l', '--label', type=str, nargs='+', default=[None], help='labels')
	parser.add_argument('-m', '--max_index', type=int, help='max index of prefix match to use')
	parser.add_argument('-a', '--alpha', type=float, default=0.9, help='alpha for emwa')
	args = parser.parse_args()

	prefix = args.prefix
	if len(prefix) != len(args.results_path):
		prefix = prefix * len(args.results_path)

	label = args.label
	if len(label) != len(args.results_path):
		label = label * len(args.results_path)

	max_epoch = 0
	for prefix, results_path, label in zip(prefix, args.results_path, label):
		pattern = '{}*.csv'.format(prefix)

		filenames = fnmatch.filter(os.listdir(results_path), pattern)
		filenames.sort(key=file_index_key)

		epoch_to_rewards = {} # epochs x trials
		nfiles = 0
		for i, f in enumerate(filenames):
			if args.max_index and i >= args.max_index:
				break

			f_in = open(os.path.join(results_path, f), 'r')
			meta = f_in.readline()
			reader = csv.reader(f_in)
			headers = next(reader, None)
			if headers != ['episode', 'reward']:
				raise ValueError('result is malformed')

			for row in reader:
				row_dict = dict(zip(headers, row))
				e = int(row_dict['episode'])
				rewards = epoch_to_rewards.get(e, [])
				r = float(re.findall(r'[-+]?\d*\.\d+|\d+', row_dict['reward'])[0])
				rewards.append(r)
				if len(rewards) == 1:
					epoch_to_rewards[e] = rewards

			nfiles += 1

		epochs = np.array(sorted([int(k) for k in epoch_to_rewards.keys()]))
		max_epoch = max(max([int(k) for k in epoch_to_rewards]), max_epoch)
		nfiles = min([len(epoch_to_rewards[k]) for k in epoch_to_rewards])
		rewards_ewma = np.zeros((epochs.shape[0], nfiles))
		rewards_avg = np.zeros(len(epochs))
		rewards_std = np.zeros(len(epochs))

		for i, e in enumerate(epochs):
			rewards_ewma[i, :] = epoch_to_rewards[e][:nfiles]

		for j in range(rewards_ewma.shape[1]):
			df = pandas.DataFrame(rewards_ewma[:, j])
			rewards_ewma[:, j] = np.array(df.ewm(alpha=args.alpha).mean()).squeeze()

		for e in enumerate(epochs):
			rewards_avg = rewards_ewma.mean(1)
			rewards_std = rewards_ewma.std(1)

		plt.plot(epochs, rewards_avg, linewidth=2, label=label)
		plt.fill_between(epochs, rewards_avg - rewards_std, 
			np.minimum(rewards_avg + rewards_std, 1), alpha=0.25)
	
	threshold_x = np.linspace(0, max_epoch, 2)
	plt.plot(threshold_x, np.ones(threshold_x.shape), 
		zorder=1, color='k', linestyle='dashed', linewidth=1, alpha=0.5, label='Oracle')
	plt.axis([0, max_epoch, 0.5, 1.01])
	plt.legend(loc='best')
	plt.ylabel('Normalized R (Optimal)')
	plt.xlabel('# Epochs')
	xtick_values = range(0, max_epoch + 1000, 1000)
	xtick_labels = ['{}k'.format(int(x/1000)) for x in xtick_values]
	xtick_labels[0] = ''
	plt.xticks(xtick_values, xtick_labels)
	ax = plt.gca()
	plt.show()

