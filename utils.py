import os 
import sys
import cPickle as cp 
import numpy as np
import time
def map_2_ix(word, word_2_ix, lower = False):
	if lower:
		word = word.lower()
	if word in word_2_ix:		
		return word_2_ix[word] + 1
	else:
		return len(word_2_ix) + 1

def map_2_feats(features, feats_2_ix):
	feature_vector = np.zeros((len(feats_2_ix),))
	for feature in features:
		if feature in feats_2_ix:
			feature_vector[feats_2_ix[feature]] = 1.
	return feature_vector

def data_generator(training_data, options):
	if type(training_data) != list:
		training_data = [training_data]
	X = []
	y = []
	for post in training_data:		
		post_y = []
		sentence = []
		feat_vectors = []
		for features in post.split('\n'):
			features = features.split(' ')			
			sentence.append(map_2_ix(features[0], options['VOCAB'], True))
			feat_vectors.append(map_2_feats(features[:-1], options['FEATURE_VOCAB']))
			if features[-1] not in options['CLASSES_2_IX']:
				assert options['USE_PARTIAL'], "Label other than type, attr, location, O and not using partially labeled data"
				assert features[-1] == '<UNK>', "The unknown tokens should be labeled <UNK>"
				post_y.append(-1)
			else:
				post_y.append(options['CLASSES_2_IX'][features[-1]])
		feat_vectors = np.array(feat_vectors)
		if options['USE_FEATURES']:
			post_X = (sentence, feat_vectors)
		else:
			post_X = sentence
		X.append(post_X)
		y.append(post_y)
	return X,y
def get_best_model_file(file_prefix, mode = "max", model_suffix = '.weights'):
	'''
		Finds the best model from a directory
	'''	
	mode = mode.lower()
	file_prefix = file_prefix.split('/')
	assert mode in set(["min", "max"])
	directory = '/'.join(file_prefix[:-1])
	model_prefix = file_prefix[-1]
	assert os.path.isdir(directory)
	best_model_metric = None
	best_model_file = None
	comparison_function = min if mode == "min" else max

	for file in os.listdir(directory):		
		if file.startswith(model_prefix) and file.endswith(model_suffix):			
			# metric = float('.'.join(file.split('_')[-1].split('.')[:-1]))
			metric = float(file.rstrip(model_suffix).split('_')[-1])
			if best_model_metric is None:
				best_model_metric = metric
				best_model_file = file
			elif metric == comparison_function(best_model_metric, metric):
				best_model_metric = metric
				best_model_file = file
	assert best_model_file is not None
	print 'LOADING WEIGHTS FROM ...'
	print best_model_file
	sys.stdout.flush()
	return directory + '/' + best_model_file

class Progbar(object):
	"""Displays a progress bar.

	# Arguments
		target: Total number of steps expected, None if unknown.
		interval: Minimum visual progress update interval (in seconds).
	"""

	def __init__(self, target, width=30, verbose=1, interval=0.05):
		self.width = width
		if target is None:
			target = -1
		self.target = target
		self.sum_values = {}
		self.unique_values = []
		self.start = time.time()
		self.last_update = 0
		self.interval = interval
		self.total_width = 0
		self.seen_so_far = 0
		self.verbose = verbose

	def update(self, current, values=None, force=False):
		"""Updates the progress bar.

		# Arguments
			current: Index of current step.
			values: List of tuples (name, value_for_last_step).
				The progress bar will display averages for these values.
			force: Whether to force visual progress update.
		"""
		values = values or []
		for k, v in values:
			if k not in self.sum_values:
				self.sum_values[k] = [v * (current - self.seen_so_far),
									  current - self.seen_so_far]
				self.unique_values.append(k)
			else:
				self.sum_values[k][0] += v * (current - self.seen_so_far)
				self.sum_values[k][1] += (current - self.seen_so_far)
		self.seen_so_far = current

		now = time.time()
		if self.verbose == 1:
			if not force and (now - self.last_update) < self.interval:
				return

			prev_total_width = self.total_width
			sys.stdout.write('\b' * prev_total_width)
			sys.stdout.write('\r')

			if self.target is not -1:
				numdigits = int(np.floor(np.log10(self.target))) + 1
				barstr = '%%%dd/%%%dd [' % (numdigits, numdigits)
				bar = barstr % (current, self.target)
				prog = float(current) / self.target
				prog_width = int(self.width * prog)
				if prog_width > 0:
					bar += ('=' * (prog_width - 1))
					if current < self.target:
						bar += '>'
					else:
						bar += '='
				bar += ('.' * (self.width - prog_width))
				bar += ']'
				sys.stdout.write(bar)
				self.total_width = len(bar)

			if current:
				time_per_unit = (now - self.start) / current
			else:
				time_per_unit = 0
			eta = time_per_unit * (self.target - current)
			info = ''
			if current < self.target and self.target is not -1:
				info += ' - ETA: %ds' % eta
			else:
				info += ' - %ds' % (now - self.start)
			for k in self.unique_values:
				info += ' - %s:' % k
				if isinstance(self.sum_values[k], list):
					avg = self.sum_values[k][0] / max(1, self.sum_values[k][1])
					if abs(avg) > 1e-3:
						info += ' %.4f' % avg
					else:
						info += ' %.4e' % avg
				else:
					info += ' %s' % self.sum_values[k]

			self.total_width += len(info)
			if prev_total_width > self.total_width:
				info += ((prev_total_width - self.total_width) * ' ')

			sys.stdout.write(info)
			sys.stdout.flush()

			if current >= self.target:
				sys.stdout.write('\n')

		if self.verbose == 2:
			if current >= self.target:
				info = '%ds' % (now - self.start)
				for k in self.unique_values:
					info += ' - %s:' % k
					avg = self.sum_values[k][0] / max(1, self.sum_values[k][1])
					if avg > 1e-3:
						info += ' %.4f' % avg
					else:
						info += ' %.4e' % avg
				sys.stdout.write(info + "\n")

		self.last_update = now

	def add(self, n, values=None):
		self.update(self.seen_so_far + n, values)

def get_weights(idx):
	BASE_DIR = '/scratch/cse/btech/cs1130773/BTP/SupervisedData/LSTM_MODELS/'
	MODEL_DIR = BASE_DIR + 'MODEL_' + str(idx) + '/'
	MAX_VAL_ACC = -1
	best_model = ''
	for filename in os.listdir(MODEL_DIR):
		if not filename.startswith('weights'):
			continue
		val_acc = int(filename.split('.')[2])
		if val_acc >= MAX_VAL_ACC:
			MAX_VAL_ACC = val_acc
			best_model = filename
	assert best_model != '', "Could Not find the best model file for directory %s" % (str(idx)) 
	print 'LOADING FOR IDX', idx, ' FROM FILE', MODEL_DIR + best_model
	sys.stdout.flush()
	return MODEL_DIR + best_model

