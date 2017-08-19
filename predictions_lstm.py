import lstm_model as lm
import keras
from keras.layers import Input, GRU, Embedding, Dense, Dropout, concatenate
from keras.models import Model
from keras.layers.wrappers import Bidirectional
from keras.callbacks import *
import keras.backend as K
from keras.optimizers import Adam
import numpy as np
import sys
from keras.utils import Progbar
import os
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import *
from keras.utils import to_categorical
import cPickle as cp
import pdb
import subprocess

def get_weights(model ,idx, options):
	BASE_DIR = '/home/bass/DataDir/BTPData/Keras_Models/'
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
	assert best_model != ''
	# print 'LOADING FOR IDX', idx, ' FROM FILE', MODEL_DIR + best_model
	sys.stdout.flush()	
	model.load_weights(MODEL_DIR + best_model)
	return model

def get_ccm_seq(filename, len_of_seq, dummy_set):
	f = open(filename)
	problem_line = f.readline()
	num_rows = int(f.readline().strip().split()[1])
	num_cols = int(f.readline().strip().split()[1])
	non_zeros = int(f.readline().strip().split()[1])
	status = f.readline()
	Objective = f.readline().strip().split()[1]
	blank_line = f.readline()
	column_line = f.readline()
	dash_line = f.readline()
	temp = ''
	ii = 0
	# ---- Reach the table with the variable and values --- #
	while(1):
		temp = f.readline()
		ii = int(filter(lambda x: x != '',temp.strip().split())[0])
		if(ii == num_rows):
			break
	blank_line = f.readline()
	column_line = f.readline()
	dash_line = f.readline()
    
	all_data = [None for idx in xrange(num_cols)] # Each col is a variable. This stores (var_name, var_value)
	curr_col = 0
	ii = 0
	counter = 1
	var_name = ''
	index = 0
	var_value = 0        
	# ---- Parse the table with the variable and values
	while(1):
		line = f.readline().strip().split('*')
		if(len(line) == 2):
			if(line[0] == ''):
				var_value = int(filter(lambda x: x != '',line[1].split())[0])           
				all_data[index - 1] = (var_name,var_value)          
			else:           
				x = line[0].split()
				index = int(x[0])
				var_name = x[1]
				var_value = int(filter(lambda x: x != '',line[1].split())[0])                       
				all_data[index - 1] = (var_name,var_value)          
		elif(len(line) == 1):
			x = line[0].split()
			index = int(x[0])
			var_name = x[1]
		if(all_data[num_cols-1] is None):
			continue
		else:
			break
	# ---- Generate the label sequence ---------- #
	pos_value = filter(lambda x: x[1] == 1, all_data) # All the variables with value 1		
	tag_seq = [None for ix in xrange(len_of_seq)]
	for (var_name, _) in pos_value:
		if var_name in dummy_set:
			# this is a dummy 
			continue
		var_name = var_name.split('_')
		ix = int(var_name[-1])
		assert tag_seq[ix] is None
		if ix == 0:
			tag_seq[ix] = var_name[1]
		else:
			assert tag_seq[ix-1] is not None and tag_seq[ix-1] == var_name[0]
			tag_seq[ix] = var_name[1]
	assert all([w is not None for w in tag_seq])
	return tag_seq

def ccm_inference(predictions, rho, options):
	# 1.1 Generate the CCM
	START_TAG = "START"
	weights = []
	for ix, pred in enumerate(predictions):
		weights.append({})
		if ix == 0:
			src = START_TAG
			for dst in options['CLASSES_2_IX']:
				weights[ix]["{}_{}".format(src,dst)] = pred[options['CLASSES_2_IX'][dst]]
		else:
			for src in options['CLASSES_2_IX']:
				for dst in options['CLASSES_2_IX']:
					weights[ix]["{}_{}".format(src,dst)] = pred[options['CLASSES_2_IX'][dst]]

	ccm_writebuf = 'maximize\n'
	# Objective function
	objective_function = ''
	for ix in xrange(len(weights)):
		if ix == 0:
			src = START_TAG
			for dest in options['CLASSES_2_IX']:				
				weight = weights[ix]["{}_{}".format(src, dest)]
				token = str(abs(weight)) + src + '_' + dest + '_' + str(ix)
				if weight >= 0.:
					objective_function = token if objective_function == '' else objective_function + ' + ' + token                            
				else:
					objective_function += ' - ' + token
		else:
			for src in options['CLASSES_2_IX']:
				for dest in options['CLASSES_2_IX']:					
					weight = weights[ix]["{}_{}".format(src, dest)]
					token = str(abs(weight)) + src + '_' + dest + '_' + str(ix)
					if weight >= 0.:
						objective_function += token if objective_function == '' else ' + ' + token                            
					else:
						objective_function += ' - ' + token

	# ---- The Attribute constraint (soft) --- #
	if rho is not None:
		objective_function += ' - ' + str(rho) + 'D1'
		dummy_set = set(['D1']) # A set of Dummy variables used to implement soft constraints
	else:
		dummy_set = set([])
	
	ccm_writebuf += objective_function
	#----- Now the constraints --- #
	ccm_writebuf += '\nsubject to\n'
	# ---- consistency for y_0 --- #
	constraints = ''
	for tag in options['CLASSES_2_IX']:
		token = START_TAG + '_' + tag + '_' + str(0)
		constraints += token if constraints == '' else ' + ' + token
	constraints += ' = 1\n'
	ccm_writebuf += constraints

	# ---- consistency between y_0 and y_1 -- #
	for src in options['CLASSES_2_IX']:
		constraints = START_TAG + '_' + src + '_' + str(0)
		for dest in options['CLASSES_2_IX']:
			token =  src + '_' + dest + '_' + str(1)
			constraints += ' - ' + token
		constraints += ' = 0\n'
		ccm_writebuf += constraints

	# ---- consistency between y_i and y_(i+1) -#
	for ix in xrange(1,len(weights)-1):
		for common_tag in options['CLASSES_2_IX']:
			constraints = ''
			for src in options['CLASSES_2_IX']:
				token =  src + '_' + common_tag + '_' + str(ix)
				constraints += token if constraints == '' else ' + ' + token
			for dest in options['CLASSES_2_IX']:
				token = common_tag + '_' + dest + '_' + str(ix + 1) 
				constraints += ' - ' + token
			constraints += ' = 0\n'
			ccm_writebuf += constraints		
	# ---- TYPE Constraint : There has to be at least one type -------- #
	constraints = START_TAG + '_' + 'type' + '_' + str(0)
	for ii in xrange(1,len(weights)):
		for src in options['CLASSES_2_IX']:
			token = src + '_' + 'type' + '_' + str(ii)
			constraints += ' + ' + token
	constraints += ' > 1\n'
	ccm_writebuf += constraints
	# --- ATTR Constraint : There has to be at least one attr (soft) -- #
	constraints = START_TAG + '_' + 'attr' + '_' + str(0)
	for ii in xrange(1,len(weights)):
		for src in options['CLASSES_2_IX']:
			token = src + '_' + 'attr' + '_' + str(ii)
			constraints += ' + ' + token
	constraints += ' D1'
	constraints += ' > 1\n'
	# --- Declare all variables as binary ------- #
	ccm_writebuf += 'binary\n'
	for ix in xrange(len(weights)):
		for tags in weights[ix]:
			variable = tags + '_' + str(ix)
			ccm_writebuf += variable + '\n'
	for dummy_vars in dummy_set:
		ccm_writebuf += dummy_vars + '\n'

	ccm_writebuf += 'end\n'

	# 1.2 Run the solver 
	FILENAME = "ilp_problem.lp"
	GLPK_LOCATION = "/usr/bin/glpsol"
	TEMP_FILENAME = "temp.out"
	open(FILENAME,'wb').write(ccm_writebuf)
	proc = subprocess.Popen([GLPK_LOCATION, '--cpxlp', FILENAME, '-o', TEMP_FILENAME], stdout = subprocess.PIPE)
	(out, err) = proc.communicate()
	if not err is None:
		print err
	seq_len = predictions.shape[0]
	# 1.3 Process the output and cleanup
	tag_seq = get_ccm_seq(TEMP_FILENAME, seq_len, dummy_set)
	proc = subprocess.Popen(['rm', FILENAME, TEMP_FILENAME], stdout = subprocess.PIPE)
	(out, err) = proc.communicate()
	return tag_seq
	

def get_prediction(model, post, idx, options, rho = None):
	# Preprocess the input
	sentence = [feat.split(' ')[0] for feat in post.split('\n')]
	sentence_vect = [options['VOCAB'][elem.lower()] + 1 if elem.lower() in options['VOCAB'] else len(options['VOCAB']) + 1 for elem in sentence]
	sentence_vect = pad_sequences([sentence_vect], maxlen=options['MAX_LEN'], padding='post')		
	model = get_weights(model, idx, options)
	predictions = model.predict(sentence_vect)
	predictions = predictions[:,:len(sentence),:] # 1 x len(sent) x num_classes
	# Sanity check	
	if rho is None:
		predictions = np.argmax(predictions, axis=-1).flatten()
		predictions_labels = [options['IX_2_CLASSES'][w] for w in predictions]
	else:		
		predictions_labels = ccm_inference(predictions[0], rho, options)		
	return '\n'.join(predictions_labels)


def get_rho(posts, ix):
	num_satisfied = 0.	
	for jx in xrange(len(posts)):
		if jx==ix:
			continue
		post = posts[jx]
		num_satisfied += 1. if (len(filter(lambda x: x == 'attr', [f.split(' ')[-1] for f in post.split('\n')]) ) > 0 ) else 0
	num_unsatisfied = len(posts) - 1 - num_satisfied
	# smoothing
	num_satisfied += 1.
	num_unsatisfied += 1.
	rho_attr = np.log(num_satisfied) - np.log(num_unsatisfied)
	return rho_attr


if __name__ == "__main__":
	options = lm.get_options()
	TRAIN_FILE = options['DATA_PATH']
	posts = open(TRAIN_FILE).read().split('\n\n')
	# rhos = [get_rho(posts, ix) for ix in xrange(len(posts))]
	rhos = [None for ix in xrange(len(posts))]
	RANGE = 136
	# RANGE = len(posts)	
	model = lm.create_model(options)
	predictions = []
	bar = Progbar(RANGE)
	for idx in xrange(RANGE):
		prediction = get_prediction(model,posts[idx], idx, options, rhos[ix])
		predictions.append(prediction)
		bar.update(idx+1)
	PREDICTION_FILE = '/home/bass/DataDir/BTPData/Predictions_New/prediction_keras.txt'
	open(PREDICTION_FILE,'wb').write('\n\n'.join(predictions))
