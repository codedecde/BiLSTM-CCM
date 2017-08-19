import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import sys
import os
import subprocess
from sklearn.metrics import accuracy_score
import numpy as np
import pdb
torch.manual_seed(1)

def to_scalar(var):
	# returns a python float
	return var.view(-1).data.tolist()[0]

def log_sum_exp_mat(matrix, axis = -1):    
	max_value, _ = torch.max(matrix, axis)    
	if axis != 0:
		ret_value = matrix -  max_value.repeat(1, matrix.size()[-1])
	else:
		ret_value = matrix - max_value.repeat(matrix.size()[0],1)    
	ret_value = torch.log( torch.sum( torch.exp(ret_value), axis) ) + max_value
	return ret_value

class CRF(nn.Module):
	'''
		This class implements a linear chain crf in pyTorch.
	'''
	def __init__(self, options, GPU = False):
		super(CRF, self).__init__()
		self.GPU = GPU		
		if self.GPU:           
			self.dtype = torch.cuda.FloatTensor
		else:            
			self.dtype = torch.FloatTensor
		self.options = options
		self.tag_to_ix = options['CLASSES_2_IX']
		self.ix_to_tag = {self.tag_to_ix[w]:w for w in self.tag_to_ix}
		
		self.tagset_size = len(self.tag_to_ix)
		
		# Matrix of transition parameters.  Entry i,j is the score of
		# transitioning *to* i *from* j.
		self.transitions = nn.Parameter(
		torch.randn(self.tagset_size, self.tagset_size)).type(self.dtype)
		
		self.initial_weights = nn.Parameter(
		torch.randn(self.tagset_size, 1)).type(self.dtype)

		self.final_weights = nn.Parameter(
		torch.randn(self.tagset_size, 1)).type(self.dtype)
		

	def _forward_alg(self, feats):		
		init_alphas = torch.Tensor(self.tagset_size, 1).fill_(0.).type(self.dtype)		
		forward_var = autograd.Variable(init_alphas).type(self.dtype)		
		for ix,feat in enumerate(feats):
			if ix == 0:
				forward_var += feat.view(self.tagset_size,1) + self.initial_weights
			else:			
				forward_var = feat.view(self.tagset_size,1) + log_sum_exp_mat( self.transitions + torch.transpose(forward_var.repeat(1, self.tagset_size), 0, 1), 1)
		terminal_var = forward_var + self.final_weights
		alpha = log_sum_exp_mat(terminal_var, 0 )
		return alpha

	def _score_sentence(self, feats, tags):
		# Gives the score of a provided tag sequence
		score = autograd.Variable(torch.Tensor([0])).type(self.dtype)		
		for i, feat in enumerate(feats):
			if i == 0:
				score = feat[tags[i]] + self.initial_weights[tags[i]]
			else:
				score = score + self.transitions[tags[i], tags[i-1]] + feat[tags[i]]
		score = score + self.final_weights[tags[-1]]
		return score

	def _viterbi_decode(self, feats):
		backpointers = []		
		init_alphas = torch.Tensor(self.tagset_size, 1).fill_(0.).type(self.dtype)		
		forward_var = autograd.Variable(init_alphas).type(self.dtype)		
		for ix,feat in enumerate(feats):
			if ix == 0:
				forward_var += feat.view(self.tagset_size, 1) + self.initial_weights
			else:				
				viterbi_vars, viterbi_idx = torch.max(self.transitions + torch.transpose( forward_var.repeat(1, self.tagset_size), 0 ,1), 1)
				forward_var = feat.view(self.tagset_size,1) + viterbi_vars
				backpointers.append(viterbi_idx)
		terminal_var = forward_var + self.final_weights						
		_ , best_tag_id = torch.max(terminal_var,0)
		best_tag_id = to_scalar(best_tag_id)
		path_score = terminal_var[best_tag_id]
		best_path = [best_tag_id]
		for bptrs_t in reversed(backpointers):
			best_tag_id = to_scalar(bptrs_t[best_tag_id])
			best_path.append(best_tag_id)			
		best_path.reverse()		
		return path_score, best_path	

	def _ccm_decode(self, lstm_feats, partial_labels = None):
		# --- Collecting the weights ----- #
		if not hasattr(self, 'FILENAME'):
			self.FILENAME = self.options['DATA_DIR'] + 'ilp_problem.lp'
		if not hasattr(self, 'TEMP_FILENAME'):
			self.TEMP_FILENAME = self.options['DATA_DIR'] + 'temp.out'
		if not hasattr(self, 'GLPK_LOCATION'):
			self.GLPK_LOCATION = '/usr/local/bin/glpsol'
		self.START_TAG = 'START'	
		weights = []        
		for ix,feat in enumerate(lstm_feats):
			weights.append({})
			if ix == 0:
				src = self.START_TAG
				for tag in self.tag_to_ix:                    
					weights[ix][src + '_' + tag] = feat[self.tag_to_ix[tag]] + self.initial_weights[self.tag_to_ix[tag]]
			else:
				for src in self.tag_to_ix:                    
					for dest in self.tag_to_ix:                        
						weights[ix][src + '_' + dest] = self.transitions[self.tag_to_ix[dest]][self.tag_to_ix[src]] + feat[self.tag_to_ix[dest]]
		
		for src in self.tag_to_ix:
			for dest in self.tag_to_ix:
				weights[-1][src + '_' + dest] += self.final_weights[self.tag_to_ix[dest]]

		# --- Now writing out the CCM ---- #
		ccm_writebuf = 'maximize\n'
		#---- The objective function ----- #
		objective_function = ''
		for ix in xrange(len(weights)):
			if ix == 0:
				src = self.START_TAG
				for dest in self.tag_to_ix:
					if self.GPU:
						weight = weights[ix][src + '_' + dest].cpu().data.numpy()[0]
					else:
						weight = weights[ix][src + '_' + dest].data.numpy()[0]					
					token = str(abs(weight)) + src + '_' + dest + '_' + str(ix)
					if weight >= 0.:
						objective_function = token if objective_function == '' else objective_function + ' + ' + token                            
					else:
						objective_function += ' - ' + token
			else:
				for src in self.tag_to_ix:
					for dest in self.tag_to_ix:
						if self.GPU:
							weight = weights[ix][src + '_' + dest].cpu().data.numpy()[0] 
						else:
							weight = weights[ix][src + '_' + dest].data.numpy()[0]

						token = str(abs(weight)) + src + '_' + dest + '_' + str(ix)
						if weight >= 0.:
							objective_function += token if objective_function == '' else ' + ' + token                            
						else:
							objective_function += ' - ' + token

		# ---- The Attribute constraint (soft) --- #
		if hasattr(self, 'constraint_penalty') and self.constraint_penalty is not None and 'AT_LEAST_ONE_ATTR' in self.constraint_penalty:
			objective_function += ' - ' + str(self.constraint_penalty['AT_LEAST_ONE_ATTR']) + 'D1'
			dummy_set = set(['D1']) # A set of Dummy variables used to implement soft constraints
			ccm_writebuf += objective_function
		else:
			dummy_set = set([])

		#----- Now the constraints --- #
		ccm_writebuf += '\nsubject to\n'
		# ---- consistency for y_0 --- #
		constraints = ''
		for tag in self.tag_to_ix:
			token = self.START_TAG + '_' + tag + '_' + str(0)
			constraints += token if constraints == '' else ' + ' + token
		constraints += ' = 1\n'
		ccm_writebuf += constraints

		# ---- consistency between y_0 and y_1 -- #
		for src in self.tag_to_ix:
			constraints = self.START_TAG + '_' + src + '_' + str(0)
			for dest in self.tag_to_ix:
				token =  src + '_' + dest + '_' + str(1)
				constraints += ' - ' + token
			constraints += ' = 0\n'
			ccm_writebuf += constraints

		# ---- consistency between y_i and y_(i+1) -#
		for ix in xrange(1,len(weights)-1):
			for common_tag in self.tag_to_ix:
				constraints = ''
				for src in self.tag_to_ix:
					token =  src + '_' + common_tag + '_' + str(ix)
					constraints += token if constraints == '' else ' + ' + token
				for dest in self.tag_to_ix:
					token = common_tag + '_' + dest + '_' + str(ix + 1) 
					constraints += ' - ' + token
				constraints += ' = 0\n'
				ccm_writebuf += constraints		
		# ---- TYPE Constraint : There has to be at least one type -------- #
		constraints = self.START_TAG + '_' + 'type' + '_' + str(0)
		for ii in xrange(1,len(weights)):
			for src in self.tag_to_ix:
				token = src + '_' + 'type' + '_' + str(ii)
				constraints += ' + ' + token
		constraints += ' > 1\n'
		ccm_writebuf += constraints
		# --- ATTR Constraint : There has to be at least one attr (soft) -- #
		constraints = self.START_TAG + '_' + 'attr' + '_' + str(0)
		for ii in xrange(1,len(weights)):
			for src in self.tag_to_ix:
				token = src + '_' + 'attr' + '_' + str(ii)
				constraints += ' + ' + token
		constraints += ' D1'
		constraints += ' > 1\n'
		# --- EM constraints --- #
		if partial_labels is not None:
			for ix in xrange(len(partial_labels)):
				if partial_labels[ix] != -1:
					constraints = ''
					dest = self.ix_to_tag[partial_labels[ix]]
					if ix == 0:
						src = self.START_TAG
						constraints = src + '_' + dest + '_' + str(ix) 
					else:
						for src in self.tag_to_ix:
							token = src + '_' + dest + '_' + str(ix)
							constraints += token if constraints == '' else ' + ' + token
					constraints += ' = 1\n'
					ccm_writebuf += constraints

		# --- Declare all variables as binary ------- #
		ccm_writebuf += 'binary\n'
		for ix in xrange(len(weights)):
			for tags in weights[ix]:
				variable = tags + '_' + str(ix)
				ccm_writebuf += variable + '\n'
		for dummy_vars in dummy_set:
			ccm_writebuf += dummy_vars + '\n'

		ccm_writebuf += 'end\n'        
		
		# --- Now call the ILP solver --------------- #		
		open(self.FILENAME,'wb').write(ccm_writebuf)		
		# os.system(self.GLPK_LOCATION + ' --cpxlp ' + self.FILENAME + ' -o ' + self.TEMP_FILENAME)
		proc = subprocess.Popen([self.GLPK_LOCATION, '--cpxlp', self.FILENAME, '-o', self.TEMP_FILENAME], stdout = subprocess.PIPE)
		(out, err) = proc.communicate()
		if not err is None:
			print err
		seq_len = int(lstm_feats.size()[0])
		tag_seq = self.get_ccm_seq(self.TEMP_FILENAME, seq_len, dummy_set)
		tag_seq_torch = torch.LongTensor(tag_seq).cuda() if self.GPU else torch.LongTensor(tag_seq)
		score = self._score_sentence(lstm_feats, tag_seq_torch)        
		# os.system('rm ' + self.FILENAME)
		# os.system('rm ' + self.TEMP_FILENAME)        

		proc = subprocess.Popen(['rm', self.FILENAME, self.TEMP_FILENAME], stdout = subprocess.PIPE)
		(out, err) = proc.communicate()
		return score, tag_seq


	def get_ccm_seq(self, filename, len_of_seq, dummy_set):
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
		tag_seq = [self.tag_to_ix[w] for w in tag_seq]
		return tag_seq
    

	def load_model(self, filename):
		# self.load_state_dict(torch.load(filename))
		raise NotImplementedError

	def save_model(self, *inputs):
		raise NotImplementedError

	def neg_log_likelihood(self, *inputs):
		raise NotImplementedError

	def _get_features(self, *inputs):
		raise NotImplementedError

	def shuffle_data(self, X, y):
		import random
		X_y = list(zip(X,y))
		random.shuffle(X_y)
		X,y = zip(*X_y)
		if type(X) != list:
			X = list(X)
			y = list(y)
		return X,y


