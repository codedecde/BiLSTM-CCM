import argparse
import subprocess
import os
import sys
def get_arguments():
	parser = argparse.ArgumentParser(description='Runner File')
	parser.add_argument('-model', action="store", default="no_features", dest="model", type=str)	
	parser.add_argument('-num_threads', action='store', default=2, dest='num_threads', type=int)
	opts = parser.parse_args(sys.argv[1:])
	return opts

if __name__ == "__main__":
	args = get_arguments()
	for thread_ix in xrange(args.num_threads):
		command =  'python BiLSTM.py'
		command += ' -train_flag True'
		command += ' -model ' + args.model 
		command += ' -thread_ix ' + str(thread_ix)
		command += ' -num_threads ' + str(args.num_threads)
		command += ' -use_partial True'
		log_file = 'logger_' + str(thread_ix) + '.log' 
		error_file = 'errlogger_' + str(thread_ix) + '.log'
		command += ' > ' + log_file + ' 2> ' + error_file
		os.system(command + ' &')