from gensim.models import Word2Vec
import sys
from utils import Progbar
import io
import re
tokenizer_exceptions = [u'(amazon\\.com)', u'(google\\.com)', u'(a\\.k\\.a\\.)', u'(r\\.i\\.p\\.)', u'(states\\.)', u'(a\\.k\\.a)', u'(r\\.i\\.p)', u'(corps\\.)', u'(ph\\.d\\.)', u'(corp\\.)', u'(j\\.r\\.)', u'(b\\.s\\.)', u'(alex\\.)', u'(d\\.c\\.)', u'(b\\.c\\.)', u'(bros\\.)', u'(j\\.j\\.)', u'(mins\\.)', u'(\\.\\.\\.)', u'(dept\\.)', u'(a\\.i\\.)', u'(u\\.k\\.)', u'(c\\.k\\.)', u'(p\\.m\\.)', u'(reps\\.)', u'(prof\\.)', u'(p\\.s\\.)', u'(l\\.a\\.)', u'(i\\.e\\.)', u'(govt\\.)', u'(u\\.s\\.)', u'(t\\.v\\.)', u'(a\\.m\\.)', u'(cons\\.)', u'(e\\.g\\.)', u'(j\\.k\\.)', u'(ave\\.)', u'(gen\\.)', u'(feb\\.)', u'(mrs\\.)', u'(etc\\.)', u'(vol\\.)', u'(gov\\.)', u'(sec\\.)', u'(nov\\.)', u'(hrs\\.)', u'(sgt\\.)', u'(mon\\.)', u'(jan\\.)', u'(min\\.)', u'(pts\\.)', u'(rev\\.)', u'(inc\\.)', u'(est\\.)', u'(cal\\.)', u'(sat\\.)', u'(dec\\.)', u'(rep\\.)', u'(lbs\\.)', u'(mr\\.)', u'(jr\\.)', u'(km\\.)', u'(dc\\.)', u'(p\\.s)', u'(pp\\.)', u'(ex\\.)', u'(op\\.)', u'(co\\.)', u'(sp\\.)', u'(u\\.s)', u'(vs\\.)', u'(kg\\.)', u'(ms\\.)', u'(iv\\.)', u'(ca\\.)', u'(sr\\.)', u'(oz\\.)', u'(bc\\.)', u'(dr\\.)', u'(ga\\.)', u'(lb\\.)', u'(mi\\.)', u'(ad\\.)', u'(ft\\.)', u'(e\\.g)', u'(ed\\.)', u'(sc\\.)', u'(lt\\.)', u'(va\\.)', u'(la\\.)', u'(mt\\.)', u'(i\\.e)', u'(st\\.)', u'(mo\\.)']

def my_tokenize(sent):
    return [''.join(x) for x in re.findall("|".join(tokenizer_exceptions)+"|([0-9]+)|('\w{1,2}[^\w])|([\w]+)|([.,!?;'])",sent)]

class Generator(object):
	def __init__(self, data, n_epochs, lowercase = True):
		self.data = data
		self.epoch_number = 0
		self.model = None
		self.model_prefix = None
		self.n_epochs = n_epochs
		self.tokenizer = my_tokenize
		self.lowercase = lowercase

	def __iter__(self):
		if self.model is not None:
			# Training started
			self.epoch_number += 1
			print 'STARTING EPOCH : (%d/%d)'%(self.epoch_number, self.n_epochs)
			sys.stdout.flush()
		self.bar = Progbar(len(self.data))
		for idx,line in enumerate(self.data):
			self.bar.update(idx+1)
			line = line.lower() if self.lowercase else line
			yield self.tokenizer(line)
		if self.model is not None:
			if self.epoch_number != self.n_epochs:
				SAVE_FILE_NAME = self.model_prefix + '_iter_' + str(self.epoch_number) + '.model'  
			else:
				# Last Epoch
				SAVE_FILE_NAME = self.model_prefix + '.model'
			self.model.save(SAVE_FILE_NAME)

ROOT_DIR = "/home/bass/DataDir/RTE/"
if __name__ == "__main__":		
	VOCAB_FILE = ROOT_DIR + 'data/vocab.pkl'
	data_file = ROOT_DIR + 'data/train.txt'	
	model_prefix = ROOT_DIR + 'gensim_models/model_all_lowercase_'
	data = []
	print 'LOADING DATA...'
	sys.stdout.flush()
	with io.open(data_file, encoding='utf-8', mode='r', errors='replace') as f:
		for line in f:			
			data.append(line)		
	print 'LOADING DONE ...'
	sys.stdout.flush()
	generator = Generator(data, n_epochs=5, lowercase = True)
	model = Word2Vec(min_count = 10, iter = 5, size = 300, workers = 5)
	model.build_vocab(generator)
	generator.model = model 
	generator.model_prefix = model_prefix
	model.train(generator, total_examples=model.corpus_count, epochs=model.iter)
	print '\nVOCAB GENERATED'

