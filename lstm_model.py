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
def get_options():
	options = {}
	options['MAX_LEN'] = 250
	options['HIDDEN_DIM'] = 300
	options['CLASSES_2_IX'] = {'O':0, 'type':1, 'attr':2, 'location':3}
	options['IX_2_CLASSES'] = {options['CLASSES_2_IX'][w]:w for w in options['CLASSES_2_IX']}
	DATA_DIR = '/home/bass/DataDir/BTPData/'
	VOCAB_PATH = DATA_DIR + 'vocab_btp.pkl'
	options['VOCAB'] = cp.load(open(VOCAB_PATH))
	options['USE_EMBEDDING'] = True
	if options['USE_EMBEDDING']:
		EMBED_PATH = DATA_DIR  + 'embedding_matrix_btp.npy'
		options['EMBEDDING_MATRIX'] = np.load(file=open(EMBED_PATH))
	options['EMBEDDING_DIM'] = 200	
	options['DATA_PATH'] = DATA_DIR + 'Data_136_with_feats.txt'
	options['BATCH_SIZE'] = 32
	return options

def get_data(options):
	X = []
	y = []
	posts = open(options['DATA_PATH']).read().split('\n\n')
	for post in posts:
		_x = []
		_y = []
		for elem in post.split('\n'):
			elem = elem.split(' ')
			_x.append(options['VOCAB'][elem[0].lower()] + 1 if elem[0].lower() in options['VOCAB'] else len(options['VOCAB']) + 1)
			_y.append(options['CLASSES_2_IX'][elem[-1]])
		X.append(_x)
		y.append(_y)
	return X,y

def get_train_val(X,y,idx, options):
	X_train = X[:idx] + X[idx+1:]
	y_train = y[:idx] + y[idx+1:]
	X_train = pad_sequences(X_train, maxlen = options['MAX_LEN'], padding = 'post', value = 0, truncating = 'post')
	y_train = pad_sequences(y_train, maxlen = options['MAX_LEN'], padding = 'post', value = 0, truncating = 'post')
	y_train_categorical = []
	for idx in xrange(y_train.shape[0]):
		y_train_categorical.append( to_categorical(y_train[idx], len(options['CLASSES_2_IX'])))
	return X_train, np.array(y_train_categorical)

def create_model(options):
	post = Input(shape=(options['MAX_LEN'],))
	if options['USE_EMBEDDING']:
		embedding = Embedding(output_dim=options['EMBEDDING_DIM'], weights = [options['EMBEDDING_MATRIX']] ,input_dim = len(options['VOCAB']) + 2, mask_zero=True ) 
	else:
		embedding = Embedding(output_dim=options['EMBEDDING_DIM'] ,input_dim = len(options['VOCAB']) + 2, mask_zero=True ) 
	embed_post = embedding(post)
	processed_post = Bidirectional(GRU(options['HIDDEN_DIM'], return_sequences = True))(embed_post)
	output = Dense(len(options['CLASSES_2_IX']), activation='softmax')(processed_post)
	model = Model(inputs = [post], outputs = output)
	adam = Adam(clipnorm=1.)
	model.compile(optimizer=adam, loss='categorical_crossentropy',metrics=['accuracy'])
	model.summary()
	return model
	


if __name__ == "__main__":
	options = get_options()
	print 'LOADING DATA ... '
	sys.stdout.flush()
	X,y = get_data(options)
	print 'DATA LOADED ...'
	sys.stdout.flush()
	bar = Progbar(len(X))	
	for idx in xrange(len(X)):
		train_X, train_y = get_train_val(X,y,idx,options)
		BASE_DIR = '/scratch/cse/btech/cs1130773/BTP/SupervisedData/LSTM_MODELS/'
		MODEL_DIR = BASE_DIR + 'MODEL_' + str(idx) + '/'
		if not os.path.exists(MODEL_DIR):
			os.makedirs(MODEL_DIR)
		file_name = MODEL_DIR + 'weights.{epoch:02d}_{val_acc:.2f}.hdf5'
		check_point = ModelCheckpoint(file_name, monitor='val_acc', save_best_only=True, save_weights_only=True, mode='auto')
		model = create_model(options)
		n_epochs = 25
		print 'STARTING MODEL FITTING ...'
		sys.stdout.flush()
		model.fit(train_X, train_y, batch_size = options['BATCH_SIZE'], validation_split = 0.1, epochs = n_epochs, callbacks = [check_point])
		print 'MODEL FITTED FOR IDX', idx
		sys.stdout.flush()
		bar.update(idx+1)
