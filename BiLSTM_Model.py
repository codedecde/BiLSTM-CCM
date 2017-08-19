import CRF as crf
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
import numpy as np
import sys
import copy
from utils import Progbar
import pdb


class BiLSTM_Model(crf.CRF):
    def __init__(self, options):
        self.options = options
        self.set_constants()

        if torch.cuda.is_available():
            self.GPU = True
        else:
            self.GPU = False
        super(BiLSTM_Model, self).__init__(options, self.GPU)
        self.embedding_dim = options['EMBEDDING_DIM']
        self.hidden_dim = options['HIDDEN_DIM']
        self.vocab_size = len(options['VOCAB']) + 2

        self.word_embeds = nn.Embedding(self.vocab_size, self.embedding_dim).type(self.dtype)
        if options['USE_EMBEDDING']:
            self.word_embeds.weight = nn.Parameter(torch.FloatTensor(options['EMBEDDING_MATRIX']).type(self.dtype))

        self.num_features = len(options['FEATURE_VOCAB'])
        if self.options['MODEL_TYPE'] == 'features_with_embeddings':
            self.lstm = nn.LSTM(self.embedding_dim + self.num_features, self.hidden_dim // 2,
                                num_layers=1, bidirectional=True).type(self.dtype)
        else:
            self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim // 2,
                                num_layers=1, bidirectional=True).type(self.dtype)

        if self.options['MODEL_TYPE'] == 'features_with_lstm':
            self.hidden2tag = nn.Linear(self.hidden_dim + self.num_features, self.tagset_size).type(self.dtype)
        else:
            self.hidden2tag = nn.Linear(self.hidden_dim, self.tagset_size).type(self.dtype)

        self.hidden = self.init_hidden()

        self.best_val_acc = None

    def set_constants(self):
        assert hasattr(self, 'options'), 'Options have not been set'
        self.FEATURE_SUPPORT = False if self.options['MODEL_TYPE'] == 'no_features' else True
        self.FILENAME = self.options['DATA_DIR'] + 'ilp_problem_' + str(self.options['THREAD_IX']) + '.lp'
        self.TEMP_FILENAME = self.options['DATA_DIR'] + 'temp_' + str(self.options['THREAD_IX']) + '.out'
        self.GLPK_LOCATION = '/usr/local/bin/glpsol'

    def provides_feature_support(self):
        return self.FEATURE_SUPPORT

    def generate_autograd_variable(self, X):
        if self.GPU:
            return autograd.Variable(torch.cuda.LongTensor(X))
        else:
            return autograd.Variable(torch.LongTensor(X))

    def init_hidden(self):
        return (autograd.Variable(torch.zeros(2, 1, self.hidden_dim // 2).type(self.dtype)),
                autograd.Variable(torch.zeros(2, 1, self.hidden_dim // 2)).type(self.dtype))

    def _get_features(self, sentence, feature_vector=None):
        self.hidden = self.init_hidden()
        if self.options['MODEL_TYPE'] == 'features_with_embeddings':
            assert feature_vector is not None
            embeds = torch.cat([self.word_embeds(sentence), feature_vector], dim=-1).view(len(sentence), 1, -1)
        else:
            embeds = self.word_embeds(sentence).view(len(sentence), 1, -1)
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        lstm_out = lstm_out.view(len(sentence), self.hidden_dim)
        if self.options['MODEL_TYPE'] == 'features_with_lstm':
            assert feature_vector is not None
            lstm_out = torch.cat([lstm_out, feature_vector], dim=-1)
        else:
            lstm_out = lstm_out.view(len(sentence), self.hidden_dim)
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats

    def neg_log_likelihood(self, sentence, feature_vector=None, tags=[]):
        feats = self._get_features(sentence, feature_vector)
        forward_score = self._forward_alg(feats)
        gold_score = self._score_sentence(feats, tags)
        return forward_score - gold_score

    def forward(self, sentence, feature_vector=None, mode='crf', partial_labels=None):
        mode = mode.lower()
        assert mode in set(['crf', 'ccm'])
        feats = self._get_features(sentence, feature_vector)
        if partial_labels is not None:
            assert mode == 'ccm'
        if mode == 'crf':
            score, tag_seq = self._viterbi_decode(feats)
        else:
            score, tag_seq = self._ccm_decode(feats, partial_labels)
        return score, tag_seq

    def get_sentence_feature_vector(self, elem):
        if type(elem) == tuple:
            sentence = self.generate_autograd_variable(elem[0])
            feature_vector = autograd.Variable(torch.Tensor(elem[1]).type(self.dtype))
        else:
            sentence = self.generate_autograd_variable(elem)
            feature_vector = None
        return sentence, feature_vector

    def predict(self, X, mode='crf', partial_labels=None, use_bar=False):
        if type(X) != list:
            X = [X]
        predictions = []
        if use_bar:
            bar = Progbar(len(X))
        else:
            bar = None
        for ix, elem in enumerate(X):
            sentence, feature_vector = self.get_sentence_feature_vector(elem)
            if partial_labels is None:
                _, prediction = self.__call__(sentence, feature_vector, mode)
            else:
                _, prediction = self.__call__(sentence, feature_vector, mode, partial_labels[ix])
                for jx in xrange(len(prediction)):
                    if partial_labels[ix][jx] != -1:
                        assert partial_labels[ix][jx] == prediction[jx]
            predictions.append(prediction)
            if bar is not None:
                bar.update(ix + 1)

        if len(predictions) == 1:
            return predictions[0]
        else:
            return predictions

    def train_epoch(self, X, y):
        optimizer = optim.Adam(self.parameters())
        bar = Progbar(len(X))
        for ix, (elem, tags) in enumerate(zip(X, y)):
            self.zero_grad()
            sentence, feature_vector = self.get_sentence_feature_vector(elem)
            if self.GPU:
                targets = torch.LongTensor(tags).cuda()
            else:
                targets = torch.LongTensor(tags)
            neg_log_likelihood = self.neg_log_likelihood(sentence, feature_vector, targets)
            neg_log_likelihood.backward()
            optimizer.step()
            bar.update(ix + 1)
        print ''
        sys.stdout.flush()

    def save_model(self, X_val, y_val, save_prefix, save_best, epoch):
        val_acc = []
        for elem, tags in zip(X_val, y_val):
            sentence, feature_vector = self.get_sentence_feature_vector(elem)
            _, predictions = self.__call__(sentence, feature_vector, mode='crf')
            val_acc.append(accuracy_score(tags, predictions))
        val_acc = np.array(val_acc)
        mean_val_acc = val_acc.mean()
        if save_best:
            if self.best_val_acc is None or mean_val_acc == max(mean_val_acc, self.best_val_acc):
                self.best_val_acc = mean_val_acc
                save_elem = {'constraint_penalty': self.constraint_penalty, 'state_dict': self.state_dict()} if hasattr(self, 'constraint_penalty') and self.constraint_penalty is not None else {'constraint_penalty': 0., 'state_dict': self.state_dict()}
                torch.save(save_elem, save_prefix + '_on_epoch_{0:d}_val_acc_{1:.3f}.weights'.format(epoch, mean_val_acc))
        else:
            save_elem = {'constraint_penalty': self.constraint_penalty, 'state_dict': self.state_dict()} if hasattr(self, 'constraint_penalty') and self.constraint_penalty is not None else {'constraint_penalty': 0., 'state_dict': self.state_dict()}
            torch.save(save_elem, save_prefix + '_on_epoch_{0:d}_val_acc_{1:.3f}.weights'.format(epoch, mean_val_acc))
        return mean_val_acc

    def load_model(self, filename):
        elem = torch.load(filename)
        self.constraint_penalty = elem['constraint_penalty']
        state_dict = elem['state_dict']
        self.load_state_dict(state_dict)

    def compute_constraint_penalty(self, X, y):
        num_satisfied = 0
        for _y in y:
            num_satisfied += 1 if len(filter(lambda x: x == self.tag_to_ix['attr'], _y)) > 0 else 0
        num_unsatisfied = len(y) - num_satisfied
        # smoothing
        num_satisfied += 0.1
        num_unsatisfied += 0.1
        rho_attr = np.log(num_satisfied) - np.log(num_unsatisfied)
        return rho_attr

    def set_constraint_penalties(self, X, y):
        self.constraint_penalty = {}        
        self.constraint_penalty['AT_LEAST_ONE_ATTR'] = self.compute_constraint_penalty(X, y)
        # print self.constraint_penalty['AT_LEAST_ONE_ATTR']

    def train_with_partial_data(self, X_train, y_train, X_unlabeled, y_unlabeled, mode='codl'):
        mode = mode.lower()
        assert mode in set(['codl', 'em']), "Found unknown mode %s" % (mode)
        NUM_ITERATIONS = 5
        if mode == 'codl':
            # CoDL
            gamma = 0.9
            original_params = copy.deepcopy(self.state_dict())            
            original_rho = self.constraint_penalty['AT_LEAST_ONE_ATTR']
            data_X = X_unlabeled
            for ix in xrange(NUM_ITERATIONS):
                print '\tStarting CoDL Iteration : %d / %d' % (ix + 1, NUM_ITERATIONS)
                print '\t Making %d Predictions ' % (len(X_unlabeled))
                y_predictions = self.predict(X_unlabeled, mode='ccm', partial_labels=y_unlabeled, use_bar=True)
                if type(y_predictions) != list:
                    y_predictions = [y_predictions]
                data_y = y_predictions
                print '\t Training on %d Observations ' % (len(data_X))
                self.set_constraint_penalties(data_X, data_y)
                self.train_epoch(data_X, data_y)
                # Now update the parameters                 
                params = self.state_dict()
                self.constraint_penalty['AT_LEAST_ONE_ATTR'] = (gamma * original_rho) + ((1. - gamma) * self.constraint_penalty['AT_LEAST_ONE_ATTR'])
                for w in params:
                    if w in original_params:
                        params[w] = (gamma * original_params[w]) + ((1. - gamma) * params[w])
                self.load_state_dict(params)

        else:
            # EM
            data_X = X_train + X_unlabeled
            for ix in xrange(NUM_ITERATIONS):
                print '\tStarting EM Iteration : %d / %d' % (ix + 1, NUM_ITERATIONS)
                # 1.1 E Step : Make predictions
                print '\t Making %d Predictions ' % (len(X_unlabeled))
                y_predictions = self.predict(X_unlabeled, mode='ccm', partial_labels=y_unlabeled, use_bar=True)
                if type(y_predictions) != list:
                    y_predictions = [y_predictions]
                data_y = y_train + y_predictions
                print '\t Training on %d Observations ' % (len(data_X))
                # 1.2 M Step : Maximize log likelihood
                # 1.2.1 Update the constraints
                self.set_constraint_penalties(data_X, data_y)
                # 1.2.2 Update the parameters
                self.train_epoch(data_X, data_y)

    def fit(self, X, y, save_prefix, val_split=0.9, shuffle=False, n_epochs=500, save_best=True, X_unlabeled=None, y_unlabeled=None, mode='em'):
        self.set_constraint_penalties(X, y)
        if shuffle:
            X, y = self.shuffle_data(X, y)
        n_train = int(val_split * len(X))

        X_train = X[:n_train]
        y_train = y[:n_train]

        X_val = X[n_train:]
        y_val = y[n_train:]
        for epoch in xrange(n_epochs):
            self.train_epoch(X_train, y_train)
            if X_unlabeled is not None:
                self.train_with_partial_data(X_train, y_train, X_unlabeled, y_unlabeled, mode)
            val_acc = self.save_model(X_val, y_val, save_prefix, save_best, epoch)
            print 'EPOCH: ', epoch, 'DONE WITH VALIDATION ACCURACY: ', val_acc
            sys.stdout.flush()
