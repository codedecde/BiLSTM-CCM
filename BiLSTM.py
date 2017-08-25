import sys
import os
from utils import Progbar

import configurations as cf
import utils as ut
import argparse
import BiLSTM_Model as MF
import pdb


def get_arguments():
    '''
        run the code as
        For training : python BiLSTM.py -train_flag True -model <MODEL>
        For Predictions : python BiLSTM.py -train_flag False -model <MODEL> -inf_mode <crf/ccm>

    '''
    parser = argparse.ArgumentParser(description='BiLSTM + CRF')
    parser.add_argument('-model', action="store", default="no_features", dest="model", type=str)
    parser.add_argument('-train_flag', action="store", default="True", dest="train_flag", type=str)
    parser.add_argument('-inf_mode', action="store", default="crf", dest="inf_mode", type=str)
    parser.add_argument('-use_partial', action="store", default="True", dest="use_partial", type=str)
    parser.add_argument('-thread_ix', action='store', default=0, dest='thread_ix', type=int)
    parser.add_argument('-num_threads', action='store', default=1, dest='num_threads', type=int)
    parser.add_argument('-partial_mode', action='store', default="em", dest='partial_mode', type=str)
    opts = parser.parse_args(sys.argv[1:])
    assert opts.model in set(['no_features', 'features_with_embeddings', 'features_with_lstm'])

    opts.train_flag = True if opts.train_flag == 'True' else False
    opts.use_partial = True if opts.use_partial == 'True' else False

    opts.inf_mode = opts.inf_mode.lower()
    assert opts.inf_mode in set(['crf', 'ccm'])
    return opts


if __name__ == "__main__":
    args = get_arguments()
    options = cf.get_options(args)
    posts = open(options['DATA_PATH']).read().split('\n\n')
    if options['USE_PARTIAL']:
        partially_labeled_posts = open(options['PARTIAL_DATA_PATH']).read().split('\n\n')
        unlabeled_X, unlabeled_y = ut.data_generator(partially_labeled_posts, options)
    else:
        unlabeled_X = None
        unlabeled_y = None

    if options['TRAIN_FLAG']:
        num_threads = options['NUM_THREADS']
        posts_per_thread = len(posts) / num_threads if (len(posts) % num_threads == 0) else (len(posts) / num_threads) + 1
        start_ix = posts_per_thread * options['THREAD_IX']
        end_ix = posts_per_thread + start_ix if posts_per_thread + start_ix < len(posts) else len(posts)
        bar = Progbar(end_ix - start_ix)
        for ix in xrange(start_ix, end_ix):
            print '\nPOST %d of (%d / %d)' % (ix - start_ix, start_ix, end_ix - 1)
            training_data = posts[:ix] + posts[ix + 1:]
            train_X, train_y = ut.data_generator(training_data, options)
            BASE_DIR = options['BASE_DIR']
            MODEL_DIR = BASE_DIR + 'MODEL_' + str(ix) + '/'
            if not os.path.exists(MODEL_DIR):
                os.makedirs(MODEL_DIR)
            file_name = MODEL_DIR + options['SAVE_PREFIX']
            model = MF.BiLSTM_Model(options)
            n_epochs = 10
            _mode = args.partial_mode if hasattr(args, 'partial_mode') else 'em'
            model.fit(X=train_X, y=train_y, val_split=0.9, shuffle=True, n_epochs=n_epochs, save_best=True, save_prefix=file_name, X_unlabeled=unlabeled_X, y_unlabeled=unlabeled_y, mode=_mode)
            bar.update(ix - start_ix + 1)
    else:
        predictions = []
        model = MF.BiLSTM_Model(options)
        for ix in xrange(len(posts)):
            BASE_DIR = options['BASE_DIR']
            MODEL_DIR = BASE_DIR + 'MODEL_' + str(ix) + '/'
            file_name = MODEL_DIR + options['SAVE_PREFIX']
            best_model_file = ut.get_best_model_file(file_name)
            model.load_model(best_model_file)
            test_X, test_y = ut.data_generator([posts[ix]], options)
            _, prediction = model.predict(test_X, mode=args.inf_mode)
            prediction = [options['IX_2_CLASSES'][x] if x in options['IX_2_CLASSES'] else 'O' for x in prediction]
            predictions.append('\n'.join(prediction))
            print 'PREDICTIONS FOR IX: ', ix, 'DONE ...'
        predictions = '\n\n'.join(predictions)
        print predictions
