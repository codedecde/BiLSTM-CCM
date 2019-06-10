import cPickle as cp
import numpy as np


def get_options(args):
    options = {}
    options['TRAIN_FLAG'] = args.train_flag if hasattr(args, 'train_flag') else False
    options['SENTENCE_MARKERS'] = args.sentence_markers if hasattr(args, 'sentence_markers') else False
    options['EMBEDDING_DIM'] = 200
    options['HIDDEN_DIM'] = 512
    options['CLASSES_2_IX'] = {'O': 0, 'type': 1, 'attr': 2, 'location': 3}
    options['IX_2_CLASSES'] = {options['CLASSES_2_IX'][w]: w for w in options['CLASSES_2_IX']}
    # DATA_DIR = '/scratch/cse/btech/cs1130773/BTP/SupervisedData/'
    DATA_DIR = 'Data/'
    VOCAB_PATH = DATA_DIR + 'vocab_btp.pkl'
    options['DATA_DIR'] = DATA_DIR
    options['VOCAB'] = cp.load(open(VOCAB_PATH))

    FEAT_VOCAB_PATH = DATA_DIR + 'features_2_ix.pkl'
    options['FEATURE_VOCAB'] = cp.load(open(FEAT_VOCAB_PATH))

    options['USE_EMBEDDING'] = True
    if options['USE_EMBEDDING']:
        EMBED_PATH = DATA_DIR + 'embedding_matrix_btp.npy'
        options['EMBEDDING_MATRIX'] = np.load(file=open(EMBED_PATH))

    options['DATA_PATH'] = DATA_DIR + 'Data_136_with_feats.txt'

    # Information associated with using partially labeled data
    options['USE_PARTIAL'] = args.use_partial if hasattr(args, 'use_partial') else False
    options['PARTIAL_DATA_PATH'] = options['DATA_DIR'] + 'partially_labeled_data_with_going_features.txt'

    # Threading information
    options['THREAD_IX'] = args.thread_ix if hasattr(args, 'thread_ix') else 0
    options['NUM_THREADS'] = args.num_threads if hasattr(args, 'num_threads') else 1

    assert options['THREAD_IX'] < options['NUM_THREADS'], "Thread Index cannot be more than number of threads"

    # Now information associated with model storage ...
    # Defaults
    options['BASE_DIR'] = 'Models/'
    options['SAVE_PREFIX'] = 'model_using_partial_data' if options['USE_PARTIAL'] else 'model'
    options['USE_FEATURES'] = False
    options['MODEL_TYPE'] = 'no_features'
    # Update information from args
    if hasattr(args, 'model'):
        options['MODEL_TYPE'] = args.model
        if args.model in set(['features_with_embeddings', 'features_with_lstm']):
            options['USE_FEATURES'] = True
            if options['MODEL_TYPE'] == 'features_with_lstm':
                options['SAVE_PREFIX'] = 'features_with_lstm_' + options['SAVE_PREFIX']
            else:
                options['SAVE_PREFIX'] = 'features_with_embeddings_' + options['SAVE_PREFIX']

    options["mode"] = args.inf_mode if hasattr(args, "inf_mode") else "ccm"

    return options
