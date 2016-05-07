# base interface for various dataset loaders
# the class has no idea of what kind of data it is,
# nor the method used for preprocess.

# prerequisite:
#   1. assume the raw data is saved in one directory
#   2. assume the raw data consists of (train, valid, test) files,
#      and their corresponding name is 'train.txt'
#
# methods:
#   1. _load: load the preprocessed data, their name should be path+'.npy'
#   2. __init__: init the path varibles

import os
from misc import *
from collections import Counter
import numpy as np

EOS_TOKEN = "_eos_"

class DataLoader(object):
    def __init__(self, data_path):
        # data_path : data directory path
        # [train, valid, test]_path is the path for raw data
        # vocab_path is the unique path for vocabulary
        self.data_path = data_path
        self.train_path = os.path.join(data_path, 'train.txt')
        self.valid_path = os.path.join(data_path, 'valid.txt')
        self.test_path = os.path.join(data_path, 'test.txt')
        self.vocab_path = os.path.join(data_path, 'vocab.pkl')
        
    
    # read raw data
    def _read_text(self, file_path):
        with open(file_path) as f:
            return f.read().replace('\n', EOS_TOKEN)
    

    # build vocabulary, leave index 0 for <EOS>, 1 for <UNK>
    def _build_vocab(self, words, vocab_path, count_threshold = 0):
        counter = Counter(words)

        count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
        count_pairs = [(k, v) for k, v in count_pairs if v >= count_threshold]
        words, _ = list(zip(*count_pairs))

        self.vocab = dict(zip(words, range(2, len(words)+2)))
        self.dict_size = len(words) + 2


    # Given wvec and vocabulary, build the vocab
    # if filt_by_wvec == True, the words not in wvec will be removed.
    def _build_wemb(self, wvec, filt_by_wvec=False):
        vocab = self.vocab
        key_wvec = set(wvec.keys())

        if filt_by_wvec:
            count_pairs = sorted(vocab.items(), key=lambda x: (-x[1], x[0]))
            words, _ = list(zip(*count_pairs))
            words = [w for w in words if w in key_wvec]
            self.vocab = dict(zip(words, range(2, len(words)+2)))
            self.dict_size = len(words) + 2
        
        self.dim_emb = len(wvec.values()[0])
        w_emb = np.random.rand(self.dict_size, self.dim_emb) * 0.001
        
        for w, idx in self.vocab.items():
            if w in key_wvec:
                w_emb[idx] = wvec[w]

        self.w_emb = w_emb



    # load the preprocessed data
    def _load(self, vocab_path, train_path, valid_path, test_path, is_npy_format=False):
        #self.vocab = load_pkl(vocab_path)
        load_fn = load_npy if is_npy_format else load_pkl
        self.train_data = load_fn(train_path)
        self.valid_data = load_fn(valid_path)
        self.test_data = load_fn(test_path)


    # load the preprocessed data
    def _save(self, vocab_path, train_path, valid_path, test_path, is_npy_format=False):
        save_pkl(vocab_path, self.vocab)
        save_fn = save_npy if is_npy_format else save_pkl
        save_fn(train_path, self.train_data)
        save_fn(valid_path, self.valid_data)
        save_fn(test_path, self.test_data)
