import os
import copy

from nltk.parse import stanford
from nltk.tree import Tree
from data_loader import *
from collections import Counter
import misc


# this class is used for constructing the dataset for stacklstm,
# in other words, the data will be parsed into binary constituent 
# parse tree.
# 
# prerequisite:
#   1. the raw data is raw text data and one line for each sample
# 
# methods:
#   1. _constituency_parse: convert the raw text into parse tree format
#   2. _convert_to_binary_tree: convert the original parse tree into
#       binary unlabeled parse tree

class PtDecLoader(DataLoader):
    def __init__(self, data_path, 
                 count_threshold=0,
                 wvec_path=None,
                 filter_by_wvec=False,
                 parser_path='/workspace/software/nlp-stanford/parser/stanford-parser-full-2015-04-20/stanford-parser.jar',
                 models_path='/workspace/software/nlp-stanford/parser/stanford-parser-full-2015-04-20/stanford-parser-3.5.2-models.jar',
                 model_path='/workspace/software/nlp-stanford/parser/stanford-parser-full-2015-04-20/englishPCFG.ser.gz',
                 ):
        # init the data path
        super(PtDecLoader, self).__init__(data_path)
        self.parser = self._init_parser(parser_path, models_path, model_path)
        self.count_threshold = count_threshold
        self.filter_by_wvec = filter_by_wvec
        self.load_data(wvec_path)
       

    def load_data(self, wvec_path):
        train_data_path = self.data_path + 'train.data.pkl'
        valid_data_path = self.data_path + 'valid.data.pkl'
        test_data_path = self.data_path + 'test.data.pkl'

        if os.path.exists(train_data_path):
            self._load(self.vocab_path, train_data_path, \
                       valid_data_path, test_data_path, False)
            self.w_emb = load_pkl(self.data_path + 'emb.pkl')
            self.dict_size, self.dim_emb = self.w_emb.shape
        else:
            self._load_strees(wvec_path)
            self.train_data = self._proc(self.train_strees)
            self.valid_data = self._proc(self.valid_strees)
            self.test_data = self._proc(self.test_strees)
            
            self._save(self.vocab_path, train_data_path, \
                       valid_data_path, test_data_path, False)
 
    # if no data, load simplified trees, i.e. no node with single offspring 
    # and not binarized
    def _load_strees(self, wvec_path):
        train_stree_path = self.data_path + 'train.stree.pkl'
        valid_stree_path = self.data_path + 'valid.stree.pkl'
        test_stree_path = self.data_path + 'test.stree.pkl'

        print(' [*] loading simplified parse trees')
        if os.path.exists(train_stree_path):
            self.train_strees = load_pkl(train_stree_path)
            self.valid_strees = load_pkl(valid_stree_path)
            self.test_strees = load_pkl(test_stree_path)
            self.vocab = load_pkl(self.vocab_path)
        else:
            self._load_trees()
            self._build_vocab(self.train_trees, self.vocab_path)
            if os.path.exists(self.data_path + 'emb.pkl'):
                self.w_emb = load_pkl(self.data_path + 'emb.pkl')
            else:
                wvec = misc.load_glove(wvec_path)
                self._build_wemb(wvec, self.filter_by_wvec)
            self.train_strees = self._tree_to_stree(self.train_trees, self.vocab)
            self.valid_strees = self._tree_to_stree(self.valid_trees, self.vocab)
            self.test_strees = self._tree_to_stree(self.test_trees, self.vocab)

            save_pkl(train_stree_path, self.train_strees)
            save_pkl(valid_stree_path, self.valid_strees)
            save_pkl(test_stree_path, self.test_strees)
            save_pkl(self.vocab_path, self.vocab)
            save_pkl(self.data_path + 'emb.pkl', self.w_emb)

    # if no data, load binarized trees
    def _load_btrees(self, wvec_path):
        train_btree_path = self.data_path + 'train.btree.pkl'
        valid_btree_path = self.data_path + 'valid.btree.pkl'
        test_btree_path = self.data_path + 'test.btree.pkl'

        print(' [*] loading binarized parse trees')
        if os.path.exists(train_btree_path):
            self.train_btrees = load_pkl(train_btree_path)
            self.valid_btrees = load_pkl(valid_btree_path)
            self.test_btrees = load_pkl(test_btree_path)
            self.vocab = load_pkl(self.vocab_path)
        else:
            self._load_trees()
            self._build_vocab(self.train_trees, self.vocab_path)
            if os.path.exists(self.data_path + 'emb.pkl'):
                self.w_emb = load_pkl(self.data_path + 'emb.pkl')
            else:
                wvec = misc.load_glove(wvec_path)
                self._build_wemb(wvec, self.filter_by_wvec)
            self.train_btrees = self._tree_to_btree(self.train_trees, self.vocab)
            self.valid_btrees = self._tree_to_btree(self.valid_trees, self.vocab)
            self.test_btrees = self._tree_to_btree(self.test_trees, self.vocab)

            save_pkl(train_btree_path, self.train_btrees)
            save_pkl(valid_btree_path, self.valid_btrees)
            save_pkl(test_btree_path, self.test_btrees)
            save_pkl(self.vocab_path, self.vocab)
            save_pkl(self.data_path + 'emb.pkl', self.w_emb)

    # if no btree, load tree
    def _load_trees(self):
        train_tree_path = self.data_path + 'train.tree.pkl'
        valid_tree_path = self.data_path + 'valid.tree.pkl'
        test_tree_path = self.data_path + 'test.tree.pkl'
        
        print(' [*] loading parse trees')
        if not os.path.exists(train_tree_path):
            self.train_trees = self._parse(self._read_text(self.train_path))
            save_pkl(train_tree_path, self.train_trees)
        else:
            self.train_trees = load_pkl(train_tree_path)

        if not os.path.exists(valid_tree_path):
            self.valid_trees = self._parse(self._read_text(self.valid_path))
            save_pkl(valid_tree_path, self.valid_trees)
        else:
            self.valid_trees = load_pkl(valid_tree_path)

        if not os.path.exists(test_tree_path):
            self.test_trees = self._parse(self._read_text(self.test_path))
            save_pkl(test_tree_path, test_trees)
        else:
            self.test_trees = load_pkl(test_tree_path)

    def _init_parser(self, parser_path, models_path, model_path):
        os.environ['STANFORD_PARSER'] = parser_path
        os.environ['STANFORD_MODELS'] = models_path

        parser = stanford.StanfordParser(model_path=model_path)
        return parser

    def _parse(self, raw_text):
        # Each sentence will be automatically tokenized and tagged by the parser
        trees = self.parser.raw_parse_sents(raw_text.split(EOS_TOKEN))
        trees = [s.next()[0] for s in trees]
        return trees

    # exatract the words and lower them and build the vocabulary
    def _build_vocab(self, trees, vocab_path):
        words = []
        for tree in trees:
            words += [w.lower() for w in tree.leaves()]

        super(PtDecLoader, self)._build_vocab(words, vocab_path)

    # reconstruct a tree with word index
    def _replace_by_widx(self, tree, vocab):
        if isinstance(tree, Tree):
            return Tree(tree.label(), [self._replace_by_widx(s, vocab) for s in tree])
        else:
            return vocab.get(tree.lower(), 1)
    
    # remove the node with only one offspring
    def _simplify_tree(self, tree):
        if not isinstance(tree, Tree):
            return copy.copy(tree)
        
        if len(tree) == 1:
            tree = tree[0]
            while len(tree) == 1 and isinstance(tree, Tree):
                tree = tree[0]
            return self._simplify_tree(tree)

        new_tree = Tree(tree.label(), [self._simplify_tree(i) for i in tree])
        return new_tree
        
    def _tree_to_stree(self, trees, vocab):
        data = []
        for tree in trees:
            data_i = self._simplify_tree(tree)
            data_i = self._replace_by_widx(data_i, vocab)
            data.append(data_i)
        return data

    # reconstruct an binary constituency parse tree
    def _convert_to_binary_tree(self, tree):
        if not isinstance(tree, Tree):
            return copy.copy(tree)

        if len(tree) == 1:
            tree = tree[0]
            while len(tree) == 1 and isinstance(tree, Tree):
                tree = tree[0]
            return self._convert_to_binary_tree(tree)

        if len(tree) == 2:
            left = self._convert_to_binary_tree(tree[0])
            right = self._convert_to_binary_tree(tree[1])
            new_tree = Tree(tree.label(), [left, right])
            return new_tree

        right = self._convert_to_binary_tree(tree[len(tree)-1])
        for idx in range(len(tree) - 1)[::-1]:
            left = self._convert_to_binary_tree(tree[idx])
            right = Tree('Newed_'+tree.label(), [left, right])
        
        return right
        
    def _tree_to_btree(self, trees, vocab):
        data = []
        for tree in trees:
            data_i = self._convert_to_binary_tree(tree)
            data_i = self._replace_by_widx(data_i, vocab)
            data.append(data_i)
        return data

    def _proc(self, trees):
        x, p, a, y = [], [], [], []
        for tree in trees:
            xi, pi, ai, yi = self._proc_single(tree)
            x.append(xi)
            p.append(pi)
            a.append(ai)
            y.append(yi)
        return x, p, a, y

    def _proc_single(self, tree):
        """
        this function is used to preproc the tree data
        @ param: single tree for one sentence
        @ rtype: x, p, a, y
        
        w: words, initilized with [0] (<SOS>)
        x: inputs at each step, 0 for tree
        y: output at each step, 0 for tree
        p: if prediction at this step
        a: father of each step
        s: stack, save all the internal state
        q: queue for extracting a
        """

        w, x, y, p, a, s, q = [0], [], [], [], [], [], [-1]
        self._traversal(tree, w, x, y, p, a, s, q)

        if isinstance(tree, Tree):
            return x[1:], p[1:], a[1:], y[1:]
        else:
            return x, [1], [0], [0]

    def _traversal(self, tree, w, x, y, p, a, s, q):
        if not isinstance(tree, Tree):
            x.append(w[-1])
            w.append(tree)
            y.append(tree)
            p.append(1)
            a.append(q[-1])
            return

        x.append(0)
        y.append(0)
        p.append(0)
        a.append(q[-1])
        s.append(tree)
        q.append(len(s)-1)

        for i in tree:
            self._traversal(i, w, x, y, p, a, s, q)

        q.pop()

if __name__ == '__main__':
    data_path = '../data/ptb_dec/'
    parser_path = '/workspace/software/nlp-stanford/parser/stanford-parser-full-2015-04-20/stanford-parser.jar'
    models_path = '/workspace/software/nlp-stanford/parser/stanford-parser-full-2015-04-20/stanford-parser-3.5.2-models.jar'
    model_path = '/workspace/software/nlp-stanford/parser/stanford-parser-full-2015-04-20/englishPCFG.ser.gz'
    loader = PtDecLoader(data_path,
                        count_threshold=3,
                        wvec_path = '../data/glove/glove.840B.300d.txt.gz',
                        filter_by_wvec=False,
                        parser_path=parser_path,
                        models_path=models_path, 
                        model_path=model_path)

    s = 'this is my house \n I can\'t believe how bad he is'
    data = loader._parse(s)
    loader._build_vocab(data, '../data/ptb/vocab.pkl')
    print data[1]
    data_i = loader._convert_to_binary_tree(data[1])
    print data_i
    data_i = loader._replace_by_widx(data_i, loader.vocab)
    print data_i

    ####
    t = [Tree('0', ['A', Tree('1', [Tree(2, ['B', 'C']), 'D'])])]
    print loader._proc(t)
    t = [Tree('0', ['A', Tree('1', [Tree(2, ['B', 'C']), 'D', Tree(2, ['E', 'F'])])])]
    print loader._proc(t)
    t = '0'
    print loader._proc(t)
