import os
import copy

from nltk.parse import stanford
from nltk.tree import Tree
from misc import *
from data_loader import *
from collections import Counter

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

class PtEncLoader(DataLoader):
    def __init__(self, data_path, parser_path, models_path, model_path):
        # init the data path
        super(PtLoader, self).__init__(data_path)
        self.parser = self._init_parser(parser_path, models_path, model_path)
        self.load_data()
       

    def load_data(self):
        train_data_path = self.data_path + 'train.data.pkl'
        valid_data_path = self.data_path + 'valid.data.pkl'
        test_data_path = self.data_path + 'test.data.pkl'

        if os.path.exists(train_data_path):
            self._load(self.vocab_path, train_data_path, \
                       valid_data_path, test_data_path, False)
        else:
            self._load_btrees()
            self.train_data = self._proc(self.train_btrees)
            self.valid_data = self._proc(self.valid_btrees)
            self.test_data = self._proc(self.test_btrees)
            
            self._save(self.vocab_path, train_data_path, \
                       valid_data_path, test_data_path, False)

    
    # if no data, load binarized trees
    def _load_btrees(self):
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
            self._load_tree()
            self._build_vocab(self.train_trees, self.vocab_path)
            self.train_btrees = self._tree_to_btree(self.train_trees, self.vocab)
            self.valid_btrees = self._tree_to_btree(self.valid_trees, self.vocab)
            self.test_btrees = self._tree_to_btree(self.test_trees, self.vocab)

            save_pkl(train_btree_path, self.train_btrees)
            save_pkl(valid_btree_path, self.valid_btrees)
            save_pkl(test_btree_path, self.test_btrees)
            save_pkl(self.vocab_path, self.vocab)

    
    # if no btree, load tree
    def _load_tree(self):
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

        super(PtLoader, self)._build_vocab(words, vocab_path)


    # reconstruct a tree with word index
    def _replace_by_widx(self, tree, vocab):
        if isinstance(tree, Tree):
            return Tree(tree.label(), [self._replace_by_widx(s, vocab) for s in tree])
        else:
            return vocab.get(tree.lower(), 1)


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
        x, a, p = [], [], []
        for tree in trees:
            xi, ai, pi = self._proc_single(tree)
            x.append(xi)
            a.append(ai)
            p.append(pi)
        return x, a, p


    def _proc_single(self, tree):
        """
        this function is used to preproc the tree data
        @ param: single tree for one sentence
        @ rtype: x, a, p
        """
        x, a, p, s, q = [], [], [], [0,], [] # s[0] is left for init_state
        self._traversal(tree, x, a, s, q, p)
        #x = [ -1 if isinstance(node, Tree) else node for node in s ]
        #a = [ 1 if isinstance(node, Tree) else 0 for node in s ]
        return x, a, p


    def _traversal(self, tree, x, a, s, q, p):
        if not isinstance(tree, Tree):
            x.append(tree)
            a.append(0)
            q0 = q[-1] if len(q) >= 1 else 0
            q1 = q[-2] if len(q) >= 2 else 0
            p.append((q0, q1))
            s.append(tree)
            q.append(len(s)-1)
            return

        for child in tree:
            self._traversal(child, x, a, s, q, p)
        
        x.append(0)
        a.append(1)
        s.append(tree)
        p.append((q.pop(), q.pop()))
        q.append(len(s)-1)

       
if __name__ == '__main__':
    data_path = '../data/ptb_enc/'
    parser_path = '/workspace/software/nlp-stanford/parser/stanford-parser-full-2015-04-20/stanford-parser.jar'
    models_path = '/workspace/software/nlp-stanford/parser/stanford-parser-full-2015-04-20/stanford-parser-3.5.2-models.jar'
    model_path = '/workspace/software/nlp-stanford/parser/stanford-parser-full-2015-04-20/englishPCFG.ser.gz'
    loader = PtLoader(data_path, parser_path, models_path, model_path)
    
    s = 'this is my house \n I can\'t believe how bad he is'
    data = loader._parse(s)
    loader._build_vocab(data, '../data/ptb/vocab.pkl')
    print data[1]
    data_i = loader._convert_to_binary_tree(data[1])
    print data_i
    data_i = loader._replace_by_widx(data_i, loader.vocab)
    print data_i

    ####
    t = [Tree('0', ['A', Tree('1', ['B', 'C'])])]
    print loader._proc(t)
