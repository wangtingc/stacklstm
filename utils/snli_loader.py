from data_loader import *
import json
import os
import misc

class SnliLoader(DataLoader):
    def __init__(self, data_path, count_threshold=0, wvec_path=None, filter_by_wvec=False):
        super(SnliLoader, self).__init__(data_path)
        self.train_path = data_path +'snli_1.0_train.jsonl'
        self.valid_path = data_path +'snli_1.0_dev.jsonl'
        self.test_path = data_path +'snli_1.0_test.jsonl'
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
            train_jsons = self._read_json_data(self.train_path)
            valid_jsons = self._read_json_data(self.valid_path)
            test_jsons = self._read_json_data(self.test_path)
            
            self._build_vocab(train_jsons)
            wvec = misc.load_glove(wvec_path)
            self._build_wemb(wvec, self.filter_by_wvec)
            self.train_data = self._json_to_data(train_jsons)
            self.valid_data = self._json_to_data(valid_jsons)
            self.test_data = self._json_to_data(test_jsons)

            self._save(self.vocab_path, train_data_path, \
                       valid_data_path, test_data_path, False)
            save_pkl(self.data_path + 'emb.pkl', self.w_emb)
        
    

    def _read_json_data(self, file_path):
        data = []
        with open(file_path) as f:
            for line in f:
                sample = json.loads(line)
                if sample['gold_label'] != u'-':
                    data.append(sample)
        return data

    
    def _build_vocab(self, json_data):
        words = []
        for sample in json_data:
            ws0 = sample['sentence1_binary_parse'].split()
            ws1 = sample['sentence2_binary_parse'].split()
            words += [w.lower() for w in ws0 if w not in ['(', ')']]
            words += [w.lower() for w in ws1 if w not in ['(', ')']]

        super(SnliLoader, self)._build_vocab(words, self.vocab_path, self.count_threshold)


    def _json_to_data(self, json_data):
        x0, x1, a0, a1, p0, p1, y = [], [], [], [], [], [], []
        l2i = {u'entailment': 0, u'neutral': 1, u'contradiction': 2}
        for sample in json_data:
            x0i, a0i, p0i = self._proc_single(sample['sentence1_binary_parse'])
            x1i, a1i, p1i = self._proc_single(sample['sentence2_binary_parse'])
            y.append(l2i[sample['gold_label']])
            x0.append(x0i)
            x1.append(x1i)
            a0.append(a0i)
            a1.append(a1i)
            p0.append(p0i)
            p1.append(p1i)
        return x0, x1, a0, a1, p0, p1, y


    def _proc_single(self, binary_parse_tree):
        buf = binary_parse_tree.split()
        x, a, p, q, s = [], [], [], [], [0,]
        for w in buf:
            if w == '(':
                continue
            elif w == ')':
                s.append(w)
                a.append(1)
                p.append((q.pop(), q.pop()))
                q.append(len(s)-1)
                x.append(0)
            else:
                s.append(self.vocab.get(w, 1))
                a.append(0)
                q0 = q[-1] if len(q) >= 1 else 0
                q1 = q[-2] if len(q) >= 2 else 0
                p.append((q0, q1))
                q.append(len(s)-1)
                x.append(self.vocab.get(w, 1))
        return x, a, p




if __name__ == '__main__':
    data_path = '../data/snli/'
    sl = SnliLoader(data_path,
                    count_threshold = 3, 
                    wvec_path = '../data/glove/glove.840B.300d.txt.gz',
                    filter_by_wvec = True)

    s = ' ( A ( B C ) )'
    sl.vocab = {'A': 0, 'B': 1, 'C': 2}

    print sl._proc_single(s)
