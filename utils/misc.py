import pprint
import cPickle
import numpy as np
import gzip
import theano

pp = pprint.PrettyPrinter()

def save_pkl(path, obj):
    with open(path, 'w') as f:
        cPickle.dump(obj, f)
        print(" [*] save %s" % path)

def load_pkl(path):
    with open(path) as f:
        obj = cPickle.load(f)
    print(" [*] load %s" % path)
    return obj

def save_npy(path, obj):
    np.save(path, obj)
    print(" [*] save %s" % path)

def load_npy(path):
    obj = np.load(path)
    print(" [*] load %s" % path)
    return obj

def prepare_data(s):
    max_len = 0
    for i in s:
        max_len = max(max_len, len(i))
    max_len += 1
    x = np.zeros([len(s), max_len], dtype='int64')
    m = np.zeros([len(s), max_len], dtype='int64')
    for idx, i in enumerate(s):
        x[idx, :len(i)] = i
        m[idx, :len(i)+1] = 1

    x = x[:, :500]
    m = m[:, :500]
    return x, m

def prepare_snli(_data):
    x, a, p = _data
    max_len = 0
    for i in x:
        max_len = max(max_len, len(i))
    
    batch_size = len(x)
    xr = np.zeros([batch_size, max_len], dtype='int64')
    mr = np.zeros([batch_size, max_len], dtype=theano.config.floatX)
    ar = np.zeros([batch_size, max_len], dtype='int64')
    pr = np.zeros([batch_size, max_len, 2], dtype='int64')

    for i in range(batch_size):
        xr[i, :len(x[i])] = x[i]
        mr[i, :len(x[i])] = 1
        ar[i, :len(x[i])] = a[i]
        pr[i, :len(x[i])] = p[i]

    return xr, mr, ar, pr


def concat_sentence(s):
    x = []
    for i in s:
        x_i = []
        for j in i:
            x_i.extend(j)
        x.append(x_i)
    return x

def filt_words(s, dict_size):
    x = []
    for i in s:
        x_i = [j if j < dict_size else 1 for j in i]
        x.append(x_i)
    return x

def configuration2str(params):
    s = '[*] printing experiment configuration' + '\n'
    for k in params:
        s += '\t[-] ' + k + ': ' + str(params[k]) + '\n'
    s += '\n'
    return s

def load_glove(filename):
    glove = {}
    f = gzip.open(filename, 'r')
    for l in f:
        k = l.split()[0]
        v = [float(i) for i in l.split()[1:]]
        glove[k] = v
    return glove

def test_prepare_data():
    s = [[1,2], [2]]
    x, m = prepare_data(s)
    print x
    print m

def test_filt_words():
    s = [[3,0]]
    x = filt_words(s, 2)
    print x


if __name__ == '__main__':
    test_prepare_data()
    test_filt_words()


