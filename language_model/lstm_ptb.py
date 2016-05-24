import sys
sys.path.append('../')
import os
import numpy as np
from datetime import datetime
import time

from utils import pt_dec_loader
from utils import batch_iterator
from utils import misc
from models.lstm_lm import LSTMLM

def init_params():
    params = {}
    params['dataset'] = 'ptb'
    params['data_path'] = '../data/ptb/'
    params['dim_emb'] = 300
    params['num_units'] = 512
    params['batch_size'] = 32
    params['num_epochs'] = 100
    params['valid_period'] = 1
    params['test_period'] = 1
    params['exp_time'] = str(datetime.now().strftime('%Y%m%d-%H%M'))
    params['emb_dropout'] = 0.0
    params['lr'] = 5e-4
    
    # paths
    params['save_dir'] = '../results/' + params['exp_time'] + '/'
    params['save_weights_path'] = params['save_dir'] + 'weights.pkl'
    params['load_weights_path'] = None
    params['log_path'] = params['save_dir'] + 'log.txt'
    params['details_log_path'] = params['save_dir'] + 'details.txt'
    return params


def train(params):
    data_loader = pt_dec_loader.PtDecLoader(params['data_path'])

    params['num_samples_train'] = len(data_loader.train_data)
    params['num_samples_valid'] = len(data_loader.valid_data)
    params['num_samples_test'] = len(data_loader.test_data)
    params['dim_emb'] = data_loader.dim_emb
    params['dict_size'] = data_loader.dict_size

    it_train = batch_iterator.BatchIterator(params['num_samples_train'], params['batch_size'],\
                                            [data_loader.train_data], False)
    it_valid = batch_iterator.BatchIterator(params['num_samples_valid'], params['batch_size'],\
                                            [data_loader.valid_data], False)
    it_test = batch_iterator.BatchIterator(params['num_samples_test'], params['batch_size'],\
                                           [data_loader.test_data], False)

    model = LSTMLM(num_units=params['num_units'],
                         dict_size=data_loader.dict_size,
                         dim_emb=data_loader.dim_emb,
                         w_emb=data_loader.w_emb,
                         emb_dropout=params['emb_dropout'],
                         lr=params['lr'],
                         )
    

    f_train = model.get_f_train()
    f_test = model.get_f_test()

    if not os.path.exists(params['save_dir']):
        os.mkdir(params['save_dir'])

    config_info = misc.configuration2str(params)
    weights_info = misc.weightsinfo2str(model.get_params())

    df = open(params['details_log_path'], 'w')
    lf = open(params['log_path'], 'w')

    df.write(config_info + '\n')
    lf.write(config_info + '\n')
    print(config_info)
    df.write(weights_info)
    lf.write(weights_info)
    print(weights_info)
    
    num_batches_train = params['num_samples_train'] / params['batch_size']
    num_batches_valid = params['num_samples_valid'] / params['batch_size']
    num_batches_test = params['num_samples_test'] / params['batch_size']
    
    cur = time.time()
    for epoch in range(params['num_epochs']):
        epoch_info = ' [*] epoch %d ' % epoch
        lf.write(epoch_info + '\n')
        df.write(epoch_info + '\n')
        print(epoch_info)
    
        out_all, llh_all, nw_all = [], [], []
        for batch in range(num_batches_train):
            x, = it_train.next()
            x, m = misc.prepare_data(x)
            out = f_train(x, m)
            out_all.append(out)
            llh_all.append(out[0] * params['batch_size'])
            nw_all.append(np.sum(m))
            df.write('\t[-] train: ' + str(out) + '')
        ppl = np.exp(np.sum(llh_all)/ np.sum(nw_all))
        train_info = '\t[-] train: ' + str(np.mean(out_all, axis=0)) + ' ' + str(np.sum(llh_all)) + ' ' + str(np.sum(nw_all)) + ' '+ str(ppl)
        print('\t[-] time:' + str(time.time() - cur))
        df.write(train_info + '\n')
        lf.write(train_info + '\n')
        print(train_info)
     
        out_all, llh_all, nw_all = [], [], []
        for batch in range(num_batches_valid):
            x, = it_valid.next()
            x, m, = misc.prepare_data(x)
            out = f_test(x, m)
            out_all.append(out)
            llh_all.append(out[0] * params['batch_size'])
            nw_all.append(np.sum(m))
            df.write('\t[-] valid: ' + str(out) + '')
        ppl = np.exp(np.sum(llh_all)/ np.sum(nw_all))
        valid_info = '\t[-] valid: ' + str(np.mean(out_all, axis=0)) + ' ' + str(np.sum(llh_all)) + ' ' + str(np.sum(nw_all)) + ' '+ str(ppl)
        print('\t[-] time:' + str(time.time() - cur))
        df.write(valid_info + '\n')
        lf.write(valid_info + '\n')
        print(valid_info)

        out_all, llh_all, nw_all = [], [], []
        for batch in range(num_batches_test):
            x, = it_test.next()
            x, m, = misc.prepare_data(x)
            out = f_test(x, m,)
            out_all.append(out)
            llh_all.append(out[0] * params['batch_size'])
            nw_all.append(np.sum(m))
            df.write('\t[-] test: ' + str(out) + '')
        ppl = np.exp(np.sum(llh_all)/ np.sum(nw_all))
        test_info = '\t[-] test: ' + str(np.mean(out_all, axis=0)) + ' ' + str(np.sum(llh_all)) + ' ' + str(np.sum(nw_all)) + ' '+ str(ppl)
        print('\t[-] time:' + str(time.time() - cur))
        df.write(test_info + '\n')
        lf.write(test_info + '\n')
        print(test_info)
        
        # todo
        # model.save_weights(params['save_weights_path'])


if __name__ == '__main__':
    params = init_params()
    train(params)
