import sys
sys.path.append('../')
import os
import numpy as np
from datetime import datetime
import time

from utils import pt_dec_loader
from utils import batch_iterator
from utils import misc
from models.stack_lstm_lm import StackLSTMLM

def init_params():
    params = {}
    params['dataset'] = 'ptb'
    params['data_path'] = '../data/ptb_dec/'
    params['dim_emb'] = 300
    params['num_units'] = 512
    params['batch_size'] = 32
    params['num_epochs'] = 100
    params['valid_period'] = 1
    params['test_period'] = 1
    params['exp_time'] = str(datetime.now().strftime('%Y%m%d-%H%M'))
    params['emb_dropout'] = 0.2
    params['lr'] = 2e-3
    
    # paths
    params['save_dir'] = '../results/' + params['exp_time'] + '/'
    params['save_weights_path'] = params['save_dir'] + 'weights.pkl'
    params['load_weights_path'] = None
    params['log_path'] = params['save_dir'] + 'log.txt'
    params['details_log_path'] = params['save_dir'] + 'details.txt'
    return params


def train(params):
    data_loader = pt_dec_loader.PtDecLoader(params['data_path'])

    params['num_samples_train'] = len(data_loader.train_data[0])
    params['num_samples_valid'] = len(data_loader.valid_data[0])
    params['num_samples_test'] = len(data_loader.test_data[0])
    params['dim_emb'] = data_loader.dim_emb
    params['dict_size'] = data_loader.dict_size

    it_train = batch_iterator.BatchIterator(params['num_samples_train'], params['batch_size'],\
                                            data_loader.train_data, False)
    it_valid = batch_iterator.BatchIterator(params['num_samples_valid'], params['batch_size'],\
                                            data_loader.valid_data, False)
    it_test = batch_iterator.BatchIterator(params['num_samples_test'], params['batch_size'],\
                                           data_loader.test_data, False)

    model = StackLSTMLM(num_units=params['num_units'],
                         dict_size=data_loader.dict_size,
                         dim_emb=data_loader.dim_emb,
                         w_emb=data_loader.w_emb,
                         emb_dropout=params['emb_dropout'],
                         lr=params['lr'],
                         )

    f_train = model.get_f_train()
    # debug
    f_test = f_train

    if not os.path.exists(params['save_dir']):
        os.mkdir(params['save_dir'])
    config_info = misc.configuration2str(params)

    df = open(params['details_log_path'], 'w')
    lf = open(params['log_path'], 'w')
    df.write(config_info + '\n')
    lf.write(config_info + '\n')
    print(config_info)
    
    num_batches_train = params['num_samples_train'] / params['batch_size']
    num_batches_valid = params['num_samples_valid'] / params['batch_size']
    num_batches_test = params['num_samples_test'] / params['batch_size']

    for epoch in range(params['num_epochs']):
        epoch_info = ' [*] epoch %d \n' % epoch
        lf.write(epoch_info)
        df.write(epoch_info)
        print(epoch_info)
    
        outs = []
        for batch in range(num_batches_train):
            x, p, a = it_train.next()
            x, m, p, a = misc.prepare_ptb([x, p, a])
            out = f_train(x, m, p, a)
            outs.append(out)
            #print('\t[-] train: ' + str(out) + '\n')
            df.write('\t[-] train: ' + str(out) + '\n')
        lf.write('\t[-] train: ' + str(np.mean(outs, axis=0)) + '\n')
        print('\t[-] train: ' + str(np.mean(outs, axis=0)) + '\n')
        
        model.save_weights(params['save_weights_path'])


if __name__ == '__main__':
    params = init_params()
    train(params)
