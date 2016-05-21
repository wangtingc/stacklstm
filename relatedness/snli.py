import sys
sys.path.append('../')
import os
import numpy as np
from datetime import datetime
import time

from utils import snli_loader
from utils import batch_iterator
from utils import misc
from models.stack_lstm_sim import StackLSTMSim

def init_params():
    params = {}
    params['dataset'] = 'snli'
    params['data_path'] = '../data/snli/'
    params['dim_emb'] = 300
    params['num_units_stack'] = 512
    params['num_units_track'] = 80
    params['num_classes'] = 2
    params['batch_size'] = 32
    params['num_epochs'] = 100
    params['valid_period'] = 1
    params['test_period'] = 1
    params['exp_time'] = str(datetime.now().strftime('%Y%m%d-%H%M'))
    params['emb_dropout'] = 0.14
    params['clf_dropout'] = 0.06
    params['word_dropout'] = 0
    params['lr'] = 2e-3
    params['lena'] = 3e-5
    params['alpha'] = 3.9
    
    # paths
    params['save_dir'] = '../results/' + params['exp_time'] + '/'
    params['save_weights_path'] = params['save_dir'] + 'weights.pkl'
    params['load_weights_path'] = None
    params['log_path'] = params['save_dir'] + 'log.txt'
    params['details_log_path'] = params['save_dir'] + 'details.txt'
    return params


def train(params):
    data_loader = snli_loader.SnliLoader(params['data_path'])

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

    model = StackLSTMSim(num_units_stack=params['num_units_stack'],
                         num_units_track=params['num_units_track'],
                         num_classes=params['num_classes'],
                         dict_size=data_loader.dict_size,
                         dim_emb=data_loader.dim_emb,
                         w_emb=data_loader.w_emb,
                         emb_dropout=params['emb_dropout'],
                         clf_dropout=params['clf_dropout'],
                         lr=params['lr'],
                         lena=params['lena'],
                         alpha=params['alpha'],
                         )

    f_train = model.get_f_train()
    f_test = model.get_f_test()

    if not os.path.exists(params['save_dir']):
        os.mkdir(params['save_dir'])

    config_info = misc.configuration2str(params)
    weights_info = misc.weight_info2str(model.get_params())

    df = open(params['details_log_path'], 'w')
    lf = open(params['log_path'], 'w')

    df.write(config_info + '\n')
    lf.write(config_info + '\n')
    print(config_info)

    df.write(weights_info + '\n')
    lf.write(config_info + '\n')
    print(weights_info)
    
    num_batches_train = params['num_samples_train'] / params['batch_size']
    num_batches_valid = params['num_samples_valid'] / params['batch_size']
    num_batches_test = params['num_samples_test'] / params['batch_size']

    for epoch in range(params['num_epochs']):
        epoch_info = ' [*] epoch %d \n' % epoch
        lf.write(epoch_info + '\n')
        df.write(epoch_info + '\n')
        print(epoch_info)
    
        out_all = []
        for batch in range(num_batches_train):
            x0, x1, a0, a1, p0, p1, y = it_train.next()
            x0, m0, a0, p0 = misc.prepare_snli([x0, a0, p0])
            x1, m1, a1, p1 = misc.prepare_snli([x1, a1, p1])
            out = f_train(x0, m0, a0, p0, x1, m1, a1, p1, y)
            out_all.append(out)
            #print('\t[-] train: ' + str(out) + '\n')
            df.write('\t[-] train: ' + str(out) + '\n')
        train_info = '\t[-] train: ' + str(np.mean(out_all, axis=0))
        lf.write(train_info + '\n')
        print(train_info)
     
        out_all = []
        for batch in range(num_batches_valid):
            x0, x1, a0, a1, p0, p1, y = it_test.next()
            x0, m0, a0, p0 = misc.prepare_snli([x0, a0, p0])
            x1, m1, a1, p1 = misc.prepare_snli([x1, a1, p1])
            out = f_test(x0, m0, a0, p0, x1, m1, a1, p1, y)
            out_all.append(out)
            #print('\t[-] train: ' + str(out) + '\n')
            df.write('\t[-] train: ' + str(out) + '\n')
        train_info = '\t[-] train: ' + str(np.mean(out_all, axis=0))
        lf.write(train_info + '\n')
        print(train_info)    

        model.save_weights(params['save_weights_path'])


if __name__ == '__main__':
    params = init_params()
    train(params)
