import sys
sys.path.append('../')

import theano
import theano.tensor as T
from helper_layers.stack_lstm_decoder import StackLSTMDecoder
from helper_layers.linear_layer import LinearLayer
from lasagne.layers import InputLayer, EmbeddingLayer, DenseLayer, get_output, DropoutLayer, get_all_params, ReshapeLayer
from lasagne.regularization import l2
from lasagne import init, nonlinearities, updates, objectives
#from lasagne.objectives import categorical_crossentropy

# TODO:
#   test function
#   l2 for w_emb?

class StackLSTMLM(object):
    def __init__(self, num_units, 
                 dict_size,
                 dim_emb,
                 w_emb,
                 emb_dropout,
                 lr = 1e-3,
                 ):
        """
        Paramters:
        =========
        """

        self.num_units = num_units
        self.lr = lr
        self.dict_size = dict_size
        w_emb = w_emb.astype(theano.config.floatX)
        
        # single example encoder
        self.l_x = InputLayer((None, None)) # flattened input
        self.l_m = InputLayer((None, None)) # mask
        self.l_p = InputLayer((None, None)) # if predict
        self.l_a = InputLayer((None, None)) # idx for ancestor in stack
        self.l_emb = EmbeddingLayer(self.l_x, dict_size, dim_emb, W=w_emb)
        self.l_emb = DropoutLayer(self.l_emb, emb_dropout)
        self.l_dec = StackLSTMDecoder([self.l_emb, self.l_m, self.l_p, self.l_a,],
                                       num_units = self.num_units,
                                       only_return_final=False,
                                       )
        self.l_prd = ReshapeLayer(self.l_dec, [-1, num_units])
        self.l_prd = DenseLayer(self.l_prd, dict_size, 
                                nonlinearity=nonlinearities.softmax)


    def _forward(self, inputs):
        x, m, p, a = inputs
        h = get_output(self.l_prd, {self.l_x: x,
                                    self.l_m: m,
                                    self.l_p: p,
                                    self.l_a: a,})
        return h
    
    def get_f_train(self,):
        x = T.lmatrix()
        m = T.matrix()
        p = T.lmatrix()
        a = T.lmatrix()
        
        inputs = [x, m, p, a]
        pred = self._forward(inputs)
        batch_size, seq_len = x.shape
        
        l = objectives.categorical_crossentropy(pred, x.flatten())
        l = l.reshape([batch_size, seq_len])
        l = (l * p).sum(axis=1)
        ppl = T.exp(T.sum(l) / T.sum(p))
        l = l.mean()

        params = self.get_params()
        grads = theano.grad(l, params)
        grads = [T.clip(g, -10, 10) for g in grads]
        grads = updates.total_norm_constraint(grads, max_norm=20)
        update = updates.adam(grads, params, self.lr)
        
        f_train = theano.function(inputs, [l, ppl], updates=update)
        return f_train
        
    def get_f_test(self,):
        x = T.lmatrix()
        m = T.matrix()
        p = T.lmatrix()
        a = T.lmatrix()
        
        inputs = [x, m, p, a]
        pred = self._forward(inputs)
        batch_size, seq_len = x.shape
        
        l = objectives.categorical_crossentropy(pred, x.flatten())
        l = l.reshape([batch_size, seq_len])
        l = (l * p).sum(axis=1)
        ppl = T.exp(T.sum(l) / T.sum(p))
        l = l.mean()
        
        f_test = theano.function(inputs, [l,ppl])
        return f_test
       
    
    def _l2_regularization(self,):
        params = self.get_params()
        l_reg = 0
        for w in params:
            if w.name == 'W':
                l_reg += self.lena * l2(w).sum()

        return l_reg


    def get_params(self,):
        params = get_all_params(self.l_dec)
        return params
