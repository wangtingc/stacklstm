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
        self.l_x = InputLayer((None, None), name='l_x') # flattened input
        self.l_m = InputLayer((None, None), name='l_m') # mask
        self.l_p = InputLayer((None, None), name='l_p') # if predict
        self.l_a = InputLayer((None, None), name='l_a') # idx for ancestor in stack
        self.l_emb = EmbeddingLayer(self.l_x, dict_size, dim_emb, W=w_emb, name='l_emb')
        self.l_emb = DropoutLayer(self.l_emb, emb_dropout, name='l_emb_drop')
        self.l_dec = StackLSTMDecoder([self.l_emb, self.l_m, self.l_p, self.l_a,],
                                       num_units = self.num_units,
                                       only_return_final=False,
                                       name='l_dec',
                                       )
        self.l_dec = ReshapeLayer(self.l_dec, [-1, num_units], name='l_reshape')
        self.l_prd_x = DenseLayer(self.l_dec, dict_size, 
                                nonlinearity=nonlinearities.softmax, name='l_prd_x')
        self.l_prd_p = DenseLayer(self.l_dec, 2, 
                                nonlinearity=nonlinearities.softmax, name='l_prd_p')


    def _forward(self, inputs, deterministic=False):
        x, m, p, a = inputs
        e = get_output(self.l_emb, {self.l_x: x})

        # shift the embedding
        e_shift = T.zeros_like(e)
        e_shift = T.set_subtensor(e_shift[:,1:,:], e[:,:-1,:])

        px, pp = get_output([self.l_prd_x, self.l_prd_p], {self.l_emb: e_shift,
                                    self.l_m: m,
                                    self.l_p: p,
                                    self.l_a: a,},
                                    deterministic=deterministic)
        return px, pp
    
    def get_f_train(self,):
        x = T.lmatrix()
        m = T.matrix()
        p = T.lmatrix()
        a = T.lmatrix()
        
        inputs = [x, m, p, a]
        px, pp = self._forward(inputs, False)
        batch_size, seq_len = x.shape
        
        lx = objectives.categorical_crossentropy(px, x.flatten())
        lx = lx.reshape([batch_size, seq_len])
        lx = (lx * p).sum(axis=1)
        lx = lx.mean()
        
        # although it is wrong to cal ppl this way, it remains for batch testing.
        ppl = T.exp(T.sum(lx) / T.sum(p))

        lp = objectives.categorical_crossentropy(pp, p.flatten())
        lp = lp.reshape([batch_size, seq_len])
        lp = (lp * m).sum(axis=1)
        lp = lp.mean()

        params = self.get_params()
        grads = theano.grad(lx+lp, params)
        grads = [T.clip(g, -10, 10) for g in grads]
        grads = updates.total_norm_constraint(grads, max_norm=20)
        update = updates.adam(grads, params, self.lr)
        
        f_train = theano.function(inputs, [lx, lp, ppl], updates=update)
        return f_train
        
    def get_f_test(self,):
        x = T.lmatrix()
        m = T.matrix()
        p = T.lmatrix()
        a = T.lmatrix()
         
        inputs = [x, m, p, a]
        px, pp = self._forward(inputs, True)
        batch_size, seq_len = x.shape
        
        lx = objectives.categorical_crossentropy(px, x.flatten())
        lx = lx.reshape([batch_size, seq_len])
        lx = (lx * p).sum(axis=1)
        lx = lx.mean()

        ppl = T.exp(T.sum(lx) / T.sum(p))

        lp = objectives.categorical_crossentropy(pp, p.flatten())
        lp = lp.reshape([batch_size, seq_len])
        lp = (lp * m).sum(axis=1)
        lp = lp.mean()

        f_test = theano.function(inputs, [lx, lp ,ppl])
        return f_test
       
    
    def _l2_regularization(self,):
        params = self.get_params()
        l_reg = 0
        for w in params:
            if w.name == 'W':
                l_reg += self.lena * l2(w).sum()

        return l_reg


    def get_params(self,):
        params = get_all_params([self.l_prd_x, self.l_prd_p])
        return params
