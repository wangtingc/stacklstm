import sys
sys.path.append('../')

import theano
import theano.tensor as T
from lasagne.layers import LSTMLayer
from helper_layers.linear_layer import LinearLayer
from lasagne.layers import InputLayer, EmbeddingLayer, DenseLayer, get_output, DropoutLayer, get_all_params, ReshapeLayer
from lasagne.regularization import l2
from lasagne import init, nonlinearities, updates, objectives
#from lasagne.objectives import categorical_crossentropy


class LSTMLM(object):
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
        self.l_emb = EmbeddingLayer(self.l_x, dict_size, dim_emb, W=w_emb, name='l_emb')
        self.l_emb = DropoutLayer(self.l_emb, emb_dropout, name='l_emb_drop')
        self.l_dec = LSTMLayer(self.l_emb, mask_input=self.l_m, 
                                       num_units = self.num_units,
                                       only_return_final=False,
                                       name='l_dec',
                                       )
        self.l_dec = ReshapeLayer(self.l_dec, [-1, num_units], name='l_reshape')
        self.l_prd_x = DenseLayer(self.l_dec, dict_size, 
                                nonlinearity=nonlinearities.softmax, name='l_prd_x')

    def _forward(self, inputs, deterministic=False):
        x, m, = inputs

        e = get_output(self.l_emb, {self.l_x: x})
        e_shift = T.zeros(e.shape)
        e_shift = T.set_subtensor(e_shift[:, 1:, :], e[:, :-1, :])

        px = get_output(self.l_prd_x, {self.l_emb: e_shift,
                                    self.l_m: m,},
                                    deterministic=deterministic)
        return px
    
    def get_f_train(self,):
        x = T.lmatrix()
        m = T.lmatrix()
        
        inputs = [x, m,]
        px = self._forward(inputs, False)
        batch_size, seq_len = x.shape
        
        lx = objectives.categorical_crossentropy(px, x.flatten())
        lx = lx.reshape([batch_size, seq_len])
        lx = (lx * m).sum(axis=1)
        lx = lx.mean()
        
        # although it is wrong to cal ppl this way, it remains for batch testing.
        ppl = T.exp(lx * batch_size / T.sum(m))

        params = self.get_params()
        grads = theano.grad(lx, params)
        grads = [T.clip(g, -10, 10) for g in grads]
        grads = updates.total_norm_constraint(grads, max_norm=20)
        update = updates.adam(grads, params, self.lr)
        
        f_train = theano.function(inputs, [lx, ppl], updates=update)
        return f_train
        
    def get_f_test(self,):
        x = T.lmatrix()
        m = T.lmatrix()
         
        inputs = [x, m, ]
        px = self._forward(inputs, True)
        batch_size, seq_len = x.shape
        
        lx = objectives.categorical_crossentropy(px, x.flatten())
        lx = lx.reshape([batch_size, seq_len])
        lx = (lx * m).sum(axis=1)
        lx = lx.mean()

        ppl = T.exp(lx * batch_size / T.sum(m))

        f_test = theano.function(inputs, [lx, ppl])
        return f_test
       
    def _l2_regularization(self,):
        params = self.get_params()
        l_reg = 0
        for w in params:
            if w.name == 'W':
                l_reg += self.lena * l2(w).sum()

        return l_reg

    def get_params(self,):
        params = get_all_params([self.l_prd_x,])
        return params
