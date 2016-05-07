import sys
sys.path.append('../')

import theano
import theano.tensor as T
from helper_layers.stack_lstm_encoder import StackLSTMEncoder
from helper_layers.linear_layer import LinearLayer
from lasagne.layers import InputLayer, EmbeddingLayer, DenseLayer, get_output, DropoutLayer
from lasagne.regularization import l2
from lasagne import init, nonlinearities, updates
from lasagne.objectives import categorical_crossentropy


class StackLSTMSim(object):
    def __init__(self, num_units_stack, 
                 num_units_track, 
                 num_classes,
                 dict_size,
                 dim_emb,
                 w_emb,
                 emb_dropout,
                 clf_dropout,
                 lr = 1e-3,
                 lena = 3e-5,
                 alpha = 3.9,
                 ):
        """
        Paramters:
        =========
            num_units_stack:
            num_units_track:
            num_classes:
            dict_size:
            dim_emb:
            w_emb: np.array or lasagne.init
            lr: learning rate
            lena: the coefficient of l2 regularization
            train_w_emb: False in the paper
        """

        self.num_units_track = num_units_track
        self.num_units_stack = num_units_stack
        self.lr = lr
        self.lena = lena
        self.alpha = alpha
        w_emb = w_emb.astype(theano.config.floatX)
        
        # single example encoder
        self.l_x = InputLayer((None, None))
        self.l_m = InputLayer((None, None))
        self.l_a = InputLayer((None, None))
        self.l_p = InputLayer((None, None, 2))
        self.l_emb = EmbeddingLayer(self.l_x, dict_size, dim_emb, W=w_emb)
        self.l_emb = DropoutLayer(self.l_emb, emb_dropout)
        self.l_xc = LinearLayer(self.l_emb, num_units_stack, W=init.HeUniform())
        self.l_xh = LinearLayer(self.l_emb, num_units_stack, W=init.HeUniform())
        self.l_enc = StackLSTMEncoder([self.l_xc, self.l_xh, self.l_m, self.l_a,
                                       self.l_p], 
                                       num_units_stack = self.num_units_stack,
                                       num_units_track = self.num_units_track
                                       )

        self.l_in_a = InputLayer([None, num_units_track])
        self.l_clf_a = DropoutLayer(self.l_in_a, clf_dropout)
        self.l_clf_a = DenseLayer(self.l_clf_a, 2, W=init.Uniform(5e-3), 
                                  nonlinearity=nonlinearities.softmax)

        self.l_in_t = InputLayer([None, 4*num_units_stack])
        self.l_clf_t = DropoutLayer(self.l_in_t, clf_dropout)
        self.l_clf_t = DenseLayer(self.l_clf_t, num_classes, W=init.Uniform(5e-3),
                                  nonlinearity=nonlinearities.softmax)


    def _forward(self, inputs):
        x0, m0, a0, p0, x1, m1, a1, p1 = inputs

        xc0, xh0 = get_output([self.l_xc, self.l_xh], {self.l_x: x0})
        xc1, xh1 = get_output([self.l_xc, self.l_xh], {self.l_x: x1})

        enc0, e0 = get_output(self.l_enc, {self.l_xc: xc0,
                         self.l_xh: xh0,
                         self.l_m: m0,
                         self.l_a: a0,
                         self.l_p: p0,})

        e0 = e0.reshape([-1, self.num_units_track])
        pa0 = get_output(self.l_clf_a, {self.l_in_a: e0})

        enc1, e1 = get_output(self.l_enc, {self.l_xc: xc1,
                         self.l_xh: xh1,
                         self.l_m: m1,
                         self.l_a: a1,
                         self.l_p: p1,})

        e1 = e1.reshape([-1, self.num_units_track])
        pa1 = get_output(self.l_clf_a, {self.l_in_a: e1})

        clf_t_in = T.concatenate([enc0, enc1, enc0-enc1, enc0*enc1], axis=1)
        t = get_output(self.l_clf_t,  {self.l_in_t: clf_t_in})
       
        # return flattened pa0 and pa1
        return pa0, pa1, t

    
    def get_f_train(self,):
        x0, x1 = T.lmatrix(), T.lmatrix()
        m0, m1 = T.matrix(), T.matrix()
        a0, a1 = T.lmatrix(), T.lmatrix()
        p0, p1 = T.ltensor3(), T.ltensor3()
        y = T.lvector()
        
        inputs = [x0, m0, a0, p0, x1, m1, a1, p1]
        pa0, pa1, t = self._forward(inputs)

        lt = categorical_crossentropy(t, y)
        lpa0 = categorical_crossentropy(pa0, a0.flatten())
        lpa1 = categorical_crossentropy(pa1, a1.flatten())

        lpa0 = lpa0.reshape(x0.shape)
        lpa1 = lpa1.reshape(x1.shape)

        lpa0 = (lpa0 * m0).sum(axis=1)
        lpa1 = (lpa1 * m1).sum(axis=1)

        l = (self.alpha*(lpa0 + lpa1) + lt).mean()
        l += self._l2_regularization()

        params = self.get_params()
        update = updates.adam(l, params, self.lr)
        
        inputs += [y]
        f_train = theano.function(inputs, l, updates=update)

        return f_train
        
    
    def _l2_regularization(self,):
        params = self.get_params()
        l_reg = 0
        for w in params:
            if w.name == 'W':
                l_reg += self.lena * l2(w).sum()

        return l_reg


    def get_params(self,):
        params = []
        params += self.l_enc.get_params()
        params += self.l_clf_a.get_params()
        params += self.l_clf_t.get_params()
        return params
