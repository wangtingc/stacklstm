from stack_lstm_decoder import StackLSTMDecoder
from lasagne.layers import InputLayer
import theano.tensor as T
import theano
import numpy as np


# TODO:
#   test_step

dm = 7
l_x = InputLayer([None, None, dm])
l_m = InputLayer([None, None])
l_p = InputLayer([None, None])
l_a = InputLayer([None, None])
num_units = 512

sld = StackLSTMDecoder([l_x, l_m, l_p, l_a],
                        num_units=num_units,
                        only_return_final=False)

def test_main(sld):
    x = T.tensor3()
    m = T.matrix()
    p = T.lmatrix()
    a = T.lmatrix()
    inputs = [x, m, p, a]
    o = sld.get_output_for(inputs)
    f = theano.function([x, m, p, a], o, on_unused_input='warn')
    
    bs = 3
    sl = 5

    x_np = np.zeros([bs, sl, dm], dtype=theano.config.floatX)
    m_np = np.zeros([bs, sl], dtype=theano.config.floatX)
    p_np = np.ones([bs, sl], dtype='int64')
    a_np = np.zeros([bs, sl], dtype='int64')

    o =  f(x_np, m_np, p_np, a_np)
    for i in o:
        print i.shape, i.dtype
    

def test_step(sld):
    x = T.tensor3()
    m = T.matrix()
    p = T.lmatrix()
    a = T.lmatrix()
    
    _x = x.dimshuffle(1, 0, 2)   # seq_len, batch_size, num_units
    _m = m.dimshuffle(1, 0)      # seq_len, batch_size
    _p = p.dimshuffle(1, 0)      # seq_len, batch_size
    _a = a.dimshuffle(1, 0)      # seq_len, batch_size, 2
    seq_len, batch_size, _ = _x.shape

    # W_left_stacked for stack
    W_i = T.concatenate(
            [sld.W_i_to_ig, sld.W_i_to_fg, sld.W_i_to_ag,
             sld.W_i_to_og, sld.W_i_to_c], axis=1)
        
    W_h = T.concatenate(
            [sld.W_h_to_ig, sld.W_h_to_fg, sld.W_h_to_ag,
             sld.W_h_to_og, sld.W_h_to_c], axis=1)
    
    W_a = T.concatenate(
            [sld.W_a_to_ig, sld.W_a_to_fg, sld.W_a_to_ag,
             sld.W_a_to_og, sld.W_a_to_c], axis=1)
    
    b = T.concatenate(
            [sld.b_ig, sld.b_fg, sld.b_ag,
             sld.b_og, sld.b_c], axis=0)
    
    stack_len = seq_len/2 + 1
    c_init = T.zeros([batch_size, sld.num_units])
    h_init = T.zeros([batch_size, sld.num_units])
    c_stack_init = T.zeros([stack_len, batch_size, sld.num_units])
    h_stack_init = T.zeros([stack_len, batch_size, sld.num_units])

    o = sld._step(0, _x[0], _m[0][:,None], _p[0][:,None], _a[0], c_init, h_init, c_stack_init, h_stack_init, W_i, W_h, W_a, b, _p)

    f = theano.function([x, m, p, a], o, on_unused_input='warn')

    bs = 3
    sl = 5

    x_np = np.zeros([bs, sl, dm], dtype=theano.config.floatX)
    m_np = np.zeros([bs, sl], dtype=theano.config.floatX)
    p_np = np.zeros([bs, sl], dtype='int64')
    a_np = np.zeros([bs, sl], dtype='int64')

    o =  f(x_np, m_np, p_np, a_np)
    for i in o:
        print i.shape, i.dtype
       

test_main(sld)
#test_step(sld)
