from stack_lstm_encoder import *
from lasagne.layers import InputLayer
import numpy as np
from lasagne import updates

num_units_stack = 100
num_units_track = 20
l_xc = InputLayer([None, None, num_units_stack])
l_xh = InputLayer([None, None, num_units_stack])
l_m = InputLayer([None, None,])
l_a = InputLayer([None, None,])
l_p = InputLayer([None, None, 2])

sle = StackLSTMEncoder([l_xc, l_xh, l_m, l_a, l_p], num_units_stack, num_units_track)

def print_params(sle):
    params = sle.get_params()
    for i in params:
        print i.name, i.get_value().shape

def test_get_output_for(sle):
    xc = T.tensor3()
    xh = T.tensor3()
    a = T.matrix()
    m = T.matrix()
    p = T.ltensor3()

    h0, h1 = sle.get_output_for([xc, xh, a, m, p])

    f = theano.function([xc, xh, a, m, p], [h0, h1])
        
    L = 11
    batch_size = 7
    xcn = np.ones([batch_size, L, num_units_stack], dtype='float32')
    xhn = np.ones([batch_size, L, num_units_stack], dtype='float32')
    mn = np.ones([batch_size, L], dtype='float32')
    an = np.ones([batch_size, L], dtype='float32')
    pn = np.ones([batch_size, L, 2], dtype='int64')

    print f(xcn, xhn, mn, an, pn)
    

def test_step(sle):
    t_n = 1
    xc_n = T.matrix()
    xh_n = T.matrix()
    m_n = T.vector()
    a_n = T.vector()
    p_n = T.lmatrix()
    prev_c_stack = T.tensor3()
    prev_h_stack = T.tensor3()
    prev_c_track = T.matrix()
    prev_h_track = T.matrix()

 
    # W_left_stacked for stack
    W_l_s = T.concatenate(
            [sle.W_l_to_lg_s, sle.W_l_to_rg_s, 
             sle.W_l_to_ig_s, sle.W_l_to_og_s,
             sle.W_l_to_c_s], axis=1)
        
    # W_right_stacked for stack
    W_r_s = T.concatenate(
            [sle.W_l_to_lg_s, sle.W_l_to_rg_s,
             sle.W_l_to_ig_s, sle.W_l_to_og_s,
             sle.W_l_to_c_s], axis=1)
    
    # W_track_stacked for stack
    W_e_s = T.concatenate(
            [sle.W_e_to_lg_s, sle.W_e_to_rg_s,
             sle.W_e_to_ig_s, sle.W_e_to_og_s,
             sle.W_e_to_c_s], axis=1)
        
    # b_stacked for stack
    b_s = T.concatenate(
            [sle.b_lg_s, sle.b_rg_s,
             sle.b_ig_s, sle.b_og_s,
             sle.b_c_s], axis=0)

    # W_in_stacked for track
    W_i_t = T.concatenate(
            [sle.W_i_to_ig_t, sle.W_i_to_fg_t,
             sle.W_i_to_og_t, sle.W_i_to_c_t], axis=1)

    # Same for hidden weight matrices
    W_h_t = T.concatenate(
            [sle.W_h_to_ig_t, sle.W_h_to_fg_t,
             sle.W_h_to_og_t, sle.W_h_to_c_t], axis=1)

    # Stack biases into a (4*num_units) vector
    b_t = T.concatenate(
            [sle.b_ig_t, sle.b_fg_t,
             sle.b_og_t, sle.b_c_t], axis=0)

    '''
    s_idx = T.eq(a_n, 0).nonzero()
    r_idx = T.eq(a_n, 1).nonzero()[0] # tuple to list
    
    p_r = p_n[r_idx, :]
    c_l = prev_c_stack[p_r[:, 0], r_idx]
    
    batch_size, _ = xc_n.shape
    h0 = prev_c_stack[p_n[:, 0], T.arange(batch_size)]
    '''
    outputs = sle._step(t_n, xc_n, xh_n, m_n, a_n, p_n, prev_c_stack, prev_h_stack,
                                   prev_c_track, prev_h_track, W_l_s, W_r_s, W_e_s, b_s,
                                   W_i_t, W_h_t, b_t)
    
    inputs = [xc_n, xh_n, m_n, a_n, p_n, prev_c_stack, prev_h_stack, prev_c_track, prev_h_track]
    f = theano.function(inputs, outputs, on_unused_input='ignore')
        
    b = 7
    ns = num_units_stack
    nt = num_units_track
    xcn = np.zeros([b, ns], dtype='float32')
    xhn = np.zeros([b, ns], dtype='float32')
    mnn = np.asarray([0,0,0,0,0], dtype='float32')
    ann = np.asarray([0,0,1,1,1], dtype='float32')
    pnn = np.asarray([[1,2,3,4,5], [2,3,4,5,0]], dtype='int64').T
    pcs = np.zeros([20, 5, ns], dtype='float32')
    phs = np.zeros([20, 5, ns], dtype='float32')
    pct = np.zeros([5, nt], dtype='float32')
    pht = np.zeros([5, nt], dtype='float32')
    o =  f(xcn, xhn, mnn, ann, pnn, pcs, phs, pct, pht)
    print o
    for i in o:
        print i.shape



def test_composite(sle):
    c_l = T.matrix()
    c_r = T.matrix()
    h_l = T.matrix()
    h_r = T.matrix()
    e = T.matrix()
 
    # W_left_stacked for stack
    W_l_s = T.concatenate(
            [sle.W_l_to_lg_s, sle.W_l_to_rg_s, 
             sle.W_l_to_ig_s, sle.W_l_to_og_s,
             sle.W_l_to_c_s], axis=1)
        
    # W_right_stacked for stack
    W_r_s = T.concatenate(
            [sle.W_l_to_lg_s, sle.W_l_to_rg_s,
             sle.W_l_to_ig_s, sle.W_l_to_og_s,
             sle.W_l_to_c_s], axis=1)
    
    # W_track_stacked for stack
    W_e_s = T.concatenate(
            [sle.W_e_to_lg_s, sle.W_e_to_rg_s,
             sle.W_e_to_ig_s, sle.W_e_to_og_s,
             sle.W_e_to_c_s], axis=1)
        
    # b_stacked for stack
    b_s = T.concatenate(
            [sle.b_lg_s, sle.b_rg_s,
             sle.b_ig_s, sle.b_og_s,
             sle.b_c_s], axis=0)

    c, h = sle._composite(c_l, h_l, c_r, h_r, e, W_l_s, W_r_s, W_e_s, b_s)

    f = theano.function([c_l, h_l, c_r, h_r, e], [c, h])

    batch_size = 5
    cln = np.ones([batch_size, num_units_stack], dtype='float32')
    hln = np.ones([batch_size, num_units_stack], dtype='float32')
    crn = np.ones([batch_size, num_units_stack], dtype='float32')
    hrn = np.ones([batch_size, num_units_stack], dtype='float32')
    en = np.ones([batch_size, num_units_track], dtype='float32')

    for i in  f(cln, hln, crn, hrn, en):
        print i.shape


def test_track(sle):
    i_n = T.matrix()
    p_h = T.matrix()
    p_c = T.matrix()
    # W_in_stacked for track
    W_i_t = T.concatenate(
            [sle.W_i_to_ig_t, sle.W_i_to_fg_t,
             sle.W_i_to_og_t, sle.W_i_to_c_t], axis=1)

    # Same for hidden weight matrices
    W_h_t = T.concatenate(
            [sle.W_h_to_ig_t, sle.W_h_to_fg_t,
             sle.W_h_to_og_t, sle.W_h_to_c_t], axis=1)

    # Stack biases into a (4*num_units) vector
    b_t = T.concatenate(
            [sle.b_ig_t, sle.b_fg_t,
             sle.b_og_t, sle.b_c_t], axis=0)
    
    c, h = sle._track(i_n, p_c, p_h, W_i_t, W_h_t, b_t)
    f = theano.function([i_n, p_c, p_h], [c, h])

    batch_size = 7
    inn = np.ones([batch_size, 3*num_units_stack], dtype='float32')
    pcn = np.ones([batch_size, num_units_track], dtype='float32')
    phn = np.ones([batch_size, num_units_track], dtype='float32')

    for i in  f(inn, pcn, phn):
        print i.shape


def test_main(sle):
    xc = T.tensor3()
    xh = T.tensor3()
    m = T.matrix()
    a = T.matrix()
    p = T.ltensor3()
    
    inputs = [xc, xh, m, a, p]
    h0, h1 = sle.get_output_for(inputs)

    loss = h0.sum()

    params = sle.get_params()
    update = updates.adam(loss, params, 1)

    f = theano.function(inputs, loss, updates=update)
    return f

#test_main(sle)

    

    
    

#print_params(sle)
#test_composite(sle)
#test_track(sle)
#test_step(sle)
test_get_output_for(sle)
