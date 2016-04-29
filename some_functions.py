#code by hzsun 2016.4.1
import theano
import theano.tensor as T
import numpy as np

def batched_get_value(x, idx):
    '''
    batched stack get top values, use Theano framework to implement.
    :param x: Tensor matrix with shape (batch_size, stack_size, word_ebd_dim), as stack
    :param idx: T.ivector index for each batch (batch_size,), int32 or int64, stack top index.
    :return: a vector (batch_size,) of stack top values
    '''
    batch_size, stack_size = x.shape[0].astype('int64'), x.shape[1].astype('int64')
    idx_in_batch = idx + T.arange(batch_size, dtype='int64') * stack_size
    return x.reshape((batch_size * stack_size, -1))[idx_in_batch]

def batched_set_value(x, v, idx):
    '''
    batched stack set top values, use Theano framework to implement.
    :param x: stack (batch_size, stack_size, word_ebd_dim) (a tmp pointer)
    :param v: values of each batch (batch_size, word_ebd_dim)
    :param idx: index for each batch (batch_size,), int32 or int64, stack top index.
    :return: new stack values
    USE stack = _batched_set_value(stack, values, index) !!!!
    '''
    batch_size, stack_size = x.shape[0].astype('int64'), x.shape[1].astype('int64')
    idx_in_batch = idx + T.arange(batch_size, dtype='int64') * stack_size
    return T.set_subtensor(x.reshape((batch_size * stack_size, -1))[idx_in_batch], v)\
        .reshape((batch_size,stack_size,-1))
    #set_subtensor allocate new space in memory, and return the pointer