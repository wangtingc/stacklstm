import numpy as np
import theano
import theano.tensor as T
from some_functions import *

stack = T.zeros((2,3,3)) +1.0
values = T.matrix('values')
idx = T.ivector('idx')
o1 = batched_get_value(stack, idx)
stack = batched_set_value(stack, values, idx)
#stack = o2
f = theano.function([values,idx] , [o1,stack])

#r = f(np.asarray([1,2]).astype('int32'))
r = f(np.random.randn(2,3).astype('float32'), np.asarray([1,2]).astype('int32'))

for i in r:
    print i





