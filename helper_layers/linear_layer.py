from lasagne.layers import Layer
from lasagne import init
import theano.tensor as T

class LinearLayer(Layer):
    def __init__(self, incoming,
                 num_units,
                 W=init.GlorotUniform(),
                 b=init.Constant(0.),):

        super(LinearLayer, self).__init__(incoming)
        self.num_units = num_units
        self.W = self.add_param(W, (self.input_shape[-1], num_units), name='W')
        self.b = self.add_param(b, (num_units,), name='b', regularizable=False)
    

    def get_output_for(self, input_shape):
        return input_shape[:-1] + [self.num_units]


    def get_output_for(self, input, **kwargs):
        return T.dot(input, self.W) + self.b
