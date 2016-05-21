import theano
import theano.tensor as T
from lasagne.layers import MergeLayer, Gate, LSTMLayer, InputLayer
from lasagne import init, nonlinearities
import numpy as np


# TODO:
#   1. add h_init
#   2. generate, predict if predict at next step
#   3. loss for prediction
#   4. if not binarized tree, explicitly memory attention loss?

class StackGate(object):
    def __init__(self, W_i=init.Normal(0.1), W_h=init.Normal(0.1),
                 W_a=init.Normal(0.1), b=init.Constant(0.),
                 nonlinearity=nonlinearities.sigmoid):
        self.W_i = W_i
        self.W_h = W_h
        self.W_a = W_a
        self.b = b
        # For the nonlinearity, if None is supplied, use identity
        if nonlinearity is None:
            self.nonlinearity = nonlinearities.identity
        else:
            self.nonlinearity = nonlinearity

class StackLSTMDecoder(MergeLayer):
    def __init__(self, incomings, 
                 num_units,
                 ig=StackGate(),
                 fg=StackGate(),
                 ag=StackGate(),
                 og=StackGate(),
                 c=StackGate(nonlinearity=nonlinearities.tanh),
                 nonlin=nonlinearities.tanh,
                 # common setting
                 grad_clipping=0,
                 only_return_final=False,
                 **kwargs): 

        """
        parameters:
        ==========
            ig: inputgate
            fg: forgetgate
            ag: fathergate
            og: outputgate
            c: cellgate
        """


        # This layer inherits from a MergerLayer, because it has [x, m, p, f]
        # five inputs. 
        [l_x, l_m, l_p , l_f] = incomings
        super(StackLSTMDecoder, self).__init__(incomings, **kwargs)

        self.nonlin = nonlin
        self.num_units = num_units
        self.grad_clipping = grad_clipping
        self.only_return_final = only_return_final
        
        num_inputs = np.prod(self.input_shapes[0][2:])

        def add_gate_params(gate, gate_name):
            return (self.add_param(gate.W_i, (num_inputs, num_units),
                                   name='W_i_to_{}'.format(gate_name)),
                    self.add_param(gate.W_h, (num_units, num_units),
                                   name='W_h_to_{}'.format(gate_name)),
                    self.add_param(gate.W_a, (num_units, num_units),
                                   name='W_a_to_{}'.format(gate_name)),
                    self.add_param(gate.b, (num_units,),
                                   name='b_{}'.format(gate_name)),
                    gate.nonlinearity)

        (self.W_i_to_ig, self.W_h_to_ig, self.W_a_to_ig, self.b_ig, self.nonlin_ig) = add_gate_params(ig, 'ig')
        (self.W_i_to_fg, self.W_h_to_fg, self.W_a_to_fg, self.b_fg, self.nonlin_fg) = add_gate_params(fg, 'fg')
        (self.W_i_to_ag, self.W_h_to_ag, self.W_a_to_ag, self.b_ag, self.nonlin_ag) = add_gate_params(ag, 'ag')
        (self.W_i_to_og, self.W_h_to_og, self.W_a_to_og, self.b_og, self.nonlin_og) = add_gate_params(og, 'og')
        (self.W_i_to_c , self.W_h_to_c , self.W_a_to_c , self.b_c , self.nonlin_c ) = add_gate_params(c , 'c' )


    def get_output_shape_for(self, input_shapes):
        input_shape = input_shapes[0]
        if self.only_return_final:
            return input_shape[0], self.num_units
        else:
            return input_shape[0], input_shape[1], self.num_units


    def get_output_for(self, inputs, **kwargs):
        x, m, p, a = inputs
        
        x = x.dimshuffle(1, 0, 2)   # seq_len, batch_size, num_units
        m = m.dimshuffle(1, 0)      # seq_len, batch_size
        p = p.dimshuffle(1, 0)      # seq_len, batch_size
        a = a.dimshuffle(1, 0)      # seq_len, batch_size, 2
        seq_len, batch_size, _ = x.shape
        
        # W_left_stacked for stack
        W_i = T.concatenate(
            [self.W_i_to_ig, self.W_i_to_fg, self.W_i_to_ag,
             self.W_i_to_og, self.W_i_to_c], axis=1)
        
        # W_right_stacked for stack
        W_h = T.concatenate(
            [self.W_h_to_ig, self.W_h_to_fg, self.W_h_to_ag,
             self.W_h_to_og, self.W_h_to_c], axis=1)
    
        # W_right_stacked for stack
        W_a = T.concatenate(
            [self.W_a_to_ig, self.W_a_to_fg, self.W_a_to_ag,
             self.W_a_to_og, self.W_a_to_c], axis=1)
    
        # b_stacked for stack
        b = T.concatenate(
            [self.b_ig, self.b_fg, self.b_ag,
             self.b_og, self.b_c], axis=0)
        
        t = T.arange(seq_len)
        seqs = [t, x, m[:, :, None], p[:, :, None], a]
        non_seqs = [W_i, W_h, W_a, b, p]
        
        # The first element is the initial state for 2 reasons:
        # 1. if the first mask is 0, there should be a previous state.
        # 2. it can be used for empty element in tracking lstm
        stack_len = seq_len // 2 + 1
        c_init = T.zeros([batch_size, self.num_units])
        h_init = T.zeros([batch_size, self.num_units])
        c_stack_init = T.zeros([stack_len, batch_size, self.num_units])
        h_stack_init = T.zeros([stack_len, batch_size, self.num_units])

        c, h, c_stack, h_stack = theano.scan(
            fn = self._step,
            sequences=seqs,
            outputs_info=[c_init, h_init, c_stack_init, h_stack_init],
            non_sequences=non_seqs,
            strict=True)[0]
        
        if self.only_return_final:
            h = h[-1] # last step and the top of stack
        else:
            h = h.dimshuffle(1,0,2)

        return h


    def _step(self, t_n, x_n, m_n, p_n, a_n, 
              prev_c, prev_h, prev_c_stack, prev_h_stack,
              W_i, W_h, W_a, b, p,):

        
        # [ct, ht] = f(ht-1, ft, xn)
        
        batch_size, _ = x_n.shape
        ac_n = prev_c_stack[a_n, T.arange(batch_size)]
        ah_n = prev_h_stack[a_n, T.arange(batch_size)]


        def slice_w(x, n):
            return x[:, n*self.num_units:(n+1)*self.num_units]

        
        gates = T.dot(x_n, W_i) + T.dot(prev_h, W_h) + T.dot(ah_n, W_a) + b


        if self.grad_clipping:
            gates = theano.gradient.grad_clip(
                gates, -self.grad_clipping, self.grad_clipping)

        ig = slice_w(gates, 0)
        fg = slice_w(gates, 1)
        ag = slice_w(gates, 2)
        og = slice_w(gates, 3)
        c_input = slice_w(gates, 4)

        ig = self.nonlin_ig(ig)
        fg = self.nonlin_fg(fg)
        ag = self.nonlin_fg(ag)
        og = self.nonlin_og(og)
        c_input = self.nonlin(c_input)

        c = ig*c_input + fg*prev_c + ag*ac_n
        h = og * self.nonlin(c)

        #save the state to stack if pred is False and mask if true
        bs_idx = T.eq(m_n*(1-p_n), 1).nonzero()[0] 
        sq_idx = (1-p)[:t_n].sum(axis=0)[bs_idx]
        c_stack = T.set_subtensor(prev_c_stack[sq_idx, bs_idx], c[bs_idx])
        h_stack = T.set_subtensor(prev_h_stack[sq_idx, bs_idx], h[bs_idx])
    
        
        # if pred is True, generate new state, int64 * float32 = float64
        h = T.cast(prev_h * (1 - p_n) + h * p_n, theano.config.floatX)
        c = T.cast(prev_c * (1 - p_n) + c * p_n, theano.config.floatX)
        

        # if not masked, pass the step
        h = prev_h * (1 - m_n) + h * m_n
        c = prev_c * (1 - m_n) + c * m_n
        h_stack = prev_h_stack * (1 - m_n) + h_stack * m_n
        c_stack = prev_c_stack * (1 - m_n) + c_stack * m_n

        return [c, h, c_stack, h_stack]
