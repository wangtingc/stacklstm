import theano
import theano.tensor as T
from lasagne.layers import MergeLayer, Gate, LSTMLayer, InputLayer
from lasagne import init, nonlinearities


# TODO:
#   1. add h_init
#   2. generate, predict if predict at next step

class StackLSTMDecoder(MergeLayer):
    def __init__(self, incomings, 
                 num_units,
                 ig=Gate(),
                 fg=Gate(),
                 c=Gate(nonlinearity=nonlinearities.tanh),
                 og=StackGate(),
                 nonlin=nonlinearities.tanh,
                 # common setting
                 grad_clipping=0,
                 only_return_final=True,
                 **kwargs): 

        """
        parameters:
        ==========
            ig: inputgate
            fg: forgetgate
            ag: fathergate
            c: cellgate
            og: outputgate
        """


        # This layer inherits from a MergerLayer, because it has [x, m, p, f]
        # five inputs. 
        [l_x, l_m, l_p , l_f] = incomings
        super(StackLSTMEncoder, self).__init__(incomings, **kwargs)

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
                    self.add_param(gate.W_f, (num_units, num_units),
                                   name='W_a_to_{}'.format(gate_name)),
                    self.add_param(gate.b, (num_units,),
                                   name='b_{}'.format(gate_name)),
                    gate.nonlinearity)

        (self.W_i_to_ig, self.W_h_to_ig, self.W_a_to_ig, self.b_ig, self.nonlin_ig) = add_gate_params(ig, 'ig')
        (self.W_i_to_fg, self.W_h_to_fg, self.W_a_to_fg, self.b_fg, self.nonlin_fg) = add_gate_params(fg, 'fg')
        (self.W_i_to_ag, self.W_h_to_ag, self.W_a_to_ag, self.b_ag, self.nonlin_ag) = add_gate_params(ag, 'ag')
        (self.W_i_to_og, self.W_h_to_og, self.W_a_to_og, self.b_og, self.nonlin_og) = add_gate_params(og, 'og')
        (self.W_i_to_c, self.W_h_to_c, self.W_a_to_c, self.b_c, self.nonlin_c) = add_gate_params(c, 'c')


    def get_output_shape_for(self, input_shapes):
        input_shape = input_shapes[0]
        if self.only_return_final:
            return input_shape[0], self.num_units
        else:
            return input_shape[0], input_shape[1], self.num_units


    def get_output_for(self, inputs):
        x, m, p, a = inputs
        
        x = x.dimshuffle(1, 0, 2)   # seq_len, batch_size, num_units_stack
        m = m.dimshuffle(1, 0)      # seq_len, batch_size
        p = p.dimshuffle(1, 0)      # seq_len, batch_size
        a = a.dimshuffle(1, 0)      # seq_len, batch_size, 2
        seq_len, batch_size, _ = x.shape
        
        # W_left_stacked for stack
        W_i = T.concatenate(
            [self.W_i_to_ig, self.W_i_to_fg, self.W_i_to_ag,
             self.W_l_to_og, self.W_l_to_c], axis=1)
        
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

        seqs = [x, m, p, a]
        non_seqs = [W_i, W_h, W_a, b]
        
        # The first element is the initial state for 2 reasons:
        # 1. if the first mask is 0, there should be a previous state.
        # 2. it can be used for empty element in tracking lstm
        stack_len = seq_len/2 + 1
        c_init = T.zeros([batch_size, self.num_units])
        h_init = T.zeros([batch_size, self.num_units])
        c_stack_init = T.zeros([stack_len, batch_size, self.num_units])
        h_stack_init = T.zeros([stack_len, batch_size, self.num_units])

        h, s = theano.scan(
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


    def _step(self, x_n, m_n, p_n, a_n, 
              prev_c, prev_h, prev_c_stack, prev_h_stack,
              W_i, W_h, W_a, b,):
        
        # [ct, ht] = f(ht-1, ft, xn)
        
        batch_size, _ = x_n.shape
        ac_n = prev_c_stack[a_n, T.arange(batch_size)]
        ah_n = prev_h_stack[a_n, T.arange(batch_size)]

        def slice_w(x, n):
            return x[:, n*self.num_units:(n+1)*self.num_units]
        
        gates = T.dot(i_n, W_i) + T.dot(prev_h, W_h) + T.dot(ah_n, W_a) + b

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
        h = og_t * self.nonlin(c)

        return [c, h]
