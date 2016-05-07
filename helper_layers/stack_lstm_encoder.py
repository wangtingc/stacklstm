import theano
import theano.tensor as T
from lasagne.layers import MergeLayer, Gate, LSTMLayer, InputLayer
from lasagne import init, nonlinearities

class StackGate(object):
    def __init__(self, W_left=init.Normal(0.1), W_right=init.Normal(0.1),
                 W_track=init.Normal(0.1), b=init.Constant(0.),
                 nonlinearity=nonlinearities.sigmoid):
        self.W_left = W_left
        self.W_right = W_right
        self.W_track = W_track
        self.b = b
        # For the nonlinearity, if None is supplied, use identity
        if nonlinearity is None:
            self.nonlinearity = nonlinearities.identity
        else:
            self.nonlinearity = nonlinearity


class StackLSTMEncoder(MergeLayer):
    def __init__(self, incomings, 
                 num_units_stack,
                 num_units_track,
                 # for stack LSTM
                 ig_s=StackGate(),
                 lg_s=StackGate(),
                 rg_s=StackGate(),
                 c_s=StackGate(nonlinearity=nonlinearities.tanh),
                 og_s=StackGate(),
                 nonlin_s=nonlinearities.tanh,
                 # for track LSTM
                 ig_t=Gate(),
                 fg_t=Gate(),
                 c_t=Gate(W_cell=None, nonlinearity=nonlinearities.tanh),
                 og_t=Gate(),
                 nonlin_t=nonlinearities.tanh,
                 # common setting
                 grad_clipping=0,
                 only_return_final=True,
                 **kwargs): 

        """
        Parameters:
        ==============
            incomings: [l_xc, l_xh, l_m, l_a, l_p] input layers
            num_units_stack: number of units for stack LSTM
            num_units_track: number of units for track LSTM
            ig_s: ingate of stack LSTM
            lg_s: leftgate of stack LSTM
            rg_s: rightgate of stack LSTM
            c_s: cell of stack LSTM
            og_s: outgate of stack LSTM
            nonlin_s: nonlinearity for stack LSTM
            ig_t: ingate of track LSTM
            fg_t: forgetgate of track LSTM
            c_t: cell of track LSTM
            og_t: outgate of track LSTM
            nonlin_t: nonlinearity for track LSTM
            grad_clipping: clip the gradient, 0 if not clipping
            only_return_final: if only return final
        """

        # This layer inherits from a MergerLayer, because it has [xc, xh, m, a, p]
        # five inputs. 
        [l_xc, l_xh, l_m, l_a, l_p] = incomings
        super(StackLSTMEncoder, self).__init__(incomings, **kwargs)

        self.nonlin_s = nonlin_s
        self.nonlin_t = nonlin_t
        self.num_units_stack = num_units_stack
        self.num_units_track = num_units_track
        self.grad_clipping = grad_clipping
        self.only_return_final = only_return_final


        # The tracking LSTM can be simply modelled by a standard LSTM,
        # although there are some considerations about inappropriate hierachical
        # class structure.
        l_in = InputLayer([None, None, 3*self.num_units_stack])
        self.trackingLSTM = LSTMLayer(l_in, self.num_units_track, mask_input = l_m)

        self._add_stack_params(lg_s, rg_s, ig_s, og_s, c_s)
        self._add_track_params(ig_t, fg_t, og_t, c_t)

    def _add_stack_params(self, lg_s, rg_s, ig_s, og_s, c_s):
        num_units_stack = self.num_units_stack
        num_units_track = self.num_units_track
        def add_stackgate_params(gate, gate_name):
            return (self.add_param(gate.W_left, (num_units_stack, num_units_stack),
                                   name='W_l_to_{}'.format(gate_name)),
                    self.add_param(gate.W_right, (num_units_stack, num_units_stack),
                                   name='W_l_to_{}'.format(gate_name)),
                    self.add_param(gate.W_track, (num_units_track, num_units_stack),
                                   name='W_e_to_{}'.format(gate_name)),
                    self.add_param(gate.b, (num_units_stack,),
                                   name='b_{}'.format(gate_name)),
                    gate.nonlinearity)

        (self.W_l_to_lg_s, self.W_r_to_lg_s, self.W_e_to_lg_s,
        self.b_lg_s, self.nonlin_lg_s) = add_stackgate_params(lg_s, 'lg_s')

        (self.W_l_to_rg_s, self.W_r_to_rg_s, self.W_e_to_rg_s,
        self.b_rg_s, self.nonlin_rg_s) = add_stackgate_params(rg_s, 'rg_s')

        (self.W_l_to_ig_s, self.W_r_to_ig_s, self.W_e_to_ig_s, 
        self.b_ig_s, self.nonlin_ig_s) = add_stackgate_params(ig_s, 'ig_s')

        (self.W_l_to_og_s, self.W_r_to_og_s, self.W_e_to_og_s,
        self.b_og_s, self.nonlin_og_s) = add_stackgate_params(og_s, 'og_s')

        (self.W_l_to_c_s, self.W_r_to_c_s, self.W_e_to_c_s,
        self.b_c_s, self.nonlin_c_s) = add_stackgate_params(c_s, 'c_s')


    def _add_track_params(self, ig_t, fg_t, og_t, c_t):
        num_units_input = 3 * self.num_units_stack
        num_units_track = self.num_units_track
        def add_gate_params(gate, gate_name):
            return (self.add_param(gate.W_in, (num_units_input, num_units_track),
                                   name="W_i_to_{}".format(gate_name)),
                    self.add_param(gate.W_hid, (num_units_track, num_units_track),
                                   name="W_h_to_{}".format(gate_name)),
                    self.add_param(gate.b, (num_units_track,),
                                   name="b_{}".format(gate_name),
                                   regularizable=False),
                    gate.nonlinearity)

        # Add in parameters from the supplied Gate instances
        (self.W_i_to_ig_t, self.W_h_to_ig_t, self.b_ig_t,
         self.nonlin_ig_t) = add_gate_params(ig_t, 'ig_t')

        (self.W_i_to_fg_t, self.W_h_to_fg_t, self.b_fg_t,
         self.nonlin_fg_t) = add_gate_params(fg_t, 'fg_t')

        (self.W_i_to_c_t, self.W_h_to_c_t, self.b_c_t,
         self.nonlin_c_t) = add_gate_params(c_t, 'c_t')

        (self.W_i_to_og_t, self.W_h_to_og_t, self.b_og_t,
         self.nonlin_og_t) = add_gate_params(og_t, 'og_t')


    def get_output_shape_for(self, input_shapes):
        input_shape = input_shapes[0]
        if self.only_return_final:
            return input_shape[0], self.num_units_stack
        else:
            return input_shape[0], input_shape[1], self.num_units_stack


    def get_output_for(self, inputs):
        xc, xh, m, a, p = inputs
        
        xc = xc.dimshuffle(1, 0, 2) # seq_len, batch_size, num_units_stack
        xh = xh.dimshuffle(1, 0, 2) # seq_len, batch_size, num_units_stack
        m = m.dimshuffle(1, 0)     # seq_len, batch_size
        a = a.dimshuffle(1, 0)     # seq_len, batch_size
        p = p.dimshuffle(1, 0, 2)  # seq_len, batch_size, 2
        seq_len, batch_size, _ = xc.shape
        
        # W_left_stacked for stack
        W_l_s = T.concatenate(
            [self.W_l_to_lg_s, self.W_l_to_rg_s, 
             self.W_l_to_ig_s, self.W_l_to_og_s,
             self.W_l_to_c_s], axis=1)
        
        # W_right_stacked for stack
        W_r_s = T.concatenate(
            [self.W_r_to_lg_s, self.W_r_to_rg_s,
             self.W_r_to_ig_s, self.W_r_to_og_s,
             self.W_r_to_c_s], axis=1)
    
        # W_track_stacked for stack
        W_e_s = T.concatenate(
            [self.W_e_to_lg_s, self.W_e_to_rg_s,
             self.W_e_to_ig_s, self.W_e_to_og_s,
             self.W_e_to_c_s], axis=1)
        
        # b_stacked for stack
        b_s = T.concatenate(
            [self.b_lg_s, self.b_rg_s,
             self.b_ig_s, self.b_og_s,
             self.b_c_s], axis=0)

        # W_in_stacked for track
        W_i_t = T.concatenate(
            [self.W_i_to_ig_t, self.W_i_to_fg_t,
             self.W_i_to_og_t, self.W_i_to_c_t], axis=1)

        # Same for hidden weight matrices
        W_h_t = T.concatenate(
            [self.W_h_to_ig_t, self.W_h_to_fg_t,
             self.W_h_to_og_t, self.W_h_to_c_t], axis=1)

        # Stack biases into a (4*num_units) vector
        b_t = T.concatenate(
            [self.b_ig_t, self.b_fg_t,
             self.b_og_t, self.b_c_t], axis=0)

        t = T.arange(seq_len) + 1
        seqs = [t, xc, xh, m, a, p]
        non_seqs = [W_l_s, W_r_s, W_e_s, b_s, W_i_t, W_h_t, b_t]
        
        # The first element is the initial state for 2 reasons:
        # 1. if the first mask is 0, there should be a previous state.
        # 2. it can be used for empty element in tracking lstm
        c_stack_init = T.zeros([seq_len + 1, batch_size, self.num_units_stack])
        h_stack_init = T.zeros([seq_len + 1, batch_size, self.num_units_stack])
        c_track_init = T.zeros([batch_size, self.num_units_track])
        h_track_init = T.zeros([batch_size, self.num_units_track])
        init_ = [c_stack_init, h_stack_init, c_track_init, h_track_init]

        c_s, h_s, c_t, h_t = theano.scan(
            fn = self._step,
            sequences=seqs,
            outputs_info=init_,
            non_sequences=non_seqs,
            strict=True)[0]
        
        c_s, h_s = c_s[-1], h_s[-1]
        
        if self.only_return_final:
            h_s = h_s[-1] # last step and the top of stack
        else:
            h_s = h_s.dimshuffle(1,0,2)[:, 1:, :] # last step

        return h_s, h_t

    def _step(self, t_n, xc_n, xh_n, m_n, a_n, p_n, 
              prev_c_stack, prev_h_stack,
              prev_c_track, prev_h_track, 
              W_l_s, W_r_s, W_e_s, b_s,
              W_i_t, W_h_t, b_t):
    
        s_idx = T.eq(a_n, 0).nonzero()[0] # indices for shift samples
        r_idx = T.eq(a_n, 1).nonzero()[0] # indices for reduce samples

        p_r = p_n[r_idx] # pop indices for reduce samples, n_r * 2

        c_l = prev_c_stack[p_r[:, 0], r_idx]
        c_r = prev_c_stack[p_r[:, 1], r_idx]
        h_l = prev_h_stack[p_r[:, 0], r_idx]
        h_r = prev_h_stack[p_r[:, 1], r_idx]

        
        batch_size, _ = xc_n.shape
        h0 = prev_h_stack[p_n[:, 0], T.arange(batch_size)]
        h1 = prev_h_stack[p_n[:, 1], T.arange(batch_size)]
        track_in = T.concatenate([h0, h1, xh_n], axis=1)


        c_t, h_t = self._track(track_in, prev_c_track, prev_h_track,
                               W_i_t, W_h_t, b_t)
        
        c_s, h_s = self._composite(c_l, h_l, c_r, h_r, h_t[r_idx],
                                   W_l_s, W_r_s, W_e_s, b_s)


        # set the top of the stack
        top_c = xc_n
        top_c = T.set_subtensor(top_c[r_idx], c_s)
        top_h = xh_n
        top_h = T.set_subtensor(top_h[r_idx], h_s)

        # mask
        top_c = T.switch(m_n[:, None], top_c, prev_c_stack[t_n-1])
        top_h = T.switch(m_n[:, None], top_h, prev_h_stack[t_n-1])

        c_s = T.set_subtensor(prev_c_stack[t_n], top_c)
        h_s = T.set_subtensor(prev_h_stack[t_n], top_h)

        return [c_s, h_s, c_t, h_t]
    

    def _composite(self, c_l, h_l, c_r, h_r, e,
                   W_l_s, W_r_s, W_e_s, b_s,
                   ):

        # At each call to scan, input_n will be (n_time_steps, 4*num_units_stack).
        # We define a slicing function that extract the input to each LSTM gate
        def slice_w(x, n):
            return x[:, n*self.num_units_stack:(n+1)*self.num_units_stack]

        gates = T.dot(h_l, W_l_s) + T.dot(h_r, W_r_s) + T.dot(e, W_e_s) + b_s

        if self.grad_clipping:
            gates = theano.gradient.grad_clip(
                gates, -self.grad_clipping, self.grad_clipping)

        lg_s = slice_w(gates, 0)
        rg_s = slice_w(gates, 1)
        ig_s = slice_w(gates, 2)
        og_s = slice_w(gates, 3)
        c_s_input = slice_w(gates, 4)

        lg_s = self.nonlin_lg_s(lg_s)
        rg_s = self.nonlin_rg_s(rg_s)
        ig_s = self.nonlin_ig_s(ig_s)
        og_s = self.nonlin_og_s(og_s)
        c_s_input = self.nonlin_c_s(c_s_input)

        c = lg_s*c_l + rg_s*c_r + ig_s*c_s_input
        h = og_s * self.nonlin_s(c)

        return [c, h]
 

    def _track(self, i_n, prev_c, prev_h, W_i_t, W_h_t, b_t):
        # At each call to scan, input_n will be (n_time_steps, 4*num_units_stack).
        # We define a slicing function that extract the input to each LSTM gate
        def slice_w(x, n):
            return x[:, n*self.num_units_track:(n+1)*self.num_units_track]
        

        gates = T.dot(i_n, W_i_t) + T.dot(prev_h, W_h_t) + b_t

        if self.grad_clipping:
            gates = theano.gradient.grad_clip(
                gates, -self.grad_clipping, self.grad_clipping)

        ig_t = slice_w(gates, 0)
        fg_t = slice_w(gates, 1)
        og_t = slice_w(gates, 2)
        c_t_input = slice_w(gates, 3)

        ig_t = self.nonlin_ig_t(ig_t)
        fg_t = self.nonlin_fg_t(fg_t)
        og_t = self.nonlin_og_t(og_t)
        c_t_input = self.nonlin_c_t(c_t_input)

        c = ig_t*c_t_input + fg_t*prev_c
        h = og_t * self.nonlin_t(c)

        return [c, h]
       

