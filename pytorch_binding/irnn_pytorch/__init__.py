import math
import torch
import warnings

from torch.nn import Parameter
from torch.nn.utils.rnn import PackedSequence


from torch.autograd import Function
from torch.autograd import Variable
from torch.nn import Module
from torch.utils.ffi import _wrap_function
import irnn_pytorch as irnn
from ._irnn import lib as _lib, ffi as _ffi

import numpy as np

__all__ = []

def _import_symbols(locals):
    for symbol in dir(_lib):
        try:
            fn = getattr(_lib, symbol)
            locals[symbol] = _wrap_function(fn, _ffi)
            __all__.append(symbol)
        except AttributeError as e:
            # since fn is not "built-in function", we should not wrap them.
            pass
        except:
            # Oops!
            pass

_import_symbols(locals())


def get_workspace_size(mode, train, input_size, hidden_size, seq_length, 
                       batch_size, bidirectional, num_layer):
    pytorch_get_workspace_size = irnn.pytorch_get_workspace_size
    if mode == 'RNN_TANH' or mode == 'RNN_RELU':
        mode_int = 1
    elif mode == 'LSTM':
        mode_int = 2
    elif mode == 'GRU':
        mode_int = 3
    if train:
        train_int = 1
    else:
        train_int = 2

    if bidirectional is True:
        bidirectional_int = 1
    else:
        bidirectional_int = 0
    workspace_size = pytorch_get_workspace_size(mode_int, train_int, input_size, 
      hidden_size, seq_length, batch_size, bidirectional_int, num_layer)
    #print("workspace_size = %d"% workspace_size)
    return workspace_size

class IRNNFunc(Function):
    def __init__(self,mode,train,num_layers,batch_sizes,seq_length,input_size,hidden_size,batch_first,dropout,bidirectional,
                 dropout_state):
        super(IRNNFunc, self).__init__()
        self.mode = mode
        self.train = train
        self.num_layers = num_layers
        self.batch_sizes = batch_sizes
        self.seq_length = seq_length
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_first = batch_first
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.dropout_state = dropout_state
    def forward(self,input,w_x,w_h,bx,bh,h0,c0,workspace):
        # call rinn/lstm/gru with inferecen/forward
        self.workspace = workspace
        self.input = input
        self.w_x = w_x
        self.w_h = w_h
        self.bx = bx
        self.bh = bh
        self.h0 = h0
        self.c0 = c0

        num_directions = 2 if self.bidirectional else 1
        self.h_state = torch.Tensor(self.seq_length, self.batch_sizes, self.hidden_size * num_directions)
        self.hy = torch.Tensor(self.num_layers*num_directions, self.batch_sizes, self.hidden_size )
        self.c_state = torch.Tensor(self.seq_length, self.batch_sizes, self.hidden_size * num_directions) #if self.mode == 'LSTM' else None
        self.cy = torch.Tensor(self.num_layers*num_directions, self.batch_sizes, self.hidden_size ) #if self.mode == 'LSTM' else None
        
        bias = bx + bh

        if self.mode == 'RNN_TANH' or self.mode == 'RNN_RELU':
            tanh = True if self.mode == 'RNN_TANH' else False
            if self.train:
                rnn_forward = irnn.pytorch_rnn_forward
                # rnn_forward(input_size,hidden_size,num_layers,batch_first,dropout,bidirectional,batch_sizes,dropout_state,
                #            workspace,input,w_x,w_h,bias,h0,h_out,tanh)
                rnn_forward(workspace, input.data, w_x, w_h, bias, h0.data, self.h_state.data,
                            self.batch_sizes, self.seq_length, self.input_size, self.hidden_size, self.num_layers)
            else:
                rnn_inference = irnn.pytorch_rnn_inference
        elif self.mode == 'LSTM':
            if self.train:
                lstm_forward = irnn.pytorch_lstm_forward
                lstm_forward(workspace, input, w_x, w_h, bias, h0, c0, self.h_state, self.c_state,self.hy,self.cy,
                             self.batch_sizes, self.seq_length, self.input_size, self.hidden_size, self.num_layers, self.bidirectional)

            else:
                lstm_inference = irnn.pytorch_lstm_inference
                #lstm_inference(self.input_size, self.hidden_size, self.num_layers, self.batch_first, self.dropout,
                #               self.bidirectional, self.batch_sizes,self.dropout_state,
                #               workspace, input, w_x, w_h, bias, h0, c0, self.h_state, self.c_state)
                #             workspace,input,w_x,w_h,bias,h0,c0,h_out,c_out)
                lstm_inference(self.workspace, input, w_x, w_h, bias, h0, c0, self.h_state, self.c_state,
                               self.batch_sizes, self.seq_length, self.input_size, self.hidden_size, self.num_layers)
        elif self.mode == 'GRU':
            if self.train:
                gru_forward = irnn.pytorch_gru_forward
                gru_forward(workspace, input, w_x, w_h, bx, bh, h0, self.h_state, 
                  self.hy, self.batch_sizes, self.seq_length, self.input_size, 
                  self.hidden_size, self.num_layers, num_directions)
            else:
                gru_inference = irnn.pytorch_gru_infer
                gru_inference(workspace, input, w_x, w_h, bx, bh, h0, self.h_state, 
                  self.hy, self.batch_sizes, self.seq_length, self.input_size, 
                  self.hidden_size, self.num_layers, num_directions)
            #self.hy[0] = self.h_state[self.seq_length-1]

        return self.h_state,self.hy,self.c_state,self.cy

    def backward(self, grad_h_state, grad_hn, grad_c_state, grad_cn):
        #print("grad_h_out sum = ", grad_h_out.sum())
        num_directions = 2 if self.bidirectional else 1
        
        if self.mode == 'RNN_TANH' or self.mode == 'RNN_RELU':
            tanh = True if self.mode == 'RNN_TANH' else False
            print("RNN backward")
        elif self.mode == 'LSTM':
            lstm_backward = irnn.pytorch_lstm_backward
            self.grad_x = torch.Tensor().resize_as_(self.input)
            self.grad_h0 = torch.Tensor().resize_as_(self.h0)
            self.grad_c0 = torch.Tensor().resize_as_(self.h0)
            self.grad_wx = torch.Tensor().resize_as_(self.w_x)
            self.grad_wh = torch.Tensor().resize_as_(self.w_h)
            self.grad_bias = torch.Tensor().resize_as_(self.bx)
            lstm_backward(self.num_layers,num_directions,self.seq_length,self.batch_sizes,self.input_size,self.hidden_size,
                self.workspace,self.input,self.h0,self.c0,self.h_state,self.c_state,self.w_x,self.w_h,grad_hn,grad_cn,self.grad_x,
                          self.grad_h0,self.grad_c0,self.grad_wx,self.grad_wh,self.grad_bias)
            self.grad_bx = self.grad_bias
            self.grad_bh = self.grad_bias
            
        elif self.mode == 'GRU':
            gru_backward = irnn.pytorch_gru_backward
            self.grad_x = torch.Tensor().resize_as_(self.input)
            self.grad_h0 = torch.Tensor().resize_as_(self.h0)
            self.grad_c0 = torch.Tensor().resize_as_(self.h0)
            self.grad_wx = torch.Tensor().resize_as_(self.w_x)
            self.grad_wh = torch.Tensor().resize_as_(self.w_h)
            self.grad_bx = torch.Tensor().resize_as_(self.bx)
            self.grad_bh = torch.Tensor().resize_as_(self.bh)
            gru_backward(self.num_layers,num_directions,self.seq_length,self.batch_sizes,self.input_size,self.hidden_size,
                self.workspace,self.input,self.h0,self.h_state,self.w_x,self.w_h,self.bx,self.bh,grad_h_state,grad_hn,self.grad_x,
                          self.grad_h0,self.grad_wx,self.grad_wh,self.grad_bx,self.grad_bh)
            self.grad_c0 = self.grad_h0

        return self.grad_x,self.grad_wx,self.grad_wh,self.grad_bx,self.grad_bh,self.grad_h0,self.grad_c0,None,None,None



class IRNNBase(Module):
    def __init__(self, mode, input_size, hidden_size,
                 num_layers=1, bias=True, batch_first=False,
                 dropout=0, bidirectional=False):
        super(IRNNBase, self).__init__()
        self.mode = mode               #model = RNN/LSTM/GRU
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout
        self.dropout_state = {}
        self.bidirectional = bidirectional
        num_directions = 2 if bidirectional else 1
        '''add for workspace update'''
        self.max_seq_length = 1
        self.max_batch_size = 1
        self.update_workspace = True
        self.workspace = torch.Tensor()

        if mode == 'LSTM':
            gate_size = 4 * hidden_size
        elif mode == 'GRU':
            gate_size = 3 * hidden_size
        else:
            gate_size = hidden_size

        self._all_weights = []
        #for layer in range(num_layers):
        #    for direction in range(num_directions):
        layer_input_size = input_size
        '''change 4H x H to H x 4H
        parameter will be double if bidirection is true(num_direction = 2)'''
        w_ih = Parameter(torch.Tensor(num_directions, layer_input_size,gate_size))
        w_hh = Parameter(torch.Tensor(num_directions, hidden_size,gate_size))
        b_ih = Parameter(torch.Tensor(gate_size * num_layers * num_directions))
        b_hh = Parameter(torch.Tensor(gate_size * num_layers * num_directions))
        # w_ih = Parameter(torch.from_numpy(np.full((num_directions, layer_input_size,gate_size), 0.1, dtype=np.float32)))
        # w_hh = Parameter(torch.from_numpy(np.full((num_directions, hidden_size,gate_size), 0.1, dtype=np.float32)))
        # b_ih = Parameter(torch.from_numpy(np.full((gate_size * num_layers * num_directions), 0.1, dtype=np.float32)))
        # b_hh = Parameter(torch.from_numpy(np.full((gate_size * num_layers * num_directions), 0.1, dtype=np.float32)))

        layer_params = (w_ih, w_hh, b_ih, b_hh)

        suffix = '_reverse' if num_directions == 1 else ''
        param_names = ['weight_ih_l{}{}', 'weight_hh_l{}{}']
        if bias:
            param_names += ['bias_ih_l{}{}', 'bias_hh_l{}{}']
        param_names = [x.format(0, suffix) for x in param_names]

        for name, param in zip(param_names, layer_params):
            setattr(self, name, param)
        self._all_weights.append(param_names)

        self.flatten_parameters()  #this works only on GPU
        self.reset_parameters()

    def flatten_parameters(self):
        """Resets parameter data pointer so that they can use faster code paths.

        Right now, this works only if the module is on the GPU and cuDNN is enabled.
        Otherwise, it's a no-op.
        """

    def _apply(self, fn):
        ret = super(IRNNBase, self)._apply(fn)
        self.flatten_parameters()
        return ret

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, input, hx=None):
        is_packed = isinstance(input, PackedSequence)
        '''if packed, input contains max_batch_size information'''
        if is_packed:
            input, batch_sizes = input
            batch_size = batch_sizes[0]
        else:
            batch_sizes = None
            batch_size = input.size(0) if self.batch_first else input.size(1)

        '''if user don't provide the h0 and c0, a zero tensor will be created.'''
        if hx is None:
            num_directions = 2 if self.bidirectional else 1
            hx = torch.autograd.Variable(input.data.new(self.num_layers *
                                                        num_directions,
                                                        batch_size,
                                                        self.hidden_size).zero_(), requires_grad=False)
            if self.mode == 'LSTM': #LSTM requires a tuple in h0
                hx = (hx, hx)

        has_flat_weights = None #= list(p.data.data_ptr() for p in self.parameters()) == self._data_ptrs
        '''TODO: add assert to avoid shape mismatch'''

        #get all weight from self.parameters()
        seq_length = input.size(0)
        weight_idx = 1
        bx = None
        bh = None
        for weight in self.parameters():
            if weight_idx == 1:
                w_x = weight
            elif weight_idx == 2:
                w_h = weight
            elif weight_idx == 3:
                bx = weight
            elif weight_idx == 4:
                bh = weight
            weight_idx = weight_idx + 1

        # Pytorch makes the assumption that all parameters passed to Function.forward() must
        # be an instance of "Variable", and it doesn't accept NoneType. Make it happy.
        if bx is None:
            bx = Variable(torch.Tensor((0)))
        if bh is None:
            bh = Variable(torch.Tensor((0)))
        #check if input seq_length or batch_size exceed the max value in self.
        if seq_length > self.max_seq_length :
            self.max_seq_length = seq_length
            self.update_workspace = True
        if batch_size > self.max_batch_size:
            self.max_batch_size = batch_size
            self.update_workspace = True
        #update workspace in the first call, or exceed happens
        if self.update_workspace:
            #print("Updating the workspace ...")
            buffer_size = get_workspace_size(self.mode, self.training, self.input_size, self.hidden_size,
                                             self.max_seq_length, self.max_batch_size, self.bidirectional, self.num_layers)
            self.workspace = Variable(torch.zeros(buffer_size), requires_grad=False)
            self.update_workspace = False


        h0 = hx[0] if self.mode == 'LSTM' else hx
        c0 = hx[1] if self.mode == 'LSTM' else Variable(torch.Tensor())

        _func = IRNNFunc(self.mode,self.training,self.num_layers,batch_size,seq_length,self.input_size,self.hidden_size,
                         self.batch_first,self.dropout,self.bidirectional,self.dropout_state)

        self.h_state, self.hy, self.c_state, self.cy = _func(input, w_x, w_h, bx, bh, h0, c0, self.workspace)
        self.h_out = (self.hy,self.cy) if self.mode=='LSTM' else self.hy
        if is_packed:
            output = PackedSequence(self.h_state, batch_sizes)
        return self.h_state, self.h_out

    def __repr__(self):
        s = '{name}({input_size}, {hidden_size}'
        if self.num_layers != 1:
            s += ', num_layers={num_layers}'
        if self.bias is not True:
            s += ', bias={bias}'
        if self.batch_first is not False:
            s += ', batch_first={batch_first}'
        if self.dropout != 0:
            s += ', dropout={dropout}'
        if self.bidirectional is not False:
            s += ', bidirectional={bidirectional}'
        s += ')'
        return s.format(name=self.__class__.__name__, **self.__dict__)

    '''QUESTION: when __setstate__ is called ?'''
    def __setstate__(self, d):
        super(IRNNBase, self).__setstate__(d)
        self.__dict__.setdefault('_data_ptrs', [])
        if 'all_weights' in d:
            self._all_weights = d['all_weights']
        if isinstance(self._all_weights[0][0], str):
            return
        num_layers = self.num_layers
        num_directions = 2 if self.bidirectional else 1
        '''QUESTION: self._all_weights is a list, contains all weight?'''
        self._all_weights = []
        for layer in range(num_layers):
            for direction in range(num_directions):
                suffix = '_reverse' if direction == 1 else ''
                weights = ['weight_ih_l{}{}', 'weight_hh_l{}{}', 'bias_ih_l{}{}', 'bias_hh_l{}{}']
                weights = [x.format(layer, suffix) for x in weights]
                if self.bx and self.bh:
                    self._all_weights += [weights]
                else:
                    self._all_weights += [weights[:2]]

    @property
    def all_weights(self):
        return [[getattr(self, weight) for weight in weights] for weights in self._all_weights]


class RNN(IRNNBase):
    r"""Applies a multi-layer Elman RNN with tanh or ReLU non-linearity to an
    input sequence.


    For each element in the input sequence, each layer computes the following
    function:

    .. math::

        h_t = \tanh(w_{ih} * x_t + b_{ih}  +  w_{hh} * h_{(t-1)} + b_{hh})

    where :math:`h_t` is the hidden state at time `t`, and :math:`x_t` is
    the hidden state of the previous layer at time `t` or :math:`input_t`
    for the first layer. If nonlinearity='relu', then `ReLU` is used instead
    of `tanh`.

    Args:
        input_size: The number of expected features in the input x
        hidden_size: The number of features in the hidden state h
        num_layers: Number of recurrent layers.
        nonlinearity: The non-linearity to use ['tanh'|'relu']. Default: 'tanh'
        bias: If ``False``, then the layer does not use bias weights b_ih and b_hh.
            Default: ``True``
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature)
        dropout: If non-zero, introduces a dropout layer on the outputs of each
            RNN layer except the last layer
        bidirectional: If ``True``, becomes a bidirectional RNN. Default: ``False``

    Inputs: input, h_0
        - **input** (seq_len, batch, input_size): tensor containing the features
          of the input sequence. The input can also be a packed variable length
          sequence. See :func:`torch.nn.utils.rnn.pack_padded_sequence`
          for details.
        - **h_0** (num_layers * num_directions, batch, hidden_size): tensor
          containing the initial hidden state for each element in the batch.

    Outputs: output, h_n
        - **output** (seq_len, batch, hidden_size * num_directions): tensor
          containing the output features (h_k) from the last layer of the RNN,
          for each k.  If a :class:`torch.nn.utils.rnn.PackedSequence` has
          been given as the input, the output will also be a packed sequence.
        - **h_n** (num_layers * num_directions, batch, hidden_size): tensor
          containing the hidden state for k=seq_len.

    Attributes:
        weight_ih_l[k]: the learnable input-hidden weights of the k-th layer,
            of shape `(input_size x hidden_size)`
        weight_hh_l[k]: the learnable hidden-hidden weights of the k-th layer,
            of shape `(hidden_size x hidden_size)`
        bias_ih_l[k]: the learnable input-hidden bias of the k-th layer,
            of shape `(hidden_size)`
        bias_hh_l[k]: the learnable hidden-hidden bias of the k-th layer,
            of shape `(hidden_size)`

    Examples::

        >>> rnn = nn.RNN(10, 20, 2)
        >>> input = Variable(torch.randn(5, 3, 10))
        >>> h0 = Variable(torch.randn(2, 3, 20))
        >>> output, hn = rnn(input, h0)
    """

    def __init__(self, *args, **kwargs):
        if 'nonlinearity' in kwargs:
            if kwargs['nonlinearity'] == 'tanh':
                mode = 'RNN_TANH'
            elif kwargs['nonlinearity'] == 'relu':
                mode = 'RNN_RELU'
            else:
                raise ValueError("Unknown nonlinearity '{}'".format(
                    kwargs['nonlinearity']))
            del kwargs['nonlinearity']
        else:
            mode = 'RNN_TANH'

        super(RNN, self).__init__(mode, *args, **kwargs)


class LSTM(IRNNBase):
    r"""Applies a multi-layer long short-term memory (LSTM) RNN to an input
    sequence.


    For each element in the input sequence, each layer computes the following
    function:

    .. math::

            \begin{array}{ll}
            i_t = \mathrm{sigmoid}(W_{ii} x_t + b_{ii} + W_{hi} h_{(t-1)} + b_{hi}) \\
            f_t = \mathrm{sigmoid}(W_{if} x_t + b_{if} + W_{hf} h_{(t-1)} + b_{hf}) \\
            g_t = \tanh(W_{ig} x_t + b_{ig} + W_{hc} h_{(t-1)} + b_{hg}) \\
            o_t = \mathrm{sigmoid}(W_{io} x_t + b_{io} + W_{ho} h_{(t-1)} + b_{ho}) \\
            c_t = f_t * c_{(t-1)} + i_t * g_t \\
            h_t = o_t * \tanh(c_t)
            \end{array}

    where :math:`h_t` is the hidden state at time `t`, :math:`c_t` is the cell
    state at time `t`, :math:`x_t` is the hidden state of the previous layer at
    time `t` or :math:`input_t` for the first layer, and :math:`i_t`,
    :math:`f_t`, :math:`g_t`, :math:`o_t` are the input, forget, cell,
    and out gates, respectively.

    Args:
        input_size: The number of expected features in the input x
        hidden_size: The number of features in the hidden state h
        num_layers: Number of recurrent layers.
        bias: If ``False``, then the layer does not use bias weights b_ih and b_hh.
            Default: ``True``
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature)
        dropout: If non-zero, introduces a dropout layer on the outputs of each
            RNN layer except the last layer
        bidirectional: If ``True``, becomes a bidirectional RNN. Default: ``False``

    Inputs: input, (h_0, c_0)
        - **input** (seq_len, batch, input_size): tensor containing the features
          of the input sequence.
          The input can also be a packed variable length sequence.
          See :func:`torch.nn.utils.rnn.pack_padded_sequence` for details.
        - **h_0** (num_layers \* num_directions, batch, hidden_size): tensor
          containing the initial hidden state for each element in the batch.
        - **c_0** (num_layers \* num_directions, batch, hidden_size): tensor
          containing the initial cell state for each element in the batch.


    Outputs: output, (h_n, c_n)
        - **output** (seq_len, batch, hidden_size * num_directions): tensor
          containing the output features `(h_t)` from the last layer of the RNN,
          for each t. If a :class:`torch.nn.utils.rnn.PackedSequence` has been
          given as the input, the output will also be a packed sequence.
        - **h_n** (num_layers * num_directions, batch, hidden_size): tensor
          containing the hidden state for t=seq_len
        - **c_n** (num_layers * num_directions, batch, hidden_size): tensor
          containing the cell state for t=seq_len

    Attributes:
        weight_ih_l[k] : the learnable input-hidden weights of the k-th layer
            `(W_ii|W_if|W_ig|W_io)`, of shape `(4*hidden_size x input_size)`
        weight_hh_l[k] : the learnable hidden-hidden weights of the k-th layer
            `(W_hi|W_hf|W_hg|W_ho)`, of shape `(4*hidden_size x hidden_size)`
        bias_ih_l[k] : the learnable input-hidden bias of the k-th layer
            `(b_ii|b_if|b_ig|b_io)`, of shape `(4*hidden_size)`
        bias_hh_l[k] : the learnable hidden-hidden bias of the k-th layer
            `(b_hi|b_hf|b_hg|b_ho)`, of shape `(4*hidden_size)`

    Examples::

        >>> rnn = nn.LSTM(10, 20, 2)
        >>> input = Variable(torch.randn(5, 3, 10))
        >>> h0 = Variable(torch.randn(2, 3, 20))
        >>> c0 = Variable(torch.randn(2, 3, 20))
        >>> output, hn = rnn(input, (h0, c0))
    """

    def __init__(self, *args, **kwargs):
        super(LSTM, self).__init__('LSTM', *args, **kwargs)


class GRU(IRNNBase):
    r"""Applies a multi-layer gated recurrent unit (GRU) RNN to an input sequence.


    For each element in the input sequence, each layer computes the following
    function:

    .. math::

            \begin{array}{ll}
            r_t = \mathrm{sigmoid}(W_{ir} x_t + b_{ir} + W_{hr} h_{(t-1)} + b_{hr}) \\
            z_t = \mathrm{sigmoid}(W_{iz} x_t + b_{iz} + W_{hz} h_{(t-1)} + b_{hz}) \\
            n_t = \tanh(W_{in} x_t + b_{in} + r_t * (W_{hn} h_{(t-1)}+ b_{hn})) \\
            h_t = (1 - z_t) * n_t + z_t * h_{(t-1)} \\
            \end{array}

    where :math:`h_t` is the hidden state at time `t`, :math:`x_t` is the hidden
    state of the previous layer at time `t` or :math:`input_t` for the first
    layer, and :math:`r_t`, :math:`z_t`, :math:`n_t` are the reset, input,
    and new gates, respectively.

    Args:
        input_size: The number of expected features in the input x
        hidden_size: The number of features in the hidden state h
        num_layers: Number of recurrent layers.
        bias: If ``False``, then the layer does not use bias weights b_ih and b_hh.
            Default: ``True``
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature)
        dropout: If non-zero, introduces a dropout layer on the outputs of each
            RNN layer except the last layer
        bidirectional: If ``True``, becomes a bidirectional RNN. Default: ``False``

    Inputs: input, h_0
        - **input** (seq_len, batch, input_size): tensor containing the features
          of the input sequence. The input can also be a packed variable length
          sequence. See :func:`torch.nn.utils.rnn.pack_padded_sequence`
          for details.
        - **h_0** (num_layers * num_directions, batch, hidden_size): tensor
          containing the initial hidden state for each element in the batch.

    Outputs: output, h_n
        - **output** (seq_len, batch, hidden_size * num_directions): tensor
          containing the output features h_t from the last layer of the RNN,
          for each t. If a :class:`torch.nn.utils.rnn.PackedSequence` has been
          given as the input, the output will also be a packed sequence.
        - **h_n** (num_layers * num_directions, batch, hidden_size): tensor
          containing the hidden state for t=seq_len

    Attributes:
        weight_ih_l[k] : the learnable input-hidden weights of the k-th layer
            (W_ir|W_iz|W_in), of shape `(3*hidden_size x input_size)`
        weight_hh_l[k] : the learnable hidden-hidden weights of the k-th layer
            (W_hr|W_hz|W_hn), of shape `(3*hidden_size x hidden_size)`
        bias_ih_l[k] : the learnable input-hidden bias of the k-th layer
            (b_ir|b_iz|b_in), of shape `(3*hidden_size)`
        bias_hh_l[k] : the learnable hidden-hidden bias of the k-th layer
            (b_hr|b_hz|b_hn), of shape `(3*hidden_size)`
    Examples::

        >>> rnn = nn.GRU(10, 20, 2)
        >>> input = Variable(torch.randn(5, 3, 10))
        >>> h0 = Variable(torch.randn(2, 3, 20))
        >>> output, hn = rnn(input, h0)
    """

    def __init__(self, *args, **kwargs):
        super(GRU, self).__init__('GRU', *args, **kwargs)
