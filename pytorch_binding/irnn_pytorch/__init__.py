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


def get_workspace_size(mode, train, L, bidirectional , T, N, I, H):
    pytorch_get_workspace_size = irnn.pytorch_get_workspace_size
    if mode == 'RNN_TANH' or mode == 'RNN_RELU':
        mode_int = 1
    elif mode == 'LSTM':
        mode_int = 2
    elif mode == 'GRU':
        mode_int = 3
    train_int = 1 if train else 2
    D = 2 if bidirectional else 1
    workspace_size = pytorch_get_workspace_size(mode_int, train_int, int(L), int(D), int(T), int(N), int(I), int(H))
    return workspace_size

class GRUFunc(Function):
    def __init__(self,mode,train,num_layers,input_size,hidden_size,batch_first,dropout,bidirectional,
                 dropout_state):
        super(GRUFunc, self).__init__()
        self.mode = mode
        self.train = train
        self.L = num_layers
        self.I = input_size
        self.H = hidden_size
        self.batch_first = batch_first
        self.dropout = dropout
        self.D = 2 if bidirectional else 1
        self.dropout_state = dropout_state
    def forward(self, ws, x, hx, wx, wh, bx, bh):
        self.T = int(x.size(0))
        self.N = int(x.size(1))
        self.ws = ws
        self.x  = x
        self.hx = hx
        self.wx = wx
        self.wh = wh
        self.bx = bx
        self.bh = bh

        self.y = torch.Tensor(self.T, self.N, self.H * self.D)
        self.hy = torch.Tensor(self.L * self.D, self.N, self.H)
        
        if self.train:
            gru_forward = irnn.pytorch_gru_forward
            gru_forward(self.L, self.D, self.T, self.N, self.I, self.H,
                ws, x, hx, wx, wh, bx, bh, self.y, self.hy)
        else:
            gru_inference = irnn.pytorch_gru_infer
            gru_inference(self.L, self.D, self.T, self.N, self.I, self.H,
                ws, x, hx, wx, wh, bx, bh, self.y, self.hy)

        return self.y,self.hy

    def backward(self, dy, dhy):
        
        gru_backward = irnn.pytorch_gru_backward
        self.dx = torch.Tensor().resize_as_(self.x)
        self.dhx = torch.Tensor().resize_as_(self.hx)
        self.dwx = torch.Tensor().resize_as_(self.wx)
        self.dwh = torch.Tensor().resize_as_(self.wh)
        self.dbx = torch.Tensor().resize_as_(self.bx)
        self.dbh = torch.Tensor().resize_as_(self.bh)
        gru_backward(self.L, self.D, self.T, self.N, self.I, self.H,
            self.ws, self.x, self.hx, self.wx, self.wh, self.dx, self.dhx,
            self.dwx, self.dwh, self.dbx, self.dbh, dy, dhy)

        return None, self.dx, self.dhx, self.dwx, self.dwh, self.dbx, self.dbh

class LSTMFunc(Function):
    def __init__(self,mode,train,num_layers,input_size,hidden_size,batch_first,dropout,bidirectional,
                 dropout_state):
        super(LSTMFunc, self).__init__()
        self.mode = mode
        self.train = train
        self.L = num_layers
        self.I = input_size
        self.H = hidden_size
        self.batch_first = batch_first
        self.dropout = dropout
        self.D = 2 if bidirectional else 1
        self.dropout_state = dropout_state
    def forward(self, ws, x, hx, cx, wx, wh, bx, bh):
        #get seq_length from input
        self.T = int(x.size(0))
        self.N = int(x.size(1))

        self.ws = ws
        self.x  = x
        self.hx = hx
        self.cx = cx
        self.wx = wx
        self.wh = wh
        self.bx = bx
        self.bh = bh

        self.y = torch.Tensor(self.T, self.N, self.H * self.D)
        self.hy = torch.Tensor(self.L * self.D, self.N, self.H)
        self.cy = torch.Tensor(self.L * self.D, self.N, self.H)
        
        bias = bx + bh

        if self.train:
            lstm_forward = irnn.pytorch_lstm_forward
            lstm_forward(self.L, self.D, self.T, self.N, self.I, self.H,
                ws, x, hx, cx, wx, wh, bias, self.y, self.hy, self.cy)
        else:
            lstm_inference = irnn.pytorch_lstm_inference
            lstm_inference(self.L, self.D, self.T, self.N, self.I, self.H,
                ws, x, hx, cx, wx, wh, bias, self.y, self.hy, self.cy)

        return self.y,self.hy,self.cy

    def backward(self, dy, dhy, dcy):
        #print("    dy size = ", dy.size(), dy.sum())  
        #print("    dhy size = ", dhy.size(), dhy.sum())                 
        #print("    dcy size = ", dcy.size(), dcy.sum())  
        
        lstm_backward = irnn.pytorch_lstm_backward
        self.dx = torch.Tensor().resize_as_(self.x)
        self.dhx = torch.Tensor().resize_as_(self.hx)
        self.dcx = torch.Tensor().resize_as_(self.cx)
        self.dwx = torch.Tensor().resize_as_(self.wx)
        self.dwh = torch.Tensor().resize_as_(self.wh)
        self.dbias = torch.Tensor().resize_as_(self.bx)
        lstm_backward(self.L, self.D, self.T, self.N, self.I, self.H,
            self.ws, self.x, self.hx, self.cx, self.wx, self.wh, self.y,
            self.hy, self.cy, self.dx, self.dhx, self.dcx, self.dwx,self.dwh,
            self.dbias, dy, dhy, dcy)
        self.dbx = self.dbias
        self.dbh = self.dbias
            
        return None, self.dx, self.dhx, self.dcx, self.dwx, self.dwh, self.dbx, self.dbh


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

        # create IRNNFunction
        if mode == 'LSTM':
            self.IRNNFunc = LSTMFunc(self.mode,self.training,self.num_layers,self.input_size,self.hidden_size,
                self.batch_first,self.dropout,self.bidirectional,self.dropout_state)
        elif mode == 'GRU':
            self.IRNNFunc = GRUFunc(self.mode,self.training,self.num_layers,self.input_size,self.hidden_size,
                         self.batch_first,self.dropout,self.bidirectional,self.dropout_state)
        


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

        '''if user don't provide the hx and cx, a zero tensor will be created.'''
        if hx is None:
            num_directions = 2 if self.bidirectional else 1
            hx = torch.autograd.Variable(input.data.new(self.num_layers *
                                                        num_directions,
                                                        batch_size,
                                                        self.hidden_size).zero_(), requires_grad=False)
            if self.mode == 'LSTM': #LSTM requires a tuple in hx
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
                wx = weight
            elif weight_idx == 2:
                wh = weight
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
            buffer_size = get_workspace_size(self.mode, self.training, self.num_layers,
                self.bidirectional, self.max_seq_length, self.max_batch_size,
                self.input_size, self.hidden_size)
            self.workspace = Variable(torch.zeros(buffer_size), requires_grad=False)
            self.update_workspace = False

        _func = self.IRNNFunc

        if self.mode == 'LSTM':
            cx = hx[1]
            hx = hx[0]
            self.y, self.hy, self.cy = _func(self.workspace, input, hx, cx, wx, wh, bx, bh)
            if is_packed:
                output = PackedSequence(self.y, batch_sizes)
            return self.y, (self.hy, self.cy)
        elif self.mode == 'GRU':
            self.y, self.hy = _func(self.workspace, input, hx, wx, wh, bx, bh)
            if is_packed:
                output = PackedSequence(self.y, batch_sizes)
            return self.y, self.hy


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
        >>> hx = Variable(torch.randn(2, 3, 20))
        >>> output, hn = rnn(input, hx)
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
        >>> hx = Variable(torch.randn(2, 3, 20))
        >>> cx = Variable(torch.randn(2, 3, 20))
        >>> output, hn = rnn(input, (hx, cx))
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
        >>> hx = Variable(torch.randn(2, 3, 20))
        >>> output, hn = rnn(input, hx)
    """

    def __init__(self, *args, **kwargs):
        super(GRU, self).__init__('GRU', *args, **kwargs)
