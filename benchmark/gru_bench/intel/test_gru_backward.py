import time
import theano
import numpy as np
import optparse

from theano import tensor as T
from theano.tensor.nnet import sigmoid
from theano.tensor import tanh

from mkl_gru import GRU
from mkl_gru_inference import GRU_Inference


# Parameters
optparser = optparse.OptionParser()
optparser.add_option("-n", "--network_type", default='gru', help="Network type (gru)")
optparser.add_option("-i", "--input_size", default=150, type='int', help="Input layer size")
optparser.add_option("-l", "--hidden_size", default=1024, type='int', help="Hidden layer size")
optparser.add_option("-s", "--seq_length", default=10, type='int', help="Sequence length")
optparser.add_option("-b", "--batch_size", default=64, type='int', help="Batch size")
opts = optparser.parse_args()[0]

network_type = opts.network_type
print(network_type)

if network_type != 'gru':
    raise Exception('Only support GRU now! '+network_type)

hidden_size = opts.hidden_size
input_size = opts.input_size
seq_length = opts.seq_length
batch_size = opts.batch_size

np.random.seed(123)
nbatch = 100

def GRU_MKL():
    x = T.ftensor3('X')
    y = T.fmatrix('Y')
    wx = theano.shared(np.random.rand(3 * hidden_size, input_size).astype(np.float32))
    wh = theano.shared(np.random.rand(3 * hidden_size, hidden_size).astype(np.float32))
    hs = theano.shared(np.zeros((hidden_size, batch_size), np.float32))

    # inference model
    out = GRU_Inference(hid=hidden_size, return_sequences=True, max_len=seq_length)(x, wx, wh, hs)
    fwd = theano.function(inputs=[x], outputs=out[-1])
    # theano.printing.pydotprint(fwd, outfile='gru_inference.png', var_with_name_simple=True)

    # training model
    out = GRU(hid=hidden_size, return_sequences=True, max_len=seq_length)(x, wx, wh, hs)
    cost = T.sum((out[0][-1,:,:] - y) ** 2)
    gi, gwx, gwh, ghs = theano.grad(cost, [x, wx, wh, hs])

    updates = [(wx, wx - 0.01 * gwx),
               (wh, wh - 0.01 * gwh),
               (hs, hs - 0.01 * ghs)]

    fbwd = theano.function(inputs=[x, y],
                          outputs=cost,
                          updates=updates
                          )
    # theano.printing.pydotprint(bwd, outfile='gru_training.png', var_with_name_simple=True)

    return fwd, fbwd


if __name__ == '__main__':
    fwd, fbwd = GRU_MKL()

    xinput = np.random.rand(seq_length, input_size, batch_size).astype(np.float32)
    ytarget = np.random.rand(hidden_size, batch_size).astype(np.float32)
    
    fwd(xinput)

    tic = time.time()
    for i in xrange(0, nbatch):
        fwd(xinput)
    toc = time.time()

    nsamples = nbatch * batch_size
    print("Forward: %s " % batch_size)
    print("--- %d samples in %s seconds (%f samples/s, %.7f s/sample) ---" %(nsamples, toc - tic, nsamples / (toc - tic), (toc - tic) / nsamples))

    fbwd(xinput, ytarget)

    tic = time.time()
    for i in xrange(0, nbatch):
        fbwd(xinput, ytarget)
    toc = time.time()

    nsamples = nbatch * batch_size
    print("Forward + Backward: %s " % batch_size)
    print("--- %d samples in %s seconds (%f samples/s, %.7f s/sample) ---" %(nsamples, toc - tic, nsamples / (toc - tic), (toc - tic) / nsamples))

