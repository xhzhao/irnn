import theano
from theano import tensor as T
import numpy as np

from mkl_gru_inference import GRU_Inference

np.random.seed(12345)

hidden_size = 1024
input_size = 1024
seq_length = 1
batch_size = 32

def sigmoid(x):
    return 1. / (1. + np.exp(-x))

x = np.random.rand(seq_length, input_size, batch_size).astype(np.float64)
w_x = np.random.rand(3*hidden_size, input_size).astype(np.float64) - np.random.rand(3*hidden_size, input_size).astype(np.float64)
w_xh = w_x[0:hidden_size, :]
w_xz = w_x[hidden_size: 2*hidden_size, :]
w_xr = w_x[2*hidden_size:, :]

w_h = np.random.rand(3*hidden_size, hidden_size).astype(np.float64) - np.random.rand(3*hidden_size, hidden_size).astype(np.float64)
w_hh = w_h[0:hidden_size, :]
w_hz = w_h[hidden_size:2*hidden_size, :]
w_hr = w_h[2*hidden_size:, :]

hid = np.zeros((hidden_size, batch_size), np.float64)


def GRU_NP():
    global x, w_xr, w_xz, w_xh, w_hr, w_hz, w_hh, hid
    for i in range(x.shape[0]):
        x_r = np.dot(w_xr, x[i])
        x_z = np.dot(w_xz, x[i])
        x_h = np.dot(w_xh, x[i])
        
        t = x_z + np.dot(w_hz, hid)
        z_t = sigmoid(t)

        t = x_r + np.dot(w_hr, hid)
        r_t = sigmoid(t)

        t0 = np.dot(w_hh, hid)
        t = x_h + r_t * t0
        can_h_t = np.tanh(t)

        hid = (1. - z_t) * hid + z_t * can_h_t
    return hid


def GRU_MKL():
    global x, w_x, w_h, hid
    # avoid hid is modified by numpy function
    hid = np.zeros((hidden_size, batch_size), np.float64)
    
    X = T.dtensor3('X')
    W_x = T.dmatrix('W_x')
    W_h = T.dmatrix('W_h')
    Hid = T.dmatrix('Hid')

    Z = GRU_Inference(hid=hidden_size, return_sequences=True, max_len=20)(X, W_x, W_h, Hid)
    f = theano.function([X, W_x, W_h, Hid], Z)

    o = f(x, w_x, w_h, hid)
    return o


if __name__ == '__main__':
    a = GRU_NP()
    b = GRU_MKL()
    #print(a)
    #print(b)
    assert np.allclose (a, b[-1])
