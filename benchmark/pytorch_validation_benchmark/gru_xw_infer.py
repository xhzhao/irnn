import time
import torch
import torch.nn as nn
from torch.autograd import Variable
import intelrnn_pytorch as irnn
import numpy as np
import sys

import argparse
parser = argparse.ArgumentParser(description='Process LSTM(xw) args.')
parser.add_argument('--check', action='store_true', default=False, help='Turn on to enable cosim with original PyTorch LSTM implementation.')
parser.add_argument('--bidirectional', action='store_true', default=True, help='Enable bi-directional LSTM')
parser.add_argument('--no_bias', action='store_true', default=False, help='GEMM with no bias.')
parser.add_argument('--count', default=100, type=int)
parser.add_argument('--num_layers', default=1, type=int, help='Number of LSTM layers.')
parser.add_argument('--store_parameters', action='store_true', default=False, help='Record all random parameters and store them to .npy files. (for ease of debug)')
parser.add_argument('--load_parameters', action='store_true', default=False, help='Load stored parameters. (for ease of debug)')
parser.add_argument('placeholder_for_check', nargs='*')

args = parser.parse_args()

check = args.check or 'check' in sys.argv
bidirectional = args.bidirectional
bias = not args.no_bias
count = args.count
num_layers = args.num_layers
store_params = args.store_parameters
load_params = args.load_parameters 

if store_params and load_params:
    raise Exception('You\'re not allowed to store and load parameters at the same time!')

sizes = [
          [2, 2, 3, 4], # Toy shape, for debugging purpose. You can comment this one if you have to do serious testing.
          [2, 2, 3, 4], # Toy shape, for debugging purpose. You can comment this one if you have to do serious testing.
          [2, 2, 3, 4], # Toy shape, for debugging purpose. You can comment this one if you have to do serious testing.
          [2, 2, 3, 4], # Toy shape, for debugging purpose. You can comment this one if you have to do serious testing.
         # [2, 2, 3, 4], # Toy shape, for debugging purpose. You can comment this one if you have to do serious testing.
         # [3, 2, 3, 4], # Toy shape, for debugging purpose. You can comment this one if you have to do serious testing.
         # [3, 2, 3, 4], # Toy shape, for debugging purpose. You can comment this one if you have to do serious testing.
         # [3, 2, 3, 4], # Toy shape, for debugging purpose. You can comment this one if you have to do serious testing.
         # [3, 2, 3, 4], # Toy shape, for debugging purpose. You can comment this one if you have to do serious testing.
         # [3, 2, 3, 4], # Toy shape, for debugging purpose. You can comment this one if you have to do serious testing.
         # [3, 4, 5, 6], # Toy shape, for debugging purpose. You can comment this one if you have to do serious testing.
         # [3, 4, 5, 6], # Toy shape, for debugging purpose. You can comment this one if you have to do serious testing.
         # [3, 4, 5, 6], # Toy shape, for debugging purpose. You can comment this one if you have to do serious testing.
         # [3, 4, 5, 6], # Toy shape, for debugging purpose. You can comment this one if you have to do serious testing.
         # [3, 4, 5, 6], # Toy shape, for debugging purpose. You can comment this one if you have to do serious testing.
         [64,30,500,500],
         [64,40,500,500],
         [64,45,500,500],
         [64,50,500,500],
         [20,50,800,800],
         [20,100,800,800],
         [20,150,800,800],
         [20,200,800,800],
         [16,25,512,512],
         [32,25,512,512],
         [64,25,512,512],
         [128,25,512,512],
         [16,25,1024,1024],
         [32,25,1024,1024],
         [64,25,1024,1024],
         [128,25,1024,1024],
         [16,25,2048,2048],
         [32,25,2048,2048],
         [64,25,2048,2048],
         [128,25,2048,2048],
         [16,25,4096,4096],
         [32,25,4096,4096],
         [64,25,4096,4096],
         [128,25,4096,4096],
        ]
#sizes = [[2,2,3,4]]
grads = {}
def save_grad(name):
    def hook(grad):
        grads[name] = grad
    return hook


for idx in range(len(sizes)):
    size = sizes[idx]
    N = size[0]    # batch size
    T = size[1]    # sentence length
    I = size[2]    # embedding size
    H = size[3]    # hidden size
    D = 2 if bidirectional else 1

    ori_input = Variable(torch.randn(T, N, I), requires_grad=True)
    ori_h0 = Variable(torch.randn(D, N, H), requires_grad=True)
    opt_input = Variable(torch.randn(T, N, I), requires_grad=True)
    opt_h0 = Variable(torch.randn(D, N, H), requires_grad=True)

    opt_input.data.copy_(ori_input.data)
    opt_h0.data.copy_(ori_h0.data)

    if store_params:
        with open('input.npy', 'wb+') as input_file:
            np.save(input_file, opt_input.data.numpy())

        with open('h0.npy', 'wb+') as h0_file:
            np.save(h0_file, opt_h0.data.numpy())


    if load_params:
        with open('input.npy', 'rb') as input_file:
            intput_data = torch.from_numpy(np.load(input_file))
            opt_input.data.copy_(intput_data)
            ori_input.data.copy_(intput_data)

        with open('h0.npy', 'rb') as h0_file:
            h0_data = torch.from_numpy(np.load(h0_file))
            opt_h0.data.copy_(h0_data)
            ori_h0.data.copy_(h0_data)


    target_hy = Variable(torch.randn(D, N, H)) # include hy
    target_y = Variable(torch.randn(T, N, H*D)) # include y
    opt_rnn = irnn.GRU(I, H, num_layers, bidirectional=bidirectional, bias=bias)
    opt_rnn.eval()
    if check:
        ori_rnn = nn.GRU(I, H, num_layers, bidirectional=bidirectional, bias=bias)
        ori_rnn.eval()
        model_ori = {}
        index=0
        for name, param in ori_rnn.named_parameters():
            model_ori[index] = param.data
            index = index + 1

            if store_params:
                with open(name + '.npy', 'wb+') as param_file:
                    np.save(param_file, param.data)

            if load_params:
                with open(name + '.npy', 'rb') as param_file:
                    param.data.copy_(torch.from_numpy(np.load(param_file)))

        model_opt = {}
        index=0
        for name, param in opt_rnn.named_parameters():
            model_opt[index] = param.data
            index = index + 1


        if bidirectional is False:
            # single directional LSTM
            model_opt[0].copy_(torch.transpose(model_ori[0],0,1))
            model_opt[1].copy_(torch.transpose(model_ori[1],0,1))
            if bias:
                model_opt[2].copy_(model_ori[2])
                model_opt[3].copy_(model_ori[3])
        else:
            # bidirectional LSTM
            if bias:
                tmp = torch.Tensor(2, model_ori[0].size(1), model_ori[0].size(0))
                tmp[0] = torch.transpose(model_ori[0], 0, 1)
                tmp[1] = torch.transpose(model_ori[4], 0, 1)
                model_opt[0].copy_(tmp)

                tmp = torch.Tensor(2, model_ori[1].size(1), model_ori[1].size(0))
                tmp[0] = torch.transpose(model_ori[1], 0, 1)
                tmp[1] = torch.transpose(model_ori[5], 0, 1)
                model_opt[1].copy_(tmp)
                
                model_opt[2].copy_(torch.cat((model_ori[2], model_ori[6]), 0))
                model_opt[3].copy_(torch.cat((model_ori[3], model_ori[7]), 0))
            else:
                tmp = torch.Tensor(2, model_ori[0].size(1), model_ori[0].size(0))
                tmp[0] = torch.transpose(model_ori[0], 0, 1)
                tmp[1] = torch.transpose(model_ori[2], 0, 1)
                model_opt[0].copy_(tmp)


                tmp = torch.Tensor(2, model_ori[1].size(1), model_ori[1].size(0))
                tmp[0] = torch.transpose(model_ori[1], 0, 1)
                tmp[1] = torch.transpose(model_ori[3], 0, 1)
                model_opt[1].copy_(tmp)
        # get original output
        ori_y, ori_ht = ori_rnn(ori_input, ori_h0)
        ori_np = ori_y.data.numpy()
        # get intelrnn output
        opt_y, opt_ht = opt_rnn(opt_input, opt_h0)
        opt_np = opt_y.data.numpy()
        #check result with 1% and 0.0001 tolerance
        rtn_y = np.allclose(ori_np, opt_np, 0.01, 1e-4)
        rtn_ht = np.allclose(ori_ht.data.numpy(), opt_ht.data.numpy(), 0.01, 1e-4)
        print("fwd check = ", rtn_y, rtn_ht)
        if rtn_y is False or rtn_ht is False:
            print("ori_np: {}".format(ori_y))
            print("====================================")
            print("opt_np: {}".format(opt_y))

'''    #warm up twice
    opt_states, opt_ht = opt_rnn(opt_input, opt_h0)
    opt_loss = opt_loss_fn(opt_ht, target_hy)
    opt_loss.backward()
    opt_states, opt_ht = opt_rnn(opt_input, opt_h0)
    opt_loss = opt_loss_fn(opt_ht, target_hy)
    opt_loss.backward()

    start = time.time()
    for j in range(count):
        opt_states, opt_ht = opt_rnn(opt_input, opt_h0)
        opt_loss = opt_loss_fn(opt_ht, target_hy)
        opt_loss.backward()
    dura = (time.time() - start)/count     # time of ONE iteration
    gflops = T*4*(N*H*I*2 + N*H*H*2)/1e9
    GFLOPS = gflops/dura                   # giga floating-point operations per second
    SPS = N/dura                           # number of processed sentences per second
    print("size = %s, duration = %.4f, SPS = %.4f" %(size,1e6*dura/N,SPS))

'''
