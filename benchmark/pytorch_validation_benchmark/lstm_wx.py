import time
import torch
import torch.nn as nn
from torch.autograd import Variable
import intelrnn_pytorch as irnn
import numpy as np
import sys

check = False
if 'check' in sys.argv:
    check = True

count = 100

sizes = [[64,15,500,500],
         [64,20,500,500],
         [64,25,500,500],
         [64,30,500,500],
         [64,35,500,500],
         [64,40,500,500],
         [64,45,500,500],
         [64,50,500,500],
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
         [128,25,4096,4096]
        ]

for idx in range(len(sizes)):
    size = sizes[idx]
    N = size[0]    # batch size
    T = size[1]    # sentence length
    D = size[2]    # embedding size
    H = size[3]    # hidden size
    input = Variable(torch.randn(T, N, D))
    h0 = Variable(torch.randn(1, N, H))
    c0 = Variable(torch.randn(1, N, H))
    opt_rnn = irnn.LSTM(D, H, 1, False)
    if check:
        ori_rnn =   nn.LSTM(D, H, 1, False)
        model = {}
        index=0
        for param in ori_rnn.parameters():
            model[index] = param.data
            index = index + 1
        opt_rnn.w_x = model[0].clone()
        opt_rnn.w_h = model[1].clone()
        #opt_rnn.bias.fill_(0)
        # get original output
        ori_output, (ori_ht, ori_ct) = ori_rnn(input, (h0, c0))
        ori_np = ori_output.data.numpy()
        # get intelrnn output
        opt_output, (opt_ht, opt_ct) = opt_rnn(input, (h0, c0))
        #opt_output_t = opt_output.transpose(1,2)
        opt_np = opt_output.data.numpy()
        #check result with 1% and 0.0001 tolerance
        check = np.allclose(ori_np, opt_np, 0.01, 1e-4)
        print("check = ", check)
        #print("ori_output sum = %.4f, opt_output = %.4f" % (ori_output.data.sum(), opt_output.data.sum() ))
    #warm up twice
    opt_output, (opt_ht, opt_ct) = opt_rnn(input, (h0, c0))
    opt_output, (opt_ht, opt_ct) = opt_rnn(input, (h0, c0))
    start = time.time()
    for j in range(count):
    	opt_output, (opt_ht, opt_ct) = opt_rnn(input, (h0, c0))
    dura = (time.time() - start)/count     # time of ONE iteration
    gflops = T*4*(N*H*D*2 + N*H*H*2)/1e9
    GFLOPS = gflops/dura                   # giga floating-point operations per second
    SPS = N/dura                           # number of processed sentences per second
    print("size = %s, duration = %.4f, SPS = %.4f" %(size,1e6*dura/N,SPS))


