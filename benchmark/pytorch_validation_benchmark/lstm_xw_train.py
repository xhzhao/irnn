import time
import torch
import torch.nn as nn
from torch.autograd import Variable
import intelrnn_pytorch as irnn
import numpy as np
import sys
import argparse

import util

parser = argparse.ArgumentParser(description='Process LSTM(xw) args.')
parser.add_argument('--check', action='store_true', default=False, help='Turn on to enable cosim with original PyTorch LSTM implementation.')
parser.add_argument('--bidirectional', action='store_true', default=False, help='Enable bi-directional LSTM')
parser.add_argument('--no_bias', action='store_true', default=False, help='GEMM with no bias.')
parser.add_argument('--count', default=100, type=int)
parser.add_argument('--forward_only', default=False, action='store_true', help='Enable forward only check')
parser.add_argument('--num_layers', default=1, type=int, help='Number of LSTM layers.')
parser.add_argument('--store_parameters', action='store_true', default=False, help='Record all random parameters and store them to .npy files. (for ease of debug)')
parser.add_argument('--load_parameters', action='store_true', default=False, help='Load stored parameters. (for ease of debug)')
parser.add_argument('--no_verbose', action='store_true', default=False, help='Suppress verbose info.')
parser.add_argument('placeholder_for_check', nargs='*')

args = parser.parse_args()

check = args.check or 'check' in sys.argv
bidirectional = args.bidirectional
bias = not args.no_bias
count = args.count
forward_only = args.forward_only
num_layers = args.num_layers
store_params = args.store_parameters
load_params = args.load_parameters
verbose = not args.no_verbose

if store_params and load_params:
    raise Exception('You\'re not allowed to store and load parameters at the same time!')

sizes = [
          [1, 2, 3, 4], # Toy shape, for debugging purpose. You can comment this one if you have to do serious testing.
          [2, 2, 4, 5], # Toy shape, for debugging purpose. You can comment this one if you have to do serious testing.
          [2, 2, 3, 4], # Toy shape, for debugging purpose. You can comment this one if you have to do serious testing.
          [2, 2, 3, 4], # Toy shape, for debugging purpose. You can comment this one if you have to do serious testing.
          [2, 2, 3, 4], # Toy shape, for debugging purpose. You can comment this one if you have to do serious testing.
          [2, 2, 3, 4], # Toy shape, for debugging purpose. You can comment this one if you have to do serious testing.
          [2, 2, 3, 4], # Toy shape, for debugging purpose. You can comment this one if you have to do serious testing.
          [3, 2, 3, 4], # Toy shape, for debugging purpose. You can comment this one if you have to do serious testing.
          [3, 2, 3, 4], # Toy shape, for debugging purpose. You can comment this one if you have to do serious testing.
          [3, 2, 3, 4], # Toy shape, for debugging purpose. You can comment this one if you have to do serious testing.
          [3, 2, 3, 4], # Toy shape, for debugging purpose. You can comment this one if you have to do serious testing.
          [3, 2, 3, 4], # Toy shape, for debugging purpose. You can comment this one if you have to do serious testing.
          [3, 4, 5, 6], # Toy shape, for debugging purpose. You can comment this one if you have to do serious testing.
          [3, 4, 5, 6], # Toy shape, for debugging purpose. You can comment this one if you have to do serious testing.
          [3, 4, 5, 6], # Toy shape, for debugging purpose. You can comment this one if you have to do serious testing.
          [3, 4, 5, 6], # Toy shape, for debugging purpose. You can comment this one if you have to do serious testing.
          [3, 4, 5, 6], # Toy shape, for debugging purpose. You can comment this one if you have to do serious testing.
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
    ori_c0 = Variable(torch.randn(D, N, H), requires_grad=True)
    opt_input = Variable(torch.randn(T, N, I), requires_grad=True)
    opt_h0 = Variable(torch.randn(D, N, H), requires_grad=True)
    opt_c0 = Variable(torch.randn(D, N, H), requires_grad=True)

    opt_input.data.copy_(ori_input.data)
    opt_h0.data.copy_(ori_h0.data)
    opt_c0.data.copy_(ori_c0.data)

    if store_params:
        util.store_inputs(opt_input, opt_h0, opt_c0)

    if load_params:
        util.load_inputs(opt_input, opt_h0, opt_c0, ori_input, ori_h0, ori_c0)


    ori_loss_fn = torch.nn.L1Loss()
    opt_loss_fn = torch.nn.L1Loss()
    #ori_output = Variable(torch.randn(2D, N, H))  # include ht and ct
    #opt_output = Variable(torch.randn(2, D, N, H))  # include ht and ct
    targets = Variable(torch.randn(2*D, N, H)) # include ht and ct
    if store_params:
        util.store_targets(targets)
    elif load_params:
        util.load_targets(targets)

    opt_rnn = irnn.LSTM(I, H, num_layers, bidirectional=bidirectional, bias=bias)

    if check:
        ori_rnn = nn.LSTM(I, H, num_layers, bidirectional=bidirectional, bias=bias)

        model_ori = {}
        index=0
        for name, param in ori_rnn.named_parameters():
            model_ori[index] = param.data
            index = index + 1

            if store_params:
                util.store_weight_bias(name, param)

            if load_params:
                util.load_weight_bias(name, param)

        model_opt = {}
        index=0
        for name, param in opt_rnn.named_parameters():
            model_opt[index] = param.data
            index = index + 1

        util.copy_weight_bias(model_ori, model_opt, bidirectional, bias)

        # get original output
        ori_states, (ori_ht, ori_ct) = ori_rnn(ori_input, (ori_h0, ori_c0))
        ori_np = ori_states.data.numpy()
        # get intelrnn output
        opt_states, (opt_ht, opt_ct) = opt_rnn(opt_input, (opt_h0, opt_c0))
        opt_np = opt_states.data.numpy()
        #check result with 1% and 0.0001 tolerance
        rtn = np.allclose(ori_np, opt_np, 0.001, 1e-4)
        ct_equal = np.allclose(ori_ct.data.numpy(), opt_ct.data.numpy(), 0.001, 1e-4)
        ht_equal = np.allclose(ori_ht.data.numpy(), opt_ht.data.numpy(), 0.001, 1e-4)
        forward_check = rtn and ct_equal and ht_equal
        print("fwd check = ", forward_check)
        if forward_check is False:
            # print("ori_ct: \n{}".format(ori_ct.data.numpy()))
            # print("====================================")
            # print("opt_ct: \n{}".format(opt_ct.data.numpy()))
            # print('shape: ', opt_ct.data.numpy().shape)

            # print("\n\nori_np: \n{}".format(ori_np))
            # print("====================================")
            # print("opt_np: \n{}".format(opt_np))

            # print("ori_ht: \n{}".format(ori_ht.data.numpy()))
            # print("====================================")
            # print("opt_ht: \n{}".format(opt_ht.data.numpy()))
            pass

        ori_ht.register_hook(save_grad('ori_ht'))
        ori_ct.register_hook(save_grad('ori_ct'))
        opt_ht.register_hook(save_grad('opt_ht'))
        opt_ct.register_hook(save_grad('opt_ct'))
        if forward_only is False:
            ori_output = torch.cat((ori_ht,ori_ct),0)
            ori_loss = ori_loss_fn(ori_output, targets)
            ori_loss.backward()
            opt_output = torch.cat((opt_ht,opt_ct),0)
            opt_loss = opt_loss_fn(opt_output, targets)
            opt_loss.backward()
            #check result with 1% and 0.0001 tolerance
            if verbose:
                print("ori_ht.grad sum = %.6f, ori_ct.grad sum = %.6f" % (grads['ori_ht'].data.sum(), grads['ori_ct'].data.sum()))
                print("opt_ht.grad sum = %.6f, opt_ct.grad sum = %.6f" %(grads['opt_ht'].data.sum(), grads['opt_ct'].data.sum()))
                print("ori_input.grad size = %s, sum = %.6f" % ( ori_input.grad.size(), ori_input.grad.data.sum()))
                print("opt_input.grad size = %s, sum = %.6f" % ( opt_input.grad.size(), opt_input.grad.data.sum()))

            rtol = 0.001
            atol = 1e-4

            input_grad_equal = np.allclose(ori_input.grad.data.numpy(), opt_input.grad.data.numpy(), rtol, atol)
            h0_grad_equal = np.allclose(ori_h0.grad.data.numpy(), opt_h0.grad.data.numpy(), rtol, atol)
            c0_grad_equal = np.allclose(ori_c0.grad.data.numpy(), opt_c0.grad.data.numpy(), rtol, atol)
            Wx_grad_equal = util.compare_wx_grad(opt_rnn, ori_rnn, bidirectional, bias)
            bx_grad_equal = util.compare_bx_grad(opt_rnn, ori_rnn, bidirectional, bias) 
            Wh_grad_equal = util.compare_wh_grad(opt_rnn, ori_rnn, bidirectional, bias)
            bh_grad_equal = util.compare_bh_grad(opt_rnn, ori_rnn, bidirectional, bias)
            

            name_list  = ['input', 'h0', 'c0', 'Wx', 'bx', 'Wh', 'bh']
            result_list = [input_grad_equal, h0_grad_equal, c0_grad_equal, Wx_grad_equal, bx_grad_equal, Wh_grad_equal, bh_grad_equal]
            rtn = all(result_list)
            print("bwd check = ", rtn)

            if rtn is False:
                for name, result in zip(name_list, result_list):
                    if result is False:
                        print('grad of "{}" fails to pass the bwd check.'.format(name))

                # print('model_opt[0]: \n', list(opt_rnn.parameters())[0].grad.data.numpy())
                # print('==========================================================')
                # print('model_ori[0]: \n', list(ori_rnn.parameters())[0].grad.data.numpy())
                # print('ori_input.grad: \n{}'.format(ori_input.grad.data.numpy()))
                # print('opt_input.grad: \n{}'.format(opt_input.grad.data.numpy()))
                # print('ori_c0.grad: \n{}'.format(ori_c0.grad.data.numpy()))
                # print('opt_c0.grad: \n{}'.format(opt_c0.grad.data.numpy()))
                # print('ori_h0.grad: \n{}'.format(ori_h0.grad.data.numpy()))
                # print('opt_h0.grad: \n{}'.format(opt_h0.grad.data.numpy()))

            #print("ori_output sum = %.4f, opt_output = %.4f" % (ori_output.data.sum(), opt_output.data.sum() ))
    #warm up twice
    opt_states, (opt_ht, opt_ct) = opt_rnn(opt_input, (opt_h0, opt_c0))
    opt_output = torch.cat((opt_ht, opt_ct), 0)
    opt_loss = opt_loss_fn(opt_output, targets)
    opt_loss.backward()
    opt_states, (opt_ht, opt_ct) = opt_rnn(opt_input, (opt_h0, opt_c0))
    opt_output = torch.cat((opt_ht, opt_ct), 0)
    opt_loss = opt_loss_fn(opt_output, targets)
    opt_loss.backward()
    start = time.time()
    for j in range(count):
        opt_states, (opt_ht, opt_ct) = opt_rnn(opt_input, (opt_h0, opt_c0))
        opt_output = torch.cat((opt_ht, opt_ct), 0)
        opt_loss = opt_loss_fn(opt_output, targets)
        opt_loss.backward()
    dura = (time.time() - start)/count     # time of ONE iteration
    gflops = T*4*(N*H*I*2 + N*H*H*2)/1e9
    GFLOPS = gflops/dura                   # giga floating-point operations per second
    SPS = N/dura                           # number of processed sentences per second
    print("size = %s, duration = %.4f, SPS = %.4f" %(size,1e6*dura/N,SPS))


