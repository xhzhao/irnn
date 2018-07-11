import time
import torch
import torch.nn as nn
from torch.autograd import Variable
import irnn_pytorch as irnn
import numpy as np
import sys
import util 

import argparse
parser = argparse.ArgumentParser(description='Process LSTM(xw) args.')
parser.add_argument('--check', action='store_true', default=True, help='Turn on to enable cosim with original PyTorch LSTM implementation.')
parser.add_argument('--bidirectional', action='store_true', default=False, help='Enable bi-directional LSTM')
parser.add_argument('--no_bias', action='store_true', default=False, help='GEMM with no bias.')
parser.add_argument('--count', default=100, type=int)
parser.add_argument('--forward_only', default=False, action='store_true', help='Enable forward only check')
parser.add_argument('--num_layers', default=1, type=int, help='Number of LSTM layers.')
parser.add_argument('--store_parameters', action='store_true', default=False, help='Record all random parameters and store them to .npy files. (for ease of debug)')
parser.add_argument('--load_parameters', action='store_true', default=False, help='Load stored parameters. (for ease of debug)')
parser.add_argument('placeholder_for_check', nargs='*')

args = parser.parse_args()
print("args = ", args)

check = args.check or 'check' in sys.argv
bidirectional = args.bidirectional
bias = not args.no_bias
count = args.count
forward_only = args.forward_only
num_layers = args.num_layers
store_params = args.store_parameters
load_params = args.load_parameters 

if store_params and load_params:
    raise Exception('You\'re not allowed to store and load parameters at the same time!')

sizes = [
          [1, 2, 1, 1], # Toy shape, for debugging purpose. You can comment this one if you have to do serious testing.
          [1, 2, 2, 1], # Toy shape, for debugging purpose. You can comment this one if you have to do serious testing.
          [1, 2, 1, 2], # Toy shape, for debugging purpose. You can comment this one if you have to do serious testing.
          [1, 2, 3, 4], # Toy shape, for debugging purpose. You can comment this one if you have to do serious testing.
          [2, 2, 3, 4], # Toy shape, for debugging purpose. You can comment this one if you have to do serious testing.
          [3, 2, 3, 4], # Toy shape, for debugging purpose. You can comment this one if you have to do serious testing.
          [3, 4, 5, 6], # Toy shape, for debugging purpose. You can comment this one if you have to do serious testing.
         [64,30,500,500],
         [64,40,500,500],
         [64,45,500,500],
         [64,50,500,500],
         [20,50,800,800],
         [20,100,800,800],
         [20,150,800,800],
         [20,200,800,800],
#         [16,25,512,512],
#         [32,25,512,512],
#         [64,25,512,512],
#         [128,25,512,512],
#         [16,25,1024,1024],
#         [32,25,1024,1024],
#         [64,25,1024,1024],
#         [128,25,1024,1024],
#         [16,25,2048,2048],
#         [32,25,2048,2048],
#         [64,25,2048,2048],
#         [128,25,2048,2048],
#         [16,25,4096,4096],
#         [32,25,4096,4096],
#         [64,25,4096,4096],
#         [128,25,4096,4096],
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
    print("N = ",N," T = ",T," I = ",I," H = ",H," D = ",D)

    ori_input = Variable(torch.randn(T, N, I), requires_grad=True)
    ori_hx = Variable(torch.randn(D, N, H), requires_grad=True)
    ori_cx = Variable(torch.randn(D, N, H), requires_grad=True)
    opt_input = Variable(torch.randn(T, N, I), requires_grad=True)
    opt_hx = Variable(torch.randn(D, N, H), requires_grad=True)
    opt_cx = Variable(torch.randn(D, N, H), requires_grad=True)

    opt_input.data.copy_(ori_input.data)
    opt_hx.data.copy_(ori_hx.data)
    opt_cx.data.copy_(ori_cx.data)

    #print("ori_input sum = ", ori_input.sum())
    #print("ori_hx sum = ", ori_hx.sum())
    #print("ori_cx sum = ", ori_cx.sum())

    if store_params:
        with open('input.npy', 'wb+') as input_file:
            np.save(input_file, opt_input.data.numpy())

        with open('hx.npy', 'wb+') as hx_file:
            np.save(hx_file, opt_hx.data.numpy())


    if load_params:
        with open('input.npy', 'rb') as input_file:
            intput_data = torch.from_numpy(np.load(input_file))
            opt_input.data.copy_(intput_data)
            ori_input.data.copy_(intput_data)

        with open('hx.npy', 'rb') as hx_file:
            hx_data = torch.from_numpy(np.load(hx_file))
            opt_hx.data.copy_(hx_data)
            ori_hx.data.copy_(hx_data)


    ori_loss_fn = torch.nn.L1Loss()
    opt_loss_fn = torch.nn.L1Loss()
    target_hy = Variable(torch.randn(D, N, H)) # include hy
    target_cy = Variable(torch.randn(D, N, H)) # include cy
    target_y = Variable(torch.randn(T, N, H*D)) # include y
    opt_rnn = irnn.LSTM(I, H, num_layers, bidirectional=bidirectional, bias=bias)

    if check:
        ori_rnn = nn.LSTM(I, H, num_layers, bidirectional=bidirectional, bias=bias)

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
        ori_y, (ori_hy, ori_cy) = ori_rnn(ori_input, (ori_hx, ori_cx))
        ori_np = ori_y.data.numpy()
        # get intelrnn output
        opt_y, (opt_hy, opt_cy) = opt_rnn(opt_input, (opt_hx, opt_cx))
        opt_np = opt_y.data.numpy()
        #check result with 1% and 0.0001 tolerance
        rtn_y = np.allclose(ori_np, opt_np, 0.01, 1e-4)
        rtn_hy = np.allclose(ori_hy.data.numpy(), opt_hy.data.numpy(), 0.01, 1e-4)
        rtn_cy = np.allclose(ori_cy.data.numpy(), opt_cy.data.numpy(), 0.01, 1e-4)
        print("fwd check = ", rtn_y, rtn_hy, rtn_cy)
        if rtn_y is False or rtn_hy is False:
            print("ori_y_np: {}".format(ori_y))
            print("opt_y_np: {}".format(opt_y))
            print("====================================")
            print("ori_hy_np: {}".format(ori_hy))
            print("opt_hy_np: {}".format(opt_hy))

        ori_y.register_hook(save_grad('ori_y'))
        opt_y.register_hook(save_grad('opt_y'))
        ori_hy.register_hook(save_grad('ori_hy'))
        opt_hy.register_hook(save_grad('opt_hy'))
        ori_cy.register_hook(save_grad('ori_cy'))
        opt_cy.register_hook(save_grad('opt_cy'))
        if forward_only is False:
            ori_loss_1 = ori_loss_fn(ori_hy, target_hy)
            ori_loss_2 = ori_loss_fn(ori_cy, target_cy)
            ori_loss_3 = ori_loss_fn(ori_y, target_y)
            ori_loss = ori_loss_1 + ori_loss_2 + ori_loss_3
            ori_loss.backward()
            opt_loss_1 = opt_loss_fn(opt_hy, target_hy)
            opt_loss_2 = opt_loss_fn(opt_cy, target_cy)
            opt_loss_3 = opt_loss_fn(opt_y, target_y)
            opt_loss = opt_loss_1 + opt_loss_2 + opt_loss_3
            opt_loss.backward()
            #check result with 1% and 0.0001 tolerance

            rtn_x = np.allclose(ori_input.grad.data.numpy(), opt_input.grad.data.numpy(), 0.001, 1e-4)
            rtn_hx = np.allclose(ori_hx.grad.data.numpy(), opt_hx.grad.data.numpy(), 0.001, 1e-4)
            rtn_cx = np.allclose(ori_cx.grad.data.numpy(), opt_cx.grad.data.numpy(), 0.001, 1e-4)

            rtol = 0.001                                                        
            atol = 1e-4                                                         
                                                                                
            input_grad_equal = np.allclose(ori_input.grad.data.numpy(), opt_input.grad.data.numpy(), rtol, atol)
            hx_grad_equal = np.allclose(ori_hx.grad.data.numpy(), opt_hx.grad.data.numpy(), rtol, atol)
            Wx_grad_equal = util.compare_wx_grad(opt_rnn, ori_rnn, bidirectional, bias)
            bx_grad_equal = util.compare_bx_grad(opt_rnn, ori_rnn, bidirectional, bias)
            Wh_grad_equal = util.compare_wh_grad(opt_rnn, ori_rnn, bidirectional, bias)
            bh_grad_equal = util.compare_bh_grad(opt_rnn, ori_rnn, bidirectional, bias)
                                                                                
                                                                                
            name_list  = ['input', 'hx', 'Wx', 'bx', 'Wh', 'bh']          
            result_list = [input_grad_equal, hx_grad_equal, Wx_grad_equal, bx_grad_equal, Wh_grad_equal, bh_grad_equal]
            rtn = all(result_list)                                              
            print("bwd check = ", rtn)     
            if rtn is False:
                print("bwd check = ", result_list)     
                print("ori_hy.grad sum = %.6f" % (grads['ori_hy'].data.sum()))
                print("opt_hy.grad sum = %.6f" %(grads['opt_hy'].data.sum()))
                print("ori_y.grad sum = %.6f" % (grads['ori_y'].data.sum()))
                print("opt_y.grad sum = %.6f" % (grads['opt_y'].data.sum()))
                print("ori_input.grad:", ori_input.grad.size(), ori_input.grad.data.sum())
                print("opt_input.grad:", opt_input.grad.size(), opt_input.grad.data.sum())
                print("ori_hx.grad:", ori_hx.grad.size(), ori_hx.grad.data.sum())
                print("opt_hx.grad:", opt_hx.grad.size(), opt_hx.grad.data.sum())
                print("ori_cx.grad:", ori_cx.grad.size(), ori_cx.grad.data.sum())
                print("opt_cx.grad:", opt_cx.grad.size(), opt_cx.grad.data.sum())

