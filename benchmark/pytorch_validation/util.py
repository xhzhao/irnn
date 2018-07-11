import numpy as np
import torch

def store_targets(targets):
    with open('targets.npy', 'wb+') as target_file:
        np.save(target_file, targets.data.numpy())

def load_targets(targets):
    with open('targets.npy', 'rb') as target_file:
        targets.data.copy_(torch.from_numpy(np.load(target_file)))

def store_inputs(opt_input, opt_h0, opt_c0):
    with open('input.npy', 'wb+') as input_file:
        np.save(input_file, opt_input.data.numpy())

    with open('h0.npy', 'wb+') as h0_file:
        np.save(h0_file, opt_h0.data.numpy())

    with open('c0.npy', 'wb+') as c0_file:
        np.save(c0_file, opt_c0.data.numpy())

def load_inputs(opt_input, opt_h0, opt_c0, ori_input, ori_h0, ori_c0):
    with open('input.npy', 'rb') as input_file:
        input_data = torch.from_numpy(np.load(input_file))
        opt_input.data.copy_(input_data)
        ori_input.data.copy_(input_data)

    with open('h0.npy', 'rb') as h0_file:
        h0_data = torch.from_numpy(np.load(h0_file))
        opt_h0.data.copy_(h0_data)
        ori_h0.data.copy_(h0_data)

    with open('c0.npy', 'rb') as c0_file:
        c0_data = torch.from_numpy(np.load(c0_file))
        opt_c0.data.copy_(c0_data)
        ori_c0.data.copy_(c0_data)

def store_weight_bias(name, param):
    with open(name + '.npy', 'wb+') as param_file:
        np.save(param_file, param.data)

def load_weight_bias(name, param):
    with open(name + '.npy', 'rb') as param_file:
        param.data.copy_(torch.from_numpy(np.load(param_file)))

def copy_weight_bias(model_ori, model_opt, bidirectional, bias):
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

def compare_wx_grad(opt_rnn, ori_rnn, bidirectional, bias):
    return compare_parameter_grad(opt_rnn, ori_rnn, bidirectional, bias, 0)

def compare_wh_grad(opt_rnn, ori_rnn, bidirectional, bias):
    return compare_parameter_grad(opt_rnn, ori_rnn, bidirectional, bias, 1)

def compare_bx_grad(opt_rnn, ori_rnn, bidirectional, bias):
    return compare_parameter_grad(opt_rnn, ori_rnn, bidirectional, bias, 2)

def compare_bh_grad(opt_rnn, ori_rnn, bidirectional, bias):
    return compare_parameter_grad(opt_rnn, ori_rnn, bidirectional, bias, 3)

def compare_parameter_grad(opt_rnn, ori_rnn, bidirectional, bias, param_index):
    if bias is False:
        return True

    offset = 4 if bias else 2

    ori_parameters = list(ori_rnn.parameters())
    ori_param_tuple = (ori_parameters[param_index], ori_parameters[param_index + offset]) if bidirectional else (ori_parameters[param_index], )

    opt_parameters = list(opt_rnn.parameters())
    opt_param = opt_parameters[param_index]

    if bidirectional:
        if param_index < 2:
            ori_param_forward_transpose = np.transpose(ori_param_tuple[0].grad.data.numpy(), (1, 0))
            ori_param_backward_transpose = np.transpose(ori_param_tuple[1].grad.data.numpy(), (1, 0))
            
            opt_param_forward = opt_param.grad.data.numpy()[0]
            opt_param_backward = opt_param.grad.data.numpy()[1]
        else:
            ori_param_forward_transpose = ori_param_tuple[0].grad.data.numpy()
            ori_param_backward_transpose = ori_param_tuple[1].grad.data.numpy()

            opt_param_forward = opt_param.grad.data.numpy()[:int(opt_param.grad.data.numpy().size / 2)]
            opt_param_backward = opt_param.grad.data.numpy()[int(opt_param.grad.data.numpy().size / 2):]

        forward_equal = np.allclose(opt_param_forward, ori_param_forward_transpose)
        backward_equal = np.allclose(opt_param_backward, ori_param_backward_transpose)

        if forward_equal is False:
            print('forward weight grad check fail')
            # print('ori: \n', ori_param_tuple[0].grad.data.numpy())
            # print('===========================================================')
            # print('opt: \n', opt_param.grad.data.numpy())
        if backward_equal is False:
            print('backward weight grad check fail')
        
        return forward_equal and backward_equal
    else:
        if param_index < 2:
            ori_param_forward_transpose = np.transpose(ori_param_tuple[0].grad.data.numpy())
            opt_param_forward = opt_param.grad.data.numpy()[0]
        else:
            ori_param_forward_transpose = ori_param_tuple[0].grad.data.numpy()
            opt_param_forward = opt_param.grad.data.numpy()
        forward_equal = np.allclose(opt_param_forward, ori_param_forward_transpose, 0.001, 1e-4)
        return forward_equal
