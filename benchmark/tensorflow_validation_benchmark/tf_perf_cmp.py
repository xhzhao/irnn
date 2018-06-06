'''
' @file      tensorflow_perf.py
' @author    zhangshu(shu.zhang@intel.com)
' @date      2017-11-16 18:11:15
' @brief
'''

import time
import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.python.ops import array_ops
import intelrnn_tensorflow
import sys

check = False
if 'check' in sys.argv:
    check = True

count = 2

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
config = tf.ConfigProto()
config.inter_op_parallelism_threads = 1
config.intra_op_parallelism_threads = 44
 

for size in sizes:
    g = tf.Graph()
    with g.as_default():
        N, T, D, H = size
        x_input = np.random.rand(T, N, D).astype(np.float32)
        y_target = np.random.rand(N, H).astype(np.float32)
        intel_x = tf.transpose(x_input, perm = [0, 2, 1])
        intel_y = tf.transpose(y_target, perm = [1, 0])

        intel_cell = intelrnn_tensorflow.BasicLSTMCell(H, forget_bias=0.0, state_is_tuple = True)
        intel_out, _ = intelrnn_tensorflow.static_rnn(intel_cell, intel_x, dtype=tf.float32)
        tmp_out = tf.unstack(intel_out, T)
        intel_cost = tf.reduce_sum((tmp_out[-1] - intel_y) ** 2)                                 #training

        tf_cell = rnn.BasicLSTMCell(H, forget_bias=0.0, state_is_tuple = True)
        tf_out, _ = rnn.static_rnn(tf_cell, tf.unstack(x_input, T), dtype=tf.float32)  #forward
        tf_cost = tf.reduce_sum((tf_out[-1] - y_target) ** 2)                                 #training

        optim = tf.train.GradientDescentOptimizer(0.01)

        tf_train_op = optim.minimize(tf_cost)
        intel_train_op = optim.minimize(intel_cost)


    if check:
        with tf.Session(graph=g, config=config) as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf_out)
            t1 = time.time()
            for _ in range(count):
                sess.run(tf_out)
            t2 = time.time()
            dura = (t2 - t1) / count
            print("size = %s, count = %d, duration = %.4fus, tf_rnn_inference:SPS = %.4f" %(size, count, 1e6*dura/N, N/dura))

            sess.run(tf_train_op)
            t1 = time.time()
            for _ in range(count):
                sess.run(tf_train_op)
            t2 = time.time()
            dura = (t2 - t1) / count
            print("size = %s, count = %d, duration = %.4fus, tf_rnn_training:SPS = %.4f" %(size, count, 1e6*dura/N, N/dura))
#            wi, wc, wf, wo = array_ops.split(cell.weights[0], 4, 1)
#            bi, bc, bf, bo = array_ops.split(cell.weights[1], 4, 0)
#            intel_w = tf.transpose(array_ops.concat([wi, wf, wc, wo], 1), perm=[1, 0])
#            
#            intel_cell.w_x, intel_cell.w_h = array_ops.split(intel_w, [D, H], 1)
#            intel_cell.bias = array_ops.concat([bf, bi, bc, bo], 0)
#            intel_out, _ = intel_cell.inference(intel_x, intel_h_0, intel_c_0)
#
#            tmp_out = sess.run(intel_out)
#            cmp_tf = tf.stack(tf_out).eval()
#            cmp_intel = tmp_out.transpose((0,2,1)) 
#            ret = np.allclose(cmp_tf, cmp_intel, 0.01, 1e-4)
#            print("check = ", ret)


    with tf.Session(graph=g, config=config) as sess:    
        sess.run(tf.global_variables_initializer())
        sess.run(intel_out)
        sess.run(intel_out)
  #      print("training-------------")
  #      for _ in range(10):
  #          print(sess.run(cost))
  #          #print(tf_cell.weights[0].eval())
  #          #print(tf_cell.weights[1].eval())
  #          sess.run(train_op)

        begin = time.time()
        for _ in range(count):
            sess.run(intel_out)
        end = time.time()
        dura = (end - begin)/count
        print("size = %s, count = %d, duration = %.4fus, intel_rnn_inference:SPS = %.4f" %(size, count, 1e6*dura/N, N/dura))

        sess.run(intel_train_op)
        sess.run(intel_train_op)
        begin = time.time()
        for _ in range(count):
            sess.run(intel_train_op)
        end = time.time()
        dura = (end - begin)/count
        print("size = %s, count = %d, duration = %.4fus, intel_rnn_inference:SPS = %.4f" %(size, count, 1e6*dura/N, N/dura))
             
