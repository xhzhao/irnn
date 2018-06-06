#!/usr/bin/env python
import time
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import rnn
import pandas as pd
from sklearn.preprocessing import StandardScaler,MinMaxScaler

def get_feed_dict(x_data, y_data=None):
    feed_dict = {}

    if y_data is not None:
        feed_dict[y] = y_data

    for i in range(x_data.shape[0]):
        feed_dict[x[i]] = x_data[i, :, :]

    return feed_dict


# Parameters
network_type = 'basic_lstm'
hidden_size = 128
input_size = 450
seq_length = 50
batch_size = 241

n_batch = 100
n_samples = batch_size * n_batch 

labelName="label"

train_all = pd.read_csv('train-3-5.csv')
test_all = pd.read_csv('train-3-5.csv')

y_train = train_all.label
y_test = test_all.label

X_Train = train_all.drop(labelName, axis = 1, inplace=False)
X_Test = test_all.drop(labelName, axis = 1, inplace=False)

sc = StandardScaler() 

X_train = sc.fit_transform(X_Train)
X_test = sc.transform(X_Test)

# Data
xinput = np.random.rand(seq_length, batch_size, input_size).astype(np.float32)
xinput = np.tile(X_train,seq_length).reshape(seq_length, batch_size, input_size)
#    for i in range(0,seq_length-1):
#        xinput[i]=X_train
    
ytarget = np.random.rand(batch_size, hidden_size).astype(np.float32)
ytarget = np.tile(y_train,hidden_size).reshape(batch_size,hidden_size)


with tf.device('/cpu:0'):
    x = [tf.placeholder(tf.float32, [batch_size, input_size], name="x") for i in range(seq_length)]
    y = tf.placeholder(tf.float32, [batch_size, hidden_size], name="y")
    
    
    cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size)
    cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=0.7)


    #print ("Compiling...")
    start = time.time()       
    output, _cell_state = rnn.static_rnn(cell, x, dtype=tf.float32)
    
    cost = tf.reduce_sum((output[-1] - y) ** 2)

    #optim = tf.train.GradientDescentOptimizer(0.01)
    #train_op = optim.minimize(cost)
    train_op = tf.train.AdamOptimizer(0.01).minimize(cost)

    #session = tf.Session()
    threadsConfig = tf.ConfigProto(inter_op_parallelism_threads=1, intra_op_parallelism_threads=56)
    session = tf.InteractiveSession(config=threadsConfig)
    
    session.run(tf.global_variables_initializer())
    #print ("Setup : compile + forward/backward x 1")
    #print ("--- %s seconds" % (time.time() - start))

    start = time.time()
    for i in range(0, n_batch):
        session.run(output[-1], feed_dict=get_feed_dict(xinput))
    end = time.time()
    print ("Forward:")
    print ("--- %i samples in %s seconds (%f samples/s, %.7f s/sample) ---" % (n_samples, end - start, n_samples / (end - start), (end - start) / n_samples))

    start = time.time()
    for i in range(0, n_batch):
        session.run(train_op, feed_dict=get_feed_dict(xinput, ytarget))
    end = time.time()
    print ("Forward + Backward:")
    print ("--- %i samples in %s seconds (%f samples/s, %.7f s/sample) ---" % (n_samples, end - start, n_samples / (end - start), (end - start) / n_samples))
