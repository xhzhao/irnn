#!/bin/bash

source ~/.bashrc
rm -rf /tmp/.theano/*

# set the correct physical cores 
export KMP_AFFINITY=granularity=core,noduplicates,compact,0,0
export OMP_NUM_THREADS=56
export MKL_DYNAMIC=false

export batchsize=32
export inputdim=128
export seqlen=30
export hidesize=128

echo "===Testing GRU of original Tensorflow in CPU==="
python tensorflow/tf_rnn_benchmarks_cpu_lstm.py -n gru -s $seqlen -l $hidesize -i $inputdim -b $batchsize
echo "===Testing GRU of original Tensorflow in GPU==="
python tensorflow/tf_rnn_benchmarks_gpu_lstm.py -n gru -s $seqlen -l $hidesize -i $inputdim -b $batchsize
echo "===Testing GRU w/ Intel RNN engine by Theano in CPU==="
python intel/test_gru_backward.py -n gru -s $seqlen -l $hidesize -i $inputdim -b $batchsize

