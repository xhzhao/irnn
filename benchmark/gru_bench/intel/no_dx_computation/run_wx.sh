#!/bin/bash


source ~/.bashrc

export KMP_AFFINITY=granularity=core,noduplicates,compact,0,0
export OMP_NUM_THREADS=56
export MKL_DYNAMIC=false

python test_gru_backward.py
python test_gru_backward.py
python test_gru_backward.py
python test_gru_backward.py
python test_gru_backward.py
