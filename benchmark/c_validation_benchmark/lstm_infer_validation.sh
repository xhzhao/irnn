#!/bin/sh
t=$1
lscpu
which python
export KMP_AFFINITY=compact,1,0,granularity=fine
cd ../../build

./test_lstm

