#!/bin/sh
t=$1
lscpu
which python

pattern='wx' #default
echo $p
if [ ! -n "$2" ] ;then
  echo "use default pattern: wx"
else
  pattern=$2
fi


export KMP_AFFINITY=compact,1,0,granularity=fine
cd ../../build

if [ $t == 'bdw' ]; then
  export OMP_NUM_THREADS=44
  ./test_lstm_inference $pattern 
fi
if [ $t == 'knl' ]; then
  export OMP_NUM_THREADS=68
  ./test_lstm_inference $pattern
fi
if [ $t == 'knm' ]; then
  export OMP_NUM_THREADS=72
  ./test_lstm_inference $pattern
fi
if [ $t == 'skx' ]; then
  export OMP_NUM_THREADS=56
  ./test_lstm_inference $pattern
fi
if [ $t == 'gpu' ]; then
  ./test_lstm_inference
fi
if [ $t == 'cur' ]; then
  ./test_lstm_inference
fi

cd ..
