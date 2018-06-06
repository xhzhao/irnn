#!/bin/sh
t=$1
lscpu
which python

pattern='xw' #default
echo $p
if [ ! -n "$2" ] ;then
  echo "use default pattern: xw"
else
  pattern=$2
fi


export KMP_AFFINITY=compact,1,0,granularity=fine
cd ../../build

if [ $t == 'bdw' ]; then
  export OMP_NUM_THREADS=44
  ./test_gru_xw_inference $pattern 
fi
if [ $t == 'knl' ]; then
  export OMP_NUM_THREADS=68
  ./test_gru_xw_inference $pattern 
fi
if [ $t == 'knm' ]; then
  export OMP_NUM_THREADS=72
  ./test_gru_xw_inference $pattern 
fi
if [ $t == 'skx' ]; then
  export OMP_NUM_THREADS=56
  ./test_gru_xw_inference $pattern 
fi
if [ $t == 'i7' ]; then
  export OMP_NUM_THREADS=8
  ./test_gru_xw_inference $pattern 
fi
if [ $t == 'gpu' ]; then
  ./test_gru_inference
fi
if [ $t == 'cur' ]; then
  ./test_gru_xw_inference
fi

cd -
