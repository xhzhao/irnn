#!/bin/sh
t=$1
lscpu
which python
export KMP_AFFINITY=compact,1,0,granularity=fine

if [ $t == 'bdw' ]; then
  export OMP_NUM_THREADS=44
  python lstm_wx.py $2
fi
if [ $t == 'knl' ]; then
  export OMP_NUM_THREADS=68
  python lstm_wx.py $2
fi
if [ $t == 'knm' ]; then
  export OMP_NUM_THREADS=72
  python lstm_wx.py $2
fi
if [ $t == 'skx' ]; then
  export OMP_NUM_THREADS=56
  python lstm_wx.py $2
fi
if [ $t == 'i7' ]; then
  export OMP_NUM_THREADS=8
  python lstm_wx.py $2
fi
