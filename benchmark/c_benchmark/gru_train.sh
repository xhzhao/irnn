#!/bin/sh

pattern='xw' #default
echo $p
if [ ! -n "$1" ] ;then
  echo "use default pattern: xw"
else
  pattern=$1
fi

CORES=`lscpu | grep Core | awk '{print $4}'`
SOCKETS=`lscpu | grep Socket | awk '{print $2}'`
TOTAL_CORES=`expr $CORES \* $SOCKETS`

#KMP_SETTING="KMP_HW_SUBSET=2s,20c,1t KMP_AFFINITY=compact,1,0,granularity=fine"
KMP_SETTING="KMP_AFFINITY=compact,1,0,granularity=fine"

export OMP_NUM_THREADS=$TOTAL_CORES
export $KMP_SETTING

echo -e "### using OMP_NUM_THREADS=$TOTAL_CORES"
echo -e "### using $KMP_SETTING\n"

cd ../../build

./rnn_bench train gru ud # unidirectional
./rnn_bench train gru bd # bidirectional

