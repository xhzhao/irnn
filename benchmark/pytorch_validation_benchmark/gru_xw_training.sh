#!/bin/sh
#example: ./gru_xw_training.sh skx
#example: ./gru_xw_training.sh skx check

lscpu
which python
export KMP_AFFINITY=compact,1,0,granularity=fine

CORES=`lscpu | grep Core | awk '{print $4}'`
SOCKETS=`lscpu | grep Socket | awk '{print $2}'`
TOTAL_CORES=`expr $CORES \* $SOCKETS`

KMP_SETTING="KMP_AFFINITY=granularity=fine,compact,1,0"

export OMP_NUM_THREADS=$TOTAL_CORES
export $KMP_SETTING

echo -e "### using OMP_NUM_THREADS=$TOTAL_CORES"
echo -e "### using $KMP_SETTING\n"

python gru_xw_train.py $1
