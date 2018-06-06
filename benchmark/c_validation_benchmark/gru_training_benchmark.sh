#!/bin/sh
lscpu
which python

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

KMP_SETTING="KMP_AFFINITY=granularity=fine,compact,1,0"

export OMP_NUM_THREADS=$TOTAL_CORES
export $KMP_SETTING

echo -e "### using OMP_NUM_THREADS=$TOTAL_CORES"
echo -e "### using $KMP_SETTING\n"

cd ../../build


./test_gru_xw_training $pattern $2

