log1=$1

cat size.log | while read line
do
  str=$line
  count1=$(grep "${str}" $log1 | wc -l)
  grep "${str}" $log1 | awk '{sum+=$10} END {print $str "\t"  sum/'$count1'}'
  #echo "count1 = "  $count1 "avg1 = " $avg1

done
