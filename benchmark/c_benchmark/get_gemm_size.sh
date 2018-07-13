log=$1
grep "gflops" $log | awk '{print $1 "\t" $2 "\t" $3 "\t" $4 "\t" $5}' | sort | uniq  &> size.log

