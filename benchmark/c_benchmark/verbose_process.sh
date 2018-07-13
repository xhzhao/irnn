echo "processing file: $1 "
grep "SGEMM" $1 | tr ',' ' ' | tr '(' ' '| sed "s/us/\ us/g" | sed "s/ms/\ ms/g"  \
    | awk '{if($17=="us") {gflops = $5*$6*$7/$16/1e3 ;print $3 "\t" $4 "\t" $5 "\t" $6 "\t" $7 "\t" $16 "\t" $17 "\t gflops = " gflops} \
            else {gflops = $5*$6*$7/$16/1e6 ;print $3 "\t" $4 "\t" $5 "\t" $6 "\t" $7 "\t" $16 "\t" $17 "\t gflops = "gflops}}' &> gflops.log

./get_gemm_size.sh gflops.log
./avg_gflops.sh gflops.log

