#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <mkl.h>
#include <sys/time.h>
#include "sru.h"

void print(float *array, int time_step, int row, int col)
{
    int i, j, k;
    for(i = 0; i < time_step; ++i)
    {
        printf("timestep: %d\n", i);
        for(j = 0; j < row; ++j)
        {
            for(k = 0; k < col; ++k)
            {
                printf("%f ", array[i * row * col + j * col + k]);
            }
            printf("\n");
        }
        printf("\n");
    }

}
void random_fill(float *parray, int len)
{
    int i;
    for(i = 0; i < len; ++i)
    {   
        parray[i] = (float)rand() / (float)RAND_MAX;
    }   
}

int main(int argc, char *argv[])
{
    int time_step = 10;
    int batch_size = 64;
    int input_dim = 150;     
    int hidden_dim = 1024;     
    //int time_step = 3;
    //int batch_size = 2;
    //int input_dim = 5;     
    //int hidden_dim = 5;     
    bool return_sequences = false;
    
    float *c_0 = (float*)mkl_calloc(batch_size * hidden_dim, sizeof(float), 64); 
    float *x_t = (float*)mkl_calloc(time_step * batch_size * input_dim, sizeof(float), 64);      
    float *h_out = NULL;
    if (return_sequences) {
        h_out = (float*)mkl_calloc(time_step * batch_size * hidden_dim, sizeof(float), 64); 
    } 
    else {
        h_out = (float*)mkl_calloc(batch_size * hidden_dim, sizeof(float), 64);
    }
    float *b_f = (float*)mkl_calloc(batch_size * hidden_dim, sizeof(float), 64); 
    float *b_r = (float*)mkl_calloc(batch_size * hidden_dim, sizeof(float), 64);
    float *w_x = (float*)mkl_calloc(hidden_dim * input_dim, sizeof(float), 64); 
    float *w_f = (float*)mkl_calloc(hidden_dim * input_dim, sizeof(float), 64); 
    float *w_r = (float*)mkl_calloc(hidden_dim * input_dim, sizeof(float), 64);
    float *w_tmp = (float*)mkl_calloc(hidden_dim * input_dim, sizeof(float), 64);
   // float* w_tmp = NULL;

    random_fill(x_t, time_step * batch_size * input_dim);
    random_fill(c_0, batch_size * hidden_dim);
    random_fill(w_x, input_dim * hidden_dim);
    random_fill(w_f, input_dim * hidden_dim);
    random_fill(w_r, input_dim * hidden_dim);
    random_fill(b_f, batch_size * hidden_dim);
    random_fill(b_r, batch_size * hidden_dim);
    random_fill(w_tmp, input_dim * hidden_dim);

    double begin, end;
    int i = 0, count = 10000;
    void *buf = mkl_malloc(sru_get_size(batch_size, hidden_dim, time_step), 64);
    memset(buf, 0, sru_get_size(batch_size, hidden_dim, time_step));
    //test_batch
    printf("batch_gemm called.\n");
    sru_inference(buf, batch_size, time_step, input_dim, hidden_dim, w_x, w_f, w_r, w_tmp, b_f, b_r, 
                  c_0, x_t, h_out, return_sequences, 0); 
    sru_inference(buf, batch_size, time_step, input_dim, hidden_dim, w_x, w_f, w_r, w_tmp, b_f, b_r, 
                  c_0, x_t, h_out, return_sequences, 0); 
   // if (return_sequences) {
   //     print(h_out, time_step, hidden_dim, batch_size);
   // }
   // else {
   //     print(h_out, 1, hidden_dim, batch_size);
   // }
    begin = dsecnd();
    for(i = 0; i < count; ++i)
    {
        sru_inference(buf, batch_size, time_step, input_dim, hidden_dim, w_x, w_f, w_r, w_tmp, b_f, b_r, 
                      c_0, x_t, h_out, return_sequences, 0); 
    }
    end = dsecnd();
    printf("samples/s:%lf\n", batch_size * count/ (end-begin));
    printf("time:%lf\n", (end-begin));

    //test_sequential
    printf("sequential_gemm called.\n");
    sru_inference(buf, batch_size, time_step, input_dim, hidden_dim, w_x, w_f, w_r, w_tmp, b_f, b_r, 
                  c_0, x_t, h_out, return_sequences, 1); 
    sru_inference(buf, batch_size, time_step, input_dim, hidden_dim, w_x, w_f, w_r, w_tmp, b_f, b_r, 
                  c_0, x_t, h_out, return_sequences, 1); 
   // if (return_sequences) {
   //     print(h_out, time_step, hidden_dim, batch_size);
   // }
   // else {
   //     print(h_out, 1, hidden_dim, batch_size);
   // }
    begin = dsecnd();
    for(i = 0; i < count; ++i)
    {
        sru_inference(buf, batch_size, time_step, input_dim, hidden_dim, w_x, w_f, w_r, w_tmp, b_f, b_r, 
                      c_0, x_t, h_out, return_sequences, 1); 
    }
    end = dsecnd();
    printf("time:%lfms\n", (end-begin)*1000.0/count);
 
 
    //test_pack
    printf("pack_gemm called.\n");
    sru_inference(buf, batch_size, time_step, input_dim, hidden_dim, w_x, w_f, w_r, w_tmp, b_f, b_r, 
                  c_0, x_t, h_out, return_sequences, 2); 
    sru_inference(buf, batch_size, time_step, input_dim, hidden_dim, w_x, w_f, w_r, w_tmp, b_f, b_r, 
                  c_0, x_t, h_out, return_sequences, 2); 
   // if (return_sequences) {
   //     print(h_out, time_step, hidden_dim, batch_size);
   // }
   // else {
   //     print(h_out, 1, hidden_dim, batch_size);
   // }
    begin = dsecnd();
    for(i = 0; i < count; ++i)
    {
        sru_inference(buf, batch_size, time_step, input_dim, hidden_dim, w_x, w_f, w_r, w_tmp, b_f, b_r, 
                      c_0, x_t, h_out, return_sequences, 2); 
    }
    end = dsecnd();
    printf("time:%lfms\n", (end-begin)*1000.0/count);
    mkl_free(x_t);
    mkl_free(h_out);
    mkl_free(c_0);

    mkl_free(b_f);
    mkl_free(b_r);
    mkl_free(w_x);
    mkl_free(w_f);
    mkl_free(w_r);
    return 0;
}
