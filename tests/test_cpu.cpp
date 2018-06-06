#include <iostream>
#include <sys/time.h>
#include <rnn.h>

int main() {
    printf("test main \n");
    srand(45678);
    int i,j;
    int output_flag = 1;
    //reuse memory
    //assume timestep changes for diff input length
    int max_len = 128;//max timestep
    int batch_size = 64;
    int time_step = 10;
    int input_dim = 150;
    int hid = 1024;

    const float** A = (const float**)mkl_malloc(4 * max_len * sizeof (float*), 64);
    const float** B = (const float**)mkl_malloc(4 * max_len * sizeof (float*), 64);
    float** A_pack = (float**)mkl_malloc(4 * max_len * sizeof (float*), 64);
    float** C = (float**)mkl_malloc(4 * max_len * sizeof (float*), 64);
    float* x_temp = (float*)mkl_malloc(max_len * 4 * batch_size * hid * sizeof (float), 64);
    float* gemmB = (float*)mkl_malloc(batch_size * (input_dim + hid) * sizeof(float), 64);
    float* gemmC = (float*)mkl_malloc(4 * batch_size * hid * sizeof(float), 64);
    float* f_t = (float*)mkl_malloc(batch_size * hid * sizeof (float), 64);
    float* i_t = (float*)mkl_malloc(batch_size * hid * sizeof (float), 64);
    float* c_wave_t = (float*)mkl_malloc(batch_size * hid * sizeof (float), 64);
    float* o_t = (float*)mkl_malloc(batch_size * hid * sizeof (float), 64);
    float* c_t = (float*)mkl_malloc(batch_size * hid * sizeof (float), 64);
    float* h_t = (float*)mkl_malloc(batch_size * hid * sizeof (float), 64);

    bool return_sequences = false;
    float* w_x;
    float* w_h;
    float* w;    //for combine_gemm
    float* b;
    float* x;
    float* h_0;
    float* c_0;
    float* y;
    w_x = (float*)mkl_malloc(4 * hid * input_dim * sizeof (float), 64);
    w_h = (float*)mkl_malloc(4 * hid * hid * sizeof (float), 64);
    w = (float*)mkl_malloc(4 * hid * (input_dim+hid) * sizeof (float), 64);
    //b = NULL;
    b = (float*)mkl_malloc(4 * hid * sizeof (float), 64);
    x = (float*)mkl_malloc(time_step * input_dim * batch_size * sizeof (float), 64);
    h_0 = (float*)mkl_malloc(hid * batch_size * sizeof (float), 64);
    c_0 = (float*)mkl_malloc(hid * batch_size * sizeof (float), 64);
    y = (float*)mkl_malloc(hid * batch_size * sizeof (float), 64);
    memset(y, 0, sizeof(float) * hid * batch_size);
    for (i = 0; i < 4 * hid * input_dim; i++) {
        w_x[i] = ((float)rand()/(float)RAND_MAX) - ((float)rand()/(float)RAND_MAX);
    }
    for (i = 0; i < 4 * hid * hid; i++) {
        w_h[i] = ((float)rand()/(float)RAND_MAX) - ((float)rand()/(float)RAND_MAX);
    }
    for (i = 0; i < 4 * hid; i++) {
        //b[i] = ((float)rand()/(float)RAND_MAX);
        b[i]=0;
    }
    for (i = 0; i < time_step * input_dim * batch_size; i++) {
        x[i] = ((float)rand()/(float)RAND_MAX * 2.0f - 1.0f);
    }
    for (i = 0; i < hid * batch_size; i++) {
        h_0[i] = ((float)rand()/(float)RAND_MAX);
    }
    for (i = 0; i < hid * batch_size; i++) {
        c_0[i] = ((float)rand()/(float)RAND_MAX);
    }

    struct timeval start, end;

    //wx_sequential_gemm
    gettimeofday(&start, NULL);
    LSTM_sequential_gemm(batch_size, time_step, input_dim, hid, w_x, w_h, b, x, h_0, c_0, y, return_sequences,f_t,i_t,c_wave_t,o_t,c_t,gemmC);       
    gettimeofday(&end, NULL);
    printf("time: %.8f\n",  (end.tv_sec-start.tv_sec) + (end.tv_usec-start.tv_usec)*1e-6);
    if ( output_flag == 1) {
        printf("output:");
        for (i = 0; i < 10; i++) {
            printf( "%f ",y[i]);
        }
    }
    printf( "\n\n");   
    
    //wx_pack_gemm
    gettimeofday(&start, NULL);
    LSTM_pack_gemm(batch_size, time_step, input_dim, hid, w_x, w_h, b, x, h_0, c_0, y, return_sequences,f_t,i_t,c_wave_t,o_t,c_t,gemmC);       
    gettimeofday(&end, NULL);
    printf("time: %.8f\n",  (end.tv_sec-start.tv_sec) + (end.tv_usec-start.tv_usec)*1e-6);
    if ( output_flag == 1) {
        printf("output:");
        for (i = 0; i < 10; i++) {
            printf( "%f ",y[i]);
        }
    }
    printf( "\n\n");
 
    //wx_combine_gemm
    gettimeofday(&start, NULL);
    for(i=0;i<4*hid;i++){
        memcpy(w+i*(input_dim+hid),w_x+i*input_dim,input_dim*sizeof(float));
        memcpy(w+i*(input_dim+hid)+input_dim,w_h+i*hid,hid*sizeof(float));
    }
    LSTM_combine_gemm(batch_size, time_step, input_dim, hid, w, w_h, b, x, h_0, c_0, y, return_sequences,f_t,i_t,c_wave_t,o_t,c_t,gemmB,gemmC);
    gettimeofday(&end, NULL);
    printf("time: %.8f\n",  (end.tv_sec-start.tv_sec) + (end.tv_usec-start.tv_usec)*1e-6);
    if ( output_flag == 1) {
        printf("output:");
        for (i = 0; i < 10; i++) {
            printf( "%f ",y[i]);
        }
    }
    printf( "\n\n");
    
    //wx_combine_pack_gemm
    gettimeofday(&start, NULL);
    LSTM_combine_pack_gemm(batch_size, time_step, input_dim, hid, w, w_h, b, x, h_0, c_0, y, return_sequences,f_t,i_t,c_wave_t,o_t,c_t,gemmB,gemmC);       
    gettimeofday(&end, NULL);
    printf("time: %.8f\n",  (end.tv_sec-start.tv_sec) + (end.tv_usec-start.tv_usec)*1e-6);
    if ( output_flag == 1) {
        printf("output:");
        for (i = 0; i < 10; i++) {
            printf( "%f ",y[i]);
        }
    }
    printf( "\n\n");
 
    //wx_batch_gemm
    gettimeofday(&start, NULL);
    LSTM_batch_gemm(batch_size, time_step, input_dim, hid, w_x, w_h, b, x, h_0, c_0, y, return_sequences,f_t,i_t,c_wave_t,o_t,c_t,A,B,C,x_temp);       
    gettimeofday(&end, NULL);
    printf("time: %.8f\n",  (end.tv_sec-start.tv_sec) + (end.tv_usec-start.tv_usec)*1e-6);
    if ( output_flag == 1) {
        printf("output:");
        for (i = 0; i < 10; i++) {
            printf( "%f ",y[i]);
        }
    }
    printf( "\n\n");

    //wx_h_pack_gemm
    memset(x_temp, 0, sizeof(float) * max_len * 4 * batch_size * hid);
    gettimeofday(&start, NULL);
    LSTM_h_pack_gemm(batch_size, time_step, input_dim, hid, w_x, w_h, b, x, h_0, c_0, y, return_sequences,f_t,i_t,c_wave_t,o_t,c_t,A,B,C,A_pack, x_temp);
    gettimeofday(&end, NULL);
    printf("time: %.8f\n",  (end.tv_sec-start.tv_sec) + (end.tv_usec-start.tv_usec)*1e-6);
    if ( output_flag == 1) {
        printf("output:");
        for (i = 0; i < 10; i++) {
            printf( "%f ",y[i]);
        }
    }
    printf( "\n\n");

    mkl_free(A);
    mkl_free(B);
    mkl_free(A_pack);
    mkl_free(C);
    mkl_free(gemmC);
    mkl_free(h_t);
    mkl_free(x_temp);
    mkl_free(f_t);
    mkl_free(i_t);
    mkl_free(c_wave_t);
    mkl_free(o_t);
    mkl_free(c_t);
    mkl_free(w_x);
    mkl_free(w_h);
    mkl_free(x);
    mkl_free(h_0);
    mkl_free(c_0);
    mkl_free(y);
}

