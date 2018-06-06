#include <iostream>
#include <sys/time.h>
#include <rnn.h>

int main() {
    printf("test main \n");
    int output_flag = 1;
    srand(45678);
    int i;
    int batch_size = 64;
    int time_step = 20;
    int input_dim = 512;
    int hid = 512;
    int max_time_step = time_step;
    int max_batch_size = batch_size;
    bool return_sequences = false;
    void * buf;
    float* w_x;     //(4H, I) 
    float* w_h;     //(4H, H) 
    float* b;       //(4H)
    float* x;       //(T, I, N)
    float* h_0;     //(H, N)
    float* c_0;     //(H, N)
    float* h_out;   //if return_sequences == true, size = (T, H, N), else size = (H, N)
    float* c_out;   //(H, N)
    buf = (void*)mkl_malloc(16 * max_time_step * sizeof (float*) + (max_time_step * 4 * max_batch_size * hid + 
          max_batch_size * (input_dim + hid) + 8 * max_batch_size * hid + 4 * hid * (hid + input_dim)) * sizeof (float), 64);
    w_x = (float*)mkl_malloc(4 * hid * input_dim * sizeof (float), 64);
    w_h = (float*)mkl_malloc(4 * hid * hid * sizeof (float), 64);
    b = (float*)mkl_malloc(4 * hid * sizeof (float), 64);
    x = (float*)mkl_malloc(time_step * input_dim * batch_size * sizeof (float), 64);
    h_0 = (float*)mkl_malloc(hid * batch_size * sizeof (float), 64);
    c_0 = (float*)mkl_malloc(hid * batch_size * sizeof (float), 64);
    h_out = (float*)mkl_malloc(/*max_time_step * */hid * batch_size * sizeof (float), 64);
    c_out = (float*)mkl_malloc(hid * batch_size * sizeof (float), 64);
    memset(h_out, 0, sizeof(float) * hid * batch_size);
    memset(c_out, 0, sizeof(float) * hid * batch_size);
    for (i = 0; i < 4 * hid * input_dim; i++) {
        w_x[i] = ((float)rand()/(float)RAND_MAX) - ((float)rand()/(float)RAND_MAX);
    }
    for (i = 0; i < 4 * hid * hid; i++) {
        w_h[i] = ((float)rand()/(float)RAND_MAX) - ((float)rand()/(float)RAND_MAX);
    }
    for (i = 0; i < 4 * hid; i++) {
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

    int buf_size = lstm_wx_infer_get_workspace_size(input_dim, hid, max_time_step, max_batch_size);
    lstm_wx_infer(buf, batch_size, time_step, input_dim, hid, max_time_step, max_batch_size, w_x, w_h, b, x, h_0, c_0, h_out, c_out, return_sequences,
    0);
   
    if ( output_flag == 1) {
        printf("buf_size = %d\n", buf_size);
        printf("h_out:");
        for (i = 0; i < 10; i++) {
            printf( "%f ",h_out[i]);
        }
        printf( "\nc_out:");
        for (i = 0; i < 10; i++) {
            printf( "%f ",c_out[i]);
        }
        printf( "\n");
    }
        lstm_wx_infer(buf, batch_size, time_step, input_dim, hid, max_time_step, max_batch_size, w_x, w_h, b, x, h_0, c_0, h_out, c_out, return_sequences,
    1);
   
    if ( output_flag == 1) {
        printf("buf_size = %d\n", buf_size);
        printf("h_out:");
        for (i = 0; i < 10; i++) {
            printf( "%f ",h_out[i]);
        }
        printf( "\nc_out:");
        for (i = 0; i < 10; i++) {
            printf( "%f ",c_out[i]);
        }
        printf( "\n");
    }
   lstm_wx_infer(buf, batch_size, time_step, input_dim, hid, max_time_step, max_batch_size, w_x, w_h, b, x, h_0, c_0, h_out, c_out, return_sequences,
    2);
   
    if ( output_flag == 1) {
        printf("buf_size = %d\n", buf_size);
        printf("h_out:");
        for (i = 0; i < 10; i++) {
            printf( "%f ",h_out[i]);
        }
        printf( "\nc_out:");
        for (i = 0; i < 10; i++) {
            printf( "%f ",c_out[i]);
        }
        printf( "\n");
    }
   lstm_wx_infer(buf, batch_size, time_step, input_dim, hid, max_time_step, max_batch_size, w_x, w_h, b, x, h_0, c_0, h_out, c_out, return_sequences,
    3);
   
    if ( output_flag == 1) {
        printf("buf_size = %d\n", buf_size);
        printf("h_out:");
        for (i = 0; i < 10; i++) {
            printf( "%f ",h_out[i]);
        }
        printf( "\nc_out:");
        for (i = 0; i < 10; i++) {
            printf( "%f ",c_out[i]);
        }
        printf( "\n");
    }
   lstm_wx_infer(buf, batch_size, time_step, input_dim, hid, max_time_step, max_batch_size, w_x, w_h, b, x, h_0, c_0, h_out, c_out, return_sequences,
    4);
   
    if ( output_flag == 1) {
        printf("buf_size = %d\n", buf_size);
        printf("h_out:");
        for (i = 0; i < 10; i++) {
            printf( "%f ",h_out[i]);
        }
        printf( "\nc_out:");
        for (i = 0; i < 10; i++) {
            printf( "%f ",c_out[i]);
        }
        printf( "\n");
    }
       lstm_wx_infer(buf, batch_size, time_step, input_dim, hid, max_time_step, max_batch_size, w_x, w_h, b, x, h_0, c_0, h_out, c_out, return_sequences,
    5);
   
    if ( output_flag == 1) {
        printf("buf_size = %d\n", buf_size);
        printf("h_out:");
        for (i = 0; i < 10; i++) {
            printf( "%f ",h_out[i]);
        }
        printf( "\nc_out:");
        for (i = 0; i < 10; i++) {
            printf( "%f ",c_out[i]);
        }
        printf( "\n");
    }

    mkl_free(buf);
    mkl_free(w_x);
    mkl_free(w_h);
    mkl_free(b);
    mkl_free(x);
    mkl_free(h_0);
    mkl_free(c_0);
    mkl_free(h_out);
    mkl_free(c_out);
}
