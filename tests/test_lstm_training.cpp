#include <iostream>
#include <sys/time.h>
#include <rnn.h>
#define max(a,b) ((a) > (b) ? (a) : (b))

int main() {
    printf("test main \n");
    int output_flag = 1;
    srand(45678);
    int i;
    int batch_size = 64;
    int time_step = 10;
    int input_dim = 150;
    int hid = 1024;
    int max_time_step = time_step;
    int max_batch_size = batch_size;
    bool return_sequences = false;
    void * buf;
    float* w_x;     //(4H, I) 
    float* w_x2;     //(4H, I) 
    float* w_h;     //(4H, H) 
    float* w_h2;     //(4H, H) 
    float* b;       //(4H)
    float* x;       //(T, I, N)
    float* h_0;     //(H, N)
    float* c_0;     //(H, N)
    float* grad_last;//(H,N)
    float* dall;   //all gradients
    int temp_size = max(4 * max_time_step * hid * max_batch_size, 4 * max_time_step * hid * input_dim);
    temp_size = max(temp_size, 4 * max_time_step * hid * hid);
    temp_size = max(temp_size, 4 * max_time_step * input_dim * max_batch_size);
    buf = (void*)mkl_malloc(12 * max_time_step * sizeof (float*) + (max_time_step * 14 * max_batch_size * hid + 4 * hid * max_batch_size + temp_size) * sizeof (float), 64);
    w_x = (float*)mkl_malloc(4 * hid * input_dim * sizeof (float), 64);
    w_x2 = (float*)mkl_malloc(4 * hid * input_dim * sizeof (float), 64);
    w_h = (float*)mkl_malloc(4 * hid * hid * sizeof (float), 64);
    w_h2 = (float*)mkl_malloc(4 * hid * hid * sizeof (float), 64);
    b = (float*)mkl_malloc(4 * hid * sizeof (float), 64);
    x = (float*)mkl_malloc(time_step * input_dim * batch_size * sizeof (float), 64);
    h_0 = (float*)mkl_malloc(hid * batch_size * sizeof (float), 64);
    c_0 = (float*)mkl_malloc(hid * batch_size * sizeof (float), 64);
    grad_last = (float*)mkl_malloc(hid * batch_size * sizeof (float), 64);
    dall = (float*)mkl_malloc( (hid * input_dim * 4 + hid * hid * 4 + hid * 4 + time_step * input_dim * batch_size) * sizeof (float), 64);
    memset(dall, 0, sizeof(float) * (hid * input_dim * 4 + hid * hid * 4 + hid * 4 + time_step * input_dim * batch_size));
    for (i = 0; i < 4 * hid * input_dim; i++) {
        w_x[i] = ((float)rand()/(float)RAND_MAX) - ((float)rand()/(float)RAND_MAX);
    }
    for (i = 0; i < 4 * hid * hid; i++) {
        w_h[i] = ((float)rand()/(float)RAND_MAX) - ((float)rand()/(float)RAND_MAX);
    }
    for (i = 0; i < hid * input_dim; i++) {
        w_x2[i] = w_x[i + hid * input_dim];
    }
    for (i = hid * input_dim; i < hid * input_dim * 2; i++) {
        w_x2[i] = w_x[i - hid * input_dim];
    }
    for (i = hid * input_dim * 2; i < hid * input_dim * 4; i++) {
        w_x2[i] = w_x[i];
    }
    for (i = 0; i < hid * hid; i++) {
        w_h2[i] = w_h[i + hid * hid];
    }
    for (i = hid * hid; i < hid * hid * 2; i++) {
        w_h2[i] = w_h[i - hid * hid];
    }
    for (i = hid * hid * 2; i < hid * hid * 4; i++) {
        w_h2[i] = w_h[i];
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
    for (i = 0; i < hid * batch_size; i++) {
        grad_last[i] = 1.0;
    }
    memset(buf, 0, sizeof(void)*(12 * max_time_step * sizeof (float*) + (max_time_step * 14 * max_batch_size * hid + 4 * hid * max_batch_size + temp_size) * sizeof (float)));
    int buf_size = lstm_wx_train_get_workspace_size(input_dim, hid, max_time_step, max_batch_size);
    lstm_wx_training(buf, batch_size, time_step, input_dim, hid, max_time_step, max_batch_size, w_x2, w_h2, b, x, h_0, c_0, grad_last, dall);
   
    if ( output_flag == 1) {
        printf("buf_size = %d\n", buf_size);
        printf("dall:");
        for (i = 0; i < 10; i++) {
            printf( "%f ",dall[i]);
        }
        printf( "\n");
    }
    //memset(buf, 0, sizeof(void)*(16 * max_time_step * sizeof (float*) + (max_time_step * 4 * max_batch_size * hid +
    //          max_batch_size * (input_dim + hid) + 8 * max_batch_size * hid + 4 * hid * (hid + input_dim)) * sizeof (float)));
    memset(dall, 0, sizeof(float) * (hid * input_dim * 4 + hid * hid * 4 + hid * 4 + time_step * input_dim * batch_size));
    lstm_wx_training(buf, batch_size, time_step, input_dim, hid, max_time_step, max_batch_size, w_x2, w_h2, b, x, h_0, c_0, grad_last, dall);
    if ( output_flag == 1) {
        printf("buf_size = %d\n", buf_size);
        printf("dall:");
        for (i = 0; i < 10; i++) {
            printf( "%f ",dall[i]);
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
    mkl_free(grad_last);
    mkl_free(dall);
}
