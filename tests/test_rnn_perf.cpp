#include <iostream>
#include <sys/time.h>
#include <rnn.h>
#include <mkl.h>
#define count 20
void test(
    int output_flag,
    int i,
    int batch_size,
    int time_step,
    int input_dim,
    int hid,
    int max_time_step,
    int max_batch_size,
    bool return_sequences,
    void * buf,
    float* w_x,     //(4H, I) 
    float* w_h,     //(4H, H) 
    float* b,       //(4H)
    float* x,       //(T, I, N)
    float* h_0,     //(H, N)
    float* h_out,   //if return_sequences == true, size = (T, H, N), else size = (H, N)
    int mode){

    struct timeval start, end;
    rnn_xw_infer(buf, batch_size, time_step, input_dim, hid, max_time_step, max_batch_size, w_x, w_h, b, x, h_0, h_out, return_sequences,mode);
    rnn_xw_infer(buf, batch_size, time_step, input_dim, hid, max_time_step, max_batch_size, w_x, w_h, b, x, h_0, h_out, return_sequences,mode);
    //gettimeofday(&start, NULL);
    double tic = dsecnd();
    for(i=0; i<count; ++i){
        rnn_xw_infer(buf, batch_size, time_step, input_dim, hid, max_time_step, max_batch_size, w_x, w_h, b, x, h_0, h_out, return_sequences,mode);
    }
    //gettimeofday(&end, NULL);
    double dura = dsecnd()-tic;
    printf("N = %d, T = %d, I = %d, H = %d, mode = %d, time(us/sentence) = %.4f, SPS = %.4f\n",batch_size, time_step, input_dim, hid,
        mode, dura*1e6/count/batch_size,batch_size*count/dura);
    //printf("model = %d, start = %d  %d, end = %d %d, count = %d, batch_size = %d\n",mode,start.tv_sec,start.tv_usec,end.tv_sec,end.tv_usec,count,batch_size);

    if ( output_flag == 1) {
        printf("h_out:");
        for (i = 0; i < 10; i++) {
            printf( "%f ",h_out[i]);
        }
        printf( "\n");
    }

}


int test_main(int batch_size, int time_step, int input_dim, int hid) {
    printf("test main \n");
    int output_flag = 0;
    srand(206);
    int i;
    //int batch_size = 64;
    //int time_step = 25;
    //int input_dim = 1024;
    //int hid = 1024;
    int max_time_step = time_step;
    int max_batch_size = batch_size;
    bool return_sequences = true;
    void * buf;
    float* w_x;     //(4H, I) 
    float* w_h;     //(4H, H) 
    float* b;       //(4H)
    float* x;       //(T, I, N)
    float* h_0;     //(H, N)
    float* c_0;     //(H, N)
    float* h_out;   //if return_sequences == true, size = (T, H, N), else size = (H, N)
    float* c_out;   //(H, N)
    // buf = (void*)mkl_malloc(16 * max_time_step * sizeof (float*) + (max_time_step * 4 * max_batch_size * hid + 
    //      max_batch_size * (input_dim + hid) + 8 * max_batch_size * hid + 4 * hid * (hid + input_dim)) * sizeof (float), 64);
    w_x = (float*)mkl_malloc(4 * hid * input_dim * sizeof (float), 64);
    w_h = (float*)mkl_malloc(4 * hid * hid * sizeof (float), 64);
    b = (float*)mkl_malloc(4 * hid * sizeof (float), 64);
    x = (float*)mkl_malloc(time_step * input_dim * batch_size * sizeof (float), 64);
    h_0 = (float*)mkl_malloc(hid * batch_size * sizeof (float), 64);
    c_0 = (float*)mkl_malloc(hid * batch_size * sizeof (float), 64);
    h_out = (float*)mkl_malloc(max_time_step *hid * batch_size * sizeof (float), 64);
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
        b[i]=(float)rand()/(float)RAND_MAX;
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
    int buf_size = rnn_xw_infer_get_workspace_size(input_dim, hid, max_time_step, max_batch_size);
    buf = (void*)mkl_malloc(buf_size, 64); 
    test(output_flag,i,batch_size,time_step,input_dim,hid,max_time_step,max_batch_size,return_sequences,buf, w_x,w_h, b,x, h_0, h_out, 0);
    test(output_flag,i,batch_size,time_step,input_dim,hid,max_time_step,max_batch_size,return_sequences,buf, w_x,w_h, b,x, h_0, h_out, 1);
    test(output_flag,i,batch_size,time_step,input_dim,hid,max_time_step,max_batch_size,return_sequences,buf, w_x,w_h, b,x, h_0, h_out, 2);
    test(output_flag,i,batch_size,time_step,input_dim,hid,max_time_step,max_batch_size,return_sequences,buf, w_x,w_h, b,x, h_0, h_out, 3);
    test(output_flag,i,batch_size,time_step,input_dim,hid,max_time_step,max_batch_size,return_sequences,buf, w_x,w_h, b,x, h_0, h_out, 4);
    test(output_flag,i,batch_size,time_step,input_dim,hid,max_time_step,max_batch_size,return_sequences,buf, w_x,w_h, b,x, h_0, h_out, 5);


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


void main()
{
    int sizes[24][4] = {{64,15,500,500},
         {64,20,500,500},
         {64,25,500,500},
         {64,30,500,500},
         {64,35,500,500},
         {64,40,500,500},
         {64,45,500,500},
         {64,50,500,500},
         {16,25,512,512},
         {32,25,512,512},
         {64,25,512,512},
         {128,25,512,512},
         {16,25,1024,1024},
         {32,25,1024,1024},
         {64,25,1024,1024},
         {128,25,1024,1024},
         {16,25,2048,2048},
         {32,25,2048,2048},
         {64,25,2048,2048},
         {128,25,2048,2048},
         {16,25,4096,4096},
         {32,25,4096,4096},
         {64,25,4096,4096},
         {128,25,4096,4096}
        };

    //sizes[0] = {64, 15, 500, 500};

    //int * size = sizes;

    int i = 0;
    for(i = 0; i < 24; i++){
        test_main(sizes[i][0], sizes[i][1], sizes[i][2], sizes[i][3]);
    }

}
