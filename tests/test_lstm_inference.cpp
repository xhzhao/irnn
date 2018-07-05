#include <iostream>
#include <sys/time.h>
#include <rnn.h>
#include <mkl.h>
#define count 100
#define WX_Pattern  0
#define XW_Pattern  1
int pattern = 0;
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
    float* c_0,     //(H, N)
    float* h_out,   //if return_sequences == true, size = (T, H, N), else size = (H, N)
    float* c_out,   //(H, N)
    int mode
){
    struct timeval start, end;
    int warmup = 3;
    for(int j = 0; j < warmup; j++) {
        if (pattern == WX_Pattern) {
            lstm_wx_infer(buf, batch_size, time_step, input_dim, hid, max_time_step, max_batch_size, w_x, w_h, b, x, h_0, c_0, h_out, c_out,
                return_sequences,mode);
        }else if(pattern == XW_Pattern) {
            lstm_xw_infer(buf, batch_size, time_step, input_dim, hid, max_time_step, max_batch_size, w_x, w_h, b, x, h_0, c_0, h_out, c_out,
                return_sequences,mode);
        }
    }
    //gettimeofday(&start, NULL);
    double tic = dsecnd();
    for(i=0; i<count; ++i){
        if (pattern == WX_Pattern) {
            lstm_wx_infer(buf, batch_size, time_step, input_dim, hid, max_time_step, max_batch_size, w_x, w_h, b, x, h_0, c_0, h_out, c_out,
                return_sequences,mode);
        }else if (pattern == XW_Pattern) {
            lstm_xw_infer(buf, batch_size, time_step, input_dim, hid, max_time_step, max_batch_size, w_x, w_h, b, x, h_0, c_0, h_out, c_out,
                return_sequences,mode);
        }
    }
    //gettimeofday(&end, NULL);
    double dura = dsecnd()-tic;
    printf("pattern = %d, N = %d, T = %d, I = %d, H = %d, mode = %d, time(us/sentence) = %.4f, SPS = %.4f\n", pattern,batch_size, time_step, input_dim, hid,
        mode, dura*1e6/count/batch_size,batch_size*count/dura);
    //printf("model = %d, start = %d  %d, end = %d %d, count = %d, batch_size = %d\n",mode,start.tv_sec,start.tv_usec,end.tv_sec,end.tv_usec,count,batch_size);

    if ( output_flag == 1) {
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

}


int test_main(int batch_size, int time_step, int input_dim, int hid) {
    printf("test main \n");
    int output_flag = 0;
    srand(45678);
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
    buf = (void*)mkl_malloc(buf_size, 64); 
    test(output_flag,i,batch_size,time_step,input_dim,hid,max_time_step,max_batch_size,return_sequences,buf, w_x,w_h, b,x, h_0,c_0, h_out, c_out, 0);
    //test(output_flag,i,batch_size,time_step,input_dim,hid,max_time_step,max_batch_size,return_sequences,buf, w_x,w_h, b,x, h_0,c_0, h_out, c_out, 1);
    test(output_flag,i,batch_size,time_step,input_dim,hid,max_time_step,max_batch_size,return_sequences,buf, w_x,w_h, b,x, h_0,c_0, h_out, c_out, 2);

    //test(output_flag,i,batch_size,time_step,input_dim,hid,max_time_step,max_batch_size,return_sequences,buf, w_x,w_h, b,x, h_0,c_0, h_out, c_out, 3);
    //test(output_flag,i,batch_size,time_step,input_dim,hid,max_time_step,max_batch_size,return_sequences,buf, w_x,w_h, b,x, h_0,c_0, h_out, c_out, 4);
    //test(output_flag,i,batch_size,time_step,input_dim,hid,max_time_step,max_batch_size,return_sequences,buf, w_x,w_h, b,x, h_0,c_0, h_out, c_out, 5);

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


void main(int argc, char ** argv)
{
    printf("argc = %d, argv[0] = %s, argv[1] = %s \n",argc,argv[0],argv[1]);
    std::string p = "wx";
    if (argc > 1) {
        p = argv[1];
        if (p == "wx" ) {
            pattern = WX_Pattern;
        }
        else if (p == "xw" ) {
            pattern = XW_Pattern;
        }
        else {
            std::cout << "invalid pattern parameters, will use wx as default"  << std::endl;
            p = "wx";
            pattern = WX_Pattern;
        }
    }
    std::cout << "pattern = " << p <<  std::endl;
    int sizes[21][4] = {
         {20,1,800,800},
         {20,50,800,800},
         {20,100,800,800},
         //{20,150,800,800},
         {20,200,800,800},
         {20,300,800,800},
         {20,400,800,800},
         {12,1,1760,1760},
         {12,50,1760,1760},
         {12,100,1760,1760},
         //{12,150,1760,1760},
         {12,200,1760,1760},
         {12,300,1760,1760},
         {12,400,1760,1760},
         {32,1,1760,1760},
         {32,50,1760,1760},
         {32,100,1760,1760},
         //{32,150,1760,1760},
         {32,200,1760,1760},
         {32,300,1760,1760},
         {32,400,1760,1760}
};

    //sizes[0] = {64, 15, 500, 500};

    //int * size = sizes;

    int i = 0;
    for(i = 0; i < 24; i++){
        test_main(sizes[i][0], sizes[i][1], sizes[i][2], sizes[i][3]);
    }

}
