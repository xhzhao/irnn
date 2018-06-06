#include <iostream>
#include <sys/time.h>
#include <rnn.h>
#include <mkl.h>
#define count 100
#define WX_Pattern  0
#define XW_Pattern  1
int pattern = 0;
void test(
    int batch_size,
    int time_step,
    int input_dim,
    int hid,
    void * buf,
    float* w_x,     //(4H, I) 
    float* w_h,     //(4H, H)
    float* b,       //(4H)
    float* x,       //(T, I, N)
    float* h_0,     //(H, N)
    float* c_0,     //(H, N)
    float* h_out,   //(T, H, N)
    float* c_out,   //(T, H, N)
    float* grad_wx,
    float* grad_wh,
    float* grad_b,
    float* grad_x,
    float* grad_h0,
    float* grad_c0,
    float* grad_h_out,
    float* grad_c_out,
    int mode,
    int bidirectional
){
    struct timeval start, end;
    int warmup = 3;
    int num_layer = 1;
    int num_direction = bidirectional + 1;
    for(int j = 0; j < warmup; j++) {
        if (pattern == WX_Pattern) {
            //lstm_wx_forward(buf, batch_size, time_step, input_dim, hid, max_time_step, max_batch_size, w_x, w_h, b, x, h_0, c_0, h_out, c_out,
            //    return_sequences,mode);
        }else if(pattern == XW_Pattern) {
            lstm_xw_forward(buf, batch_size, time_step, input_dim, hid, w_x, w_h, b, x, h_0, c_0, h_out, c_out,mode, bidirectional);
            lstm_xw_backward(buf,num_layer,num_direction,time_step,batch_size,input_dim,hid,x,h_0,c_0,w_x,w_h,h_out, c_out,
               grad_h_out,grad_c_out, grad_x,grad_h0, grad_c0, grad_wx,grad_wh,grad_b,mode);
        }
    }

    double fwd_dura = 0;
    double tic = dsecnd();
    for(int i=0; i<count; ++i){
        double start = dsecnd();
        if (pattern == WX_Pattern) {
            //lstm_wx_forward(buf, batch_size, time_step, input_dim, hid, max_time_step, max_batch_size, w_x, w_h, b, x, h_0, c_0, h_out, c_out,
            //    return_sequences,mode);
        }else if (pattern == XW_Pattern) {
            lstm_xw_forward(buf, batch_size, time_step, input_dim, hid, w_x, w_h, b, x, h_0, c_0, h_out, c_out,mode);
            fwd_dura += (dsecnd() - start);
            lstm_xw_backward(buf,num_layer,num_direction,time_step,batch_size,input_dim,hid,x,h_0,c_0,w_x,w_h,h_out, c_out,
               grad_h_out,grad_c_out, grad_x,grad_h0, grad_c0, grad_wx,grad_wh,grad_b,mode);
        }
    }
    double dura = dsecnd()-tic;
    double bdw_dura = dura - fwd_dura;

    printf("pattern = %d, D = %d, N = %d, T = %d, I = %d, H = %d, mode = %d, total_time = %.4f, fwd_time=%.4f, bdw_time=%.4f, SPS = %.4f\n", pattern,num_direction,batch_size, time_step, input_dim, hid,
        mode, dura,fwd_dura,bdw_dura,batch_size*count/dura);
}


int test_main(int batch_size, int time_step, int input_dim, int hid, int bidirectional) {
    printf("test main \n");
    int output_flag = 0;
    srand(45678);
    long i;
    //int batch_size = 64;
    //int time_step = 25;
    //int input_dim = 1024;
    //int hid = 1024;
    int num_direction = bidirectional + 1;
    int max_time_step = time_step;
    int max_batch_size = batch_size;
    bool return_sequences = true;
    void * buf;
    float* w_x;     //(4H, I)
    float* grad_wx; //(4H, I)
    float* w_h;     //(4H, H)
    float* grad_wh; //(4H, H)
    float* b;       //(4H)
    float* grad_b;  //(4H)
    float* x;       //(T, I, N)
    float* grad_x;  //(T, I, N)
    float* h_0;     //(H, N)
    float* grad_h0;     //(H, N)
    float* c_0;     //(H, N)
    float* grad_c0;     //(H, N)
    float* h_out;   //(T, H, N)
    float* grad_h_out;   //(T, H, N)
    float* c_out;   //(T, H, N)
    float* grad_c_out;   //(T, H, N)
    // buf = (void*)mkl_malloc(16 * max_time_step * sizeof (float*) + (max_time_step * 4 * max_batch_size * hid + 
    //      max_batch_size * (input_dim + hid) + 8 * max_batch_size * hid + 4 * hid * (hid + input_dim)) * sizeof (float), 64);
    w_x = (float*)mkl_malloc(num_direction * 4 * hid * input_dim * sizeof(float) ,64);
    grad_wx = (float*)mkl_malloc(num_direction * 4 * hid * input_dim * sizeof(float) ,64);
    w_h = (float*)mkl_malloc(num_direction * 4 * hid * hid * sizeof(float),64);
    grad_wh = (float*)mkl_malloc(num_direction * 4 * hid * hid * sizeof(float),64);
    b = (float*)mkl_malloc(num_direction * 4 * hid * sizeof(float),64);
    grad_b = (float*)mkl_malloc(num_direction * 4 * hid * sizeof(float),64);
    x = (float*)mkl_malloc(time_step * input_dim * batch_size * sizeof (float),64);
    grad_x = (float*)mkl_malloc(time_step * input_dim * batch_size * sizeof (float),64);
    h_0 = (float*)mkl_malloc(num_direction * hid * batch_size * sizeof(float),64);
    grad_h0 = (float*)mkl_malloc(num_direction * hid * batch_size * sizeof(float),64);
    c_0 = (float*)mkl_malloc(num_direction * hid * batch_size * sizeof(float),64);
    grad_c0 = (float*)mkl_malloc(num_direction * hid * batch_size * sizeof(float),64);
    h_out = (float*)mkl_malloc(num_direction * max_time_step *hid * batch_size * sizeof(float),64);
    grad_h_out = (float*)mkl_malloc(num_direction * max_time_step *hid * batch_size * sizeof(float),64);
    c_out = (float*)mkl_malloc(num_direction * max_time_step *hid * batch_size * sizeof(float),64);
    grad_c_out = (float*)mkl_malloc(num_direction * max_time_step *hid * batch_size * sizeof(float),64);
    memset(h_out, 0, sizeof(float) * num_direction * max_time_step* hid * batch_size);
    memset(c_out, 0, sizeof(float) * num_direction * max_time_step* hid * batch_size);
    memset(grad_wx, 0, sizeof(float) * num_direction * 4 * hid * input_dim);

    for (i = 0; i < num_direction * 4 * hid * input_dim; i++) {
        w_x[i] = ((float)rand()/(float)RAND_MAX) - ((float)rand()/(float)RAND_MAX);
    }

    for (i = 0; i < num_direction * 4 * hid * hid; i++) {
        w_h[i] = ((float)rand()/(float)RAND_MAX) - ((float)rand()/(float)RAND_MAX);
    }
    for (i = 0; i < num_direction * 4 * hid; i++) {
        b[i]=0;
    }
    for (i = 0; i < time_step * input_dim * batch_size; i++) {
        x[i] = ((float)rand()/(float)RAND_MAX * 2.0f - 1.0f);
    }
    for (i = 0; i < num_direction * hid * batch_size; i++) {
        h_0[i] = ((float)rand()/(float)RAND_MAX);
    }
    for (i = 0; i < num_direction * hid * batch_size; i++) {
        c_0[i] = ((float)rand()/(float)RAND_MAX);
    }
    for (i = 0; i < num_direction * max_time_step * hid * batch_size; i++) {
        grad_h_out[i] = ((float)rand()/(float)RAND_MAX);
        grad_c_out[i] = ((float)rand()/(float)RAND_MAX);
    }
    int buf_size = lstm_xw_train_get_workspace_size(input_dim, hid, max_time_step, max_batch_size, bidirectional);
    buf = (void*)mkl_malloc(buf_size*sizeof(float), 64);
    test(batch_size,time_step,input_dim,hid,buf, w_x,w_h, b,x, h_0,c_0, h_out, c_out,
       grad_wx,grad_wh, grad_b,grad_x, grad_h0,grad_c0, grad_h_out, grad_c_out,0, bidirectional);





    //test(output_flag,i,batch_size,time_step,input_dim,hid,max_time_step,max_batch_size,return_sequences,buf, w_x,w_h, b,x, h_0,c_0, h_out, c_out, 1);
    //test(output_flag,i,batch_size,time_step,input_dim,hid,max_time_step,max_batch_size,return_sequences,buf, w_x,w_h, b,x, h_0,c_0, h_out, c_out, 2);

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
    int bidirectional = 0;
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
            p = "xw";
            pattern = XW_Pattern;
        }

        if (argc > 2) {
            bidirectional = 1;
        }
    }
    std::cout << "pattern = " << p <<  std::endl;
    int sizes[24][4] = {
         {64,30,500,500},
         {64,40,500,500},
         {64,45,500,500},
         {64,50,500,500},
         {20,50,800,800},
         {20,100,800,800},
         {20,150,800,800},
         {20,200,800,800},
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
        test_main(sizes[i][0], sizes[i][1], sizes[i][2], sizes[i][3], bidirectional);
    }

}
