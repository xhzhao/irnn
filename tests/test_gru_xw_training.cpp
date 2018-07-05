#include <iostream>
#include <sys/time.h>
#include <rnn.h>
#include <mkl.h>

#define count 100
#define WX_Pattern  0
#define XW_Pattern  1
int pattern = 1;
int num_direction = 1;
void test(
    int batch_size,
    int time_step,
    int input_dim,
    int hid,
    float* buf,
    float* w_x,     //(4H, I) 
    float* w_h,     //(4H, H)
    float* b_x,       //(4H)
    float* b_h,       //(4H)
    float* x,       //(T, I, N)
    float* h_0,     //(H, N)
    float* h_out,   //(T, H, N)
    float* grad_h_out,   //(T, H, N)
    float* hy,
    float* grad_hy,
    float* grad_wx,
    float* grad_wh,
    float* grad_bx,
    float* grad_bh,
    float* grad_x,
    float* grad_h0,
    int mode
){
    struct timeval start, end;
    int warmup = 3;
    int num_layer = 1;
    double fwd_time[4] = {0,0,0,0};
    double bwd_time[4] = {0,0,0,0};
    double time[4] = {0,0,0,0};
    for(int j = 0; j < warmup; j++) {
        if (pattern == WX_Pattern) {
            continue;
        }
        else if(pattern == XW_Pattern) {
            gru_xw_forward_prof(num_layer, time_step, num_direction, batch_size,
              input_dim, hid, x, h_0, w_x, w_h, b_x, b_h, h_out, hy, buf, mode, 
              time);
            gru_xw_backward_prof(num_layer, time_step, num_direction, batch_size,
              input_dim, hid, grad_h_out, grad_hy, x, h_0, w_x, w_h, grad_x, 
              grad_h0, grad_wx, grad_wh, grad_bx, grad_bh, buf, mode, time);
        }
    }

    double fwd_dura = 0;
    double tic = dsecnd();
    for(int i=0; i<count; ++i){
        double start = dsecnd();
        if (pattern == WX_Pattern) {
            continue;
        }
        else if(pattern == XW_Pattern) {
            gru_xw_forward_prof(num_layer, time_step, num_direction, batch_size,
              input_dim, hid, x, h_0, w_x, w_h, b_x, b_h, h_out, hy, buf, mode, 
              fwd_time);
            fwd_dura += (dsecnd() - start);
            gru_xw_backward_prof(num_layer, time_step, num_direction, batch_size,
              input_dim, hid, grad_h_out, grad_hy, x, h_0, w_x, w_h, grad_x, 
              grad_h0, grad_wx, grad_wh, grad_bx, grad_bh, buf, mode, bwd_time);
        }
    }
    double dura = dsecnd()-tic;
    double bwd_dura = dura - fwd_dura;

    printf("pattern = %d, N = %d, T = %d, I = %d, H = %d, SPS = %.4f\n", 
      pattern, batch_size, time_step, input_dim, hid, batch_size*count/dura);
    printf("fwd_dura = %f, bwd_dura = %f\n", fwd_dura, bwd_dura);
    printf("fwd_time: %f, %f, %f, %f\n", fwd_time[0], fwd_time[1], fwd_time[2], fwd_time[3]);
    printf("bwd_time: %f, %f, %f, %f\n", bwd_time[0], bwd_time[1], bwd_time[2], bwd_time[3]);
/*
    printf("pattern = %d, N = %d, T = %d, I = %d, H = %d, mode = %d,\ 
      total_time = %.4f, fwd_time=%.4f, bdw_time=%.4f, SPS = %.4f\n", 
      pattern,batch_size, time_step, input_dim, hid, mode, dura,fwd_dura,
      bdw_dura,batch_size*count/dura);*/
}


int test_main(int batch_size, int time_step, int input_dim, int hid) {
    int output_flag = 0;
    srand(45678);
    long i;
    //int max_time_step = time_step;
    //int max_batch_size = batch_size;
    //bool return_sequences = true;
    float * buf;
    float* w_x;      //(I, 3H)
    float* grad_wx;  //(I, 3H)
    float* w_h;      //(H, 3H)
    float* grad_wh;  //(H, 3H)
    float* b_x;      //(3H)
    float* grad_bx;  //(3H)
    float* b_h;      //(3H)
    float* grad_bh;  //(3H)
    float* x;        //(T, N, I)
    float* grad_x;   //(T, N, I)
    float* h_0;      //(N, H)
    float* grad_h0;  //(N, H)
    float* h_out;    //(T, N, H)
    float* grad_h_out;    //(T, N, H)
    float* hy;  //(T, N, H)
    float* grad_hy;
    int D = num_direction;
    w_x = (float*)mkl_malloc(D * 3 * hid * input_dim * sizeof(float) ,64);
    grad_wx = (float*)mkl_malloc(D * 3 * hid * input_dim * sizeof(float) ,64);
    w_h = (float*)mkl_malloc(D * 3 * hid * hid * sizeof(float),64);
    grad_wh = (float*)mkl_malloc(D * 3 * hid * hid * sizeof(float),64);
    b_x = (float*)mkl_malloc(D * 3 * hid * sizeof(float),64);
    grad_bx = (float*)mkl_malloc(D * 3 * hid * sizeof(float),64);
    b_h = (float*)mkl_malloc(D * 3 * hid * sizeof(float),64);
    grad_bh = (float*)mkl_malloc(D * 3 * hid * sizeof(float),64);
    x = (float*)mkl_malloc(time_step * input_dim * batch_size * sizeof (float),64);
    grad_x = (float*)mkl_malloc(time_step * input_dim * batch_size * sizeof (float),64);
    h_0 = (float*)mkl_malloc(D * hid * batch_size * sizeof(float),64);
    grad_h0 = (float*)mkl_malloc(D * hid * batch_size * sizeof(float),64);
    h_out = (float*)mkl_malloc(D * time_step *hid * batch_size * sizeof(float),64);
    grad_h_out = (float*)mkl_malloc(D * time_step *hid * batch_size * sizeof(float),64);
    hy = (float*)mkl_malloc(D * hid * batch_size * sizeof(float),64);
    grad_hy = (float*)mkl_malloc(D * hid * batch_size * sizeof(float),64);
    memset(h_out, 0, sizeof(float) * time_step * hid * batch_size * D);
    memset(grad_wx, 0, sizeof(float) * 3 * hid * input_dim * D);

    for (i = 0; i < D * 3 * hid * input_dim; i++) {
        w_x[i] = ((float)rand()/(float)RAND_MAX) - ((float)rand()/(float)RAND_MAX);
    }

    for (i = 0; i < D * 3 * hid * hid; i++) {
        w_h[i] = ((float)rand()/(float)RAND_MAX) - ((float)rand()/(float)RAND_MAX);
    }
    for (i = 0; i < D * 3 * hid; i++) {
        b_x[i]=0;
        b_h[i]=0;
    }
    for (i = 0; i < time_step * input_dim * batch_size; i++) {
        x[i] = ((float)rand()/(float)RAND_MAX * 2.0f - 1.0f);
    }
    for (i = 0; i < D * hid * batch_size; i++) {
        h_0[i] = ((float)rand()/(float)RAND_MAX);
    }
    for (i = 0; i < D * hid * batch_size; i++) {
        grad_hy[i] = ((float)rand()/(float)RAND_MAX);
    }
    for (i = 0; i <  D * time_step * hid * batch_size; i++) {
        grad_h_out[i] = ((float)rand()/(float)RAND_MAX);
    }
    int buf_size = gru_xw_train_get_workspace_size(
                     input_dim, hid, time_step, batch_size, (D==2), 1);
    buf = (float*)mkl_malloc(buf_size*sizeof(float), 64);
    test(batch_size, time_step, input_dim, hid, buf, w_x, w_h, b_x, b_h, x, h_0,
      h_out, grad_h_out, hy, grad_hy, grad_wx, grad_wh, grad_bx, grad_bh, 
      grad_x, grad_h0,0);

    mkl_free(buf);
    mkl_free(w_x);
    mkl_free(w_h);
    mkl_free(b_x);
    mkl_free(b_h);
    mkl_free(x);
    mkl_free(h_0);
    mkl_free(hy);
    mkl_free(h_out);
    mkl_free(grad_h_out);
    mkl_free(grad_wx);
    mkl_free(grad_wh);
    mkl_free(grad_bx);
    mkl_free(grad_bh);
    mkl_free(grad_x);
    mkl_free(grad_h0);
    mkl_free(grad_hy);
    
}


void main(int argc, char ** argv)
{
    printf("argc = %d, argv[0] = %s, argv[1] = %s \n",argc,argv[0],argv[1]);
    std::string p = "xw";
    if (argc > 1) {
        p = argv[1];
        if (p == "wx" ) {
            pattern = WX_Pattern;
        }
        else if (p == "xw" ) {
            pattern = XW_Pattern;
        }
        else {
            std::cout << "invalid pattern, will use wx as default" << std::endl;
            p = "xw";
            pattern = XW_Pattern;
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
    for(i = 0; i < 18; i++){
        test_main(sizes[i][0], sizes[i][1], sizes[i][2], sizes[i][3]);
    }

}
