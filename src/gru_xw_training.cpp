#include <iostream>
#include <sys/time.h>
#include <rnn.h>
#include <cstddef>
#include <iostream>
#include <rnn.h>
#include <mkl.h>
#include <omp.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <assert.h>
#include <pthread.h>

//#define WS_THREAD  80  //80 float = 300Byte, 2 * 8 for threadid, 136 * 2 for message
#define MESSAGE_SIZE (64*3) // 192, nearest number to 136
//#define WS_THREAD  (64*6+16)  //394, 2 * 8 for threadid, 192 * 2 for message
#define WS_THREAD  0  //394, 2 * 8 for threadid, 192 * 2 for message

double get_time(void)
{
#if 1
    struct timeval start;
    gettimeofday(&start,NULL);
    double time = start.tv_sec * 1000 + start.tv_usec /1000;
    return time; 
#else
    double time = dsecnd() * 1000;
    return time;
#endif
}

void elemwise_opt(int D, int N, int H, float* rt, float* nt, float* zt,
    float* gemmC1_t, float* gemmC2, float* Mnht, float* bx, float* bh,
    float* ht, float* ht_1) {
    
    if (bx and bh){

        #pragma omp parallel for
        for (int i = 0; i < N; ++i) {
            int rtb = i * 3 * H;
            int ztb = i * 3 * H + H;
            int ntb = i * 3 * H + 2 * H;
            
            for (int j = 0; j < H; ++j) {

                Mnht[i * H + j] = gemmC2[ntb + j] + bh[2*H+j];
                gemmC2[rtb + j] = -gemmC1_t[rtb + j] - gemmC2[rtb + j] 
                    - bx[j] - bh[j];
                gemmC2[ztb + j] = -gemmC1_t[ztb + j]-gemmC2[ztb + j]   
                    - bx[H + j] - bh[H + j];
            }

            vsExp(2 * H, gemmC2 + rtb, gemmC2 + rtb);
            for (int j = 0; j < H; ++j) {                                       
                rt[i * H + j] = 1/(1 + gemmC2[rtb + j]);                   
                zt[i * H + j] = 1/(1 + gemmC2[ztb + j]);
                gemmC2[ntb + j] = gemmC1_t[ntb + j] + bx[2 * H + j] +           
                    rt[i * H + j] * (gemmC2[ntb + j] + bh[2 * H + j]);
            }
            vsTanh(H, gemmC2 + ntb, nt + i * H);
            for (int j = 0; j < H; ++j) {                                       
                ht[i * D * H + j] = (1-zt[i * H + j]) * nt[i * H + j] +         
                    zt[i * H + j] * ht_1[i * D * H + j];                        
            }
        }
    } else {

        #pragma omp parallel for
        for (int i = 0; i < N; ++i) {
            int rtb = i * 3 * H;
            int ztb = i * 3 * H + H;
            int ntb = i * 3 * H + 2 * H;
            
            for (int j = 0; j < H; ++j) {

                Mnht[i * H + j] = gemmC2[ntb + j];
                gemmC2[rtb + j] = -gemmC1_t[rtb + j] - gemmC2[rtb + j];
                gemmC2[ztb + j] = -gemmC1_t[ztb + j]-gemmC2[ztb + j];
            }

            vsExp(2 * H, gemmC2 + rtb, gemmC2 + rtb);
            for (int j = 0; j < H; ++j) {                                       
                rt[i * H + j] = 1/(1 + gemmC2[rtb + j]);                   
                zt[i * H + j] = 1/(1 + gemmC2[ztb + j]);
                gemmC2[ntb + j] = gemmC1_t[ntb + j] +
                    rt[i * H + j] * (gemmC2[ntb + j]);
            }
            vsTanh(H, gemmC2 + ntb, nt + i * H);
            for (int j = 0; j < H; ++j) {                                       
                ht[i * D * H + j] = (1-zt[i * H + j]) * nt[i * H + j] +         
                    zt[i * H + j] * ht_1[i * D * H + j];                        
            }
        }
    }

}

typedef struct fwd_thread_para{
    int T;
    int D;
    int N;
    int I;
    int H;
    float* x;
    float* hx;
    float* wx;
    float* wh;
    float* bx;
    float* bh;
    float* y;
    float* hy;
    float* gateR;
    float* gateZ;
    float* gateN;
    float* Mnh;
    float* ws;
    int d;
}FwdThreadPara;


#define COUNT 1
void* fwd_thread_one_direction(void * p){

    FwdThreadPara* para = (FwdThreadPara*)p;
    int T = para->T;
    int D = para->D;
    int N = para->N;
    int I = para->I;
    int H = para->H;
    int d = para->d;
    double start = get_time();
    for(int c =0; c < COUNT; c++){

    float* x = para->x;
    float* hx = para->hx;
    float* wx = para->wx;
    float* wh = para->wh;
    float* bx = para->bx;
    float* bh = para->bh;
    float* y = para->y;
    float* hy = para->hy;
    float* gateR = para->gateR;
    float* gateZ = para->gateZ;
    float* gateN = para->gateN;
    float* Mnh = para->Mnh;
    float* ws = para->ws;

    int m = N;

    int n = 3*H;
    int k = I;
    float* ht = y;
    float* ht_1 = y;
    float* back_ht_1 = y + (T-1)*N*H*D + H;
    float* back_ht = back_ht_1;

    float* gemmC1  = ws;              // [D,T,N,3H]
    float* gemmC2  = gemmC1 + D*T*N*3*H;  // N*3H
    float* rt = gateR;
    float* zt = gateZ;
    float* nt = gateN;
    float* back_wx = wx + I * 3 * H;
    float* back_wh = wh + H * 3 * H;
    float* back_bx = (bx != NULL)? bx + 3 * H : NULL;
    float* back_bh = (bh != NULL)? bh + 3 * H : NULL;
    float* back_gateR = gateR + T * N * H;
    float* back_gateZ = gateZ + T * N * H;
    float* back_gateN = gateN + T * N * H;
    float* back_Mnh = Mnh + T * N * H;
    float* back_gemmC1 = gemmC1 + T * N * 3 * H;
    float* back_gemmC2 = gemmC2 + N * 3 * H;
    float* gemmC1_t = gemmC1;

    int cores = 40 ? (D==1):20;
    if(D == 1){
        omp_set_num_threads(40);
    }else{
        mkl_set_num_threads_local(20);
        omp_set_num_threads(20);
    }

    if(d == 0){
        //bind pthread to socket2
        if(D ==2){
            cpu_set_t cpuset;
            pthread_t thread = pthread_self();
            CPU_ZERO(&cpuset);
            for (int i = 20; i < 40; i++){
                CPU_SET(i, &cpuset);
            }
            pthread_setaffinity_np(thread, sizeof(cpu_set_t), &cpuset);
        }

        #pragma omp parallel for collapse(2)
        for (int i = 0; i < N; i++){
            for (int j = 0; j < H; j++) {
                y[i * D * H + j] = hx[i * H + j];
            }
        }
        //x*wx : [T*N,I] * [I,3H]
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N*T, n, k, 1, x, k, 
          wx, n, 0.0, gemmC1, n);
        for (int t =0; t < T; t++) {
            //  perform the first direction, X*wx and H * wh for each step
            //  ht-1*wh, ht-1:[N,H] wh:[H, 3H]
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, H, 1, 
                        ht_1, D*H, wh, n, 0.0, gemmC2, n);
            rt = gateR + t * N * H;
            zt = gateZ + t * N * H;
            nt = gateN + t * N * H;
            gemmC1_t = gemmC1 + t * N * 3 * H;
            float* Mnht = Mnh + t * N * H;
            elemwise_opt(D, N, H, rt, nt, zt, gemmC1_t, gemmC2, Mnht, bx, bh, ht, ht_1);
            ht_1 = ht;
            ht = ht + D * H * N;
        }
        float* y_start = y + (T - 1) * N * H * D;
        #pragma omp parallel for collapse(2)
        for (int i = 0; i < N; i++){
            for (int j = 0; j < H; j++) {
                hy[i * H + j] = y_start[i * D * H + j];
            }
        }
    } else if(d == 1){
        cpu_set_t cpuset;
        pthread_t thread = pthread_self();
        CPU_ZERO(&cpuset);
        for (int i = 0; i < 20 ; i++){
            CPU_SET(i, &cpuset);
        }
        pthread_setaffinity_np(thread, sizeof(cpu_set_t), &cpuset);
        #pragma omp parallel for collapse(2)
        for (int i = 0; i < N; i++){
            for (int j = 0; j < H; j++) {
                back_ht_1[i *D * H + j] = hx[N * H + i * H + j];
            }
        }
        cblas_sgemm(CblasRowMajor,CblasNoTrans, CblasNoTrans, N*T, n, k, 1, x, 
          k, back_wx, n, 0.0, back_gemmC1, n);
        for (int t =0; t < T; t++) {
            rt = back_gateR + (T - 1 - t) * N * H;
            zt = back_gateZ + (T - 1 - t) * N * H;
            nt = back_gateN + (T - 1 - t) * N * H;
            gemmC1_t = back_gemmC1 + (T - 1 - t) * N * 3 * H;
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, H, 1, 
                        back_ht_1, D * H, back_wh, n, 0.0, back_gemmC2, n);

            float* back_Mnht = back_Mnh + (T-1-t)*N*H;
            elemwise_opt(D, N, H, rt, nt, zt, gemmC1_t, back_gemmC2, back_Mnht, back_bx, back_bh, back_ht, back_ht_1);
            back_ht_1 = back_ht;
            back_ht = back_ht - D * H * N;
        }
        float* y_back_start = y + H;
        #pragma omp parallel for collapse(2)
        for (int i = 0; i < N; i++){
            for (int j = 0; j < H; j++) {
                hy[N * H + i * H + j] = y_back_start[i * D * H + j];
            }
        }
    }
    }
#if 0
    double dura = (get_time() - start) /1e3;                                                  
    float SPS = N * COUNT / dura;                                               
    double one_iter = 0;                                                        
    one_iter = (N * H * I * 2 + N * H * H * 2) / 1e6 * 3;            
    double GFLOPS = COUNT * one_iter * T / dura / 1e3;                      
    printf("thread inside d = %d, L = 1, D = %d, N = %d, T = %d, I = %d, H = %d, GFLOPS = %.4f, SPS = %.4f\n", d, D, N, T, I, H, GFLOPS, SPS);
#endif


}



/*
 * @brief:  gru_forward_single_sequential
 *          single layer forward computation 
 *
 * @params: T:  seq_length
 *          D:  direction, D = 2 if bidirectional else 1
 *          N:  batch_size
 *          I:  input_dim
 *          H:  hidden_size
 *          x:  [T, N, I] input features
 *          hx: [D*N, H] initial hidden state
 *          wx: [D,I,3H]
 *          wh: [D,H,3H]
 *          bx: [D,3H]
 *          bh: [D,3H]
 *          y:  [T, N, D*H] output features for the every step
 *          hy: [D*N, H] output feature for the last time step
 *          gateR: reserved matrix of gate R
 *          gateZ: reserved matrix of gate Z
 *          gateN: reserved matrix of gate N
 *          Mnh: reserved matrix of (ht-1 * Wnh + bnh)
 *          ws: workspace, temporary buffer
 *
 * @formula_list:  rt = sigmoid(xt * Wrx + brx + ht-1 * Wrh + brh)
 *                 zt = sigmoid(xt * Wzx + bzx + ht-1 * Wzh + bzh)
 *                 nt = tanh(xt * Wnx + bnx + rt * (ht-1 * Wnh + bnh))
 *                 ht = (1 - zt) * nt + zt * ht-1
 */

void gru_forward_single_sequential(int T, int D, int N, int I, int H, 
    float* x,    //[T,N,H*D]
    float* hx,   //[D*N,H] 
    float* wx,   //2*(I*3H + H*3H + 6H)
    float* wh,   //2*(I*3H + H*3H + 6H)
    float* bx,   //2*(I*3H + H*3H + 6H)
    float* bh,   //2*(I*3H + H*3H + 6H)
    float* y,    //[T,N,H*D]
    float* hy,   //[D*N,H]
    float* gateR,//(L*)D*T*N*H
    float* gateZ,//(L*)D*T*N*H
    float* gateN,//(L*)D*T*N*H
    float* Mnh,  //(L*)*D*T*N*H
    float* ws
){
    // create thread for ud
    pthread_t thread1, thread2;
    int err1 = 0;
    int err2 = 0;
    FwdThreadPara* para1 = NULL;
    FwdThreadPara* para2 = NULL;
    para1 = (FwdThreadPara*)malloc(sizeof(FwdThreadPara));
    para1->T = T;
    para1->D = D;
    para1->N = N;
    para1->I = I;
    para1->H = H;
    para1->x = x;
    para1->hx = hx;
    para1->wx = wx;
    para1->wh = wh;
    para1->bx = bx;
    para1->bh = bh;
    para1->y = y;
    para1->hy = hy;
    para1->gateR = gateR;
    para1->gateZ = gateZ;
    para1->gateN = gateN;
    para1->Mnh = Mnh;
    para1->ws = ws;
    para1->d = 0;

    //get threadid from ws
    //thread1 = (pthread_t) ws;

    // if threadid is not avaliable, create the thread

    err1 = pthread_create(&thread1, NULL, fwd_thread_one_direction, para1);
    // create thread for ud
    if(D ==2){
        para2 = (FwdThreadPara*)malloc(sizeof(FwdThreadPara));
        memcpy(para2, para1, sizeof(FwdThreadPara));
        para2->d = 1;
        err2 = pthread_create(&thread2, NULL, fwd_thread_one_direction, para2);
    }
    // wait for the threat job done, children thread alwasy running
    if(err1 | err2){
        printf("error in creating thread \n");
    }
    err1 = pthread_join(thread1, NULL);                                         
    if(D == 2){
        err2 = pthread_join(thread2, NULL);                                         
    }
    if (err1 | err2){                                                           
        printf("error in joining thread \n");                                   
    }
    free(para1);
    if(D == 2){
        free(para2);
    }

}

/*
 * @brief:  gru_xw_seq_forward
 *          multi-layer forward computation 
 *
 * @params: L:  num_layers
 *          T:  seq_length
 *          D:  direction, D = 2 if bidirectional else 1
 *          N:  batch_size
 *          I:  input_dim
 *          H:  hidden_size
 *          x:  [T,N,I] input features
 *          hx: [L,D*N,H] initial hidden state
 *          wx: [L,D,I,3H]
 *          wh: [L,D,H,3H]
 *          bx: [L,D,3H]
 *          bh: [L,D,3H]
 *          y:  [T,N,D*H] output features for the every step
 *          hy: [L,D*N,H] output feature for the last time step
 *          ws: workspace, reserved buffer & temporary buffer
 *
 * @desc: sequantial implementation of GRU forward
 *        call function 'gru_forward_single_sequential' to compute each layer
 */

void gru_xw_seq_forward(int L, int T, int D, int N, int I, int H, 
    float* x, float* hx, float* wx, float* wh, float* bx, float* bh, float* y, 
    float* hy, float* ws //[4*L*T*D*N*H + 2*N*3H*D]
){
    float* gateR_l = ws + WS_THREAD;
    float* gateZ_l = gateR_l + L * T * D * N * H;
    float* gateN_l = gateZ_l + L * T * D * N * H;
    float* y_l = gateN_l + L * T * D * N * H;
    float* Mnh_l = y_l + L * T * N * H * D;
    float* ws2 = Mnh_l + L * D * T * N * H;
    float* wx_l = wx;
    float* wh_l = wh;
    float* bx_l = bx;
    float* bh_l = bh;
    float* x_l = x;
    float* hx_l = hx;
    float* hy_l = hy;
    
    for (int l = 0; l < L; l++) {  //  for each Layer
        if(l != 0){
            x_l = y_l;
            I = 2 * H;
        }
        
        gru_forward_single_sequential(T, D, N, I, H, x_l, hx_l, wx_l, wh_l,
          bx_l, bh_l, y_l, hy_l, gateR_l, gateZ_l, gateN_l, Mnh_l, ws2);
        gateR_l = gateR_l + T * D * N * H;
        gateZ_l = gateZ_l + T * D * N * H;
        gateN_l = gateN_l + T * D * N * H;
        Mnh_l = Mnh_l +  T * D * N * H;
        y_l = y_l + T * N * H * D;
        hx_l = hx_l + D * N * H;
        hy_l = hy_l + D * N * H;        
        wh_l = wh_l + H * 3 * H * D;
        bx_l = bx_l + 3 * H * D;
        bh_l = bh_l + 3 * H * D;

        if (l == 0) {
            wx_l = wx_l + I * H * 3 * D;
        } else {
            wx_l = wx_l + D * H * H * 3 * D;
        }
    }
    y_l = y_l - T * N * H * D;
    #pragma omp parallel for 
    for (int i = 0; i < T * N * H * D; i++){
        y[i] = y_l[i];
    }
}

/*
 * @brief:  gru_xw_single_bwd
 *          single layer backward computation 
 *
 * @params: T:  seq_length
 *          D:  direction, D = 2 if bidirectional else 1
 *          N:  batch_size
 *          I:  input_dim
 *          H:  hidden_size
 *          x:  [T, N, I] input features
 *          hx: [D*N, H] initial hidden state
 *          wx: [D,I,3H]
 *          wh: [D,H,3H]
 *          dwx: [D,I,3H] output gradient
 *          dwh: [D,H,3H] output gradient
 *          dx: [T,N,I] output gradient
 *          dhx: [D,N,H] output gradient
 *          dbx: [D,3H] output gradient
 *          dbh: [D,3H] output gradient
 *          dy:  [T, N, D*H] input gradient
 *          dhy: [D*N, H] input gradient
 *          gateR: [D,T,N,H] reserved matrix of gate R
 *          gateZ: [D,T,N,H] reserved matrix of gate Z
 *          gateN: [D,T,N,H] reserved matrix of gate N
 *          Mnh: [D,T,N,H] reserved matrix of (ht-1 * Wnh + bnh)
 *          ws: workspace, temporary buffer
 *
 * @formulas: dan = dht * (1 - zt) * (1 - nt^2)
 *            daz = dht * (ht-1 - nt) * zt * (1 - zt)
 *            dar = dan * (ht-1 @ Wnh + bnh) * rt * (1-rt)
 *            dxt = dan @ Wnx.T + daz @ Wzx.T + dar @ Wrx.T
 *            dht-1 = dht * zt + (dan*rt) @ Wnh.T + daz @ Wzh.T + dar @ Wrh.T
 *            dWnx = xt.T @ dan
 *            dWzx = xt.T @ daz
 *            dWrx = xt.T @ dar
 *            dWnh = ht-1.T @ (dan*rt)
 *            dWzh = ht-1.T @ daz
 *            dWrh = ht-1.T @ dar
 *            dbnx = sum(dan,axis=0)
 *            dbnh = sum(dan*rt,axis=0)
 *            dbzx = dbzh = sum(daz,axis=0)
 *            dbrx = dbrh = sum(dar,axis=0)
 */

typedef struct bwd_thread_para{
    int T;
    int D;
    int N;
    int I;
    int H;
    float* x;
    float* hx;
    float* wx;
    float* wh;
    float* y;
    float* dwx;
    float* dwh;
    float* dx;
    float* dhx;
    float* dbx;
    float* dbh;
    float* dy;
    float* dhy;
    float* gateR;
    float* gateZ;
    float* gateN;
    float* Mnh;
    float* ws;
    int d;
}BwdThreadPara;

void* bwd_thread_one_direction(void * p){
    BwdThreadPara* para = (BwdThreadPara*)p;
    int T = para->T;
    int D = para->D;
    int N = para->N;
    int I = para->I;
    int H = para->H;
    int d = para->d;
    float* x = para->x;
    float* hx = para->hx;
    float* y  = para->y;
    float* wx = para->wx;
    float* wh = para->wh;
    float* dwx = para->dwx;
    float* dwh = para->dwh;
    float* dx = para->dx;
    float* dhx = para->dhx;
    float* dbx = para->dbx;
    float* dbh = para->dbh;
    float* dy = para->dy;
    float* dhy = para->dhy;
    float* gateR = para->gateR;
    float* gateZ = para->gateZ;
    float* gateN = para->gateN;
    float* Mnh = para->Mnh;
    float* ws = para->ws;

    int i,j,t;
    double start;
    double t_ew = 0;
    double t_cp = 0;
    double t_sg = 0;
    double t_bg = 0;

    float* dyt;
    float* ht1;//[N,D,H]
    float* back_ht1;//not necessary
    float* dxt;
    float* xt;
    float* rt;
    float* zt;
    float* nt;
    float* dat;
    float* dart;
    float* dar = ws; //[D,T,N,3H]
    float* da = dar + D * T * N * 3 * H; //[D,T,N,3H]
    float* dht1 = da + D * T * N * 3 * H;  //[D,N,H]
    float* hx_ = dht1 + D * N * H; //[N,D,H] 
    float* back_dht1 = dht1 + N*H; //[N,H]
    float* Mnht = Mnh;
    float* back_Mnht = Mnh + T*N*H;
    float* back_gateR = gateR + T * N * H;
    float* back_gateZ = gateZ + T * N * H;
    float* back_gateN = gateN + T * N * H;
    float* back_wx = wx + I*3*H;
    float* back_wh = wh + H*3*H;
    float* back_dwx = dwx + I*3*H;
    float* back_dwh = dwh + H*3*H;
    float* back_dbx = dbx + 3*H;
    float* back_dbh = dbh + 3*H;
    float* back_dar = dar + T * N * 3 * H;
    float* back_da = da + T * N * 3 * H;
    float* back_dat;
    float* back_dart;
    if(D == 1){
        omp_set_num_threads(40);
    }else{
        mkl_set_num_threads_local(20);
        omp_set_num_threads(20);
    }

    if(d == 0){
        if(D ==2){
            cpu_set_t cpuset;
            pthread_t thread = pthread_self();
            CPU_ZERO(&cpuset);
            for (int i = 20; i < 40; i++){
                CPU_SET(i, &cpuset);
            }
            pthread_setaffinity_np(thread, sizeof(cpu_set_t), &cpuset);
        }
        #pragma omp parallel for
        for(i = 0; i <  H * 3 * H; ++i){
            dwh[i]=0;
        }
        if(dbx and dbh){
            #pragma omp parallel for
            for(i = 0; i < 3 * H; ++i){
                dbx[i]=0;
                dbh[i]=0;
            }
        }
        #pragma omp parallel for collapse(2)
        for(i=0; i < N; ++i){
            for(j=0; j < H; ++j){
                hx_[i*D*H+j] = hx[i*H+j];
                dht1[i * H + j] = dhy[i * H + j];
            }
        }
    }else if(d == 1){
        if(D ==2){
            cpu_set_t cpuset;
            pthread_t thread = pthread_self();
            CPU_ZERO(&cpuset);
            for (int i = 0; i < 20; i++){
                CPU_SET(i, &cpuset);
            }
            pthread_setaffinity_np(thread, sizeof(cpu_set_t), &cpuset);
        }
        #pragma omp parallel for
        for(i = H * 3 * H; i < D * H * 3 * H; ++i){
            dwh[i]=0;
        }
        if(dbx and dbh){
            #pragma omp parallel for
            for(i = 3 * H; i < D * 3 * H; ++i){
                dbx[i]=0;
                dbh[i]=0;
            }
        }
        #pragma omp parallel for collapse(2)
        for(i=0; i < N; ++i){
            for(j=0; j < H; ++j){
                hx_[i*D*H + H + j] = hx[N*H + i*H + j];
                back_dht1[i * H + j] = dhy[N * H + i * H + j];
            }
        }
    }
    if(d == 0){
        for(t = T - 1; t >= 0; --t){
            if(t){
                ht1 = y + (t-1) * N * D * H;
            }
            else{
                ht1 = hx_;
            }
            //add dy[T,N,D,H] to dhy[D,N,H]
            dyt = dy + t * N * D * H;
            #pragma omp parallel for collapse(2)
            for(i=0; i < N; ++i){
                for(j=0; j < H; ++j){
                    dht1[i*H+j] += dyt[i*D*H+j];
                }
            }

            rt = gateR + t * N * H;
            zt = gateZ + t * N * H;
            nt = gateN + t * N * H;
            Mnht = Mnh +  t * N * H;
            dat = da + t * N * 3 * H;
            dart = dar + t * N * 3 * H;
            #pragma omp parallel for collapse(2)
            for( i=0; i < N; ++i){
                for( j=0; j < H; ++j){
                    int nid = i * 3 * H + 2 * H + j;
                    int zid = i * 3 * H + H + j;
                    int rid = i * 3 * H + j;
                    int id = i * H + j;
                    dat[nid] = dht1[id] * (1 - zt[id]) * (1 - nt[id] * nt[id]);
                    dart[zid] = dat[zid] = dht1[id] * (ht1[i*D*H + j] - nt[id]) * 
                      zt[id] * (1 - zt[id]);
                    dart[rid] = dat[rid] = dat[nid] * Mnht[id] * rt[id] * 
                      (1 - rt[id]);
                    dart[nid] = dat[nid] * rt[id];
                    dht1[id] = dht1[id] * zt[id];
                }
            }

            // dht1 = da * wh.T    [N,H] = [N,3H] * [3H,H]
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, N, H, 3 * H, 1.0, 
              dart, 3 * H, wh, 3 * H, 1.0, dht1, H);

            // dwh = ht1.T * da    [H,3H] = [H,N] * [N,3H]
            cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, H, 3 * H, 
              N, 1, ht1, D*H, dart, 3 * H, 1, dwh, 3 * H);
        }
        // dbx = e * da       [1,3H] = [1,N] * [N,3H]
        if(dbx and dbh){
            #pragma omp parallel for
            for(i = 0; i < 3 * H; ++i){
                for(j = 0; j < N * T; ++j){
                    dbx[i] += da[j * 3 * H + i];
                    dbh[i] += dar[j * 3 * H + i];
                }
            }
        }
        //}
        // dx = da * wx.T    [T*N,I] = [T*N,3H] * [3H,I]
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, T*N, I, 3 * H, 1, 
          da, 3 * H, wx, 3 * H, 0, dx, I);
        // dwx = x.T * da    [I,3H] = [I,T*N] * [T*N,3H]
        cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, I, 3 * H, T*N, 1, x, 
          I, da, 3 * H, 0, dwx, 3 * H);
        #pragma omp parallel for
        for(i = 0; i <  N * H; ++i){
            dhx[i] = dht1[i];
        }

    }
    if(d == 1){
        for(t = 0; t < T; ++t){
            if(t == T-1){
                back_ht1 = hx_;
            }
            else{
                back_ht1 = y + (t+1) * N * D * H;
            }
            //add dy[T,N,D,H] to dhy[D,N,H]
            dyt = dy + t * N * D * H;
            #pragma omp parallel for collapse(2)
            for(i=0; i < N; ++i){
                for(j=0; j < H; ++j){
                    back_dht1[i*H+j] += dyt[i*D*H + H + j];
                }
            }
            rt = back_gateR + t * N * H;
            zt = back_gateZ + t * N * H;
            nt = back_gateN + t * N * H;
            back_Mnht = Mnh + (T+t)*N*H;
            dat = back_da + t * N * 3 * H;
            dart = back_dar + t * N * 3 * H;
            #pragma omp parallel for collapse(2)
            for( i=0; i < N; ++i){
                for( j=0; j < H; ++j){
                    int nid = i * 3 * H + 2 * H + j;
                    int zid = i * 3 * H + H + j;
                    int rid = i * 3 * H + j;
                    int id = i * H + j;
                    dat[nid] = back_dht1[id] * (1 - zt[id]) * (1 - nt[id] * nt[id]);
                    dart[zid] = dat[zid] = back_dht1[id] * (back_ht1[i*D*H + H + j] - 
                      nt[id]) * zt[id] * (1 - zt[id]);
                    dart[rid] = dat[rid] = dat[nid] * back_Mnht[id] * rt[id] * 
                      (1 - rt[id]);
                    dart[nid] = dat[nid] * rt[id];
                    back_dht1[id] = back_dht1[id] * zt[id];
                }
            }
            // dht1 = da * wh.T    [N,H] = [N,3H] * [3H,H]
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, N, H, 3 * H, 1, 
              dart, 3 * H, back_wh, 3 * H, 1, back_dht1, H);
            // dwh = ht1.T * da    [H,3H] = [H,N] * [N,3H]
            cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, H, 3 * H, 
              N, 1, back_ht1+H, D*H, dart, 3 * H, 1, back_dwh, 3 * H);

        }
        // dbx = e * da       [1,3H] = [1,N] * [N,3H]
        if(dbx and dbh){
            #pragma omp parallel for
            for(i = 0; i < 3 * H; ++i){
                for(j = 0; j < N * T; ++j){
                    back_dbx[i] += back_da[j * 3 * H + i];
                    back_dbh[i] += back_dar[j * 3 * H + i];
                }
            }
        }

        // dwx = xt.T * da    [I,3H] = [I,N] * [N,3H]
        cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, I, 3 * H, T*N, 1, x, 
          I, back_da, 3 * H, 0, back_dwx, 3 * H);

        // dxt = da * wx.T    [T*N,I] = [T*N,3H] * [3H,I]
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, T*N, I, 3 * H, 1, 
            back_da, 3 * H, back_wx, 3 * H, 1, dx, I);
        #pragma omp parallel for
        for(i = N * H; i < D * N * H; ++i){
            dhx[i] = dht1[i];
        }
    }

}

int tid = 0;
void gru_xw_single_bwd(int T, int D, int N, int I, int H,
            float* x,  //[T,N,I]
            float* hx, //[D*N,H]
            float* y,  //[T,N,D,H]
            float* wx, //[I,3H]
            float* wh, //[H,3H]
            float* dwx, //[I,3H]
            float* dwh, //[H,3H]
            float* dx,  //[T,N,I]
            float* dhx, //[D,N,H]
            float* dbx, //[1,3H]
            float* dbh,
            float* dy, //[T,N,D,H] 
            float* dhy, //[D,N,H] 
            float* gateR, //[T,D,N,H]
            float* gateZ, //[T,D,N,H]
            float* gateN, //[T,D,N,H]
            float* Mnh, //[D,T,N,H]
            float* ws //D * (N*3H + N*H + N*H)
            )
{

    pthread_t thread1, thread2;
    int err1 = 0;
    int err2 = 0;
    BwdThreadPara* para1 = NULL;
    BwdThreadPara* para2 = NULL;
    para1 = (BwdThreadPara*)malloc(sizeof(BwdThreadPara));
    para1->T = T;
    para1->D = D;
    para1->N = N;
    para1->I = I;
    para1->H = H;
    para1->x = x;
    para1->hx = hx;
    para1->wx = wx;
    para1->wh = wh;
    para1->y = y;
    para1->dwx = dwx;
    para1->dwh = dwh;
    para1->dx = dx;
    para1->dhx = dhx;
    para1->dbx = dbx;
    para1->dbh = dbh;
    para1->dy = dy;
    para1->dhy = dhy;
    para1->gateR = gateR;
    para1->gateZ = gateZ;
    para1->gateN = gateN;
    para1->Mnh = Mnh;
    para1->ws = ws;
    para1->d = 0;

    //get threadid from ws
    //thread1 = (pthread_t) ws;

    // if threadid is not avaliable, create the thread

    err1 = pthread_create(&thread1, NULL, bwd_thread_one_direction, para1);
    // create thread for ud
    if(D ==2){
        para2 = (BwdThreadPara*)malloc(sizeof(BwdThreadPara));
        memcpy(para2, para1, sizeof(BwdThreadPara));
        para2->d = 1;
        err2 = pthread_create(&thread2, NULL, bwd_thread_one_direction, para2);
    }
    // wait for the threat job done, children thread alwasy running
    if(err1 | err2){
        printf("error in creating thread \n");
    }
    err1 = pthread_join(thread1, NULL);                                         
    if(D == 2){
        err2 = pthread_join(thread2, NULL);                                         
    }
    if (err1 | err2){                                                           
        printf("error in joining thread \n");                                   
    }
    free(para1);
    if(D == 2){
        free(para2);
    }
}
/*
 * @brief:  gru_xw_seq_backward
 *          multi-layer backward computation 
 *
 * @params: L:  num_layers
 *          T:  seq_length
 *          D:  direction, D = 2 if bidirectional else 1
 *          N:  batch_size
 *          I:  input_dim
 *          H:  hidden_size
 *          dy:  [T,N,D*H] input gradient
 *          dhy: [L,D*N,H] input gradient
 *          x:  [T,N,I] input features
 *          hx: [L,D*N,H] initial hidden state
 *          wx: [L,D,I,3H]
 *          wh: [L,D,H,3H]
 *          dx: [T,N,I] output gradient
 *          dhx: [L,D*N,H] output gradient
 *          dwx: [L,D,I,3H] output gradient
 *          dwh: [L,D,H,3H] output gradient
 *          dbx: [L,D,3H] output gradient
 *          dbh: [L,D,3H] output gradient
 *          ws: workspace, reserved buffer & temporary buffer
 *
 * @desc: sequantial implementation of GRU backward
 *        call function 'gru_xw_single_bwd' to compute each layer
 */

void gru_xw_seq_bwd(int L, int T, int D, int N, int I, int H,
    float* dy,  //[T,N,D*H]
    float* dhy, //[L,D*N,H]
    float* x,   //[T,N,I]
    float* hx,  //[L,D*N,H]
    float* wx,  //[L,D,I,3H]
    float* wh,  //[L,D,H,3H]
    float* dx,  //[T,N,I]
    float* dhx, //[L,D*N,H]
    float* dwx, //[L,D,I,3H]
    float* dwh, //[L,D,H,3H]
    float* dbx, //[L,D,3H]
    float* dbh, //[L,D,3H]
    float* ws
){
    float* gateR_l = ws + WS_THREAD + (L-1) * T * D * N * H;
    float* gateZ_l = gateR_l + L * T * D * N * H;
    float* gateN_l = gateZ_l + L * T * D * N * H;
    float* y_l = gateN_l + L * T * D * N * H;
    float* Mnh_l = y_l + L * T * N * H * D;
    float* ws2 = Mnh_l + T * N * H * D;
    float* wx_l = (L == 1)? wx : wx + (L-2)*D*(D*H)*3*H + D*I*3*H;
    float* wh_l = wh + (L-1) * D * H * 3 * H;
    float* x_l = x;
    float* hx_l = hx + (L-1) * D * N * H;
    float* dhy_l = dhy + (L-1) * D * N * H;
    float* dwx_l = (L == 1)? dwx : dwx + (L-2)*D*(D*H)*3*H + D*I*3*H;
    float* dwh_l = dwh + (L-1) * D * H * 3 * H;
    float* dbx_l = dbx + (L-1) * D * 3 * H;
    float* dbh_l = dbh + (L-1) * D * 3 * H;
    float* dx_l = (L == 1)? dx : dx;//TODO
    float* dhx_l = dhx + (L-1) * D * N * H;
    float* dy_l = dy;

    gru_xw_single_bwd(T, D, N, I, H, x_l, hx_l, y_l, wx_l, wh_l, dwx_l, dwh_l, 
      dx_l, dhx_l, dbx_l, dbh_l,dy_l, dhy_l, gateR_l, gateZ_l, gateN_l, Mnh_l,
      ws2);

}

 /*
 * @brief:  gru_xw_train_get_workspace_size
 *          get the size of buffer space for GRU training
 *
 * @params: I:  input_dim
 *          H:  hidden_size
 *          T:  seq_length
 *          N:  batch_size
 *          bi: whether bi-directional or not (1 or 0)
 *          L:  num_layers
 *
 * @workspace: RESERVED
 *             gateR [L,D,T,N,H]
 *             gateZ [L,D,T,N,H]
 *             gateN [L,D,T,N,H]
 *             y     [L,D,T,N,H]
 *             Mnh   [L,D,T,N,H] matrix for (ht-1 * Wnh + bnh)
 *             thread [WS_THREAD]
 *
 *             TEMP for FORWARD
 *             gemmC1 [D,T,N,3H]
 *             gemmC2 [D,N,3H]
 *
 *             TEMP for BACKWARD
 *             dar  [D,T,N,3H]
 *             da   [D,T,N,3H]
 *             dht1 [D,N,H]
 *             hx_  [N,D,H]
 *
 */ 
int gru_xw_train_get_workspace_size(int L, int D, int T, int N, int I, int H)
{
    return WS_THREAD + 5 * L * T * D * N * H + 7 * N * H + 2 * D * T * N * 3 * H;
}

/*
 * @brief:  gru_xw_forward
 *          interface function of GRU forward
 *
 * @params: L:  num_layers
 *          T:  seq_length
 *          D:  direction, D = 2 if bidirectional else 1
 *          N:  batch_size
 *          I:  input_dim
 *          H:  hidden_size
 *          x:  [T,N,I] input features
 *          hx: [L,D*N,H] initial hidden state
 *          wx: [L,D,I,3H]
 *          wh: [L,D,H,3H]
 *          bx: [L,D,3H]
 *          bh: [L,D,3H]
 *          y:  [T,N,D*H] output features for the every step
 *          hy: [L,D*N,H] output feature for the last time step
 *          ws: workspace, reserved buffer & temporary buffer
 *          mode: specify the mode of implementation
 *
 * @desc: call different implementations of forward functions.
 *        (Currently, there is only sequential version.)
 *
 */

int gru_xw_forward(RNNForwardDesc desc){

    gru_xw_seq_forward(desc.L, desc.T, desc.D, desc.N, desc.I, desc.H, desc.x,
        desc.hx, desc.wx, desc.wh, desc.bx, desc.bh, desc.y, desc.hy, desc.ws);
}


int gru_xw_backward(RNNBackwardDesc desc){

    gru_xw_seq_bwd(desc.L, desc.T, desc.D, desc.N, desc.I, desc.H, desc.dy, 
        desc.dhy, desc.x, desc.hx, desc.wx, desc.wh, desc.dx, desc.dhx,
        desc.dwx, desc.dwh, desc.dbx, desc.dbh, desc.ws);
    
}


