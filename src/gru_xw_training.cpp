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
    float* ws,   //
    double* time
){
    double t_ew = 0;
    double t_cp = 0;
    double t_sg = 0;
    double t_bg = 0;
    double start, end;
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
    float* gemmC1_t = gemmC1;
    start = dsecnd();
    if (D == UNIDIRECT) {
        #pragma omp parallel for collapse(2)
        for (int i = 0; i < N; i++)
            for (int j = 0; j < H; j++) {
                y[i * H + j] = hx[i * H + j];
            }
    } else {
        #pragma omp parallel for collapse(2)
        for (int i = 0; i < N; i++)
            for (int j = 0; j < H; j++) {
                y[i * D * H + j] = hx[i * H + j];
                back_ht_1[i *D * H + j] = hx[N * H + i * H + j];
            }
    }
    t_cp += dsecnd() - start;
    //x*wx : [T*N,I] * [I,3H]
    start = dsecnd();
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N*T, n, k, 1, x, k, 
      wx, n, 0.0, gemmC1, n);
    if(D == BIDIRECT){
        cblas_sgemm(CblasRowMajor,CblasNoTrans, CblasNoTrans, N*T, n, k, 1, x, 
          k, back_wx, n, 0.0, back_gemmC1, n);
    }
    t_bg += dsecnd() - start;
    for (int t =0; t < T; t++) {
        //  perform the first direction, X*wx and H * wh for each step
        //  ht-1*wh, ht-1:[N,H] wh:[H, 3H]
        start = dsecnd();
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, H, 1, 
                    ht_1, D*H, wh, n, 0.0, gemmC2, n);
        t_sg += dsecnd() - start;
        
        start = dsecnd();
        rt = gateR + t * N * H;
        zt = gateZ + t * N * H;
        nt = gateN + t * N * H;
        gemmC1_t = gemmC1 + t * N * 3 * H;
        float* Mnht = Mnh + t * N * H;
        if (bx and bh){
        #pragma omp parallel for collapse(2)
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < H; ++j) {
                //printf("N(%d)=%d, H(%d)=%d\n", N, i, H,j);
                int rtb = i * 3 * H;
                int ztb = i * 3 * H + H;
                int ntb = i * 3 * H + 2 * H;
                Mnht[i * H + j] = gemmC2[ntb + j] + bh[2*H+j];
                rt[i * H + j] = 1/(1 + exp(-gemmC1_t[rtb + j] - gemmC2[rtb + j]
                    - bx[j] - bh[j]));
                zt[i * H + j] = 1/(1 + exp(-gemmC1_t[ztb + j]-gemmC2[ztb + j]
                    - bx[H + j] - bh[H + j]));
                nt[i * H + j] = tanh(gemmC1_t[ntb + j] + bx[2 * H + j] +
                    rt[i * H + j] * (gemmC2[ntb + j] + bh[2 * H + j]));
                ht[i * D * H + j] = (1-zt[i * H + j]) * nt[i * H + j] +
                    zt[i * H + j] * ht_1[i * D * H + j];
            }
        }}
        else{
        #pragma omp parallel for collapse(2)
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < H; ++j) {
                int rtb = i * 3 * H;
                int ztb = i * 3 * H + H;
                int ntb = i * 3 * H + 2 * H;
                Mnht[i * H + j] = gemmC2[ntb + j];
                rt[i * H + j] = 1 /
                  (1 + exp(-gemmC1_t[rtb + j] - gemmC2[rtb + j]));
                zt[i * H + j] = 1/
                  (1 + exp(-gemmC1_t[ztb + j]-gemmC2[ztb + j]));
                nt[i * H + j] = tanh(gemmC1_t[ntb + j] + 
                  rt[i * H + j] * gemmC2[ntb + j]);
                ht[i * D * H + j] = (1-zt[i * H + j]) * nt[i * H + j] +
                  zt[i * H + j] * ht_1[i * D * H + j];
            } 
        }}
        t_ew += dsecnd() - start;


        ht_1 = ht;
        ht = ht + D * H * N;
        //  perform the second direction
        if (D == BIDIRECT) {
            rt = back_gateR + (T - 1 - t) * N * H;
            zt = back_gateZ + (T - 1 - t) * N * H;
            nt = back_gateN + (T - 1 - t) * N * H;
            gemmC1_t = back_gemmC1 + (T - 1 - t) * N * 3 * H;
            start = dsecnd();
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, H, 1, 
                        back_ht_1, D * H, back_wh, n, 0.0, gemmC2, n);
            t_sg += dsecnd() - start;

            start = dsecnd();
            float* back_Mnht = back_Mnh + (T-1-t)*N*H;
            if(back_bx and back_bh){
            #pragma omp parallel for collapse(2)
            for (int i = 0; i < N; ++i) {
                for (int j = 0; j < H; ++j) {
                    int rtb = i * 3 * H;
                    int ztb = i * 3 * H + H;
                    int ntb = i * 3 * H + 2 * H;
                    back_Mnht[i * H + j] = gemmC2[ntb + j] + back_bh[2*H+j];
                    rt[i * H + j] = 1 / (1 + exp(-gemmC1_t[rtb + j] -
                        gemmC2[rtb + j] - back_bx[j] - back_bh[j]));
                    zt[i * H + j] = 1 / (1 + exp(-gemmC1_t[ztb + j] -
                        gemmC2[ztb + j] - back_bx[H + j]- back_bh[H + j]));
                    nt[i * H + j] = tanh(gemmC1_t[ntb + j] + back_bx[ 2 * H + j]
                        + rt[i * H + j] * (gemmC2[ntb + j] + back_bh[2*H+j]));
                    back_ht[i * D * H + j] = (1 - zt[i * H + j]) * nt[i * H + j]
                        + zt[i * H + j] * back_ht_1[i * D * H + j];
                }
            }}
            else{
            #pragma omp parallel for collapse(2)
            for (int i = 0; i < N; ++i) {
                for (int j = 0; j < H; ++j) {
                    int rtb = i * 3 * H;
                    int ztb = i * 3 * H + H;
                    int ntb = i * 3 * H + 2 * H;
                    back_Mnht[i * H + j] = gemmC2[ntb + j];
                    rt[i * H + j] = 1 / (1 + exp(-gemmC1_t[rtb + j] -
                        gemmC2[rtb + j]));
                    zt[i * H + j] = 1 / (1 + exp(-gemmC1_t[ztb + j] -
                        gemmC2[ztb + j]));
                    nt[i * H + j] = tanh(gemmC1_t[ntb + j] + rt[i * H + j] * 
                        gemmC2[ntb + j]);
                    back_ht[i * D * H + j] = (1 - zt[i * H + j]) * nt[i * H + j]
                        + zt[i * H + j] * back_ht_1[i * D * H + j];
                }
            }}
            t_ew += dsecnd() - start;
            back_ht_1 = back_ht;
            back_ht = back_ht - D * H * N;
        }
    }
    start = dsecnd();
    //  copy last state to hy, from(N,H*D) to (D,N,H)
    if (hy != 0) {
        if (D == UNIDIRECT) {
            float* y_start = y + (T - 1) * N * H;
            #pragma omp parallel for collapse(2)
            for (int i = 0; i < N; i++)
                for (int j = 0; j < H; j++) {
                    hy[i * H + j] = y_start[i * H + j];
                }
        } else {
            float* y_start = y + (T - 1) * N * H * D;
            float* y_back_start = y + H;
            #pragma omp parallel for collapse(2)
            for (int i = 0; i < N; i++)
                for (int j = 0; j < H; j++) {
                    hy[i * H + j] = y_start[i * D * H + j];
                    hy[N * H + i * H + j] = y_back_start[i * D * H + j];
                }
        }
    }
    t_cp += dsecnd() - start;
    time[0] += t_cp;
    time[1] += t_ew;
    time[2] += t_sg;
    time[3] += t_bg;
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
    float* hy, float* ws, double* time //[4*L*T*D*N*H + 2*N*3H*D]
){
    float* gateR_l = ws;
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
          bx_l, bh_l, y_l, hy_l, gateR_l, gateZ_l, gateN_l, Mnh_l, ws2, time);
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
    double start = dsecnd();
    y_l = y_l - T * N * H * D;
    #pragma omp parallel for 
    for (int i = 0; i < T * N * H * D; i++){
        y[i] = y_l[i];
    }
    time[0] += dsecnd() - start;
   
    //memcpy(y, y_l - T * N * H * D, T * N * H * D * sizeof(float));
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

void gru_xw_single_bwd(int T, int D, int N, int I, int H,
            float* x,  //[T,N,I]
            float* hx, //[D*N,H]
            float* y,  //[T,N,D,H]
            float* wx, //[I,3H]
            float* wh, //[H,3H]
   /*out*/  float* dwx, //[I,3H]
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
            float* ws, //D * (N*3H + N*H + N*H)
            double* time)
{  
    int i,j,t;
    double start;
    double t_ew = 0;
    double t_cp = 0;
    double t_sg = 0;
    double t_bg = 0;
    start = dsecnd(); 
    #pragma omp parallel for
    for(i = 0; i < D * H * 3 * H; ++i){
        dwh[i]=0;
    }
    if(dbx and dbh){
        #pragma omp parallel for
        for(i = 0; i < D * 3 * H; ++i){
            dbx[i]=0;
            dbh[i]=0;
        }
    }
    t_cp += dsecnd() - start;
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
    float* dar = ws; //[T,N,3H]
    float* da = dar + T * N * 3 * H; //[T,N,3H]
    float* dht1 = da + T * N * 3 * H;  //[D,N,H]
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
    start = dsecnd();
    #pragma omp parallel for
    for(i = 0; i < N * H; ++i){
        dht1[i] = dhy[i];
    }
    #pragma omp parallel for collapse(2)
    for(i=0; i < N; ++i){
        for(j=0; j < H; ++j){
            hx_[i*D*H+j] = hx[i*H+j];
        }
    }
    if(D==2){
        #pragma omp parallel for
        for(i = 0; i < N * H; ++i){
            back_dht1[i] = dhy[N*H + i];
        }
        #pragma omp parallel for collapse(2)
        for(i=0; i < N; ++i){
            for(j=0; j < H; ++j){
                hx_[i*D*H + H + j] = hx[N*H + i*H + j];
            }
        }
    }
    t_cp += dsecnd() - start; 
    for(t = T - 1; t >= 0; --t){
        if(t){
            ht1 = y + (t-1) * N * D * H;
        }
        else{
            ht1 = hx_;
        }
        start = dsecnd();
        //add dy[T,N,D,H] to dhy[D,N,H]
        dyt = dy + t * N * D * H;
        #pragma omp parallel for collapse(2)
        for(i=0; i < N; ++i){
            for(j=0; j < H; ++j){
                dht1[i*H+j] += dyt[i*D*H+j];
            }
        }
        t_ew += dsecnd() - start;

        rt = gateR + t * N * H;
        zt = gateZ + t * N * H;
        nt = gateN + t * N * H;
        Mnht = Mnh +  t * N * H;
        dat = da + t * N * 3 * H;
        dart = dar + t * N * 3 * H;
        start = dsecnd();
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
        t_ew += dsecnd() - start;

        start = dsecnd(); 
        // dht1 = da * wh.T    [N,H] = [N,3H] * [3H,H]
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, N, H, 3 * H, 1.0, 
          dart, 3 * H, wh, 3 * H, 1.0, dht1, H);

        // dwh = ht1.T * da    [H,3H] = [H,N] * [N,3H]
        cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, H, 3 * H, 
          N, 1, ht1, D*H, dart, 3 * H, 1, dwh, 3 * H);
        t_sg += dsecnd() - start;
    }
    start = dsecnd();
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
    t_ew += dsecnd() - start;
    //}
    start = dsecnd();
    // dx = da * wx.T    [T*N,I] = [T*N,3H] * [3H,I]
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, T*N, I, 3 * H, 1, 
      da, 3 * H, wx, 3 * H, 0, dx, I);
    // dwx = x.T * da    [I,3H] = [I,T*N] * [T*N,3H]
    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, I, 3 * H, T*N, 1, x, 
      I, da, 3 * H, 0, dwx, 3 * H);
    t_bg += dsecnd() - start;

    if(D==2){
    for(t = 0; t < T; ++t){
        if(t == T-1){
            back_ht1 = hx_;
        }
        else{
            back_ht1 = y + (t+1) * N * D * H;
        }
        //add dy[T,N,D,H] to dhy[D,N,H]
        dyt = dy + t * N * D * H;
        start = dsecnd();
        #pragma omp parallel for collapse(2)
        for(i=0; i < N; ++i){
            for(j=0; j < H; ++j){
                back_dht1[i*H+j] += dyt[i*D*H + H + j];
            }
        }
        t_ew += dsecnd() - start;
        rt = back_gateR + t * N * H;
        zt = back_gateZ + t * N * H;
        nt = back_gateN + t * N * H;
        back_Mnht = Mnh + (T+t)*N*H;
        dat = da + t * N * 3 * H;
        dart = dar + t * N * 3 * H;
        start = dsecnd();
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
        t_ew += dsecnd() - start;
        start = dsecnd();
        // dht1 = da * wh.T    [N,H] = [N,3H] * [3H,H]
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, N, H, 3 * H, 1, 
          dart, 3 * H, back_wh, 3 * H, 1, back_dht1, H);
        // dwh = ht1.T * da    [H,3H] = [H,N] * [N,3H]
        cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, H, 3 * H, 
          N, 1, back_ht1+H, D*H, dart, 3 * H, 1, back_dwh, 3 * H);
        t_sg += dsecnd() - start;
    }
    start = dsecnd();
    // dbx = e * da       [1,3H] = [1,N] * [N,3H]
    if(dbx and dbh){
        #pragma omp parallel for
        for(i = 0; i < 3 * H; ++i){
            for(j = 0; j < N * T; ++j){
                back_dbx[i] += da[j * 3 * H + i];
                back_dbh[i] += dar[j * 3 * H + i];
            }
        }
    }
    t_ew += dsecnd() - start;
    //} 
    start = dsecnd();
    // dxt = da * wx.T    [T*N,I] = [T*N,3H] * [3H,I]
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, T*N, I, 3 * H, 1, 
      da, 3 * H, back_wx, 3 * H, 1, dx, I);

    // dwx = xt.T * da    [I,3H] = [I,N] * [N,3H]
    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, I, 3 * H, T*N, 1, x, 
      I, da, 3 * H, 0, back_dwx, 3 * H);
    }
    t_bg += dsecnd() - start;

    start = dsecnd();
    #pragma omp parallel for
    for(i = 0; i < D * N * H; ++i){
        dhx[i] = dht1[i];
    }
    //memcpy(dhx, dht1, N * H * D * sizeof(float));
    t_cp += dsecnd() - start;
    time[0] += t_cp;
    time[1] += t_ew;
    time[2] += t_sg;
    time[3] += t_bg;

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
    float* ws,
    double* time
){
    float* gateR_l = ws + (L-1) * T * D * N * H;
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
      ws2, time);

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
 *
 *             TEMP for FORWARD
 *             gemmC1 [D,T,N,3H]
 *             gemmC2 [N,3H]
 *
 *             TEMP for BACKWARD
 *             dar  [T,N,3H]
 *             da   [T,N,3H]
 *             dht1 [D,N,H]
 *             hx_  [N,D,H]
 *
 */ 
int gru_xw_train_get_workspace_size(int L, int D, int T, int N, int I, int H)
{
    return 5 * L * T * D * N * H + 7 * N * H + 2 * T * N * 3 * H;
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

    double time[4] = {0,0,0,0}; 
    gru_xw_seq_forward(desc.L, desc.T, desc.D, desc.N, desc.I, desc.H, desc.x,
        desc.hx, desc.wx, desc.wh, desc.bx, desc.bh, desc.y, desc.hy, desc.ws,
        time);
}


int gru_xw_backward(RNNBackwardDesc desc){

    double time[4] = {0,0,0,0};
    gru_xw_seq_bwd(desc.L, desc.T, desc.D, desc.N, desc.I, desc.H, desc.dy, 
        desc.dhy, desc.x, desc.hx, desc.wx, desc.wh, desc.dx, desc.dhx,
        desc.dwx, desc.dwh, desc.dbx, desc.dbh, desc.ws, time);
    
}

