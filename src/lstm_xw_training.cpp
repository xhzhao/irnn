#include <cstddef>                                                                                                                                                                                                
#include <mkl.h>
#include <omp.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include "rnn.h"

extern "C" {
inline float sigmoid(float x){
    return 1.0f / (1.0f + exp(-x));
}

void lstm_forward_single_sequential(int T, int D, int N, int I, int H, 
    float* ws,   //
    float* x,    //[T,N,H*D]
    float* hx,   //[D*N,H]
    float* cx,   //[D*N,H]
    float* wx,   //D*I*4H
    float* wh,   //D*H*4H
    float* bx, //D*4H
    float* y,    //[T,N,H*D]
    float* hy,   //[D*N,H]
    float* cy,   //[D*N,H]
    float* gateI,//(L*)D*T*N*H
    float* gateF,//(L*)D*T*N*H
    float* gateG,//(L*)D*T*N*H
    float* gateO,//(L*)D*T*N*H
    float* gateC//(L*)[T,N,D,H]
){
    int m = N;
    int n = 4*H;
    int k = I;
    float* ht = y;
    float* ht_1 = y;
    float* back_ht_1 = y + (T - 1) * N * H * D + H;
    float* back_ht = back_ht_1;
    float* ct = gateC;
    float* ct_1 = gateC;
    float* back_ct_1 = gateC + (T - 1) * N * H * D + H;
    float* back_ct = back_ct_1;


    float* gemmC1  = ws;              // [D,T,N,4H]
    float* gemmC2  = gemmC1 + D * T * N * 4 * H;  // N*4H
    float* it = gateI;
    float* ft = gateF;
    float* gt = gateG;
    float* ot = gateO;
    float* back_wx = wx + I * 4 * H;
    float* back_wh = wh + H * 4 * H;
    float* back_bx = (bx != NULL)? bx + 4 * H : NULL;
    float* back_gateI = gateI + T * N * H;
    float* back_gateF = gateF + T * N * H;
    float* back_gateG = gateG + T * N * H;
    float* back_gateO = gateO + T * N * H;
    float* back_gemmC1 = gemmC1 + T * N * 4 * H;
    float* gemmC1_t = gemmC1;
    if (D == UNIDIRECT) {
        #pragma omp parallel for collapse(2)
        for (int i = 0; i < N; i++)
            for (int j = 0; j < H; j++) {
                ht[i * H + j] = hx[i * H + j];
                ct[i * H + j] = cx[i * H + j];
            }
    } else {
        #pragma omp parallel for collapse(2)
        for (int i = 0; i < N; i++)
            for (int j = 0; j < H; j++) {
                ht[i * D * H + j] = hx[i * H + j]; // [D,N,H] --> [N,H,D]
                back_ht_1[i * D * H + j] = hx[N * H + i * H + j];
                ct[i * D * H + j] = cx[i * H + j];
                back_ct_1[i * D * H + j] = cx[N * H + i * H + j];
            }
    }
    //x*wx : [T*N,I] * [I,4H]
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N*T, n, k, 1, x, k, 
      wx, n, 0.0, gemmC1, n);
    if(D == BIDIRECT){
        cblas_sgemm(CblasRowMajor,CblasNoTrans, CblasNoTrans, N*T, n, k, 1, x, 
          k, back_wx, n, 0.0, back_gemmC1, n);
    }
    for (int t =0; t < T; t++) {
        //  perform the first direction, X*wx and H * wh for each step
        //  ht-1*wh, ht-1:[N,H] wh:[H, 4H]
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, H, 1, 
                    ht_1, D*H, wh, n, 0.0, gemmC2, n);

        it = gateI + t * N * H;
        ft = gateF + t * N * H;
        gt = gateG + t * N * H;
        ot = gateO + t * N * H;
        gemmC1_t = gemmC1 + t * N * 4 * H;
        if (bx){
            #pragma omp parallel for collapse(2)
            for (int i = 0; i < N; ++i) {
                for (int j = 0; j < H; ++j) {
                    int itb = i * 4 * H;
                    int ftb = i * 4 * H + H;
                    int gtb = i * 4 * H + 2 * H;
                    int otb = i * 4 * H + 3 * H;
                    it[i * H + j] = sigmoid(gemmC1_t[itb + j] + gemmC2[itb + j] + bx[j]);
                    ft[i * H + j] = sigmoid(gemmC1_t[ftb + j] + gemmC2[ftb + j] + bx[H + j]);
                    gt[i * H + j] = tanh(   gemmC1_t[gtb + j] + gemmC2[gtb + j] + bx[2 * H + j]);
                    ot[i * H + j] = sigmoid(gemmC1_t[otb + j] + gemmC2[otb + j] + bx[3 * H + j]);
                    ct[i * D * H + j] = ft[i * H + j] * ct_1[i * D * H + j] + it[i * H + j] * gt[i * H + j];
                    ht[i * D * H + j] = ot[i * H + j] * tanh(ct[i * D * H + j]);
                }
            }
        }
        else{
            #pragma omp parallel for collapse(2)
            for (int i = 0; i < N; ++i) {
                for (int j = 0; j < H; ++j) {
                    int itb = i * 4 * H;
                    int ftb = i * 4 * H + H;
                    int gtb = i * 4 * H + 2 * H;
                    int otb = i * 4 * H + 3 * H;
                    it[i * H + j] = sigmoid(gemmC1_t[itb + j] + gemmC2[itb + j]);
                    ft[i * H + j] = sigmoid(gemmC1_t[ftb + j] + gemmC2[ftb + j]);
                    gt[i * H + j] = tanh(   gemmC1_t[gtb + j] + gemmC2[gtb + j]);
                    ot[i * H + j] = sigmoid(gemmC1_t[otb + j] + gemmC2[otb + j]);
                    ct[i * D * H + j] = ft[i * H + j] * ct_1[i * D * H + j] + it[i * H + j] * gt[i * H + j];
                    ht[i * D * H + j] = ot[i * H + j] * tanh(ct[i * D * H + j]);
                }
            }
        }


        ht_1 = ht;
        ht = ht + D * H * N;
        ct_1 = ct;
        ct = ct + D * H * N;

        //  perform the second direction
        if (D == BIDIRECT) {
            it = back_gateI + (T - 1 - t) * N * H;
            ft = back_gateF + (T - 1 - t) * N * H;
            gt = back_gateG + (T - 1 - t) * N * H;
            ot = back_gateO + (T - 1 - t) * N * H;
            gemmC1_t = back_gemmC1 + (T - 1 - t) * N * 4 * H;
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, H, 1, 
                        back_ht_1, D * H, back_wh, n, 0.0, gemmC2, n);
            if(back_bx){
                #pragma omp parallel for collapse(2)
                for (int i = 0; i < N; ++i) {
                    for (int j = 0; j < H; ++j) {
                        int itb = i * 4 * H;
                        int ftb = i * 4 * H + H;
                        int gtb = i * 4 * H + 2 * H;
                        int otb = i * 4 * H + 3 * H;
                        it[i * H + j] = sigmoid(gemmC1_t[itb + j] + gemmC2[itb + j] + back_bx[j]);
                        ft[i * H + j] = sigmoid(gemmC1_t[ftb + j] + gemmC2[ftb + j] + back_bx[H + j]);
                        gt[i * H + j] = tanh(   gemmC1_t[gtb + j] + gemmC2[gtb + j] + back_bx[2 * H + j]);
                        ot[i * H + j] = sigmoid(gemmC1_t[otb + j] + gemmC2[otb + j] + back_bx[3 * H + j]);
                        back_ct[i * D * H + j] = ft[i * H + j] * back_ct_1[i * D * H + j] + it[i * H + j] * gt[i * H + j];
                        back_ht[i * D * H + j] = ot[i * H + j] * tanh(back_ct[i * D * H + j]);
                    }
                }
            }
            else{
                #pragma omp parallel for collapse(2)
                for (int i = 0; i < N; ++i) {
                    for (int j = 0; j < H; ++j) {
                        int itb = i * 4 * H;
                        int ftb = i * 4 * H + H;
                        int gtb = i * 4 * H + 2 * H;
                        int otb = i * 4 * H + 3 * H;
                        it[i * H + j] = sigmoid(gemmC1_t[itb + j] + gemmC2[itb + j]);
                        ft[i * H + j] = sigmoid(gemmC1_t[ftb + j] + gemmC2[ftb + j]);
                        gt[i * H + j] = tanh(   gemmC1_t[gtb + j] + gemmC2[gtb + j]);
                        ot[i * H + j] = sigmoid(gemmC1_t[otb + j] + gemmC2[otb + j]);
                        back_ct[i * D * H + j] = ft[i * H + j] * back_ct_1[i * D * H + j] + it[i * H + j] * gt[i * H + j];
                        back_ht[i * D * H + j] = ot[i * H + j] * tanh(back_ct[i * D * H + j]);
                    }
                }
            }
            back_ht_1 = back_ht;
            back_ht = back_ht - D * H * N;
            back_ct_1 = back_ct;
            back_ct = back_ct - D * H * N;
        }

    }


    //  copy last state to hy/cy, from(N,H*D) to (D,N,H)
    if (hy != 0) {
        if (D == UNIDIRECT) {
            float* ht_last = y + (T - 1) * N * H;
            float* ct_last = gateC + (T - 1) * N * H;
            #pragma omp parallel for collapse(2)
            for (int i = 0; i < N; i++)
                for (int j = 0; j < H; j++) {
                    hy[i * H + j] = ht_last[i * H + j];
                    cy[i * H + j] = ct_last[i * H + j];
                }
        } else {
            float* ht_last = y + (T - 1) * N * H * D;
            float* ht_back_last = y + H;
            float* ct_last = gateC + (T - 1) * N * H * D;
            float* ct_back_last = gateC + H;
            #pragma omp parallel for collapse(2)
            for (int i = 0; i < N; i++)
                for (int j = 0; j < H; j++) {
                    hy[i * H + j] = ht_last[i * D * H + j];
                    hy[N * H + i * H + j] = ht_back_last[i * D * H + j];
                    cy[i * H + j] = ct_last[i * D * H + j];
                    cy[N * H + i * H + j] = ct_back_last[i * D * H + j];
                }
        }
    }
}

#if 1

void lstm_xw_single_bwd(int T, int D, int N, int I, int H,
            float* ws, //D * (N*4H + N*H + N*H)
            float* x,  //[T,N,I]
            float* hx, //[D*N,H]
            float* cx, //[D*N,H]
            float* y,  //[T,N,D,H]
            float* wx, //[I,4H]
            float* wh, //[H,4H]
            float* dx,  //[T,N,I]
            float* dhx, //[D,N,H]
            float* dcx, //[D,N,H]
            float* dwx, //[I,4H]
            float* dwh, //[H,4H]
            float* dbx, //[1,4H]
            float* dy, //[T,N,D,H] 
            float* dhy, //[D,N,H]
            float* dcy, //[D,N,H]  
            float* gateI,//(L*)D*T*N*H
            float* gateF,//(L*)D*T*N*H
            float* gateG,//(L*)D*T*N*H
            float* gateO,//(L*)D*T*N*H
            float* gateC//(L*)[T,N,D,H]
)
{  
    int i,j,t,d;
    float* it;
    float* ft;
    float* gt;
    float* ot;
    float* ct;
    float* ht1;//[N,D,H]
    float* ct1;
    float* dxt;
    float* dyt;
    float* dat;
    float* da = ws; //[T,N,4H]
    float* dht1 = da + T * N * 4 * H;  //[D,N,H]
    float* dct1 = dht1 + D * N * H; //[D,N,H]
    float* hx_ = dct1 + D * N * H; //[N,D,H] 
    float* cx_ = hx_ + D * N * H; //[N,D,H]
    float* back_ht1;
    float* back_y = y + H; 
    float* back_dy = dy + H;
    float* back_dyt;
    float* back_dht1 = dht1 + N * H; //[N,H]
    float* back_dct1 = dct1 + N * H; //[N,H]
    float* back_dhy = dhy + N * H;
    float* back_dcx = dcx + N * H;
    float* back_dcy = dcy + N * H;
    float* back_gateI = gateI + T * N * H;
    float* back_gateF = gateF + T * N * H;
    float* back_gateG = gateG + T * N * H;
    float* back_gateO = gateO + T * N * H;
    float* back_gateC = gateC + H;
    float* back_wx = wx + I*4*H;
    float* back_wh = wh + H*4*H;
    float* back_dwx = dwx + I*4*H;
    float* back_dwh = dwh + H*4*H;
    float* back_dbx = dbx + 4*H;
    float* dht = dht1;
    float* dct = dct1;
    float* back_dht = back_dht1;
    float* back_dct = back_dct1;

    #pragma omp parallel for
    for(i = 0; i < D * H * 4 * H; ++i){
        dwh[i]=0;
    }
    if(dbx){
        #pragma omp parallel for
        for(i = 0; i < D * 4 * H; ++i){
            dbx[i]=0;
        }
    }
    // copy dhy,dcy to internal buffer,[D,N,H]
    #pragma omp parallel for
    for(i = 0; i < D * N * H; ++i){
        dht[i] = dhy[i];
        dct[i] = dcy[i];
    }
    // to make following iteration smooth, [D,N,H] -> [N,D,H]
    #pragma omp parallel for collapse(2)
    for(i=0; i < N; ++i){
        for(d=0; d < D; ++d){
            for(j=0; j < H; ++j){
                hx_[i * D * H + d * H + j] = hx[d * N * H + i * H + j];
                cx_[i * D * H + d * H + j] = cx[d * N * H + i * H + j];
            }
        }
    }

    for(t = T - 1; t >= 0; --t){
        //add dy[T,N,D,H] to dht[D,N,H]
        dyt = dy + t * N * D * H;
        #pragma omp parallel for collapse(2)
        for(i=0; i < N; ++i){
            for(j=0; j < H; ++j){
                dht[i*H+j] += dyt[i*D*H+j];
            }
        }
        it = gateI + t * N * H;
        ft = gateF + t * N * H;
        gt = gateG + t * N * H;
        ot = gateO + t * N * H;
        ct = gateC + t * N * D * H; //[T,N,D,H]
        ct1 = (t != 0)? (gateC + (t-1) * N * D * H) : (cx_);
        ht1 = (t != 0)? (y + (t-1) * N * D * H) : (hx_);
        dat = da + t * N * 4 * H;                               
        #pragma omp parallel for collapse(2)
        for( i=0; i < N; ++i){
            for( j=0; j < H; ++j){
                int base = i * H + j;
                int itb = i * 4 * H;
                int ftb = i * 4 * H + H;
                int gtb = i * 4 * H + 2 * H;
                int otb = i * 4 * H + 3 * H;
                
                float tanh_ct = tanh(ct[i * D * H + j]);
                dct[base] = dct[base] + dht[base] * ot[base] * (1 - tanh_ct * tanh_ct);
                dat[itb + j] = dct[base] * gt[base] * it[base] * (1 - it[base]);
                dat[ftb + j] = dct[base] * ct1[i * D * H + j] * ft[base] * (1 - ft[base]);
                dat[gtb + j] = dct[base] * it[base] * (1 - gt[base] * gt[base]);
                dat[otb + j] = dht[base] * tanh_ct * ot[base] * (1 - ot[base]);
                dct1[base] = dct[base] * ft[base];
            }
        }
        // dht1 = dat * wh.T    [N,H] = [N,4H] * [4H,H]
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, N, H, 4 * H, 1.0, 
          dat, 4 * H, wh, 4 * H, 0.0, dht1, H);

        // dwh = ht1.T * dat    [H,4H] = [H,N] * [N,4H]
        cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, H, 4 * H, 
          N, 1.0, ht1, D*H, dat, 4 * H, 1.0, dwh, 4 * H);
    }
    // dbx = e * da       [T,N,4H] -> [1,4H] reduce
    if(dbx){
        #pragma omp parallel for
        for(i = 0; i < 4 * H; ++i){
            for(j = 0; j < N * T; ++j){
                dbx[i] += da[j * 4 * H + i];
            }
        }
    }
    // dx = da * wx.T    [T*N,I] = [T*N,4H] * [4H,I]
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, T*N, I, 4 * H, 1, 
      da, 4 * H, wx, 4 * H, 0, dx, I);
    // dwx = x.T * da    [I,4H] = [I,T*N] * [T*N,4H]
    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, I, 4 * H, T*N, 1, x, 
      I, da, 4 * H, 0, dwx, 4 * H);

    /******************************  D = 2 *******************************/

    if(D == 2){
        for(t = 0; t < T; ++t){
            //add dy[T,N,D,H] to dhy[D,N,H]
            back_dyt = back_dy + t * N * D * H;
            #pragma omp parallel for collapse(2)
            for(i=0; i < N; ++i){
                for(j=0; j < H; ++j){
                    back_dht[i*H+j] += back_dyt[i*D*H + j];
                }
            }
            it = back_gateI + t * N * H;
            ft = back_gateF + t * N * H;
            gt = back_gateG + t * N * H;
            ot = back_gateO + t * N * H;
            ct = back_gateC + t * N * D * H; //[T,N,D,H]
            // reuse ct1 in D=2, but dct could not be reused
            ct1 = (t != T-1)? (back_gateC + (t+1) * N * D * H) : (cx_ + H);
            // reuse ht1 in D=2, but dht could not be reused
            ht1 = (t != T-1)? (back_y + (t+1) * N * D * H) : (hx_ + H);
            // reuse dat in D=2
            dat = da + t * N * 4 * H; //reuse buffer for D=1 and D=2
            #pragma omp parallel for collapse(2)
            for( i=0; i < N; ++i){
                for( j=0; j < H; ++j){
                    int base = i * H + j;
                    int itb = i * 4 * H;
                    int ftb = i * 4 * H + H;
                    int gtb = i * 4 * H + 2 * H;
                    int otb = i * 4 * H + 3 * H;
                    
                    float tanh_ct = tanh(ct[i * D * H + j]);
                    back_dct[base] = back_dct[base] + back_dht[base] * ot[base] * (1 - tanh_ct * tanh_ct);
                    dat[itb + j] = back_dct[base] * gt[base] * it[base] * (1 - it[base]);
                    dat[ftb + j] = back_dct[base] * ct1[i * D * H + j] * ft[base] * (1 - ft[base]);
                    dat[gtb + j] = back_dct[base] * it[base] * (1 - gt[base] * gt[base]);
                    dat[otb + j] = back_dht[base] * tanh_ct * ot[base] * (1 - ot[base]);
                    back_dct1[base] = back_dct[base] * ft[base];
                }
            }

            // dht1 = da * wh.T    [N,H] = [N,4H] * [4H,H]
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, N, H, 4 * H, 1, 
              dat, 4 * H, back_wh, 4 * H, 0, back_dht1, H);
            // dwh = ht1.T * da    [H,3H] = [H,N] * [N,4H]
            cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, H, 4 * H, 
              N, 1, ht1, D*H, dat, 4 * H, 1, back_dwh, 4 * H);
        }

        // dbx = e * da       [T,N,4H] -> [1,4H] reduce
        if(dbx){
            #pragma omp parallel for
            for(i = 0; i < 4 * H; ++i){
                for(j = 0; j < N * T; ++j){
                    back_dbx[i] += da[j * 4 * H + i];
                }
            }
        }
        // dxt = da * wx.T    [T*N,I] = [T*N,4H] * [4H,I]
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, T*N, I, 4 * H, 1, 
          da, 4 * H, back_wx, 4 * H, 1, dx, I);

        // dwx = xt.T * da    [I,4H] = [I,N] * [N,4H]
        cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, I, 4 * H, T*N, 1, x, 
          I, da, 4 * H, 0, back_dwx, 4 * H);
    }

    #pragma omp parallel for
    for(i = 0; i < D * N * H; ++i){
        dhx[i] = dht1[i];
        dcx[i] = dct1[i];
    }

}

#endif

 /*
 * @brief:  lstm_xw_train_get_workspace_size
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
 *             gateI [L,D,T,N,H]
 *             gateF [L,D,T,N,H]
 *             gateG [L,D,T,N,H]
 *             gateO [L,D,T,N,H]
 *             gateC [L,T,N,D,H]
 *
 *             TEMP for FORWARD
 *             gemmC1 [D,T,N,4H]
 *             gemmC2 [N,4H]
 *
 *             TEMP for BACKWARD
 *             da   [T,N,4H]
 *             dht1 [D,N,H]
 *             dct1 [D,N,H]
 *             hx_  [N,D,H]
 *             cx_  [N,D,H]
 *
 */ 

int lstm_xw_train_get_workspace_size(int I, int H, int T, int N, int bi, int L){
    int D = (bi==1)? 2:1;
    return 5 * L * T * D * N * H + 8 * N * H + 2 * T * N * 4 * H;
}


void print_forward_desc(RNNForwardDesc desc){
    printf("    L = %d, D = %d, T = %d, N = %d, I = %d, H = %d \n",
        desc.L,desc.D,desc.T,desc.N,desc.I,desc.H);
    printf("x = 0x%x, hx = 0x%x, cx = 0x%x, wx = 0x%x, wh = 0x%x, bx = 0x%x, hy = 0x%x, cy = 0x%x \n",
            desc.x, desc.hx, desc.cx, desc.wx, desc.wh, desc.bx, desc.hy, desc.cy);
}
void print_backward_desc(RNNBackwardDesc desc){
    printf("    L = %d, D = %d, T = %d, N = %d, I = %d, H = %d \n",
        desc.L,desc.D,desc.T,desc.N,desc.I,desc.H);
    printf("x = 0x%x, hx = 0x%x, cx = 0x%x, wx = 0x%x, wh = 0x%x, hy = 0x%x, cy = 0x%x \n",
            desc.x, desc.hx, desc.cx, desc.wx, desc.wh, desc.hy, desc.cy);
    printf("dx = 0x%x, dhx = 0x%x, dcx = 0x%x, dwx = 0x%x, dwh = 0x%x, dbx = 0x%x, dhy = 0x%x, dcy = 0x%x \n",
            desc.dx, desc.dhx, desc.dcx, desc.dwx, desc.dwh, desc.dbx, desc.dhy, desc.dcy);
}


int lstm_xw_forward(RNNForwardDesc desc){
    //print_forward_desc(desc);
    float* gateI = desc.ws;
    float* gateF = gateI + desc.D * desc.T * desc.N * desc.H;
    float* gateG = gateF + desc.D * desc.T * desc.N * desc.H;
    float* gateO = gateG + desc.D * desc.T * desc.N * desc.H;
    float* gateC = gateO + desc.D * desc.T * desc.N * desc.H;
    float* ws_new = gateC + desc.D * desc.T * desc.N * desc.H;
    lstm_forward_single_sequential(desc.T, desc.D, desc.N, desc.I, desc.H,
        ws_new, desc.x, desc.hx, desc.cx, desc.wx, desc.wh, desc.bx, desc.y, desc.hy, desc.cy,
        gateI, gateF, gateG, gateO, gateC);
    return 0;

}
int lstm_xw_backward(RNNBackwardDesc desc){
    //print_backward_desc(desc);
    float* gateI = desc.ws;
    float* gateF = gateI + desc.D * desc.T * desc.N * desc.H;
    float* gateG = gateF + desc.D * desc.T * desc.N * desc.H;
    float* gateO = gateG + desc.D * desc.T * desc.N * desc.H;
    float* gateC = gateO + desc.D * desc.T * desc.N * desc.H;
    float* ws_new = gateC + desc.D * desc.T * desc.N * desc.H;

    lstm_xw_single_bwd(desc.T, desc.D, desc.N, desc.I, desc.H,
        ws_new, desc.x, desc.hx, desc.cx, desc.y, desc.wx, desc.wh, desc.dx,
        desc.dhx, desc.dcx, desc.dwx, desc.dwh, desc.dbx, desc.dy, desc.dhy,
        desc.dcy, gateI, gateF, gateG, gateO, gateC);

    return 0;

}


}
