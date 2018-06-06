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
#define UNIDIRECT 1
#define BIDIRECT 2
/*
 * @brief:  gru_xw_single_infer 
 *          single layer infering computation 
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
 *          ws: workspace, temporary buffer
 *
 * @formula_list:  rt = sigmoid(xt * Wrx + brx + ht-1 * Wrh + brh)
 *                 zt = sigmoid(xt * Wzx + bzx + ht-1 * Wzh + bzh)
 *                 nt = tanh(xt * Wnx + bnx + rt * (ht-1 * Wnh + bnh))
 *                 ht = (1 - zt) * nt + zt * ht-1
 */

void gru_xw_single_infer(int T, int D, int N, int I, int H, 
    float* x,    //[T,N,H*D]
    float* hx,   //[D*N,H] 
    float* wx,   //2*(I*3H + H*3H + 6H)
    float* wh,   //2*(I*3H + H*3H + 6H)
    float* bx,   //2*(I*3H + H*3H + 6H)
    float* bh,   //2*(I*3H + H*3H + 6H)
    float* y,    //[T,N,H*D]
    float* hy,   //[D*N,H]
    float* ws
){
    int m = N;
    int n = 3*H;
    int k = I;
    float* ht = y;
    float* ht_1 = y;
    float* back_ht_1 = y + (T-1)*N*H*D + H;
    float* back_ht = back_ht_1;

    float* gemmC1  = ws;              // [D,T,N,3H]
    float* gemmC2  = gemmC1 + D*T*N*3*H;  // N*3H
    float* rt = gemmC2 + N*3*H;
    float* zt = rt + N*H;
    float* nt = zt + N*H;
    float* back_wx = wx + I * 3 * H;
    float* back_wh = wh + H * 3 * H;
    float* back_bx = (bx != NULL)? bx + 3 * H : NULL;
    float* back_bh = (bh != NULL)? bh + 3 * H : NULL;
    float* back_gemmC1 = gemmC1 + T * N * 3 * H;
    float* gemmC1_t = gemmC1;
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
    //x*wx : [T*N,I] * [I,3H]
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N*T, n, k, 1, x, k, 
      wx, n, 0.0, gemmC1, n);
    if(D == BIDIRECT){
        cblas_sgemm(CblasRowMajor,CblasNoTrans, CblasNoTrans, N*T, n, k, 1, x, 
          k, back_wx, n, 0.0, back_gemmC1, n);
    }
    for (int t =0; t < T; t++) {
        //  perform the first direction, X*wx and H * wh for each step
        //  ht-1*wh, ht-1:[N,H] wh:[H, 3H]
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, H, 1, 
                    ht_1, D*H, wh, n, 0.0, gemmC2, n);
        
        gemmC1_t = gemmC1 + t * N * 3 * H;
        if (bx and bh){
        #pragma omp parallel for collapse(2)
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < H; ++j) {
                //printf("N(%d)=%d, H(%d)=%d\n", N, i, H,j);
                int rtb = i * 3 * H;
                int ztb = i * 3 * H + H;
                int ntb = i * 3 * H + 2 * H;
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


        ht_1 = ht;
        ht = ht + D * H * N;
        //  perform the second direction
        if (D == BIDIRECT) {
            gemmC1_t = back_gemmC1 + (T - 1 - t) * N * 3 * H;
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, H, 1, 
                        back_ht_1, D * H, back_wh, n, 0.0, gemmC2, n);

            if(back_bx and back_bh){
            #pragma omp parallel for collapse(2)
            for (int i = 0; i < N; ++i) {
                for (int j = 0; j < H; ++j) {
                    int rtb = i * 3 * H;
                    int ztb = i * 3 * H + H;
                    int ntb = i * 3 * H + 2 * H;
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
            back_ht_1 = back_ht;
            back_ht = back_ht - D * H * N;
        }
    }
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
}

/*
 * @brief:  gru_xw_seq_infer
 *          multi-layer infering computation 
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
 *          ws: workspace, temporary buffer
 *
 * @desc: sequantial implementation of GRU inference
 *        call function 'gru_xw_single_infer' to compute each layer
 */

void gru_xw_seq_infer(int L, int T, int D, int N, int I, int H, 
    float* x, float* hx, float* wx, float* wh, float* bx, float* bh, float* y, 
    float* hy, float* ws 
){
    float* y_tmp = ws;
    float* y_l = x;
    float* ws2 = y_tmp + D * T * N * H;
    float* wx_l = wx;
    float* wh_l = wh;
    float* bx_l = bx;
    float* bh_l = bh;
    float* x_l = x;
    float* hx_l = hx;
    float* hy_l = hy;
    
    for (int l = 0; l < L; l++) {  //  for each Layer
        x_l = y_l;
        if((L+l)%2){
            y_l = y;
        }
        else{
            y_l = y_tmp;
        }
        
        gru_xw_single_infer(T, D, N, I, H, x_l, hx_l, wx_l, wh_l, bx_l, bh_l, 
          y_l, hy_l, ws2);
        hx_l = hx_l + D * N * H;
        hy_l = hy_l + D * N * H;        
        wh_l = wh_l + H * 3 * H * D;
        bx_l = bx_l + 3 * H * D;
        bh_l = bh_l + 3 * H * D;

        if (l == 0) {
            wx_l = wx_l + I * H * 3 * D;
            I = 2 * H;
        } else {
            wx_l = wx_l + D * H * H * 3 * D;
        }
    }
}


/*
 * @brief:  gru_xw_infer
 *          interface function of GRU inference
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
 *          ws: workspace, temporary buffer
 *          mode: specify the mode of implementation
 *
 * @desc: call different implementations of inference functions.
 *        (Currently, there is only sequential version.)
 *
 */

int  gru_xw_infer(int L,
                  int T,
                  int D,
                  int N,
                  int I,
                  int H,
                  float* x,
                  float* hx,
                  float* wx,
                  float* wh,
                  float* bx,
                  float* bh,
                  float* y,
                  float* hy,
                  float* ws,
                  int mode)
{
    gru_xw_seq_infer(L, T, D, N, I, H, x, hx, wx, wh, bx, bh, y, hy, ws);
}


 /*
 * @brief:  gru_xw_infer_get_workspace_size
 *          get the size of buffer space for GRU inference
 *
 * @params: I:  input_dim
 *          H:  hidden_size
 *          T:  seq_length
 *          N:  batch_size
 *          bi: whether bi-directional or not (1 or 0)
 *          L:  num_layers
 *
 * @workspace: gemmC1 [D,T,N,3H]
 *             gemmC2 [N,3H]
 *             rt     [N,H]
 *             zt     [N,H]
 *             nt     [N,H]
 *             y_tmp  [T,N,D*H]
 *
 */ 

int gru_xw_infer_get_workspace_size(int I, int H, int T, int N, int bi, int L)
{
    int D = (bi==1)? 2:1;
    return 4 * T * D * N * H + 6 * N * H;
}

