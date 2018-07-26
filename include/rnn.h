#pragma once

#ifdef __cplusplus
#include <cstddef>
#include <mkl.h>
#include <omp.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

extern "C" {
#endif

#define ENABLE_OMP_SETTING  0
#define UNIDIRECT 1
#define BIDIRECT 2
// descriptor support RNN/LSTM/GRU
typedef struct ForwardDesc{
    int L;
    int D;
    int T;
    int N;
    int I;
    int H;
    float * ws;
    float * x;
    float * hx;
    float * cx;
    float * wx;
    float * wh; 
    float * bx; 
    float * bh; 
    float * y;
    float * hy;
    float * cy;
    int algo;
    double * time; 
} RNNForwardDesc;

typedef struct BackwardDesc{
    int L;
    int D;
    int T;
    int N;
    int I;
    int H;
    float * ws;
    float * x;
    float * hx; 
    float * cx; 
    float * wx; 
    float * wh;
    float * y;
    float * hy;
    float * cy;
    float * dx;
    float * dhx;
    float * dcx;
    float * dwx;
    float * dwh;
    float * dbx;
    float * dbh;
    float * dy;
    float * dhy;
    float * dcy;
    int algo;
    double * time; 
} RNNBackwardDesc;


int lstm_xw_infer_get_workspace_size(int L, int D, int T, int N, int I, int H);
int lstm_xw_train_get_workspace_size(int L, int D, int T, int N, int I, int H);
int gru_xw_train_get_workspace_size(int L, int D, int T, int N, int I, int H);
int gru_xw_infer_get_workspace_size(int L, int D, int T, int N, int I, int H);

int gru_xw_infer(RNNForwardDesc desc);
int gru_xw_forward(RNNForwardDesc desc);
int gru_xw_backward(RNNBackwardDesc desc);
int lstm_xw_infer(RNNForwardDesc desc);
int lstm_xw_forward(RNNForwardDesc desc);
int lstm_xw_backward(RNNBackwardDesc desc);


#ifdef __cplusplus
}
#endif
