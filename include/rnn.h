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

//will remain
int rnn_xw_infer_get_workspace_size(int input_dim, int hid, int max_time_step, 
  int max_batch_size);
int lstm_wx_train_get_workspace_size(int input_dim, int hid, int max_time_step, 
  int max_batch_size);
int lstm_wx_infer_get_workspace_size(int input_dim, int hid, int max_time_step, 
  int max_batch_size);
int lstm_xw_infer_get_workspace_size(int input_dim, int hid, int max_time_step, 
  int max_batch_size);
int lstm_xw_train_get_workspace_size(int input_dim, int hid, int max_time_step, 
  int max_batch_size, int bidirectional);
int gru_xw_train_get_workspace_size(int input_dim, int hid, int max_time_step, 
  int max_batch_size, int bi, int num_layer);
int gru_xw_infer_get_workspace_size(int input_dim, int hid, int max_time_step, 
  int max_batch_size, int bi, int num_layer);

int lstm_wx_infer(void* buf, int batch_size, int time_step, int input_dim, int hid, int max_time_step, int max_batch_size, 
    float* w_x, float* w_h, float* b, float* x, float* h_0, float* c_0, float* h_out, float* c_out, bool return_sequences, int mode = 0);
int lstm_xw_infer(void* buf, int batch_size, int time_step, int input_dim, int hid, int max_time_step, int max_batch_size, 
    float* w_x, float* w_h, float* b, float* x, float* h_0, float* c_0, float* h_out, float* c_out, bool return_sequences, int mode = 0);

int rnn_xw_infer(void* buf, int batch_size, int time_step, int input_dim, int hid, int max_time_step, int max_batch_size, 
    float* w_x, float* w_h, float* b, float* x, float* h_0, float* h_out, bool return_sequences, int mode = 0);

int lstm_wx_training(void* buf, int batch_size, int time_step, int input_dim, int hid, int max_time_step, int max_batch_size,
    float* w_x, float* w_h, float* b, float* x, float* h_0, float* c_0, float* grad_last, float* dall);

//int lstm_xw_forward(float* buf, int batch_size, int time_step, int input_dim, int hid, int max_time_step, int max_batch_size,
//    float* w_x, float* w_h, float* b, float* x, float* h_0, float* c_0, float* h_out, float* c_out, int mode = 0);

int lstm_xw_forward(void* buf,
                    int batch_size,
                    int time_step,
                    int input_dim,
                    int hidden_dim,
                    float* wx,      //I*4H      
                    float* wh,      //H*4H      
                    float* bias,    //4H, because bx == bh, so just need one
                    float* x, //T*N*I
                    float* h_0,     //N*H
                    float* c_0,     //N*H
                    float* h_out,    //T*N*H
                    float* c_out,   //T*N*H
                    int mode = 0,
                    int bidirectional=0
                    ); 

int lstm_xw_backward(
        void* buf,
        int num_layer,
        int num_direction,
        int time_step,
        int batch_size,
        int input_dim,
        int hidden_size,
        float * x,
        float * h_0,
        float * c_0,
        float * wx,
        float * wh,
        float * h_out,
        float * c_out,
        float * grad_h_out,     //(D, N, H)
        float * grad_c_out,     //(D, N, H)
        float * grad_x_input,   //(T, N, I)
        float * grad_h0,        //(D, N, H)
        float * grad_c0,        //(D, N, H)
        float * grad_wx,        //(D, I, 4H
        float * grad_wh,        //(D, H, 4H)
        float * grad_bias,       //(D, 4H)
        int mode);

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
                    int mode);

int  gru_xw_forward(int L,
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
                    int mode);

int gru_xw_backward( int num_layers,
                     int time_step,
                     int num_direction,
                     int batch_size,
                     int input_dim,
                     int hidden_size,
                     float* dy,
                     float* dhy,
                     float* x,
                     float* hx,
                     float* wx,
                     float* wh,
                     float* dx,
                     float* dhx,
                     float* dwx,
                     float* dwh,
                     float* dbx,
                     float* dbh,
                     float* ws,
                     int mode);

int  gru_xw_forward_prof(int L,
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
                    int mode,
                    double* time);

int gru_xw_backward_prof(int num_layers,
                     int time_step,
                     int num_direction,
                     int batch_size,
                     int input_dim,
                     int hidden_size,
                     float* dy,
                     float* dhy,
                     float* x,
                     float* hx,
                     float* wx,
                     float* wh,
                     float* dx,
                     float* dhx,
                     float* dwx,
                     float* dwh,
                     float* dbx,
                     float* dbh,
                     float* ws,
                     int mode,
                     double* time);


#ifdef __cplusplus
}
#endif
