#include <iostream>
#include "rnn.h"
#include <assert.h>
#include "TH.h"

extern "C"{

#define MODE_RNN        1
#define MODE_LSTM       2
#define MODE_GRU        3

#define TRAINING  1
#define INFERENCE 2


float * getTHBuffer(THFloatTensor * input)
{
    if(input == NULL || input->size == 0){
        return NULL;
    }else{
        return ((float *)(input->storage->data + input->storageOffset));
    }
}

void copy_c_state_to_cy(float *c_state, float *cy, int T, int N, int H, int D) {
    float *cy_forward = cy;
    float *cy_backward = cy + N * H;

    float (*c_state_forward)[H * D] = (float (*)[H * D])(c_state + (T - 1) * N * H * D);
    float (*c_state_backward)[H * D] = (float (*)[H * D])(c_state + H);


    // copy forward cy:
    int n;
    for (n = 0; n < N; n ++) {
        memcpy(cy_forward + n * H, (float *)(c_state_forward[n]), H * sizeof(float));
    }

    // copy backward cy, if "bidirectional"
    if (D == 2) {
        for (n = 0; n < N; n ++) {
            memcpy(cy_backward + n * H, (float *)(c_state_backward[n]), H * sizeof(float));
        }
    }
}

void copy_h_state_to_hy(float *h_state, float *hy, int T, int N, int H, int D) {
    copy_c_state_to_cy(h_state, hy, T, N, H, D);
}

int pytorch_get_workspace_size(int mode, int train, int input_size, 
    int hidden_size, int time_step, int batch_size, int bidirectional,
    int num_layer){
    int rtn = 0;
    if (mode == MODE_LSTM){
        if (train == TRAINING){
            rtn = lstm_xw_train_get_workspace_size(input_size, hidden_size, time_step, batch_size, bidirectional);
        } else if(train == INFERENCE){
            rtn = lstm_xw_infer_get_workspace_size(input_size, hidden_size, time_step, batch_size);
        }
    }
    if (mode == MODE_GRU){
        if (train == TRAINING){
            rtn = gru_xw_train_get_workspace_size(input_size, hidden_size, 
                  time_step, batch_size, bidirectional, num_layer);
        } else if(train == INFERENCE){
            rtn = gru_xw_infer_get_workspace_size(input_size, hidden_size, 
                  time_step, batch_size, 1, 1);
        }
    }
    return rtn;
}

int pytorch_lstm_inference(
    THFloatTensor * workspace,
    THFloatTensor * input,
    THFloatTensor * w_x,
    THFloatTensor * w_h,
    THFloatTensor * bias,
    THFloatTensor * h0,
    THFloatTensor * c0,
    THFloatTensor * h_out,
    THFloatTensor * c_out,
    int batch_size,
    int seq_length,
    int input_size,
    int hidden_size,
    int num_layers)
{
    //printf("C      pytorch_lstm_forward in binding.cpp, batch_size = %d, seq_length = %d, input_size = %d, hidden_size = %d, \
    //    num_layers = %d\n", batch_size, seq_length, input_size, hidden_size, num_layers);

    int max_time_step = seq_length;
    int max_batch_size = batch_size;
    lstm_xw_infer(
        getTHBuffer(workspace),
        batch_size,
        seq_length,
        input_size,
        hidden_size,
        max_time_step,
        max_batch_size,
        getTHBuffer(w_x),
        getTHBuffer(w_h),
        getTHBuffer(bias),
        getTHBuffer(input),
        getTHBuffer(h0),
        getTHBuffer(c0),
        getTHBuffer(h_out),
        getTHBuffer(c_out),
        true,
        0
        );
    //float * ht = getTHBuffer(h_out);
    //printf("h_out = %.4f, %.4f, %.4f \n", ht[0], ht[1], ht[2]);
}


int pytorch_lstm_forward(
    THFloatTensor * workspace,
    THFloatTensor * input, 
    THFloatTensor * w_x, 
    THFloatTensor * w_h, 
    THFloatTensor * bias, 
    THFloatTensor * h0, 
    THFloatTensor * c0, 
    THFloatTensor * h_state,  //(T, N,D*H) 
    THFloatTensor * c_state,  //(T, N,D*H)
    THFloatTensor * hy,    //(L*D, N, H)
    THFloatTensor * cy,    //(L*D, N, H)
    int batch_size,
    int seq_length,
    int input_size,
    int hidden_size,
    int num_layers,
    bool bidirectional)
{
    int max_time_step = seq_length;
    int max_batch_size = batch_size;

    lstm_xw_forward(
        getTHBuffer(workspace), 
        batch_size, 
        seq_length,
        input_size,
        hidden_size,
        getTHBuffer(w_x),
        getTHBuffer(w_h),
        getTHBuffer(bias),
        getTHBuffer(input),
        getTHBuffer(h0),
        getTHBuffer(c0),
        getTHBuffer(h_state),
        getTHBuffer(c_state),
        0,
        bidirectional);

        int D = bidirectional ? 2 : 1;

        int i=0;
        float * hy_p = getTHBuffer(hy);
        float * cy_p = getTHBuffer(cy);
        float * h_state_p = getTHBuffer(h_state);
        float * c_state_p = getTHBuffer(c_state);
        int start = (seq_length-1)*batch_size*hidden_size;
        
        //
        // We have to be very carefully here, since there are format conversions.
        // c_state: (T, N, H*D)
        // c_y    : (D, N, H)
        // 

        //copy h_state to h_out
        copy_h_state_to_hy(h_state_p, hy_p, seq_length, batch_size, hidden_size, D);

        //copy c_state to c_out
        copy_c_state_to_cy(c_state_p, cy_p, seq_length, batch_size, hidden_size, D);

}


int pytorch_lstm_backward(
        int num_layer,
        int num_direction,
        int time_step,
        int batch_size,
        int input_dim,
        int hidden_size,
        THFloatTensor * workspace,
        THFloatTensor * x,
        THFloatTensor * h0,
        THFloatTensor * c0,
        THFloatTensor * h_out,
        THFloatTensor * c_out,
        THFloatTensor * w_x,
        THFloatTensor * w_h,
        THFloatTensor * grad_h_out,     //(D, N, H)
        THFloatTensor * grad_c_out,     //(D, N, H)
        THFloatTensor * grad_x,         //(T, N, I)
        THFloatTensor * grad_h0,        //(D, N, H)
        THFloatTensor * grad_c0,        //(D, N, H)
        THFloatTensor * grad_wx,        //(D, I, 4H
        THFloatTensor * grad_wh,        //(D, H, 4H)
        THFloatTensor * grad_bias      //(D, 4H)
        ){
        assert(workspace != NULL);

        lstm_xw_backward(
            getTHBuffer (workspace),
            num_layer,
            num_direction,
            time_step,
            batch_size,
            input_dim,
            hidden_size,
            getTHBuffer (x),
            getTHBuffer (h0),
            getTHBuffer (c0),
            getTHBuffer ( w_x),
            getTHBuffer ( w_h),
            getTHBuffer (h_out),
            getTHBuffer (c_out),
            getTHBuffer ( grad_h_out),     //(D, N, H)
            getTHBuffer ( grad_c_out),     //(D, N, H)
            getTHBuffer ( grad_x),         //(T, N, I)
            getTHBuffer ( grad_h0),        //(D, N, H)
            getTHBuffer ( grad_c0),        //(D, N, H)
            getTHBuffer ( grad_wx),        //(D, I, 4H
            getTHBuffer ( grad_wh),        //(D, H, 4H)
            getTHBuffer ( grad_bias),      //(D, 4H)
            0
        );

}
int pytorch_gru_infer(
    THFloatTensor * workspace,
    THFloatTensor * input, 
    THFloatTensor * w_x, 
    THFloatTensor * w_h, 
    THFloatTensor * b_x, 
    THFloatTensor * b_h, 
    THFloatTensor * h0, 
    THFloatTensor * h_out,
    THFloatTensor * h_t,
    int batch_size,
    int seq_length,
    int input_size,
    int hidden_size,
    int num_layers,
    int num_direction)
{
    int max_time_step = seq_length;
    int max_batch_size = batch_size;
    gru_xw_infer(
        num_layers,
        seq_length,
        num_direction,
        batch_size, 
        input_size,
        hidden_size,
        getTHBuffer(input),
        getTHBuffer(h0),
        getTHBuffer(w_x),
        getTHBuffer(w_h),
        getTHBuffer(b_x),
        getTHBuffer(b_h),
        getTHBuffer(h_out),
        getTHBuffer(h_t),
        getTHBuffer(workspace), 
        0);
}
int pytorch_gru_forward(
    THFloatTensor * workspace,
    THFloatTensor * input, 
    THFloatTensor * w_x, 
    THFloatTensor * w_h, 
    THFloatTensor * b_x, 
    THFloatTensor * b_h, 
    THFloatTensor * h0, 
    THFloatTensor * h_out,
    THFloatTensor * h_t,
    int batch_size,
    int seq_length,
    int input_size,
    int hidden_size,
    int num_layers,
    int num_direction)
{
    int max_time_step = seq_length;
    int max_batch_size = batch_size;
    gru_xw_forward(
        num_layers,
        seq_length,
        num_direction,
        batch_size, 
        input_size,
        hidden_size,
        getTHBuffer(input),
        getTHBuffer(h0),
        getTHBuffer(w_x),
        getTHBuffer(w_h),
        getTHBuffer(b_x),
        getTHBuffer(b_h),
        getTHBuffer(h_out),
        getTHBuffer(h_t),
        getTHBuffer(workspace), 
        0);
}

int pytorch_gru_backward(
    int num_layers,
    int num_direction,
    int time_step,
    int batch_size,
    int input_dim,
    int hidden_size,
    THFloatTensor * workspace,
    THFloatTensor * x,
    THFloatTensor * h0,
    THFloatTensor * h_out,
    THFloatTensor * w_x,
    THFloatTensor * w_h,
    THFloatTensor * b_x,
    THFloatTensor * b_h,
    THFloatTensor * grad_y,     //(D, N, H)
    THFloatTensor * grad_hy,     //(D, N, H)
    THFloatTensor * grad_x,         //(T, N, I)
    THFloatTensor * grad_h0,        //(D, N, H)
    THFloatTensor * grad_wx,        //(D, N, H)
    THFloatTensor * grad_wh,        //(D, I, 4H
    THFloatTensor * grad_bx,        //(D, H, 4H)
    THFloatTensor * grad_bh      //(D, 4H)
    )
{
    assert(workspace != NULL);
    assert(x != NULL);
    assert(h0 != NULL);
    assert(h_out != NULL);
    assert(w_x != NULL);
    assert(w_h != NULL);
    assert(grad_y != NULL);
    assert(grad_hy != NULL);
    assert(grad_x != NULL);
    assert(grad_h0 != NULL);
    assert(grad_wx != NULL);
    assert(grad_wh != NULL);
    gru_xw_backward(
        num_layers,
        time_step,
        num_direction,
        batch_size,
        input_dim,
        hidden_size,
        getTHBuffer (grad_y),     //(D, N, H)
        getTHBuffer (grad_hy),     //(D, N, H)
        getTHBuffer (x),
        getTHBuffer (h0),
        getTHBuffer (w_x),
        getTHBuffer (w_h),
        getTHBuffer (grad_x),         //(T, N, I)
        getTHBuffer (grad_h0),        //(D, N, H)
        getTHBuffer (grad_wx),     //(D, N, H)
        getTHBuffer (grad_wh),     //(D, N, H)
        getTHBuffer (grad_bx),        //(D, I, 4H
        getTHBuffer (grad_bh),        //(D, H, 4H)
        getTHBuffer (workspace),
        0);
}


int pytorch_rnn_forward(
    THFloatTensor * workspace,
    THFloatTensor * input, 
    THFloatTensor * w_x, 
    THFloatTensor * w_h, 
    THFloatTensor * bias, 
    THFloatTensor * h0, 
    THFloatTensor * h_out, 
    int batch_size,
    int seq_length,
    int input_size,
    int hidden_size,
    int num_layers)
{

    int max_time_step = seq_length;
    int max_batch_size = batch_size;
    rnn_xw_infer(
        getTHBuffer(workspace), 
        batch_size, 
        seq_length,
        input_size,
        hidden_size,
        max_time_step,
        max_batch_size,
        getTHBuffer(w_x),
        getTHBuffer(w_h),
        getTHBuffer(bias),
        getTHBuffer(input),
        getTHBuffer(h0),
        getTHBuffer(h_out),
        true,
        0
        );
}

}




