//define parameter in workspace
#define MODE_RNN  1
#define MODE_LSTM 2
#define MODE_GRU  3

#define TRAINING   1
#define INFERENCE  2

typedef int bool;
#define TRUE 1
#define FALSE 0

int pytorch_get_workspace_size(int mode, int train, int input_size, int hid, 
  int time_step, int batch_size, int bidirectional, int num_layers);

/*LSTM interface*/
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
    int num_layers);

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
    THFloatTensor * h_out,    //(L*D, N, H)
    THFloatTensor * c_out,    //(L*D, N, H)
    int batch_size,
    int seq_length,
    int input_size,
    int hidden_size,
    int num_layers,
    bool bidirectional);



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
        THFloatTensor * grad_bias       //(D, 4H)
        );

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
    int num_layer);

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
    int num_direction);

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
    int num_direction);

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
    );

