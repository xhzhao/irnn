//define parameter in workspace
#define MODE_RNN  1
#define MODE_LSTM 2
#define MODE_GRU  3

#define TRAINING   1
#define INFERENCE  2

#define TRUE 1
#define FALSE 0

int pytorch_get_workspace_size(int mode, int train, int L, int D, int T, int N, int I, int H);

int pytorch_gru_infer(
    int L,
    int D,
    int T,
    int N,
    int I,
    int H,
    THFloatTensor * ws,
    THFloatTensor * x, 
    THFloatTensor * hx, 
    THFloatTensor * wx, 
    THFloatTensor * wh, 
    THFloatTensor * bx, 
    THFloatTensor * bh, 
    THFloatTensor * y,
    THFloatTensor * hy);

int pytorch_gru_forward(
    int L,
    int D,
    int T,
    int N,
    int I,
    int H,
    THFloatTensor * ws,
    THFloatTensor * x, 
    THFloatTensor * hx, 
    THFloatTensor * wx, 
    THFloatTensor * wh, 
    THFloatTensor * bx, 
    THFloatTensor * bh, 
    THFloatTensor * y,
    THFloatTensor * hy);

int pytorch_gru_backward(
    int L,
    int D,
    int T,
    int N,
    int I,
    int H,
    THFloatTensor * ws,
    THFloatTensor * x,
    THFloatTensor * hx,
    THFloatTensor * wx,
    THFloatTensor * wh,
    THFloatTensor * dx, 
    THFloatTensor * dhx,
    THFloatTensor * dwx,
    THFloatTensor * dwh,
    THFloatTensor * dbx,
    THFloatTensor * dbh,
    THFloatTensor * dy, 
    THFloatTensor * dhy);

int pytorch_lstm_infer(
    int L,
    int D,
    int T,
    int N,
    int I,
    int H,
    THFloatTensor * ws,
    THFloatTensor * x, 
    THFloatTensor * hx, 
    THFloatTensor * cx, 
    THFloatTensor * wx, 
    THFloatTensor * wh, 
    THFloatTensor * bx, 
    THFloatTensor * y,
    THFloatTensor * hy,
    THFloatTensor * cy);
int pytorch_lstm_forward(
    int L,
    int D,
    int T,
    int N,
    int I,
    int H,
    THFloatTensor * ws,
    THFloatTensor * x, 
    THFloatTensor * hx, 
    THFloatTensor * cx, 
    THFloatTensor * wx, 
    THFloatTensor * wh, 
    THFloatTensor * bx, 
    THFloatTensor * y,
    THFloatTensor * hy,
    THFloatTensor * cy);
int pytorch_lstm_backward(
    int L,
    int D,
    int T,
    int N,
    int I,
    int H,
    THFloatTensor * ws,
    THFloatTensor * x,
    THFloatTensor * hx,
    THFloatTensor * cx,
    THFloatTensor * wx,
    THFloatTensor * wh,
    THFloatTensor * y,
    THFloatTensor * hy,
    THFloatTensor * cy,
    THFloatTensor * dx, 
    THFloatTensor * dhx,
    THFloatTensor * dcx,
    THFloatTensor * dwx,
    THFloatTensor * dwh,
    THFloatTensor * dbx,
    THFloatTensor * dy, 
    THFloatTensor * dhy, 
    THFloatTensor * dcy);
