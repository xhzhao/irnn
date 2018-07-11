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

    int pytorch_get_workspace_size(int mode, int train, int input_size,
        int hidden_size, int time_step, int batch_size, int bidirectional,
        int num_layer){
        int rtn = 0;
        if (mode == MODE_LSTM){
            if (train == TRAINING){
                rtn = lstm_xw_train_get_workspace_size(input_size, hidden_size, 
                      time_step, batch_size, bidirectional, num_layer);
            } else if(train == INFERENCE){
                //rtn = lstm_xw_infer_get_workspace_size(input_size, hidden_size, time_step, batch_size);
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
        THFloatTensor * hy)
    {
        assert(L != 0 && D != 0 && T != 0 && N !=0 && I != 0 && H != 0);
        assert(ws != NULL);
        assert(x  != NULL);
        assert(hx != NULL);
        assert(wx != NULL);
        assert(wh != NULL);
        assert(bx != NULL);
        assert(bh != NULL);
        assert(y  != NULL);
        assert(hy != NULL);
        RNNForwardDesc fwd_desc = {L, D, T, N, I, H,
            getTHBuffer(ws),
            getTHBuffer(x),
            getTHBuffer(hx),
            NULL,
            getTHBuffer(wx),
            getTHBuffer(wh),
            getTHBuffer(bx),
            getTHBuffer(bh),
            getTHBuffer(y),
            getTHBuffer(hy),
            NULL,
            0};
        gru_xw_infer(fwd_desc);
        return 1;
    }

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
        THFloatTensor * hy)
    {
        assert(L != 0 && D != 0 && T != 0 && N !=0 && I != 0 && H != 0);
        assert(ws != NULL);
        assert(x  != NULL);
        assert(hx != NULL);
        assert(wx != NULL);
        assert(wh != NULL);
        assert(bx != NULL);
        assert(bh != NULL);
        assert(y  != NULL);
        assert(hy != NULL);
        RNNForwardDesc fwd_desc = {L, D, T, N, I, H,
            getTHBuffer(ws),
            getTHBuffer(x),
            getTHBuffer(hx),
            NULL,
            getTHBuffer(wx),
            getTHBuffer(wh),
            getTHBuffer(bx),
            getTHBuffer(bh),
            getTHBuffer(y),
            getTHBuffer(hy),
            NULL,
            0};
        gru_xw_forward(fwd_desc);
        return 1;
    }

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
        THFloatTensor * dhy)
    {
        assert(L != 0 && D != 0 && T != 0 && N !=0 && I != 0 && H != 0);
        assert(ws != NULL);
        assert(x  != NULL);
        assert(hx != NULL);
        assert(wx != NULL);
        assert(wh != NULL);
        assert(dx != NULL);
        assert(dhx != NULL);
        assert(dwx != NULL);
        assert(dwh != NULL);
        assert(dbx != NULL);
        assert(dbh != NULL);
        assert(dy != NULL);
        assert(dhy != NULL);

        RNNBackwardDesc bwd_desc = {L, D, T, N, I, H,
            getTHBuffer(ws),
            getTHBuffer(x),
            getTHBuffer(hx),
            NULL,
            getTHBuffer(wx),
            getTHBuffer(wh),
            NULL,
            NULL,
            NULL,
            getTHBuffer(dx),
            getTHBuffer(dhx),
            NULL,
            getTHBuffer(dwx),
            getTHBuffer(dwh),
            getTHBuffer(dbx),
            getTHBuffer(dbh),
            getTHBuffer(dy),
            getTHBuffer(dhy),
            NULL,
            0};
        gru_xw_backward(bwd_desc);
        return 1;
    }

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
        THFloatTensor * cy)
    {
        assert(L != 0 && D != 0 && T != 0 && N !=0 && I != 0 && H != 0);
        assert(ws != NULL);
        assert(x  != NULL);
        assert(hx != NULL);
        assert(cx != NULL);
        assert(wx != NULL);
        assert(wh != NULL);
        assert(bx != NULL);
        assert(y  != NULL);
        assert(hy != NULL);
        assert(cy != NULL);
        RNNForwardDesc fwd_desc = {L, D, T, N, I, H,
            getTHBuffer(ws),
            getTHBuffer(x),
            getTHBuffer(hx),
            getTHBuffer(cx),
            getTHBuffer(wx),
            getTHBuffer(wh),
            getTHBuffer(bx),
            NULL,
            getTHBuffer(y),
            getTHBuffer(hy),
            getTHBuffer(cy),
            0};
        lstm_xw_infer(fwd_desc);
        return 1;
    }

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
        THFloatTensor * cy)
    {

        assert(L != 0 && D != 0 && T != 0 && N !=0 && I != 0 && H != 0);
        assert(ws != NULL);
        assert(x  != NULL);
        assert(hx != NULL);
        assert(cx != NULL);
        assert(wx != NULL);
        assert(wh != NULL);
        assert(bx != NULL);
        assert(y  != NULL);
        assert(hy != NULL);
        assert(cy != NULL);
        RNNForwardDesc fwd_desc = {L, D, T, N, I, H,
            getTHBuffer(ws),
            getTHBuffer(x),
            getTHBuffer(hx),
            getTHBuffer(cx),
            getTHBuffer(wx),
            getTHBuffer(wh),
            getTHBuffer(bx),
            NULL,
            getTHBuffer(y),
            getTHBuffer(hy),
            getTHBuffer(cy),
            0};
        lstm_xw_forward(fwd_desc);
        return 1;
    }

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
        THFloatTensor * dcy)
    {
        assert(L != 0 && D != 0 && T != 0 && N !=0 && I != 0 && H != 0);
        assert(ws != NULL);
        assert(x  != NULL);
        assert(hx != NULL);
        assert(wx != NULL);
        assert(wh != NULL);
        assert(dx != NULL);
        assert(dhx != NULL);
        assert(dwx != NULL);
        assert(dwh != NULL);
        assert(dbx != NULL);
        assert(dy != NULL);
        assert(dhy != NULL);

        RNNBackwardDesc bwd_desc = {L, D, T, N, I, H,
            getTHBuffer(ws),
            getTHBuffer(x),
            getTHBuffer(hx),
            getTHBuffer(cx),
            getTHBuffer(wx),
            getTHBuffer(wh),
            getTHBuffer(y),
            getTHBuffer(hy),
            getTHBuffer(cy),
            getTHBuffer(dx),
            getTHBuffer(dhx),
            getTHBuffer(dcx),
            getTHBuffer(dwx),
            getTHBuffer(dwh),
            getTHBuffer(dbx),
            NULL,
            getTHBuffer(dy),
            getTHBuffer(dhy),
            getTHBuffer(dcy),
            0};
        lstm_xw_backward(bwd_desc);
        return 1;
    }

}




