/*  a benchmark for all the supported cell: LSTM/GRU
    this benchmark support D=1/D=2
    ./benchmark train/infer lstm/gru ud/bd
*/

#include <iostream>                                                             
#include <sys/time.h>                                                           
#include <rnn.h>                                                                
#include <mkl.h>                                                                

#define warmup 20                                                                               
#define count 100
#define align 64

void buffer_init(float * buf, int size){
    int i = 0;
    for(int i = 0; i < size; ++i){
        buf[i] = ((float)rand()/(float)RAND_MAX) - ((float)rand()/(float)RAND_MAX);
    }
}

int test_main(int is_train, std::string cell_type, int L, int D, int T, int N, int I, int H){

    srand(45678);
    int (*call_forward)(RNNForwardDesc desc) = NULL;
    int (*call_backward)(RNNBackwardDesc desc) = NULL;

    int ws_size = 0;
    int Gate = (cell_type == "gru") ? 3 : 4;
    if(cell_type == "gru"){
        if(is_train){
            ws_size = gru_xw_train_get_workspace_size(L, D, T, N, I, H);
            call_forward = gru_xw_forward;
            call_backward = gru_xw_backward;
        }else{
            ws_size = gru_xw_infer_get_workspace_size(L, D, T, N, I, H);
            call_forward = gru_xw_infer;
        }
    }else if(cell_type == "lstm"){
        if(is_train){
            ws_size = lstm_xw_train_get_workspace_size(L, D, T, N, I, H);
            call_forward = lstm_xw_forward;
            call_backward = lstm_xw_backward;
        }else{
            ws_size = lstm_xw_infer_get_workspace_size(L, D, T, N, I, H);
            call_forward = lstm_xw_infer;
        }
    }
    float* ws = (float*)mkl_malloc(ws_size * sizeof(float), align);

    // malloc buffer for input: x, hx, cx
    float* x  = (float*)mkl_malloc(T * N * I * sizeof(float), align);
    float* hx = (float*)mkl_malloc(L * D * N * H * sizeof(float), align);
    float* cx = (float*)mkl_malloc(L * D * N * H * sizeof(float), align);

    // malloc buffer for weight/bias: ws, wh, bx, bh
    float* wx = (float*)mkl_malloc(L * D * I * Gate * H * sizeof(float), align);
    float* wh = (float*)mkl_malloc(L * D * H * Gate * H * sizeof(float), align);
    float* bx = (float*)mkl_malloc(L * D * Gate * H * sizeof(float), align);
    float* bh = (float*)mkl_malloc(L * D * Gate * H * sizeof(float), align);

    // malloc buffer for input: y, hy, cy
    float* y  = (float*)mkl_malloc(T * N * D * H * sizeof(float), align);
    float* hy = (float*)mkl_malloc(L * D * N * H * sizeof(float), align);
    float* cy = (float*)mkl_malloc(L * D * N * H * sizeof(float), align);

    //buffer init
    buffer_init(x, T * N * I);
    buffer_init(hx, L * D * N * H);
    buffer_init(cx, L * D * N * H);
    buffer_init(wx, L * D * I * Gate * H);
    buffer_init(wh, L * D * H * Gate * H);
    buffer_init(bx, L * D * Gate * H);
    buffer_init(bh, L * D * Gate * H);

    int algo = 0;
    RNNForwardDesc desc_fwd = {L, D, T, N, I, H, ws, x, hx, cx, wx, wh, bx, bh, y, hy, cy, algo};
    RNNBackwardDesc desc_bwd;
    float *dx, *dhx, *dcx, *dwx, *dwh, *dbx, *dbh, *dy, *dhy, *dcy;

    if(is_train){
        dx  = (float*)mkl_malloc(T * N * I * sizeof(float), align);
        dhx = (float*)mkl_malloc(L * D * N * H * sizeof(float), align);
        dcx = (float*)mkl_malloc(L * D * N * H * sizeof(float), align);

        dwx = (float*)mkl_malloc(L * D * I * Gate * H * sizeof(float), align);
        dwh = (float*)mkl_malloc(L * D * H * Gate * H * sizeof(float), align);
        dbx = (float*)mkl_malloc(L * D * Gate * H * sizeof(float), align);
        dbh = (float*)mkl_malloc(L * D * Gate * H * sizeof(float), align);

        dy  = (float*)mkl_malloc(T * N * D * H * sizeof(float), align);
        dhy = (float*)mkl_malloc(L * D * N * H * sizeof(float), align);
        dcy = (float*)mkl_malloc(L * D * N * H * sizeof(float), align);

        //buffer init
        buffer_init(dy,  T * N * D * H);
        buffer_init(dhy, L * D * N * H);
        buffer_init(dcy, L * D * N * H);

        desc_bwd = {L, D, T, N, I, H, ws, x, hx, cx, wx, wh, y, hy, cy,
            dx, dhx, dcx, dwx, dwh, dbx, dbh, dy, dhy, dcy, algo};
    }
    
    int i = 0;
    double start, end;
    for(i = 0; i < warmup + count; ++i){
        if(i == warmup){
            start = dsecnd();
        }
        call_forward(desc_fwd);
        if(is_train){
            call_backward(desc_bwd);
        }

    }
    end = dsecnd();
    double dura = end - start;
    float SPS = N * count / dura;
    printf("L = %d, D = %d, N = %d, T = %d, I = %d, H = %d, SPS = %.4f\n", L, D, N, T, I, H, SPS);

    // free memory
    mkl_free(ws);
    mkl_free(x);
    mkl_free(hx);
    mkl_free(cx);
    mkl_free(wx);
    mkl_free(wh);
    mkl_free(bx);
    mkl_free(bh);
    mkl_free(y);
    mkl_free(hy);
    mkl_free(cy);
    if(is_train){
        mkl_free(dx);
        mkl_free(dhx);
        mkl_free(dcx);
        mkl_free(dwx);
        mkl_free(dwh);
        mkl_free(dbx);
        mkl_free(dbh);
        mkl_free(dy);
        mkl_free(dhy);
        mkl_free(dcy);
    }
}


int main(int argc, char ** argv)                                               
{

    std::string type = "train";     // default train
    std::string cell_type = "lstm"; // default lstm
    std::string direction = "ud";   // default D=1

    if (argc > 1){
        std::string argv1 = argv[1];
        std::string argv2 = argv[2];
        std::string argv3 = argv[3];
        type        = (argv1 == "infer") ? "infer" : "train";
        cell_type   = (argv2 == "gru")   ? "gru"   : "lstm";
        direction   = (argv3 == "bd")    ? "bd"    : "ud";
    }
    std::cout << "Type = " << type << ", CellType = "<< cell_type <<
        ", direction = " << direction << std::endl;

    int is_train = (type == "train") ? 1 : 0;
    int D = (direction == "bd") ? 2 : 1;

    /* N, T, I ,H*/
    int size[18][4] = {                                                        
         {20,1,800,800},                                                        
         {20,50,800,800},                                                       
         {20,100,800,800},                                                  
         {20,200,800,800},
         {20,300,800,800},
         {20,400,800,800},
         {12,1,1760,1760},
         {12,50,1760,1760},
         {12,100,1760,1760},
         {12,200,1760,1760},
         {12,300,1760,1760},
         {12,400,1760,1760},
         {32,1,1760,1760},
         {32,50,1760,1760},
         {32,100,1760,1760},                                                 
         {32,200,1760,1760},
         {32,300,1760,1760},
         {32,400,1760,1760}
    };      

    int L = 1;
    int i = 0;
    for(i = 0; i < 18; i++){
        test_main(is_train, cell_type, L, D, size[i][1], size[i][0], size[i][2], size[i][3]);
    }


    return 1;

}
