unsigned long sru_get_size(int batch_size, int hidden_dim, int time_step);
int sru_inference(void* buf,
                  int batch_size,     //N
                  int time_step,      //T
                  int input_dim,      //I
                  int hidden_dim,     //H
                  float* w_x,         //H*I
                  float* w_f,         //H*I
                  float* w_r,         //H*I
                  float* w_tmp,       //H*I, if hidden_dim not equal to input_dim, add one more linear transform: w_tmp * x_t
                  float* b_f,         //H*N
                  float* b_r,         //H*N
                  float* c_0,         //H*N
                  float* x_t,         //T*I*N
                  float* h_out,       //if return_sequences == true, size = T*H*N, else size = H*N
                  bool return_sequences,
                  int mode);
