#include <cstddef>
#include <iostream>
#include <rnn.h>
#include <mkl.h>
#include <omp.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

extern "C" {
    struct rnn_handle {
        const float** A;
        const float** B;
        float** A_pack;
        float** C;
        float* x_temp;
        float* gemmA;
        float* gemmB;
        float* gemmC;
        float* f_t;
        float* i_t;
        float* c_wave_t;
        float* o_t;
        //float* c_t;
    };

    //method: (wx) simplest realization
    void rnn_xw_sequential_infer(
        int batch_size, 
        int time_step, 
        int input_dim, 
        int hid, 
        float* w_x,   //I*H
        float* w_h,   //H*H
        float* b,     //H 
        float* x,     //T*N*I
        float* h_0,   //N*H
        float* h_out,   //UNCERTAIN
        bool return_sequences,
        float* gemmC){
        
        //printf("C      rnn_xw_sequential_infer called, time_step = %d\n", time_step);
  
        int i,j,p;
        int m = batch_size;
	int n = hid;
	int k = input_dim;
	float beta = 0.0;
        float* h_t=h_out;
        float* h_t1=h_out;
	memcpy(h_out, h_0, hid*batch_size * sizeof(float));	
        for (i = 0; i < time_step; i++) {
	    //Wx*x; Wx:[I,H] x:[T, N, I]
	    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1, x+i*k*m, k, w_x, n, beta, gemmC, n);
            //Wh*h+tmp; Wh:[H,H] h:[N,H]
	    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, hid, 1, h_t1, hid, w_h, n, 1, gemmC, n);
            // sigmoid for f,i,o, tanh for c_wave
            #pragma omp parallel for
            for (j = 0; j < hid*batch_size; j++) {
                if (b==NULL){
		    h_t[j] = tanh(gemmC[j]);
                }
                else{
 		    h_t[j] = tanh(gemmC[j] + b[j%hid]);
                }
            }
	    if(return_sequences){
                h_t1 = h_t;
                h_t = h_t + hid*batch_size;
            }
            else{
                h_t1 = h_t;
                h_t = h_out;
            }
        }
    }/*
    void rnn_xw_sequential_infer(
        int batch_size, 
        int time_step, 
        int input_dim, 
        int hid, 
        float* w_x,   //I*H
        float* w_h,   //H*H
        float* b,     //H 
        float* x,     //T*N*I
        float* h_0,   //N*H
        float* h_out,   //UNCERTAIN
        bool return_sequences,
        float* gemmC){
        
        //printf("C      rnn_xw_sequential_infer called, time_step = %d\n", time_step);
  
        int i,j,p;
        int m = batch_size;
	int n = hid;
	int k = input_dim;
	float beta = (b == NULL) ? 0.0 : 1.0;
        float* h_t=h_out;
        float* h_t1=h_out;
	memcpy(h_out, h_0, hid*batch_size * sizeof(float));	
        for (i = 0; i < time_step; i++) {
	    if (b) {
                #pragma omp parallel for collapse(2)
                for(p=0;p<hid;p++){
	            for (j = 0; j < batch_size; j++) {
		        gemmC[batch_size*j + p] = b[p];
		    }
		}
	    }

            //Wx*x; Wx:[I,H] x:[T, N, I]
	    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1, x+i*k*m, k, w_x, n, beta, gemmC, n);
            //Wh*h+tmp; Wh:[H,H] h:[N,H]
	    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, hid, 1, h_t1, hid, w_h, n, 1, gemmC, n);
            // sigmoid for f,i,o, tanh for c_wave
            #pragma omp parallel for
            for (j = 0; j < hid*batch_size; j++) {
		    h_t[j] = tanh(gemmC[j]);
            }
	    if(return_sequences){
                h_t1 = h_t;
                h_t = h_t + hid*batch_size;
            }
            else{
                h_t1 = h_t;
                h_t = h_out;
            }
        }
    }
*/
    void rnn_xw_pack_infer(
        int batch_size, 
        int time_step, 
        int input_dim, 
        int hid, 
        float* w_x,   //I8H
        float* w_h,   //H*H
        float* b,     //H 
        float* x,     //T*N*I
        float* h_0,   //N*H
        float* h_out,   //UNCERTAIN
        bool return_sequences,
        float* gemmC){
 
        //printf("C      RNN_xw_pack_infer called \n");
	int i, j, p;
	int m = batch_size;
	int n = hid;
	int k = input_dim;
	float beta = 0.0;
	float* h_t = h_out;
	float* h_t1 = h_out;
	memcpy(h_out, h_0, hid*batch_size * sizeof(float));
	float* w_x_pack = cblas_sgemm_alloc(CblasBMatrix, m, n, k);
	float* w_h_pack = cblas_sgemm_alloc(CblasBMatrix, m, n, hid);
	cblas_sgemm_pack(CblasRowMajor, CblasBMatrix, CblasNoTrans, m, n, k, 1.0, w_x, n, w_x_pack);
	cblas_sgemm_pack(CblasRowMajor, CblasBMatrix, CblasNoTrans, m, n, hid, 1.0, w_h, n, w_h_pack);
	for (i = 0; i < time_step; i++) {
            //Wx*x+b; Wx:[4H,D] x:[T, D, N]
            cblas_sgemm_compute(CblasRowMajor, CblasNoTrans, CblasPacked, m, n, k, x + i*k*m, k, w_x_pack, n, beta, gemmC, n);
            //Wh*h+tmp; Wh:[4H,H] h:[H,N]
            cblas_sgemm_compute(CblasRowMajor, CblasNoTrans, CblasPacked, m, n, hid, h_t1, hid, w_h_pack, n, 1, gemmC, n);
            // sigmoid for f,i,o, tanh for c_wave
            #pragma omp parallel for
            for (j = 0; j < hid*batch_size; j++) {
                if (b==NULL){
		    h_t[j] = tanh(gemmC[j]);
                }
                else{
 		    h_t[j] = tanh(gemmC[j] + b[j%hid]);
                }
            }
	    if(return_sequences){
                h_t1 = h_t;
                h_t = h_t + hid*batch_size;
            }
            else{
                h_t1 = h_t;
                h_t = h_out;
            }

        }
        cblas_sgemm_free(w_x_pack);
        cblas_sgemm_free(w_h_pack);
    }
    void rnn_xw_combine_infer(
        int batch_size, 
        int time_step, 
        int input_dim, 
        int hid, 
        float* w_x,   //4H*D
        float* w_h,   //4H*H
        float* b,     //4H 
        float* x,     //T*D*N
        float* h_0,   //H*N
        float* h_out,   //UNCERTAIN
        bool return_sequences,
        float* gemmA,
        float* gemmB,
        float* gemmC){
	
        //printf("C      rnn_xw_combine_infer called\n");
	int i, j, p;
	int n = hid;
	int m = batch_size;
	int k = input_dim+hid;
	float beta = 0.0;
	float* h_t = h_out;
	float* h_t1 = h_out;
	memcpy(h_out, h_0, hid*batch_size * sizeof(float));
        memcpy(gemmB, w_x, input_dim * hid * sizeof(float));
	memcpy(gemmB + input_dim * hid, w_h, hid * hid * sizeof(float));
        for (i = 0; i < time_step; i++) {
            for( j = 0; j < batch_size; j++){
                memcpy(gemmA+j*(input_dim+hid), x+i*input_dim*batch_size+j*input_dim, input_dim*sizeof(float));
                memcpy(gemmA+j*(input_dim+hid)+input_dim, h_t1+j*hid, hid*sizeof(float));
            }
            //W*[x;h]+b;  W:[4H,D+H]  [x,h]:[D+H,N]
	    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1.0, gemmA, k, gemmB, n, beta, gemmC, n);
            #pragma omp parallel for
            for (j = 0; j < hid*batch_size; j++) {
                if (b==NULL){
		    h_t[j] = tanh(gemmC[j]);
                }
                else{
 		    h_t[j] = tanh(gemmC[j] + b[j%hid]);
                }
            }
	    if(return_sequences){
                h_t1 = h_t;
                h_t = h_t + hid*batch_size;
            }
            else{
                h_t1 = h_t;
                h_t = h_out;
            }
 	}
    }
    void rnn_xw_combine_pack_infer(
        int batch_size, 
        int time_step, 
        int input_dim, 
        int hid, 
        float* w_x,   //4H*D
        float* w_h,   //4H*H
        float* b,     //4H 
        float* x,     //T*D*N
        float* h_0,   //H*N
        float* h_out,   //UNCERTAIN
        bool return_sequences,
        float* gemmA,
        float* gemmB,
        float* gemmC){
 
	//printf("C      rnn_xw_combine_pack_infer called \n");
	int i, j, p;
	int n = hid;
	int m = batch_size;
	int k = input_dim+hid;
	float beta = 0.0;
	float* h_t = h_out;
	float* h_t1 = h_out;
	memcpy(h_out, h_0, hid*batch_size * sizeof(float));
        memcpy(gemmB, w_x, input_dim * hid * sizeof(float));
	memcpy(gemmB + input_dim * hid, w_h, hid * hid * sizeof(float));
        float* pack = cblas_sgemm_alloc(CblasBMatrix, m, n, k);
	cblas_sgemm_pack(CblasRowMajor, CblasBMatrix, CblasNoTrans, m, n, k, 1.0, gemmB, n, pack);
	for (i = 0; i < time_step; i++) {
            for( j = 0; j < batch_size; j++){
                memcpy(gemmA+j*(input_dim+hid), x+i*input_dim*batch_size+j*input_dim, input_dim*sizeof(float));
                memcpy(gemmA+j*(input_dim+hid)+input_dim, h_t1+j*hid, hid*sizeof(float));
            }
	    cblas_sgemm_compute(CblasRowMajor, CblasNoTrans, CblasPacked, m, n, k, gemmA, k, pack, n, beta, gemmC, n);
            // sigmoid for f,i,o, tanh for c_wave
            #pragma omp parallel for
            for (j = 0; j < hid*batch_size; j++) {
                if (b==NULL){
		    h_t[j] = tanh(gemmC[j]);
                }
                else{
 		    h_t[j] = tanh(gemmC[j] + b[j%hid]);
                }
            }
	    if(return_sequences){
                h_t1 = h_t;
                h_t = h_t + hid*batch_size;
            }
            else{
                h_t1 = h_t;
                h_t = h_out;
            }
        }
        cblas_sgemm_free(pack);
    }

    //method: wx_batch_gemm
    int rnn_xw_batch_infer(
        int batch_size, 
        int time_step, 
        int input_dim, 
        int hid, 
        float* w_x, 
        float* w_h, 
        float* b, 
        float* x,
        float* h_0,
        float* h_out,
        bool return_sequences,
        //float* c_t,
        const float** A,
        const float** B, 
        float**  C,
        float* x_temp) {

        //printf("C      rnn_xw_batch_infer called \n");
        int i,j,p;
        // w_x * x
        MKL_INT m[1]; 
        MKL_INT n[1]; 
        MKL_INT k[1]; 
        
        MKL_INT lda[1]; 
        MKL_INT ldb[1]; 
        MKL_INT ldc[1]; 
        
        CBLAS_TRANSPOSE transA[1]; 
        CBLAS_TRANSPOSE transB[1]; 
        
        float alpha[1]; 
        float beta[1]; 
        MKL_INT size_per_grp[1]; 
    
        n[0] = hid;
        k[0] = input_dim;
        m[0] = batch_size;
        
        lda[0] = k[0]; 
        ldb[0] = n[0]; 
        ldc[0] = n[0]; 
        
        transB[0] = CblasNoTrans; 
        transA[0] = CblasNoTrans; 
        alpha[0] = 1.0; 
        beta[0] = 0.0;
    
        size_per_grp[0] = time_step;
    
        if (NULL == A || NULL == B || NULL == C || NULL == x_temp) {
            //printf( "\n ERROR: malloc global buffers failed \n\n");
            return -3;
        }
        #pragma omp parallel for 
        for (i = 0; i < time_step; i++) { 
            B[i] = w_x;                                       // w_fx
            A[i] = x + i * k[0] * m[0]; 
            C[i] = x_temp + i * m[0] * n[0]; 
        } 
        cblas_sgemm_batch(CblasRowMajor, transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, 1, size_per_grp); 
    
        // loop on step
        n[0] = hid;
        k[0] = hid;
        m[0] = batch_size;
        
        beta[0] = 1.0;
    
        lda[0] = k[0]; 
        ldb[0] = n[0]; 
        ldc[0] = n[0]; 
        size_per_grp[0] = 1;
        
        B[0] = w_h;                //w_fh     
        A[0] = h_0;
    
        size_t mn = m[0] * n[0];
    
        for (i = 0; i < time_step; i++) {
            // f,i,c_wave,o
            C[0] = x_temp + i * m[0] * n[0];
    
            cblas_sgemm_batch(CblasRowMajor, transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, 1, size_per_grp);
    
            //h
            float* y_ptr = NULL;
            if (return_sequences) {
                y_ptr = h_out + i * batch_size * hid;
            } else {
                y_ptr = h_out;
            }
            #pragma omp parallel for
            for (j = 0; j < mn; j++) {
                if (b==NULL){
                    y_ptr[j]  = tanh((float)(C[0][j]));
                }
                else{
                    y_ptr[j]  = tanh((float)(C[0][j])+b[j%hid]);
                }
            }
            // update
            A[0] = y_ptr;
        }
        return 0;
    }

    //method: wx_batch_gemm_h_pack_gemm
    int rnn_xw_h_pack_infer(
        int batch_size, 
        int time_step, 
        int input_dim, 
        int hid, 
        float* w_x, 
        float* w_h, 
        float* b, 
        float* x,
        float* h_0,
        float* h_out,
        bool return_sequences,
        const float** A,
        const float** B, 
        float**  C, 
        float** B_pack, 
        float* x_temp){
    
        //printf("C      wx_batch_gemm_h_pack_gemm \n");
        int i,j,p;
        // w_x * x
        MKL_INT m[1]; 
        MKL_INT n[1]; 
        MKL_INT k[1]; 
        
        MKL_INT lda[1]; 
        MKL_INT ldb[1]; 
        MKL_INT ldc[1]; 
        
        CBLAS_TRANSPOSE transA[1]; 
        CBLAS_TRANSPOSE transB[1]; 
        
        float alpha[1]; 
        float beta[1]; 
        MKL_INT size_per_grp[1]; 
    
        n[0] = hid;
        k[0] = input_dim;
        m[0] = batch_size;
        
        lda[0] = k[0]; 
        ldb[0] = n[0]; 
        ldc[0] = n[0]; 
        
        transB[0] = CblasNoTrans; 
        transA[0] = CblasNoTrans; 
        
        alpha[0] = 1.0; 
        beta[0] = 0.0;
    
        size_per_grp[0] = time_step;
    
        if (NULL == A || NULL == B || NULL == C || NULL == x_temp) {
            //printf( "\n ERROR: malloc global buffers failed \n\n");
            return -3;
        }
        #pragma omp parallel for 
        for (i = 0; i < time_step; i++){ 
            B[i] = w_x;
            A[i] = x + i * k[0] * m[0]; 
            C[i] = x_temp + i * m[0] * n[0]; 
        } 
        cblas_sgemm_batch(CblasRowMajor, transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, 1, size_per_grp); 
        // loop on step
        n[0] = hid;
        k[0] = hid;
        m[0] = batch_size;
        
        beta[0] = 1.0;
        B_pack[0] = cblas_sgemm_alloc(CblasAMatrix, m[0], n[0], k[0]);//w_fh
        cblas_sgemm_pack(CblasRowMajor, CblasBMatrix, CblasNoTrans, m[0], n[0], k[0], alpha[0], w_h, k[0], B_pack[0]);
        A[0] = h_0; 
    
        size_t mn = m[0] * n[0];
    
        for (i = 0; i < time_step; i++) {
            // f,i,c_wave,o
            C[0] = x_temp + i * m[0] * n[0];
   
            cblas_sgemm_compute(CblasRowMajor, CblasNoTrans, CblasPacked, m[0], n[0], k[0], A[0], k[0], B_pack[0], n[0], beta[0], C[0], n[0]);
            //h
            float* y_ptr = NULL;
            if (return_sequences) {
                y_ptr = h_out + i * batch_size * hid;
            } else {
                y_ptr = h_out;
            }
            #pragma omp parallel for
            for (j = 0; j < mn; j++) {
                if (b==NULL){
                    y_ptr[j]  = tanh((float)(C[0][j]));
                }
                else{
                    y_ptr[j]  = tanh((float)(C[0][j])+b[j%hid]);
                }
            }
            // update
            A[0] = y_ptr;
        }
        cblas_sgemm_free(B_pack[0]);
        return 0;
    }

    int rnn_create_instance(void* buf, rnn_handle* rnn_han, int input_dim, int hid, int max_time_step, int max_batch_size) { 
        if (max_time_step == 0) {
            max_time_step = 128;
        }
        if (max_batch_size == 0) {
            max_batch_size = 64;
        }
        rnn_han->A = (const float**) buf;
        rnn_han->B = (const float**) (buf + max_time_step * sizeof (float*));
        rnn_han->A_pack = (float**) (buf + 2 * max_time_step * sizeof (float*));
        rnn_han->C = (float**) (buf + 3 * max_time_step * sizeof (float*));
        rnn_han->x_temp = (float*) (buf + 4 * max_time_step * sizeof (float*));
        rnn_han->gemmB = (float*) (buf + 4 * max_time_step * sizeof (float*) + max_time_step *  max_batch_size * hid * sizeof (float));
        rnn_han->gemmC = (float*) (buf + 4 * max_time_step * sizeof (float*) +
                         (max_time_step *  max_batch_size * hid + hid * (input_dim + hid)) * sizeof (float)); 
        rnn_han->gemmA = (float*) (buf + 4 * max_time_step * sizeof (float*) +
                       (max_time_step * max_batch_size * hid + hid * (input_dim + hid) + max_batch_size * hid) * sizeof (float));
        return 0;
    }

    // rnn c interface
    // I: input dimension
    // H: hidden size
    // N: batch size
    // T: time step
    int rnn_xw_infer(
        void* buf, 
        int batch_size, 
        int time_step, 
        int input_dim, 
        int hid, 
        int max_time_step,
        int max_batch_size,
        float* w_x,     //(4H, I) 
        float* w_h,     //(4H, H) 
        float* b,       //(4H)
        float* x,       //(T, I, N)
        float* h_0,     //(H, N)
        float* h_out,   //if return_sequences == true, size = (T, H, N), else size = (H, N)
        bool return_sequences,
        int mode){

        //printf("\n C     batch_size = %d \n", batch_size);
        rnn_handle* rnn_han = (rnn_handle*)mkl_malloc(sizeof(rnn_handle), 64);
        rnn_create_instance(buf, rnn_han, input_dim, hid, max_time_step, max_batch_size);
        const float** A = rnn_han->A;
        const float** B = rnn_han->B;
        float** A_pack = rnn_han->A_pack;
        float** C = rnn_han->C;
        float* x_temp = rnn_han->x_temp;
        float* gemmA = rnn_han->gemmA;
        float* gemmB = rnn_han->gemmB;
        float* gemmC = rnn_han->gemmC;
        //float* f_t = rnn_han->f_t;
        //float* i_t = rnn_han->i_t;
        //float* c_wave_t = rnn_han->c_wave_t;
        //float* o_t = rnn_han->o_t;
        //printf("mode = %d\n",mode);
        switch(mode) {
            case 0:
                rnn_xw_sequential_infer(batch_size, time_step, input_dim, hid, w_x, w_h, b, x, h_0, h_out, return_sequences,gemmC);       
                break;
            case 1:
                rnn_xw_pack_infer(batch_size, time_step, input_dim, hid, w_x, w_h, b, x, h_0, h_out,return_sequences,gemmC);
                break;
            case 2:
                rnn_xw_combine_infer(batch_size, time_step, input_dim, hid, w_x, w_h, b, x, h_0, h_out, return_sequences, gemmA, gemmB, gemmC);       
                break;
            case 3:
                rnn_xw_combine_pack_infer(batch_size, time_step, input_dim, hid, w_x, w_h, b, x, h_0, h_out, return_sequences, gemmA,gemmB,gemmC);
                break;
            case 4:
                rnn_xw_batch_infer(batch_size, time_step, input_dim, hid, w_x, w_h, b, x, h_0, h_out, return_sequences,A, B, C, x_temp);       
                break;
            case 5:
                rnn_xw_h_pack_infer(batch_size, time_step, input_dim, hid, w_x, w_h, b, x, h_0, h_out, return_sequences, A, B, C, A_pack, x_temp);       
                break;
            default: 
                rnn_xw_sequential_infer(batch_size, time_step, input_dim, hid, w_x, w_h, b, x, h_0, h_out, return_sequences,gemmC);       
       }
       mkl_free(rnn_han);
       return 0;
    }
    
    int rnn_xw_infer_get_workspace_size(int input_dim, int hid, int max_time_step, int max_batch_size){
        int max_ws = 16 * max_time_step * sizeof (float*) + (max_time_step * 4 * max_batch_size * hid 
                     + max_batch_size * (input_dim + hid) + 8 * max_batch_size * hid + 4 * hid * (input_dim + hid)) * sizeof (float);
        return max_ws;
    }
}
