#include <cstddef>
#include <iostream>
#include <rnn.h>
#include <mkl.h>
#include <omp.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

extern "C" {
    struct lstm_handle {
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
    void lstm_wx_total_sequential_infer(
        int batch_size,
        int time_step,
        int input_dim,
        int hid,
        float* w_x,   //4H*D
        float* w_h,   //4H*H
        float* b,     //4H
        float* x,     //T*D*N
        float* h_0,   //H*N
        float* c_0,   //H*N
        float* h_out,   //UNCERTAIN
        float* c_out,
        bool return_sequences,
        float* f_t,
        float* i_t,
        float* c_wave_t,
        float* o_t,
        float* gemmC){

        //printf("C      lstm_wx_sequential_infer called, batch_size = %d\n", batch_size);

        int i,j,p;
        int m = hid;
        int n = batch_size;
        int k = input_dim;
        float beta = 0.0;
        float* h_t=h_out;
        float* h_t1=h_out;
        float* c_t=c_out;
        memcpy(h_out, h_0, hid*batch_size * sizeof(float));
        memcpy(c_out, c_0, hid*batch_size * sizeof(float));
        for (i = 0; i < time_step; i++) {
            /*
            if(b){
                #pragma omp parallel for collapse(2)
                for(p=0;p<4*hid;p++){
	            for (j = 0; j < batch_size; j++) {
                        gemmC[batch_size*p+j]=b[p];
                    }
	        }
            }
            */
           //Wx*x+b; Wx:[4H,D] x:[T, D, N]
           cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1, w_x, k,x+i*k*n, n, beta, gemmC, n);
           cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1, w_x+hid*input_dim, k,x+i*k*n, n, beta, gemmC+hid*batch_size, n);
           cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1, w_x+2*hid*input_dim, k,x+i*k*n, n, beta, gemmC+2*hid*batch_size, n);
           cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1, w_x+3*hid*input_dim, k,x+i*k*n, n, beta, gemmC+3*hid*batch_size, n);
           //Wh*h+tmp; Wh:[4H,H] h:[H,N]
           cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, hid, 1, w_h, hid, h_t1, n, 1, gemmC, n);
           cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1, w_x, k,x+i*k*n, n, beta, gemmC, n);
           cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1, w_x, k,x+i*k*n, n, beta, gemmC, n);
           cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1, w_x, k,x+i*k*n, n, beta, gemmC, n);
           // sigmoid for f,i,o, tanh for c_wave
           #pragma omp parallel for
           for (j = 0; j < hid*batch_size; j++) {
               if (b==NULL){
		           i_t[j] = 1 / (1 + exp(-gemmC[j]));
		           f_t[j] = 1 / (1 + exp(-gemmC[j+hid*batch_size]));
		           c_wave_t[j] = tanh(gemmC[j+2*hid*batch_size]);
		           o_t[j] = 1 / (1 + exp(-gemmC[j+3*hid*batch_size]));
               }
               else{
 		           i_t[j] = 1 / (1 + exp(-gemmC[j])) + b[j/batch_size];
		           f_t[j] = 1 / (1 + exp(-gemmC[j+hid*batch_size])) + b[j/batch_size + hid];
		           c_wave_t[j] = tanh(gemmC[j+2*hid*batch_size]) + b[j/batch_size + 2*hid];
		           o_t[j] = 1 / (1 + exp(-gemmC[j+3*hid*batch_size])) + b[j/batch_size + 3*hid];
               }
               c_t[j] = f_t[j] * c_t[j] + i_t[j] * c_wave_t[j];
		       h_t[j] = o_t[j] * tanh(c_t[j]);
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
    //method: (wx) simplest realization
    void lstm_wx_sequential_infer(
        int batch_size, 
        int time_step, 
        int input_dim, 
        int hid, 
        float* w_x,   //4H*D
        float* w_h,   //4H*H
        float* b,     //4H 
        float* x,     //T*D*N
        float* h_0,   //H*N
        float* c_0,   //H*N
        float* h_out,   //UNCERTAIN
        float* c_out,
        bool return_sequences,
        float* f_t,
        float* i_t,
        float* c_wave_t,
        float* o_t,
        float* gemmC){
        
        //printf("C      lstm_wx_sequential_infer called, batch_size = %d\n", batch_size);
  
        int i,j,p;
        int m = hid*4;
        int n = batch_size;
        int k = input_dim;
        float beta = 0.0;
        float* h_t=h_out;
        float* h_t1=h_out;
        float* c_t=c_out;
        memcpy(h_out, h_0, hid*batch_size * sizeof(float));	
        memcpy(c_out, c_0, hid*batch_size * sizeof(float));
        for (i = 0; i < time_step; i++) {
            /*
            if(b){
                #pragma omp parallel for collapse(2)
                for(p=0;p<4*hid;p++){
	            for (j = 0; j < batch_size; j++) {
                        gemmC[batch_size*p+j]=b[p];
                    }
	        }
            }
            */
           //Wx*x+b; Wx:[4H,D] x:[T, D, N]
           cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1, w_x, k,x+i*k*n, n, beta, gemmC, n);
           //Wh*h+tmp; Wh:[4H,H] h:[H,N]
           cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, hid, 1, w_h, hid, h_t1, n, 1, gemmC, n);
           // sigmoid for f,i,o, tanh for c_wave
           #pragma omp parallel for
           for (j = 0; j < hid*batch_size; j++) {
               if (b==NULL){
		           i_t[j] = 1 / (1 + exp(-gemmC[j]));
		           f_t[j] = 1 / (1 + exp(-gemmC[j+hid*batch_size]));
		           c_wave_t[j] = tanh(gemmC[j+2*hid*batch_size]);
		           o_t[j] = 1 / (1 + exp(-gemmC[j+3*hid*batch_size]));
               }
               else{
 		           i_t[j] = 1 / (1 + exp(-gemmC[j])) + b[j/batch_size];
		           f_t[j] = 1 / (1 + exp(-gemmC[j+hid*batch_size])) + b[j/batch_size + hid];
		           c_wave_t[j] = tanh(gemmC[j+2*hid*batch_size]) + b[j/batch_size + 2*hid];
		           o_t[j] = 1 / (1 + exp(-gemmC[j+3*hid*batch_size])) + b[j/batch_size + 3*hid];
               }
               c_t[j] = f_t[j] * c_t[j] + i_t[j] * c_wave_t[j];
		       h_t[j] = o_t[j] * tanh(c_t[j]);                
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
    void lstm_wx_pack_infer(
        int batch_size, 
        int time_step, 
        int input_dim, 
        int hid, 
        float* w_x,   //4H*D
        float* w_h,   //4H*H
        float* b,     //4H 
        float* x,     //T*D*N
        float* h_0,   //H*N
        float* c_0,   //H*N
        float* h_out,   //UNCERTAIN
        float* c_out,
        bool return_sequences,
        float* f_t,
        float* i_t,
        float* c_wave_t,
        float* o_t,
        float* gemmC){
 
        //printf("C      lstm_wx_pack_infer called \n");
        int i, j, p;
        int m = hid * 4;
        int n = batch_size;
        int k = input_dim;
        float beta = (b == NULL) ? 0.0 : 1.0;
        float* h_t = h_out;
        float* h_t1 = h_out;
        float* c_t = c_out;
        memcpy(h_out, h_0, hid*batch_size * sizeof(float));
        memcpy(c_out, c_0, hid*batch_size * sizeof(float));
        float* w_x_pack = cblas_sgemm_alloc(CblasAMatrix, m, n, k);
        float* w_h_pack = cblas_sgemm_alloc(CblasAMatrix, m, n, hid);
        cblas_sgemm_pack(CblasRowMajor, CblasAMatrix, CblasNoTrans, m, n, k, 1.0, w_x, k, w_x_pack);
        cblas_sgemm_pack(CblasRowMajor, CblasAMatrix, CblasNoTrans, m, n, hid, 1.0, w_h, hid, w_h_pack);
        for (i = 0; i < time_step; i++) {
            if (b) {
                #pragma omp parallel for collapse(2)
                for(p=0;p<4*hid;p++){
	            for (j = 0; j < batch_size; j++) {
		            gemmC[batch_size*p + j] = b[p];
		        }
		        }
	        }
            //Wx*x+b; Wx:[4H,D] x:[T, D, N]
            cblas_sgemm_compute(CblasRowMajor, CblasPacked, CblasNoTrans, m, n, k, w_x_pack, k, x + i*k*n, n, beta, gemmC, n);
            //Wh*h+tmp; Wh:[4H,H] h:[H,N]
            cblas_sgemm_compute(CblasRowMajor, CblasPacked, CblasNoTrans, m, n, hid, w_h_pack, hid, h_t1, n, 1, gemmC, n);
            // sigmoid for f,i,o, tanh for c_wave
            #pragma omp parallel for
            for (j = 0; j < hid*batch_size; j++) {
                i_t[j] = 1 / (1 + exp(-gemmC[j]));
                f_t[j] = 1 / (1 + exp(-gemmC[j+hid*batch_size]));
                c_wave_t[j] = tanh(gemmC[j+2*hid*batch_size]);
                o_t[j] = 1 / (1 + exp(-gemmC[j+3*hid*batch_size]));

                c_t[j] = f_t[j] * c_t[j] + i_t[j] * c_wave_t[j];
                h_t[j] = o_t[j] * tanh(c_t[j]);                
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
    void lstm_wx_combine_infer(
        int batch_size, 
        int time_step, 
        int input_dim, 
        int hid, 
        float* w_x,   //4H*D
        float* w_h,   //4H*H
        float* b,     //4H 
        float* x,     //T*D*N
        float* h_0,   //H*N
        float* c_0,   //H*N
        float* h_out,   //UNCERTAIN
        float* c_out,
        bool return_sequences,
        float* f_t,
        float* i_t,
        float* c_wave_t,
        float* o_t,
        float* gemmA,
        float* gemmB,
        float* gemmC){
	
        //printf("C      lstm_wx_combine_infer called\n");
        int i, j, p;
        int m = hid * 4;
        int n = batch_size;
        int k = input_dim+hid;
        float beta = (b == NULL) ? 0.0 : 1.0;
        float* h_t = h_out;
        float* h_t1 = h_out;
        float* c_t = c_out;
        memcpy(h_out, h_0, hid*batch_size * sizeof(float));
        memcpy(c_out, c_0, hid*batch_size * sizeof(float));
        for( i = 0; i < 4 * hid; i++){
            memcpy(gemmA+i*(input_dim+hid), w_x+i*input_dim, input_dim*sizeof(float));
            memcpy(gemmA+i*(input_dim+hid)+input_dim, w_h+i*hid, hid*sizeof(float));
        }
        for (i = 0; i < time_step; i++) {
            if (b) {
                #pragma omp parallel for collapse(2)
                for(p=0;p<4*hid;p++){
	            for (j = 0; j < batch_size; j++) {
                    gemmC[batch_size*p + j] = b[p];
                }
		        }
            }
            memcpy(gemmB, x + i*input_dim*batch_size, input_dim*batch_size * sizeof(float));
            memcpy(gemmB+input_dim*batch_size, h_t1, hid*batch_size * sizeof(float));
            //W*[x;h]+b;  W:[4H,D+H]  [x,h]:[D+H,N]
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1, gemmA, k, gemmB, n, beta, gemmC, n);
            // sigmoid for f,i,o, tanh for c_wave
            #pragma omp parallel for
            for (j = 0; j < hid*batch_size; j++) {
                i_t[j] = 1 / (1 + exp(-gemmC[j]));
                f_t[j] = 1 / (1 + exp(-gemmC[j + hid*batch_size]));
                c_wave_t[j] = tanh(gemmC[j + 2 * hid*batch_size]);
                o_t[j] = 1 / (1 + exp(-gemmC[j + 3 * hid*batch_size]));
                c_t[j] = f_t[j] * c_t[j] + i_t[j] * c_wave_t[j];
		        h_t[j] = o_t[j] * tanh(c_t[j]);                
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
    void lstm_wx_combine_pack_infer(
        int batch_size, 
        int time_step, 
        int input_dim, 
        int hid, 
        float* w_x,   //4H*D
        float* w_h,   //4H*H
        float* b,     //4H 
        float* x,     //T*D*N
        float* h_0,   //H*N
        float* c_0,   //H*N
        float* h_out,   //UNCERTAIN
        float* c_out,
        bool return_sequences,
        float* f_t,
        float* i_t,
        float* c_wave_t,
        float* o_t,
        float* gemmA,
        float* gemmB,
        float* gemmC){
 
	//printf("C      lstm_wx_combine_pack_infer called \n");
        int i, j, p;
        int m = hid * 4;
        int n = batch_size;
        int k = input_dim+hid;
        float beta = (b == NULL) ? 0.0 : 1.0;
        float* h_t = h_out;
        float* h_t1 = h_out;
        float* c_t = c_out;
        memcpy(h_out, h_0, hid*batch_size * sizeof(float));
        memcpy(c_out, c_0, hid*batch_size * sizeof(float));
        for( i = 0; i < 4 * hid; i++){
            memcpy(gemmA+i*(input_dim+hid), w_x+i*input_dim, input_dim*sizeof(float));
            memcpy(gemmA+i*(input_dim+hid)+input_dim, w_h+i*hid, hid*sizeof(float));
        }
        float* w_pack = cblas_sgemm_alloc(CblasAMatrix, m, n, k);
        cblas_sgemm_pack(CblasRowMajor, CblasAMatrix, CblasNoTrans, m, n, k, 1.0, gemmA, k, w_pack);
        for (i = 0; i < time_step; i++) {
            if (b) {
                #pragma omp parallel for collapse(2)
                for(p=0;p<4*hid;p++){
	            for (j = 0; j < batch_size; j++) {
                    gemmC[batch_size*p + j] = b[p];
                }
	  	        }
	        }
	        memcpy(gemmB, x + i*input_dim*batch_size, input_dim*batch_size * sizeof(float));
	        memcpy(gemmB+input_dim*batch_size, h_t1, hid*batch_size * sizeof(float));
            //W*[x;h]+b;  W:[4H,D+H]  [x,h]:[D+H,N]
	        cblas_sgemm_compute(CblasRowMajor, CblasPacked, CblasNoTrans, m, n, k, w_pack, k, gemmB, n, beta, gemmC, n);
	        // sigmoid for f,i,o, tanh for c_wave
            #pragma omp parallel for
	        for (j = 0; j < hid*batch_size; j++) {
                i_t[j] = 1 / (1 + exp(-gemmC[j]));
                f_t[j] = 1 / (1 + exp(-gemmC[j + hid*batch_size]));
                c_wave_t[j] = tanh(gemmC[j + 2 * hid*batch_size]);
                o_t[j] = 1 / (1 + exp(-gemmC[j + 3 * hid*batch_size]));
                c_t[j] = f_t[j] * c_t[j] + i_t[j] * c_wave_t[j];
                h_t[j] = o_t[j] * tanh(c_t[j]);                
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
        cblas_sgemm_free(w_pack);
    }

    //method: wx_batch_gemm
    int lstm_wx_batch_infer(
        int batch_size, 
        int time_step, 
        int input_dim, 
        int hid, 
        float* w_x, 
        float* w_h, 
        float* b, 
        float* x,
        float* h_0,
        float* c_0,
        float* h_out,
        float* c_out,
        bool return_sequences,
        float* f_t, 
        float* i_t,
        float* c_wave_t,
        float* o_t,
        //float* c_t,
        const float** A,
        const float** B, 
        float**  C,
        float* x_temp) {

        //printf("C      lstm_wx_batch_infer called \n");
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
    
        m[0] = hid;
        k[0] = input_dim;
        n[0] = batch_size;
        
        lda[0] = k[0]; 
        ldb[0] = n[0]; 
        ldc[0] = n[0]; 
        
        transB[0] = CblasNoTrans; 
        transA[0] = CblasNoTrans; 
        
        alpha[0] = 1.0; 
        if (b == NULL) {
            beta[0] = 0.0;
        }
        else {
            beta[0] = 1.0;
            #pragma omp parallel for collapse(3)
            for (i = 0; i < time_step; i++) { 
                for (j = 0; j < hid; j++) { 
                    for (p = 0; p < batch_size; p++) { 
                        size_t offset0 = i * batch_size * hid + j * batch_size + p; 
                        size_t offset1 = (i + time_step) * batch_size * hid + j * batch_size + p; 
                        size_t offset2 = (i + 2 * time_step) * batch_size * hid + j * batch_size + p; 
                        size_t offset3 = (i + 3 * time_step) * batch_size * hid + j * batch_size + p; 
            
                        x_temp[offset0] = b[j]; 
                        x_temp[offset1] = b[j + hid]; 
                        x_temp[offset2] = b[j + 2 * hid]; 
                        x_temp[offset3] = b[j + 3 * hid]; 
                    } 
                } 
            } 
    
        }
        size_per_grp[0] = 4 * time_step;
    
        if (NULL == A || NULL == B || NULL == C || NULL == x_temp || NULL == f_t || NULL == i_t || NULL == c_wave_t || NULL == o_t) {
            //printf( "\n ERROR: malloc global buffers failed \n\n");
            return -3;
        }
        #pragma omp parallel for 
        for (i = 0; i < time_step; i++) { 
            A[i] = w_x;                                       // w_fx
            A[i + time_step] = w_x + input_dim * hid;         // w_ix
            A[i + 2 * time_step] = w_x + 2 * input_dim * hid; // w_cx 
            A[i + 3 * time_step] = w_x + 3 * input_dim * hid; // w_ox 
        
            B[i] = x + i * k[0] * n[0]; 
            B[i + time_step] = B[i]; 
            B[i + 2 * time_step] = B[i]; 
            B[i + 3 * time_step] = B[i]; 
        
            C[i] = x_temp + i * m[0] * n[0]; 
            C[i + time_step] = x_temp + (i + time_step) * m[0] * n[0]; 
            C[i + 2 * time_step] = x_temp + (i + 2 * time_step) * m[0] * n[0]; 
            C[i + 3 * time_step] = x_temp + (i + 3 * time_step) * m[0] * n[0]; 
        } 
        cblas_sgemm_batch(CblasRowMajor, transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, 1, size_per_grp); 
    
        // loop on step
        m[0] = hid;
        k[0] = hid;
        n[0] = batch_size;
        
        beta[0] = 1.0;
    
        lda[0] = k[0]; 
        ldb[0] = n[0]; 
        ldc[0] = n[0]; 
        size_per_grp[0] = 4;
        
        A[0] = w_h;                //w_fh
        A[1] = w_h + hid * hid;    //w_ih
        A[2] = w_h + 2 * hid * hid;//w_ch
        A[3] = w_h + 3 * hid * hid;//w_oh
        
        B[0] = h_0;
        B[1] = h_0;
        B[2] = h_0;
        B[3] = h_0;
    
        size_t mn = m[0] * n[0];
        #pragma omp parallel for
        for (j = 0; j < mn; j++) {
            c_out[j] = c_0[j];
        }
    
        for (i = 0; i < time_step; i++) {
            // f,i,c_wave,o
            C[0] = x_temp + i * m[0] * n[0];
            C[1] = x_temp + (i + time_step) * m[0] * n[0];
            C[2] = x_temp + (i + 2 * time_step) * m[0] * n[0];
            C[3] = x_temp + (i + 3 * time_step) * m[0] * n[0];
    
            cblas_sgemm_batch(CblasRowMajor, transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, 1, size_per_grp);
    
            // sigmoid for f,i,o, tanh for c_wave
            #pragma omp parallel for
            for (j = 0; j < mn; j++) {
                float exp_i = exp((float)(C[0][j]));
                float exp_f = exp((float)(C[1][j]));
                c_wave_t[j] = tanh((float)(C[2][j]));
                float exp_o = exp((float)(C[3][j]));
                f_t[j] = exp_f / ((float)1.0 + exp_f);        
                i_t[j] = exp_i / ((float)1.0 + exp_i);
                o_t[j] = exp_o / ((float)1.0 + exp_o);
            }
            //c
            #pragma omp parallel for 
            for (j = 0; j < mn; j++) { 
                c_out[j] = (float)((float)(f_t[j]) * (float)(c_out[j]) + (float)(i_t[j]) * (float)(c_wave_t[j])); 
            }
            //h
            float* y_ptr = NULL;
            if (return_sequences) {
                y_ptr = h_out + i * batch_size * hid;
            } else {
                y_ptr = h_out;
            }
            #pragma omp parallel for
            for (j = 0; j < mn; j++) {
                y_ptr[j] = (float)(o_t[j]) * tanh((float)(c_out[j]));
            }
            // update
            B[0] = y_ptr;
            B[1] = B[0];
            B[2] = B[0];
            B[3] = B[0];
        }
        return 0;
    }

    //method: wx_batch_gemm_h_pack_gemm
    int lstm_wx_h_pack_infer(
        int batch_size, 
        int time_step, 
        int input_dim, 
        int hid, 
        float* w_x, 
        float* w_h, 
        float* b, 
        float* x,
        float* h_0,
        float* c_0,
        float* h_out,
        float* c_out,
        bool return_sequences,
        float* f_t, 
        float* i_t,
        float* c_wave_t,
        float* o_t,
        const float** A,
        const float** B, 
        float**  C, 
        float** A_pack, 
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
    
        m[0] = hid;
        k[0] = input_dim;
        n[0] = batch_size;
        
        lda[0] = k[0]; 
        ldb[0] = n[0]; 
        ldc[0] = n[0]; 
        
        transB[0] = CblasNoTrans; 
        transA[0] = CblasNoTrans; 
        
        alpha[0] = 1.0; 
        if (b == NULL) {
            beta[0] = 0.0;
        }
        else {
            beta[0] = 1.0;
            #pragma omp parallel for collapse(3)
            for (i = 0; i < time_step; i++) { 
                for (j = 0; j < hid; j++) { 
                    for (p = 0; p < batch_size; p++) { 
                        size_t offset0 = i * batch_size * hid + j * batch_size + p; 
                        size_t offset1 = (i + time_step) * batch_size * hid + j * batch_size + p; 
                        size_t offset2 = (i + 2 * time_step) * batch_size * hid + j * batch_size + p; 
                        size_t offset3 = (i + 3 * time_step) * batch_size * hid + j * batch_size + p; 
            
                        x_temp[offset0] = b[j]; 
                        x_temp[offset1] = b[j + hid]; 
                        x_temp[offset2] = b[j + 2 * hid]; 
                        x_temp[offset3] = b[j + 3 * hid]; 
                    } 
                } 
            } 
    
        }
        size_per_grp[0] = 4 * time_step;
    
        if (NULL == A || NULL == B || NULL == C || NULL == x_temp || NULL == f_t || NULL == i_t || NULL == c_wave_t || NULL == o_t) {
            //printf( "\n ERROR: malloc global buffers failed \n\n");
            return -3;
        }
        #pragma omp parallel for 
        for (i = 0; i < time_step; i++) { 
            A[i] = w_x;                                       // w_fx
            A[i + time_step] = w_x + input_dim * hid;         // w_ix
            A[i + 2 * time_step] = w_x + 2 * input_dim * hid; // w_cx 
            A[i + 3 * time_step] = w_x + 3 * input_dim * hid; // w_ox 
        
            B[i] = x + i * k[0] * n[0]; 
            B[i + time_step] = B[i]; 
            B[i + 2 * time_step] = B[i]; 
            B[i + 3 * time_step] = B[i]; 
        
            C[i] = x_temp + i * m[0] * n[0]; 
            C[i + time_step] = x_temp + (i + time_step) * m[0] * n[0]; 
            C[i + 2 * time_step] = x_temp + (i + 2 * time_step) * m[0] * n[0]; 
            C[i + 3 * time_step] = x_temp + (i + 3 * time_step) * m[0] * n[0]; 
        } 
        cblas_sgemm_batch(CblasRowMajor, transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, 1, size_per_grp); 
    
        // loop on step
        m[0] = hid;
        k[0] = hid;
        n[0] = batch_size;
        
        beta[0] = 1.0;
    
        float* w_fh = w_h;                //w_fh
        float* w_ih = w_h + hid * hid;    //w_ih
        float* w_ch = w_h + 2 * hid * hid;//w_ch
        float* w_oh = w_h + 3 * hid * hid;//w_oh
        //A_pack[0] = cblas_sgemm_alloc(CblasAMatrix, 4*m[0], n[0], k[0]);
        A_pack[0] = cblas_sgemm_alloc(CblasAMatrix, m[0], n[0], k[0]);//w_fh
        A_pack[1] = cblas_sgemm_alloc(CblasAMatrix, m[0], n[0], k[0]);//w_ih
        A_pack[2] = cblas_sgemm_alloc(CblasAMatrix, m[0], n[0], k[0]);//w_ch
        A_pack[3] = cblas_sgemm_alloc(CblasAMatrix, m[0], n[0], k[0]);//w_oh
        //cblas_sgemm_pack(CblasRowMajor, CblasAMatrix, CblasNoTrans, 4*m[0], n[0], k[0], alpha[0], w_h, k[0], A_pack[0]);
        cblas_sgemm_pack(CblasRowMajor, CblasAMatrix, CblasNoTrans, m[0], n[0], k[0], alpha[0], w_fh, k[0], A_pack[0]);
        cblas_sgemm_pack(CblasRowMajor, CblasAMatrix, CblasNoTrans, m[0], n[0], k[0], alpha[0], w_ih, k[0], A_pack[1]);
        cblas_sgemm_pack(CblasRowMajor, CblasAMatrix, CblasNoTrans, m[0], n[0], k[0], alpha[0], w_ch, k[0], A_pack[2]);
        cblas_sgemm_pack(CblasRowMajor, CblasAMatrix, CblasNoTrans, m[0], n[0], k[0], alpha[0], w_oh, k[0], A_pack[3]);
        B[0] = h_0;
        B[1] = h_0;
        B[2] = h_0;
        B[3] = h_0;
    
        size_t mn = m[0] * n[0];
        #pragma omp parallel for
        for (j = 0; j < mn; j++) {
            c_out[j] = c_0[j];
        }
    
        for (i = 0; i < time_step; i++) {
            // f,i,c_wave,o
            C[0] = x_temp + i * m[0] * n[0];
            C[1] = x_temp + (i + time_step) * m[0] * n[0];
            C[2] = x_temp + (i + 2 * time_step) * m[0] * n[0];
            C[3] = x_temp + (i + 3 * time_step) * m[0] * n[0];
   
            //cblas_sgemm_compute(CblasRowMajor, CblasPacked, CblasNoTrans, 4*m[0], n[0], k[0], A[0], k[0], B[0], n[0], beta[0], C[0], n[0]); 
            //#pragma omp parallel for
            for (j = 0; j < 4; j++){
                cblas_sgemm_compute(CblasRowMajor, CblasPacked, CblasNoTrans, m[0], n[0], k[0], A_pack[j], k[0], B[0], n[0], beta[0], C[j], n[0]);
                //cblas_sgemm_compute(CblasRowMajor, CblasPacked, CblasNoTrans, m[0], n[0], k[0], A_pack[1], k[0], B[1], n[0], beta[0], C[1], n[0]);
                //cblas_sgemm_compute(CblasRowMajor, CblasPacked, CblasNoTrans, m[0], n[0], k[0], A_pack[2], k[0], B[2], n[0], beta[0], C[2], n[0]);
                //cblas_sgemm_compute(CblasRowMajor, CblasPacked, CblasNoTrans, m[0], n[0], k[0], A_pack[3], k[0], B[3], n[0], beta[0], C[3], n[0]);
            }
            // sigmoid for f,i,o, tanh for c_wave
            #pragma omp parallel for
            for (j = 0; j < mn; j++) {
                float exp_i = exp((float)(C[0][j]));
                float exp_f = exp((float)(C[1][j]));
                c_wave_t[j] = tanh((float)(C[2][j]));
                float exp_o = exp((float)(C[3][j]));
                f_t[j] = exp_f / ((float)1.0 + exp_f);        
                i_t[j] = exp_i / ((float)1.0 + exp_i);
                o_t[j] = exp_o / ((float)1.0 + exp_o);
            }
            //c
            #pragma omp parallel for 
            for (j = 0; j < mn; j++) { 
                c_out[j] = (float)((float)(f_t[j]) * (float)(c_out[j]) + (float)(i_t[j]) * (float)(c_wave_t[j])); 
            }
            //h
            float* y_ptr = NULL;
            if (return_sequences) {
                y_ptr = h_out + i * batch_size * hid;
            } else {
                y_ptr = h_out;
            }
            #pragma omp parallel for
            for (j = 0; j < mn; j++) {
                y_ptr[j] = (float)(o_t[j]) * tanh((float)(c_out[j]));
            }
            // update
            B[0] = y_ptr;
            B[1] = B[0];
            B[2] = B[0];
            B[3] = B[0];
        }
        cblas_sgemm_free(A_pack[0]);
        cblas_sgemm_free(A_pack[1]);
        cblas_sgemm_free(A_pack[2]);
        cblas_sgemm_free(A_pack[3]);

        return 0;
    }

    static int lstm_create_instance(void* buf, lstm_handle* lstm_han, int input_dim, int hid, int max_time_step, int max_batch_size) { 
        if (max_time_step == 0) {
            max_time_step = 128;
        }
        if (max_batch_size == 0) {
            max_batch_size = 64;
        }
        lstm_han->A = (const float**) buf;
        lstm_han->B = (const float**) (buf + 4 * max_time_step * sizeof (float*));
        lstm_han->A_pack = (float**) (buf + 8 * max_time_step * sizeof (float*));
        lstm_han->C = (float**) (buf + 12 * max_time_step * sizeof (float*));
        lstm_han->x_temp = (float*) (buf + 16 * max_time_step * sizeof (float*));
        lstm_han->gemmB = (float*) (buf + 16 * max_time_step * sizeof (float*) + max_time_step * 4 * max_batch_size * hid * sizeof (float));
        lstm_han->gemmC = (float*) (buf + 16 * max_time_step * sizeof (float*) +
                         (max_time_step * 4 * max_batch_size * hid + max_batch_size * (input_dim + hid)) * sizeof (float)); 
        lstm_han->f_t = (float*) (buf + 16 * max_time_step * sizeof (float*) + 
                       (max_time_step * 4 * max_batch_size * hid + max_batch_size * (input_dim + hid) + 4 * max_batch_size * hid) * sizeof (float));
        lstm_han->i_t = (float*) (buf + 16 * max_time_step * sizeof (float*) +
                       (max_time_step * 4 * max_batch_size * hid + max_batch_size * (input_dim + hid) + 5 * max_batch_size * hid) * sizeof (float));
        lstm_han->c_wave_t = (float*) (buf + 16 * max_time_step * sizeof (float*) +
                            (max_time_step * 4 * max_batch_size * hid + max_batch_size * (input_dim + hid) + 6 * max_batch_size * hid) * sizeof (float));
        lstm_han->o_t = (float*) (buf + 16 * max_time_step * sizeof (float*) +
                       (max_time_step * 4 * max_batch_size * hid + max_batch_size * (input_dim + hid) + 7 * max_batch_size * hid) * sizeof (float));
        //lstm_han->c_t = (float*) (buf + 16 * max_time_step * sizeof (float*) +
        //               (max_time_step * 4 * max_batch_size * hid + max_batch_size * (input_dim + hid) + 8 * max_batch_size * hid) * sizeof (float));
        lstm_han->gemmA = (float*) (buf + 16 * max_time_step * sizeof (float*) +
                       (max_time_step * 4 * max_batch_size * hid + max_batch_size * (input_dim + hid) + 8 * max_batch_size * hid) * sizeof (float));
        return 0;
    }

    // lstm c interface
    // I: input dimension
    // H: hidden size
    // N: batch size
    // T: time step
    int lstm_wx_infer(
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
        float* c_0,     //(H, N)
        float* h_out,   //if return_sequences == true, size = (T, H, N), else size = (H, N)
        float* c_out,   //(H, N)
        bool return_sequences,
        int mode){
#if ENABLE_OMP_SETTING
        #pragma omp parallel default(shared)
        {
            int ompTid = omp_get_thread_num();
            int numomp = omp_get_num_threads();
            int numprc = omp_get_num_procs();
            int ompmax = omp_get_max_threads();
            kmp_affinity_mask_t new_omp_mask;
            kmp_create_affinity_mask(&new_omp_mask);
            kmp_set_affinity_mask_proc(ompTid, &new_omp_mask);
            kmp_set_affinity_mask_proc(ompTid + ompmax, &new_omp_mask);
            if (kmp_set_affinity(&new_omp_mask) != 0)
            {
                printf("Error: kmp_set_affinity(%d, &new_omp_mask)\n", ompTid);
            }
        }
#endif
        lstm_handle* lstm_han = (lstm_handle*)mkl_malloc(sizeof(lstm_handle), 64);
        lstm_create_instance(buf, lstm_han, input_dim, hid, max_time_step, max_batch_size);
        const float** A = lstm_han->A;
        const float** B = lstm_han->B;
        float** A_pack = lstm_han->A_pack;
        float** C = lstm_han->C;
        float* x_temp = lstm_han->x_temp;
        float* gemmA = lstm_han->gemmA;
        float* gemmB = lstm_han->gemmB;
        float* gemmC = lstm_han->gemmC;
        float* f_t = lstm_han->f_t;
        float* i_t = lstm_han->i_t;
        float* c_wave_t = lstm_han->c_wave_t;
        float* o_t = lstm_han->o_t;
        //printf("mode = %d\n",mode);
        switch(mode) {
            case 4:
                lstm_wx_batch_infer(batch_size, time_step, input_dim, hid, w_x, w_h, b, x, h_0, c_0, h_out, c_out, 
                                        return_sequences,f_t,i_t,c_wave_t,o_t,A,B,C,x_temp);       
                break;
            case 5:
                lstm_wx_h_pack_infer(batch_size, time_step, input_dim, hid, w_x, w_h, b, x, h_0, c_0, h_out, c_out, 
                                         return_sequences,f_t,i_t,c_wave_t,o_t,A,B,C,A_pack,x_temp);
                break;
            case 0:
                lstm_wx_sequential_infer(batch_size, time_step, input_dim, hid, w_x, w_h, b, x, h_0, c_0, h_out, c_out, 
                                             return_sequences,f_t,i_t,c_wave_t,o_t,gemmC);       
                break;
            case 1:
                lstm_wx_combine_infer(batch_size, time_step, input_dim, hid, w_x, w_h, b, x, h_0, c_0, h_out, c_out, 
                                          return_sequences,f_t,i_t,c_wave_t,o_t,gemmA,gemmB,gemmC);
                break;
            case 2:
                lstm_wx_pack_infer(batch_size, time_step, input_dim, hid, w_x, w_h, b, x, h_0, c_0, h_out, c_out, 
                                       return_sequences,f_t,i_t,c_wave_t,o_t,gemmC);       
                break;
            case 3:
                lstm_wx_combine_pack_infer(batch_size, time_step, input_dim, hid, w_x, w_h, b, x, h_0, c_0, h_out, c_out, 
                                               return_sequences,f_t,i_t,c_wave_t,o_t,gemmA,gemmB,gemmC);       
                break;
            default: 
                lstm_wx_batch_infer(batch_size, time_step, input_dim, hid, w_x, w_h, b, x, h_0, c_0, h_out, c_out, 
                                        return_sequences,f_t,i_t,c_wave_t,o_t,A,B,C,x_temp);       
        }
        mkl_free(lstm_han);
        return 0;
    }

    int lstm_wx_infer_get_workspace_size(int input_dim, int hid, int max_time_step, int max_batch_size)
    {
        int max_ws = 16 * max_time_step * sizeof (float*) + (max_time_step * 4 * max_batch_size * hid 
                     + max_batch_size * (input_dim + hid) + 8 * max_batch_size * hid + 4 * hid * (input_dim + hid)) * sizeof (float);
        return max_ws;
    }
}
