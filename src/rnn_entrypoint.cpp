#include <cstddef>
#include <iostream>
#include <rnn.h>
#include <mkl.h>
#include <omp.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

extern "C" {

    void lstm_forward(float * input, float * weight, float * output){}
    void LSTM_combine_pack_gemm(int batch_size, int time_step, int input_dim, int hid,
        float* w_x,   //4H*(D+H)
        float* w_h,   //None
        float* b,     //4H or None 
        float* x,     //T*D*N
        float* h_0,   //H*N
        float* c_0,   //H*N
        float* h_out,   //UNCERTAIN
        bool return_sequences,
        float* f_t,
	float* i_t,
	float* c_wave_t,
	float* o_t,
	float* c_t,
	float* gemmB, //(D+H)*N
	float* gemmC) {

	printf("C      lstm_combine_pack_gemm called \n");
	int i, j, p;
	int m = hid * 4;
	int n = batch_size;
	int k = input_dim+hid;
	float beta = (b == NULL) ? 0.0 : 1.0;
	float* h_t = h_0;
	memcpy(c_t, c_0, hid*batch_size * sizeof(float));
	float* w_pack = cblas_sgemm_alloc(CblasAMatrix, m, n, k);
	cblas_sgemm_pack(CblasRowMajor, CblasAMatrix, CblasNoTrans, m, n, k, 1.0, w_x, k, w_pack);
	//memcpy(h_t, h_0, hid*batch_size * sizeof(float));
	for (i = 0; i < time_step; i++) {
            if (b) {
		#pragma omp parallel for
                for (j = 0; j < batch_size; j++) {
		    for (p = 0; p<4 * hid; p++) {
			gemmC[batch_size*p + j] = b[p];
                    }
	  	}
	    }
	    memcpy(gemmB, x + i*input_dim*batch_size, input_dim*batch_size * sizeof(float));
	    memcpy(gemmB+input_dim*batch_size, h_t, hid*batch_size * sizeof(float));
            //W*[x;h]+b;  W:[4H,D+H]  [x,h]:[D+H,N]
	    cblas_sgemm_compute(CblasRowMajor, CblasPacked, CblasNoTrans, m, n, k, w_pack, k, gemmB, n, beta, gemmC, n);
	    // sigmoid for f,i,o, tanh for c_wave
            #pragma omp parallel for
	    for (j = 0; j < hid*batch_size; j++) {
		f_t[j] = 1 / (1 + exp(-gemmC[j]));
		i_t[j] = 1 / (1 + exp(-gemmC[j + hid*batch_size]));
		c_wave_t[j] = tanh(gemmC[j + 2 * hid*batch_size]);
		o_t[j] = 1 / (1 + exp(-gemmC[j + 3 * hid*batch_size]));

		c_t[j] = f_t[j] * c_t[j] + i_t[j] * c_wave_t[j];
		if (return_sequences) {
	            h_out[j + i*hid*batch_size] = o_t[j] * tanh(c_t[j]);
	        }
	        else {
		    h_out[j] = o_t[j] * tanh(c_t[j]);
	        }
	    }
            if(return_sequences){
                h_t = h_out + i*hid*batch_size;
            }
            else{
                h_t = h_out;
            }
        }
    }
    void LSTM_combine_gemm(int batch_size, int time_step, int input_dim, int hid,
        float* w_x,   //4H*(D+H)
		float* w_h,   //None
		float* b,     //4H or None 
		float* x,     //T*D*N
		float* h_0,   //H*N
		float* c_0,   //H*N
		float* h_out,   //UNCERTAIN
		bool return_sequences,
		float* f_t,
		float* i_t,
		float* c_wave_t,
		float* o_t,
		float* c_t,
		float* gemmB, //(D+H)*N
		float* gemmC) {

		printf("C      lstm_combine_gemm called \n");
		int i, j, p;
		int m = hid * 4;
		int n = batch_size;
		int k = input_dim+hid;
		float beta = (b == NULL) ? 0.0 : 1.0;
		float* h_t = h_0;
		memcpy(c_t, c_0, hid*batch_size * sizeof(float));
		//memcpy(h_t, h_0, hid*batch_size * sizeof(float));
		for (i = 0; i < time_step; i++) {
			if (b) {
                                #pragma omp parallel for
				for (j = 0; j < batch_size; j++) {
					for (p = 0; p<4 * hid; p++) {
						gemmC[batch_size*p + j] = b[p];
					}
				}
			}
			memcpy(gemmB, x + i*input_dim*batch_size, input_dim*batch_size * sizeof(float));
			memcpy(gemmB+input_dim*batch_size, h_t, hid*batch_size * sizeof(float));
                	//W*[x;h]+b;  W:[4H,D+H]  [x,h]:[D+H,N]
			cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1, w_x, k, gemmB, n, beta, gemmC, n);
			// sigmoid for f,i,o, tanh for c_wave
                        #pragma omp parallel for
			for (j = 0; j < hid*batch_size; j++) {
				f_t[j] = 1 / (1 + exp(-gemmC[j]));
				i_t[j] = 1 / (1 + exp(-gemmC[j + hid*batch_size]));
				c_wave_t[j] = tanh(gemmC[j + 2 * hid*batch_size]);
				o_t[j] = 1 / (1 + exp(-gemmC[j + 3 * hid*batch_size]));

				c_t[j] = f_t[j] * c_t[j] + i_t[j] * c_wave_t[j];
				if (return_sequences) {
					h_out[j + i*hid*batch_size] = o_t[j] * tanh(c_t[j]);
					h_t = h_out + i*hid*batch_size;
				}
				else {
					h_out[j] = o_t[j] * tanh(c_t[j]);
					h_t = h_out;
				}
			}
		}
	}
    void LSTM_pack_gemm(int batch_size, int time_step, int input_dim, int hid,
        float* w_x,   //4H*D
        float* w_h,   //4H*H
        float* b,     //4H 
        float* x,     //T*D*N
        float* h_0,   //H*N
        float* c_0,   //H*N
        float* h_out,   //UNCERTAIN
        bool return_sequences,
        float* f_t,
        float* i_t,
        float* c_wave_t,
        float* o_t,
        float* c_t,
        float* gemmC) {
        
        printf("C      lstm_pack_gemm called \n");
	int i, j, p;
	int m = hid * 4;
	int n = batch_size;
	int k = input_dim;
	float beta = (b == NULL) ? 0.0 : 1.0;
	float* h_t = h_0;
	memcpy(c_t, c_0, hid*batch_size * sizeof(float));
	float* w_x_pack = cblas_sgemm_alloc(CblasAMatrix, m, n, k);
	float* w_h_pack = cblas_sgemm_alloc(CblasAMatrix, m, n, hid);
	cblas_sgemm_pack(CblasRowMajor, CblasAMatrix, CblasNoTrans, m, n, k, 1.0, w_x, k, w_x_pack);
	cblas_sgemm_pack(CblasRowMajor, CblasAMatrix, CblasNoTrans, m, n, hid, 1.0, w_h, hid, w_h_pack);
	for (i = 0; i < time_step; i++) {
	    if (b) {
                #pragma omp parallel for
	        for (j = 0; j < batch_size; j++) {
	            for (p = 0; p<4 * hid; p++) {
		        gemmC[batch_size*p + j] = b[p];
		    }
		}
	    }
            //Wx*x+b; Wx:[4H,D] x:[T, D, N]
            cblas_sgemm_compute(CblasRowMajor, CblasPacked, CblasNoTrans, m, n, k, w_x_pack, k, x + i*k*n, n, beta, gemmC, n);
            //Wh*h+tmp; Wh:[4H,H] h:[H,N]
            cblas_sgemm_compute(CblasRowMajor, CblasPacked, CblasNoTrans, m, n, hid, w_h_pack, hid, h_t, n, 1, gemmC, n);

            // sigmoid for f,i,o, tanh for c_wave
            #pragma omp parallel for
            for (j = 0; j < hid*batch_size; j++) {
                f_t[j] = 1 / (1 + exp(-gemmC[j]));
                i_t[j] = 1 / (1 + exp(-gemmC[j + hid*batch_size]));
                c_wave_t[j] = tanh(gemmC[j + 2 * hid*batch_size]);
                o_t[j] = 1 / (1 + exp(-gemmC[j + 3 * hid*batch_size]));

                c_t[j] = f_t[j] * c_t[j] + i_t[j] * c_wave_t[j];
                if (return_sequences) {
    	            h_out[j + i*hid*batch_size] = o_t[j] * tanh(c_t[j]);
                }
                else {
	            h_out[j] = o_t[j] * tanh(c_t[j]);
	        }
    	    }
            if(return_sequences){
                h_t = h_out + i*hid*batch_size;
            }
            else{
                h_t = h_out;
            }
        }
        cblas_sgemm_free(w_x_pack);
        cblas_sgemm_free(w_h_pack);
    }
    void LSTM_sequential_gemm(int batch_size, int time_step, int input_dim, int hid, 
	float* w_x,   //4H*D
	float* w_h,   //4H*H
	float* b,     //4H 
	float* x,     //T*D*N
	float* h_0,   //H*N
	float* c_0,   //H*N
	float* h_out,   //UNCERTAIN
	bool return_sequences,
        float* f_t,
        float* i_t,
        float* c_wave_t,
        float* o_t,
        float* c_t,
        float* gemmC){
        
        printf("C      lstm_sequential_gemm called \n");
        int i,j,p;
        int m = hid*4;
	int n = batch_size;
	int k = input_dim;
	float beta = (b == NULL) ? 0.0 : 1.0;
        float* h_t=h_0;
	memcpy(c_t, c_0, hid*batch_size * sizeof(float));
	//memcpy(h_t, h_0, hid*batch_size * sizeof(float));
	for (i = 0; i < time_step; i++) {
            if(b){
                #pragma omp parallel for
	        for (j = 0; j < batch_size; j++) {
                    for(p=0;p<4*hid;p++){
                        gemmC[batch_size*p+j]=b[p];
                    }
	        }
            }
	    //Wx*x+b; Wx:[4H,D] x:[T, D, N]
	    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1, w_x, k,x+i*k*n, n, beta, gemmC, n);
            //Wh*h+tmp; Wh:[4H,H] h:[H,N]
	    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, hid, 1, w_h, hid, h_t, n, 1, gemmC, n);
            // sigmoid for f,i,o, tanh for c_wave
            #pragma omp parallel for
            for (j = 0; j < hid*batch_size; j++) {
		f_t[j] = 1 / (1 + exp(-gemmC[j]));
		i_t[j] = 1 / (1 + exp(-gemmC[j+hid*batch_size]));
		c_wave_t[j] = tanh(gemmC[j+2*hid*batch_size]);
		o_t[j] = 1 / (1 + exp(-gemmC[j+3*hid*batch_size]));

		c_t[j] = f_t[j] * c_t[j] + i_t[j] * c_wave_t[j];
		if (return_sequences) {
		    h_out[j+i*hid*batch_size] = o_t[j] * tanh(c_t[j]);
		}
		else {
		    h_out[j] = o_t[j] * tanh(c_t[j]);
		}
            }
	    if(return_sequences){
                h_t = h_out + i*hid*batch_size;
            }
            else{
                h_t = h_out;
            }
        }
    }
    void  LSTM_batch_gemm(int batch_size, int time_step, int input_dim, int hid, float* w_x, float* w_h, float* b, float* x, float* h_0, float* c_0, /*out*/float* y, bool return_sequences, /*temp memory*/float* f_t, float* i_t, float* c_wave_t, float* o_t, float* c_t, const float** A, const float** B, float**  C, float* x_temp){
        printf("C      lstm_batch_gemm called \n");
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
            #pragma omp parallel for 
            for (i = 0; i < time_step; i++) { 
                for (j = 0; j < batch_size; j++) { 
                    for (p = 0; p < hid; p++) { 
                        size_t offset0 = i * batch_size * hid + j * hid + p; 
                        size_t offset1 = (i + time_step) * batch_size * hid + j * hid + p; 
                        size_t offset2 = (i + 2 * time_step) * batch_size * hid + j * hid + p; 
                        size_t offset3 = (i + 3 * time_step) * batch_size * hid + j * hid + p; 
            
                        x_temp[offset0] = b[p]; 
                        x_temp[offset1] = b[p + hid]; 
                        x_temp[offset2] = b[p + 2 * hid]; 
                        x_temp[offset2] = b[p + 3 * hid]; 
                    } 
                } 
            } 
    
        }
        size_per_grp[0] = 4 * time_step;
    
        if (NULL == A || NULL == B || NULL == C || NULL == x_temp || NULL == f_t || NULL == i_t || NULL == c_wave_t || NULL == o_t || NULL == c_t) {
            printf( "\n ERROR: malloc global buffers failed \n\n");
            return;
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
            c_t[j] = c_0[j];
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
                float exp_f = exp((float)(C[0][j]));
                float exp_i = exp((float)(C[1][j]));
                c_wave_t[j] = tanh((float)(C[2][j]));
                float exp_o = exp((float)(C[3][j]));
                f_t[j] = exp_f / ((float)1.0 + exp_f);        
                i_t[j] = exp_i / ((float)1.0 + exp_i);
                o_t[j] = exp_o / ((float)1.0 + exp_o);
            }
            //c
            #pragma omp parallel for 
            for (j = 0; j < mn; j++) { 
                c_t[j] = (float)((float)(f_t[j]) * (float)(c_t[j]) + (float)(i_t[j]) * (float)(c_wave_t[j])); 
            }
            //h
            float* y_ptr = NULL;
            if (return_sequences) {
                y_ptr = y + i * batch_size * hid;
            } else {
                y_ptr = y;
            }
            #pragma omp parallel for
            for (j = 0; j < mn; j++) {
                y_ptr[j] = (float)(o_t[j]) * tanh((float)(c_t[j]));
            }
            // update
            B[0] = y_ptr;
            B[1] = B[0];
            B[2] = B[0];
            B[3] = B[0];
        }
    }
    void  LSTM_h_pack_gemm(int batch_size, int time_step, int input_dim, int hid, float* w_x, float* w_h, float* b, float* x, float* h_0, float* c_0, /*out*/float* y, bool return_sequences, /*temp memory*/float* f_t, float* i_t, float* c_wave_t, float* o_t, float* c_t, const float** A, const float** B, float**  C, float** A_pack, float* x_temp){
        printf("C      lstm_pack_gemm called \n");
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
            #pragma omp parallel for 
            for (i = 0; i < time_step; i++) { 
                for (j = 0; j < batch_size; j++) { 
                    for (p = 0; p < hid; p++) { 
                        size_t offset0 = i * batch_size * hid + j * hid + p; 
                        size_t offset1 = (i + time_step) * batch_size * hid + j * hid + p; 
                        size_t offset2 = (i + 2 * time_step) * batch_size * hid + j * hid + p; 
                        size_t offset3 = (i + 3 * time_step) * batch_size * hid + j * hid + p; 
            
                        x_temp[offset0] = b[p]; 
                        x_temp[offset1] = b[p + hid]; 
                        x_temp[offset2] = b[p + 2 * hid]; 
                        x_temp[offset2] = b[p + 3 * hid]; 
                    } 
                } 
            } 
    
        }
        size_per_grp[0] = 4 * time_step;
    
        if (NULL == A || NULL == B || NULL == C || NULL == x_temp || NULL == f_t || NULL == i_t || NULL == c_wave_t || NULL == o_t || NULL == c_t) {
            printf( "\n ERROR: malloc global buffers failed \n\n");
            return;
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
        A_pack[0] = cblas_sgemm_alloc(CblasAMatrix, m[0], n[0], k[0]);//w_fh
        A_pack[1] = cblas_sgemm_alloc(CblasAMatrix, m[0], n[0], k[0]);//w_ih
        A_pack[2] = cblas_sgemm_alloc(CblasAMatrix, m[0], n[0], k[0]);//w_ch
        A_pack[3] = cblas_sgemm_alloc(CblasAMatrix, m[0], n[0], k[0]);//w_oh
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
            c_t[j] = c_0[j];
        }
    
        for (i = 0; i < time_step; i++) {
            // f,i,c_wave,o
            C[0] = x_temp + i * m[0] * n[0];
            C[1] = x_temp + (i + time_step) * m[0] * n[0];
            C[2] = x_temp + (i + 2 * time_step) * m[0] * n[0];
            C[3] = x_temp + (i + 3 * time_step) * m[0] * n[0];
    
            cblas_sgemm_compute(CblasRowMajor, CblasPacked, CblasNoTrans, m[0], n[0], k[0], A_pack[0], k[0], B[0], n[0], beta[0], C[0], n[0]);
            cblas_sgemm_compute(CblasRowMajor, CblasPacked, CblasNoTrans, m[0], n[0], k[0], A_pack[1], k[0], B[1], n[0], beta[0], C[1], n[0]);
            cblas_sgemm_compute(CblasRowMajor, CblasPacked, CblasNoTrans, m[0], n[0], k[0], A_pack[2], k[0], B[2], n[0], beta[0], C[2], n[0]);
            cblas_sgemm_compute(CblasRowMajor, CblasPacked, CblasNoTrans, m[0], n[0], k[0], A_pack[3], k[0], B[3], n[0], beta[0], C[3], n[0]);

            // sigmoid for f,i,o, tanh for c_wave
            #pragma omp parallel for
            for (j = 0; j < mn; j++) {
                float exp_f = exp((float)(C[0][j]));
                float exp_i = exp((float)(C[1][j]));
                c_wave_t[j] = tanh((float)(C[2][j]));
                float exp_o = exp((float)(C[3][j]));
                f_t[j] = exp_f / ((float)1.0 + exp_f);        
                i_t[j] = exp_i / ((float)1.0 + exp_i);
                o_t[j] = exp_o / ((float)1.0 + exp_o);
            }
            //c
            #pragma omp parallel for 
            for (j = 0; j < mn; j++) { 
                c_t[j] = (float)((float)(f_t[j]) * (float)(c_t[j]) + (float)(i_t[j]) * (float)(c_wave_t[j])); 
            }
            //h
            float* y_ptr = NULL;
            if (return_sequences) {
                y_ptr = y + i * batch_size * hid;
            } else {
                y_ptr = y;
            }
            #pragma omp parallel for
            for (j = 0; j < mn; j++) {
                y_ptr[j] = (float)(o_t[j]) * tanh((float)(c_t[j]));
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
    }
}

