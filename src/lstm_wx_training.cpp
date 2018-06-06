#include <cstddef>
#include <iostream>
#include <rnn.h>
#include <mkl.h>
#include <omp.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#define max(a,b) ((a) > (b) ? (a) : (b))

extern "C" {
    struct lstm_handle {
        const float** A;
        const float** B;
        float** C;
        float* x_temp;

        float* hf;
        float* hi;
        float* hc;
        float* ho;
        float* c;
        float* h;

        //bp
        float* dh;
        float* dc;
        float* dh_next;
        float* dc_next;
        float* dhf;
        float* dhi;
        float* dhc;
        float* dho;
        float* temp;
    };

void  LSTM_forward(int batch_size, int time_step, int input_dim, int hid, 
                      const float* w_x, const float* w_h, const float* b, const float* x, const float* h_0, const float* c_0, 
               /*out*/float *o_t, float *f_t, float *i_t, float* c_wave_t, //hid * batch_size
                      float* c_t, float* h,//time_Step * hid * batch_size
            /*global*/const float** A, const float** B, float** C, float* x_temp){
    ////global
    //const float** A = (const float**)mkl_malloc(4 * time_step * sizeof (float*), 64);
    //const float** B = (const float**)mkl_malloc(4 * time_step * sizeof (float*), 64);
    //float** C = (float**)mkl_malloc(4 * time_step * sizeof (float*), 64);
    //float* x_temp = (float*)mkl_malloc(time_step * 4 * batch_size * hid * sizeof (float), 64);
    
    memset(x_temp, 0, sizeof(float) * (time_step * 4 * batch_size * hid));
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

    if (NULL == A || NULL == B || NULL == C || NULL == x_temp) {
        printf( "\n ERROR: malloc global buffers failed \n\n");
        return;
    }
    #pragma omp parallel for 
    for (i = 0; i < time_step; i++) { 
        A[i] = w_x;                                       // w_ix
        A[i + time_step] = w_x + input_dim * hid;         // w_fx
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
    
    A[0] = w_h;                //w_ih
    A[1] = w_h + hid * hid;    //w_fh
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

    float* c_t_ptr = NULL;
    float* f_t_ptr = NULL;
    float* i_t_ptr = NULL;
    float* c_wave_t_ptr = NULL;
    float* o_t_ptr = NULL;
    for (i = 0; i < time_step; i++) {
        // f,i,c_wave,o
        C[0] = x_temp + i * m[0] * n[0];
        C[1] = x_temp + (i + time_step) * m[0] * n[0];
        C[2] = x_temp + (i + 2 * time_step) * m[0] * n[0];
        C[3] = x_temp + (i + 3 * time_step) * m[0] * n[0];

        cblas_sgemm_batch(CblasRowMajor, transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, 1, size_per_grp);

        // sigmoid for f,i,o, tanh for c_wave
        c_t_ptr = c_t + i * mn;
        f_t_ptr = f_t + i * mn;
        i_t_ptr = i_t + i * mn;
        o_t_ptr = o_t + i * mn;
        c_t_ptr = c_t + i * mn;
        c_wave_t_ptr = c_wave_t + i * mn;
        #pragma omp parallel for
        for (j = 0; j < mn; j++) {
            float exp_i = exp((float)(C[0][j]));
            float exp_f = exp((float)(C[1][j]));
            c_wave_t_ptr[j] = tanh((float)(C[2][j]));
            float exp_o = exp((float)(C[3][j]));
            f_t_ptr[j] = exp_f / ((float)1.0 + exp_f);        
            i_t_ptr[j] = exp_i / ((float)1.0 + exp_i);
            o_t_ptr[j] = exp_o / ((float)1.0 + exp_o);
        }
        //c
        const float* c_tm1 = NULL;
        if(i == 0) 
            c_tm1 = c_0;
        else
            c_tm1 = c_t_ptr - mn;
        #pragma omp parallel for 
        for (j = 0; j < mn; j++) { 
            c_t_ptr[j] = (float)((float)(f_t_ptr[j]) * (float)(c_tm1[j]) + (float)(i_t_ptr[j]) * (float)(c_wave_t_ptr[j])); 
        }
        float* y_ptr = NULL;
        y_ptr = h + i * mn;
        //h:all time_step
        #pragma omp parallel for
        for (j = 0; j < mn; j++) {
            y_ptr[j] = (float)(o_t_ptr[j]) * tanh((float)(c_t_ptr[j]));
        }
        // update
        B[0] = y_ptr;
        B[1] = B[0];
        B[2] = B[0];
        B[3] = B[0];
    }
    //mkl_free(A);
    //mkl_free(B);
    //mkl_free(C);
    //mkl_free(x_temp);
}

void  LSTM_backward(int batch_size, int time_step, int input_dim, int hid, 
                    const float* w_x, const float* w_h, const float* b, const float* x, const float* h_0, const float* c_0,//same with forward input
                    float *ho, float *hf, float *hi, float* hc, float* c, float* h,//forward output: time_step * hid * batch_size
                    float* grad_last,//last gradient
             /*out*/float* dwix, float* dwfx, float* dwcx, float* dwox,
                    float* dwih, float* dwfh, float* dwch, float* dwoh, 
                    float* dbi, float* dbf, float* dbc, float* dbo, 
                    float* dx,
                    const float** A, const float** B, float** C,  float* dh, float* dc, float* dh_next, float* dc_next, float* dhf, float* dhi, float* dhc, float* dho, float* x_temp){
    //global
    //const float** A = (const float**)mkl_malloc(4 * time_step * sizeof (float*), 64);
    //const float** B = (const float**)mkl_malloc(4 * time_step * sizeof (float*), 64);
    //float** C = (float**)mkl_malloc(4 * time_step * sizeof (float*), 64);
    memset(dwfx, 0, sizeof(float) * hid * input_dim);
    memset(dwix, 0, sizeof(float) * hid * input_dim);
    memset(dwcx, 0, sizeof(float) * hid * input_dim);
    memset(dwox, 0, sizeof(float) * hid * input_dim);
    memset(dwfh, 0, sizeof(float) * hid * hid);
    memset(dwih, 0, sizeof(float) * hid * hid);
    memset(dwch, 0, sizeof(float) * hid * hid);
    memset(dwoh, 0, sizeof(float) * hid * hid);
    memset(dbf, 0, sizeof(float) * hid);
    memset(dbi, 0, sizeof(float) * hid);
    memset(dbc, 0, sizeof(float) * hid);
    memset(dbo, 0, sizeof(float) * hid);
    
    int i,j,p;
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

    //from last timestep
    memset(dh_next, 0, sizeof(float) * hid * batch_size);
    memset(dc_next, 0, sizeof(float) * hid * batch_size);
    
    //cache: hf hi hc ho c, c=[c_0, all c_t]i
    //calculate all gf, gi, gc_wave, go
    // loop on step
    m[0] = hid;
    k[0] = hid;
    n[0] = batch_size;
    
    beta[0] = 0.0;

    lda[0] = m[0]; 
    ldb[0] = n[0]; 
    ldc[0] = n[0]; 
    transA[0] = CblasTrans; 
    transB[0] = CblasNoTrans; 
    
    alpha[0] = 1.0; 
    size_per_grp[0] = 4;
    
    A[0] = w_h;                //w_ih
    A[1] = w_h + hid * hid;    //w_fh
    A[2] = w_h + 2 * hid * hid;//w_ch
    A[3] = w_h + 3 * hid * hid;//w_oh
    
    size_t bh = batch_size * hid;
    size_t tbi = batch_size * input_dim * time_step;
    size_t ib = input_dim * batch_size;
    size_t hh = hid * hid;
    C[0] = x_temp;
    C[1] = x_temp + bh;
    C[2] = x_temp + bh * 2;
    C[3] = x_temp + bh * 3;
    float* c_ptr = NULL;
    float* hf_ptr = NULL;
    float* hi_ptr = NULL;
    float* hc_ptr = NULL;
    float* ho_ptr = NULL;
    float* dhf_ptr = NULL;
    float* dhi_ptr = NULL;
    float* dhc_ptr = NULL;
    float* dho_ptr = NULL;

    for(i = time_step - 1; i >= 0; i--) {
        int kk = i * bh;
        c_ptr = c + kk;
        hf_ptr = hf + kk;
        hi_ptr = hi + kk;
        hc_ptr = hc + kk;
        ho_ptr = ho + kk;
        dhf_ptr = dhf + kk;
        dhi_ptr = dhi + kk;
        dhc_ptr = dhc + kk;
        dho_ptr = dho + kk;
        
        const float *c_old;
        if(i != 0)
            c_old = c_ptr - bh;
        else
            c_old = c_0;
        if(i == time_step - 1) {
            #pragma omp parallel for
            for(j = 0; j < bh; j++ ) {
                float tanh_c = tanh(c_ptr[j]);
                //dh[j] = 1.0 + dh_next[j];
                dh[j] = grad_last[j] + dh_next[j];
                dho_ptr[j] = ho_ptr[j] * (1.0 - ho_ptr[j]) * tanh_c * dh[j];
                dc[j] = ho_ptr[j] * dh[j] * (1.0 - tanh_c * tanh_c) + dc_next[j];
                dhf_ptr[j] = hf_ptr[j] * (1.0 - hf_ptr[j]) * c_old[j] * dc[j];
                dhi_ptr[j] = hi_ptr[j] * (1.0 - hi_ptr[j]) * hc_ptr[j] * dc[j];
                dhc_ptr[j] = (1.0 - hc_ptr[j] * hc_ptr[j]) * hi_ptr[j] * dc[j];
            }
        }
        else {
            #pragma omp parallel for
            for(j = 0; j < bh; j++ ) {
                float tanh_c = tanh(c_ptr[j]);
                dh[j] = dh_next[j];
                dho_ptr[j] = ho_ptr[j] * (1.0 - ho_ptr[j]) * tanh_c * dh[j];
                dc[j] = ho_ptr[j] * dh[j] * (1.0 - tanh_c * tanh_c) + dc_next[j];
                dhf_ptr[j] = hf_ptr[j] * (1.0 - hf_ptr[j]) * c_old[j] * dc[j];
                dhi_ptr[j] = hi_ptr[j] * (1.0 - hi_ptr[j]) * hc_ptr[j] * dc[j];
                dhc_ptr[j] = (1.0 - hc_ptr[j] * hc_ptr[j]) * hi_ptr[j] * dc[j];
            }
        }
        B[0] = dhi_ptr;
        B[1] = dhf_ptr;
        B[2] = dhc_ptr;
        B[3] = dho_ptr;
        cblas_sgemm_batch(CblasRowMajor, transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, 1, size_per_grp);
    
        //calculate dbf, dbi, dbc, dbo
        #pragma omp parallel for
        for(j = 0; j < bh; j++ ) {
            dh_next[j] = C[0][j] + C[1][j] + C[2][j] + C[3][j];
            dc_next[j] = hf_ptr[j] * dc[j];
        }
        #pragma omp parallel for
        for(j = 0; j < hid; j++) {
            for(p = 0; p < batch_size; p++) {
                int index = j * batch_size + p;
                dbf[j] += dhf_ptr[index];
                dbi[j] += dhi_ptr[index];
                dbc[j] += dhc_ptr[index];
                dbo[j] += dho_ptr[index];
            }
        } 
    }
    //calculate dwfx, dwix, dwcx, dwox
    m[0] = hid;
    k[0] = batch_size;
    n[0] = input_dim;

    lda[0] = k[0];
    ldb[0] = k[0];
    ldc[0] = n[0];
    transA[0] = CblasNoTrans;
    transB[0] = CblasTrans;
    
    size_per_grp[0] = 4 * time_step;
    for (i = 0; i < time_step; i++) {
        A[i] = dhf + i * bh;                 
        A[i + time_step] = dhi + i * bh;    
        A[i + 2 * time_step] = dhc + i * bh; 
        A[i + 3 * time_step] = dho + i * bh; 
    
        B[i] = x + i * input_dim * batch_size; 
        B[i + time_step] = B[i]; 
        B[i + 2 * time_step] = B[i]; 
        B[i + 3 * time_step] = B[i]; 
    
        C[i] = x_temp + i * hid * input_dim;
        C[i + time_step] = x_temp + (i + time_step) * hid * input_dim; 
        C[i + 2 * time_step] = x_temp + (i + 2 * time_step) * hid * input_dim; 
        C[i + 3 * time_step] = x_temp + (i + 3 * time_step) * hid * input_dim; 
    }
    cblas_sgemm_batch(CblasRowMajor, transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, 1, size_per_grp);
    #pragma omp parallel for
    for(i = 0; i < hid * input_dim; i++) {
        for(j = 0; j < time_step; j++) {
            dwfx[i] += C[j][i];
            dwix[i] += C[j + time_step][i];
            dwcx[i] += C[j + 2 * time_step][i];
            dwox[i] += C[j + 3 * time_step][i];
        }
    }
    //calculate dwfh, dwih, dwch, dwoh
    m[0] = hid;
    k[0] = batch_size;
    n[0] = hid;

    lda[0] = k[0];
    ldb[0] = k[0];
    ldc[0] = n[0];
    transA[0] = CblasNoTrans;
    transB[0] = CblasTrans;
    
    size_per_grp[0] = 4 * time_step;
    for (i = 0; i < time_step; i++) {
        A[i] = dhf + i * bh;                 
        A[i + time_step] = dhi + i * bh;    
        A[i + 2 * time_step] = dhc + i * bh; 
        A[i + 3 * time_step] = dho + i * bh; 
   
        if(i == 0) {
            B[i] = h_0; 
            B[i + time_step] = B[i]; 
            B[i + 2 * time_step] = B[i]; 
            B[i + 3 * time_step] = B[i]; 
        }    
        else {
            B[i] = h + (i - 1) * bh; 
            B[i + time_step] = B[i]; 
            B[i + 2 * time_step] = B[i]; 
            B[i + 3 * time_step] = B[i]; 
        } 
        C[i] = x_temp + i * hh;
        C[i + time_step] = x_temp + (i + time_step) * hh; 
        C[i + 2 * time_step] = x_temp + (i + 2 * time_step) * hh; 
        C[i + 3 * time_step] = x_temp + (i + 3 * time_step) * hh; 
    } 
    cblas_sgemm_batch(CblasRowMajor, transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, 1, size_per_grp);
    #pragma omp parallel for
    for(i = 0; i < hid * hid; i++) {
        for(j = 0; j < time_step; j++) {
            dwfh[i] += C[j][i];
            dwih[i] += C[j + time_step][i];
            dwch[i] += C[j + 2 * time_step][i];
            dwoh[i] += C[j + 3 * time_step][i];
        }
    }

    //calculate dx
    m[0] = input_dim;
    k[0] = hid;
    n[0] = batch_size;
    
    lda[0] = m[0]; 
    ldb[0] = n[0]; 
    ldc[0] = n[0]; 
    transA[0] = CblasTrans;
    transB[0] = CblasNoTrans;
    size_per_grp[0] = 4 * time_step;
    for (i = 0; i < time_step; i++) { 
        A[i] = w_x;                                       // w_ix
        A[i + time_step] = w_x + input_dim * hid;         // w_fx
        A[i + 2 * time_step] = w_x + 2 * input_dim * hid; // w_cx 
        A[i + 3 * time_step] = w_x + 3 * input_dim * hid; // w_ox 
    
        B[i] = dhi + i * bh; 
        B[i + time_step] = dhf + i * bh; 
        B[i + 2 * time_step] = dhc + i * bh; 
        B[i + 3 * time_step] = dho + i * bh; 
    
        C[i] = x_temp + i * ib;
        C[i + time_step] = x_temp + (i + time_step) * ib; 
        C[i + 2 * time_step] = x_temp + (i + 2 * time_step)* ib; 
        C[i + 3 * time_step] = x_temp + (i + 3 * time_step)* ib; 
    }
    cblas_sgemm_batch(CblasRowMajor, transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, 1, size_per_grp); 
    for(i = 0; i < tbi; i++) {
        dx[i] = x_temp[i] + x_temp[i + tbi] + x_temp[i + 2 * tbi] + x_temp[i + 3 * tbi];
    }
}
    int lstm_wx_training_create_instance(void* buf, lstm_handle* lstm_han, int input_dim, int hid, int max_time_step, int max_batch_size) { 
        if (max_time_step == 0) {
            max_time_step = 128;
        }
        if (max_batch_size == 0) {
            max_batch_size = 64;
        }
        lstm_han->A = (const float**) buf;
        lstm_han->B = (const float**) (buf + 4 * max_time_step * sizeof (float*));
        lstm_han->C = (float**) (buf + 8 * max_time_step * sizeof (float*));
        lstm_han->x_temp = (float*) (buf + 12 * max_time_step * sizeof (float*));
        lstm_han->hf = (float*) (buf + 12 * max_time_step * sizeof (float*) + max_time_step * 4 * max_batch_size * hid * sizeof (float));
        lstm_han->hi = (float*) (buf + 12 * max_time_step * sizeof (float*) + max_time_step * 5 * max_batch_size * hid * sizeof (float));
        lstm_han->hc = (float*) (buf + 12 * max_time_step * sizeof (float*) + max_time_step * 6 * max_batch_size * hid * sizeof (float));
        lstm_han->ho = (float*) (buf + 12 * max_time_step * sizeof (float*) + max_time_step * 7 * max_batch_size * hid * sizeof (float));
        lstm_han->c = (float*) (buf + 12 * max_time_step * sizeof (float*) + max_time_step * 8 * max_batch_size * hid * sizeof (float));
        lstm_han->h = (float*) (buf + 12 * max_time_step * sizeof (float*) + max_time_step * 9 * max_batch_size * hid * sizeof (float));
         
        //bp 
        lstm_han->dh = (float*)(buf + 12 * max_time_step * sizeof (float*) + max_time_step * 10 * max_batch_size * hid * sizeof (float)); 
        lstm_han->dc = (float*)(buf + 12 * max_time_step * sizeof (float*) + (max_time_step * 10 * max_batch_size * hid + hid * max_batch_size) * sizeof (float));
        lstm_han->dh_next = (float*)(buf + 12 * max_time_step * sizeof (float*) + (max_time_step * 10 * max_batch_size * hid + 2 * hid * max_batch_size) * sizeof (float)); 
        lstm_han->dc_next = (float*)(buf + 12 * max_time_step * sizeof (float*) + (max_time_step * 10 * max_batch_size * hid + 3 * hid * max_batch_size) * sizeof (float));
        lstm_han->dhf = (float*)(buf + 12 * max_time_step * sizeof (float*) + (max_time_step * 10 * max_batch_size * hid + 4 * hid * max_batch_size) * sizeof (float));
        lstm_han->dhi = (float*)(buf + 12 * max_time_step * sizeof (float*) + (max_time_step * 11 * max_batch_size * hid + 4 * hid * max_batch_size) * sizeof (float));
        lstm_han->dhc = (float*)(buf + 12 * max_time_step * sizeof (float*) + (max_time_step * 12 * max_batch_size * hid + 4 * hid * max_batch_size) * sizeof (float));
        lstm_han->dho = (float*)(buf + 12 * max_time_step * sizeof (float*) + (max_time_step * 13 * max_batch_size * hid + 4 * hid * max_batch_size) * sizeof (float));
        lstm_han->temp = (float*)(buf + 12 * max_time_step * sizeof (float*) + (max_time_step * 14 * max_batch_size * hid + 4 * hid * max_batch_size) * sizeof (float));
        return 0;
    }
    int lstm_wx_training(
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
        float* grad_last,//(H, N)
        float* dall){   //all gradients
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
        lstm_wx_training_create_instance(buf, lstm_han, input_dim, hid, max_time_step, max_batch_size);
        const float** A = lstm_han->A;
        const float** B = lstm_han->B;
        float** C = lstm_han->C;
        float* x_temp = lstm_han->x_temp;
        float* hf = lstm_han->hf;
        float* hi = lstm_han->hi;
        float* hc = lstm_han->hc;
        float* ho = lstm_han->ho;
        float* c = lstm_han->c;
        float* h = lstm_han->h;
        //bp
        float* dh = lstm_han->dh;
        float* dc = lstm_han->dc;
        float* dh_next = lstm_han->dh_next;
        float* dc_next = lstm_han->dc_next;
        float* dhf = lstm_han->dhf;
        float* dhi = lstm_han->dhi;
        float* dhc = lstm_han->dhc;
        float* dho = lstm_han->dho;
        float* temp = lstm_han->temp;
        //memset(A, 0, sizeof(float*) * (4 * max_time_step));
        //memset(B, 0, sizeof(float*) * (4 * max_time_step));
        //memset(C, 0, sizeof(float*) * (4 * max_time_step));
        //memset(x_temp, 0, sizeof(float*) * (max_time_step * 4 * max_batch_size * hid));
        //memset(hf, 0, sizeof(float*) * (max_time_step * max_batch_size * hid));
        //memset(hi, 0, sizeof(float*) * (max_time_step * max_batch_size * hid));
        //memset(hc, 0, sizeof(float*) * (max_time_step * max_batch_size * hid));
        //memset(ho, 0, sizeof(float*) * (max_time_step * max_batch_size * hid));
        //memset(c, 0, sizeof(float*) * (max_time_step * max_batch_size * hid));
        //memset(h, 0, sizeof(float*) * (max_time_step * max_batch_size * hid));
        LSTM_forward(batch_size, time_step, input_dim, hid,
                         w_x, w_h, b, x, h_0, c_0,
                         ho, hf, hi, hc, //hid * batch_size
                         c, h,//time_Step * hid * batch_size
                         A, B, C, x_temp);
        LSTM_backward(batch_size, time_step, input_dim, hid, 
                    w_x, w_h, b, x, h_0, c_0,//same with forward input
                    ho, hf, hi, hc, c, h,//forward output: time_step * hid * batch_size
                    grad_last,//gz: (H,N)
                    dall,
                    dall + hid * input_dim * 2,
                    dall + hid * input_dim,
                    dall + hid * input_dim * 3,//dwxi,f,c,o

                    dall + hid * input_dim * 4, 
                    dall + hid * input_dim * 4 + hid * hid * 2, 
                    dall + hid * input_dim * 4 + hid * hid, 
                    dall + hid * input_dim * 4 + hid * hid * 3,//dwhi,f,c,o

                    dall + hid * input_dim * 4 + hid * hid * 4, 
                    dall + hid * input_dim * 4 + hid * hid * 4 + hid * 2,
                    dall + hid * input_dim * 4 + hid * hid * 4 + hid, 
                    dall + hid * input_dim * 4 + hid * hid * 4 + hid * 3,//dbi,f,c,o 
                    dall + hid * input_dim * 4 + hid * hid * 4 + hid * 4,//dx
                    A, B, C, dh, dc, dh_next, dc_next, dhf, dhi, dhc, dho, temp);
                    //icfo    
        mkl_free(lstm_han);
        return 0;
    }
    int lstm_wx_train_get_workspace_size(int input_dim, int hid, int max_time_step, int max_batch_size)
    {
        int temp_size = max(4 * max_time_step * hid * max_batch_size, 4 * max_time_step * hid * input_dim);
        temp_size = max(temp_size, 4 * max_time_step * hid * hid);
        temp_size = max(temp_size, 4 * max_time_step * input_dim * max_batch_size);
                
        int max_ws = 12 * max_time_step * sizeof (float*) + (max_time_step * 14 * max_batch_size * hid + 4 * hid * max_batch_size + temp_size) * sizeof (float);
        return max_ws;
    }
}
