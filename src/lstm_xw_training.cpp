#include <cstddef>                                                                                                                                                                                                
#include <mkl.h>
#include <omp.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>

extern "C" {
inline float sigmoid(float x){
    return 1.0f / (1.0f + exp(-x));
}

int lstm_xw_train_get_workspace_size(const int I, const int H, const int T, const int N, const int bidirectional) {
    int D = bidirectional ? 2 : 1;
    int forward_size = D * (T + 1) * N * 4 * H + 2 * D * N * H;
    // int backward_size = T * N * H * 9;
    int backward_size = D * (T + 1) * N * 4 * H + D * T * N * 4 * H + D * T * N * H;
    return forward_size > backward_size ? forward_size : backward_size;
}

//
// Reorder the initial hidden layer: (D, N, H) => (N, D, H).
// 
// We have to do this because LSTM's final result has a format of (T, N, D*H), 
// so (N, D, H) is more convienent in computation.
//
float *
reorder_hidden(const float *h0, float *dest, int d, int n, int h)
{
    float (*ori_h0)[n][h] = (float (*)[n][h])h0;
    float (*new_h0)[d][h] = (float (*)[d][h])dest;

    assert(new_h0 != NULL);

    int d_index, n_index, h_index;
    for (d_index = 0; d_index < d; d_index ++) {
        for (n_index = 0; n_index < n; n_index ++) {
            // this is the start address of a size "h" block, let's compute it's new address.
            float *current_location = (float *)(ori_h0[d_index][n_index]);
            float *new_location = (float *)(new_h0[n_index][d_index]);
            memcpy(new_location, current_location, h * sizeof(float));
        }
    }

    return (float *)new_h0;
}

int lstm_xw_sequential_forward(void* buf,
                               const int N,                         // batch size
                               const int T,                         // time_step(seq_len)
                               const int I,                         // input size
                               const int H,                         // hidden size
                               const int bidirectional,             // enable bidirectional
                               const float* x,                      // (T, N, I)
                               const float* h_0,                    // (D, N, H)
                               const float* c_0,                    // (D, N, H)
                               const float* wx,                     // (D, I, 4*H)      
                               const float* wh,                     // (D, H, 4*H)      
                               const float* bias,                   // (D * 4H),   because bx == bh, so just need one
                               float* h_out,                        // (T, N, H*D)
                               float* c_out                         // (T, N, H*D)
                               ) {
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
    MKL_INT gemm_m = T*N, gemm_n = 4*H, gemm_k = I; 
    MKL_INT lda = gemm_k, ldb = gemm_n, ldc = gemm_n; 
    MKL_INT num_directions = bidirectional ? 2 : 1;

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, gemm_m, gemm_n, gemm_k, 1.0f, x, lda, wx, ldb, 0.0f, (float*)buf, ldc); 

    // Format h0 and c0, for the ease of computation.
    const float *internal_h_0 = h_0;
    const float *internal_c_0 = c_0;
    if (bidirectional) {
        internal_h_0 = reorder_hidden(h_0, (float *)buf + 2 * (T + 1) * N * 4 * H, num_directions, N, H);
        internal_c_0 = reorder_hidden(c_0, (float *)buf + 2 * (T + 1) * N * 4 * H + num_directions * N * H, num_directions, N, H);
    }
    
    //
    // Prepare buffer for "forward" LSTM layer:
    //

    //[ix|gx|fx|ox] : [T*N, 4H]
    float (*ix)[gemm_n] = (float(*)[gemm_n])buf; 
    float (*fx)[gemm_n] = (float(*)[gemm_n])((float*)ix + H);
    float (*gx)[gemm_n] = (float(*)[gemm_n])((float*)fx + H); 
    float (*ox)[gemm_n] = (float(*)[gemm_n])((float*)gx + H);

    // [ih|fh|gh|oh]: [N, 4H]
    float (*ih)[gemm_n] = ix + gemm_m;
    float (*fh)[gemm_n] = fx + gemm_m;
    float (*gh)[gemm_n] = gx + gemm_m; 
    float (*oh)[gemm_n] = ox + gemm_m;
    float *h_buf = (float*)ih;

    const float *bi = bias;
    const float *bf = bi + H;
    const float *bg = bf + H;
    const float *bo = bg + H;
    const float *h_pre = internal_h_0;
    const float (*c_pre)[num_directions * H] = (const float(*)[num_directions * H])internal_c_0;
    float (*c)[num_directions * H] = (float(*)[num_directions * H])c_out;
    float (*h)[num_directions * H] = (float(*)[num_directions * H])h_out;

    //
    // Prepare buffer for "backward" LSTM layer:
    //

    float (*ix_reverse)[gemm_n] = NULL;
    float (*fx_reverse)[gemm_n] = NULL;
    float (*gx_reverse)[gemm_n] = NULL;
    float (*ox_reverse)[gemm_n] = NULL;
    float (*ih_reverse)[gemm_n] = NULL;
    float (*fh_reverse)[gemm_n] = NULL;
    float (*gh_reverse)[gemm_n] = NULL;
    float (*oh_reverse)[gemm_n] = NULL;
    float *h_reverse_buf = NULL;
    const float *bi_reverse = NULL;
    const float *bf_reverse = NULL;
    const float *bg_reverse = NULL;
    const float *bo_reverse = NULL;
    const float *h_pre_reverse = NULL;
    const float (*c_pre_reverse)[2 * H] = NULL;
    float (*c_reverse)[2 * H] = NULL;
    float (*h_reverse)[2 * H] = NULL;

    if (bidirectional) {
        ix_reverse = ih + N;
        fx_reverse = fh + N;
        gx_reverse = gh + N;
        ox_reverse = oh + N;

        ih_reverse = ix_reverse + gemm_m;
        fh_reverse = fx_reverse + gemm_m;
        gh_reverse = gx_reverse + gemm_m;
        oh_reverse = ox_reverse + gemm_m;

        h_reverse_buf = (float *)ih_reverse;

        bi_reverse = bi + 4 * H;
        bf_reverse = bf + 4 * H;
        bg_reverse = bg + 4 * H;
        bo_reverse = bo + 4 * H;

        h_pre_reverse = (float *)internal_h_0 + H;
        c_pre_reverse = (const float(*)[2 * H])((float *)internal_c_0 + H);

        c_reverse = (float (*)[2 * H])((float *)c + H);
        h_reverse = (float (*)[2 * H])((float *)h + H);
    }

    //
    // x * wx_reverse
    //
    const float *wx_reverse = NULL;
    const float *wh_reverse = NULL;
    if (bidirectional) {
        wx_reverse = wx + I * 4 * H;

        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, gemm_m, gemm_n, gemm_k, 1.0f, x, lda, wx_reverse, ldb, 0.0f, (float*)buf + (T + 1) * N * 4 * H, ldc);
        
        wh_reverse = wh + H * 4 * H;
    }



    int i = 0, j = 0, k = 0;

    if (bi != NULL) {
        
        // "Forward" (non-reversed) LSTM layer. 
        
        for (i = 0; i < gemm_m; i += N) {
            //h_t-1 * wh
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N, gemm_n, H, 1.0f, h_pre, num_directions * H, wh, ldb, 0.0f, h_buf, ldc);
            #pragma omp parallel for collapse(2)
            for (j = 0; j < N; ++j) {
                for (k = 0; k < H; ++k) {
                    ix[i+j][k] = sigmoid(ix[i+j][k] + ih[j][k] + bi[k]);
                    fx[i+j][k] = sigmoid(fx[i+j][k] + fh[j][k] + bf[k]);
                    gx[i+j][k] =    tanh(gx[i+j][k] + gh[j][k] + bg[k]);
                    ox[i+j][k] = sigmoid(ox[i+j][k] + oh[j][k] + bo[k]);
                    c[i+j][k] = c_pre[j][k] * fx[i+j][k] + ix[i+j][k] * gx[i+j][k];
                    h[i+j][k] = ox[i+j][k] * tanh(c[i+j][k]);
                }
            }
            c_pre = c + i;
            h_pre = (float*)(h + i);
        }
        
        // "Backward" (reversed) LSTM layer.

        if (bidirectional) {
            for (i = gemm_m - N; i >= 0; i -= N) {
                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N, gemm_n, H, 1.0f, h_pre_reverse, num_directions * H, wh_reverse, ldb, 0.0f, h_reverse_buf, ldc);
                #pragma omp parallel for collapse(2)
                for (j = 0; j < N; ++j) {
                    for (k = 0; k < H; ++k) {
                        ix_reverse[i+j][k] = sigmoid(ix_reverse[i+j][k] + ih_reverse[j][k] + bi_reverse[k]);
                        fx_reverse[i+j][k] = sigmoid(fx_reverse[i+j][k] + fh_reverse[j][k] + bf_reverse[k]);
                        gx_reverse[i+j][k] =    tanh(gx_reverse[i+j][k] + gh_reverse[j][k] + bg_reverse[k]);
                        ox_reverse[i+j][k] = sigmoid(ox_reverse[i+j][k] + oh_reverse[j][k] + bo_reverse[k]);
                        c_reverse[i+j][k] = c_pre_reverse[j][k] * fx_reverse[i+j][k] + ix_reverse[i+j][k] * gx_reverse[i+j][k];
                        h_reverse[i+j][k] = ox_reverse[i+j][k] * tanh(c_reverse[i+j][k]);
                    }
                }

                c_pre_reverse = c_reverse + i;
                h_pre_reverse = (float *)(h_reverse + i);
            }
        }
    } else {
        for (i = 0; i < gemm_m; i += N) {
            //h_t-1 * wh
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N, gemm_n, H, 1.0f, h_pre, num_directions * H, wh, ldb, 0.0f, h_buf, ldc);
            #pragma omp parallel for collapse(2)
            for (j = 0; j < N; ++j) {
                for (k = 0; k < H; ++k) {
                    ix[i+j][k] = sigmoid(ix[i+j][k] + ih[j][k]);
                    fx[i+j][k] = sigmoid(fx[i+j][k] + fh[j][k]);
                    gx[i+j][k] =    tanh(gx[i+j][k] + gh[j][k]);
                    ox[i+j][k] = sigmoid(ox[i+j][k] + oh[j][k]);
                    c[i+j][k] = c_pre[j][k] * fx[i+j][k] + ix[i+j][k] * gx[i+j][k];
                    h[i+j][k] = ox[i+j][k] * tanh(c[i+j][k]);
                }
            }
            c_pre = c + i;
            h_pre = (float*)(h + i);
        }

        if (bidirectional) {
            for (i = gemm_m - N; i >= 0; i -= N) {
                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N, gemm_n, H, 1.0f, h_pre_reverse, num_directions * H, wh_reverse, ldb, 0.0f, h_reverse_buf, ldc);
                #pragma omp parallel for collapse(2)
                for (j = 0; j < N; ++j) {
                    for (k = 0; k < H; ++k) {
                        ix_reverse[i+j][k] = sigmoid(ix_reverse[i+j][k] + ih_reverse[j][k]);
                        fx_reverse[i+j][k] = sigmoid(fx_reverse[i+j][k] + fh_reverse[j][k]);
                        gx_reverse[i+j][k] =    tanh(gx_reverse[i+j][k] + gh_reverse[j][k]);
                        ox_reverse[i+j][k] = sigmoid(ox_reverse[i+j][k] + oh_reverse[j][k]);
                        c_reverse[i+j][k] = c_pre_reverse[j][k] * fx_reverse[i+j][k] + ix_reverse[i+j][k] * gx_reverse[i+j][k];
                        h_reverse[i+j][k] = ox_reverse[i+j][k] * tanh(c_reverse[i+j][k]);
                    }
                }

                c_pre_reverse = c_reverse + i;
                h_pre_reverse = (float *)(h_reverse + i);
            }
        }
    }

    return 0;
}
int lstm_xw_sequential_backward(void* buf,
                                const int N,                // batch size
                                const int T,                // time_step(seq_len)
                                const int I,                // input size
                                const int H,                // hidden size
                                const int D,                // num of directions
                                const float* x,             // (T, N, I)
                                const float* h_0,           // (D, N, H)
                                const float* c_0,           // (D, N, H)
                                const float* wx,            // (D, I, 4*H)      
                                const float* wh,            // (D, H, 4*H)      
                                const float* h_state,       // (T, N, H*D)
                                const float* c_state,       // (T, N, H*D)
                                const float* grad_hy,       // (D, N, H)
                                const float* grad_cy,       // (D, N, H) 
                                float* dwx,                 // (D, I, 4*H)
                                float* dwh,                 // (D, H, 4*H)
                                float* db,                  // (D * 4H)
                                float* dx,                  // (T, N, I)
                                float* dh0,                 // (D, N, H)
                                float* dc0                  // (D, N, H)
                                ) {

#if ENABLE_OMP_SETTING
    #pragma omp parallel default(shared)
    {
        int ompTid = omp_get_thread_num();
        int numomp = omp_get_num_threads();
        int numprc = omp_get_num_procs();
        int ompmax = omp_get_max_threads();
        //printf("ompmax:%d\n", ompmax);
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
    MKL_INT gemm_m = T * N, gemm_n = 4 * H;
    memset(dwh, 0, sizeof(float) * D * H * gemm_n);
    if (db != NULL) {
        memset(db, 0, sizeof(float) * D * gemm_n);
    }
    memset(dh0, 0, sizeof(float) * D * N * H);

    //
    // Variables for bidirectional LSTM's "forward" path:
    //

    // [ix|gx]fx|ox]: (T*N, 4*H)
    float (*it)[gemm_n] = (float(*)[gemm_n])buf;
    float (*ft)[gemm_n] = (float(*)[gemm_n])((float*)it + H);
    float (*gt)[gemm_n] = (float(*)[gemm_n])((float*)ft + H);
    float (*ot)[gemm_n] = (float(*)[gemm_n])((float*)gt + H);

    // [cout] = [ct|ct_reverse]
    const float (*ct)[D * H] = (float(*)[D * H])c_state;
    // [hout] = [ht|ht_reverse]
    const float (*ht)[D * H] = (float(*)[D * H])h_state;

    //[di|dg|df|do] : size=[T*N, 4*H]
    float (*deta_i)[gemm_n] = (float(*)[gemm_n])((float *)buf + D * (T + 1) * N * 4 * H);
    float (*deta_f)[gemm_n] = (float(*)[gemm_n])((float *)deta_i + H);
    float (*deta_g)[gemm_n] = (float(*)[gemm_n])((float *)deta_f + H);
    float (*deta_o)[gemm_n] = (float(*)[gemm_n])((float *)deta_g + H);

    //dc:[T*N, H]
    float (*deta_c)[H] = (float(*)[H])(deta_i + gemm_m);

    //dh: (N, H)
    float (*deta_h)[H] = (float(*)[H])dh0;



    //
    // Variables for bidirectional LSTM's "backward(reverse)" path:
    //

    // [ix_reverse|gx_reverse|fx_reverse|ox_reverse]: (T*N, 4*H)
    float (*it_reverse)[gemm_n] = NULL;
    float (*ft_reverse)[gemm_n] = NULL;
    float (*gt_reverse)[gemm_n] = NULL;
    float (*ot_reverse)[gemm_n] = NULL;

    const float (*ct_reverse)[D * H] = NULL;
    const float (*ht_reverse)[D * H] = NULL;

    float (*deta_i_reverse)[gemm_n] = NULL;
    float (*deta_f_reverse)[gemm_n] = NULL;
    float (*deta_g_reverse)[gemm_n] = NULL;
    float (*deta_o_reverse)[gemm_n] = NULL;

    float (*deta_c_reverse)[H] = NULL;
    float (*deta_h_reverse)[H] = NULL;

    if (D == 2) {
        it_reverse = (float(*)[gemm_n])((float *)buf + (T + 1) * N * 4 * H);
        ft_reverse = (float(*)[gemm_n])((float *)it_reverse + H);
        gt_reverse = (float(*)[gemm_n])((float *)ft_reverse + H);
        ot_reverse = (float(*)[gemm_n])((float *)gt_reverse + H);

        deta_i_reverse = deta_i + gemm_m;
        deta_f_reverse = deta_f + gemm_m;
        deta_g_reverse = deta_g + gemm_m;
        deta_o_reverse = deta_o + gemm_m;

        ct_reverse = (float(*)[D * H])(c_state + H);
        ht_reverse = (float(*)[D * H])(h_state + H);

        deta_c_reverse = (float (*)[H])(deta_i_reverse + gemm_m);
        deta_h_reverse = (float (*)[H])(dh0 + N * H);

    }

    //
    // Compute derivation of bidirectional LSTM's "forward" path:
    //

    int i = 0, j = 0, k = 0;
    
    float tc = 0.0f;                                                                                                                                                                                              
    for (i = gemm_m - N; i >= 0; i -= N) {
        #pragma omp parallel for collapse(2)
        for (j = 0; j < N; ++j) {
            for (k = 0; k < H; ++k) {
                tc = tanh(ct[i+j][k]);
                if (i == gemm_m - N) {
                    deta_h[j][k] = grad_hy[j*H+k];
                    deta_c[i+j][k] = deta_h[j][k] * ot[i+j][k] * (1 - tc * tc) + grad_cy[j*H+k];
                }
                else {
                    deta_c[i+j][k] = deta_h[j][k] * ot[i+j][k] * (1 - tc * tc) + deta_c[i+j+N][k] * ft[i+j+N][k];
                }
                deta_i[i+j][k] = deta_c[i+j][k] * gt[i+j][k] * it[i+j][k] * (1 - it[i+j][k]);
                deta_g[i+j][k] = deta_c[i+j][k] * it[i+j][k] * (1 - gt[i+j][k] * gt[i+j][k]);
                deta_o[i+j][k] = deta_h[j][k] * tc * ot[i+j][k] * (1 - ot[i+j][k]);
                if (i != 0) {
                    deta_f[i+j][k] = deta_c[i+j][k] * ct[i+j-N][k] * ft[i+j][k] * (1 - ft[i+j][k]);
                }
                else {
                    dc0[j*H + k] = deta_c[j][k] * ft[j][k];
                    deta_f[j][k] = deta_c[j][k] * c_0[j*H+k] * ft[j][k] * (1 - ft[j][k]);
                }
            }
        }

        if (i != 0) {
            cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, H, gemm_n, N, 1.0f, (float*)(ht+i-N), D * H, (float*)(deta_i+i), gemm_n, 1.0f, dwh, gemm_n);
        }
        else {
            cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, H, gemm_n, N, 1.0f, h_0, H, (float*)(deta_i+i), gemm_n, 1.0f, dwh, gemm_n);
        }

        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, N, H, gemm_n, 1.0f, (float*)(deta_i+i), gemm_n, wh, gemm_n, 0.0f, dh0, H);
    }

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, gemm_m, I, gemm_n, 1.0f, (float*)(deta_i), gemm_n, wx, gemm_n, 0.0f, dx, I);
    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, I, gemm_n, gemm_m, 1.0f, x, I, (float*)(deta_i), gemm_n, 0.0f, dwx, gemm_n);
    
    if (db != NULL)
    {
        for(i = 0; i < gemm_m; ++i) {
            for (j = 0; j < gemm_n; ++j) {
                db[j] += deta_i[i][j];
            }
        }
    }


    //
    // Compute derivation of bidirectional LSTM's "backward(reverse)" path:
    //

    if (D == 2) {
        for (i = 0; i < gemm_m; i += N) {
            #pragma omp parallel for collapse(2)
            for (j = 0; j < N; j ++) {
                for (k = 0; k < H; k ++) {
                    tc = tanh(ct_reverse[i + j][k]);
                    if (i == 0) {
                        deta_h_reverse[j][k] = grad_hy[N * H + j * H + k];
                        deta_c_reverse[i + j][k] = deta_h_reverse[j][k] * ot_reverse[i + j][k] * (1 - tc * tc) + grad_cy[N * H + j * H + k];
                    }
                    else {
                        deta_c_reverse[i+j][k] = deta_h_reverse[j][k] * ot_reverse[i + j][k] * (1 - tc * tc) + deta_c_reverse[i + j - N][k] * ft_reverse[i + j - N][k];
                    }
                    deta_i_reverse[i + j][k] = deta_c_reverse[i + j][k] * gt_reverse[i + j][k] * it_reverse[i + j][k] * (1 - it_reverse[i + j][k]); 
                    deta_g_reverse[i + j][k] = deta_c_reverse[i + j][k] * it_reverse[i + j][k] * (1 - gt_reverse[i + j][k] * gt_reverse[i + j][k]);
                    deta_o_reverse[i + j][k] = deta_h_reverse[j][k] * tc * ot_reverse[i + j][k] * (1 - ot_reverse[i + j][k]);
                    if (i != gemm_m - N) {
                        deta_f_reverse[i + j][k] = deta_c_reverse[i + j][k] * ct_reverse[i + j + N][k] * ft_reverse[i + j][k] * (1 - ft_reverse[i + j][k]);
                    }
                    else {
                        dc0[N*H + j*H + k] = deta_c_reverse[i + j][k] * ft_reverse[i + j][k];
                        deta_f_reverse[i + j][k] = deta_c_reverse[i + j][k] * c_0[N*H + j*H + k] * ft_reverse[i + j][k] * (1 - ft_reverse[i + j][k]);
                    }
                }
            }

            if (i != gemm_m - N) {
                cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, H, gemm_n, N, 1.0f, (float*)(ht_reverse+i+N), D * H, (float*)(deta_i_reverse+i), gemm_n, 1.0f, dwh + H * 4 * H, gemm_n);
            } else {
                cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, H, gemm_n, N, 1.0f, h_0 + N * H, H, (float*)(deta_i_reverse+i), gemm_n, 1.0f, dwh + H * 4 * H, gemm_n);
            }
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, N, H, gemm_n, 1.0f, (float*)(deta_i_reverse + i), gemm_n, wh + H * 4 * H, gemm_n, 0.0f, dh0 + N * H, H);
    
        }

        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, gemm_m, I, gemm_n, 1.0f, (float*)(deta_i_reverse), gemm_n, wx + I * 4 * H, gemm_n, 1.0f, dx, I);
        cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, I, gemm_n, gemm_m, 1.0f, x, I, (float*)(deta_i_reverse), gemm_n, 0.0f, dwx + I * 4 * H, gemm_n);
    
        if (db != NULL)
        {
            float *db_reverse = db + 4 * H;
            for(i = 0; i < gemm_m; ++i) {
                for (j = 0; j < gemm_n; ++j) {
                    db_reverse[j] += deta_i_reverse[i][j];
                }
            }
        }
    }

    
    
    
    return 0;
}
int lstm_xw_forward(void* buf,
                    int batch_size,
                    int time_step,
                    int input_dim,
                    int hidden_dim,
                    float* wx,      //I*4H      
                    float* wh,      //H*4H      
                    float* bias,    //4H, because bx == bh, so just need one
                    float* x, //T*N*I
                    float* h_0,     //N*H
                    float* c_0,     //N*H
                    float* h_out,    //T*N*H
                    float* c_out,   //T*N*H
                    int mode,
                    int bidirectional) {
    switch(mode) {
        case 0:
            lstm_xw_sequential_forward(buf, batch_size, time_step, input_dim, hidden_dim, bidirectional, x, h_0, c_0, wx, wh, bias, h_out, c_out);
            break;
        default:
            lstm_xw_sequential_forward(buf, batch_size, time_step, input_dim, hidden_dim, bidirectional, x, h_0, c_0, wx, wh, bias, h_out, c_out);
    }
    return 0;
}

int lstm_xw_backward(
        void* buf,
        int num_layer,
        int num_direction,
        int time_step,
        int batch_size,
        int input_dim,
        int hidden_dim,
        float * x,
        float * h_0,
        float * c_0,
        float * wx,
        float * wh,
        float * h_out,
        float * c_out,
        float * grad_h_out,     //(D, N, H)
        float * grad_c_out,     //(D, N, H)
        float * grad_x_input,   //(T, N, I)
        float * grad_h0,        //(D, N, H)
        float * grad_c0,        //(D, N, H)
        float * grad_wx,        //(D, I, 4H
        float * grad_wh,        //(D, H, 4H)
        float * grad_bias,       //(D, 4H)
        int mode) {
    switch(mode) {
        case 0:
            lstm_xw_sequential_backward(buf, batch_size, time_step, input_dim, hidden_dim, num_direction, x, h_0, c_0, wx, wh, h_out, c_out, 
                                        grad_h_out, grad_c_out, grad_wx, grad_wh, grad_bias, grad_x_input, grad_h0, grad_c0);
            break;
        default:
            lstm_xw_sequential_backward(buf, batch_size, time_step, input_dim, hidden_dim, num_direction, x, h_0, c_0, wx, wh, h_out, c_out, 
                                        grad_h_out, grad_c_out, grad_wx, grad_wh, grad_bias, grad_x_input, grad_h0, grad_c0);
    }
    return 0;
}

}
