
import theano
from theano import tensor, gof
from theano.tensor import TensorType
from theano.tensor.blas import ldflags


class GRUGradients(gof.Op):
    __props__ = ('hid', 'step', 'dim', 'return_sequences', 'max_len', 'bias')

    def __init__(self, hid, step=None, dim=None, return_sequences=False, max_len=None, bias=False):
        self.hid = hid
        self.step = step
        self.dim = dim
        self.return_sequences = return_sequences
        self.max_len = max_len    # it's not needed in backward computation
        self.bias = bias
        super(GRUGradients, self).__init__()

    def make_node(self, X, Wx, Wh, hid, hid_init, zt, rt, hcan, hht, grads):
        """
        Parameters:
            X: All inputs, should be a 3D tensor with shape
               (time_step, batch_size, embed_dims)

            Wx: Weight for X which merges Wx_r, Wx_z and Wx_h together,
                2D tensor with shape (3 * embed_dims, hid_dims)

            Wh: Weight for hidden state, 2D tensor with shape (3 * hid_dims, hid_dims)

            hid: Ouput of GRU forward, 3D tensor with shape
                 (time_step, batch_size, hid_dims)

            hid_init: the initial hidden state

            zt: intermediate results of forward calculation

            rt: intermediate results of forward calculation

            hcan: intermediate results of forward calculation

            hht: intermediate results of forward calculation

            grads: gradient of next layer

        """
        inp = [X, Wx, Wh, hid, hid_init, zt, rt, hcan, hht, grads]
        inp = map(tensor.as_tensor_variable, inp)

        assert inp[0].type.ndim is 3
        assert inp[1].type.ndim is 2
        assert inp[2].type.ndim is 2
        assert inp[3].type.ndim is 3
        assert inp[4].type.ndim is 2
        assert inp[5].type.ndim is 3
        assert inp[6].type.ndim is 3
        assert inp[7].type.ndim is 3
        assert inp[8].type.ndim is 3

        out = [X.type(), Wx.type(), Wh.type(), hid_init.type()]
        if self.bias:
            out = out + [TensorType(dtype=Wx.type.dtype, broadcastable=(False,))()]
        return gof.Apply(self, inp, out)

    def c_headers(self):
        headers = ['<mkl.h>', '<omp.h>']
        return headers

    def c_libraries(self):
        return ldflags()

    def c_support_code_struct(self, node, name):
        if node.inputs[0].type.dtype == 'float32':
            dtype = 'float'
        elif node.inputs[0].type.dtype == 'float64':
            dtype = 'double'
        else:
            raise TypeError('GRUGradients: dtype %s is not supported.'
                            % (node.inputs[0].type.dtype))

        ccode = """
            size_t time_step;
            size_t batch_size;
            size_t embed_dims;
            size_t hid_dims;

            %(dtype)s* dhnext;

            %(dtype)s* d0;
            %(dtype)s* d1;
            %(dtype)s* d2;
            %(dtype)s* d3;
            %(dtype)s* d5;
            %(dtype)s* d7;
            %(dtype)s* d8;
            %(dtype)s* d10;
            %(dtype)s* d11;
            %(dtype)s* d14;
        """ % locals()
        return ccode

    def c_init_code_struct(self, node, name, sub):

        ccode = """
            time_step = 0;
            batch_size = 0;
            embed_dims = 0;
            hid_dims = 0;

            dhnext = NULL;

            d0 = NULL;
            d1 = NULL;
            d2 = NULL;
            d3 = NULL;
            d5 = NULL;
            d7 = NULL;
            d8 = NULL;
            d10 = NULL;
            d11 = NULL;
            d14 = NULL;
        """ % locals()
        return ccode

    def c_cleanup_code_struct(self, node, name):
        ccode = """
            if (dhnext) {
                mkl_free(dhnext);
                dhnext = NULL;
            }

            if (d0) {
                mkl_free(d0);
                d0 = NULL;
            }

            if (d1) {
                mkl_free(d1);
                d1 = NULL;
            }

            if (d2) {
                mkl_free(d2);
                d2 = NULL;
            }

            if (d3) {
                mkl_free(d3);
                d3 = NULL;
            }

            if (d5) {
                mkl_free(d5);
                d5 = NULL;
            }

            if (d7) {
                mkl_free(d7);
                d7 = NULL;
            }

            if (d8) {
                mkl_free(d8);
                d8 = NULL;
            }

            if (d10) {
                mkl_free(d10);
                d10 = NULL;
            }

            if (d11) {
                mkl_free(d11);
                d11 = NULL;
            }

            if (d14) {
                mkl_free(d14);
                d14 = NULL;
            }
        """
        return ccode

    def c_code(self, node, name, inputs, outputs, sub):
        X, Wx, Wh, hid_state, hid_init, zt, rt, hcan, hht, gz = inputs

        if self.bias:
            with_bias = 1
            gi, gwx, gwh, ghinit, gb = outputs
        else:
            with_bias = 0
            gi, gwx, gwh, ghinit = outputs

        hid = self.hid
        if self.return_sequences:
            return_sequences = 1
        else:
            return_sequences = 0

        if node.inputs[0].type.dtype == 'float32':
            dtype = 's'
            d = 'float'
        elif node.inputs[0].type.dtype == 'float64':
            dtype = 'd'
            d = 'double'
        else:
            raise TypeError('GRUGradients: dtype %s is not supported.'
                            % (node.inputs[0].type.dtype))

        ccode = """
            time_step  = PyArray_DIMS(%(X)s)[0];
            embed_dims = PyArray_DIMS(%(X)s)[1];
            batch_size = PyArray_DIMS(%(X)s)[2];

            hid_dims = PyArray_DIMS(%(hid_state)s)[1];
            assert (hid_dims == %(hid)s);

            %(d)s* x_ptr     = NULL;
            %(d)s* wx_ptr    = NULL;
            %(d)s* wh_ptr    = NULL;
            %(d)s* hid_ptr   = NULL;
            %(d)s* hinit_ptr = NULL;

            PyArrayObject* x_src     = NULL;
            PyArrayObject* wx_src    = NULL;
            PyArrayObject* wh_src    = NULL;
            PyArrayObject* hid_src   = NULL;
            PyArrayObject* hinit_src = NULL;

            vmlSetMode(vmlGetMode() & 0xFFFFFFF0 | VML_EP);

            if (!PyArray_IS_C_CONTIGUOUS(%(X)s)) {
                printf(\"Warning: GRUGradients need convert X to C-Contiguous\\n\");
                x_src = (PyArrayObject*)PyArray_ContiguousFromAny((PyObject*)%(X)s,
                                            PyArray_TYPE(%(X)s),
                                            PyArray_NDIM(%(X)s),
                                            PyArray_NDIM(%(X)s));
                if (!x_src) {
                    PyErr_SetString(PyExc_RuntimeError, \"GRUGradients: fail to cast X to contiguous array\");
                    goto gru_backward_fail;
                }
                x_ptr = (%(d)s*) PyArray_DATA(x_src);
            } else {
                x_ptr = (%(d)s*) PyArray_DATA(%(X)s);
            }

            if (!PyArray_IS_C_CONTIGUOUS(%(Wx)s)) {
                printf(\"Warning: GRUGradients need convert Wx to C-Contiguous\\n\");
                wx_src = (PyArrayObject*)PyArray_ContiguousFromAny((PyObject*)%(Wx)s,
                                            PyArray_TYPE(%(Wx)s),
                                            PyArray_NDIM(%(Wx)s),
                                            PyArray_NDIM(%(Wx)s));
                if (!wx_src) {
                    PyErr_SetString(PyExc_RuntimeError, \"GRUGradients: fail to cast Wx to contiguous array\");
                    goto gru_backward_fail;
                }
                wx_ptr = (%(d)s*) PyArray_DATA(wx_src);
            } else {
                wx_ptr = (%(d)s*) PyArray_DATA(%(Wx)s);
            }

            if (!PyArray_IS_C_CONTIGUOUS(%(Wh)s)) {
                printf(\"Warning: GRUGradients need convert Wh to C-Contiguous\\n\");
                wh_src = (PyArrayObject*)PyArray_ContiguousFromAny((PyObject*)%(Wh)s,
                                            PyArray_TYPE(%(Wh)s),
                                            PyArray_NDIM(%(Wh)s),
                                            PyArray_NDIM(%(Wh)s));
                if (!wh_src) {
                    PyErr_SetString(PyExc_RuntimeError, \"GRUGradients: fail to cast Wh to contiguous array\");
                    goto gru_backward_fail;
                }
                wh_ptr = (%(d)s*) PyArray_DATA(wh_src);
            } else {
                wh_ptr = (%(d)s*) PyArray_DATA(%(Wh)s);
            }

            if (!PyArray_IS_C_CONTIGUOUS(%(hid_state)s)) {
                printf(\"Warning: GRUGradients need convert hidden state to C-Contiguous\\n\");
                hid_src = (PyArrayObject*)PyArray_ContiguousFromAny((PyObject*)%(hid_state)s,
                                            PyArray_TYPE(%(hid_state)s),
                                            PyArray_NDIM(%(hid_state)s),
                                            PyArray_NDIM(%(hid_state)s));
                if (!hid_src) {
                    PyErr_SetString(PyExc_RuntimeError, \"GRUGradients: fail to cast hidden state to contiguous array\");
                    goto gru_backward_fail;
                }
                hid_ptr = (%(d)s*) PyArray_DATA(hid_src);
            } else {
                hid_ptr = (%(d)s*) PyArray_DATA(%(hid_state)s);
            }

            if (!PyArray_IS_C_CONTIGUOUS(%(hid_init)s)) {
                printf(\"Warning: GRUGradients need convert hid_init to C-Contiguous\\n\");
                hinit_src = (PyArrayObject*)PyArray_ContiguousFromAny((PyObject*)%(hid_init)s,
                                            PyArray_TYPE(%(hid_init)s),
                                            PyArray_NDIM(%(hid_init)s),
                                            PyArray_NDIM(%(hid_init)s));
                if (!hinit_src) {
                    PyErr_SetString(PyExc_RuntimeError, \"GRUGradients: fail to cast hid_init to contiguous array\");
                    goto gru_backward_fail;
                }
                hinit_ptr = (%(d)s*) PyArray_DATA(hinit_src);
            } else {
                hinit_ptr = (%(d)s*) PyArray_DATA(%(hid_init)s);
            }

            //// construct outputs
            npy_intp dims[3] = {0, 0, 0};
            if (%(gi)s == NULL || PyArray_NDIM(%(gi)s) != 3 ||
                PyArray_DIMS(%(gi)s)[0] != time_step ||
                PyArray_DIMS(%(gi)s)[1] != embed_dims ||
                PyArray_DIMS(%(gi)s)[2] != batch_size) {
                Py_XDECREF(%(gi)s);

                dims[0] = time_step;
                dims[1] = embed_dims;
                dims[2] = batch_size;
                %(gi)s = (PyArrayObject*) PyArray_ZEROS(3, dims, PyArray_TYPE(%(X)s), 0);
            }

            if (%(gwx)s == NULL || PyArray_NDIM(%(gwx)s) != 2 ||
                PyArray_DIMS(%(gwx)s)[0] != 3 * hid_dims ||
                PyArray_DIMS(%(gwx)s)[1] != embed_dims) {
                Py_XDECREF(%(gwx)s);

                dims[0] = 3 * hid_dims;
                dims[1] = embed_dims;
                %(gwx)s = (PyArrayObject*) PyArray_ZEROS(2, dims, PyArray_TYPE(%(Wx)s), 0);
            }

            if (%(gwh)s == NULL || PyArray_NDIM(%(gwh)s) != 2 ||
                PyArray_DIMS(%(gwh)s)[0] != 3 * hid_dims ||
                PyArray_DIMS(%(gwh)s)[1] != hid_dims) {
                Py_XDECREF(%(gwh)s);

                dims[0] = 3 * hid_dims;
                dims[1] = hid_dims;
                %(gwh)s = (PyArrayObject*) PyArray_ZEROS(2, dims, PyArray_TYPE(%(Wh)s), 0);
            }

            if (%(ghinit)s == NULL || PyArray_NDIM(%(ghinit)s) != 2 ||
                PyArray_DIMS(%(ghinit)s)[0] != hid_dims ||
                PyArray_DIMS(%(ghinit)s)[1] != batch_size) {
                Py_XDECREF(%(ghinit)s);

                dims[0] = hid_dims;
                dims[1] = batch_size;
                %(ghinit)s = (PyArrayObject*) PyArray_ZEROS(2, dims, PyArray_TYPE(%(hid_init)s), 0);
            }

            if (NULL == %(gi)s || NULL == %(gwx)s || NULL == %(gwh)s || NULL == %(ghinit)s) {
                PyErr_SetString(PyExc_RuntimeError, \"GRUGradients: create output array failed\");
                goto gru_backward_fail;
            }
            """ % locals()

        if self.bias:
            ccode += """
                if (%(gb)s == NULL || PyArray_NDIM(%(gb)s) != 1 ||
                    PyArray_DIMS(%(gb)s)[0] != 3 * hid_dims) {
                    Py_XDECREF(%(gb)s);

                    dims[0] = 3 * hid_dims;
                    %(gb)s = (PyArrayObject*) PyArray_ZEROS(1, dims, PyArray_TYPE(%(Wx)s), 0);
                }

                if (NULL == %(gb)s) {
                    PyErr_SetString(PyExc_RuntimeError, \"GRUGradients: create output array for gradient bias fasiled\");
                    goto gru_backward_fail;
                }

                %(d)s* gb_ptr = (%(d)s*) PyArray_DATA(%(gb)s);
                """ % locals()
        else:
            ccode += """
                // no gradient bias for this apply..
                %(d)s* gb_ptr = NULL;
                """ % locals()

        ccode += """

            if (NULL == d0) {
                d0 = (%(d)s*) mkl_malloc(batch_size * hid_dims * sizeof (%(d)s), 64);
            }

            if (NULL == d1) {
                d1 = (%(d)s*) mkl_malloc(batch_size * hid_dims * sizeof (%(d)s), 64);
            }

            if (NULL == d2) {
                d2 = (%(d)s*) mkl_malloc(batch_size * hid_dims * sizeof (%(d)s), 64);
            }

            if (NULL == d3) {
                d3 = (%(d)s*) mkl_malloc(batch_size * hid_dims * sizeof (%(d)s), 64);
            }

            if (NULL == d5) {
                d5 = (%(d)s*) mkl_malloc(batch_size * hid_dims * sizeof (%(d)s), 64);
            }

            if (NULL == d7) {
                d7 = (%(d)s*) mkl_malloc(batch_size * hid_dims * sizeof (%(d)s), 64);
            }

            if (NULL == d8) {
                d8 = (%(d)s*) mkl_malloc(batch_size * hid_dims * sizeof (%(d)s), 64);
            }

            if (NULL == d10) {
                d10 = (%(d)s*) mkl_malloc(batch_size * hid_dims * sizeof (%(d)s), 64);
            }

            if (NULL == d11) {
                d11 = (%(d)s*) mkl_malloc(batch_size * hid_dims * sizeof (%(d)s), 64);
            }

            if (NULL == d14) {
                d14 = (%(d)s*) mkl_malloc(batch_size * hid_dims * sizeof (%(d)s), 64);
            }

            if (NULL == dhnext) {
                dhnext = (%(d)s*) mkl_calloc(batch_size * hid_dims, sizeof (%(d)s), 64);
            }

            if (dhnext) {
                // memset((void*)dhnext, 0, batch_size * hid_dims * sizeof (%(d)s));
            } else {
                PyErr_SetString(PyExc_MemoryError, \"GRUGradients: create dhnext buffer failed\");
                goto gru_backward_fail;
            }

            %(d)s* gz_ptr   = (%(d)s*) PyArray_DATA(%(gz)s);
            %(d)s* zt_ptr   = (%(d)s*) PyArray_DATA(%(zt)s);
            %(d)s* rt_ptr   = (%(d)s*) PyArray_DATA(%(rt)s);
            %(d)s* hcan_ptr = (%(d)s*) PyArray_DATA(%(hcan)s);
            %(d)s* hht_ptr  = (%(d)s*) PyArray_DATA(%(hht)s);

            if (NULL == zt_ptr || NULL == rt_ptr || NULL == hcan_ptr || NULL == hht_ptr) {
                PyErr_SetString(PyExc_RuntimeError, \"GRUGradients: input workspace is NULL\");
                goto gru_backward_fail;
            }

            %(d)s* wxh_ptr = wx_ptr;
            %(d)s* wxz_ptr = wx_ptr + embed_dims * hid_dims;
            %(d)s* wxr_ptr = wx_ptr + 2 * embed_dims * hid_dims;

            %(d)s* whh_ptr = wh_ptr;
            %(d)s* whz_ptr = wh_ptr + hid_dims * hid_dims;
            %(d)s* whr_ptr = wh_ptr + 2 * hid_dims * hid_dims;

            // arguments for batch gemm
            %(d)s* A[6] = {NULL};
            %(d)s* B[6] = {NULL};
            %(d)s* C[6] = {NULL};

            MKL_INT m[2] = {hid_dims, hid_dims};
            MKL_INT k[2] = {batch_size, batch_size};
            MKL_INT n[2] = {embed_dims, hid_dims};

            MKL_INT lda[2] = {batch_size, batch_size};
            MKL_INT ldb[2] = {batch_size, batch_size};
            MKL_INT ldc[2] = {embed_dims, hid_dims};

            CBLAS_TRANSPOSE transA[2] = {CblasNoTrans, CblasNoTrans};
            CBLAS_TRANSPOSE transB[2] = {CblasTrans, CblasTrans};

            %(d)s alpha[2] = {1.0, 1.0};
            %(d)s beta[2] = {1.0, 1.0};

            MKL_INT size_per_grp[2] = {3, 3};

            //// step on time_step
            // loop on step, reverse
            size_t size_of_batch = batch_size * hid_dims;
            double tic = dsecnd();
            double acc1 = 0.0f;
            double acc2 = 0.0f;
            double acc3 = 0.0f;
            double acc4 = 0.0f;
            double toc;
            for (int i = time_step - 1; i >= 0; i--) {
                // dh = dy + dhnext
                toc = dsecnd();
                if (PyArray_NDIM(%(gz)s) == 3) {
                    v%(dtype)sAdd(size_of_batch, gz_ptr + i * size_of_batch, dhnext, dhnext);
                } else if (PyArray_NDIM(%(gz)s) == 2) {
                    if (i == time_step - 1) {
                        v%(dtype)sAdd(size_of_batch, gz_ptr, dhnext, dhnext);
                    }
                } else {
                    PyErr_SetString(PyExc_RuntimeError, \"GRUGradients: dimension of input gz is wrong\");
                    goto gru_backward_fail;
                }
                acc1 += dsecnd() - toc;

                toc = dsecnd();
                #pragma omp parallel for simd
                for (int t = 0; t < size_of_batch; t++) {
                    d1[t] = dhnext[t] * (zt_ptr + i * size_of_batch)[t];
                    d0[t] = dhnext[t] - d1[t];
                    d2[t] = d1[t] * (1.0f - (hcan_ptr + i * size_of_batch)[t] * (hcan_ptr + i * size_of_batch)[t]);
                    d3[t] = d2[t] * (rt_ptr + i * size_of_batch)[t];

                    // d5 = dh*h(t-1)
                    if (0 == i) {
                        d5[t] = 0;
                    } else {
                        d5[t] = dhnext[t] * (hid_ptr + (i - 1) * size_of_batch)[t];
                    }

                    d7[t] = dhnext[t] * (hcan_ptr + i * size_of_batch)[t];
                    d8[t] = d7[t] - d5[t];

                    d10[t] = d2[t] * (hht_ptr + i * size_of_batch)[t];
                    d11[t] = d10[t] * ((rt_ptr + i * size_of_batch)[t] - (rt_ptr + i * size_of_batch)[t] * (rt_ptr + i * size_of_batch)[t]);
                    d14[t] = d8[t] * ((zt_ptr + i * size_of_batch)[t] - (zt_ptr + i * size_of_batch)[t] * (zt_ptr + i * size_of_batch)[t]);
                }
                acc2 += dsecnd() - toc;

                // GEMM, dhnext(t-1), in-place add
                toc = dsecnd();
                memcpy((void*)dhnext, (void*)d0, size_of_batch * sizeof (%(d)s));
                cblas_%(dtype)sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, hid_dims, batch_size, hid_dims,
                                    1.0, whh_ptr, hid_dims, d3, batch_size, 1.0, dhnext, batch_size);
                cblas_%(dtype)sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, hid_dims, batch_size, hid_dims,
                                    1.0, whz_ptr, hid_dims, d14, batch_size, 1.0, dhnext, batch_size);
                cblas_%(dtype)sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, hid_dims, batch_size, hid_dims,
                                    1.0, whr_ptr, hid_dims, d11, batch_size, 1.0, dhnext, batch_size);
                // GEMM, dX(t)
                %(d)s* gi_ptr = (%(d)s*) PyArray_DATA(%(gi)s) + i * batch_size * embed_dims;
                #ifndef  _GRU_NO_DX_
                cblas_%(dtype)sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, embed_dims, batch_size, hid_dims,
                                    1.0, wxh_ptr, embed_dims, d2, batch_size, 0.0, gi_ptr, batch_size);
                cblas_%(dtype)sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, embed_dims, batch_size, hid_dims,
                                    1.0, wxz_ptr, embed_dims, d14, batch_size, 1.0, gi_ptr, batch_size);
                cblas_%(dtype)sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, embed_dims, batch_size, hid_dims,
                                    1.0, wxr_ptr, embed_dims, d11, batch_size, 1.0, gi_ptr, batch_size);
                #endif
                acc3 += dsecnd() - toc;

                // GEMM, dWx
                B[0] = x_ptr + i * batch_size * embed_dims;
                B[1] = B[0];
                B[2] = B[0];

                A[0] = d2;
                A[1] = d14;
                A[2] = d11;

                C[0] = (%(d)s*) PyArray_DATA(%(gwx)s);
                C[1] = C[0] + embed_dims * hid_dims;
                C[2] = C[0] + 2 * embed_dims * hid_dims;

                if (i > 0) {
                    B[3] = hid_ptr + (i - 1) * batch_size * hid_dims;
                } else {
                    B[3] = hinit_ptr;
                }

                B[4] = B[3];
                B[5] = B[3];

                A[3] = d3;
                A[4] = d14;
                A[5] = d11;

                C[3] = (%(d)s*) PyArray_DATA(%(gwh)s);
                C[4] = C[3] + hid_dims * hid_dims;
                C[5] = C[3] + 2 * hid_dims * hid_dims;

                toc = dsecnd();
                cblas_%(dtype)sgemm_batch(CblasRowMajor, transA, transB, m, n, k,
                                          alpha, A, lda, B, ldb, beta, C, ldc, 2, size_per_grp);
                acc4 += dsecnd() - toc;
                // gb, reduction to one column
                if (gb_ptr) {
                    #pragma omp parallel for
                    for (int h = 0; h < hid_dims; h++) {
                        for (int b = 0; b < batch_size; b++) {
                            gb_ptr[h] = gb_ptr[h] + (d2 + h * batch_size)[b];                   // for bh
                            gb_ptr[h + hid_dims] = gb_ptr[h] + (d14 + h * batch_size)[b];       // for bz
                            gb_ptr[h + 2 * hid_dims] = gb_ptr[h] + (d11 + h * batch_size)[b];   // for br
                        }
                    }
                }
            }
            // printf(\"step: %%.8f sec, %%.8f, %%.8f, %%.8f, %%.8f \\n\", dsecnd() - tic, acc1, acc2, acc3, acc4);

            // dhinit is dhnext(0)
            memcpy ((void*)PyArray_DATA(%(ghinit)s), (void*)dhnext, batch_size * hid_dims * sizeof (%(d)s));
            gru_backward_fail:
            Py_XDECREF(x_src);
            Py_XDECREF(wx_src);
            Py_XDECREF(wh_src);
            Py_XDECREF(hid_src);
            Py_XDECREF(hinit_src);
        """ % locals()
        return ccode

    def c_code_cache_version(self):
        return (1, 0, 1)

