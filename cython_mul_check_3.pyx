from __future__ import division
import numpy as np
import math
cimport numpy as np
cimport cython
DTYPE = np.float64

ctypedef np.float64_t DTYPE_t

cdef inline double sigmoid (x):
    #sigmoid activation
    
    val = 1.0/float((1+np.exp(-x)))
    return val

cdef inline double tanh(x):
    #tanh activation
    return (math.tanh(x))

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def forward_propagation_multidimension(np.ndarray[DTYPE_t,ndim=3] X, np.ndarray[DTYPE_t,ndim=2] W_xi, np.ndarray[DTYPE_t,ndim=2] W_xf, np.ndarray[DTYPE_t,ndim=2] \
                                    W_xo, np.ndarray[DTYPE_t,ndim=2] W_xg, np.ndarray [DTYPE_t,ndim=2]W_hi_d1, np.ndarray[DTYPE_t,ndim=2] W_hf_d1, np.ndarray[DTYPE_t,ndim=2] \
                                    W_ho_d1, np.ndarray[DTYPE_t,ndim=2] W_hg_d1, np.ndarray[DTYPE_t,ndim=2] W_hi_d2, np.ndarray[DTYPE_t,ndim=2] W_hf_d2, np.ndarray[DTYPE_t,ndim=2] W_ho_d2, \
                                    np.ndarray[DTYPE_t,ndim=2] W_hg_d2, np.ndarray[DTYPE_t,ndim=1] b_i, np.ndarray[DTYPE_t,ndim=1] b_f_d1, np.ndarray[DTYPE_t,ndim=1] b_f_d2,  \
                                       np.ndarray[DTYPE_t,ndim=1] b_o, np.ndarray[DTYPE_t,ndim=1] b_g , np.ndarray[DTYPE_t,ndim=3] h, np.ndarray[DTYPE_t,ndim=3] arr_i,np.ndarray[DTYPE_t,ndim=3] arr_f_d1,\
                                       np.ndarray[DTYPE_t,ndim=3] arr_f_d2, np.ndarray[DTYPE_t,ndim=3] arr_o ,np.ndarray[DTYPE_t,ndim=3] arr_g, np.ndarray[DTYPE_t,ndim=3] cell_state):

    
    #Performing Forward propagation for multidimension LSTM
    
    assert X.dtype == DTYPE and  W_xi.dtype == DTYPE   and W_xf.dtype == DTYPE   and W_xo.dtype == DTYPE   and W_xg.dtype == DTYPE   and W_hi_d1.dtype == DTYPE  \
        and  W_hf_d1.dtype == DTYPE   and W_ho_d1.dtype == DTYPE   and W_hg_d1.dtype == DTYPE   and W_hi_d2.dtype == DTYPE   and W_hf_d2.dtype == DTYPE   and \
        W_ho_d2.dtype == DTYPE   and W_hg_d2.dtype == DTYPE   and b_i.dtype == DTYPE   and b_f_d1.dtype == DTYPE  and b_f_d2.dtype == DTYPE   and b_o.dtype == DTYPE and \
        b_g.dtype == DTYPE and h.dtype == DTYPE  

    cdef int T_d1 = X.shape[1]    # along Y direction 
    cdef int T_d2 = X.shape[2]   # along x direction
    cdef int rows =   W_xi.shape[0]
    cdef int cols = W_xi.shape[1]
    cdef int rows_h = W_hf_d1.shape[0]
    cdef int t_d1
    cdef int t_d2
    cdef int i
    cdef int j
    cdef int k
    cdef DTYPE_t value_i
    cdef DTYPE_t value_f1
    cdef DTYPE_t value_f2
    cdef DTYPE_t value_o
    cdef DTYPE_t value_g
    cdef int x_inp = X.shape[0]
    cdef int h_inp = h.shape[0]
    for t_d2 in range(1,T_d2):
        for t_d1 in range(1,T_d1):
            for i in range(cols):
                value_i= 0
                value_f1 = 0
                value_f2=0
                value_o = 0
                value_g = 0
                for j in xrange(rows):
                    
                    value_i +=  X[j,t_d1,t_d2] * W_xi[j,i] 
                    value_f1 += X[j,t_d1,t_d2] * W_xf[j,i]
                    value_f2 += X[j,t_d1,t_d2] * W_xf[j,i] 
                    value_o += X[j,t_d1,t_d2] * W_xo[j,i] 
                    value_g += X[j,t_d1,t_d2] * W_xg[j,i] 

                for k in xrange(rows_h):
                    
                    value_i +=  h[k,t_d1-1,t_d2]*W_hi_d1[k,i] + h[k,t_d1,t_d2-1] * W_hi_d2[k,i]
                    value_f1 +=  h[k,t_d1-1,t_d2] * W_hf_d1[k,i]
                    value_f2 += h[k,t_d1,t_d2-1] * W_hf_d2[k,i]
                    value_o += h[k,t_d1-1,t_d2] * W_ho_d1[k,i] + h[k,t_d1,t_d2-1] * W_ho_d2[k,i]
                    value_g += h[k,t_d1-1,t_d2] * W_hg_d1[k,i] + h[k,t_d1,t_d2-1] * W_hg_d2[k,i]

                value_i += b_i[i]
                value_f1+= b_f_d1[i]
                value_f2 += b_f_d2[i]
                value_o += b_o[i]
                value_g += b_g[i]
                arr_i[i,t_d1,t_d2] = value_i 
                arr_f_d1[i,t_d1,t_d2] = value_f1
                arr_f_d2[i,t_d1,t_d2] = value_f2
                arr_o[i,t_d1,t_d2] =  value_o 
                arr_g[i,t_d1,t_d2] = value_g
                cell_state[i,t_d1,t_d2] = sigmoid(value_i) * tanh(value_g)  +  sigmoid(value_f1) * cell_state[i,t_d1-1,t_d2] + sigmoid(value_f2) * cell_state[i,t_d1,t_d2-1]
                h[i,t_d1,t_d2] = sigmoid(value_o) * tanh(cell_state[i,t_d1,t_d2])

    return h,X,cell_state,arr_i,arr_f_d1,arr_f_d2,arr_o,arr_g


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def backward_propagation_multidimension(np.ndarray[DTYPE_t,ndim=3] dout, np.ndarray[DTYPE_t,ndim=3] X,np.ndarray[DTYPE_t,ndim=3] h,np.ndarray[DTYPE_t,ndim=3] cell_state, \
                                        np.ndarray[DTYPE_t,ndim=3] arr_i, np.ndarray[DTYPE_t,ndim=3] arr_f_back_pass_d1, np.ndarray[DTYPE_t,ndim=3] arr_f_back_pass_d2, np.ndarray[DTYPE_t,ndim=3] arr_o,\
                                        np.ndarray[DTYPE_t,ndim=3] arr_g,np.ndarray[DTYPE_t,ndim=2] W_xi, np.ndarray[DTYPE_t,ndim=2] W_xf, np.ndarray[DTYPE_t,ndim=2] W_xo, np.ndarray[DTYPE_t,ndim=2] W_xg, \
                                        np.ndarray [DTYPE_t,ndim=2]W_hi_d1, np.ndarray[DTYPE_t,ndim=2] W_hf_d1, np.ndarray[DTYPE_t,ndim=2] W_ho_d1, np.ndarray[DTYPE_t,ndim=2] W_hg_d1, \
                                        np.ndarray[DTYPE_t,ndim=2] W_hi_d2, np.ndarray[DTYPE_t,ndim=2] W_hf_d2, np.ndarray[DTYPE_t,ndim=2] W_ho_d2, np.ndarray[DTYPE_t,ndim=2] W_hg_d2, np.ndarray[DTYPE_t,ndim=1] b_i, \
                                        np.ndarray[DTYPE_t,ndim=1] b_f_d1, np.ndarray[DTYPE_t,ndim=1] b_f_d2,  np.ndarray[DTYPE_t,ndim=1] b_o, np.ndarray[DTYPE_t,ndim=1] b_g,np.ndarray[DTYPE_t,ndim=3] dout_ec,\
                                        np.ndarray[DTYPE_t,ndim=4] dout_h, np.ndarray[DTYPE_t,ndim=3] dout_es, np.ndarray[DTYPE_t,ndim=2] dW_xi, np.ndarray[DTYPE_t,ndim=2] dW_xf, np.ndarray[DTYPE_t,ndim=2] dW_xo,\
                                        np.ndarray[DTYPE_t,ndim=2] dW_xg, np.ndarray[DTYPE_t,ndim=2] dW_hi_d1, np.ndarray[DTYPE_t,ndim=2] dW_hf_d1, np.ndarray[DTYPE_t,ndim=2] dW_ho_d1,\
                                        np.ndarray[DTYPE_t,ndim=2] dW_hg_d1,  np.ndarray[DTYPE_t,ndim=2] dW_hi_d2, np.ndarray[DTYPE_t,ndim=2] dW_hf_d2, np.ndarray[DTYPE_t,ndim=2] dW_ho_d2,\
                                        np.ndarray[DTYPE_t,ndim=2] dW_hg_d2, np.ndarray[DTYPE_t,ndim=1] db_i, np.ndarray[DTYPE_t,ndim=1] db_f_d1, np.ndarray[DTYPE_t,ndim=1] db_f_d2,\
                                        np.ndarray[DTYPE_t,ndim=1] db_o, np.ndarray[DTYPE_t,ndim=1] db_g, np.ndarray[DTYPE_t,ndim=3] dx):


    
    #Backward propagation for MDLSTM
    

    assert X.dtype == DTYPE and  W_xi.dtype == DTYPE   and W_xf.dtype == DTYPE   and W_xo.dtype == DTYPE   and W_xg.dtype == DTYPE   and W_hi_d1.dtype == DTYPE  \
    and  W_hf_d1.dtype == DTYPE   and W_ho_d1.dtype == DTYPE   and W_hg_d1.dtype == DTYPE   and W_hi_d2.dtype == DTYPE   and W_hf_d2.dtype == DTYPE   and \
    W_ho_d2.dtype == DTYPE   and W_hg_d2.dtype == DTYPE   and b_i.dtype == DTYPE   and b_f_d1.dtype == DTYPE  and b_f_d2.dtype == DTYPE   and b_o.dtype == DTYPE and \
    b_g.dtype == DTYPE and h.dtype == DTYPE and dout.dtype == DTYPE  and   cell_state.dtype == DTYPE  and arr_i.dtype == DTYPE  and arr_f_back_pass_d1.dtype == DTYPE and  \
    arr_f_back_pass_d2.dtype == DTYPE  and  arr_o.dtype == DTYPE  and arr_g.dtype == DTYPE 

    cdef int T_d1 = dout.shape[1] 
    cdef int T_d2 = dout.shape[2]
    cdef int n_gates = 5
    cdef int f_h = W_xi.shape[0]
    cdef int f_w = W_xi.shape[1]
    cdef int f_h_hidden = W_hi_d1.shape[0]
    cdef int f_w_hidden = W_hi_d1.shape[1]
    cdef int rows = f_h_hidden
    cdef int cols = f_w_hidden
    cdef int inp_C = X.shape[0]
    cdef int t_d2,t_d1,i,j,k,iter_h
    cdef DTYPE_t v_dout
    
                       
    for t_d2 in reversed(xrange(1,T_d2)):
        for t_d1 in reversed(xrange(1,T_d1)):
            for i in xrange(rows):
                v_dout = 0
                for j in xrange(cols):
                    v_dout += W_hi_d1[i,j] * dout_h[j,t_d1+1,t_d2,0]  + W_hi_d2[i,j] * dout_h[j,t_d1,t_d2+1,0]  + W_hf_d1[i,j] * dout_h[j,t_d1+1,t_d2,1]+ W_hf_d2[i,j] * dout_h[j,t_d1,t_d2+1,2] + W_ho_d1[i,j] * dout_h[j,t_d1+1,t_d2,3] +W_ho_d2[i,j] * dout_h[j,t_d1,t_d2+1,3]+ W_hg_d1[i,j] * dout_h[j,t_d1+1,t_d2,4] + W_hg_d2[i,j] * dout_h[j,t_d1,t_d2+1,4]

                dout_ec[i,t_d1,t_d2] = v_dout + dout[i,t_d1,t_d2]   
                dout_h[i,t_d1,t_d2,3] = (dout_ec[i,t_d1,t_d2] * tanh(cell_state[i,t_d1,t_d2])) * sigmoid(arr_o[i,t_d1,t_d2]) * (1-sigmoid(arr_o[i,t_d1,t_d2]))
                
                dout_es[i,t_d1,t_d2] = sigmoid(arr_o[i,t_d1,t_d2]) * (1-(tanh(cell_state[i,t_d1,t_d2]) * tanh(cell_state[i,t_d1,t_d2]))) * dout_ec[i,t_d1,t_d2]\
 + (dout_es[i,t_d1+1,t_d2] * sigmoid(arr_f_back_pass_d1[i,t_d1+1,t_d2]))\
 + dout_es[i,t_d1,t_d2+1] * sigmoid(arr_f_back_pass_d2[i,t_d1,t_d2+1])
                
                dout_h[i,t_d1,t_d2,1] = dout_es[i,t_d1,t_d2] * cell_state[i,t_d1-1,t_d2] * (sigmoid(arr_f_back_pass_d1[i,t_d1,t_d2])*(1-sigmoid(arr_f_back_pass_d1[i,t_d1,t_d2])))

                dout_h[i,t_d1,t_d2,2] = dout_es[i,t_d1,t_d2] *  cell_state[i,t_d1,t_d2-1] * (sigmoid(arr_f_back_pass_d2[i,t_d1,t_d2]) * (1-sigmoid(arr_f_back_pass_d2[i,t_d1,t_d2])))

                dout_h[i,t_d1,t_d2,0] = dout_es[i,t_d1,t_d2] * tanh(arr_g[i,t_d1,t_d2]) * (sigmoid(arr_i[i,t_d1,t_d2])*(1-sigmoid(arr_i[i,t_d1,t_d2])))

                dout_h[i,t_d1,t_d2,4] = dout_es[i,t_d1,t_d2] * sigmoid(arr_i[i,t_d1,t_d2])  * (1-(tanh(arr_g[i,t_d1,t_d2]) * tanh(arr_g[i,t_d1,t_d2])) )               

                for k in xrange(inp_C):
                    dW_xi[k,i] += X[k,t_d1,t_d2] * dout_h[i,t_d1,t_d2,0]
                    dW_xf[k,i] += (X[k,t_d1,t_d2] * dout_h[i,t_d1,t_d2,1] + X[k,t_d1,t_d2] * dout_h[i,t_d1,t_d2,2])
                    dW_xo[k,i] += X[k,t_d1,t_d2] * dout_h[i,t_d1,t_d2,3]
                    dW_xg[k,i] +=X[k,t_d1,t_d2] * dout_h[i,t_d1,t_d2,4]
                    dx[k,t_d1,t_d2] += W_xi[k,i] * dout_h[i,t_d1,t_d2,0] + W_xf[k,i] * dout_h[i,t_d1,t_d2,1] + W_xf[k,i] * dout_h[i,t_d1,t_d2,2]+W_xo[k,i] * dout_h[i,t_d1,t_d2,3] + W_xg[k,i] * dout_h[i,t_d1,t_d2,4]

                for iter_h in xrange(cols):
                    dW_hi_d1[iter_h,i] += h[iter_h,t_d1-1,t_d2] * dout_h[i,t_d1,t_d2,0]
                    dW_hi_d2[iter_h,i] +=h[iter_h,t_d1,t_d2-1]* dout_h[i,t_d1,t_d2,0] 
                    dW_hf_d1[iter_h,i] += h[iter_h,t_d1-1,t_d2] * dout_h[i,t_d1,t_d2,1]
                    dW_hf_d2[iter_h,i] +=  h[iter_h,t_d1,t_d2-1] * dout_h[i,t_d1,t_d2,2]
                    dW_ho_d1[iter_h,i] +=  h[iter_h,t_d1-1,t_d2] * dout_h[i,t_d1,t_d2,3]
                    dW_ho_d2[iter_h,i] += h[iter_h,t_d1,t_d2-1] * dout_h[i,t_d1,t_d2,3] 
                    dW_hg_d1[iter_h,i] +=   h[iter_h,t_d1-1,t_d2]* dout_h[i,t_d1,t_d2,4]
                    dW_hg_d2[iter_h,i] += h[iter_h,t_d1,t_d2-1] * dout_h[i,t_d1,t_d2,4]
                db_i[i] += dout_h[i,t_d1,t_d2,0] 
                db_f_d1[i] +=dout_h[i,t_d1,t_d2,1]
                db_f_d2[i] += dout_h[i,t_d1,t_d2,2]    
                db_o[i] +=dout_h[i,t_d1,t_d2,3]
                db_g[i] +=dout_h[i,t_d1,t_d2,4]          

    return dx,dW_xi,dW_xf,dW_xo,dW_xg,dW_hi_d1,dW_hi_d2,dW_hf_d1,dW_hf_d2,dW_ho_d1,dW_ho_d2,dW_hg_d1,dW_hg_d2,db_i,db_f_d1,db_f_d2,db_o,db_g


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def get_input_block(np.ndarray[DTYPE_t,ndim=4] x_new, np.ndarray[DTYPE_t,ndim=3]  img_inp_in_block,int width, int height,int filter_W,int filter_H):
    
    #creating input blocks for feeding it to MDLSTM layers
    
    cdef int N = x_new.shape[0]
    cdef int C = x_new.shape[1]
    cdef int WW = width
    cdef int HH = height
    cdef int row_traverse = 0
    cdef int column_traverse = 0
    cdef int flag =1
    cdef int count = 0
    cdef int i , iter_ch ,iter_row,iter_col,row_count, column_count
 
    for i in xrange(N):
        row_traverse = 0
        flag =1
        for row_count in xrange(filter_H):
            column_traverse = 0
            row_traverse = HH * row_count
            
            for column_count in xrange(filter_W):
                count = 0
                for iter_ch in xrange(C):
                    for iter_row in xrange(row_traverse,row_traverse+HH):
                        for iter_col in xrange(column_traverse,column_traverse+WW):
                            img_inp_in_block[count,row_count, column_count ] = x_new[i,iter_ch,iter_row,iter_col]
                            count = count + 1
                column_traverse = column_traverse+WW
                
    return img_inp_in_block

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def get_img_to_col(np.ndarray[DTYPE_t,ndim=4] x_new, np.ndarray[DTYPE_t,ndim=2]  img_to_col,int width, int height,int filter_W,int filter_H):

    # img_to_col required for convolutional subsampling layers
    
    cdef int N = x_new.shape[0]
    cdef int C = x_new.shape[1]
    cdef int WW = width
    cdef int HH = height
    cdef int row_traverse = 0
    cdef int column_traverse = 0
    cdef int flag =1
    cdef int count = 0
    cdef int count_col=0
    cdef int i , iter_ch ,iter_row,iter_col,row_count, column_count
 
    for row_count in xrange(filter_H):
        column_traverse = 0
        row_traverse = HH * row_count
        for column_count in xrange(filter_W):
            count = 0
            for iter_ch in xrange(C):
                for iter_row in xrange(row_traverse,row_traverse+HH):
                    for iter_col in xrange(column_traverse,column_traverse+WW):
                        img_to_col[count,count_col] = x_new[0,iter_ch,iter_row,iter_col]
                        count = count + 1
            column_traverse = column_traverse+WW
            count_col = count_col + 1
                
    return img_to_col


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def get_col_to_img(np.ndarray[DTYPE_t,ndim=2] x_col_deconvolve, np.ndarray[DTYPE_t,ndim=4] x_new_deconvolve,int width, int height,int filter_W,int filter_H):

    # Required for convolutional subsampling layers.
    
    cdef int C = x_new_deconvolve.shape[1]
    cdef int WW = width
    cdef int HH = height
    cdef int row_traverse = 0
    cdef int column_traverse = 0
    cdef int flag =1
    cdef int count = 0
    cdef int count_col=0
    cdef int i , iter_ch ,iter_row,iter_col,row_count, column_count
 
    for row_count in xrange(filter_H):
        column_traverse = 0
        row_traverse = HH * row_count
        for column_count in xrange(filter_W):
            count = 0
            for iter_ch in xrange(C):
                for iter_row in xrange(row_traverse,row_traverse+HH):
                    for iter_col in xrange(column_traverse,column_traverse+WW):
                        x_new_deconvolve[0,iter_ch,iter_row,iter_col] = x_new_deconvolve[0,iter_ch,iter_row,iter_col] + x_col_deconvolve[count,count_col]
                        count = count + 1
            column_traverse = column_traverse+WW
            count_col = count_col + 1
                
    return x_new_deconvolve


