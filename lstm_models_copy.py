#author : Suhas Pillai

import numpy as np
import pdb
from lstm_layer import *
from lstm_net import *
from ctc_loss import *
from lstm_net_cythonic import *

class Models:
    '''
    This class is the deep network, created using  many sub modules from lstm_layer_net.py and lstm_layer.py files 
    '''
    def __init__(self):
        pass

    def MDLSTM_model_graves(self,X,model,dict_conv_param,grd_truth_seq=None):
        '''
        The function is the forward pass for Deep Network
        '''
        
        lstm_layer_obj = Layer()
        lstm_net_obj = LSTM_net()
        lstm_net_obj_cython = LSTM_net_cythonic()
        ctc_obj = CTC()
        conv_param_1 = dict_conv_param['conv_1']
        width = conv_param_1['width']
        height = conv_param_1['height']
        X = lstm_layer_obj.get_input_block_cython(X,width,height)     # Get input blocks

        #----------------------------------------------MDLSTM lstm_conv_feedforward block----------------------------------------------------#
        model_2 = model['model_2']
        conv_param_2 = dict_conv_param['conv_2']
        # call to cython
        out_2, cache_2 = lstm_net_obj_cython.MDLSTM_lstm_conv_feed_layer_forward(X,model_2,conv_param_2)
        _,_,_,_,_,_,_,_,dout_conv_frwd_2 = cache_2
  
        # pass through tanh
        out_tanh_2, cache_tanh_2 = lstm_layer_obj.tanh_forward(out_2)
        
        #----------------------------------------------MDLSTM lstm_conv_feedward block----------------------------------------------------#
        out_tanh_2 = out_tanh_2.T
        N_conv_2,C_conv_2,W_conv_2,H_conv_2 = dout_conv_frwd_2.shape
        out_tanh_2_re = out_tanh_2.reshape(out_tanh_2.shape[0],W_conv_2,H_conv_2)
        model_3 = model['model_3']
        conv_param_3 = dict_conv_param['conv_3']
        
        # call to cython
        out_3, cache_3 = lstm_net_obj_cython.MDLSTM_lstm_conv_feed_layer_forward(out_tanh_2_re,model_3,conv_param_3)
        
        _,_,_,_,_,_,_,_,dout_conv_frwd_3 = cache_3
    
        # pass through tanh
        
        out_tanh_3, cache_tanh_3 = lstm_layer_obj.tanh_forward(out_3)
        
        #-----------------------------------------------------------------last lstm layer---------------------------------------------#

        out_tanh_3 = out_tanh_3.T
        N_conv_3,C_conv_3,W_conv_3,H_conv_3= dout_conv_frwd_3.shape
        out_tanh_3_re = out_tanh_3.reshape(out_tanh_3.shape[0],W_conv_3,H_conv_3)
        model_4 = model['model_4']
        h_frwd_4,h_bckd_4,h_frwd_unflip_4,h_bckd_unflip_align_4,cache_lstm_4 = lstm_net_obj.MDLSTM_lstm_forward_layer(out_tanh_3_re,model_4)
        # calling cython
        # Some reshaping required     
        C=  h_frwd_4.shape[0]
        W = h_frwd_4.shape[1]-1
        H = h_frwd_4.shape[2]-1
        h_frwd_4 = h_frwd_4[:,1:,1:].reshape(C,W*H)
        h_frwd_4  = h_frwd_4 .T
        h_bckd_4 = h_bckd_4[:,1:,1:].reshape(C,W*H)
        h_bckd_4  = h_bckd_4 .T
        h_frwd_unflip_4 = h_frwd_unflip_4[:,1:,1:].reshape(C,W*H)
        h_frwd_unflip_4 = h_frwd_unflip_4.T
        h_bckd_unflip_align_4 = h_bckd_unflip_align_4[:,1:,1:].reshape(C,W*H)
        h_bckd_unflip_align_4 =h_bckd_unflip_align_4.T

        #------------------------------------------------------------Fully Connnected---------------------------------------------------# 
        W_h_to_o_4_frwd = model_4['ff']['W_h_to_o_frwd']
        b_h_to_o_4_frwd  = model_4['ff']['b_h_to_o_frwd']
        W_h_to_o_4_bckd = model_4['ff']['W_h_to_o_bckd']
        b_h_to_o_4_bckd  = model_4['ff']['b_h_to_o_bckd']
        W_h_to_o_4_frwd_flip = model_4['ff']['W_h_to_o_frwd_flip']
        b_h_to_o_4_frwd_flip  = model_4['ff']['b_h_to_o_frwd_flip']
        W_h_to_o_4_bckd_flip = model_4['ff']['W_h_to_o_bckd_flip']
        b_h_to_o_4_bckd_flip  = model_4['ff']['b_h_to_o_bckd_flip']

        out_4_frwd,cache_affine_forward_4_frwd = lstm_layer_obj.affine_forward(h_frwd_4,W_h_to_o_4_frwd, b_h_to_o_4_frwd)
        out_4_bckd,cache_affine_forward_4_bckd = lstm_layer_obj.affine_forward(h_bckd_4,W_h_to_o_4_bckd, b_h_to_o_4_bckd)
        out_4_frwd_flip,cache_affine_forward_4_frwd_flip = lstm_layer_obj.affine_forward(h_frwd_unflip_4,W_h_to_o_4_frwd_flip, b_h_to_o_4_frwd_flip)
        out_4_bckd_flip,cache_affine_forward_4_bckd_flip = lstm_layer_obj.affine_forward(h_bckd_unflip_align_4, W_h_to_o_4_bckd_flip, b_h_to_o_4_bckd_flip)
        
        cache_affine_forward_4 = (cache_affine_forward_4_frwd, cache_affine_forward_4_bckd, cache_affine_forward_4_frwd_flip,cache_affine_forward_4_bckd_flip) 

        #----------------------------------------------------Sum across fully connected--------------------------------------------------#
        out_4_sum= out_4_frwd + out_4_bckd + out_4_frwd_flip +  out_4_bckd_flip
        out_4_sum = out_4_sum.T
        out_4 = out_4_sum.reshape(out_4_sum.shape[0],W,H)
        out_4=out_4.sum(1)        #sum across vertical dimensions.

         
        if grd_truth_seq==None:
            
            out_4= out_4- np.max(out_4,axis=0)
            out_4 = np.exp(out_4)
            out_4 = out_4 / np.sum(out_4,axis=0)
            return out_4

        check_prob = out_4.copy()
        check_prob = check_prob-np.max(check_prob,axis=0)
        check_prob = np.exp(check_prob) 
        check_prob = check_prob / np.sum(check_prob,axis=0)
        check_prob = np.sum(check_prob,1)/check_prob.shape[1]

       #----------------------------------------------------Calling CTC cost function-------------------------------------------------------#
        grd_truth_seq = np.asarray(grd_truth_seq)
        loss,dscores,check = ctc_obj.ctc_loss_mass(out_4,grd_truth_seq,0,False)
        dscores = dscores.T
        cache  = (cache_2 ,cache_tanh_2,cache_3,cache_tanh_3,cache_lstm_4,cache_affine_forward_4)
        return loss,dscores,check,cache,check_prob
        
    def MDLSTM_model_backward_graves(self,dscores,cache):
        '''
        The function is used for backward propagation of Deep network.
        '''       
        lstm_net_obj_cython = LSTM_net_cythonic()
        lstm_layer_obj = Layer()
        lstm_net_obj = LSTM_net()
        cache_2 ,cache_tanh_2,cache_3,cache_tanh_3,cache_lstm_4,cache_affine_forward_4 = cache
        
        cache_lstm_frwd,cache_lstm_bckd,cache_lstm_frwd_flip,cache_lstm_bckd_flip = cache_lstm_4
        X,model,h,cell_state,arr_i,arr_f_d1,arr_f_d2,arr_o,arr_g = cache_lstm_frwd
        C =  h.shape[0]
        W= h.shape[1]-1
        H=h.shape[2]-1
        
        dout_sum = np.zeros((dscores.shape[1],W, H))
        
        # giving error with respect to every pixel, which was summed.  
        for iter_i in xrange(H):
            for iter_j in xrange(W):
                    dout_sum[:,iter_j,iter_i] = dscores[iter_i,:]

        cache_affine_forward_4_frwd, cache_affine_forward_4_bckd, cache_affine_forward_4_frwd_flip,cache_affine_forward_4_bckd_flip = cache_affine_forward_4         
        dout_sum = dout_sum.reshape(dout_sum.shape[0],dout_sum.shape[1]*dout_sum.shape[2])
        dout_sum = dout_sum.T

        #-----------------------------------------------Calling affine Backward-------------------------------------------------------------#

        dout_frwd_4, dW_h_to_o_4_frwd,db_h_to_o_4_frwd = lstm_layer_obj.affine_backward(dout_sum, cache_affine_forward_4_frwd)
        dout_frwd_4 = dout_frwd_4.T
        dout_frwd_4 = dout_frwd_4.reshape(C,W,H)
        dout_bckd_4, dW_h_to_o_4_bckd,db_h_to_o_4_bckd = lstm_layer_obj.affine_backward(dout_sum, cache_affine_forward_4_bckd)
        dout_bckd_4 =dout_bckd_4.T
        dout_bckd_4 = dout_bckd_4.reshape(C,W,H)
        dout_frwd_flip_4, dW_h_to_o_4_frwd_flip,db_h_to_o_4_frwd_flip = lstm_layer_obj.affine_backward(dout_sum, cache_affine_forward_4_frwd_flip)
        dout_frwd_flip_4 = dout_frwd_flip_4.T
        dout_frwd_flip_4 = dout_frwd_flip_4.reshape(C,W,H)
        dout_bckd_flip_4, dW_h_to_o_4_bckd_flip,db_h_to_o_4_bckd_flip = lstm_layer_obj.affine_backward(dout_sum, cache_affine_forward_4_bckd_flip)
        dout_bckd_flip_4 = dout_bckd_flip_4.T
        dout_bckd_flip_4 = dout_bckd_flip_4.reshape(C,W,H)
        
        dx_4,grads_frwd_4,grads_bckd_4,grads_frwd_flip_4,grads_bckd_flip_4 = lstm_net_obj.MDLSTM_backward_layer(dout_frwd_4,dout_bckd_4,dout_frwd_flip_4,dout_bckd_flip_4,cache_lstm_4)
        # calling cython
        dx_4 = dx_4.reshape(dx_4.shape[0],dx_4.shape[1] * dx_4.shape[2])
        dx_4 = dx_4.T
        
        # passing through tanh
        dx_tanh_3 = lstm_layer_obj.tanh_backward(dx_4,cache_tanh_3)

        #-----------------------------------------------Backward Propagation of lstm_conv_feed_layer----------------------------------------#
        
        dx_tanh_3 = dx_tanh_3.T     #(W*H,C) --->(C,W*H)
        dx_3, grads_conv_3, grads_frwd_3, grads_bckd_3, grads_frwd_flip_3, grads_bckd_flip_3 = lstm_net_obj_cython. MDLSTM_lstm_conv_feed_layer_backward(dx_tanh_3,cache_3)
        dx_3 = dx_3.reshape(dx_3.shape[0],dx_3.shape[1] * dx_3.shape[2])
        dx_3 = dx_3.T

       

        #pass through tanh
        dx_tanh_2 = lstm_layer_obj.tanh_backward(dx_3,cache_tanh_2)

        #-----------------------------------------------Backward Propagation of lstm_conv_feed_layer----------------------------------------#   
        # calling cython
        dx_tanh_2 = dx_tanh_2.T
        dx_2,grads_conv_2, grads_frwd_2, grads_bckd_2, grads_frwd_flip_2, grads_bckd_flip_2 = lstm_net_obj_cython. MDLSTM_lstm_conv_feed_layer_backward(dx_tanh_2,cache_2)
         
        dx_2 = dx_2.reshape(1,dx_2.shape[0],dx_2.shape[1],dx_2.shape[2])

        # --------------------------------------------------Storing Gradients----------------------------------------------------------------#          
        grads_model_2 = {'conv':grads_conv_2,'forward':grads_frwd_2,'backward':grads_bckd_2,'forward_flip':grads_frwd_flip_2,'backward_flip':grads_bckd_flip_2}
        grads_model_3 = {'conv':grads_conv_3,'forward':grads_frwd_3,'backward':grads_bckd_3,'forward_flip':grads_frwd_flip_3,'backward_flip':grads_bckd_flip_3}
        grads_model_4 = {'ff':{'W_h_to_o_frwd':dW_h_to_o_4_frwd,'b_h_to_o_frwd':db_h_to_o_4_frwd,'W_h_to_o_bckd':dW_h_to_o_4_bckd,'b_h_to_o_bckd':db_h_to_o_4_bckd,\
                               'W_h_to_o_frwd_flip':dW_h_to_o_4_frwd_flip,'b_h_to_o_frwd_flip':db_h_to_o_4_frwd_flip,'W_h_to_o_bckd_flip':dW_h_to_o_4_bckd_flip, \
                               'b_h_to_o_bckd_flip':db_h_to_o_4_bckd_flip},'forward':grads_frwd_4,'backward':grads_bckd_4,'forward_flip':grads_frwd_flip_4,'backward_flip':grads_bckd_flip_4}

        return dx_2,grads_model_2, grads_model_3, grads_model_4

    def Model_MDLSTM(self,X,model,dict_conv_param,grd_truth_seq,reg):
        '''
        The method calls forward and backward pass, i.e the above 2 methods.
        '''           
        loss_forward,dscores,check,cache,check_prob = self.MDLSTM_model_graves(X,model,dict_conv_param,grd_truth_seq)
        dx_2,grads_model_2, grads_model_3, grads_model_4 = self.MDLSTM_model_backward_graves(dscores,cache)
        grads={'model_2':grads_model_2,'model_3':grads_model_3,'model_4':grads_model_4}
 
        # L2 regularization
        reg_loss = 0.0
        for model_name in model:
            for field in model[model_name]:
                for key in model[model_name][field]:
                    grads[model_name][field][key] += reg * model[model_name][field][key]
                    reg_loss +=  np.sum(model[model_name][field][key]**2)
                    
        reg_loss = 0.5 * reg * reg_loss
        loss_forward +=  reg_loss
        return loss_forward, grads,check,check_prob
    
        
