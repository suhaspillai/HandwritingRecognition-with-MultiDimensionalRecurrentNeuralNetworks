#author : Suhas Pillai


import numpy as np
from lstm_layer import *
import time


class LSTM_net:
    '''
    The class creates different layers for your network like blocks of layer, which can then be combined to form a big deep network.
    '''
    def __init__(self):
        pass
        

    def MDLSTM_lstm_conv_feed_layer_forward(self,X,model,conv_param):
        '''
        The method is used for forward propagation of MDLSTM and Convolutional subsampling layer.
        '''
            
        lstm_layer_obj = Layer()
        model_frwd = model['forward']
        model_bckd = model['backward']
        model_frwd_flip = model['forward_flip']
        model_bckd_flip = model['backward_flip']
        
        C,W,H = X.shape
        X_frwd = np.zeros((C,W+1,H+1))
        X_frwd[:,1:,1:] = X
        iter_C,iter_W,iter_H = X_frwd.shape
        X_frwd_flip = np.zeros(X_frwd.shape)
        
        for i in xrange(1,iter_W):
            X_frwd_flip[:,i,:] = X_frwd[:,iter_W-i,:]
        
        X_bckd = np.zeros(X_frwd.shape)
        for i in xrange(1,iter_H):
            X_bckd[:,:,i] = X_frwd[:,:,iter_H-i] 
        W_xi = model_frwd['W_xi']
        
        h_prev_frwd = np.zeros((W_xi.shape[1],X_frwd.shape[1],X_frwd.shape[2]))
        h_prev_bckd = h_prev_frwd.copy()
        h_frwd,cache_lstm_frwd = lstm_layer_obj.forward_propagation_multidimension(X_frwd, model_frwd,h_prev_frwd) 
        h_bckd,cache_lstm_bckd = lstm_layer_obj.forward_propagation_multidimension(X_bckd,model_bckd,h_prev_bckd)  
        # realign
        h_bckd_align = np.zeros(h_bckd.shape)
        for i in xrange(1,iter_H):
            h_bckd_align[:,:,i] = h_bckd[:,:,iter_H-i]

        # now for flip image
        X_bckd_flip  = np.zeros(X_frwd_flip.shape)
        C_iter,W_iter,H_iter = X_bckd.shape
        for i in xrange(1,iter_H):
            X_bckd_flip[:,:,i] = X_frwd_flip[:,:,iter_H-i]

        h_prev_frwd_flip = np.zeros((W_xi.shape[1],X_frwd_flip.shape[1],X_frwd_flip.shape[2]))
        h_prev_bckd_flip = h_prev_frwd_flip.copy()

        h_frwd_flip,cache_lstm_frwd_flip = lstm_layer_obj.forward_propagation_multidimension(X_frwd_flip, model_frwd_flip,h_prev_frwd_flip) 
        h_bckd_flip,cache_lstm_bckd_flip = lstm_layer_obj.forward_propagation_multidimension(X_bckd_flip,model_bckd_flip,h_prev_bckd_flip)  

        h_bckd_flip_align = np.zeros(h_bckd_flip.shape)
        for i in xrange(1,iter_H):
            h_bckd_flip_align[:,:,i] = h_bckd_flip[:,:,iter_H-i]

        # now unflip both the images
        
        h_frwd_unflip = np.zeros(h_frwd_flip.shape)
        h_bckd_unflip_align  = np.zeros(h_bckd_flip_align.shape)

        for i in xrange(1,iter_W):
            h_frwd_unflip[:,i,:] = h_frwd_flip[:,iter_W-i,:]
            h_bckd_unflip_align[:,i,:] = h_bckd_flip_align[:,iter_W-i,:]

        
        # give this to conv layer all as different input
        W_conv = model['conv']['W_conv']
        b_conv = model['conv']['b_conv']
        C,W,H = h_frwd.shape                       
        h_frwd_new = h_frwd[:,1:,1:].reshape(1,C,W-1,H-1) 
        h_bckd_align_new = h_bckd_align[:,1:,1:].reshape(1, C,(W-1),(H-1))
        h_frwd_unflip_new = h_frwd_unflip[:,1:,1:].reshape(1,C,(W-1),(H-1))
        h_bckd_unflip_align_new =h_bckd_unflip_align[:,1:,1:].reshape(1,C,(W-1),(H-1))

        dout_conv_frwd, cache_conv_frwd = lstm_layer_obj.conv_subsampling_forward_multidim(h_frwd_new,W_conv, b_conv, conv_param)
       
        dout_conv_bckd,cache_conv_bckd = lstm_layer_obj.conv_subsampling_forward_multidim( h_bckd_align_new , W_conv, b_conv, conv_param)
        dout_conv_frwd_flip,cache_conv_frwd_flip = lstm_layer_obj.conv_subsampling_forward_multidim(h_frwd_unflip_new, W_conv, b_conv, conv_param)
        dout_conv_bckd_flip,cache_conv_bckd_flip = lstm_layer_obj.conv_subsampling_forward_multidim(h_bckd_unflip_align_new , W_conv, b_conv, conv_param)
        N_ff, C_ff, W_ff, H_ff = dout_conv_frwd.shape
        
        # sum all the convo outputs
        out_conv = dout_conv_frwd + dout_conv_bckd + dout_conv_frwd_flip + dout_conv_bckd_flip
        out_conv = out_conv.reshape(W_ff * H_ff, C_ff)
        cache = (cache_lstm_frwd,cache_lstm_bckd  ,cache_lstm_frwd_flip,cache_lstm_bckd_flip,cache_conv_frwd,cache_conv_bckd, \
                       cache_conv_frwd_flip,cache_conv_bckd_flip)
        return out_conv,cache      


        

    def MDLSTM_lstm_conv_feed_layer_backward(self,dscores,cache):
        
        '''
        The method is used for backpropagation of MDLSTM and convolutional subsampling Layers. 
        '''

        lstm_layer_obj = Layer()
        cache_lstm_frwd,cache_lstm_bckd  ,cache_lstm_frwd_flip,cache_lstm_bckd_flip,cache_conv_frwd,cache_conv_bckd, \
        cache_conv_frwd_flip,cache_conv_bckd_flip,cache_affine_frwd,cache_affine_bckd,cache_affine_frwd_flip,cache_affine_bckd_flip,dout_conv_frwd = cache
        
        dout_ff_frwd,dW_h_to_o_frwd, db_h_to_o_frwd = lstm_layer_obj.affine_backward(dscores, cache_affine_frwd)
        dout_ff_frwd = dout_ff_frwd.T
        dout_ff_bckd,dW_h_to_o_bckd, db_h_to_o_bckd = lstm_layer_obj.affine_backward(dscores, cache_affine_bckd)        
        dout_ff_bckd = dout_ff_bckd.T        
        dout_ff_frwd_flip,dW_h_to_o_frwd_flip, db_h_to_o_frwd_flip = lstm_layer_obj.affine_backward(dscores, cache_affine_frwd_flip)
        dout_ff_frwd_flip = dout_ff_frwd_flip.T
        dout_ff_bckd_flip,dW_h_to_o_bckd_flip, db_h_to_o_bckd_flip = lstm_layer_obj.affine_backward(dscores, cache_affine_bckd_flip)
        dout_ff_bckd_flip = dout_ff_bckd_flip.T

        # pass then through convo layers, but reshape them to prorper format.
        N,C,W,H = dout_conv_frwd.shape
        dout_ff_frwd_conv = dout_ff_frwd.reshape(N,C,W,H)
        dout_ff_bckd_conv = dout_ff_bckd.reshape(N,C,W,H)
        dout_ff_frwd_flip_conv = dout_ff_frwd_flip.reshape(N,C,W,H)
        dout_ff_bckd_flip_conv = dout_ff_bckd_flip.reshape(N,C,W,H)

        dx_frwd_conv,dw_frwd_conv,db_frwd_conv = lstm_layer_obj.conv_subsampling_backward_multidim(dout_ff_frwd_conv, cache_conv_frwd)
        dx_bckd_conv,dw_bckd_conv,db_bckd_conv = lstm_layer_obj.conv_subsampling_backward_multidim(dout_ff_bckd_conv, cache_conv_bckd)
        dx_frwd_flip_conv,dw_frwd_flip_conv,db_frwd_flip_conv = lstm_layer_obj.conv_subsampling_backward_multidim(dout_ff_frwd_flip_conv, cache_conv_frwd_flip)
        dx_bckd_flip_conv,dw_bckd_flip_conv,db_bckd_flip_conv = lstm_layer_obj.conv_subsampling_backward_multidim(dout_ff_bckd_flip_conv, cache_conv_bckd_flip)

        dw_conv = dw_frwd_conv + dw_bckd_conv + dw_frwd_flip_conv + dw_bckd_flip_conv
        db_conv = db_frwd_conv + db_bckd_conv + db_frwd_flip_conv + db_bckd_flip_conv
        
        # passing through lstm layer
        N_conv,C_conv,W_conv,H_conv = dx_frwd_conv.shape
        X,model,h,cell_state,arr_i,arr_f_d1,arr_f_d2,arr_o,arr_g = cache_lstm_frwd
        dout_frwd_new = np.zeros(h.shape)
         
        dout_frwd = dx_frwd_conv.reshape(C_conv,W_conv,H_conv)
        dout_bckd = dx_bckd_conv.reshape(C_conv,W_conv,H_conv)
        dout_frwd_new[:,1:,1:] = dout_frwd
        iter_C,iter_W,iter_H = dout_frwd_new.shape
        dout_bckd_new = np.zeros(h.shape)
        C,W,H = dout_bckd.shape
        for i in xrange(H):
            dout_bckd_new[:,1:,i+1] = dout_bckd[:,:,H-i-1]

        dx_frwd,grads_frwd = lstm_layer_obj.backward_propagation_multidimension(dout_frwd_new,cache_lstm_frwd)
        dx_bckd, grads_bckd = lstm_layer_obj.backward_propagation_multidimension(dout_bckd_new,cache_lstm_bckd)

        dx_bckd_realign = np.zeros(dx_bckd.shape)
        for i in xrange(1,iter_H):
            dx_bckd_realign[:,:,i] = dx_bckd[:,:,iter_H-i]


        # now for flip images
        dout_frwd_flip = dx_frwd_flip_conv.reshape(C_conv,W_conv,H_conv)
        dout_bckd_flip = dx_bckd_flip_conv.reshape(C_conv,W_conv,H_conv)
        dout_frwd_flip_new = np.zeros(h.shape)
        dout_frwd_flip_new[:,1:,1:] = dout_frwd_flip
        dout_bckd_flip_new = np.zeros(h.shape)
        dout_bckd_flip_new[:,1:,1:] = dout_bckd_flip
                
        # flip both
        C,W,H = dout_frwd_flip.shape
        for i in xrange (W):
            dout_frwd_flip_new[:,i+1,1:] = dout_frwd_flip[:,W-i-1,:]
            dout_bckd_flip_new[:,i+1,1:] = dout_bckd_flip[:,W-i-1,:]

        # align the bckd_flip
        iter_C,iter_W,iter_H = dout_bckd_flip_new.shape
        dout_bckd_flip_new_align = np.zeros(dout_bckd_flip_new.shape) 
        for i in xrange(1,iter_H):
            dout_bckd_flip_new_align[:,:,i] = dout_bckd_flip_new[:,:,iter_H-i]

        dx_frwd_flip,grads_frwd_flip = lstm_layer_obj.backward_propagation_multidimension(dout_frwd_flip_new,cache_lstm_frwd_flip)
        dx_bckd_flip, grads_bckd_flip = lstm_layer_obj.backward_propagation_multidimension(dout_bckd_flip_new_align,cache_lstm_bckd_flip)

        # now unflip
        dx_frwd_unflip = np.zeros(dx_frwd_flip.shape)
        dx_bckd_flip_realign = np.zeros(dx_bckd_flip.shape)

        for i in xrange(1,iter_H):
            dx_bckd_flip_realign[:,:,i] = dx_bckd_flip[:,:,iter_H-i]


        dx_bckd_realign_unflip = np.zeros(dx_bckd_flip_realign.shape)
        for i in xrange(1,iter_W):
            dx_frwd_unflip[:,i,:] = dx_frwd_flip[:,iter_W-i,:]
            dx_bckd_realign_unflip[:,i,:] = dx_bckd_flip_realign[:,iter_W-i,:]
             
        # realign bckd_flip
            
        dx = dx_frwd[:,1:,1:] +  dx_bckd_realign[:,1:,1:] + dx_frwd_unflip[:,1:,1:] + dx_bckd_realign_unflip[:,1:,1:]

        grads_ff ={'W_h_to_o_frwd':dW_h_to_o_frwd, 'b_h_to_o_frwd':db_h_to_o_frwd, 'W_h_to_o_bckd':dW_h_to_o_bckd, 'b_h_to_o_bckd':db_h_to_o_bckd ,\
                                 'W_h_to_o_frwd_flip':dW_h_to_o_frwd_flip, 'b_h_to_o_frwd_flip':db_h_to_o_frwd_flip ,'W_h_to_o_bckd_flip':dW_h_to_o_bckd_flip, 'b_h_to_o_bckd_flip':db_h_to_o_bckd_flip}
        grads_conv = {'W_conv':dw_conv,'b_conv':db_conv}
        
        return dx, grads_ff, grads_conv, grads_frwd, grads_bckd, grads_frwd_flip, grads_bckd_flip  




    def MDLSTM_lstm_forward_layer(self,X,model):
        '''
        The method performs Multidimensional Forward propagation
        '''
        lstm_layer_obj = Layer()
        model_frwd = model['forward']
        model_bckd = model['backward']
        model_frwd_flip = model['forward_flip']
        model_bckd_flip = model['backward_flip']
        
        C,W,H = X.shape
        X_frwd = np.zeros((C,W+1,H+1))
        X_frwd[:,1:,1:] = X
        iter_C,iter_W,iter_H = X_frwd.shape
        X_frwd_flip = np.zeros(X_frwd.shape)
        
        for i in xrange(1,iter_W):
            X_frwd_flip[:,i,:] = X_frwd[:,iter_W-i,:]
        
        
        X_bckd = np.zeros(X_frwd.shape)
        for i in xrange(1,iter_H):
            X_bckd[:,:,i] = X_frwd[:,:,iter_H-i] 
        
        W_xi = model_frwd['W_xi']
        
        h_prev_frwd = np.zeros((W_xi.shape[1],X_frwd.shape[1],X_frwd.shape[2]))
        h_prev_bckd = h_prev_frwd.copy()
        
        h_frwd,cache_lstm_frwd = lstm_layer_obj.forward_propagation_multidimension(X_frwd, model_frwd,h_prev_frwd) 
        h_bckd,cache_lstm_bckd = lstm_layer_obj.forward_propagation_multidimension(X_bckd,model_bckd,h_prev_bckd)  
        
        # realign
    
        h_bckd_align = np.zeros(h_bckd.shape)
        for i in xrange(1,iter_H):
            h_bckd_align[:,:,i] = h_bckd[:,:,iter_H-i]

        # now for flip image
        X_bckd_flip  = np.zeros(X_frwd_flip.shape)
        C_iter,W_iter,H_iter = X_bckd.shape
        for i in xrange(1,iter_H):
            X_bckd_flip[:,:,i] = X_frwd_flip[:,:,iter_H-i]

        h_prev_frwd_flip = np.zeros((W_xi.shape[1],X_frwd_flip.shape[1],X_frwd_flip.shape[2]))
        h_prev_bckd_flip = h_prev_frwd_flip.copy()

        h_frwd_flip,cache_lstm_frwd_flip = lstm_layer_obj.forward_propagation_multidimension(X_frwd_flip, model_frwd_flip,h_prev_frwd_flip) 
        h_bckd_flip,cache_lstm_bckd_flip = lstm_layer_obj.forward_propagation_multidimension(X_bckd_flip,model_bckd_flip,h_prev_bckd_flip)  

        h_bckd_flip_align = np.zeros(h_bckd_flip.shape)
        for i in xrange(1,iter_H):
            h_bckd_flip_align[:,:,i] = h_bckd_flip[:,:,iter_H-i]

        # now unflip both the images
        
        h_frwd_unflip = np.zeros(h_frwd_flip.shape)
        h_bckd_unflip_align  = np.zeros(h_bckd_flip_align.shape)

        for i in xrange(1,iter_W):
            h_frwd_unflip[:,i,:] = h_frwd_flip[:,iter_W-i,:]
            h_bckd_unflip_align[:,i,:] = h_bckd_flip_align[:,iter_W-i,:]

        cache = (cache_lstm_frwd,cache_lstm_bckd,cache_lstm_frwd_flip,cache_lstm_bckd_flip)
        return h_frwd,h_bckd_align,h_frwd_unflip,h_bckd_unflip_align,cache

    def MDLSTM_backward_layer(self,dout_frwd,dout_bckd,dout_frwd_flip,dout_bckd_flip,cache):
        '''
        The method is used for backward propagation in Multidimensional LSTM  
        ''' 
        
        lstm_layer_obj = Layer()
        cache_lstm_frwd,cache_lstm_bckd,cache_lstm_frwd_flip,cache_lstm_bckd_flip = cache
        X,model,h,cell_state,arr_i,arr_f_d1,arr_f_d2,arr_o,arr_g = cache_lstm_frwd
        dout_frwd_new = np.zeros(h.shape)
        
        dout_frwd = dout_frwd.reshape(h.shape[0],h.shape[1]-1,h.shape[2]-1)
        dout_bckd = dout_bckd.reshape(h.shape[0],h.shape[1]-1,h.shape[2]-1)
        dout_frwd_new[:,1:,1:] = dout_frwd
        iter_C,iter_W,iter_H = dout_frwd_new.shape
        dout_bckd_new = np.zeros(h.shape)
        C,W,H = dout_bckd.shape
        for i in xrange(H):
            dout_bckd_new[:,1:,i+1] = dout_bckd[:,:,H-i-1]

        
        dx_frwd,grads_frwd = lstm_layer_obj.backward_propagation_multidimension(dout_frwd_new,cache_lstm_frwd)
        dx_bckd, grads_bckd = lstm_layer_obj.backward_propagation_multidimension(dout_bckd_new,cache_lstm_bckd)

        dx_bckd_realign = np.zeros(dx_bckd.shape)
        for i in xrange(1,iter_H):
            dx_bckd_realign[:,:,i] = dx_bckd[:,:,iter_H-i]


        # now for flip images
        dout_frwd_flip = dout_frwd_flip.reshape(h.shape[0],h.shape[1]-1,h.shape[2]-1)
        dout_bckd_flip = dout_bckd_flip.reshape(h.shape[0],h.shape[1]-1,h.shape[2]-1)
        dout_frwd_flip_new = np.zeros(h.shape)
        dout_frwd_flip_new[:,1:,1:] = dout_frwd_flip
        dout_bckd_flip_new = np.zeros(h.shape)
        dout_bckd_flip_new[:,1:,1:] = dout_bckd_flip
                
        # flip both
        C,W,H = dout_frwd_flip.shape
        for i in xrange (W):
            dout_frwd_flip_new[:,i+1,1:] = dout_frwd_flip[:,W-i-1,:]
            dout_bckd_flip_new[:,i+1,1:] = dout_bckd_flip[:,W-i-1,:]

        # align the bckd_flip
        iter_C,iter_W,iter_H = dout_bckd_flip_new.shape
        dout_bckd_flip_new_align = np.zeros(dout_bckd_flip_new.shape) 
        for i in xrange(1,iter_H):
            dout_bckd_flip_new_align[:,:,i] = dout_bckd_flip_new[:,:,iter_H-i]

        dx_frwd_flip,grads_frwd_flip = lstm_layer_obj.backward_propagation_multidimension(dout_frwd_flip_new,cache_lstm_frwd_flip)
        dx_bckd_flip, grads_bckd_flip = lstm_layer_obj.backward_propagation_multidimension(dout_bckd_flip_new_align,cache_lstm_bckd_flip)

        # now unflip
        dx_frwd_unflip = np.zeros(dx_frwd_flip.shape)
        dx_bckd_flip_realign = np.zeros(dx_bckd_flip.shape)

        for i in xrange(1,iter_H):
            dx_bckd_flip_realign[:,:,i] = dx_bckd_flip[:,:,iter_H-i]


        dx_bckd_realign_unflip = np.zeros(dx_bckd_flip_realign.shape)
        for i in xrange(1,iter_W):
            dx_frwd_unflip[:,i,:] = dx_frwd_flip[:,iter_W-i,:]
            dx_bckd_realign_unflip[:,i,:] = dx_bckd_flip_realign[:,iter_W-i,:]
             
        # realign bckd_flip
        dx = dx_frwd[:,1:,1:] +  dx_bckd_realign[:,1:,1:] + dx_frwd_unflip[:,1:,1:] + dx_bckd_realign_unflip[:,1:,1:]
        return dx,grads_frwd,grads_bckd,grads_frwd_flip,grads_bckd_flip
