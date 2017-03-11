#author : Suhas Pillai


import numpy as np
import pdb
import time
import cython_mul_check_3


class Layer:
    '''
    The class conatains all the layers that are required to create MDLSTM NETWORK.
    '''
    def __init__(self):
        pass
    
    def affine_forward(self,x, w, b):

        """
        Computes forward pass for an affine (fully-connected) layer.
        """
        out = None
        x_new=x.reshape(x.shape[0],np.prod(x.shape[1:]))
        out=x_new.dot(w)+b.T    #change to match dimensions
        cache = (x, w, b)
        return out, cache

    def affine_forward_bidirectional(self,x_frwd, w_frwd, b_frwd, x_bckd, w_bckd, b_bckd):
        

        """
        Computes bidirectional forward pass for an affine (fully-connected) layer.
        """
        out = None
        #forward 
        x_new_frwd=x_frwd.reshape(x_frwd.shape[0],np.prod(x_frwd.shape[1:]))
        out_frwd=x_new_frwd.dot(w_frwd)+b_frwd.T    #change to match dimensions
        
        #backward
        x_bckd_rev = np.zeros(x_bckd.shape)
        N  = x_bckd.shape[0] 
        
        for i in xrange(1,N):
            x_bckd_rev[N-i] = x_bckd[i]


        x_new_bckd=x_bckd_rev.reshape(x_bckd_rev.shape[0],np.prod(x_bckd_rev.shape[1:]))

        out_bckd=x_new_bckd.dot(w_bckd)+b_bckd.T    #change to match dimensions
        out = out_frwd + out_bckd 
        cache = (x_frwd, w_frwd, b_frwd, x_bckd_rev, w_bckd, b_bckd)
        return out, cache

    def affine_backward(self,dout, cache):
        """
        Compute backward pass for an affine layer.
        """
        
        x, w, b = cache
        x_new=x.reshape(x.shape[0],np.prod(x.shape[1:]))
        dx, dw, db = None, None, None
        dw=np.zeros(w.shape)
        db=np.zeros(b.shape)
        dx=np.zeros(x_new.shape)
        N =x.shape[0]
        db = db + np.sum(dout,axis=0)
        dx=dout.dot(w.T)
        dx=dx.reshape(x.shape)
        dw=(x_new.T).dot(dout)
        return dx, dw, db

    def affine_backward_birectional(self,dout, cache):
        """
        Computes bidirectional backward pass for an affine layer.
        """
        
        x_frwd, w_frwd, b_frwd, x_bckd, w_bckd, b_bckd  = cache
        x_new_frwd=x_frwd.reshape(x_frwd.shape[0],np.prod(x_frwd.shape[1:]))
        x_new_bckd = x_bckd.reshape(x_bckd.shape[0],np.prod(x_bckd.shape[1:]))

        dx_frwd, dw_frwd, db_frwd,dx_bckd,dw_bckd,db_bckd = None, None, None, None, None, None

        dw_frwd=np.zeros(w_frwd.shape)
        db_frwd=np.zeros(b_frwd.shape)
        dx_frwd=np.zeros(x_new_frwd.shape)
        dw_bckd=np.zeros(w_bckd.shape)
        db_bckd=np.zeros(b_bckd.shape)
        dx_bckd=np.zeros(x_new_bckd.shape)

        N =x_frwd.shape[0]
        #Forward 
        db_frwd = db_frwd + np.sum(dout,axis=0)
        dx_frwd=dout.dot(w_frwd.T)
        dx_frwd=dx_frwd.reshape(x_frwd.shape)
        dw_frwd=(x_new_frwd.T).dot(dout)
        
        #Backward
        db_bckd = db_bckd + np.sum(dout,axis=0)
        dx_bckd=dout.dot(w_bckd.T)
        dx_bckd=dx_bckd.reshape(x_bckd.shape)
        dw_bckd=(x_new_bckd.T).dot(dout)

        N = dx_bckd.shape[0]
        dx_bckd_rev = np.zeros(dx_bckd.shape)
        
        # This for backward propagation for Backward pass, need to reverse the dout, so that we can call the normal backward propagation, where now the first entry will be 20 and not 1.
 
        for i in xrange(1,N):
            dx_bckd_rev[N-i] = dx_bckd[i]

          
        return dx_frwd, dw_frwd, db_frwd, dx_bckd_rev, dw_bckd, db_bckd

    def conv_subsampling_forward(self,x, w, b, conv_param):
        """
        The method is used to compute both convolution and subsampling i.e pooling (Forward Pass)
        """
        out = None     
        N,C,H,W=x.shape
        F,C,HH,WW=w.shape
        stride_H=conv_param['stride_H']
        stride_W = conv_param['stride_W']
        # Pad based on filter size.
        rem_w = W%WW
        rem_h = H%HH
        
        if rem_w !=0 and  rem_h!=0:
            pad_row = HH-rem_h
            pad_col = WW-rem_w
            x_new =  np.lib.pad(x,((0,0),(0,0),(pad_row,0),(0,pad_col)),'constant')
            
        elif rem_w!=0:
            pad_col = WW-rem_w
            x_new =  np.lib.pad(x,((0,0),(0,0),(0,0),(0,pad_col)),'constant')

        elif rem_h!=0:
            pad_row = HH-rem_h
            x_new =  np.lib.pad(x,((0,0),(0,0),(pad_row,0),(0,0)),'constant')    
        else:
            x_new = x
            
        N,C,H_new,W_new = x_new.shape
        filter_H = 1 + (H_new + 2 * pad_row - HH) / stride_H
        filter_W = 1 + (W_new + 2 * pad_col - WW) / stride_W

        x_col=np.zeros((HH*WW*C,filter_H*filter_W)) 
        out=np.zeros((N,F,filter_H,filter_W))
      
        for i in xrange(N):
          count=0
          row_traverse=0
          flag=1
          for channel in xrange(C):
              for row_count in xrange(filter_H):
                column_traverse=0
                if flag==0:
                  row_traverse=row_traverse+stride
                for column_count in xrange(filter_W):
                  x_col[:,count]= x_new[i,:C,row_traverse:row_traverse+HH,column_traverse:column_traverse+WW].reshape((C*WW*HH))
                  column_traverse=column_traverse+stride
                  count=count+1
                  flag=0
                  
          
          w_new=w.reshape((F,np.prod(C*HH*WW)))
          w_new=w_new.dot(x_col)+b.reshape(b.shape[0],1)
          out[i]=w_new.reshape((F,filter_H,filter_W))

        cache = (x, w, b, conv_param)
        # need to convert out to a proper shape for LSTM layer.
        #Convert out to proper dimensions for LSTM  layer input.
        o_N,o_C,o_WW,o_HH = out.shape

        # Thsis for giving input to lstm. we need to five extra column
        #for the existing col dimension, to make matrix multiplication in lstm layer
        temp = np.zeros((o_N,o_C,o_WW,o_HH+1))
        temp[:,:,:,1:] = out
        t_N,t_C,t_WW,t_HH = temp.shape
        # For 1D lstm which takes input as time steps * values i.e w * h
        out_new = np.zeros((t_HH,(t_N*t_C*t_WW)) )
        # When you do ravel the pixels are arrange channel wise, i.e 1,2,3,4 th pixel all of 1st channel, then all pixles for 2nd channel....
        # You need to make sure while propagating gradients back correctly.
        for i in xrange(t_HH):
            out_new[i] = temp[:,:,:,i].ravel()

        return out_new, cache

    def conv_subsampling_backward(self,dout, cache):
        """
        The method is used to compute both convolution and subsampling i.e pooling (Backward Pass)
        """
        # Get the dout in proper dimensions
        #pdb.set_trace()        
        dx, dw, db = None, None, None
        x,w,b,conv_param=cache
        N,C,H,W=x.shape
        F,C,HH,WW=w.shape
        stride_H=conv_param['stride_H']
        stride_W = conv_param['stride_W']
        # cuz dout = H * W (i.e time steps * columns)
        #depth=F

        rem_w = W%WW
        rem_h = H%HH
        pad_row=0
        pad_col  = 0
        if rem_w !=0 and  rem_h!=0:
            pad_row = HH-rem_h
            pad_col = WW-rem_w
            x_new =  np.lib.pad(x,((0,0),(0,0),(pad_row,0),(0,pad_col)),'constant')
            
        elif rem_w!=0:
            pad_col = WW-rem_w
            x_new =  np.lib.pad(x,((0,0),(0,0),(0,0),(0,pad_col)),'constant')

        elif rem_h!=0:
            pad_row = HH-rem_h
            x_new =  np.lib.pad(x,((0,0),(0,0),(pad_row,0),(0,0)),'constant')    
        else:
            x_new = x
            
        N,C,H_new,W_new = x_new.shape
        filter_H = 1 + (H_new + 2 * pad_row - HH) / stride_H
        filter_W = 1 + (W_new + 2 * pad_col - WW) / stride_W

        x_col=np.zeros((HH*WW*C,filter_H*filter_W)) 

        dout_new = np.zeros((N,F,filter_H,filter_W))  
        
        # The reason we have to do this is because ravel arranges (3,W,H) images as 1st all pixels of 1st channel , then of 2nd and then 3rd.
        # As a result while back propagating we have to carefully put elements across channels.
        no_to_traverse= dout.shape[1]/F  # This will giive number of elements column wise for 1 channel.
        for i in xrange(1,dout.shape[0]):  # starng from 1 because added a column for lstm layer calculation
            counter = 0 
            for j in xrange(F):
                dout_new[:,j,:,i-1] = dout[i][counter:counter+no_to_traverse]    # Converting in proper shape for convolution operation.
                counter = counter + no_to_traverse
        dout = dout_new
        #Now we have gradient for every pixel and its channel.
        dw=np.zeros((w.shape[0],np.prod(w.shape[1:])))
        db=np.zeros((b.shape))
        w_new=w.reshape((F,np.prod(C*HH*WW)))
        dx=np.zeros((x.shape))
        x_col_deconvolve=np.zeros(x_col.shape)
        dout=dout.reshape((dout.shape[0],dout.shape[1],np.prod(dout.shape[2:])))

        for i in xrange(N):
          count=0
          row_traverse=0
          flag=1
          for channel in xrange(C):
            for row_count in xrange(filter_H):
              column_traverse=0
              if flag==0:
                row_traverse=row_traverse+stride
              for column_count in xrange(filter_W):
                x_col[:,count]= x_new[i,:C,row_traverse:row_traverse+HH,column_traverse:column_traverse+WW].reshape((C*WW*HH))
                column_traverse=column_traverse+stride
                count=count+1
                flag=0
                
         
          dw=dw+dout[i].dot(x_col.T)
          db=db+np.sum(dout[i],1)
          x_col_deconvolve= w_new.T.dot(dout[i])

          #deconvolve the image to the original size
          count=0
          row_traverse=0
          flag=1
          x_new_deconvolve=np.zeros((x_new.shape))
          
          for row_count in xrange(filter_H):
            column_traverse=0
            if flag==0:
              row_traverse=row_traverse+stride
            for column_count in xrange(filter_W):
              x_new_deconvolve[i,:C,row_traverse:row_traverse+HH,column_traverse:column_traverse+WW]=x_new_deconvolve[i,:C,row_traverse:row_traverse+HH,column_traverse:column_traverse+WW]+x_col_deconvolve[:,count].reshape((C,WW,HH))
              count=count+1
              column_traverse=column_traverse+stride
              flag=0

          if rem_w !=0 and  rem_h!=0:
              dx[i,:C,:,:]=x_new_deconvolve[i,:C,pad_row:,:-pad_col]   # both row col were padded
          elif rem_w!=0:
              dx[i,:C,:,:]=x_new_deconvolve[i,:C,:,:-pad_col]     # only col was padded
          elif rem_h!=0:
              dx[i,:C,:,:]=x_new_deconvolve[i,:C,pad_row:,:]  # only row was padded.
          else:
              dx[i,:C,:,:]=x_new_deconvolve[i,:C,:,:] 

                        
        dw=dw.reshape((w.shape))
        return dx, dw, db

    def conv_subsampling_forward_multidim(self,x, w, b, conv_param):
        """
        The method is used to compute convolution and subsampling i.e pooling for multi dimension LSTM (Forward Pass)
        """
        out = None
      
        N,C,H,W=x.shape
        F,C,HH,WW=w.shape
        stride_H=conv_param['stride_H']
        stride_W = conv_param['stride_W']

        # Pad based on filter size.
        rem_w = W%WW
        rem_h = H%HH
        pad_row = 0
        pad_col = 0
        if rem_w !=0 and  rem_h!=0:
            pad_row = HH-rem_h
            pad_col = WW-rem_w
            x_new =  np.lib.pad(x,((0,0),(0,0),(pad_row,0),(0,pad_col)),'constant')
            
        elif rem_w!=0:
            pad_col = WW-rem_w
            x_new =  np.lib.pad(x,((0,0),(0,0),(0,0),(0,pad_col)),'constant')

        elif rem_h!=0:
            pad_row = HH-rem_h
            x_new =  np.lib.pad(x,((0,0),(0,0),(pad_row,0),(0,0)),'constant')    
        else:
            x_new = x
            
        N,C,H_new,W_new = x_new.shape
        filter_W = W_new/WW
        filter_H = H_new/HH

        x_col=np.zeros((HH*WW*C,filter_H*filter_W)) 
        out=np.zeros((N,F,filter_H,filter_W))
        for i in xrange(N):
          count=0
          row_traverse=0
          flag=1
          for row_count in xrange(filter_H):
              column_traverse=0
              if flag==0:
                  row_traverse=row_traverse+stride_H
              for column_count in xrange(filter_W):
                  x_col[:,count]= x_new[i,:C,row_traverse:row_traverse+HH,column_traverse:column_traverse+WW].reshape((C*WW*HH))
                  column_traverse=column_traverse+stride_W
                  count=count+1
                  flag=0
          w_new=w.reshape((F,np.prod(C*HH*WW)))
          w_new=w_new.dot(x_col)+b.reshape(b.shape[0],1)
          out[i]=w_new.reshape((F,filter_H,filter_W))

        cache = (x, w, b, conv_param)
        return out, cache

    def conv_subsampling_forward_multidim_cython(self,x, w, b, conv_param):
        """
        Cython implementation of convolutional subsampling for fast execution (Forward Pass)
        """
        out = None
      
        N,C,H,W=x.shape
        F,C,HH,WW=w.shape
        stride_H=conv_param['stride_H']
        stride_W = conv_param['stride_W']

        # Pad based on filter size.
        rem_w = W%WW
        rem_h = H%HH
        pad_row = 0
        pad_col = 0
        if rem_w !=0 and  rem_h!=0:
            pad_row = HH-rem_h
            pad_col = WW-rem_w
            x_new =  np.lib.pad(x,((0,0),(0,0),(pad_row,0),(0,pad_col)),'constant')
            
        elif rem_w!=0:
            pad_col = WW-rem_w
            x_new =  np.lib.pad(x,((0,0),(0,0),(0,0),(0,pad_col)),'constant')

        elif rem_h!=0:
            pad_row = HH-rem_h
            x_new =  np.lib.pad(x,((0,0),(0,0),(pad_row,0),(0,0)),'constant')    
        else:
            x_new = x
            
        N,C,H_new,W_new = x_new.shape
        filter_W = W_new/WW
        filter_H = H_new/HH

        x_col=np.zeros((HH*WW*C,filter_H*filter_W)) 
        out=np.zeros((N,F,filter_H,filter_W))
        cython_mul_check_3.get_img_to_col(x_new,x_col,WW,HH,filter_W,filter_H)
        w_new=w.reshape((F,np.prod(C*HH*WW)))
        w_new=w_new.dot(x_col)+b.reshape(b.shape[0],1)
        out[0]=w_new.reshape((F,filter_H,filter_W))
        cache = (x, w, b, conv_param)
        return out, cache

    def conv_subsampling_backward_multidim(self,dout, cache):
        """
        The method is used to compute convolution and subsampling i.e pooling for multi dimension LSTM (Backward Pass)
        """
        dx, dw, db = None, None, None
        x,w,b,conv_param=cache
        N,C,H,W=x.shape
        F,C,HH,WW=w.shape
        stride_H=conv_param['stride_H']
        stride_W = conv_param['stride_W']

        rem_w = W%WW
        rem_h = H%HH
        pad_row=0
        pad_col  = 0
        if rem_w !=0 and  rem_h!=0:
            pad_row = HH-rem_h
            pad_col = WW-rem_w
            x_new =  np.lib.pad(x,((0,0),(0,0),(pad_row,0),(0,pad_col)),'constant')
            
        elif rem_w!=0:
            pad_col = WW-rem_w
            x_new =  np.lib.pad(x,((0,0),(0,0),(0,0),(0,pad_col)),'constant')

        elif rem_h!=0:
            pad_row = HH-rem_h
            x_new =  np.lib.pad(x,((0,0),(0,0),(pad_row,0),(0,0)),'constant')    
        else:
            x_new = x
            
        N,C,H_new,W_new = x_new.shape
        filter_W = W_new/WW
        filter_H = H_new/HH

        x_col=np.zeros((HH*WW*C,filter_H*filter_W)) 

        dw=np.zeros((w.shape[0],np.prod(w.shape[1:])))
        db=np.zeros((b.shape))
        w_new=w.reshape((F,np.prod(C*HH*WW)))
        dx=np.zeros((x.shape))
        x_col_deconvolve=np.zeros(x_col.shape)
        dout=dout.reshape((dout.shape[0],dout.shape[1],np.prod(dout.shape[2:])))

        for i in xrange(N):
          count=0
          row_traverse=0
          flag=1
          for row_count in xrange(filter_H):
              column_traverse=0
              if flag==0:
                  row_traverse=row_traverse+stride_H
              for column_count in xrange(filter_W):
                  x_col[:,count]= x_new[i,:C,row_traverse:row_traverse+HH,column_traverse:column_traverse+WW].reshape((C*WW*HH))
                  column_traverse=column_traverse+stride_W
                  count=count+1
                  flag=0

                  
          dw=dw+dout[i].dot(x_col.T)
          db=db+np.sum(dout[i],1)
          x_col_deconvolve= w_new.T.dot(dout[i])

          #deconvolve the image to the original size
          count=0
          row_traverse=0
          flag=1
          x_new_deconvolve=np.zeros((x_new.shape))
          
          for row_count in xrange(filter_H):
            column_traverse=0
            if flag==0:
              row_traverse=row_traverse+stride_H
            for column_count in xrange(filter_W):
              x_new_deconvolve[i,:C,row_traverse:row_traverse+HH,column_traverse:column_traverse+WW]=x_new_deconvolve[i,:C,row_traverse:row_traverse+HH,column_traverse:column_traverse+WW]+x_col_deconvolve[:,count].reshape((C,HH,WW))
              count=count+1
              column_traverse=column_traverse+stride_W
              flag=0

          if rem_w !=0 and  rem_h!=0:
              dx[i,:C,:,:]=x_new_deconvolve[i,:C,pad_row:,:-pad_col]   # both row col were padded
          elif rem_w!=0:
              dx[i,:C,:,:]=x_new_deconvolve[i,:C,:,:-pad_col]     # only col was padded
          elif rem_h!=0:
              dx[i,:C,:,:]=x_new_deconvolve[i,:C,pad_row:,:]  # only row was padded.
          else:
              dx[i,:C,:,:]=x_new_deconvolve[i,:C,:,:] 

                        
        dw=dw.reshape((w.shape))
        return dx, dw, db

    def conv_subsampling_backward_multidim_cython(self,dout, cache):
        """
        Cython implementation of convolutional subsampling for faster execution (Backward Pass)
        """
        dx, dw, db = None, None, None
        x,w,b,conv_param=cache
        N,C,H,W=x.shape
        F,C,HH,WW=w.shape
        stride_H=conv_param['stride_H']
        stride_W = conv_param['stride_W']

        rem_w = W%WW
        rem_h = H%HH
        pad_row=0
        pad_col  = 0
        if rem_w !=0 and  rem_h!=0:
            pad_row = HH-rem_h
            pad_col = WW-rem_w
            x_new =  np.lib.pad(x,((0,0),(0,0),(pad_row,0),(0,pad_col)),'constant')
            
        elif rem_w!=0:
            pad_col = WW-rem_w
            x_new =  np.lib.pad(x,((0,0),(0,0),(0,0),(0,pad_col)),'constant')

        elif rem_h!=0:
            pad_row = HH-rem_h
            x_new =  np.lib.pad(x,((0,0),(0,0),(pad_row,0),(0,0)),'constant')    
        else:
            x_new = x
            
        N,C,H_new,W_new = x_new.shape
        filter_W = W_new/WW
        filter_H = H_new/HH

        x_col=np.zeros((HH*WW*C,filter_H*filter_W)) 
        dw=np.zeros((w.shape[0],np.prod(w.shape[1:])))
        db=np.zeros((b.shape))
        w_new=w.reshape((F,np.prod(C*HH*WW)))
        dx=np.zeros((x.shape))
        x_col_deconvolve=np.zeros(x_col.shape)
        dout=dout.reshape((dout.shape[0],dout.shape[1],np.prod(dout.shape[2:])))

        for i in xrange(N):
          cython_mul_check_3.get_img_to_col(x_new,x_col,WW,HH,filter_W,filter_H)      
          dw=dw+dout[i].dot(x_col.T)
          db=db+np.sum(dout[i],1)
          x_col_deconvolve= w_new.T.dot(dout[i])
              
          x_new_deconvolve=np.zeros((x_new.shape))
        
          #calling cython
          cython_mul_check_3.get_col_to_img(x_col_deconvolve,  x_new_deconvolve, WW, HH, filter_W,filter_H)
          if rem_w !=0 and  rem_h!=0:
              dx[i,:C,:,:]=x_new_deconvolve[i,:C,pad_row:,:-pad_col]   # both row col were padded
          elif rem_w!=0:
              dx[i,:C,:,:]=x_new_deconvolve[i,:C,:,:-pad_col]     # only col was padded
          elif rem_h!=0:
              dx[i,:C,:,:]=x_new_deconvolve[i,:C,pad_row:,:]  # only row was padded.
          else:
              dx[i,:C,:,:]=x_new_deconvolve[i,:C,:,:] 

                        
        dw=dw.reshape((w.shape))
        return dx, dw, db

    def tanh_forward(self,x):

        """
        Computing tanh forward pass
        """
        out = None
        #pdb.set_trace()
        out = np.tanh(x)
        cache = (x)
        return out,cache

    def tanh_backward(self,dout,cache):
        '''
        Computing tanh backward pass
        '''

        x = cache
        temp = np.tanh(x)
        dx = dout * (1-(temp**2))
        return dx

    def sigmoid(self, X):
        '''
        Computing sigmoid backward pass
        '''
        x_exp = np.exp(-X)   
        x_sigmoid = 1/(1+x_exp) 
        return x_sigmoid

    def tanh(self,X):
        '''
        Computing tanh activation
        '''
        val = np.tanh(X)
        return val

    def forward_propagation_multidimension(self, X, model,h):
        '''
        Forward pass for multidimension LSTM
        '''
        
        T_d1 = X.shape[1]    # along Y direction 
        T_d2 = X.shape[2]   # along x direction
        
        W_xi = model['W_xi']
        W_xf = model['W_xf']
        W_xo = model['W_xo']
        W_xg = model['W_xg']

        W_hi_d1 = model['W_hi_d1']
        W_hf_d1 = model['W_hf_d1']
        W_ho_d1 = model['W_ho_d1']
        W_hg_d1 = model['W_hg_d1']
   
        W_hi_d2 = model['W_hi_d2']
        W_hf_d2 = model['W_hf_d2']
        W_ho_d2 = model['W_ho_d2']
        W_hg_d2 = model['W_hg_d2']
        b_i = model['b_i']
        b_f_d1 = model['b_f_d1']
        b_f_d2 = model['b_f_d2']
        b_o = model['b_o']
        b_g = model['b_g']

        cols = W_xi.shape[1]
        arr_i = np.zeros((cols,T_d1,T_d2))
        arr_f_d1 = np.zeros((cols,T_d1,T_d2))
        arr_f_d2 = np.zeros((cols,T_d1,T_d2))
        arr_o = np.zeros((cols,T_d1,T_d2))
        arr_g = np.zeros((cols,T_d1,T_d2))
        cell_state = np.zeros((cols,T_d1,T_d2))

        # Now doing forward pass across time steps
        
        for t_d2 in xrange(1,T_d2):                      # along x direction
            for t_d1 in xrange(1,T_d1):                  # along y direction
                 
                arr_i[:,t_d1,t_d2] = np.dot(X[:,t_d1,t_d2],W_xi) + np.dot(h[:,t_d1-1,t_d2],W_hi_d1)+ np.dot(h[:,t_d1,t_d2-1],W_hi_d2) + b_i

                arr_f_d1[:,t_d1,t_d2] = np.dot(X[:,t_d1,t_d2],W_xf) + np.dot(h[:,t_d1-1,t_d2],W_hf_d1) + b_f_d1

                arr_f_d2[:,t_d1,t_d2] = np.dot(X[:,t_d1,t_d2],W_xf) + np.dot(h[:,t_d1,t_d2-1],W_hf_d2) + b_f_d2

                arr_o[:,t_d1,t_d2] = np.dot(X[:,t_d1,t_d2],W_xo) + np.dot(h[:,t_d1-1,t_d2],W_ho_d1) + np.dot(h[:,t_d1,t_d2-1],W_ho_d2) + b_o

                arr_g[:,t_d1,t_d2] = np.dot(X[:,t_d1,t_d2],W_xg) + np.dot(h[:,t_d1-1,t_d2],W_hg_d1) + np.dot(h[:,t_d1,t_d2-1],W_hg_d2) + b_g

                i_gate = self.sigmoid(arr_i[:,t_d1,t_d2])
                f_gate_d1 = self.sigmoid(arr_f_d1[:,t_d1,t_d2])
                f_gate_d2 = self.sigmoid(arr_f_d2[:,t_d1,t_d2])
                o_gate = self.sigmoid(arr_o[:,t_d1,t_d2])
                g_input = self.tanh(arr_g[:,t_d1,t_d2])
                cell_state[:,t_d1,t_d2] = i_gate * g_input  +  (f_gate_d1 * cell_state[:,t_d1-1,t_d2] + f_gate_d2 * cell_state[:,t_d1,t_d2-1])

                h[:,t_d1,t_d2] = o_gate * self.tanh(cell_state[:,t_d1,t_d2])
            
        
        cache = (X,model,h,cell_state,arr_i,arr_f_d1,arr_f_d2,arr_o,arr_g)

        return h,cache

    def backward_propagation_multidimension(self,dout,cache):
        
        '''
        Backward pass for multidimension LSTM
        '''
        T_d1 = dout.shape[1] 
        T_d2 = dout.shape[2]
        X,model,h,cell_state,arr_i,arr_f_d1,arr_f_d2,arr_o,arr_g = cache

        W_xi = model['W_xi']
        W_xf = model['W_xf']
        W_xo = model['W_xo']
        W_xg = model['W_xg']
   
        W_hi_d1 = model['W_hi_d1']
        W_hf_d1 = model['W_hf_d1']
        W_ho_d1 = model['W_ho_d1']
        W_hg_d1 = model['W_hg_d1']
        W_hi_d2 = model['W_hi_d2']
        W_hf_d2 = model['W_hf_d2']
        W_ho_d2 = model['W_ho_d2']
        W_hg_d2 = model['W_hg_d2']

        b_i = model['b_i']
        b_f_d1 = model['b_f_d1']
        b_f_d2 = model['b_f_d2']
        b_g = model['b_g']
        b_o = model['b_o']
        cols = dout.shape[0]
        n_gates = 5
        dout_ec = np.zeros((cols,T_d1,T_d2))
        dout_h = np.zeros((cols,T_d1+1,T_d2+1,n_gates)) # to keep derivatives of gates
        dout_es = np.zeros((cols,T_d1+1,T_d2+1))  # to keep cell_state derivative

        arr_f_back_pass_d1 = np.zeros((arr_f_d1.shape[0],arr_f_d1.shape[1]+1,arr_f_d1.shape[2]))
        arr_f_back_pass_d1[:,:-1,:] = arr_f_d1 # for y direction
        arr_f_back_pass_d2 = np.zeros((arr_f_d2.shape[0],arr_f_d2.shape[1],arr_f_d2.shape[2]+1))
        arr_f_back_pass_d2[:,:,:-1] = arr_f_d2  # for x direction 


        dW_xi = np.zeros(W_xi.shape)
        dW_xf = np.zeros(W_xf.shape)
        dW_xo = np.zeros(W_xo.shape)
        dW_xg = np.zeros(W_xg.shape)

        dW_hi_d1 = np.zeros(W_hi_d1.shape)
        dW_hf_d1 = np.zeros(W_hf_d1.shape)
        dW_ho_d1 = np.zeros(W_ho_d1.shape)
        dW_hg_d1 = np.zeros(W_hg_d1.shape)
 
        dW_hi_d2 = np.zeros(W_hi_d2.shape)
        dW_hf_d2 = np.zeros(W_hf_d2.shape)
        dW_ho_d2 = np.zeros(W_ho_d2.shape)
        dW_hg_d2 = np.zeros(W_hg_d2.shape)

        db_i = np.zeros(b_i.shape)
        db_f_d1 = np.zeros(b_f_d1.shape)
        db_f_d2 = np.zeros(b_f_d2.shape)
        db_o = np.zeros(b_o.shape)
        db_g = np.zeros(b_g.shape)
        dx = np.zeros(X.shape)
        N_x = X.shape[0]
        N_h = h.shape[0]

        for t_d2 in reversed(xrange(1,T_d2)):

            for t_d1 in reversed(xrange(1,T_d1)):

                #dout_ec[t] = np.dot(W_h_to_o,dout[t]) + np.dot(W_hi,dout_h[t+1,0]) + np.dot(W_hf,dout_h[t+1,1]) + np.dot(W_ho,dout_h[t+1,2]) + np.dot(W_hg,dout_h[t+1,3])
                # 0 ---> input gate dout
                # 1 ---> forget gate 1 dout
                # 2 ---> forget gatye 2 dout
                # 3 ---> output  gate  dout
                # 4 ---> dout wrt to input
  

                dout_ec[:,t_d1,t_d2] = dout[:,t_d1,t_d2]\
 + (np.dot(W_hi_d1,dout_h[:,t_d1+1,t_d2,0]) + np.dot(W_hi_d2,dout_h[:,t_d1,t_d2+1,0]))\
 + (np.dot(W_hf_d1,dout_h[:,t_d1+1,t_d2,1]) + np.dot(W_hf_d2,dout_h[:,t_d1,t_d2+1,2]))\
 + (np.dot(W_ho_d1,dout_h[:,t_d1+1,t_d2,3]) + np.dot(W_ho_d2,dout_h[:,t_d1,t_d2+1,3]))\
 + (np.dot(W_hg_d1,dout_h[:,t_d1+1,t_d2,4]) + np.dot(W_hg_d2,dout_h[:,t_d1,t_d2+1,4]))
           
                dout_h[:,t_d1,t_d2,3] = (dout_ec[:,t_d1,t_d2] * self.tanh(cell_state[:,t_d1,t_d2])) * self.sigmoid(arr_o[:,t_d1,t_d2]) * (1-self.sigmoid(arr_o[:,t_d1,t_d2]))

                dout_es[:,t_d1,t_d2] = self.sigmoid(arr_o[:,t_d1,t_d2]) * (1-(self.tanh(cell_state[:,t_d1,t_d2]))**2) * dout_ec[:,t_d1,t_d2]\
 + (dout_es[:,t_d1+1,t_d2] * self.sigmoid(arr_f_back_pass_d1[:,t_d1+1,t_d2]))\
 + dout_es[:,t_d1,t_d2+1] * self.sigmoid(arr_f_back_pass_d2[:,t_d1,t_d2+1])


                dout_h[:,t_d1,t_d2,1] = dout_es[:,t_d1,t_d2] * cell_state[:,t_d1-1,t_d2] * (self.sigmoid(arr_f_back_pass_d1[:,t_d1,t_d2])*(1-self.sigmoid(arr_f_back_pass_d1[:,t_d1,t_d2])))

                dout_h[:,t_d1,t_d2,2] = dout_es[:,t_d1,t_d2] *  cell_state[:,t_d1,t_d2-1] * (self.sigmoid(arr_f_back_pass_d2[:,t_d1,t_d2]) * (1-self.sigmoid(arr_f_back_pass_d2[:,t_d1,t_d2])))



                dout_h[:,t_d1,t_d2,0] = dout_es[:,t_d1,t_d2] * self.tanh(arr_g[:,t_d1,t_d2]) * (self.sigmoid(arr_i[:,t_d1,t_d2])*(1-self.sigmoid(arr_i[:,t_d1,t_d2])))

                dout_h[:,t_d1,t_d2,4] = dout_es[:,t_d1,t_d2] * self.sigmoid(arr_i[:,t_d1,t_d2])  * (1-(self.tanh(arr_g[:,t_d1,t_d2]))**2)

                # Calculating the gradients
                temp_x =  X[:,t_d1,t_d2].reshape(N_x,1)
                temp_h_d1 = h[:,t_d1-1,t_d2].reshape(N_h,1)
                temp_h_d2 = h[:,t_d1,t_d2-1].reshape(N_h,1)
                
                # gradients wrt to  inputs weights 
                dW_xi = dW_xi + temp_x * dout_h[:,t_d1,t_d2,0]
                dW_xf = dW_xf + (temp_x * dout_h[:,t_d1,t_d2,1] + temp_x * dout_h[:,t_d1,t_d2,2])
                dW_xo = dW_xo + temp_x * dout_h[:,t_d1,t_d2,3]
                dW_xg = dW_xg + temp_x * dout_h[:,t_d1,t_d2,4]

                # wrt to input gate 
                dW_hi_d1 = dW_hi_d1 + temp_h_d1 * dout_h[:,t_d1,t_d2,0]
                dW_hi_d2 = dW_hi_d2 + temp_h_d2 * dout_h[:,t_d1,t_d2,0] 

                # wrt to forget gates
                dW_hf_d1 = dW_hf_d1 + temp_h_d1 * dout_h[:,t_d1,t_d2,1]
                dW_hf_d2 = dW_hf_d2 + temp_h_d2 * dout_h[:,t_d1,t_d2,2]

                # wrt to output gates
                dW_ho_d1 = dW_ho_d1 + temp_h_d1 * dout_h[:,t_d1,t_d2,3]
                dW_ho_d2 = dW_ho_d2 + temp_h_d2 * dout_h[:,t_d1,t_d2,3] 

                # wrt to inputs 
                dW_hg_d1 = dW_hg_d1 + temp_h_d1 * dout_h[:,t_d1,t_d2,4]
                dW_hg_d2 = dW_hg_d2 + temp_h_d2 * dout_h[:,t_d1,t_d2,4]

                db_i = db_i + dout_h[:,t_d1,t_d2,0] 
                db_f_d1 = db_f_d1 + dout_h[:,t_d1,t_d2,1]
                db_f_d2 = db_f_d2 + dout_h[:,t_d1,t_d2,2]    
                db_o = db_o + dout_h[:,t_d1,t_d2,3]
                db_g = db_g + dout_h[:,t_d1,t_d2,4]
                dx[:,t_d1,t_d2] = np.dot(W_xi,dout_h[:,t_d1,t_d2,0]) + np.dot(W_xf,dout_h[:,t_d1,t_d2,1]) + np.dot(W_xf,dout_h[:,t_d1,t_d2,2])+np.dot(W_xo,dout_h[:,t_d1,t_d2,3]) + np.dot(W_xg,dout_h[:,t_d1,t_d2,4])

        grads={'W_xi':dW_xi,'W_xf' :dW_xf,'W_xo' :dW_xo, 'W_xg' :dW_xg,'W_hi_d1':dW_hi_d1,'W_hi_d2' :dW_hi_d2, 'W_hf_d1' :dW_hf_d1,'W_hf_d2' :dW_hf_d2,'W_ho_d1' :dW_ho_d1, 'W_ho_d2' :dW_ho_d2, 'W_hg_d1' :dW_hg_d1,'W_hg_d2' :dW_hg_d2,'b_i' : db_i, 'b_f_d1' :db_f_d1, 'b_f_d2' :db_f_d2, 'b_o' :db_o,'b_g' :db_g}
           
        return dx, grads

    def forward_propagation_cythonic(self, X, model,h):
        '''
        Performing forward propagation (calling cython code for faster execution)
        '''
        T_d1 = X.shape[1]    # along Y direction 
        T_d2 = X.shape[2]   # along x direction
        W_xi = model['W_xi']
        W_xf = model['W_xf']
        W_xo = model['W_xo']
        W_xg = model['W_xg']

        W_hi_d1 = model['W_hi_d1']
        W_hf_d1 = model['W_hf_d1']
        W_ho_d1 = model['W_ho_d1']
        W_hg_d1 = model['W_hg_d1']

        W_hi_d2 = model['W_hi_d2']
        W_hf_d2 = model['W_hf_d2']
        W_ho_d2 = model['W_ho_d2']
        W_hg_d2 = model['W_hg_d2']
        #W_h_to_o = model['W_h_to_o']
        b_i = model['b_i']
        b_f_d1 = model['b_f_d1']
        b_f_d2 = model['b_f_d2']
        b_o = model['b_o']
        b_g = model['b_g']

        cols = W_xi.shape[1]
        arr_i = np.zeros((cols,T_d1,T_d2))
        arr_f_d1 = np.zeros((cols,T_d1,T_d2))
        arr_f_d2 = np.zeros((cols,T_d1,T_d2))
        arr_o = np.zeros((cols,T_d1,T_d2))
        arr_g = np.zeros((cols,T_d1,T_d2))
        cell_state = np.zeros((cols,T_d1,T_d2))


        out,X,cell_state,arr_i,arr_f_d1,arr_f_d2,arr_o,arr_g =  cython_mul_check_3. forward_propagation_multidimension(X,  W_xi,  W_xf,  W_xo,  W_xg,  W_hi_d1,  W_hf_d1,  W_ho_d1,\
                                                                        W_hg_d1,  W_hi_d2,  W_hf_d2,  W_ho_d2,  W_hg_d2,  b_i,  b_f_d1, b_f_d2,  b_o,b_g ,h,arr_i, arr_f_d1,arr_f_d2,  arr_o , arr_g,  cell_state)
        cache = (X,model,h,cell_state,arr_i,arr_f_d1,arr_f_d2,arr_o,arr_g)
        return out,cache
    
    def backward_propagation_cythonic(self, dout,cache):
        '''
        Backward propagation for multidimension LSTM in cython (calling cython code for faster execution)
        '''
        
        T_d1 = dout.shape[1] 
        T_d2 = dout.shape[2]
        X,model,h,cell_state,arr_i,arr_f_d1,arr_f_d2,arr_o,arr_g = cache
        W_xi = model['W_xi']
        W_xf = model['W_xf']
        W_xo = model['W_xo']
        W_xg = model['W_xg']
   
        W_hi_d1 = model['W_hi_d1']
        W_hf_d1 = model['W_hf_d1']
        W_ho_d1 = model['W_ho_d1']
        W_hg_d1 = model['W_hg_d1']
        
        W_hi_d2 = model['W_hi_d2']
        W_hf_d2 = model['W_hf_d2']
        W_ho_d2 = model['W_ho_d2']
        W_hg_d2 = model['W_hg_d2']

        b_i = model['b_i']
        b_f_d1 = model['b_f_d1']
        b_f_d2 = model['b_f_d2']
        b_g = model['b_g']
        b_o = model['b_o']

        cols = dout.shape[0]
        n_gates = 5

        dout_ec = np.zeros((cols,T_d1,T_d2))
        
        dout_h = np.zeros((cols,T_d1+1,T_d2+1,n_gates)) # to keep derivatives of gates

        
        dout_es = np.zeros((cols,T_d1+1,T_d2+1))  # to keep cell_state derivative

        arr_f_back_pass_d1 = np.zeros((arr_f_d1.shape[0],arr_f_d1.shape[1]+1,arr_f_d1.shape[2]))
        arr_f_back_pass_d1[:,:-1,:] = arr_f_d1 # for y direction
        arr_f_back_pass_d2 = np.zeros((arr_f_d2.shape[0],arr_f_d2.shape[1],arr_f_d2.shape[2]+1))
        arr_f_back_pass_d2[:,:,:-1] = arr_f_d2  # for x direction 


        dW_xi = np.zeros(W_xi.shape)
        dW_xf = np.zeros(W_xf.shape)
        dW_xo = np.zeros(W_xo.shape)
        dW_xg = np.zeros(W_xg.shape)

        dW_hi_d1 = np.zeros(W_hi_d1.shape)
        dW_hf_d1 = np.zeros(W_hf_d1.shape)
        dW_ho_d1 = np.zeros(W_ho_d1.shape)
        dW_hg_d1 = np.zeros(W_hg_d1.shape)
 
        dW_hi_d2 = np.zeros(W_hi_d2.shape)
        dW_hf_d2 = np.zeros(W_hf_d2.shape)
        dW_ho_d2 = np.zeros(W_ho_d2.shape)
        dW_hg_d2 = np.zeros(W_hg_d2.shape)


        
        db_i = np.zeros(b_i.shape)
        db_f_d1 = np.zeros(b_f_d1.shape)
        db_f_d2 = np.zeros(b_f_d2.shape)
        db_o = np.zeros(b_o.shape)
        db_g = np.zeros(b_g.shape)
        
        dx = np.zeros(X.shape)

        dx,dW_xi,dW_xf,dW_xo,dW_xg,dW_hi_d1,dW_hi_d2,dW_hf_d1,dW_hf_d2,dW_ho_d1,dW_ho_d2,dW_hg_d1,dW_hg_d2,db_i,db_f_d1,db_f_d2,db_o,db_g = \
        cython_mul_check_3.backward_propagation_multidimension( dout,  X, h,cell_state, arr_i,  arr_f_back_pass_d1, arr_f_back_pass_d2,arr_o,  arr_g, W_xi, W_xf, W_xo, \
        W_xg, W_hi_d1,  W_hf_d1,  W_ho_d1,  W_hg_d1,  W_hi_d2, W_hf_d2,  W_ho_d2,  W_hg_d2, b_i,  b_f_d1, b_f_d2, b_o, b_g ,dout_ec,dout_h, dout_es,dW_xi, dW_xf,\
        dW_xo,dW_xg,  dW_hi_d1,  dW_hf_d1,  dW_ho_d1,dW_hg_d1,   dW_hi_d2,  dW_hf_d2,  dW_ho_d2,dW_hg_d2,  db_i,  db_f_d1,  db_f_d2, db_o,  db_g,  dx)

        grads={'W_xi':dW_xi,'W_xf' :dW_xf,'W_xo' :dW_xo, 'W_xg' :dW_xg,'W_hi_d1':dW_hi_d1,'W_hi_d2' :dW_hi_d2, 'W_hf_d1' :dW_hf_d1,'W_hf_d2' :dW_hf_d2,'W_ho_d1' :dW_ho_d1, 'W_ho_d2' :dW_ho_d2, 'W_hg_d1' :dW_hg_d1,'W_hg_d2' :dW_hg_d2,'b_i' : db_i, 'b_f_d1' :db_f_d1, 'b_f_d2' :db_f_d2, 'b_o' :db_o,'b_g' :db_g}
        return dx, grads

    def dropout_train_forward(self, X,p):
        
        '''
        Drop out layer forward
        ''' 
        H = np.maximum(0,X)
        U = (np.random.rand(*H.shape)<p)/p
        H *=U
        cache = (X,U)
        return H,cache

    def dropout_train_backward(self,dout,cache):
        '''
        Drop out layer backward
        '''
        X,U = cache
        dout = U * dout
        dx = np.zeros(dout.shape)
        dx = (np.maximum(0,X)>0)  * dout
        return dx       

    def softmax_loss(self,x, y):
        """
        Computes the loss and gradient for softmax classification.
        """

        f_open = open('prob.txt','a')
        probs = np.exp(x - np.max(x, axis=1, keepdims=True))

        probs /= np.sum(probs, axis=1, keepdims=True)
        f_open.write(str(probs))

        N = x.shape[0]

        loss = -np.sum(np.log(probs[np.arange(N), y])) / N

        dx = probs.copy()
        dx[np.arange(N), y] -= 1
        dx /= N
        return loss, dx

    def get_input_block(self, x,width,height):
        '''
        To convert image data into blocks and then start processing 
        '''

        x = x.reshape(1,1,x.shape[0],x.shape[1])
        N,C,H,W=x.shape
        WW  = width
        HH =  height
        rem_w = W%WW
        rem_h = H%HH
        pad_row = 0
        pad_col = 0
        if rem_w !=0 and  rem_h!=0:
            pad_row = HH-rem_h
            pad_col = WW-rem_w
            x_new =  np.lib.pad(x,((0,0),(0,0),(pad_row,0),(0,pad_col)),'constant')
            
        elif rem_w!=0:
            pad_col = WW-rem_w
            x_new =  np.lib.pad(x,((0,0),(0,0),(0,0),(0,pad_col)),'constant')

        elif rem_h!=0:
            pad_row = HH-rem_h
            x_new =  np.lib.pad(x,((0,0),(0,0),(pad_row,0),(0,0)),'constant')    
        else:
            x_new = x

        N,C,H_new,W_new = x_new.shape
        filter_W = W_new/WW
        filter_H = H_new/HH

        no_of_elem = WW*HH
        img_inp_in_block = np.zeros((no_of_elem,filter_H,filter_W))
        stride_H = HH
        stride_W = WW

        for i in xrange(N):
            
            row_traverse=0
            flag=1
            for row_count in xrange(filter_H):
                column_traverse=0
                if flag==0:
                    row_traverse=row_traverse+stride_H
                for column_count in xrange(filter_W):
                    img_inp_in_block[:,row_count,column_count]= x_new[i,:C,row_traverse:row_traverse+HH,column_traverse:column_traverse+WW].reshape((C*WW*HH))
                    column_traverse=column_traverse+stride_W
                    flag=0

        return img_inp_in_block
    
    def get_input_block_cython(self, x,width,height):
        '''
        Cython implementation to convert image data into blocks and then start processing
        '''        
        x = x.reshape(1,1,x.shape[0],x.shape[1])
        N,C,H,W=x.shape
        WW  = width
        HH =  height
        rem_w = W%WW
        rem_h = H%HH
        pad_row = 0
        pad_col = 0
        if rem_w !=0 and  rem_h!=0:
            pad_row = HH-rem_h
            pad_col = WW-rem_w
            x_new =  np.lib.pad(x,((0,0),(0,0),(pad_row,0),(0,pad_col)),'constant')
            
        elif rem_w!=0:
            pad_col = WW-rem_w
            x_new =  np.lib.pad(x,((0,0),(0,0),(0,0),(0,pad_col)),'constant')

        elif rem_h!=0:
            pad_row = HH-rem_h
            x_new =  np.lib.pad(x,((0,0),(0,0),(pad_row,0),(0,0)),'constant')    
        else:
            x_new = x

        N,C,H_new,W_new = x_new.shape
        filter_W = W_new/WW
        filter_H = H_new/HH

        no_of_elem = WW*HH
        img_inp_in_block = np.zeros((no_of_elem,filter_H,filter_W))
        cython_mul_check_3.get_input_block(x_new,img_inp_in_block,WW,HH,filter_W,filter_H)
        return img_inp_in_block
