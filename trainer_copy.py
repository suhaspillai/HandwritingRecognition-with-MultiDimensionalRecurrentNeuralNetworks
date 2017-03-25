#author : Suhas Pillai

import numpy as np
from pylab import *
from PIL import Image
import time
import cPickle as cp
from lstm_models_copy import *
from editDistance import *
from multiprocessing import Pool
import dill

def MDLSTM_train(params):
    func,args= dill.loads(params)
    X_arr,model,dict_conv_param,grd_truth_seq,reg= args[0]
    return func(X_arr,model,dict_conv_param,grd_truth_seq,reg)


def MDLSTM_val(params):
    func,args= dill.loads(params)
    X_arr,model,dict_conv_param,rand_no,grd_truth_seq,reg= args[0]
   
    return func(X_arr,model,dict_conv_param,grd_truth_seq,reg),rand_no



class Trainer:
    '''
    The class is used to call forward pass and backward pass, along with updating parameters.
    '''

    def __init__(self):
        self.dict_global={'model_1':{'conv':{}},'model_2':{'conv':{},'forward':{},'backward':{},'forward_flip':{},'backward_flip':{},'ff':{}},\
                         'model_3':{'conv':{},'forward':{},'backward':{},'forward_flip':{},'backward_flip':{},'ff':{}},'model_4':{'forward':{},\
                                                                                                                                  'backward':{},'forward_flip':{},'backward_flip':{},'ff':{}}}
        self.loss_list=[]
        self.total_prob =np.zeros(80)


     
    def train_parallel(self,X_arr,model,dict_conv_param,grd_truth_seq,reg):
        lstm_model_obj = Models()
        X = X_arr
        loss, grads,check,check_prob = lstm_model_obj.Model_MDLSTM(X,model,dict_conv_param,grd_truth_seq,reg)
        return loss,grads,check,check_prob 
       
    def val_parallel(self,X_arr,model,dict_conv_param,grd_truth_seq,reg):
         lstm_model_obj = Models()
         out_probs_val = lstm_model_obj.MDLSTM_model_graves(X_arr,model,dict_conv_param)
         return out_probs_val 

    '''
    Train function, it calls train_parallel , to executre parallel jobs across cores
    '''

    def train(self,dict_data_train,list_data_train, validation_dict,validation_list, model,dict_conv_param,char_to_ix,ix_to_char,max_iter,learning_rate,momentum,reg,batch_size,update,epoch,prev_cer):
        
        
        file_loss = open('loss.txt','a')
        file_loss.write('\n Starting Epoch'+ str(epoch)+'\n')         
        ed = Check_edit_distance()
        self.total_prob = np.zeros(80)
        lstm_model_obj = Models()
        beta1=0.9
        beta2=0.999
        total_iter = len(dict_data_train)
        channel = 1
        counter_iter = 0
        count_batch = 0
        mean = 206.186924193
        std = 58.919279934 
        start = time.time()
        list_data = list_data_train
        dict_data = dict_data_train   
        cer = prev_cer 

        while count_batch < max_iter:
            pool = Pool(16)
            end  = min(count_batch+batch_size,max_iter)
            list_imgs = list_data[count_batch:end]
            list_img_arr = []
            list_track=[]
            flag = True
            batch_loss=0.0
            for iter_val in xrange(len(list_imgs)):
                try:
                    X = array(Image.open(list_imgs[iter_val]))
                except:
                    continue
                    #Add up something here if this will affect the further process
                X = (X-mean)/std	  
                seq = dict_data[list_imgs[iter_val]]
                grd_truth_seq = [char_to_ix[i]  for i in seq]
                list_img_arr.append((X,model,dict_conv_param,grd_truth_seq,reg))

            for iter_count in xrange(len(list_img_arr)):
                list_track.append(pool.apply_async(MDLSTM_train,(dill.dumps((self.train_parallel,[list_img_arr[iter_count]])),)))
        
            pool.close()
            pool.join()  
            for iter_count in xrange(len(list_img_arr)):
                loss, grads,check,check_prob = list_track[iter_count].get()    
                if check :
                    continue
                if flag:
                    batch_grads = grads
                    batch_loss=loss
                    flag = False
                    self.total_prob +=check_prob
                else:
                    batch_loss +=loss
                    self.total_prob +=check_prob
                    for model_name in grads:
                        sub_model = grads[model_name]
                        for field in sub_model:
                            sub_sub_model = sub_model[field]
                            for key in sub_sub_model:
                                batch_grads[model_name][field][key]+=grads[model_name][field][key]
                       
            loss = batch_loss/batch_size
            if counter_iter >0:
                self.loss_list.append(0.01*loss  + 0.99 * self.loss_list[-1])
            else:
                self.loss_list.append(loss)
            
            print 'Loss is = %f and Prev Cer = %f' % (self.loss_list[-1],cer)

            #Call after every 10000 samples                       
            if count_batch %10000==0:
                file_loss = open('loss.txt','a')    
                file_loss.write(str(self.loss_list[-1])+'\n')
                file_loss.close()
                
            # Calculate CER
            # Randomly sample from the validation dataset, to test the CER on alidation data  

                list_data_val = []
                list_track_val=[]                    
               
                for iter_val_count in xrange(batch_size):
                    rand_no = np.random.randint(0,len(validation_list)) 
                    X = array(Image.open(validation_list[rand_no]))  
                    X = (X-mean)/std
                    list_data_val.append((X,model,dict_conv_param,rand_no,None,reg))


                pool = Pool(16) #create threads for parallel execution
                for iter_count in xrange(len(list_data_val)):
                    list_track_val.append(pool.apply_async(MDLSTM_val,(dill.dumps((self.val_parallel,[list_data_val[iter_count]])),)))
    
                pool.close()
                pool.join()
                total_dist = 0
                total_corr = 0

                # calculate CER on validation samples
                for iter_count in xrange(len(list_track_val)):
                    out_probs,rand_no = list_track_val[iter_count].get() 
                    seq = validation_dict[validation_list[rand_no]] 
                    grd_truth_seq = [char_to_ix[i]  for i in seq]
                    hyp,dist = ed.decode_best_path(out_probs, grd_truth_seq)
                    if len(hyp)>0:
                        dist, corr = ed.disp(grd_truth_seq,hyp)
                        total_dist +=dist
                        total_corr += corr
             
                cer = 100 * (total_dist /(total_dist+total_corr+1e-8))
                prev_cer = cer 
               
                                       
     #-----------------------------------------------------------update parameter----------------------------------------------------------#

            cache = 'step_cache'
        
            for model_name in model:
               
                sub_model = batch_grads[model_name]
                for field in sub_model:
                    
                    sub_model_field = sub_model[field]
                    if update == 'momentum':                        
                        if cache not in self.dict_global[model_name][field]:
                            self.dict_global[model_name][field]['step_cache']={}
                        for p in sub_model_field:
                            if p not in self.dict_global[model_name][field]['step_cache']:
                                self.dict_global[model_name][field]['step_cache'][p] = np.zeros(batch_grads[model_name][field][p].shape)
                                dx = np.zeros_like(batch_grads[model_name][field][p])
                            
                            dx=momentum * self.dict_global[model_name][field]['step_cache'][p] - learning_rate* (batch_grads[model_name][field][p]/batch_size)
                            self.dict_global[model_name][field]['step_cache'][p] = dx
                            model[model_name][field][p] +=dx
                            
                    elif update=='sgd':
                         for p in sub_model_field:
                             dx = -learning_rate * (batch_grads[model_name][field][p]/batch_size)
                             model[model_name][field][p] += dx

                    elif update=="rmsprop":
                        
                        decay_rate = 0.9
                        if cache not in self.dict_global[model_name][field]:
                            self.dict_global[model_name][field]['step_cache']={}
                        for p in sub_model_field:
                            if p not in self.dict_global[model_name][field]['step_cache']:
                                self.dict_global[model_name][field]['step_cache'][p] = np.zeros(batch_grads[model_name][field][p].shape)
                                dx = np.zeros_like(batch_grads[model_name][field][p])

                            dx = batch_grads[model_name][field][p]/batch_size
                            self.dict_global[model_name][field]['step_cache'][p] = decay_rate * self.dict_global[model_name][field]['step_cache'][p] + (1-decay_rate) * dx**2
                            dx = (-learning_rate * dx) / (np.sqrt(self.dict_global[model_name][field]['step_cache'][p] )+ 1e-8)
                            model[model_name][field][p] += dx
      
            count_batch = count_batch+batch_size
            counter_iter = counter_iter+1
              
        end = time.time()
        print 'Total Time =%f' % (end-start)
        file_parameters = open('dict_parameters','wb')
        file_model = open('model_parameters','wb')
        cp.dump(self.dict_global,file_parameters)
        cp.dump(model,file_model) 
        file_parameters.close()
        file_model.close()
        return model,prev_cer   


    '''
    Method :   initialize_parameters_MDLSTM
    The method is used to initialize parameters of MDLSTM
    '''
    
    def initialize_parameters_MDLSTM(self,input_size,hidden_size,weight_scale=0.001,bias_scale=0):
        model={}
        
        model['W_xi'] = np.random.randn(input_size,hidden_size)/np.sqrt(input_size+2*hidden_size)
        model['W_xf'] =  np.random.randn(input_size,hidden_size)/np.sqrt(input_size+hidden_size)
        model['W_xo']= np.random.randn(input_size,hidden_size)/np.sqrt(input_size+2*hidden_size)
        model['W_xg']= np.random.randn(input_size,hidden_size)/np.sqrt(input_size+2*hidden_size)

        model['W_hi_d1']=  np.random.randn(hidden_size,hidden_size)/np.sqrt(input_size+2*hidden_size)
        model['W_hf_d1']= np.random.randn(hidden_size,hidden_size)/np.sqrt(input_size+hidden_size)
        model['W_ho_d1']=  np.random.randn(hidden_size,hidden_size)/np.sqrt(input_size+2*hidden_size)
        model['W_hg_d1']= np.random.randn(hidden_size,hidden_size)/np.sqrt(input_size+2*hidden_size)
        model['W_hi_d2']=  np.random.randn(hidden_size,hidden_size)/np.sqrt(input_size+2*hidden_size)
        model['W_hf_d2']=  np.random.randn(hidden_size,hidden_size)/np.sqrt(input_size+hidden_size)
        model['W_ho_d2']= np.random.randn(hidden_size,hidden_size)/np.sqrt(input_size+2*hidden_size)
        model['W_hg_d2']=  np.random.randn(hidden_size,hidden_size)/np.sqrt(input_size+2*hidden_size)

        model['b_i']=   bias_scale * np.random.randn(hidden_size)/np.sqrt(input_size+2*hidden_size)
        model['b_f_d1']=   bias_scale * np.random.randn(hidden_size)/np.sqrt(input_size+hidden_size)
        model['b_f_d2']=  bias_scale * np.random.randn(hidden_size)/np.sqrt(input_size+hidden_size)
        model['b_o']=  bias_scale * np.random.randn(hidden_size)/np.sqrt(input_size+2*hidden_size)
        model['b_g']=   bias_scale * np.random.randn(hidden_size)/np.sqrt(input_size+2*hidden_size)
         

        return model	


    

    def cer_val (self,dict_data,list_data, model,dict_conv_param,char_to_ix,ix_to_char,max_iter,reg):
        '''
        The method is used for calculating cer 
        '''

        ed = Check_edit_distance()
        list_data_val = []
        list_track_val=[]
        mean = 206.186924193
        std = 58.919279934

        
        for iter_val_count in xrange(max_iter,len(list_data)): 
            rand_no = iter_val_count
            X = array(Image.open(list_data[rand_no]))
            X = (X-mean)/std
            list_data_val.append((X,model,dict_conv_param,rand_no,None,reg))
          
        pool = Pool(16)
        for iter_count in xrange(len(list_data_val)):
            list_track_val.append(pool.apply_async(MDLSTM_val,(dill.dumps((self.val_parallel,[list_data_val[iter_count]])),)))

        pool.close()
        pool.join()        
        total_dist = 0
        total_corr = 0
        total_chars = 0
        
        for iter_count in xrange(len(list_track_val)):
            out_probs,rand_no = list_track_val[iter_count].get()
            seq = dict_data[list_data[rand_no]]
            grd_truth_seq = [char_to_ix[i]  for i in seq]
            total_chars +=len(grd_truth_seq)
            hyp,dist = ed.decode_best_path(out_probs, grd_truth_seq)
            if len(hyp)>0:
                dist, corr = ed.disp(grd_truth_seq,hyp)
                total_dist +=dist
                total_corr += corr
        
        cer = 100 * (total_dist /(total_dist+total_corr+1e-8))   
        return cer 

