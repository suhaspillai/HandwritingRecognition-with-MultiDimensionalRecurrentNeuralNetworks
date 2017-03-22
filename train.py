#author : Suhas Pillai

import numpy as  np
from loadData import *
from trainer_copy import *
from PIL import Image
from pylab import *
import pdb
import pstats, cProfile
import sys
import argparse

parser  = argparse.ArgumentParser()
parser.add_argument('--learning_rate',default=0.001,type=float)
parser.add_argument('--momentum',default=0.9,type=float)
parser.add_argument('--reg',default=0.0,type=float)
parser.add_argument('--update',default='rmsprop')
parser.add_argument('--batch_size',default=200,type=int)
parser.add_argument('--epochs',default=50,type=int)
args = parser.parse_args()

'''
Main file to train the network
'''

def main():
    print '\n ..................Start Training..................\n'
    file_dict_data = open('dict_data','rb')
    dict_data = cp.load(file_dict_data)
    f_train = open('training_data','rb')
    list_data_train = cp.load(f_train)
    f_val = open('validation_data','rb')
    list_data_val = cp.load(f_val)
    dict_data_train={}
    dict_data_val={}

    # create dictionary for training and validation
    for img in list_data_train:
        dict_data_train[img] = dict_data[img]

    for img in list_data_val:
        dict_data_val[img] = dict_data[img]

    file_chars = open('chars_data','rb')
    chars = cp.load(file_chars)
    list_vocab = []
    list_vocab.append('blank')
    for i in xrange(len(chars)):
        list_vocab.append(chars.pop())

    char_to_ix = {list_vocab[i]:i for i in xrange(len(list_vocab))}
    ix_to_char = {i:list_vocab[i] for i in xrange(len(list_vocab))}
    vocab_size = len(char_to_ix)

    #Intialize weights
    #-------------------------------------------------Model_1 --------------------------------------------------------#

    bias_scale = 0.0
    conv_param_1={'width':2,'height':2}
    width = conv_param_1['width']
    height = conv_param_1['height']
    #-------------------------------------------------Model_2----------------------------------------------------------#

    trainer_obj = Trainer()
    model_2={}
    input_size = width * height
    hidden_size_1 = 2
    model_2['forward']=trainer_obj.initialize_parameters_MDLSTM(input_size,hidden_size_1)
    model_2['backward']=trainer_obj.initialize_parameters_MDLSTM(input_size,hidden_size_1)
    model_2['forward_flip']=trainer_obj.initialize_parameters_MDLSTM(input_size,hidden_size_1)
    model_2['backward_flip']=trainer_obj.initialize_parameters_MDLSTM(input_size,hidden_size_1)
    num_filters_2 = 6
    W = 4
    H = 2
    C = hidden_size_1
    model_conv={}
    model_conv['W_conv_frwd'] = np.random.rand(num_filters_2,C,W,H)
    model_conv['W_conv_bckd'] = np.random.rand(num_filters_2,C,W,H)
    model_conv['W_conv_frwd_flip'] = np.random.rand(num_filters_2,C,W,H)
    model_conv['W_conv_bckd_flip'] = np.random.rand(num_filters_2,C,W,H)
    model_conv['b_conv_frwd'] = bias_scale *  np.random.randn(num_filters_2)
    model_conv['b_conv_bckd'] = bias_scale *  np.random.randn(num_filters_2)
    model_conv['b_conv_frwd_flip'] = bias_scale *  np.random.randn(num_filters_2)
    model_conv['b_conv_bckd_flip'] = bias_scale *  np.random.randn(num_filters_2)
    model_2['conv'] = model_conv
    conv_param_2={'stride_W':4,'stride_H':2}

    #------------------------------------------------Model_3---------------------------------------------------------#

    model_3={}
    input_size = num_filters_2
    hidden_size_3 = 10
    model_3['forward']=trainer_obj.initialize_parameters_MDLSTM(input_size,hidden_size_3)
    model_3['backward']=trainer_obj.initialize_parameters_MDLSTM(input_size,hidden_size_3)
    model_3['forward_flip']=trainer_obj.initialize_parameters_MDLSTM(input_size,hidden_size_3)
    model_3['backward_flip']=trainer_obj.initialize_parameters_MDLSTM(input_size,hidden_size_3)
    num_filters_3 = 20
    W = 4
    H = 2
    C = hidden_size_3
    model_conv={}
    model_conv['W_conv_frwd'] = np.random.rand(num_filters_3,C,W,H)
    model_conv['W_conv_bckd'] = np.random.rand(num_filters_3,C,W,H)
    model_conv['W_conv_frwd_flip'] = np.random.rand(num_filters_3,C,W,H)
    model_conv['W_conv_bckd_flip'] = np.random.rand(num_filters_3,C,W,H)
    model_conv['b_conv_frwd'] = bias_scale * np.random.randn(num_filters_3)
    model_conv['b_conv_bckd'] = bias_scale * np.random.randn(num_filters_3)
    model_conv['b_conv_frwd_flip'] = bias_scale * np.random.randn(num_filters_3)
    model_conv['b_conv_bckd_flip'] = bias_scale * np.random.randn(num_filters_3)
    model_3['conv'] = model_conv

    conv_param_3={'stride_W':4,'stride_H':2}

    #-------------------------------------------------Model_4 ----------------------------------------------------------#

    model_4={}
    input_size = num_filters_3
    hidden_size_4 = 50
    model_4['forward']=trainer_obj.initialize_parameters_MDLSTM(input_size,hidden_size_4)
    model_4['backward']=trainer_obj.initialize_parameters_MDLSTM(input_size,hidden_size_4)
    model_4['forward_flip']=trainer_obj.initialize_parameters_MDLSTM(input_size,hidden_size_4)
    model_4['backward_flip']=trainer_obj.initialize_parameters_MDLSTM(input_size,hidden_size_4)
    model_temp={}
    model_temp['W_h_to_o_frwd'] = np.random.randn( hidden_size_4, vocab_size)/(np.sqrt( hidden_size_4))
    model_temp['b_h_to_o_frwd'] = bias_scale * np.random.randn(vocab_size)
    model_temp['W_h_to_o_bckd'] = np.random.randn( hidden_size_4, vocab_size)/(np.sqrt( hidden_size_4))
    model_temp['b_h_to_o_bckd'] = bias_scale * np.random.randn(vocab_size)
    model_temp['W_h_to_o_frwd_flip'] = np.random.randn( hidden_size_4, vocab_size)/(np.sqrt( hidden_size_4))
    model_temp['b_h_to_o_frwd_flip'] = bias_scale * np.random.randn(vocab_size)
    model_temp['W_h_to_o_bckd_flip'] = np.random.randn( hidden_size_4, vocab_size)/(np.sqrt( hidden_size_4))
    model_temp['b_h_to_o_bckd_flip'] = bias_scale * np.random.randn(vocab_size)
    model_4['ff'] = model_temp
    model={'model_2':model_2,'model_3':model_3,'model_4':model_4}
    dict_conv_param = {'conv_1':conv_param_1,'conv_2':conv_param_2,'conv_3':conv_param_3}

    #--------------------------------------------------------------------------------------------------------------------------------------#
    max_iter = len(list_data_train)
    print ('Training data =%d Validation data = %d') % (len(list_data_train),len(list_data_val))
    lr = args.learning_rate
    momentum = args.momentum
    reg = args.reg
    update = args.update
    batch_size = args.batch_size
    epochs = args.epochs
    cer = 0
    
    # Start Training
    for i in xrange (epochs):
        print '\nEpoc no = %d' % (i)
        epoch = i+1
        model,cer = trainer_obj. train (dict_data_train, list_data_train, dict_data_val, list_data_val, model,dict_conv_param,char_to_ix,ix_to_char,max_iter,lr,momentum,reg,batch_size,update,epoch,cer)


    print ('Finished Training CER on validation set = %f') % (cer)

if __name__=='__main__':
    main()
