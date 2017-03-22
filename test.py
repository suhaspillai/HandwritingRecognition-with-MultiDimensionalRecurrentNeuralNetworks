#author : Suhas Pillai

import numpy as  np
from loadData import *
from trainer_copy import *
from PIL import Image
from pylab import *
import pdb
import pstats, cProfile
import argparse

parser  = argparse.ArgumentParser()
parser.add_argument('--learning_rate',default=0.001,type=float)
parser.add_argument('--momentum',default=0.9,type=float)
parser.add_argument('--reg',default=0.0,type=float)
parser.add_argument('--update',default='rmsprop')
parser.add_argument('--batch_size',default=200,type=int)
parser.add_argument('--epochs',default=50,type=int)
args = parser.parse_args()


def main():
    '''
    The script is used to test the system.
    '''
    #Load data
    file_dict_data = open('dict_data','rb')
    dict_data = cp.load(file_dict_data)
    f_test = open('testing_data','rb')
    list_data_test = cp.load(f_test) 
    dict_data_test={}

    for img in list_data_test:
        dict_data_test[img] = dict_data[img]
   
    file_chars = open('chars_data','rb')
    chars = cp.load(file_chars) 
    list_vocab = []
    list_vocab.append('blank')
    for i in xrange(len(chars)):
        list_vocab.append(chars.pop())
 
    char_to_ix = {list_vocab[i]:i for i in xrange(len(list_vocab))}
    ix_to_char = {i:list_vocab[i] for i in xrange(len(list_vocab))}
    vocab_size = len(char_to_ix)

    #Conv filters size and stride
    conv_param_1={'width':2,'height':2}
    conv_param_2={'stride_W':4,'stride_H':2}
    conv_param_3={'stride_W':4,'stride_H':2} 
    dict_conv_param = {'conv_1':conv_param_1,'conv_2':conv_param_2,'conv_3':conv_param_3}
    
    lr = args.learning_rate
    momentum = args.momentum
    reg = args.reg
    update = args.update
    batch_size = args.batch_size
    epochs = args.epochs
    max_iter = 0
    print 'Total data for testing  = %d ' % (len(list_data_test)-max_iter)
    print 'Loading model parameters'      
    model = cp.load(open('model_parameters','rb'))    # Parameters from the trained model.
    print '\n.....................Start Testing.....................\n'
    trainer_obj = Trainer() 
    cer = trainer_obj.cer_val(dict_data_test,list_data_test,model,dict_conv_param,char_to_ix,ix_to_char,max_iter,reg)
    print ('Percentage CER on Test set = %f') % (cer)

if __name__=='__main__':
    main()
