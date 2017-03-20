from loadData import *
import cPickle as cp
import sys


'''
The script is used for creating training, validation and testing data.
'''

def main(argv):
    
    load_obj = loadData()
    file_xml_path = argv[0]
    file_location = argv[1] 
    no_of_train = int(argv[2])
    no_of_val = int(argv[3]) 
    dict_data,chars,list_data = load_obj.loadData_word(file_location,file_xml_path)    
    len_dict = len(dict_data)
    list_data_random = []
    dict_data_random={}
    while(len(list_data)>0):
        rand_seed = np.random.randint(len(list_data))
        img_id = list_data[rand_seed]
        list_data_random.append(img_id)
        dict_data_random[img_id] = dict_data[img_id]
        list_data.remove(img_id)
    
    train_data = list_data_random[:no_of_train]
    val_data = list_data_random[no_of_train:no_of_train+no_of_val] 
    test_data = list_data_random[no_of_train+no_of_val:]
    print ('Total no of training samples created =%d') % (len(train_data))
    print ('Total no of validation samples created =%d') % (len(val_data)) 
    print ('Total no of testing samples created =%d') % (len(test_data))
    file_dict_data = open('dict_data','wb')  
    file_data_chars = open('chars_data','wb')
    file_data_train = open('training_data','wb')
    file_data_val = open('validation_data','wb') 
    file_data_test = open('testing_data','wb')
    cp.dump(dict_data_random,file_dict_data)
    cp.dump(train_data,file_data_train)
    cp.dump(val_data,file_data_val) 
    cp.dump(test_data,file_data_test)  
    cp.dump(chars,file_data_chars)
    file_dict_data.close()
    file_data_train.close() 
    file_data_val.close()
    file_data_test.close()
    file_data_chars.close()
    print 'Finished Creating data'

if __name__=='__main__':
    main(sys.argv[1:])
