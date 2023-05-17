# shuffle & data augmentation
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
pd.set_option('display.max_columns', None)

gene_dict = {'A':[0,0,0,1], 'T':[0,0,1,0], 'C':[0,1,0,0], 'G':[1,0,0,0], 
             'a':[0,0,0,1], 't':[0,0,1,0], 'c':[0,1,0,0], 'g':[1,0,0,0],
             'P':[0,0,0,0]} # padding

# 1-d label: sigmoid          
label_dict = {'0':0, # Enhancer
              '1':1, # Repressor
              }

def augmentation(model_size): # try data augmentation for train set
    path = '../datasets/' + model_size + '/train_' + model_size + '.pkl'
    out = '../datasets/' + model_size + '/train_aug_' + model_size + '.pkl'

    train_set = pd.read_pickle(path)
    #print(train_set)
    column_list = ['variant_51_seq', 'seq_between_variant_tss', 'variant_51_seq_after_mutation', 'atac_between', 'atac_variant_51', 'label']
    data_new_train = {'variant_51_seq':[],  'seq_between_variant_tss':[], 'variant_51_seq_after_mutation':[], 
                  'atac_between':[], 'atac_variant_51':[],  'label':[], 'slope':[]}

    for i in range(len(train_set)):
        variant_51_seq = train_set['variant_51_seq'].values[i]
        seq_between_variant_tss = train_set['seq_between_variant_tss'].values[i]    
        variant_51_seq_after_mutation = train_set['variant_51_seq_after_mutation'].values[i]
        seq_between_variant_tss_after_mutation = train_set['seq_between_variant_tss_after_mutation'].values[i]
        atac_between = train_set['atac_between'].values[i]
        atac_variant_51 = train_set['atac_variant_51'].values[i]
        label = train_set['label'].values[i]
        slope = float(train_set['slope'].values[i])
    
        # original 
        data_new_train['variant_51_seq'].append(variant_51_seq)
        data_new_train['seq_between_variant_tss'].append(seq_between_variant_tss)
        data_new_train['variant_51_seq_after_mutation'].append(variant_51_seq_after_mutation)
        data_new_train['atac_between'].append(atac_between)
        data_new_train['atac_variant_51'].append(atac_variant_51)
        data_new_train['label'].append(label)
        data_new_train['slope'].append(slope)
        
        # reverse
        if(str(label) == '0'):        
            data_new_train['variant_51_seq'].append(variant_51_seq_after_mutation)
            data_new_train['seq_between_variant_tss'].append(seq_between_variant_tss_after_mutation)
            data_new_train['variant_51_seq_after_mutation'].append(variant_51_seq)
            data_new_train['atac_between'].append(atac_between)
            data_new_train['atac_variant_51'].append(atac_variant_51)
            data_new_train['slope'].append(slope*(-1))
            data_new_train['label'].append('1')  
        elif(str(label) == '1'):
            data_new_train['variant_51_seq'].append(variant_51_seq_after_mutation)
            data_new_train['seq_between_variant_tss'].append(seq_between_variant_tss_after_mutation)
            data_new_train['variant_51_seq_after_mutation'].append(variant_51_seq)
            data_new_train['atac_between'].append(atac_between)
            data_new_train['atac_variant_51'].append(atac_variant_51)
            data_new_train['slope'].append(slope*(-1))
            data_new_train['label'].append('0')
        
    train_new = pd.DataFrame(data_new_train)
    train_new = shuffle(train_new)

    c_re, c_en = 0, 0
    for i in range(len(train_new)):
        label = train_new['label'].values[i]
        if(str(label) == '0'):
            c_re += 1
        elif(str(label) == '1'):
            c_en += 1  
    print(c_re)
    print(c_en)

    train_new.to_pickle(out)

def reverse_testset(model_size):
    path = '../datasets/' + model_size + '/test_' + model_size + '.pkl'
    out = '../datasets/' + model_size + '/test_rev_' + model_size + '.pkl'

    test_set = pd.read_pickle(path)
    #print(test_set)
    column_list = ['variant_51_seq', 'seq_between_variant_tss', 'variant_51_seq_after_mutation', 'atac_between', 'atac_variant_51', 'label']
    data_new_test = {'variant_51_seq':[],  'seq_between_variant_tss':[], 'variant_51_seq_after_mutation':[], 
                  'atac_between':[], 'atac_variant_51':[],  'label':[], 'slope':[]}

    for i in range(len(test_set)):
        variant_51_seq = test_set['variant_51_seq'].values[i]
        seq_between_variant_tss = test_set['seq_between_variant_tss'].values[i]    
        variant_51_seq_after_mutation = test_set['variant_51_seq_after_mutation'].values[i]
        seq_between_variant_tss_after_mutation = test_set['seq_between_variant_tss_after_mutation'].values[i]
        atac_between = test_set['atac_between'].values[i]
        atac_variant_51 = test_set['atac_variant_51'].values[i]
        label = test_set['label'].values[i]
        slope = float(test_set['slope'].values[i])
        
        # reverse
        if(str(label) == '0'):        
            data_new_test['variant_51_seq'].append(variant_51_seq_after_mutation)
            data_new_test['seq_between_variant_tss'].append(seq_between_variant_tss_after_mutation)
            data_new_test['variant_51_seq_after_mutation'].append(variant_51_seq)
            data_new_test['atac_between'].append(atac_between)
            data_new_test['atac_variant_51'].append(atac_variant_51)
            data_new_test['slope'].append(slope*(-1))
            data_new_test['label'].append('1')  
        elif(str(label) == '1'):
            data_new_test['variant_51_seq'].append(variant_51_seq_after_mutation)
            data_new_test['seq_between_variant_tss'].append(seq_between_variant_tss_after_mutation)
            data_new_test['variant_51_seq_after_mutation'].append(variant_51_seq)
            data_new_test['atac_between'].append(atac_between)
            data_new_test['atac_variant_51'].append(atac_variant_51)
            data_new_test['slope'].append(slope*(-1))
            data_new_test['label'].append('0')
        
    test_new = pd.DataFrame(data_new_test)
    test_new = shuffle(test_new)
    test_new.to_pickle(out)

def pkl2npy(model_size): # prepare .npy datasets with shuffle (easy to OOM, Data Generator is suggested)
    
    trainpath = '../datasets/' + model_size + '/train_' + model_size + '.pkl'
    testpath = '../datasets/' + model_size + '/test_' + model_size + '.pkl'

    train_pkl = pd.read_pickle(trainpath)
    test_pkl = pd.read_pickle(testpath)
    
    # check model size
    if(model_size == 'small'):
        padding = 1000
    elif(model_size == 'middle'):
        padding = 10000
    elif(model_size == 'large'):
        padding = 100000
    elif(model_size == 'huge'):
        padding = 1000000
    
    flg = 0
    for pkl in [train_pkl, test_pkl]:
        flg += 1
        before_51, after_51, bet_seq, atac_51, atac_bet, label = [], [], [], [], [], []
        for i in range(len(pkl)):

            ######## 51-d
            seq_list = []
            value = pkl['variant_51_seq'].values[i]
            for str in value:
                seq_list.append(gene_dict[str])
            before_51.append(seq_list)

            seq_list = []
            value = pkl['variant_51_seq_after_mutation'].values[i]
            for str in value:
                seq_list.append(gene_dict[str])
            after_51.append(seq_list)

            seq_list = []
            value = list(pkl['atac_variant_51'].values[i])
            for item in value:
                seq_list.append(item)
            atac_51.append(seq_list)

            ######## between: need padding
            seq_between_variant_tss = pkl['seq_between_variant_tss'].values[i]
            seq_between_variant_tss = seq_between_variant_tss.ljust(padding,'P')
            atac_between = list(pkl['atac_between'].values[i])
            atac_between = atac_between + [0] * (padding - len(atac_between))

            seq_list = []
            for str in seq_between_variant_tss:
                seq_list.append(gene_dict[str])
            bet_seq.append(seq_list)

            seq_list = []
            for item in seq_between_variant_tss:
                seq_list.append(item)
            atac_bet.append(seq_list)

            ######## label
            value = pkl['label'].values[i]
            label.append(value)

        # to npy before_51, after_51, bet_seq, atac_51, atac_bet, label
        np_before_51 = np.array(before_51)
        print(np_before_51.shape) # (sample, 51, 4)
        np_after_51 = np.array(after_51)
        print(np_after_51.shape) # (sample, 51, 4)
        np_bet_seq = np.array(bet_seq)
        print(np_bet_seq.shape) # (sample, 1000, 4)
        np_atac_51 = np.array(atac_51)
        print(np_atac_51.shape) # (sample, 51)
        np_atac_bet = np.array(atac_bet)
        print(np_atac_bet.shape) # (sample, 1000)
        np_label = np.array(label)
        print(np_label.shape) # (sample,)

        # save
        if(flg == 1):
            np.save('../datasets/' + model_size + '/ori/train_before_51.npy', np_before_51)
            np.save('../datasets/' + model_size + '/ori/train_after_51.npy', np_after_51)
            np.save('../datasets/' + model_size + '/ori/train_bet_seq.npy', np_bet_seq)
            np.save('../datasets/' + model_size + '/ori/train_atac_51.npy', np_atac_51)
            np.save('../datasets/' + model_size + '/ori/train_atac_bet.npy', np_atac_bet)
            np.save('../datasets/' + model_size + '/ori/train_label.npy', np_label)
        elif(flg == 2):
            np.save('../datasets/' + model_size + '/ori/test_before_51.npy', np_before_51)
            np.save('../datasets/' + model_size + '/ori/test_after_51.npy', np_after_51)
            np.save('../datasets/' + model_size + '/ori/test_bet_seq.npy', np_bet_seq)
            np.save('../datasets/' + model_size + '/ori/test_atac_51.npy', np_atac_51)
            np.save('../datasets/' + model_size + '/ori/test_atac_bet.npy', np_atac_bet)
            np.save('../datasets/' + model_size + '/ori/test_label.npy', np_label)

if __name__ == '__main__':
    
    augmentation('small')
    augmentation('middle')
    augmentation('large')
    augmentation('huge')

    reverse_testset('small')
    reverse_testset('middle')
    reverse_testset('large')
    reverse_testset('huge')

    '''
    pkl2npy('small')   
    pkl2npy('middle')
    #pkl2npy('large') # easy to OOM
    #pkl2npy('huge')  # easy to OOM    
    '''                                                                                                           