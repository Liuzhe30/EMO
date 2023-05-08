# prepare .npy datasets with shuffle
import pandas as pd
import numpy as np
pd.set_option('display.max_columns', None)

gene_dict = {'A':[0,0,0,1], 'T':[0,0,1,0], 'C':[0,1,0,0], 'G':[1,0,0,0], 
             'a':[0,0,0,1], 't':[0,0,1,0], 'c':[0,1,0,0], 'g':[1,0,0,0],
             'P':[0,0,0,0]} # padding

# 1-d label: sigmoid          
label_dict = {'0':0, # Enhancer
              '1':1, # Repressor
              }

def pkl2npy(model_size):
    
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


def augmentation():
    pass


if __name__ == '__main__':
    
    pkl2npy('small')   
    pkl2npy('middle')
    pkl2npy('large')
    pkl2npy('huge')                                                                                                                 