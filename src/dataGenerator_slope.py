# class for data generation
import pandas as pd
import numpy as np
import copy
pd.set_option('display.max_columns', None)

class dataGenerator():

    def __init__(self, dataset, batch_size, model_size):
        self.dataset = dataset
        self.batch_size = batch_size
        self.model_size = model_size

        self.gene_dict = {'A':[0,0,0,1], 'T':[0,0,1,0], 'C':[0,1,0,0], 'G':[1,0,0,0], 
             'a':[0,0,0,1], 't':[0,0,1,0], 'c':[0,1,0,0], 'g':[1,0,0,0],
             'N':[0,0,0,0], 'n':[0,0,0,0],
             'P':[0,0,0,0]} # padding

        self.mask_dict = {'A':1, 'T':1, 'C':1, 'G':1, 
             'a':1, 't':1, 'c':1, 'g':1,
             'N':0, 'n':0,
             'P':0} # padding

    def check_padding(self):
        
        # check model size
        if(self.model_size == 'small'):
            padding = 1000
        elif(self.model_size == 'middle'):
            padding = 10000
        elif(self.model_size == 'large'):
            padding = 100000
        elif(self.model_size == 'huge'):
            padding = 1000000
        
        return padding

    def generate_batch(self):

        # fetch padding
        padding = self.check_padding()

        while 1:
            i = 0
            while i < (len(self.dataset) - self.batch_size):
                before_51, after_51, bet_seq, atac_51, atac_bet, dataY_batch, mask1_batch, mask2_batch = [], [], [], [], [], [], [], []
                for j in range(i, i + self.batch_size):

                    # generate label
                    label = float(self.dataset['slope'].values[j])
                    dataY_batch.append(label)

                    # generate input_51 
                    seq_list = []
                    value = self.dataset['variant_51_seq'].values[j]
                    for strr in value:
                        seq_list.append(self.gene_dict[strr])
                    before_51.append(seq_list)

                    seq_list = []
                    value = self.dataset['variant_51_seq_after_mutation'].values[j] 
                    for strr in value:
                        seq_list.append(self.gene_dict[strr])
                    after_51.append(seq_list)

                    seq_list = []
                    value = list(self.dataset['atac_variant_51'].values[j])
                    for item in value:
                        seq_list.append(item)
                    atac_51.append(seq_list)

                    # generate mask_51 
                    mask1_sample = []
                    for idx in range(51):
                        line = copy.copy(self.mask_dict[self.dataset['variant_51_seq'].values[j][idx]])  
                        mask1_sample.append(line)
                    mask1_batch.append(mask1_sample)  

                    # generate input_bet
                    seq_between_variant_tss = self.dataset['seq_between_variant_tss'].values[j]
                    seq_between_variant_tss = seq_between_variant_tss.ljust(padding,'P')
                    atac_between = list(self.dataset['atac_between'].values[j])
                    atac_between = atac_between + [0] * (padding - len(atac_between))   

                    seq_list = []
                    for strr in seq_between_variant_tss:
                        seq_list.append(self.gene_dict[strr])
                    bet_seq.append(seq_list)

                    seq_list = []
                    for item in atac_between:
                        seq_list.append(item)
                    atac_bet.append(seq_list)

                    # generate mask_bet
                    mask2_sample = []
                    for idx in range(padding):
                        line = copy.copy(self.mask_dict[seq_between_variant_tss[idx]])   
                        mask2_sample.append(line)
                    mask2_batch.append(mask2_sample)  

                input_before_51 = np.array(before_51)
                input_after_51 = np.array(after_51)
                input_atac_51 = np.array(atac_51)
                input_bet_seq = np.array(bet_seq)
                input_atac_bet = np.array(atac_bet)
                y = np.array(dataY_batch)
                input_mask1 = np.array(mask1_batch)
                input_mask2 = np.array(mask2_batch)

                i += self.batch_size
                
                yield ([input_before_51, input_after_51, input_atac_51, input_bet_seq, input_atac_bet, input_mask1, input_mask2], y)

    def generate_validation(self):

        # fetch padding
        padding = self.check_padding()
        
        before_51, after_51, bet_seq, atac_51, atac_bet, dataY_batch, mask1_batch, mask2_batch = [], [], [], [], [], [], [], []
        for j in range(len(self.dataset)):

            # generate label
            label = float(self.dataset['slope'].values[j])
            dataY_batch.append(label)

            # generate input_51 
            seq_list = []
            value = self.dataset['variant_51_seq'].values[j]
            for strr in value:
                seq_list.append(self.gene_dict[strr])
            before_51.append(seq_list)

            seq_list = []
            value = self.dataset['variant_51_seq_after_mutation'].values[j] 
            for strr in value:
                seq_list.append(self.gene_dict[strr])
            after_51.append(seq_list)

            seq_list = []
            value = list(self.dataset['atac_variant_51'].values[j])
            for item in value:
                seq_list.append(item)
            atac_51.append(seq_list)

            # generate mask_51 
            mask1_sample = []
            for idx in range(51):
                line = copy.copy(self.mask_dict[self.dataset['variant_51_seq'].values[j][idx]])  
                mask1_sample.append(line)
            mask1_batch.append(mask1_sample)  

            # generate input_bet
            seq_between_variant_tss = self.dataset['seq_between_variant_tss'].values[j]
            seq_between_variant_tss = seq_between_variant_tss.ljust(padding,'P')
            atac_between = list(self.dataset['atac_between'].values[j])
            atac_between = atac_between + [0] * (padding - len(atac_between))   

            seq_list = []
            for strr in seq_between_variant_tss:
                seq_list.append(self.gene_dict[strr])
            bet_seq.append(seq_list)

            seq_list = []
            for item in atac_between:
                seq_list.append(item)
            atac_bet.append(seq_list)

            # generate mask_bet
            mask2_sample = []
            for idx in range(padding):
                line = copy.copy(self.mask_dict[seq_between_variant_tss[idx]])   
                mask2_sample.append(line)
            mask2_batch.append(mask2_sample)  

        input_before_51 = np.array(before_51)
        input_after_51 = np.array(after_51)
        input_atac_51 = np.array(atac_51)
        input_bet_seq = np.array(bet_seq)
        input_atac_bet = np.array(atac_bet)
        y = np.array(dataY_batch)
        input_mask1 = np.array(mask1_batch)
        input_mask2 = np.array(mask2_batch)

        return ([input_before_51, input_after_51, input_atac_51, input_bet_seq, input_atac_bet, input_mask1, input_mask2], y)