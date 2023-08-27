import os
import numpy as np
import pandas as pd

gene_dict = {'A':[0,0,0,1], 'T':[0,0,1,0], 'C':[0,1,0,0], 'G':[1,0,0,0], 
             'a':[0,0,0,1], 't':[0,0,1,0], 'c':[0,1,0,0], 'g':[1,0,0,0],
             } 

output_path = '/data/eqtl/sceQTL/datasets/'
file_path = '/data/eqtl/sceQTL/merged/'

cell_list = ['B_Naive','CD4_memory','CD4_naive','CD8_memory','Dendritic_Cell','Natural_killer_Cell']

down_dict = {'B_Naive':938,'CD4_memory':1772,'CD4_naive':1123,
            'CD8_memory':1153,'Dendritic_Cell':977,'Natural_killer_Cell':1535}

# 1 delete wrong sequence & down-sampling
#'''
for cell in cell_list:
    print(cell)
    data = pd.read_pickle(file_path + cell + '_seq_0.2_atac.pkl')
    new_data = data[0:1]
    down_num = 0
    print(data['label'].value_counts())
    for i in range(1,len(data)):     
        if(int(data['label'][i]) == 0 and down_num < down_dict[cell]): # [938,1772,1123/1123,1153,977,1535]
            down_num += 1
            new_data = pd.concat([new_data,data[i:i+1]])
        elif(int(data['label'][i]) == 1):
            new_data = pd.concat([new_data,data[i:i+1]])
    
    new_data = new_data.reset_index(drop=True)
    print(new_data)
    print(new_data['label'].value_counts())

    for i in range(0,len(new_data)):     
        del_row = []
        for str in new_data['variant_51_seq'].values[i]:
            if(str not in gene_dict.keys() and new_data['RSID'].values[i] not in del_row):
                del_row.append(new_data['RSID'][i])
        for str in new_data['variant_51_seq_after_mutation'].values[i]:
            if(str not in gene_dict.keys() and new_data['RSID'].values[i] not in del_row):
                del_row.append(new_data['RSID'][i])  
        for str in new_data['seq_between_variant_tss'].values[i]:
            if(str not in gene_dict.keys() and new_data['RSID'].values[i] not in del_row):
                del_row.append(new_data['RSID'][i])
    
    for idx in del_row:
        new_data = new_data.drop(new_data[new_data['RSID']==idx].index)
    new_data = new_data.reset_index(drop=True)

    
    print(new_data)
    print(new_data['label'].value_counts())
    new_data.to_pickle('/data/eqtl/sceQTL/merged/' + cell + '_final.pkl')
#'''

# 2 shuffle, split dataset with sequence length

for cell in cell_list:
    print(cell)
    data = pd.read_pickle(file_path + cell + '_final.pkl')

    # small
    df = pd.DataFrame()
    for i in range(len(data)):        
        position = data['POS'][i]
        tss_position = data['TSS'][i]
        tss_distance = abs(position - tss_position)
        if(tss_distance < 1000):
            df = pd.concat([df, data[i:i+1]])

    df = df.sample(frac=1).reset_index(drop=True)
    train = df[0:int(0.9*df.shape[0])].reset_index(drop=True)
    test = df[int(0.9*df.shape[0]):].reset_index(drop=True)
    print(train.shape[0])
    print(test.shape[0])

    train.to_pickle(output_path + cell + '_train_small.pkl')
    test.to_pickle(output_path + cell + '_test_small.pkl')

    # middle
    df = pd.DataFrame()
    for i in range(len(data)):        
        position = data['POS'][i]
        tss_position = data['TSS'][i]
        tss_distance = abs(position - tss_position)
        if(tss_distance >= 1000 and tss_distance < 10000):
            df = pd.concat([df, data[i:i+1]])

    df = df.sample(frac=1).reset_index(drop=True)
    train = df[0:int(0.9*df.shape[0])].reset_index(drop=True)
    test = df[int(0.9*df.shape[0]):].reset_index(drop=True)
    print(train.shape[0])
    print(test.shape[0])

    train.to_pickle(output_path + cell + '_train_middle.pkl')
    test.to_pickle(output_path + cell + '_test_middle.pkl')

    # large
    df = pd.DataFrame()
    for i in range(len(data)):        
        position = data['POS'][i]
        tss_position = data['TSS'][i]
        tss_distance = abs(position - tss_position)
        if(tss_distance >= 10000 and tss_distance < 100000):
            df = pd.concat([df, data[i:i+1]])

    df = df.sample(frac=1).reset_index(drop=True)
    train = df[0:int(0.9*df.shape[0])].reset_index(drop=True)
    test = df[int(0.9*df.shape[0]):].reset_index(drop=True)
    print(train.shape[0])
    print(test.shape[0])

    train.to_pickle(output_path + cell + '_train_large.pkl')
    test.to_pickle(output_path + cell + '_test_large.pkl')

    # huge
    df = pd.DataFrame()
    for i in range(len(data)):        
        position = data['POS'][i]
        tss_position = data['TSS'][i]
        tss_distance = abs(position - tss_position)
        if(tss_distance >= 100000):
            df = pd.concat([df, data[i:i+1]])

    df = df.sample(frac=1).reset_index(drop=True)
    train = df[0:int(0.9*df.shape[0])].reset_index(drop=True)
    test = df[int(0.9*df.shape[0]):].reset_index(drop=True)
    print(train.shape[0])
    print(test.shape[0])

    train.to_pickle(output_path + cell + '_train_huge.pkl')
    test.to_pickle(output_path + cell + '_test_huge.pkl')
