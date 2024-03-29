# split 5-fold dataset
import pandas as pd
from pandas import read_parquet
pd.set_option('display.max_columns', None)
#pd.set_option('display.max_rows', None)

gene_dict = {'A':[0,0,0,1], 'T':[0,0,1,0], 'C':[0,1,0,0], 'G':[1,0,0,0], 
             'a':[0,0,0,1], 't':[0,0,1,0], 'c':[0,1,0,0], 'g':[1,0,0,0],
             } 

# dataset path
data_path_enhancer = '/data/eqtl/datasets/enhancer/'
data_path_repressor = '/data/eqtl/datasets/repressor/'

with open("../gtex_list.txt") as r:
    lines = r.readlines()
    gtex_bulk_list = []
    for line in lines:
        gtex_bulk_list.append(line.strip())
    print(gtex_bulk_list)

train_all_df = pd.DataFrame()
test_all_df = pd.DataFrame()
for bulk in gtex_bulk_list:
    data_en = pd.read_pickle(data_path_enhancer + bulk + '.pkl')
    data_en.insert(data_en.shape[1], 'bulk', bulk)
    data_en.insert(data_en.shape[1], 'label', 1)

    data_re = pd.read_pickle(data_path_repressor + bulk + '.pkl')
    data_re.insert(data_re.shape[1], 'bulk', bulk)
    data_re.insert(data_re.shape[1], 'label', 0)

    merged_df = pd.concat([data_en, data_re])
    merged_df = merged_df.drop(['atac_tss_51', 'tss_51_seq','level_0','index'], axis=1)

    # delete wrong sequence
    del_row = []
    for i in range(len(merged_df)):
        for str in merged_df['variant_51_seq'].values[i]:
            if(str not in gene_dict.keys() and merged_df['variant_id'].values[i] not in del_row):
                del_row.append(merged_df['variant_id'].values[i])
        for str in merged_df['variant_51_seq_after_mutation'].values[i]:
            if(str not in gene_dict.keys() and merged_df['variant_id'].values[i] not in del_row):
                del_row.append(merged_df['variant_id'].values[i])   
        for str in merged_df['seq_between_variant_tss'].values[i]:
            if(str not in gene_dict.keys() and merged_df['variant_id'].values[i] not in del_row):
                del_row.append(merged_df['variant_id'].values[i])  
    #print(del_row)
    for i in del_row:
        merged_df = merged_df[~(merged_df['variant_id'] == i)]

    shuffle_df = merged_df.sample(frac=1).reset_index(drop=True)

    train_df = shuffle_df[0:int(shuffle_df.shape[0]*0.9)].reset_index(drop=True)
    train_df.to_pickle('/data/eqtl/datasets/train/' + bulk + '.pkl')
    print(bulk, 'train', train_df.shape)

    train_all_df = pd.concat([train_all_df, train_df])

    test_df = shuffle_df[int(shuffle_df.shape[0]*0.9):].reset_index(drop=True)
    test_df.to_pickle('/data/eqtl/datasets/test/' + bulk + '.pkl')
    print(bulk, 'test', test_df.shape)

    test_all_df = pd.concat([test_all_df, test_df])
    
print('train all', train_all_df.shape)
train_all_df.to_pickle('/data/eqtl/datasets/train.pkl')
print('test all', test_all_df.shape)
test_all_df.to_pickle('/data/eqtl/datasets/test.pkl')
print(test_all_df)