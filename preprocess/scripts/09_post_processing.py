# post process the 'nan' in ATAC-seq
import numpy as np
import pandas as pd
import math
pd.set_option('display.max_columns', None)

def check(check_list):
    for elem in check_list:
        if(float(elem) == np.nan):
            return 0
        if(math.isnan(elem) == True):
            return 0
    return 1

# small model
train_df = pd.read_pickle('../../datasets/small/train_small.pkl')
train_df_check = pd.read_pickle('../../datasets/small/train_small.pkl')
test_df = pd.read_pickle('../../datasets/small/test_small.pkl')
test_df_check = pd.read_pickle('../../datasets/small/test_small.pkl')

print(train_df.shape)
for i in range(train_df_check.shape[0]):
    variant_id = train_df_check['variant_id'].values[i]
    slope = train_df_check['slope'].values[i]
    seq_between_variant_tss = train_df_check['seq_between_variant_tss'][i]
    atac_between = train_df_check['atac_between'][i]
    atac_variant_51 = list(train_df_check['atac_variant_51'][i])
    if(check(atac_between)==0 or check(atac_variant_51)==0):
        train_df = train_df[~(train_df['variant_id'].isin([variant_id])&train_df['slope'].isin([slope]))]
train_df = train_df.reset_index(drop=True)
train_df.to_pickle('../../datasets/small/train_small_post.pkl')
print(train_df.shape)

for i in range(test_df_check.shape[0]):
    variant_id = test_df_check['variant_id'].values[i]
    slope = test_df_check['slope'].values[i]
    seq_between_variant_tss = test_df_check['seq_between_variant_tss'][i]
    atac_between = test_df_check['atac_between'][i]
    atac_variant_51 = test_df_check['atac_variant_51'][i]
    if(check(atac_between)==0 or check(atac_variant_51)==0):
        test_df = test_df[~(test_df['variant_id'].isin([variant_id])&test_df['slope'].isin([slope]))]
test_df = test_df.reset_index(drop=True)
test_df.to_pickle('../../datasets/small/test_small_post.pkl')

# middle model
train_df = pd.read_pickle('../../datasets/middle/train_middle.pkl')
train_df_check = pd.read_pickle('../../datasets/middle/train_middle.pkl')
test_df = pd.read_pickle('../../datasets/middle/test_middle.pkl')
test_df_check = pd.read_pickle('../../datasets/middle/test_middle.pkl')

print(train_df.shape)
for i in range(train_df_check.shape[0]):
    variant_id = train_df_check['variant_id'].values[i]
    slope = train_df_check['slope'].values[i]
    seq_between_variant_tss = train_df_check['seq_between_variant_tss'][i]
    atac_between = train_df_check['atac_between'][i]
    atac_variant_51 = list(train_df_check['atac_variant_51'][i])
    if(check(atac_between)==0 or check(atac_variant_51)==0):
        train_df = train_df[~(train_df['variant_id'].isin([variant_id])&train_df['slope'].isin([slope]))]
train_df = train_df.reset_index(drop=True)
train_df.to_pickle('../../datasets/middle/train_middle_post.pkl')
print(train_df.shape)

for i in range(test_df_check.shape[0]):
    variant_id = test_df_check['variant_id'].values[i]
    slope = test_df_check['slope'].values[i]
    seq_between_variant_tss = test_df_check['seq_between_variant_tss'][i]
    atac_between = test_df_check['atac_between'][i]
    atac_variant_51 = test_df_check['atac_variant_51'][i]
    if(check(atac_between)==0 or check(atac_variant_51)==0):
        test_df = test_df[~(test_df['variant_id'].isin([variant_id])&test_df['slope'].isin([slope]))]
test_df = test_df.reset_index(drop=True)
test_df.to_pickle('../../datasets/middle/test_middle_post.pkl')

# large model
train_df = pd.read_pickle('../../datasets/large/train_large.pkl')
train_df_check = pd.read_pickle('../../datasets/large/train_large.pkl')
test_df = pd.read_pickle('../../datasets/large/test_large.pkl')
test_df_check = pd.read_pickle('../../datasets/large/test_large.pkl')

print(train_df.shape)
for i in range(train_df_check.shape[0]):
    variant_id = train_df_check['variant_id'].values[i]
    slope = train_df_check['slope'].values[i]
    seq_between_variant_tss = train_df_check['seq_between_variant_tss'][i]
    atac_between = train_df_check['atac_between'][i]
    atac_variant_51 = list(train_df_check['atac_variant_51'][i])
    if(check(atac_between)==0 or check(atac_variant_51)==0):
        train_df = train_df[~(train_df['variant_id'].isin([variant_id])&train_df['slope'].isin([slope]))]
train_df = train_df.reset_index(drop=True)
train_df.to_pickle('../../datasets/large/train_large_post.pkl')
print(train_df.shape)

for i in range(test_df_check.shape[0]):
    variant_id = test_df_check['variant_id'].values[i]
    slope = test_df_check['slope'].values[i]
    seq_between_variant_tss = test_df_check['seq_between_variant_tss'][i]
    atac_between = test_df_check['atac_between'][i]
    atac_variant_51 = test_df_check['atac_variant_51'][i]
    if(check(atac_between)==0 or check(atac_variant_51)==0):
        test_df = test_df[~(test_df['variant_id'].isin([variant_id])&test_df['slope'].isin([slope]))]
test_df = test_df.reset_index(drop=True)
test_df.to_pickle('../../datasets/large/test_large_post.pkl')

# huge model
train_df = pd.read_pickle('../../datasets/huge/train_huge.pkl')
train_df_check = pd.read_pickle('../../datasets/huge/train_huge.pkl')
test_df = pd.read_pickle('../../datasets/huge/test_huge.pkl')
test_df_check = pd.read_pickle('../../datasets/huge/test_huge.pkl')

print(train_df.shape)
for i in range(train_df_check.shape[0]):
    variant_id = train_df_check['variant_id'].values[i]
    slope = train_df_check['slope'].values[i]
    seq_between_variant_tss = train_df_check['seq_between_variant_tss'][i]
    atac_between = train_df_check['atac_between'][i]
    atac_variant_51 = list(train_df_check['atac_variant_51'][i])
    if(check(atac_between)==0 or check(atac_variant_51)==0):
        train_df = train_df[~(train_df['variant_id'].isin([variant_id])&train_df['slope'].isin([slope]))]
train_df = train_df.reset_index(drop=True)
train_df.to_pickle('../../datasets/huge/train_huge_post.pkl')
print(train_df.shape)

for i in range(test_df_check.shape[0]):
    variant_id = test_df_check['variant_id'].values[i]
    slope = test_df_check['slope'].values[i]
    seq_between_variant_tss = test_df_check['seq_between_variant_tss'][i]
    atac_between = test_df_check['atac_between'][i]
    atac_variant_51 = test_df_check['atac_variant_51'][i]
    if(check(atac_between)==0 or check(atac_variant_51)==0):
        test_df = test_df[~(test_df['variant_id'].isin([variant_id])&test_df['slope'].isin([slope]))]
test_df = test_df.reset_index(drop=True)
test_df.to_pickle('../../datasets/huge/test_huge_post.pkl')