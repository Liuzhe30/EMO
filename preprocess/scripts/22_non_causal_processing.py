import numpy as np
import pandas as pd
import math
pd.set_option('display.max_columns', None)

non_causal_path = '../../4_ATAC_0.001_merged/'
output_path = '../../datasets_new/non_causal/'
with open("../causal_gtex_list.txt") as r:
    lines = r.readlines()
    gtex_bulk_list = []
    for line in lines:
        gtex_bulk_list.append(line.strip())
    print(gtex_bulk_list)

# step 1: slope filtering
filtered_df = pd.DataFrame()
for bulk in gtex_bulk_list:
    print(bulk)
    bulk_df = pd.read_pickle(non_causal_path + bulk + '.pkl')
    bulk_df['slope'] = bulk_df['slope'].astype(float)
    slope_df = bulk_df[(bulk_df['slope']>-0.5) & (bulk_df['slope']<0.5)]
    filtered_df = pd.concat([filtered_df,slope_df])
shuffle_df = filtered_df.sample(frac=1).reset_index(drop=True)
shuffle_df.to_pickle(output_path + 'slope_0.5.pkl')

# step 2: generate datasets
test_all = pd.read_pickle(output_path + 'slope_0.5.pkl')

# small model
test_df = pd.DataFrame()
for i in range(len(test_all)):        
    tss_distance = abs(int(test_all['tss_distance'].values[i]))
    if(tss_distance < 1000):
        test_df = pd.concat([test_df, test_all[i:i+1]])
test_df = test_df.reset_index(drop=True)
test_df.to_pickle(output_path + 'small/test_small.pkl')

# middle model
test_df = pd.DataFrame()
for i in range(len(test_all)):        
    tss_distance = abs(int(test_all['tss_distance'].values[i]))
    if(tss_distance >= 1000 and tss_distance < 10000):
        test_df = pd.concat([test_df, test_all[i:i+1]])
test_df = test_df.reset_index(drop=True)
test_df.to_pickle(output_path + 'middle/test_middle.pkl')

# large model
test_df = pd.DataFrame()
for i in range(len(test_all)):        
    tss_distance = abs(int(test_all['tss_distance'].values[i]))
    if(tss_distance >= 10000 and tss_distance < 100000):
        test_df = pd.concat([test_df, test_all[i:i+1]])
test_df = test_df.reset_index(drop=True)
test_df.to_pickle(output_path + 'large/test_large.pkl')

# huge model
test_df = pd.DataFrame()
for i in range(len(test_all)):        
    tss_distance = abs(int(test_all['tss_distance'].values[i]))
    if(tss_distance >= 100000):
        test_df = pd.concat([test_df, test_all[i:i+1]])
test_df = test_df.reset_index(drop=True)
test_df.to_pickle(output_path + 'huge/test_huge.pkl')
