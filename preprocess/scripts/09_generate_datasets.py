import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)

train_all = pd.read_pickle('../../datasets/train.pkl')
test_all = pd.read_pickle('../../datasets/test.pkl')

# 1-1000 small
train_df = pd.DataFrame()
test_df = pd.DataFrame()

for i in range(len(train_all)):        
    tss_distance = abs(int(train_all['tss_distance'].values[i]))
    if(tss_distance < 1000):
        train_df = pd.concat([train_df, train_all[i:i+1]])
train_df = train_df.reset_index(drop=True)

for i in range(len(test_all)):        
    tss_distance = abs(int(test_all['tss_distance'].values[i]))
    if(tss_distance < 1000):
        test_df = pd.concat([test_df, test_all[i:i+1]])
test_df = test_df.reset_index(drop=True)

train_df.to_pickle('../../datasets/small/train_small.pkl')
test_df.to_pickle('../../datasets/small/test_small.pkl')

# 1001-10000 middle
train_df = pd.DataFrame()
test_df = pd.DataFrame()

for i in range(len(train_all)):        
    tss_distance = abs(int(train_all['tss_distance'].values[i]))
    if(tss_distance >= 1000 and tss_distance < 10000):
        train_df = pd.concat([train_df, train_all[i:i+1]])
train_df = train_df.reset_index(drop=True)

for i in range(len(test_all)):        
    tss_distance = abs(int(test_all['tss_distance'].values[i]))
    if(tss_distance >= 1000 and tss_distance < 10000):
        test_df = pd.concat([test_df, test_all[i:i+1]])
test_df = test_df.reset_index(drop=True)

train_df.to_pickle('../../datasets/middle/train_middle.pkl')
test_df.to_pickle('../../datasets/middle/test_middle.pkl')

# 10001-100000 large
train_df = pd.DataFrame()
test_df = pd.DataFrame()

for i in range(len(train_all)):        
    tss_distance = abs(int(train_all['tss_distance'].values[i]))
    if(tss_distance >= 10000 and tss_distance < 100000):
        train_df = pd.concat([train_df, train_all[i:i+1]])
train_df = train_df.reset_index(drop=True)

for i in range(len(test_all)):        
    tss_distance = abs(int(test_all['tss_distance'].values[i]))
    if(tss_distance >= 10000 and tss_distance < 100000):
        test_df = pd.concat([test_df, test_all[i:i+1]])
test_df = test_df.reset_index(drop=True)

train_df.to_pickle('../../datasets/large/train_large.pkl')
test_df.to_pickle('../../datasets/large/test_large.pkl')

# 100001-1000000 huge
train_df = pd.DataFrame()
test_df = pd.DataFrame()

for i in range(len(train_all)):        
    tss_distance = abs(int(train_all['tss_distance'].values[i]))
    if(tss_distance >= 100000):
        train_df = pd.concat([train_df, train_all[i:i+1]])
train_df = train_df.reset_index(drop=True)

for i in range(len(test_all)):        
    tss_distance = abs(int(test_all['tss_distance'].values[i]))
    if(tss_distance >= 100000):
        test_df = pd.concat([test_df, test_all[i:i+1]])
test_df = test_df.reset_index(drop=True)

train_df.to_pickle('../../datasets/huge/train_huge.pkl')
test_df.to_pickle('../../datasets/huge/test_huge.pkl')