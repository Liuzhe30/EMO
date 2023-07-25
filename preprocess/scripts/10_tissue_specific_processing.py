import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)

with open("/data/eqtl/datasets/gtex_list.txt") as r:
    lines = r.readlines()
    gtex_bulk_list = []
    for line in lines:
        gtex_bulk_list.append(line.strip())
    print(gtex_bulk_list)

for tissue in gtex_bulk_list:

    train_all = pd.read_pickle('/data/eqtl/datasets/train/' + tissue + '.pkl')
    test_all = pd.read_pickle('/data/eqtl/datasets/test/' + tissue + '.pkl')

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

    train_df.to_pickle('/data/eqtl/datasets/tissue_specific/small/train_small_' + tissue + '.pkl')
    test_df.to_pickle('/data/eqtl/datasets/tissue_specific/small/test_small_' + tissue + '.pkl')
    print(tissue, train_df.shape[0])

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

    train_df.to_pickle('/data/eqtl/datasets/tissue_specific/middle/train_middle_' + tissue + '.pkl')
    test_df.to_pickle('/data/eqtl/datasets/tissue_specific/middle/test_middle_' + tissue + '.pkl')
    print(tissue, train_df.shape[0])

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

    train_df.to_pickle('/data/eqtl/datasets/tissue_specific/large/train_large_' + tissue + '.pkl')
    test_df.to_pickle('/data/eqtl/datasets/tissue_specific/large/test_large_' + tissue + '.pkl')
    print(tissue, train_df.shape[0])