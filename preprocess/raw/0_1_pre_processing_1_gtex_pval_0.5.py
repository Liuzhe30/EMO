#fetch gtex pval>= 0.5

import pandas as pd
from pandas import read_parquet
from pandas import DataFrame
import pickle
pd.set_option('display.max_columns', None)

column_list = ['phenotype_id', 'variant_id', 'tss_distance', 'maf', 'ma_samples', 'ma_count', 'pval_nominal', 'slope', 'slope_se']

with open("0_gtex_list.txt") as r:
    lines = r.readlines()
    gtex_bulk_list = []
    for line in lines:
        gtex_bulk_list.append(line.strip())
    print(gtex_bulk_list)
    
file_path = 'E:/SNP_eQTL_rawdata/GTEx_Analysis_v8_EUR_eQTL_all_associations/'
for bulk in gtex_bulk_list:
    for chr_num in range(1, 24): # and chrX
        bulk_file_name = file_path + bulk + '/' + 'GTEx_Analysis_v8_QTLs_GTEx_Analysis_v8_EUR_eQTL_all_associations_' + bulk + '.v8.EUR.allpairs.chr' + str(chr_num) + '.parquet'
        print(bulk_file_name)
        data = read_parquet(bulk_file_name)
        print(data)
        new_df = pd.DataFrame(columns = column_list)
        for i in range(len(data)):
        #for i in range(4):
            if(float(data['pval_nominal'].values[i]) >= 0.5):
                print(data['pval_nominal'].values[i])
                #print(type(data.iloc[i]))
                new_df = new_df.append(data.iloc[i])
        new_df = new_df.reset_index()        
        print(new_df)
        new_file_name = 'D:/eQTL_SNP/dataset/0_pval_0.5/' + bulk + '_' + str(chr_num) + '.pkl'
        new_df.to_pickle(new_file_name)
    # chrX
    chr_num = 'X'
    bulk_file_name = file_path + bulk + '/' + 'GTEx_Analysis_v8_QTLs_GTEx_Analysis_v8_EUR_eQTL_all_associations_' + bulk + '.v8.EUR.allpairs.chr' + str(chr_num) + '.parquet'
    print(bulk_file_name)
    data = read_parquet(bulk_file_name)
    print(data)
    new_df = pd.DataFrame(columns = column_list)
    for i in range(len(data)):
    #for i in range(4):
        if(float(data['pval_nominal'].values[i]) >= 0.5):
            print(data['pval_nominal'].values[i])
            #print(type(data.iloc[i]))
            new_df = new_df.append(data.iloc[i])
    new_df = new_df.reset_index()        
    print(new_df)
    new_file_name = 'D:/eQTL_SNP/dataset/0_pval_0.5/' + bulk + '_' + str(chr_num) + '.pkl'
    new_df.to_pickle(new_file_name)            