#fetch fine-mapping ppc>=0.9 & ppc<=0.001

import pandas as pd
from pandas import read_parquet
from pandas import DataFrame
import pickle
pd.set_option('display.max_columns', None)

column_list = ['phenotype_id', 'variant_id', 'tss_distance', 'maf', 'ma_samples', 'ma_count', 'pval_nominal', 'slope', 'slope_se']

with open("../gtex_list.txt") as r:
    lines = r.readlines()
    gtex_bulk_list = []
    for line in lines:
        gtex_bulk_list.append(line.strip())
    print(gtex_bulk_list)
    
# gtex_bulk_list = ['Colon_Transverse', 'Thyroid', 'Nerve_Tibial'] # split for multithreading working
    
file_path_1 = '../../datasets/GTEx_Analysis_v8_EUR_eQTL_all_associations/'
file_path_2 = '../../datasets/caviar_output_GTEx_LD/'
for bulk in gtex_bulk_list:
    for chr_num in range(1, 23): # ignore chrX
        bulk_file_name = file_path_1 + bulk + '/' + 'GTEx_Analysis_v8_QTLs_GTEx_Analysis_v8_EUR_eQTL_all_associations_' + bulk + '.v8.EUR.allpairs.chr' + str(chr_num) + '.parquet'
        print(bulk_file_name)
        data = read_parquet(bulk_file_name)
        print(data)
        new_df_up = pd.DataFrame(columns = column_list)
        new_df_down = pd.DataFrame(columns = column_list)
        for i in range(len(data)):
        #for i in range(100):
            if(float(data['pval_nominal'].values[i]) < 0.5):
                #new_df = new_df.append(data.iloc[i])
                phenotype_id = data['phenotype_id'].values[i]
                variant_id = data['variant_id'].values[i]
                chr_str = variant_id.split('_')[0]
                position = variant_id.split('_')[1]
                #print(phenotype_id)
                try:
                    with open(file_path_2 + bulk + '.allpairs.txt/' + chr_str + '/post/' + phenotype_id + '_new.out_post') as file:
                        lines = file.readlines()
                        for line in lines:
                            if(line.split()[0] == position and float(line.split()[2]) >= 0.9):
                                print(line)
                                new_df_up = new_df_up.append(data.iloc[i])
                            elif(line.split()[0] == position and float(line.split()[2]) <= 0.001):
                                new_df_down = new_df_down.append(data.iloc[i])
                                #print(line)
                except FileNotFoundError:
                    continue
        '''
        new_df_up = new_df_up.reset_index()        
        print(new_df_up)
        new_file_up = '../../datasets/0_ppc_0.9/' + bulk + '_' + str(chr_num) + '.pkl'
        new_df_up.to_pickle(new_file_up)
        '''
        new_df_down = new_df_down.reset_index() 
        print(new_df_down)
        new_file_down = '../../datasets/0_ppc_0.001/' + bulk + '_' + str(chr_num) + '.pkl'
        new_df_down.to_pickle(new_file_down)        