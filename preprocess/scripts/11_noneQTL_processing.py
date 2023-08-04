# merge ATAC_0.9 and split sequence length
import pandas as pd
import numpy as np
pd.set_option('display.max_columns', None)
#pd.set_option('display.max_rows', None)

gene_dict = {'A':[0,0,0,1], 'T':[0,0,1,0], 'C':[0,1,0,0], 'G':[1,0,0,0], 
             'a':[0,0,0,1], 't':[0,0,1,0], 'c':[0,1,0,0], 'g':[1,0,0,0],
             } 

with open("../gtex_list.txt") as r:
    lines = r.readlines()
    gtex_bulk_list = []
    for line in lines:
        gtex_bulk_list.append(line.strip())
    print(gtex_bulk_list)

file_path = '/data/eqtl/3_ATAC_0.001/'

for bulk in gtex_bulk_list:
    for chr_no in range(1, 23):
        merged_df = pd.DataFrame()
        merged_df = pd.read_pickle(file_path + bulk + '_' + str(chr_no) + '.pkl')
        
        print('delete started!')
        print(merged_df.shape)
        # delete wrong sequence
        del_row = []
        for i in range(len(merged_df)):
            for strr in merged_df['variant_51_seq'].values[i]:
                if(strr not in gene_dict.keys() and merged_df['variant_id'].values[i] not in del_row):
                    del_row.append(merged_df['variant_id'].values[i])
            for strr in merged_df['variant_51_seq_after_mutation'].values[i]:
                if(strr not in gene_dict.keys() and merged_df['variant_id'].values[i] not in del_row):
                    del_row.append(merged_df['variant_id'].values[i])   
            for strr in merged_df['seq_between_variant_tss'].values[i]:
                if(strr not in gene_dict.keys() and merged_df['variant_id'].values[i] not in del_row):
                    del_row.append(merged_df['variant_id'].values[i])  
        #print(del_row)
        for i in del_row:
            merged_df = merged_df[~(merged_df['variant_id'] == i)]

        # delete nan ATAC values
        del_row = []
        for i in range(len(merged_df)):
            for num in merged_df['atac_variant_51'].values[i]:
                if(pd.isnull(num) and merged_df['variant_id'].values[i] not in del_row):
                    del_row.append(merged_df['variant_id'].values[i])
            for num in merged_df['atac_between'].values[i]:
                if(pd.isnull(num) and merged_df['variant_id'].values[i] not in del_row):
                    del_row.append(merged_df['variant_id'].values[i])
                    continue
        for i in del_row:
            merged_df = merged_df[~(merged_df['variant_id'] == i)]

        print('delete finished!')
        print(merged_df.shape)

        small_df = pd.DataFrame()
        for i in range(len(merged_df)):        
            tss_distance = abs(int(merged_df['tss_distance'].values[i]))
            if(tss_distance < 1000):
                small_df = pd.concat([small_df, merged_df[i:i+1]])
        small_df = small_df.reset_index(drop=True)
        small_df.to_pickle('/data/eqtl/noneqtl/small/' + bulk + '_' + str(chr_no) + '.pkl')

        middle_df = pd.DataFrame()
        for i in range(len(merged_df)):        
            tss_distance = abs(int(merged_df['tss_distance'].values[i]))
            if(tss_distance >= 1000 and tss_distance < 10000):
                middle_df = pd.concat([middle_df, merged_df[i:i+1]])
        middle_df = middle_df.reset_index(drop=True)
        middle_df.to_pickle('/data/eqtl/noneqtl/middle/' + bulk + '_' + str(chr_no) + '.pkl')

        large_df = pd.DataFrame()
        for i in range(len(merged_df)):        
            tss_distance = abs(int(merged_df['tss_distance'].values[i]))
            if(tss_distance >= 10000 and tss_distance < 100000):
                large_df = pd.concat([large_df, merged_df[i:i+1]])
        large_df = large_df.reset_index(drop=True)
        large_df.to_pickle('/data/eqtl/noneqtl/large/' + bulk + '_' + str(chr_no) + '.pkl')

        huge_df = pd.DataFrame()
        for i in range(len(merged_df)):        
            tss_distance = abs(int(merged_df['tss_distance'].values[i]))
            if(tss_distance >= 100000):
                huge_df = pd.concat([huge_df, merged_df[i:i+1]])
        huge_df = huge_df.reset_index(drop=True)
        huge_df.to_pickle('/data/eqtl/noneqtl/huge/' + bulk + '_' + str(chr_no) + '.pkl')

