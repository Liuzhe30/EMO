# split slope ATAC_0.9
import pandas as pd
from pandas import read_parquet
pd.set_option('display.max_columns', None)
#pd.set_option('display.max_rows', None)

file_path = 'dataset/3_ATAC_0.9/'
file_path2 = 'dataset/3_ATAC_0.9_0/'

with open("0_gtex_list.txt") as r:
    lines = r.readlines()
    gtex_bulk_list = []
    for line in lines:
        gtex_bulk_list.append(line.strip())
    print(gtex_bulk_list)
    
for bulk in gtex_bulk_list:
    for chr_no in range(1, 23):
        data = pd.read_pickle(file_path + bulk + '_' + str(chr_no) + '.pkl')
        #print(data)
        data_check = pd.read_pickle(file_path + bulk + '_' + str(chr_no) + '.pkl')
        for i in range(len(data_check)):
            slope = data_check['slope'].values[i]     
            variant_id = data_check['variant_id'].values[i]
            if(float(slope) >= 0):
                data = data[~data['variant_id'].isin([variant_id])]
        print(data)
        data = data.reset_index(drop=True)
        new_file_down = file_path2 + bulk + '_' + str(chr_no) + '.pkl'
        data.to_pickle(new_file_down)        