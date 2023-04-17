# count the distribution of sequences
import pandas as pd
from pandas import read_parquet
pd.set_option('display.max_columns', None)
#pd.set_option('display.max_rows', None)

file_path = 'D:/eQTL_SNP/dataset/2_ppc_0.9_seq_mapping/'
#file_path = 'D:/eQTL_SNP/dataset/2_ppc_0.001_seq_mapping/'

len_dict = {'1-1000':0,'1001-10000':0,'10001-100000':0,'100001-1000000':0,'1000000+':0}
# count sequence len
with open("0_gtex_list.txt") as r:
    lines = r.readlines()
    gtex_bulk_list = []
    for line in lines:
        gtex_bulk_list.append(line.strip())
    print(gtex_bulk_list)
    
for bulk in gtex_bulk_list:
    for chr_no in range(1, 23):
        data = pd.read_pickle(file_path + bulk + '_' + str(chr_no) + '.pkl')
        for i in range(len(data)):        
            tss_distance = abs(int(data['tss_distance'].values[i]))
            #print(tss_distance)
            if(tss_distance <= 1000):
                len_dict['1-1000'] += 1
            elif(tss_distance > 1000 and tss_distance <= 10000):
                len_dict['1001-10000'] += 1
            elif(tss_distance > 10000 and tss_distance <= 100000):
                len_dict['10001-100000'] += 1
            elif(tss_distance > 100000 and tss_distance <= 1000000):
                len_dict['100001-1000000'] += 1
            else:
                len_dict['1000000+'] += 1
        
print(len_dict)