# make mutation single
import pandas as pd
from pandas import read_parquet
pd.set_option('display.max_columns', None)
#pd.set_option('display.max_rows', None)

'''
# test 
file_path = '../../datasets/GTEx_Analysis_v8_EUR_eQTL_all_associations/'
bulk = 'Adipose_Subcutaneous'
data = read_parquet(file_path + bulk + '/GTEx_Analysis_v8_QTLs_GTEx_Analysis_v8_EUR_eQTL_all_associations_' + bulk + '.v8.EUR.allpairs.chr1.parquet')
#print(data.count())
#data.head()
print(data)
print(len(data))
for i in range(20):
    variant_id = data['variant_id'].values[i]
    chr_str = variant_id.split('_')[0]
    position = variant_id.split('_')[1]
    z_no = int(float(position) / 50)
    x_no = int(position) % 50    
    with open(fasta_path + chr_str + '.fa') as fa:
        line = fa.readline()
        for i in range(0, z_no):
            line = fa.readline()
        if(x_no == 0):
            pos_fasta = line.strip()[-1].upper()
        else:
            line = fa.readline()
            pos_fasta = line[x_no - 1].upper()
    print(variant_id)
    print(pos_fasta)
'''

fasta_path = '../../datasets/chr_fasta_hg38/'
file_path = '../../datasets/0_ppc_0.001/'
#file_path_2 = '../../datasets/1_pval_0.5_single_mutation/'
with open("../0_gtex_list.txt") as r:
    lines = r.readlines()
    gtex_bulk_list = []
    for line in lines:
        gtex_bulk_list.append(line.strip())
    print(gtex_bulk_list)

# single mutation    
column_list = ['phenotype_id', 'variant_id', 'tss_distance', 'maf', 'ma_samples', 'ma_count', 'pval_nominal', 'slope', 'slope_se']
for bulk in gtex_bulk_list:
    for chr_no in range(1, 23):
        data = pd.read_pickle(file_path + bulk + '_' + str(chr_no) + '.pkl')
        data_check = pd.read_pickle(file_path + bulk + '_' + str(chr_no) + '.pkl')
        print(len(data))
        print(data)
        
        for i in range(len(data_check)):
            variant_id = data_check['variant_id'].values[i]
            before_mutation = variant_id.split('_')[2]
            after_mutation = variant_id.split('_')[3]
            if(len(after_mutation) > 1 or len(before_mutation) > 1):
                data = data[~data['variant_id'].isin([variant_id])]
        print(data)
        data = data.reset_index()
        new_file_down = '../../datasets/1_ppc_0.001_single_mutation/' + bulk + '_' + str(chr_no) + '.pkl'
        data.to_pickle(new_file_down)


'''
# count
file_path_3 = '../../datasets/1_ppc_0.001_single_mutation/'

for bulk in gtex_bulk_list:
    count = 0
    for chr_no in range(1, 23):
        data = pd.read_pickle(file_path_3 + bulk + '_' + str(chr_no) + '.pkl')
        #print(data)
        count += len(data)
    print(count)
'''