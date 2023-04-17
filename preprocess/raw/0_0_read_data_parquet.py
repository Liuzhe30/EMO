import pandas as pd
from pandas import read_parquet
pd.set_option('display.max_columns', None)


file_path = 'E:/SNP_eQTL_rawdata/GTEx_Analysis_v8_EUR_eQTL_all_associations/'
bulk = 'Adipose_Subcutaneous'
data = read_parquet(file_path + bulk + '/GTEx_Analysis_v8_QTLs_GTEx_Analysis_v8_EUR_eQTL_all_associations_' + bulk + '.v8.EUR.allpairs.chr1.parquet')
#print(data.count())
#data.head()
print(data)
print(len(data))