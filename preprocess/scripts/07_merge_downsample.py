# merge chrom & down-sample 
import pandas as pd
from pandas import read_parquet
pd.set_option('display.max_columns', None)
#pd.set_option('display.max_rows', None)

enhancer_path = '/data/eqtl/3_ATAC_0.9_1/'
repressor_path = '/data/eqtl/3_ATAC_0.9_0/'
datasets_path = '/data/eqtl/datasets/'

with open("../gtex_list.txt") as r:
    lines = r.readlines()
    gtex_bulk_list = []
    for line in lines:
        gtex_bulk_list.append(line.strip())
    print(gtex_bulk_list)

downsample_list = ['Esophagus_Mucosa', 'Heart_Left_Ventricle']

# enhancer
for bulk in gtex_bulk_list:
    if(bulk in downsample_list):
        continue
    bulk_frame = pd.DataFrame()
    for chr_no in range(1, 23):
        data = pd.read_pickle(enhancer_path + bulk + '_' + str(chr_no) + '.pkl')
        bulk_frame = pd.concat([bulk_frame, data])
    bulk_frame.to_pickle(datasets_path + 'enhancer/' + bulk + '.pkl')

bulk_frame = pd.DataFrame()
for chr_no in range(1, 23):
    data = pd.read_pickle(enhancer_path + 'Esophagus_Mucosa' + '_' + str(chr_no) + '.pkl')
    bulk_frame = pd.concat([bulk_frame, data])
shuffle_df = bulk_frame.sample(frac=1).reset_index(drop=True)
bulk_frame = shuffle_df[0:715].reset_index(drop=True)
bulk_frame.to_pickle(datasets_path + 'enhancer/' + 'Esophagus_Mucosa' + '.pkl')
print('Esophagus_Mucosa enhancer: ', bulk_frame.shape)

bulk_frame = pd.DataFrame()
for chr_no in range(1, 23):
    data = pd.read_pickle(enhancer_path + 'Heart_Left_Ventricle' + '_' + str(chr_no) + '.pkl')
    bulk_frame = pd.concat([bulk_frame, data])
shuffle_df = bulk_frame.sample(frac=1).reset_index(drop=True)
bulk_frame = shuffle_df[0:715].reset_index(drop=True)
bulk_frame.to_pickle(datasets_path + 'enhancer/' + 'Heart_Left_Ventricle' + '.pkl')
print('Heart_Left_Ventricle enhancer: ', bulk_frame.shape)

# repressor
for bulk in gtex_bulk_list:
    if(bulk in downsample_list):
        continue
    bulk_frame = pd.DataFrame()
    for chr_no in range(1, 23):
        data = pd.read_pickle(repressor_path + bulk + '_' + str(chr_no) + '.pkl')
        bulk_frame = pd.concat([bulk_frame, data])
    bulk_frame.to_pickle(datasets_path + 'repressor/' + bulk + '.pkl')

bulk_frame = pd.DataFrame()
for chr_no in range(1, 23):
    data = pd.read_pickle(repressor_path + 'Esophagus_Mucosa' + '_' + str(chr_no) + '.pkl')
    bulk_frame = pd.concat([bulk_frame, data])
shuffle_df = bulk_frame.sample(frac=1).reset_index(drop=True)
bulk_frame = shuffle_df[0:648].reset_index(drop=True)
bulk_frame.to_pickle(datasets_path + 'repressor/' + 'Esophagus_Mucosa' + '.pkl')
print('Esophagus_Mucosa repressor: ', bulk_frame.shape)

bulk_frame = pd.DataFrame()
for chr_no in range(1, 23):
    data = pd.read_pickle(repressor_path + 'Heart_Left_Ventricle' + '_' + str(chr_no) + '.pkl')
    bulk_frame = pd.concat([bulk_frame, data])
shuffle_df = bulk_frame.sample(frac=1).reset_index(drop=True)
bulk_frame = shuffle_df[0:648].reset_index(drop=True)
bulk_frame.to_pickle(datasets_path + 'repressor/' + 'Heart_Left_Ventricle' + '.pkl')
print('Heart_Left_Ventricle repressor: ', bulk_frame.shape)