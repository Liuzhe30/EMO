# mapping ATAC-seq
import pandas as pd
from pandas import read_parquet
import pyBigWig
import numpy as np
pd.set_option('display.max_columns', None)
#pd.set_option('display.max_rows', None)

atac_path = '../../datasets/epimap_ATAC_tissue_new/'
new_path = '../../datasets/3_ATAC_0.9/'
file_path = '../../datasets/2_ppc_0.9_seq_mapping/'

gtex_bulk_list = ['Adipose_Subcutaneous','Artery_Tibial','Brain_Cerebellum','Brain_Cortex','Liver','Nerve_Tibial','Ovary','Prostate','Spleen','Testis']

for bulk in gtex_bulk_list:
    for chr_no in range(1, 23):
        data = pd.read_pickle(file_path + bulk + '_' + str(chr_no) + '.pkl')    
        bw = pyBigWig.open(atac_path + bulk + ".bigWig")
        max_atac_len = int(bw.chroms("chr" + str(chr_no)))
        print(data)
        data['atac_between'] = 0
        data['atac_variant_51'] = 0
        data['atac_tss_51'] = 0
        data['atac_between'] = data['atac_between'].astype('object')
        data['atac_variant_51'] = data['atac_variant_51'].astype('object')
        data['atac_tss_51'] = data['atac_tss_51'].astype('object')
        for i in range(len(data)):
            variant_id = data['variant_id'].values[i]
            seq_between = data['seq_between_variant_tss'].values[i]
            #print(len(seq_between))
            position = int(variant_id.split('_')[1])  
            print(position) 
            tss_distance = int(data['tss_distance'].values[i])
            tss_position = position - tss_distance
            print(tss_position)
            if(tss_distance > 0):
                if(tss_position > max_atac_len):
                    atac_between = np.ones(tss_distance + 1).tolist()
                elif(tss_position <= max_atac_len and position > max_atac_len):
                    atac_between = bw.values("chr" + str(chr_no), tss_position - 1, max_atac_len)
                    for idx in range(position - max_atac_len):
                        atac_between.append(0)
                else:
                    atac_between = bw.values("chr" + str(chr_no), tss_position - 1, position)
            else:
                if(position > max_atac_len):
                    atac_between = np.ones(-tss_distance + 1).tolist()
                elif(position <= max_atac_len and tss_position > max_atac_len):
                    atac_between = bw.values("chr" + str(chr_no), position - 1, max_atac_len)
                    for idx in range(tss_position - max_atac_len):
                        atac_between.append(0)
                else:                    
                    atac_between = bw.values("chr" + str(chr_no), position - 1, tss_position)
            #print(len(atac_between))
            
            if(position - 25 > max_atac_len):
                atac_variant_51 = np.ones(51).tolist()
            elif(position - 25 <= max_atac_len and position + 25 > max_atac_len):
                atac_variant_51 = bw.values("chr" + str(chr_no), position - 26, max_atac_len)
                for idx in range(51 - max_atac_len):
                    atac_variant_51.append(0)
            else:
                atac_variant_51 = bw.values("chr" + str(chr_no), position - 26, position + 25)
                
            if(tss_position - 25 > max_atac_len):
                atac_tss_51 = np.ones(51).tolist()
            elif(tss_position - 25 <= max_atac_len and tss_position + 25 > max_atac_len):
                atac_tss_51 = bw.values("chr" + str(chr_no), tss_position - 26, max_atac_len)
                for idx in range(51 - max_atac_len):
                    atac_tss_51.append(0)
            else:
                atac_tss_51 = bw.values("chr" + str(chr_no), tss_position - 26, tss_position + 25)    
            
            #atac_between = [round(item, 4) for item in atac_between]
            #atac_variant_51 = [round(item, 4) for item in atac_variant_51]
            #atac_tss_51 = [round(item, 4) for item in atac_tss_51]
            
            atac_between = atac_between.replace('nan', 0)
            atac_variant_51 = atac_variant_51.replace('nan', 0)
            atac_tss_51 = atac_tss_51.replace('nan', 0)
            
            data.at[i, 'atac_between'] = atac_between
            data.at[i, 'atac_variant_51'] = atac_variant_51
            data.at[i, 'atac_tss_51'] = atac_tss_51
        print(data)
        new_file_down = new_path + bulk + '_' + str(chr_no) + '.pkl'
        #print(data)
        data.to_pickle(new_file_down)