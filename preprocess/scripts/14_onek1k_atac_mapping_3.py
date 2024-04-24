import pandas as pd
import pyBigWig
import numpy as np
import math
pd.set_option('display.max_columns', None)

atac_path = '/data/eqtl/scATAC/'
new_path = '/data/eqtl/sceQTL/merged/'
file_path = '/data/eqtl/sceQTL/merged/'

#cell_list = ['B_Naive','CD4_memory','CD4_naive','CD8_memory','Dendritic_Cell','Natural_killer_Cell']
cell_list = ['CD4_naive','Natural_killer_Cell']

for cell in cell_list:
    error_id = []
    data = pd.read_pickle(file_path + cell + '_seq_0.2.pkl')
    print(cell)
    
    bw_1 = pyBigWig.open(atac_path + cell + "_1.bw")
    bw_2 = pyBigWig.open(atac_path + cell + "_2.bw")
    bw_3 = pyBigWig.open(atac_path + cell + "_3.bw")

    data['atac_between'] = 0
    data['atac_variant_51'] = 0
    data['atac_tss_51'] = 0
    data['atac_between'] = data['atac_between'].astype('object')
    data['atac_variant_51'] = data['atac_variant_51'].astype('object')
    data['atac_tss_51'] = data['atac_tss_51'].astype('object')
    for i in range(data.shape[0]):
        chr_no = int(data['CHR'][i])
        max_atac_len = min(int(bw_1.chroms("chr" + str(chr_no))),int(bw_2.chroms("chr" + str(chr_no))),int(bw_3.chroms("chr" + str(chr_no))))
        seq_between = data['seq_between_variant_tss'].values[i]
        position = data['POS'][i]
        tss_position = data['TSS'][i]
        tss_distance = position - tss_position

        if(tss_distance > 0):
            if(tss_position > max_atac_len):
                atac_between = np.ones(tss_distance + 1).tolist()
                error_id.append(data['RSID'][i])
            elif(tss_position <= max_atac_len and position > max_atac_len):
                atac_between_1 = bw_1.values("chr" + str(chr_no), tss_position - 1, max_atac_len)
                atac_between_1 = [0 if item == 'nan' else item for item in atac_between_1]
                atac_between_2 = bw_2.values("chr" + str(chr_no), tss_position - 1, max_atac_len)
                atac_between_2 = [0 if item == 'nan' else item for item in atac_between_2]
                atac_between_3 = bw_3.values("chr" + str(chr_no), tss_position - 1, max_atac_len)
                atac_between_3 = [0 if item == 'nan' else item for item in atac_between_3]
                atac_between = [atac_between_1, atac_between_2, atac_between_3]
                atac_between = list(np.mean(atac_between, axis=0))
                for idx in range(position - max_atac_len):
                    atac_between.append(0)
            else:
                atac_between_1 = bw_1.values("chr" + str(chr_no), tss_position - 1, position)
                atac_between_1 = [0 if item == 'nan' else item for item in atac_between_1]
                atac_between_2 = bw_2.values("chr" + str(chr_no), tss_position - 1, position)
                atac_between_2 = [0 if item == 'nan' else item for item in atac_between_2]
                atac_between_3 = bw_3.values("chr" + str(chr_no), tss_position - 1, position)
                atac_between_3 = [0 if item == 'nan' else item for item in atac_between_3]
                atac_between = [atac_between_1, atac_between_2, atac_between_3]
                atac_between = list(np.mean(atac_between, axis=0))
        else:
            if(position > max_atac_len):
                atac_between = np.ones(-tss_distance + 1).tolist()
                error_id.append(data['RSID'][i])
            elif(position <= max_atac_len and tss_position > max_atac_len):
                atac_between_1 = bw_1.values("chr" + str(chr_no), position - 1, max_atac_len)
                atac_between_1 = [0 if item == 'nan' else item for item in atac_between_1]
                atac_between_2 = bw_2.values("chr" + str(chr_no), position - 1, max_atac_len)
                atac_between_2 = [0 if item == 'nan' else item for item in atac_between_2]
                atac_between_3 = bw_3.values("chr" + str(chr_no), position - 1, max_atac_len)
                atac_between_3 = [0 if item == 'nan' else item for item in atac_between_3]
                atac_between = [atac_between_1, atac_between_2, atac_between_3]
                atac_between = list(np.mean(atac_between, axis=0))
                for idx in range(tss_position - max_atac_len):
                    atac_between.append(0)
            else:                    
                atac_between_1 = bw_1.values("chr" + str(chr_no), position - 1, tss_position)
                atac_between_1 = [0 if item == 'nan' else item for item in atac_between_1]
                atac_between_2 = bw_2.values("chr" + str(chr_no), position - 1, tss_position)
                atac_between_2 = [0 if item == 'nan' else item for item in atac_between_2]
                atac_between_3 = bw_3.values("chr" + str(chr_no), position - 1, tss_position)
                atac_between_3 = [0 if item == 'nan' else item for item in atac_between_3]
                atac_between = [atac_between_1, atac_between_2, atac_between_3]
                atac_between = list(np.mean(atac_between, axis=0))
        #print(len(atac_between))
            
        if(position - 25 > max_atac_len):
            atac_variant_51 = np.ones(51).tolist()
            error_id.append(data['RSID'][i])
        elif(position - 25 <= max_atac_len and position + 25 > max_atac_len):
            atac_variant_51_1 = bw_1.values("chr" + str(chr_no), position - 26, max_atac_len)
            atac_variant_51_1 = [0 if item == 'nan' else item for item in atac_variant_51_1]
            atac_variant_51_2 = bw_2.values("chr" + str(chr_no), position - 26, max_atac_len)
            atac_variant_51_2 = [0 if item == 'nan' else item for item in atac_variant_51_2]
            atac_variant_51_3 = bw_3.values("chr" + str(chr_no), position - 26, max_atac_len)
            atac_variant_51_3 = [0 if item == 'nan' else item for item in atac_variant_51_3]
            atac_variant_51 = [atac_variant_51_1, atac_variant_51_2, atac_variant_51_3]
            atac_variant_51 = list(np.mean(atac_variant_51, axis=0))
            for idx in range(51 - max_atac_len):
                atac_variant_51.append(0)
        else:
            atac_variant_51_1 = bw_1.values("chr" + str(chr_no), position - 26, position + 25)
            atac_variant_51_1 = [0 if item == 'nan' else item for item in atac_variant_51_1]
            atac_variant_51_2 = bw_2.values("chr" + str(chr_no), position - 26, position + 25)
            atac_variant_51_2 = [0 if item == 'nan' else item for item in atac_variant_51_2]
            atac_variant_51_3 = bw_3.values("chr" + str(chr_no), position - 26, position + 25)
            atac_variant_51_3 = [0 if item == 'nan' else item for item in atac_variant_51_3]
            atac_variant_51 = [atac_variant_51_1, atac_variant_51_2, atac_variant_51_3]
            atac_variant_51 = list(np.mean(atac_variant_51, axis=0))
                
        if(tss_position - 25 > max_atac_len):
            error_id.append(data['RSID'][i])
            atac_tss_51 = np.ones(51).tolist()
        elif(tss_position - 25 <= max_atac_len and tss_position + 25 > max_atac_len):
            atac_tss_51_1 = bw_1.values("chr" + str(chr_no), tss_position - 26, max_atac_len)
            atac_tss_51_1 = [0 if item == 'nan' else item for item in atac_tss_51_1]
            atac_tss_51_2 = bw_2.values("chr" + str(chr_no), tss_position - 26, max_atac_len)
            atac_tss_51_2 = [0 if item == 'nan' else item for item in atac_tss_51_2]
            atac_tss_51_3 = bw_3.values("chr" + str(chr_no), tss_position - 26, max_atac_len)
            atac_tss_51_3 = [0 if item == 'nan' else item for item in atac_tss_51_3]
            atac_tss_51 = [atac_tss_51_1, atac_tss_51_2, atac_tss_51_3]
            atac_tss_51 = list(np.mean(atac_tss_51, axis=0))
            for idx in range(51 - max_atac_len):
                atac_tss_51.append(0)
        else:
            atac_tss_51_1 = bw_1.values("chr" + str(chr_no), tss_position - 26, tss_position + 25)  
            atac_tss_51_1 = [0 if item == 'nan' else item for item in atac_tss_51_1]
            atac_tss_51_2 = bw_2.values("chr" + str(chr_no), tss_position - 26, tss_position + 25)  
            atac_tss_51_2 = [0 if item == 'nan' else item for item in atac_tss_51_2]
            atac_tss_51_3 = bw_3.values("chr" + str(chr_no), tss_position - 26, tss_position + 25)  
            atac_tss_51_3 = [0 if item == 'nan' else item for item in atac_tss_51_3]
            atac_tss_51 = [atac_tss_51_1, atac_tss_51_2, atac_tss_51_3]
            atac_tss_51 = list(np.mean(atac_tss_51, axis=0))    
            
        #atac_between = [round(item, 4) for item in atac_between]
        #atac_variant_51 = [round(item, 4) for item in atac_variant_51]
        #atac_tss_51 = [round(item, 4) for item in atac_tss_51]
            
        data.at[i, 'atac_between'] = atac_between
        data.at[i, 'atac_variant_51'] = atac_variant_51
        data.at[i, 'atac_tss_51'] = atac_tss_51

    for idx in error_id:
        data = data.drop(data[data['RSID']==idx].index)
    data = data.reset_index(drop=True)

    print(data)
    print(data['label'].value_counts())
    data.to_pickle('/data/eqtl/sceQTL/merged/' + cell + '_seq_0.2_atac.pkl')
    
    '''
    for i in range(data.shape[0]):
        if(np.abs(data['RHO'][i]) <= 0.2):
            error_id.append(data['RSID'][i])

    for idx in error_id:
        data = data.drop(data[data['RSID']==idx].index)
    data = data.reset_index(drop=True)
    print(data)
    data.to_pickle('/data/eqtl/sceQTL/merged/' + cell + '_seq_0.2.pkl')
    '''