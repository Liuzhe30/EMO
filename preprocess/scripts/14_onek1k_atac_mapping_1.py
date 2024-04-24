import pandas as pd
import pyBigWig
import numpy as np
import math
pd.set_option('display.max_columns', None)

atac_path = '/data/eqtl/scATAC/'
new_path = '/data/eqtl/sceQTL/merged/'
file_path = '/data/eqtl/sceQTL/merged/'

#cell_list = ['B_Naive','CD4_memory','CD4_naive','CD8_memory','Dendritic_Cell','Natural_killer_Cell']
cell_list = ['B_Naive','CD4_memory','CD8_memory','Dendritic_Cell']

for cell in cell_list:
    error_id = []
    data = pd.read_pickle(file_path + cell + '_seq_0.2.pkl')
    print(cell)
    
    bw = pyBigWig.open(atac_path + cell + ".bw")
    data['atac_between'] = 0
    data['atac_variant_51'] = 0
    data['atac_tss_51'] = 0
    data['atac_between'] = data['atac_between'].astype('object')
    data['atac_variant_51'] = data['atac_variant_51'].astype('object')
    data['atac_tss_51'] = data['atac_tss_51'].astype('object')
    for i in range(data.shape[0]):
        chr_no = int(data['CHR'][i])
        max_atac_len = int(bw.chroms("chr" + str(chr_no)))
        seq_between = data['seq_between_variant_tss'].values[i]
        position = data['POS'][i]
        tss_position = data['TSS'][i]
        tss_distance = position - tss_position

        if(tss_distance > 0):
            if(tss_position > max_atac_len):
                atac_between = np.ones(tss_distance + 1).tolist()
                error_id.append(data['RSID'][i])
            elif(tss_position <= max_atac_len and position > max_atac_len):
                atac_between = bw.values("chr" + str(chr_no), tss_position - 1, max_atac_len)
                for idx in range(position - max_atac_len):
                    atac_between.append(0)
            else:
                atac_between = bw.values("chr" + str(chr_no), tss_position - 1, position)
        else:
            if(position > max_atac_len):
                atac_between = np.ones(-tss_distance + 1).tolist()
                error_id.append(data['RSID'][i])
            elif(position <= max_atac_len and tss_position > max_atac_len):
                atac_between = bw.values("chr" + str(chr_no), position - 1, max_atac_len)
                for idx in range(tss_position - max_atac_len):
                    atac_between.append(0)
            else:                    
                atac_between = bw.values("chr" + str(chr_no), position - 1, tss_position)
        #print(len(atac_between))
            
        if(position - 25 > max_atac_len):
            atac_variant_51 = np.ones(51).tolist()
            error_id.append(data['RSID'][i])
        elif(position - 25 <= max_atac_len and position + 25 > max_atac_len):
            atac_variant_51 = bw.values("chr" + str(chr_no), position - 26, max_atac_len)
            for idx in range(51 - max_atac_len):
                atac_variant_51.append(0)
        else:
            atac_variant_51 = bw.values("chr" + str(chr_no), position - 26, position + 25)
                
        if(tss_position - 25 > max_atac_len):
            error_id.append(data['RSID'][i])
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
        #print(atac_between)

        atac_between = [0 if math.isnan(x) else x for x in atac_between]
        atac_variant_51 = [0 if math.isnan(x) else x for x in atac_variant_51]
        atac_tss_51 = [0 if math.isnan(x) else x for x in atac_tss_51]
            
        #atac_between = atac_between.replace('nan', 0)
        #atac_variant_51 = atac_variant_51.replace('nan', 0)
        #atac_tss_51 = atac_tss_51.replace('nan', 0)
            
        data.at[i, 'atac_between'] = atac_between
        data.at[i, 'atac_variant_51'] = atac_variant_51
        data.at[i, 'atac_tss_51'] = atac_tss_51

        #print(data)

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