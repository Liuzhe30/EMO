# mapping ATAC-seq
import pandas as pd
from pandas import read_parquet
import pyBigWig
import numpy as np
import math
pd.set_option('display.max_columns', None)
#pd.set_option('display.max_rows', None)

snp_list = ['rs2476601','rs3806624', 'rs7731626', 'rs2234067','rs2233424','rs947474','rs3824660','rs968567','rs3218251']
model_dict = {"small":1_000,"middle":10_000,"large":100_000,"huge":1_000_000}

atac_path = 'hg38_ATAC/hg38_t0_atac.bigWig'
output_path = 'data/atac_mapping_t0/'
file_path = 'data/seq_mapping/'

for snp in snp_list:
    for model_type in model_dict:
        data = pd.read_pickle(file_path + snp + '_' + model_type + '.dataset')
        bw = pyBigWig.open(atac_path)
        data['atac_between'] = 0
        data['atac_variant_51'] = 0
        data['atac_tss_51'] = 0
        data['atac_between'] = data['atac_between'].astype('object')
        data['atac_variant_51'] = data['atac_variant_51'].astype('object')
        data['atac_tss_51'] = data['atac_tss_51'].astype('object')
        
        for i in range(len(data)):
            variant_id = data['variant_id'].values[i]
            chr_str = data['CHR'].values[i]
            position = data['SNP_POS'].values[i]
            before_mutation = data['Ref'].values[i]
            after_mutation = data['Alt'].values[i]
            tss_distance = int(data['tss_distance'].values[i])
            tss_position = int(data['TSS_POS'].values[i])
            seq_between = data['seq_between_variant_tss'].values[i]

            max_atac_len = int(bw.chroms(chr_str))
            if(tss_distance > 0):
                if(tss_position > max_atac_len):
                    atac_between = np.ones(tss_distance + 1).tolist()
                elif(tss_position <= max_atac_len and position > max_atac_len):
                    atac_between = bw.values(chr_str, tss_position - 1, max_atac_len)
                    for idx in range(position - max_atac_len):
                        atac_between.append(0)
                else:
                    atac_between = bw.values(chr_str, tss_position - 1, position)
            else:
                if(position > max_atac_len):
                    atac_between = np.ones(-tss_distance + 1).tolist()
                elif(position <= max_atac_len and tss_position > max_atac_len):
                    atac_between = bw.values(chr_str, position - 1, max_atac_len)
                    for idx in range(tss_position - max_atac_len):
                        atac_between.append(0)
                else:                    
                    atac_between = bw.values(chr_str, position - 1, tss_position)
            #print(len(atac_between))
            
            if(position - 25 > max_atac_len):
                atac_variant_51 = np.ones(51).tolist()
            elif(position - 25 <= max_atac_len and position + 25 > max_atac_len):
                atac_variant_51 = bw.values(chr_str, position - 26, max_atac_len)
                for idx in range(51 - max_atac_len):
                    atac_variant_51.append(0)
            else:
                atac_variant_51 = bw.values(chr_str, position - 26, position + 25)
                
            if(tss_position - 25 > max_atac_len):
                atac_tss_51 = np.ones(51).tolist()
            elif(tss_position - 25 <= max_atac_len and tss_position + 25 > max_atac_len):
                atac_tss_51 = bw.values(chr_str, tss_position - 26, max_atac_len)
                for idx in range(51 - max_atac_len):
                    atac_tss_51.append(0)
            else:
                atac_tss_51 = bw.values(chr_str, tss_position - 26, tss_position + 25)    
            
            atac_between = [0 if math.isnan(x) else x for x in atac_between]
            atac_variant_51 = [0 if math.isnan(x) else x for x in atac_variant_51]
            atac_tss_51 = [0 if math.isnan(x) else x for x in atac_tss_51]
            
            data.at[i, 'atac_between'] = atac_between
            data.at[i, 'atac_variant_51'] = atac_variant_51
            data.at[i, 'atac_tss_51'] = atac_tss_51
        #print(data)
        data.to_pickle(output_path + snp + '_' + model_type + '.dataset')


atac_path = 'hg38_ATAC/hg38_t24h_atac.bigWig'
output_path = 'data/atac_mapping_t24/'
file_path = 'data/seq_mapping/'

for snp in snp_list:
    for model_type in model_dict:
        data = pd.read_pickle(file_path + snp + '_' + model_type + '.dataset')
        bw = pyBigWig.open(atac_path)
        data['atac_between'] = 0
        data['atac_variant_51'] = 0
        data['atac_tss_51'] = 0
        data['atac_between'] = data['atac_between'].astype('object')
        data['atac_variant_51'] = data['atac_variant_51'].astype('object')
        data['atac_tss_51'] = data['atac_tss_51'].astype('object')
        
        for i in range(len(data)):
            variant_id = data['variant_id'].values[i]
            chr_str = data['CHR'].values[i]
            position = data['SNP_POS'].values[i]
            before_mutation = data['Ref'].values[i]
            after_mutation = data['Alt'].values[i]
            tss_distance = int(data['tss_distance'].values[i])
            tss_position = int(data['TSS_POS'].values[i])
            seq_between = data['seq_between_variant_tss'].values[i]

            max_atac_len = int(bw.chroms(chr_str))
            if(tss_distance > 0):
                if(tss_position > max_atac_len):
                    atac_between = np.ones(tss_distance + 1).tolist()
                elif(tss_position <= max_atac_len and position > max_atac_len):
                    atac_between = bw.values(chr_str, tss_position - 1, max_atac_len)
                    for idx in range(position - max_atac_len):
                        atac_between.append(0)
                else:
                    atac_between = bw.values(chr_str, tss_position - 1, position)
            else:
                if(position > max_atac_len):
                    atac_between = np.ones(-tss_distance + 1).tolist()
                elif(position <= max_atac_len and tss_position > max_atac_len):
                    atac_between = bw.values(chr_str, position - 1, max_atac_len)
                    for idx in range(tss_position - max_atac_len):
                        atac_between.append(0)
                else:                    
                    atac_between = bw.values(chr_str, position - 1, tss_position)
            #print(len(atac_between))
            
            if(position - 25 > max_atac_len):
                atac_variant_51 = np.ones(51).tolist()
            elif(position - 25 <= max_atac_len and position + 25 > max_atac_len):
                atac_variant_51 = bw.values(chr_str, position - 26, max_atac_len)
                for idx in range(51 - max_atac_len):
                    atac_variant_51.append(0)
            else:
                atac_variant_51 = bw.values(chr_str, position - 26, position + 25)
                
            if(tss_position - 25 > max_atac_len):
                atac_tss_51 = np.ones(51).tolist()
            elif(tss_position - 25 <= max_atac_len and tss_position + 25 > max_atac_len):
                atac_tss_51 = bw.values(chr_str, tss_position - 26, max_atac_len)
                for idx in range(51 - max_atac_len):
                    atac_tss_51.append(0)
            else:
                atac_tss_51 = bw.values(chr_str, tss_position - 26, tss_position + 25)    
            
            atac_between = [0 if math.isnan(x) else x for x in atac_between]
            atac_variant_51 = [0 if math.isnan(x) else x for x in atac_variant_51]
            atac_tss_51 = [0 if math.isnan(x) else x for x in atac_tss_51]
            
            data.at[i, 'atac_between'] = atac_between
            data.at[i, 'atac_variant_51'] = atac_variant_51
            data.at[i, 'atac_tss_51'] = atac_tss_51
        #print(data)
        data.to_pickle(output_path + snp + '_' + model_type + '.dataset')