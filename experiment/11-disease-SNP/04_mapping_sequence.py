# mapping sequences
import pandas as pd
from pandas import read_parquet
pd.set_option('display.max_columns', None)
#pd.set_option('display.max_rows', None)

fasta_path = '../../datasets/chr_fasta_hg38/'
file_path = 'data/snp_tss/'
output_path = 'data/seq_mapping/'

'''
    TSS_POS   gene variant_id  slope Ref Alt    CHR   SNP_POS  label  tss_distance
0  37149916  IL2RB  rs3218251      0   T   A  chr22  37149465      0           451
'''

snp_list = ['rs2476601','rs3806624', 'rs7731626', 'rs2234067','rs2233424','rs947474','rs3824660','rs968567','rs3218251']

model_dict = {"small":1_000,"middle":10_000,"large":100_000,"huge":1_000_000}
for snp in snp_list:
    for model_type in model_dict:
        data = pd.read_pickle(file_path + snp + '_' + model_type + '.pkl')
        # add new columns
        data['variant_51_seq'] = 0
        data['tss_51_seq'] = 0
        data['seq_between_variant_tss'] = 0
        data['variant_51_seq_after_mutation'] = 0
        data['seq_between_variant_tss_after_mutation'] = 0

        for i in range(len(data)):        
            variant_id = data['variant_id'].values[i]
            chr_str = data['CHR'].values[i]
            position = data['SNP_POS'].values[i]
            before_mutation = data['Ref'].values[i]
            after_mutation = data['Alt'].values[i]
            tss_distance = int(data['tss_distance'].values[i])
            tss_position = int(data['TSS_POS'].values[i])

            with open(fasta_path + chr_str + '_new.fasta') as fa:
                line = fa.readline()                
                variant_51_seq = line[position - 26:position + 25]  
                #print(variant_51_seq)
                tss_51_seq = line[tss_position - 26:tss_position + 25]
                #print(tss_51_seq)
                if(tss_distance > 0):
                    seq_between_variant_tss = line[tss_position - 1:position]
                else:
                    seq_between_variant_tss = line[position -1: tss_position]
                #print(seq_between_variant_tss)
                if(line[position - 1] >= 'a' and line[position - 1] <= 'z'):
                    variant_51_seq_after_mutation = line[position - 26: position - 1] + after_mutation.lower() + line[position: position + 25]
                    if(tss_distance > 0):
                        seq_between_variant_tss_after_mutation = seq_between_variant_tss[0:-2] + after_mutation.lower()
                    else:
                        seq_between_variant_tss_after_mutation = after_mutation.lower() + seq_between_variant_tss[1:-1]
                else:
                    variant_51_seq_after_mutation = line[position - 26: position - 1] + after_mutation + line[position: position + 25]
                    if(tss_distance > 0):
                        seq_between_variant_tss_after_mutation = seq_between_variant_tss[0:-2] + after_mutation
                    else:
                        seq_between_variant_tss_after_mutation = after_mutation + seq_between_variant_tss[1:-1]                    
                #print(variant_51_seq_after_mutation)
                
                data.loc[i, 'variant_51_seq'] = variant_51_seq
                data.loc[i, 'tss_51_seq'] = tss_51_seq
                data.loc[i, 'seq_between_variant_tss'] = seq_between_variant_tss
                data.loc[i, 'variant_51_seq_after_mutation'] = variant_51_seq_after_mutation
                data.loc[i, 'seq_between_variant_tss_after_mutation'] = seq_between_variant_tss_after_mutation

        data.to_pickle(output_path + snp + '_' + model_type + '.dataset')