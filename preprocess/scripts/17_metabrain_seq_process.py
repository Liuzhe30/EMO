import os
import numpy as np
import pandas as pd

# load hg38 fasta
fasta_path = '../../datasets/chr_fasta_hg38/'
tissue_list = ['cerebellum','cortex','hippocampus','spinalcord']
for tissue in tissue_list:
    data = pd.read_pickle('../../datasets/metabrain/filtered/' + tissue + '_tss.pickle')

    data['variant_51_seq'] = 0
    data['tss_51_seq'] = 0
    data['seq_between_variant_tss'] = 0
    data['variant_51_seq_after_mutation'] = 0
    data['seq_between_variant_tss_after_mutation'] = 0

    error_id_list = []

    for i in range(len(data)):
        chr_str = 'chr' + str(data['CHR'][i])
        position = data['POS'][i]
        tss_position = data['TSS'][i]
        before_mutation = data['A1'][i]
        after_mutation = data['A2'][i]
        tss_distance = position - tss_position

        with open(fasta_path + chr_str + '_new.fasta') as fa:
            line = fa.readline()
            if(line[position - 1].upper() != before_mutation):
                #print('error!')
                #print(data['label'][i])
                error_id_list.append([data['GENE'][i],data['POS'][i]])

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
    
    for id in error_id_list:
        data = data.drop(data[(data['GENE']==id[0]) & (data['POS']==id[1])].index)
    data = data.reset_index(drop=True)
    print(data)
    data.to_pickle('../../datasets/metabrain/filtered/' + tissue + '_seq.pkl')