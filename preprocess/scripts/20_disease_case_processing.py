import os
import numpy as np
import pandas as pd
import pyBigWig
import math

disease_list = ['AD','PD']

# 1 mapping TSS (3 large models)

# 2 check training set
tissue_file = '/data/eqtl/datasets/tissue_specific/'
model = 'large'
check_str = ['chr10_80503927_G_A_b38','chr4_15735725_G_A_b38']
data = pd.read_pickle(tissue_file + model + '/train_' + model + '_Brain_Cortex.pkl')
select1 = data[(data['variant_id'] == check_str[0])]
print(select1) # null
select2 = data[(data['variant_id'] == check_str[1])]
print(select2) # null

# 3 split and save pickle
cases = pd.read_csv("/data/eqtl/metabrain/disease/metabrain_disease_case.csv")
print(cases)
cases[0:1].reset_index(drop=True).to_pickle('/data/eqtl/metabrain/disease/AD.pkl')
cases[1:2].reset_index(drop=True).to_pickle('/data/eqtl/metabrain/disease/PD.pkl')

# 4 sequence mapping
fasta_path = '/data/eqtl/rawdata/chr_fasta_hg38/'
for tissue in disease_list:
    data = pd.read_pickle('/data/eqtl/metabrain/disease/' + tissue + '.pkl')

    data['variant_51_seq'] = 0
    data['tss_51_seq'] = 0
    data['seq_between_variant_tss'] = 0
    data['variant_51_seq_after_mutation'] = 0
    data['seq_between_variant_tss_after_mutation'] = 0

    error_id_list = []

    for i in range(len(data)):
        chr_str = 'chr' + str(data['chr'][i])
        position = data['pos'][i]
        tss_position = data['tss'][i]
        before_mutation = data['A1'][i]
        after_mutation = data['A2'][i]
        tss_distance = position - tss_position

        with open(fasta_path + chr_str + '_new.fasta') as fa:
            line = fa.readline()
            if(line[position - 1].upper() != before_mutation):
                #print('error!')
                #print(data['label'][i])
                error_id_list.append([data['Gene'][i],data['pos'][i]])

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
        data = data.drop(data[(data['Gene']==id[0]) & (data['Pos']==id[1])].index)
    data = data.reset_index(drop=True)
    print(data)
    data.to_pickle('/data/eqtl/metabrain/disease/' + tissue + '_seq.pkl')

# 5 atac mapping
brain_atac = '/data/eqtl/rawdata/epimap_ATAC_tissue_new/Brain_Cerebellum.bigWig'
file_path = '/data/eqtl/metabrain/disease/'

for cell in ['AD','PD']:
    error_id = []
    data = pd.read_pickle(file_path + cell + '_seq.pkl')
    print(cell)
    
    bw = pyBigWig.open(brain_atac)
    data['atac_between'] = 0
    data['atac_variant_51'] = 0
    data['atac_tss_51'] = 0
    data['atac_between'] = data['atac_between'].astype('object')
    data['atac_variant_51'] = data['atac_variant_51'].astype('object')
    data['atac_tss_51'] = data['atac_tss_51'].astype('object')
    for i in range(data.shape[0]):
        chr_no = int(data['chr'][i])
        max_atac_len = int(bw.chroms("chr" + str(chr_no)))
        seq_between = data['seq_between_variant_tss'].values[i]
        position = data['pos'][i]
        tss_position = data['tss'][i]
        tss_distance = position - tss_position

        if(tss_distance > 0):
            if(tss_position > max_atac_len):
                atac_between = np.ones(tss_distance + 1).tolist()
                error_id.append([data['Gene'][i],data['pos'][i]])
            elif(tss_position <= max_atac_len and position > max_atac_len):
                atac_between = bw.values("chr" + str(chr_no), tss_position - 1, max_atac_len)
                for idx in range(position - max_atac_len):
                    atac_between.append(0)
            else:
                atac_between = bw.values("chr" + str(chr_no), tss_position - 1, position)
        else:
            if(position > max_atac_len):
                atac_between = np.ones(-tss_distance + 1).tolist()
                error_id.append([data['Gene'][i],data['pos'][i]])
            elif(position <= max_atac_len and tss_position > max_atac_len):
                atac_between = bw.values("chr" + str(chr_no), position - 1, max_atac_len)
                for idx in range(tss_position - max_atac_len):
                    atac_between.append(0)
            else:                    
                atac_between = bw.values("chr" + str(chr_no), position - 1, tss_position)
        #print(len(atac_between))
            
        if(position - 25 > max_atac_len):
            atac_variant_51 = np.ones(51).tolist()
            error_id.append([data['Gene'][i],data['pos'][i]])
        elif(position - 25 <= max_atac_len and position + 25 > max_atac_len):
            atac_variant_51 = bw.values("chr" + str(chr_no), position - 26, max_atac_len)
            for idx in range(51 - max_atac_len):
                atac_variant_51.append(0)
        else:
            atac_variant_51 = bw.values("chr" + str(chr_no), position - 26, position + 25)
                
        if(tss_position - 25 > max_atac_len):
            error_id.append([data['Gene'][i],data['pos'][i]])
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

    for id in error_id:
        data = data.drop(data[(data['Gene']==id[0]) & (data['pos']==id[1])].index)
    data = data.reset_index(drop=True)

    print(data)
    print(data['label'].value_counts())
    data.to_pickle(file_path + cell + '_seq_atac.pkl')

# 6 check sequence
gene_dict = {'A':[0,0,0,1], 'T':[0,0,1,0], 'C':[0,1,0,0], 'G':[1,0,0,0], 
             'a':[0,0,0,1], 't':[0,0,1,0], 'c':[0,1,0,0], 'g':[1,0,0,0],
             } 
for cell in disease_list:
    print(cell)
    new_data = pd.read_pickle(file_path + cell + '_seq_atac.pkl')
    del_row = []
    for i in range(0,len(new_data)):     
        for str in new_data['variant_51_seq'].values[i]:
            if(str not in gene_dict.keys() and [new_data['Gene'][i],new_data['pos'][i]] not in del_row):
                del_row.append([new_data['Gene'][i],new_data['pos'][i]])
        for str in new_data['variant_51_seq_after_mutation'].values[i]:
            if(str not in gene_dict.keys() and [new_data['Gene'][i],new_data['pos'][i]] not in del_row):
                del_row.append([new_data['Gene'][i],new_data['Pos'][i]])  
        for str in new_data['seq_between_variant_tss'].values[i]:
            if(str not in gene_dict.keys() and [new_data['GENE'][i],new_data['pos'][i]] not in del_row):
                del_row.append([new_data['Gene'][i],new_data['Pos'][i]])
    for id in del_row:
        new_data = new_data.drop(new_data[(new_data['Gene']==id[0]) & (new_data['pos']==id[1])].index)
    new_data = new_data.reset_index(drop=True)
    
    print(new_data)
    print(new_data['label'].value_counts())
    new_data.to_pickle(file_path + cell + '_final.pkl')