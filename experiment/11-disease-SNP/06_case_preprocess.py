# case preprocess
import pandas as pd
from pandas import read_parquet
import pyBigWig
import numpy as np
import math
pd.set_option('display.max_columns', None)

# step 1, tss mapping
tss_df = pd.read_csv('data/tss_annotations.csv')
snp_df = pd.read_csv('data/case/case_raw.csv')

for i in range(len(snp_df)):
    snp_small = pd.DataFrame()
    snp_middle = pd.DataFrame()
    snp_large = pd.DataFrame()
    snp_huge = pd.DataFrame()

    rsid = snp_df['RSID'][i]
    chr = str(snp_df['CHR'][i])
    pos = int(snp_df['POS'][i])
    ref = snp_df['Ref'][i]
    alt = snp_df['Alt'][i]

    tss_chr_df = tss_df[tss_df['chrom'] == "chr" + chr].reset_index(drop=True)
    for j in range(len(tss_chr_df)):
        tss_pos = tss_chr_df['tss'][j]
        gene_name = tss_chr_df['gene_name'][j]
        distance = pos - tss_pos
        if(pos-1_000 < tss_pos < pos+1_000):
            snp_small = snp_small._append({'TSS_POS':tss_pos, 'gene':gene_name,'variant_id':rsid,'slope':0,'Ref':ref,'Alt':alt,'CHR':'chr'+chr,
                                            'SNP_POS':pos,'label':0,'tss_distance':distance},ignore_index=True)
        elif(pos-10_000 < tss_pos < pos-1_000 or pos+1_000 < tss_pos < pos+10_000):
            snp_middle = snp_middle._append({'TSS_POS':tss_pos, 'gene':gene_name,'variant_id':rsid,'slope':0,'Ref':ref,'Alt':alt,'CHR':'chr'+chr,
                                            'SNP_POS':pos,'label':0,'tss_distance':distance},ignore_index=True)         
        elif(pos-100_000 < tss_pos < pos-10_000 or pos+10_000 < tss_pos < pos+100_000):
            snp_large = snp_large._append({'TSS_POS':tss_pos, 'gene':gene_name,'variant_id':rsid,'slope':0,'Ref':ref,'Alt':alt,'CHR':'chr'+chr,
                                            'SNP_POS':pos,'label':0,'tss_distance':distance},ignore_index=True)                           
        elif(pos-1_000_000 < tss_pos < pos-100_000 or pos+100_000 < tss_pos < pos+1_000_000):
            snp_huge = snp_huge._append({'TSS_POS':tss_pos, 'gene':gene_name,'variant_id':rsid,'slope':0,'Ref':ref,'Alt':alt,'CHR':'chr'+chr,
                                            'SNP_POS':pos,'label':0,'tss_distance':distance},ignore_index=True)

    #print(snp_small)
    #print(snp_middle)
    print(snp_large)
    #print(snp_huge)

    snp_large.to_pickle('data/case/' + rsid + '_large.pkl')

'''
    TSS_POS          gene variant_id  slope Ref Alt    CHR   SNP_POS  label  \
0  42403902       UBASH3A  rs1893592      0   A   G  chr21  42434957      0
1  42513852       SLC37A1  rs1893592      0   A   G  chr21  42434957      0
2  42396052       TMPRSS3  rs1893592      0   A   G  chr21  42434957      0
3  42389149       TMPRSS3  rs1893592      0   A   G  chr21  42434957      0
4  42499622       SLC37A1  rs1893592      0   A   G  chr21  42434957      0
5  42366535          TFF1  rs1893592      0   A   G  chr21  42434957      0
6  42496539  LOC101930094  rs1893592      0   A   G  chr21  42434957      0
7  42350994          TFF2  rs1893592      0   A   G  chr21  42434957      0
8  42496224         RSPH1  rs1893592      0   A   G  chr21  42434957      0

   tss_distance
0         31055
1        -78895
2         38905
3         45808
4        -64665
5         68422
6        -61582
7         83963
8        -61267
'''

# step 2, mapping sequence
fasta_path = '../../datasets/chr_fasta_hg38/'
file_path = 'data/case/rs1893592_large.pkl'
output_path = 'data/case/rs1893592_large_seq.pkl'

data = pd.read_pickle(file_path)
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

data.to_pickle(output_path)

# step 3, atac mapping
atac_path = 'hg38_ATAC/hg38_t0_atac.bigWig'
output_path = 'data/case/rs1893592_large_atac_t0.dataset'
file_path = 'data/case/rs1893592_large_seq.pkl'

data = pd.read_pickle(file_path)
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
data.to_pickle(output_path)

# step 3, mapping ATAC-seq
atac_path = 'hg38_ATAC/hg38_t24h_atac.bigWig'
output_path = 'data/case/rs1893592_large_atac_t24.dataset'
file_path = 'data/case/rs1893592_large.pkl'

data = pd.read_pickle(file_path)
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
data.to_pickle(output_path)