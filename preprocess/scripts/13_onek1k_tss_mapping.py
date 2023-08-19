import os
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# 1 filtering the legal chrom
#'''
genome_field = pd.read_csv('../../datasets/sceQTL/genome_all_field.csv',sep='\t')
genome_field = genome_field.drop(['exonEnds','score','cdsStartStat','cdsEndStat','exonFrames','exonStarts','cdsStart','cdsEnd','exonCount'],axis=1)
#print(genome_field['chrom'].unique())
chr_list = ['chrX' 'chrY']
for i in range(1,22):
    chr_list.append('chr' + str(i))
new_genome_field = genome_field
for i in range(new_genome_field.shape[0]):
    if(new_genome_field['chrom'][i] not in chr_list):
        new_genome_field = new_genome_field.drop([i])
new_genome_field.sort_values('name2',inplace=True)
new_genome_field = new_genome_field.reset_index(drop=True)

print(new_genome_field.head())
new_genome_field.to_csv('../../datasets/sceQTL/clean_genome_tss.csv',index=False)
#'''

# 2 mapping TSS with chrom
#'''
cell_list = ['B_Naive','CD4_memory','CD4_naive','CD8_memory','Dendritic_Cell','Natural_killer_Cell']
genome_field = pd.read_csv('../../datasets/sceQTL/clean_genome_tss.csv')
new_genome_field = pd.DataFrame(columns=['name', 'name2', 'chrom', 'txStart', 'txEnd'])
check_list = []
for cell in cell_list:
    data = pd.read_pickle('../../datasets/sceQTL/head/' + cell + '.pkl')
    for j in range(data.shape[0]):
        pair = [data['CHR'][j],data['GENE'][j]]
        if(pair not in check_list):
            check_list.append(pair)
            select = genome_field[(genome_field['chrom'] == 'chr' + str(data['CHR'][j])) & (genome_field['name2'] == data['GENE'][j])]
            select = select.reset_index(drop=True)
            #print(select)
            for i in range(select.shape[0]):
                new_genome_field.append([{'name': select['name'][i], 'name2': select['name2'][i], 'chrom': select['chrom'][i], 
                                        'txStart': select['txStart'][i], 'txEnd': select['txEnd'][i]}], ignore_index=True)
print(new_genome_field)
new_genome_field.to_csv('../../datasets/sceQTL/filtered_genome_tss.csv',index=False)
#'''

# 3 filtering the nearest TSS
gene_list = []