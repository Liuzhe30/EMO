import os
import numpy as np
import pandas as pd
import warnings
import json
warnings.filterwarnings('ignore')

cell_list = ['B_Naive','CD4_memory','CD4_naive','CD8_memory','Dendritic_Cell','Natural_killer_Cell']

# 1 filter TSS annotation
#'''
tss = pd.read_csv('../../datasets/sceQTL/farthest_tss.csv')
gene_list = tss['name2'].values
#print(gene_list)

for cell in cell_list:
    data = pd.read_pickle('../../datasets/sceQTL/head/' + cell + '.pkl')
    new_data = pd.DataFrame(columns=['RSID', 'CHR', 'GENE','A1','A2','RHO'])
    for i in range(data.shape[0]):
        gene = data['GENE'][i]
        if(gene in gene_list and np.abs(data['SPEARMANS_RHO'][i]) > 0.2): # update in 14_onek1k_atac_mapping.py
            new_data = new_data.append([{'RSID': data['RSID'][i], 'CHR': data['CHR'][i], 'GENE': data['GENE'][i], 
                                        'A1': data['A1'][i], 'A2': data['A2'][i], 'RHO': data['SPEARMANS_RHO'][i]}], ignore_index=True)
    print(new_data.shape[0])
    new_data.to_pickle('../../datasets/sceQTL/head/' + cell + '_filtered.pkl')
#'''
#'''
# 2 mapping RSID

#rs_list = []
for cell in cell_list:
    with open('../../datasets/sceQTL/rsid/' + cell + '.txt', 'w+') as w:
        data = pd.read_pickle('../../datasets/sceQTL/head/' + cell + '_filtered.pkl')
        for j in range(data.shape[0]):
            RSID = data['RSID'][j]
            w.write(RSID + '\n')

#'''
# 3 propess VEP files
#'''
for cell in cell_list:
    rs_id = []
    rs_dict = {}
    with open('../../datasets/sceQTL/rsid/' + cell + '_vep.txt') as r:
        line = r.readline()
        while line:
            if(line[0] == 'r'):
                rs = line.split()[0]
                if(rs not in rs_id):
                    rs_id.append(rs)
                    pos = line.split()[1].split('-')[-1]
                    rs_dict[rs] = int(pos)
            line = r.readline()
    with open('../../datasets/sceQTL/rsid/' + cell + '_mapping.json','w') as f:
        json.dump(rs_dict,f)
#'''

# 4 convert hg19 position to hg38 & mapping tss & merge file

tss = pd.read_csv('../../datasets/sceQTL/farthest_tss.csv')

for cell in cell_list:
    print(cell)
    with open('../../datasets/sceQTL/rsid/' + cell + '_mapping.json') as f:
        rs_dict = json.load(f)
        data = pd.read_pickle('../../datasets/sceQTL/head/' + cell + '_filtered.pkl')
        new_data = pd.DataFrame(columns=['RSID', 'CHR', 'GENE','A1','A2','RHO','TSS','POS','label'])
        for i in range(data.shape[0]):
            if(data['RSID'][i] in rs_dict.keys()):
                select = tss[(tss['chrom'] == 'chr' + str(data['CHR'][i])) & (tss['name2'] == data['GENE'][i])]
                TSS = select['txStart'].values[0]
                if(float(data['RHO'][i] > 0)):
                    label = 1
                else:
                    label = 0
                POS = rs_dict[data['RSID'][i]]
            new_data = new_data.append([{'RSID': data['RSID'][i], 'CHR': data['CHR'][i], 'GENE': data['GENE'][i], 
                                        'A1': data['A1'][i], 'A2': data['A2'][i], 
                                        'RHO': data['RHO'][i],'TSS':TSS,'POS':POS,
                                        'label':label}], ignore_index=True)
        
        print(new_data)
        print(new_data['label'].value_counts())
        new_data.to_pickle('../../datasets/sceQTL/merged/' + cell + '_hg38.pkl')
