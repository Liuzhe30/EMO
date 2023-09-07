import os
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

tissue_list = ['cerebellum','cortex','hippocampus','spinalcord']
#tissue_list = ['cerebellum']

# 1 check effect allele and fetch columns
#'''
for tissue in tissue_list:
    data = pd.read_csv('../../datasets/metabrain/raw/' + tissue + '.txt', sep='\t')
    #print(data)
    new_data = pd.DataFrame(columns=['gene', 'chr','pos', 'A1', 'A2', 'MetaBeta', 'label'])
    for i in range(data.shape[0]):
        gene = data['GeneSymbol'][i]
        chr = data['SNPChr'][i]
        pos = data['SNPPos'][i]
        A1 = data['SNPAlleles'][i].split('/')[0]
        A2 = data['SNPAlleles'][i].split('/')[1]
        MetaBeta = float(data['MetaBeta'][i])
        if(A2 != data['SNPEffectAllele'][i]):
            continue
        if(MetaBeta > 0):
            label = 1
        else:
            label = 0
        new_data = new_data.append([{'gene': gene, 'chr': chr, 'pos':pos, 'A1': A1, 'A2': A2, 'MetaBeta': MetaBeta, 'label':label}], ignore_index=True)
    print(new_data)
    new_data.to_pickle('../../datasets/metabrain/filtered/' + tissue + '.pickle')
#'''

# 2 mapping tss
tss = pd.read_csv('../../datasets/metabrain/farthest_tss.csv')
for tissue in tissue_list:
    data = pd.read_pickle('../../datasets/metabrain/filtered/' + tissue + '.pickle')
    new_data = pd.DataFrame(columns=['CHR', 'GENE','A1','A2','RHO','TSS','POS','label'])
    for i in range(data.shape[0]):
        select = tss[(tss['chrom'] == 'chr' + str(data['chr'][i])) & (tss['name2'] == data['gene'][i])]
        if(select.shape[0] != 0):
            TSS = select['txStart'].values[0]
            new_data = new_data.append([{'CHR': data['chr'][i], 'GENE': data['gene'][i], 
                                        'A1': data['A1'][i], 'A2': data['A2'][i], 
                                        'RHO': data['MetaBeta'][i],'TSS':TSS,'POS':data['pos'][i],
                                        'label':data['label'][i]}], ignore_index=True)
    print(new_data)
    new_data.to_pickle('../../datasets/metabrain/filtered/' + tissue + '_tss.pickle')    

        