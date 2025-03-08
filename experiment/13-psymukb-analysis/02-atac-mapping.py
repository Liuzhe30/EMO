# Hippocampus: BSS00091 
# Cerebellum: BSS00207 
# Prefrontal Cortex: BSS00369 
# Temporal Lobe: BSS01714
# Substantia Nigra: BSS01675

import pandas as pd
import pyBigWig
import math
import numpy as np
pd.set_option('display.max_columns', None)

disease_list = ['Autism_ASD','Developmental_Delay_DD','Fetal_non-Preterm_birth_non-PTB','Fetal_preterm_birth_PTB','Mix_Autism_or_Schizophrenia','Sibling_Control']
# 1000,24,863,497,198,1000
tissue_list = ['hippocampus','cerebellum','prefrontal_cortex','temporal_lobe','substantia_nigra']

def check(check_list):
    for elem in check_list:
        if(float(elem) == np.nan):
            return 0
        if(math.isnan(elem) == True):
            return 0
    return 1

for disease in disease_list:
    data = pd.read_csv('datasets/' + disease + '.csv')
    for tissue in tissue_list:
        bw = pyBigWig.open('datasets/' + tissue + ".bigWig")
        data[tissue + '_between'] = 0
        data[tissue + '_variant'] = 0
        data[tissue + '_between'] = data[tissue + '_between'].astype('object')
        data[tissue + '_variant'] = data[tissue + '_variant'].astype('object')
        for i in range(len(data)):
            chr_no = data['Chr'].values[i]
            max_atac_len = int(bw.chroms("chr" + str(chr_no)))
            position = int(data['position'].values[i])
            tss_distance = int(data['TSS_distance'].values[i])
            tss_position = int(data['TSS_position'].values[i])

            if(tss_distance > 0):
                if(tss_position > max_atac_len):
                    atac_between = np.ones(tss_distance + 1).tolist()
                elif(tss_position <= max_atac_len and position > max_atac_len):
                    atac_between = bw.values("chr" + str(chr_no), tss_position - 1, max_atac_len)
                    for idx in range(position - max_atac_len):
                        atac_between.append(0)
                else:
                    atac_between = bw.values("chr" + str(chr_no), tss_position - 1, position)
            else:
                if(position > max_atac_len):
                    atac_between = np.ones(-tss_distance + 1).tolist()
                elif(position <= max_atac_len and tss_position > max_atac_len):
                    atac_between = bw.values("chr" + str(chr_no), position - 1, max_atac_len)
                    for idx in range(tss_position - max_atac_len):
                        atac_between.append(0)
                else:                    
                    atac_between = bw.values("chr" + str(chr_no), position - 1, tss_position)
            
            if(position - 25 > max_atac_len):
                atac_variant_51 = np.ones(51).tolist()
            elif(position - 25 <= max_atac_len and position + 25 > max_atac_len):
                atac_variant_51 = bw.values("chr" + str(chr_no), position - 26, max_atac_len)
                for idx in range(51 - max_atac_len):
                    atac_variant_51.append(0)
            else:
                atac_variant_51 = bw.values("chr" + str(chr_no), position - 26, position + 25)  
            
            atac_between = [0 if math.isnan(x) else x for x in atac_between]
            atac_variant_51 = [0 if math.isnan(x) else x for x in atac_variant_51]
            
            data.at[i, tissue + '_between'] = atac_between
            data.at[i, tissue + '_variant'] = atac_variant_51
    print(data.head())
    data.to_pickle('datasets/' + disease + '.pkl')

# post-process
for disease in disease_list:
    data = pd.read_pickle('datasets/' + disease + '.pkl')
    data_check = pd.read_pickle('datasets/' + disease + '.pkl')
    for tissue in tissue_list:
        for i in range(data_check.shape[0]):
            atac_between = data_check[tissue + '_between'][i]
            atac_variant_51 = list(data_check[tissue + '_variant'][i])
            chr_no = data_check['Chr'].values[i]
            ref = data_check['Ref'].values[i]
            alt = data_check['Alt'].values[i]
            if(check(atac_between)==0 or check(atac_variant_51)==0):
                data = data[~(data['chr_no'].isin([chr_no])&data['ref'].isin([ref])&data['alt'].isin([alt]))]
    data = data.reset_index(drop=True)
    print(data.head())
    data.to_pickle('datasets/' + disease + '_post.pkl')


'''
   Chr Ref Alt PrimaryPhenotype      DisorderCategory Gene.refGene   position  \
0    9   T   C     Autism (ASD)  psychiatric disorder     ANKRD19P   92819845
1   10   T   A     Autism (ASD)  psychiatric disorder    ZMIZ1-AS1   78996445
2   10   A   G     Autism (ASD)  psychiatric disorder    WDR11-AS1  120824851
3    9   A   G     Autism (ASD)  psychiatric disorder    VLDLR-AS1    2614615
4   14   G   T     Autism (ASD)  psychiatric disorder    LINC01550   97970375

   TSS_position  TSS_distance  \
0      92809610         10235
1      79067448        -71003
2     120851179        -26328
3       2622373         -7758
4      97978124         -7749

                                 hippocampus_between  \
0  [0.3700000047683716, 0.3700000047683716, 0.370...
1  [0.9700000286102295, 0.9700000286102295, 0.970...
2  [0.6700000166893005, 0.6700000166893005, 0.670...
3  [0.8799999952316284, 0.8799999952316284, 0.879...
4  [0.47999998927116394, 0.47999998927116394, 0.4...

                                 hippocampus_variant  \
0  [0.7699999809265137, 0.7699999809265137, 0.769...
1  [0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.9700000...
2  [0.6899999976158142, 0.6899999976158142, 0.689...
3  [0.9800000190734863, 0.9800000190734863, 0.980...
4  [0.5199999809265137, 0.47999998927116394, 0.47...

                                  cerebellum_between  \
0  [0.4300000071525574, 0.4300000071525574, 0.430...
1  [0.7099999785423279, 0.7099999785423279, 0.709...
2  [0.5600000023841858, 0.5600000023841858, 0.560...
3  [0.9200000166893005, 0.9200000166893005, 0.920...
4  [0.5600000023841858, 0.5, 0.5, 0.5, 0.5, 0.5, ...

                                  cerebellum_variant  \
0  [0.800000011920929, 0.800000011920929, 0.80000...
1  [0.7300000190734863, 0.7300000190734863, 0.730...
2  [0.5299999713897705, 0.5299999713897705, 0.529...
3  [1.0099999904632568, 1.0099999904632568, 1.009...
4  [0.5899999737739563, 0.5600000023841858, 0.560...

                           prefrontal_cortex_between  \
0  [0.3400000035762787, 0.3400000035762787, 0.340...
1  [0.5899999737739563, 0.5899999737739563, 0.589...
2  [0.6499999761581421, 0.6499999761581421, 0.649...
3  [0.8799999952316284, 0.8799999952316284, 0.879...
4  [0.38999998569488525, 0.36000001430511475, 0.3...

                           prefrontal_cortex_variant  \
0  [0.6899999976158142, 0.6899999976158142, 0.689...
1  [0.46000000834465027, 0.46000000834465027, 0.4...
2  [0.6800000071525574, 0.6800000071525574, 0.680...
3  [0.8999999761581421, 0.8999999761581421, 0.899...
4  [0.46000000834465027, 0.38999998569488525, 0.3...

                               temporal_lobe_between  \
0  [0.3499999940395355, 0.3499999940395355, 0.349...
1  [0.8799999952316284, 0.8799999952316284, 0.879...
2  [0.6499999761581421, 0.6499999761581421, 0.649...
3  [0.9800000190734863, 0.9800000190734863, 0.980...
4  [0.5099999904632568, 0.4399999976158142, 0.439...

                               temporal_lobe_variant  \
0  [0.800000011920929, 0.800000011920929, 0.80000...
1  [0.7699999809265137, 0.7699999809265137, 0.769...
2  [0.6899999976158142, 0.6899999976158142, 0.689...
3  [1.0099999904632568, 1.0099999904632568, 1.009...
4  [0.5600000023841858, 0.5099999904632568, 0.509...

                            substantia_nigra_between  \
0  [0.36000001430511475, 0.36000001430511475, 0.3...
1  [0.7400000095367432, 0.7400000095367432, 0.740...
2  [0.6899999976158142, 0.6899999976158142, 0.689...
3  [0.9399999976158142, 0.9399999976158142, 0.939...
4  [0.4699999988079071, 0.4300000071525574, 0.430...

                            substantia_nigra_variant
0  [0.7599999904632568, 0.7599999904632568, 0.759...
1  [0.7599999904632568, 0.7599999904632568, 0.759...
2  [0.6399999856948853, 0.6399999856948853, 0.639...
3  [0.949999988079071, 0.949999988079071, 0.94999...
4  [0.5, 0.4699999988079071, 0.4699999988079071, ...
'''
