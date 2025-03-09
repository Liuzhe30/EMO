import pandas as pd
import math
import numpy as np
pd.set_option('display.max_columns', None)

from src.utils_sign_prediction import *
from src.utils_slope_prediction import *

window_len = 51
disease_list = ['Autism_ASD','Developmental_Delay_DD','Fetal_non-Preterm_birth_non-PTB','Fetal_preterm_birth_PTB','Mix_Autism_or_Schizophrenia','Sibling_Control']
# 1000,24,863,497,198,1000
tissue_list = ['hippocampus','cerebellum','prefrontal_cortex','temporal_lobe','substantia_nigra']

# Define path of reference genome 
genome_path = 'reference_genome_hg38/' # In this case, 'reference_genome_hg38/chr19.fasta' will be used.
# Define path of pretrained model weights 
weights_path = 'trained_weights/' #  In this case, 'trained_weights/small_trained_weights.tf' and 'trained_weights/small_slope_weights.tf' will be used.

for disease in disease_list:
    data = pd.read_pickle('datasets/' + disease + '_post.pkl')
    for tissue in tissue_list:
        data[tissue + '_slope'] = 0
        data[tissue + '_sign'] = 0
        for i in range(len(data)):
            chr_no = data['Chr'].values[i]
            ref = data['Ref'].values[i]
            alt = data['Alt'].values[i]
            position = int(data['position'].values[i])

            input_variant = 'chr' + str(chr_no) + '_' + str(position) + '_' + ref + '_' + alt
            TSS_distance = int(data['TSS_distance'].values[i])
            atac_between = np.array(data[tissue + '_between'].values[i])
            atac_variant = np.array(data[tissue + '_variant'].values[i])

            slope_prediction_output = get_slope_prediction_result(input_variant, TSS_distance, atac_variant, atac_between, genome_path, weights_path) 
            sign_prediction_output = get_sign_prediction_result(input_variant, TSS_distance, atac_variant, atac_between, genome_path, weights_path) 
            data[tissue + '_slope'][i] = slope_prediction_output
            data[tissue + '_sign'][i] = sign_prediction_output
        print(data.head())    
    print(data.head()) 
    data.to_pickle('datasets/' + disease + '_pred.pkl')