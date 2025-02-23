# python=3.9, tensorflow-gpu=2.7

import os
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub

# https://github.com/google-deepmind/deepmind-research/blob/master/enformer/README.md, one hot encoded in order 'ACGT'
eyes = np.eye(4)
gene_dict = {'A':eyes[0], 'C':eyes[1], 'G':eyes[2], 'T':eyes[3], 'N':np.zeros(4),
             'a':eyes[0], 'c':eyes[1], 'g':eyes[2], 't':eyes[3], 'n':np.zeros(4)
             }

enformer = hub.load("https://www.kaggle.com/models/deepmind/enformer/frameworks/TensorFlow2/variations/enformer/versions/1").model
print('test load successful!')
maxlen = 393216
fasta_path = '../../datasets/chr_fasta_hg38/'
compare_tissue_list = ['Adipose_Subcutaneous','Artery_Tibial','Breast_Mammary_Tissue','Colon_Transverse','Nerve_Tibial','Thyroid']

def mutation_center_seq(variant_id):
    chr_str = variant_id.split('_')[0]
    position = int(variant_id.split('_')[1])
    before_mutation = variant_id.split('_')[2]
    after_mutation = variant_id.split('_')[3]
    with open(fasta_path + chr_str + '_new.fasta') as fa:
        line = fa.readline()        
        range_seq = int(maxlen/2)       

        start_before = max(position - range_seq, 0)
        end_before = min(position + range_seq, len(line))
        start_after = max(position - range_seq, 0)
        end_after = min(position + range_seq, len(line))

        sequence_before = line[start_before:end_before]
        sequence_after = line[start_after:position - 1] + after_mutation + line[position:end_after]

        # padding 'N' for short sequences
        if start_before == 0:
            sequence_before = 'N' * (range_seq - position) + sequence_before
        if end_before == len(line):
            sequence_before = sequence_before + 'N' * (position + range_seq - len(line))
        if start_after == 0:
            sequence_after = 'N' * (range_seq - position) + sequence_after
        if end_after == len(line):
            sequence_after = sequence_after + 'N' * (position + range_seq - len(line))

    return sequence_before, sequence_after

def fetch_enformer_results(sequence):
    seq_list = []
    for strr in sequence:
        seq_list.append(gene_dict[strr])
    seq_array = np.array(seq_list)
    tensor = tf.convert_to_tensor(seq_array, tf.float32)
    tensor = tf.expand_dims(tensor, axis=0)
    result = enformer.predict_on_batch(tensor)['human']
    return np.array(result)

for tissue in compare_tissue_list:
    for model_size in ['large','middle','small']:
        for splittype in ['train','test']:
            data_all = pd.read_pickle('../../datasets/tissue_specific/' + model_size + '/' + splittype + '_' + model_size + '_' + tissue + '.pkl')[['phenotype_id','variant_id','tss_distance','label','bulk']]
            new_df = pd.DataFrame(columns=['variant_id', 'label', 'result_before', 'result_after'])
            for i in range(data_all.shape[0]):
                variant_id = data_all['variant_id'].values[i]
                sequence_before, sequence_after = mutation_center_seq(variant_id)
                result_before = fetch_enformer_results(sequence_before)
                result_after = fetch_enformer_results(sequence_after)
                #print(result_before.shape)
                new_df = new_df._append([{'variant_id': data_all['variant_id'][i], 'label': data_all['label'][i], 'result_before': result_before, 
                                                'result_after': result_after}], ignore_index=True)
            print(new_df.head())
            new_df.to_pickle('../../datasets/tissue_specific/enformer/' + splittype + '_' + model_size + '_' + tissue + '.pkl')