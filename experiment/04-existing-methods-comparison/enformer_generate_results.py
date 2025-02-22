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
fasta_path = '/data/eqtl/chr_fasta_hg38/'
compare_tissue_list = ['Adipose_Subcutaneous','Artery_Tibial','Breast_Mammary_Tissue','Colon_Transverse','Nerve_Tibial','Testis','Thyroid']

def mutation_center_seq(variant_id):
    chr_str = variant_id.split('_')[0]
    position = int(variant_id.split('_')[1])
    before_mutation = variant_id.split('_')[2]
    after_mutation = variant_id.split('_')[3]
    with open(fasta_path + chr_str + '_new.fasta') as fa:
        line = fa.readline()        
        range_seq = int(maxlen/2)       
        sequence_before = line[position - range_seq:position + range_seq]
        if(line[position - 1] >= 'a' and line[position - 1] <= 'z'):
            sequence_after = line[position - range_seq: position - 1] + after_mutation.lower() + line[position: position + range_seq]
        else:
            sequence_after = line[position - range_seq: position - 1] + after_mutation + line[position: position + range_seq]
    return sequence_before, sequence_after

def fetch_enformer_results(sequence):
    seq_list = []
    for strr in sequence:
        seq_list.append(gene_dict[strr])
    seq_array = np.array(seq_list)
    tensor = tf.convert_to_tensor(seq_array, tf.float32)
    tensor = tf.expand_dims(tensor, axis=0)
    result = enformer.predict_on_batch(tensor)['human']
    return result

for tissue in compare_tissue_list:

    # 1-1000 small
    train_all = pd.read_pickle('/data/eqtl/datasets/tissue_specific/small/train_small_' + tissue + '.pkl')
    test_all = pd.read_pickle('/data/eqtl/datasets/tissue_specific/small/test_small_' + tissue + '.pkl')
    
    train_df = pd.DataFrame(columns=['variant_id', 'label', 'result_before', 'result_after'])
    test_df = pd.DataFrame(columns=['variant_id', 'label', 'result_before', 'result_after'])

    for i in range(train_all.shape[0]):
        variant_id = train_all['variant_id'].values[i]
        sequence_before, sequence_after = mutation_center_seq(variant_id)
        if(len(sequence_before) != maxlen or len(sequence_after) != maxlen):
            continue
        result_before = fetch_enformer_results(sequence_before)
        result_after = fetch_enformer_results(sequence_after)
        print(result_before.shape)
        train_df = train_df._append([{'variant_id': train_all['variant_id'][i], 'label': train_all['label'][i], 'result_before': result_before, 
                                        'result_after': result_after}], ignore_index=True)
    
    for i in range(test_all.shape[0]):
        variant_id = test_all['variant_id'].values[i]
        sequence_before, sequence_after = mutation_center_seq(variant_id)
        if(len(sequence_before) != maxlen or len(sequence_after) != maxlen):
            continue
        result_before = fetch_enformer_results(sequence_before)
        result_after = fetch_enformer_results(sequence_after)
        test_df = test_df._append([{'variant_id': test_all['variant_id'][i], 'label': test_all['label'][i], 'result_before': result_before, 
                                        'result_after': result_after}], ignore_index=True)
    
    train_df.to_pickle('/data/eqtl/datasets/tissue_specific/enformer/train_small_' + tissue + '.pkl')
    test_df.to_pickle('/data/eqtl/datasets/tissue_specific/enformer/test_small_' + tissue + '.pkl')

    # 1001-10000 middle
    train_all = pd.read_pickle('/data/eqtl/datasets/tissue_specific/middle/train_middle_' + tissue + '.pkl')
    test_all = pd.read_pickle('/data/eqtl/datasets/tissue_specific/middle/test_middle_' + tissue + '.pkl')
    
    train_df = pd.DataFrame(columns=['variant_id', 'label', 'result_before', 'result_after'])
    test_df = pd.DataFrame(columns=['variant_id', 'label', 'result_before', 'result_after'])

    for i in range(train_all.shape[0]):
        variant_id = train_all['variant_id'].values[i]
        sequence_before, sequence_after = mutation_center_seq(variant_id)
        if(len(sequence_before) != maxlen or len(sequence_after) != maxlen):
            continue
        result_before = fetch_enformer_results(sequence_before)
        result_after = fetch_enformer_results(sequence_after)
        train_df = train_df._append([{'variant_id': train_all['variant_id'][i], 'label': train_all['label'][i], 'result_before': result_before, 
                                        'result_after': result_after}], ignore_index=True)
    
    for i in range(test_all.shape[0]):
        variant_id = test_all['variant_id'].values[i]
        sequence_before, sequence_after = mutation_center_seq(variant_id)
        if(len(sequence_before) != maxlen or len(sequence_after) != maxlen):
            continue
        result_before = fetch_enformer_results(sequence_before)
        result_after = fetch_enformer_results(sequence_after)
        test_df = test_df._append([{'variant_id': test_all['variant_id'][i], 'label': test_all['label'][i], 'result_before': result_before, 
                                        'result_after': result_after}], ignore_index=True)
    
    train_df.to_pickle('/data/eqtl/datasets/tissue_specific/enformer/train_middle_' + tissue + '.pkl')
    test_df.to_pickle('/data/eqtl/datasets/tissue_specific/enformer/test_middle_' + tissue + '.pkl')

    # 10001-100000 large
    train_all = pd.read_pickle('/data/eqtl/datasets/tissue_specific/large/train_large_' + tissue + '.pkl')
    test_all = pd.read_pickle('/data/eqtl/datasets/tissue_specific/large/test_large_' + tissue + '.pkl')
    
    train_df = pd.DataFrame(columns=['variant_id', 'label', 'result_before', 'result_after'])
    test_df = pd.DataFrame(columns=['variant_id', 'label', 'result_before', 'result_after'])

    for i in range(train_all.shape[0]):
        variant_id = train_all['variant_id'].values[i]
        sequence_before, sequence_after = mutation_center_seq(variant_id)
        if(len(sequence_before) != maxlen or len(sequence_after) != maxlen):
            continue
        result_before = fetch_enformer_results(sequence_before)
        result_after = fetch_enformer_results(sequence_after)
        train_df = train_df._append([{'variant_id': train_all['variant_id'][i], 'label': train_all['label'][i], 'result_before': result_before, 
                                        'result_after': result_after}], ignore_index=True)
    
    for i in range(test_all.shape[0]):
        variant_id = test_all['variant_id'].values[i]
        sequence_before, sequence_after = mutation_center_seq(variant_id)
        if(len(sequence_before) != maxlen or len(sequence_after) != maxlen):
            continue
        result_before = fetch_enformer_results(sequence_before)
        result_after = fetch_enformer_results(sequence_after)
        test_df = test_df._append([{'variant_id': test_all['variant_id'][i], 'label': test_all['label'][i], 'result_before': result_before, 
                                        'result_after': result_after}], ignore_index=True)
    
    train_df.to_pickle('/data/eqtl/datasets/tissue_specific/enformer/train_large_' + tissue + '.pkl')
    test_df.to_pickle('/data/eqtl/datasets/tissue_specific/enformer/test_large_' + tissue + '.pkl')




