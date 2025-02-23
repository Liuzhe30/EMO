# model weights downloaded from https://github.com/calico/basenji/tree/master/manuscripts/cross2020
# python=3.8, tensorflow-gpu=2.7
import numpy as np
import json
import h5py
import tensorflow as tf
import pandas as pd
from basenji import seqnn
from basenji import dna_io

maxlen = 131072
model_file = 'bin/model_human.h5'
params_file = 'bin/params_human.json'
fasta_path = 'chr_fasta_hg38/'
# read model parameters
with open(params_file) as params_open:
    params = json.load(params_open)
params_model = params['model']
params_train = params['train']
seqnn_model = seqnn.SeqNN(params_model)
seqnn_model.restore(model_file)
seqnn_model.build_slice(target_slice=None)  # Not using a slice of targets
targets_length = seqnn_model.target_lengths[0]
num_targets = seqnn_model.num_targets()

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
            
        sequence_before = sequence_before.upper()
        sequence_after = sequence_after.upper()
    return sequence_before, sequence_after

def process_dna_sequence(dna_sequence):
    seq_1hot = dna_io.dna_1hot(dna_sequence)  # OneHot encode the sequence
    seq_1hot = np.expand_dims(seq_1hot, axis=0)  # Add batch dimension
    preds = seqnn_model.predict(seq_1hot)  # Get the predictions from the model
    last_dense_output = preds[-1]  
    return last_dense_output

compare_tissue_list = ['Adipose_Subcutaneous','Artery_Tibial','Breast_Mammary_Tissue','Colon_Transverse','Nerve_Tibial','Thyroid']
for tissue in compare_tissue_list:
    for model_size in ['small','middle']:
        for splittype in ['test','train']:
            data_all = pd.read_pickle(model_size + '/' + splittype + '_' + model_size + '_' + tissue + '.pkl')[['phenotype_id','variant_id','tss_distance','label','bulk']]
            new_df = pd.DataFrame(columns=['variant_id', 'label', 'result_before', 'result_after'])
            for i in range(data_all.shape[0]):
                variant_id = data_all['variant_id'].values[i]
                sequence_before, sequence_after = mutation_center_seq(variant_id)
                if(len(sequence_before) != maxlen or len(sequence_after) != maxlen):
                    continue
                result_before = process_dna_sequence(sequence_before)
                result_after = process_dna_sequence(sequence_after)
                #print(result_before.shape) #(896,5313)
                new_df = new_df._append([{'variant_id': data_all['variant_id'][i], 'label': data_all['label'][i], 'result_before': result_before, 
                                        'result_after': result_after}], ignore_index=True)
            print(new_df.head())
            new_df.to_pickle('basenji_results/' + splittype + '_' + model_size + '_' + tissue + '.pkl')