from chrombpnet.evaluation.variant_effect_prediction.snp_generator import SNPGenerator
from scipy.spatial.distance import jensenshannon
from tensorflow.keras.utils import get_custom_objects
from tensorflow.keras.models import load_model
import chrombpnet.training.utils.losses as losses
import pandas as pd
import numpy as np
import pickle as pkl
import tensorflow as tf

# Define column names for SNP data
SNP_SCHEMA = ["CHR", "POS0", "REF", "ALT", "META_DATA"]

compare_tissue_list = ['Adipose_Subcutaneous','Artery_Tibial','Breast_Mammary_Tissue','Colon_Transverse','Nerve_Tibial','Thyroid']

def softmax(x, temp=1):
    # Compute softmax with temperature scaling
    norm_x = x - np.mean(x, axis=1, keepdims=True)
    return np.exp(temp * norm_x) / np.sum(np.exp(temp * norm_x), axis=1, keepdims=True)

def load_model_wrapper(model_path):
    # Load the ChromBPNet model from .h5 file
    custom_objects = {"tf": tf, "multinomial_nll": losses.multinomial_nll}
    get_custom_objects().update(custom_objects)
    model = load_model(model_path, compile=False)
    print("Model loaded successfully")
    return model

def fetch_snp_predictions(model, snp_regions, inputlen, genome_fasta, batch_size, debug_mode_on=False):
    '''
    Returns model predictions (counts and profile probabilities) for reference and alternate alleles.
    If the SNP is at the edge of the chromosome (unable to form a complete input sequence), it is skipped.

    Arguments:
        model: ChromBPNet model (.h5 file).
        snp_regions: DataFrame with SNP information, columns: ["CHR", "POS0", "REF", "ALT"].
        inputlen: Input sequence length, SNP is inserted in the middle.
        genome_fasta: Path to reference genome FASTA file.
        batch_size: Batch size for model predictions.
        debug_mode_on: Debug mode (0 or 1).

    Returns:
        rsids: List of SNP IDs.
        ref_logcount_preds: Log count predictions for reference allele.
        alt_logcount_preds: Log count predictions for alternate allele.
        ref_prob_preds: Profile probability predictions for reference allele.
        alt_prob_preds: Profile probability predictions for alternate allele.
    '''
    rsids = []
    ref_logcount_preds = []
    alt_logcount_preds = []
    ref_prob_preds = []
    alt_prob_preds = []

    # SNP sequence generator
    snp_gen = SNPGenerator(snp_regions=snp_regions,
                          inputlen=inputlen,
                          genome_fasta=genome_fasta,
                          batch_size=batch_size,
                          debug_mode_on=debug_mode_on)

    for i in range(len(snp_gen)):
        batch_rsids, ref_seqs, alt_seqs = snp_gen[i]

        # Get model predictions for reference and alternate sequences
        ref_batch_preds = model.predict(ref_seqs)
        alt_batch_preds = model.predict(alt_seqs)

        # Append predictions to lists
        ref_logcount_preds.extend(np.squeeze(ref_batch_preds[1]))
        alt_logcount_preds.extend(np.squeeze(alt_batch_preds[1]))

        ref_prob_preds.extend(np.squeeze(softmax(ref_batch_preds[0])))
        alt_prob_preds.extend(np.squeeze(softmax(alt_batch_preds[0])))

        rsids.extend(batch_rsids)

    return np.array(rsids), np.array(ref_logcount_preds), np.array(alt_logcount_preds), np.array(ref_prob_preds), np.array(alt_prob_preds)

def predict_snp_effect_scores(rsids, ref_count_preds, alt_count_preds, ref_prob_preds, alt_prob_preds):
    '''
    Computes variant effect scores based on model predictions.

    Arguments:
        ref_logcount_preds: Log count predictions for reference allele.
        alt_logcount_preds: Log count predictions for alternate allele.
        ref_prob_preds: Profile probability predictions for reference allele.
        alt_prob_preds: Profile probability predictions for alternate allele.

    Returns:
        log_counts_diff: Difference in log count predictions between alternate and reference alleles.
        log_probs_diff_abs_sum: Sum of absolute differences in log probabilities between alternate and reference alleles.
        probs_jsd_diff: Jensen-Shannon distance between profile probability predictions of alternate and reference alleles.
    '''
    log_counts_diff = alt_count_preds - ref_count_preds
    log_probs_diff_abs_sum = np.sum(np.abs(np.log(alt_prob_preds) - np.log(ref_prob_preds)), axis=1) * np.sign(log_counts_diff)
    probs_jsd_diff = np.array([jensenshannon(x, y) for x, y in zip(alt_prob_preds, ref_prob_preds)]) * np.sign(log_counts_diff)

    return log_counts_diff, log_probs_diff_abs_sum, probs_jsd_diff

for tissue in compare_tissue_list:
    for model_size in ['small']:
        for splittype in ['test','train']:
            data = pd.read_pickle(model_size + '/' + splittype + '_' + model_size + '_' + tissue + '.pkl')[['phenotype_id','variant_id','tss_distance','label','bulk']]

            # Extract mutation information
            snp_regions = data["variant_id"].str.split("_", expand=True)
            snp_regions.columns = ["CHR", "POS0", "REF", "ALT", "GENOME"]
            snp_regions["META_DATA"] = data['label']
            snp_regions["label"] = data['label']
            snp_regions["POS0"] = snp_regions["POS0"].astype(int)

            # Paths to model and reference genome
            model_path = "ENCSR146KFX_bias_fold_0.h5"
            genome_fasta = "../ExPecto/resources/hg19.fa"

            # Load the model
            model = load_model_wrapper(model_path)

            # Infer input length from the model
            inputlen = model.input_shape[1]
            print("Inferred input length: ", inputlen)

            # Fetch model predictions for SNPs
            rsids, ref_logcount_preds, alt_logcount_preds, ref_prob_preds, alt_prob_preds = fetch_snp_predictions(
                model, snp_regions, inputlen, genome_fasta, batch_size=64, debug_mode_on=False
            )

            # Compute variant effect scores
            log_counts_diff, log_probs_diff_abs_sum, probs_jsd_diff = predict_snp_effect_scores(
                rsids, ref_logcount_preds, alt_logcount_preds, ref_prob_preds, alt_prob_preds
            )

            # Save results to a TSV file
            snp_effect_scores_pd = pd.DataFrame()
            snp_effect_scores_pd[["CHR", "POS0", "REF", "ALT", "META_DATA"]] = pd.Series(rsids).str.split('_', expand=True)
            snp_effect_scores_pd["log_counts_diff"] = log_counts_diff
            snp_effect_scores_pd["log_probs_diff_abs_sum"] = log_probs_diff_abs_sum
            snp_effect_scores_pd["probs_jsd_diff"] = probs_jsd_diff
            print(snp_effect_scores_pd.head())
            snp_effect_scores_pd.to_pickle('chrombpnet_results/' + splittype + '_' + model_size + '_' + tissue + '.pkl')

