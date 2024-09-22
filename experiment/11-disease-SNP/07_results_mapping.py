# mapping results
import pandas as pd
import numpy as np

pd.set_option('display.max_columns', None)
import warnings
warnings.filterwarnings('ignore')

snp_list = ['rs2476601','rs3806624', 'rs7731626', 'rs2234067','rs2233424','rs947474','rs3824660','rs968567','rs3218251']

pred_results_path = 'data/prediction_results/'
full_t0_path = 'data/atac_mapping_t0/'
full_t24_path = 'data/atac_mapping_t24/'
output_path = 'data/pred_merged/'

model_dict = {"small":1_000,"middle":10_000,"large":100_000,"huge":1_000_000}

for snp in snp_list:
    for model in model_dict.keys():
        full_t0_data = pd.read_pickle(full_t0_path + snp + '_' + model + '.dataset')
        if(len(full_t0_data)!=0):
            full_t24_data = pd.read_pickle(full_t24_path + snp + '_' + model + '.dataset')
            t0_npy = np.load(pred_results_path + snp + '_' + model + '_t0.npy')
            t24_npy = np.load(pred_results_path + snp + '_' + model + '_t24.npy')

            full_t0_data['t0'] = 0.0
            full_t24_data['t24'] = 0.0
            for i in range(len(full_t0_data)):
                full_t0_data['t0'][i] = t0_npy[i][0]
            for i in range(len(full_t24_data)):
                full_t24_data['t24'][i] = t24_npy[i][0]

            # crossmap
            gene_list1 = full_t0_data['gene'].tolist()
            gene_list2 = full_t24_data['gene'].tolist()
            final_list = [x for x in gene_list1 if x in gene_list2]

            final_df = pd.DataFrame()
            for item in final_list:
                gene = full_t0_data[full_t0_data['gene']==item]['gene'].values[0]
                snp = full_t0_data[full_t0_data['gene']==item]['variant_id'].values[0]
                ref = full_t0_data[full_t0_data['gene']==item]['Ref'].values[0]
                alt = full_t0_data[full_t0_data['gene']==item]['Alt'].values[0]
                chr = full_t0_data[full_t0_data['gene']==item]['CHR'].values[0]
                tss_pos = full_t0_data[full_t0_data['gene']==item]['TSS_POS'].values[0]
                snp_pos = full_t0_data[full_t0_data['gene']==item]['SNP_POS'].values[0]
                distance = full_t0_data[full_t0_data['gene']==item]['tss_distance'].values[0]
                t0 = full_t0_data[full_t0_data['gene']==item]['t0'].values[0]
                t24 = full_t24_data[full_t24_data['gene']==item]['t24'].values[0]
                delta = t24 - t0
                final_df = final_df._append({'gene':gene,'SNP':snp,'Ref':ref,'Alt':alt,'CHR':chr,'TSS_POS':tss_pos,
                                                'SNP_POS':snp_pos,'t0':t0,'t24':t24,'delta':delta,'tss_distance':distance},ignore_index=True)
            print(final_df.head())
            final_df.to_csv(output_path + snp + '_' + model + '.csv', index=False)

'''
      gene        SNP Ref Alt    CHR   TSS_POS   SNP_POS        t0       t24  \
0    SSTR3  rs3218251   T   A  chr22  37210689  37149465  0.372962 -1.472712
1  C1QTNF6  rs3218251   T   A  chr22  37199423  37149465  0.023470 -1.858921
2   KCTD17  rs3218251   T   A  chr22  37051736  37149465 -0.783935 -1.978821
3  C1QTNF6  rs3218251   T   A  chr22  37199423  37149465  0.023470 -1.858921
4  TMPRSS6  rs3218251   T   A  chr22  37109713  37149465 -0.044932 -1.362033

      delta  tss_distance
0 -1.845673        -61224
1 -1.882391        -49958
2 -1.194886         97729
3 -1.882391        -49958
4 -1.317100         39752
'''