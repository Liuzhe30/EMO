# counting results
import pandas as pd
import numpy as np

pd.set_option('display.max_columns', None)
import warnings
warnings.filterwarnings('ignore')

output_path = 'data/pred_merged/'
snp_list = ['rs3806624', 'rs7731626', 'rs947474','rs3824660','rs968567','rs3218251','rs2234067','rs2233424']

model_dict = {"small":1_000,"middle":10_000,"large":100_000,"huge":1_000_000}

cutoff = 0.7
merged_df = pd.DataFrame()
for snp in snp_list:
    
    delta1,delta2,delta3,delta4=0,0,0,0
    for model in model_dict.keys():
        try:
            data = pd.read_csv(output_path + snp + '_' + model + '.csv')

            # |Δslope|>0.5,slope(t0)>=0
            delta1_ = len(data[(data['delta']>cutoff)&(data['t0']>0)].reset_index(drop=True))
            delta1 += len(data[(data['delta']>cutoff)&(data['t0']>0)].reset_index(drop=True))
            
            # |Δslope|<=0.5
            delta2_ = len(data[(data['delta']>=-cutoff)&(data['delta']<=cutoff)].reset_index(drop=True))
            delta2 += len(data[(data['delta']>=-cutoff)&(data['delta']<=cutoff)].reset_index(drop=True))
            
            # |Δslope|>0.5,slope(t0)<0 
            delta3_ = len(data[(data['delta']<-cutoff)&(data['t0']<0)].reset_index(drop=True))
            delta3 += len(data[(data['delta']<-cutoff)&(data['t0']<0)].reset_index(drop=True))
            
            # others
            delta4 += len(data) - delta1_ - delta2_ - delta3_

        except FileNotFoundError:
            continue
    
    merged_df = merged_df._append({'SNP':snp,'type':'Strong affected','count of TSS':delta1+delta3+delta4},ignore_index=True)
    merged_df = merged_df._append({'SNP':snp,'type':'Weak affected','count of TSS':delta2},ignore_index=True)

print(merged_df)
merged_df.to_csv(output_path + 'merged_all.csv')
