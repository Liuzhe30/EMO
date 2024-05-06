import numpy as np
import pandas as pd
import math
pd.set_option('display.max_columns', None)

test_path = '../../datasets_new/'
output_up_path = '../../datasets_new/non_causal/up_causal/'
output_down_path = '../../datasets_new/non_causal/down_causal/'

# up-regulation testset
# small model
test_all = pd.read_pickle(test_path + 'small/test_small.pkl')
test_all['slope'] = test_all['slope'].astype(float)
test_df = test_all[test_all['slope']>-0.5]
test_df = test_df.reset_index(drop=True)
test_df.to_pickle(output_up_path + 'small/test_small.pkl')
test_df = test_all[test_all['slope']<-0.5]
test_df = test_df.reset_index(drop=True)
test_df.to_pickle(output_down_path + 'small/test_small.pkl')

# middle model
test_all = pd.read_pickle(test_path + 'middle/test_middle.pkl')
test_all['slope'] = test_all['slope'].astype(float)
test_df = test_all[test_all['slope']>-0.5]
test_df = test_df.reset_index(drop=True)
test_df.to_pickle(output_up_path + 'middle/test_middle.pkl')
test_df = test_all[test_all['slope']<-0.5]
test_df = test_df.reset_index(drop=True)
test_df.to_pickle(output_down_path + 'middle/test_middle.pkl')

# large model
test_all = pd.read_pickle(test_path + 'large/test_large.pkl')
test_all['slope'] = test_all['slope'].astype(float)
test_df = test_all[test_all['slope']>-0.5]
test_df = test_df.reset_index(drop=True)
test_df.to_pickle(output_up_path + 'large/test_large.pkl')
test_df = test_all[test_all['slope']<-0.5]
test_df = test_df.reset_index(drop=True)
test_df.to_pickle(output_down_path + 'large/test_large.pkl')

# huge model
test_all = pd.read_pickle(test_path + 'huge/test_huge.pkl')
test_all['slope'] = test_all['slope'].astype(float)
test_df = test_all[test_all['slope']>-0.5]
test_df = test_df.reset_index(drop=True)
test_df.to_pickle(output_up_path + 'huge/test_huge.pkl')
test_df = test_all[test_all['slope']<-0.5]
test_df = test_df.reset_index(drop=True)
test_df.to_pickle(output_down_path + 'huge/test_huge.pkl')