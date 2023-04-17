import pandas as pd
from pandas import read_parquet
import pyBigWig
pd.set_option('display.max_columns', None)

atac_path = 'E:/SNP_eQTL_rawdata/epimap_ATAC_tissue/'

bw = pyBigWig.open(atac_path + "FINAL_ATAC-seq_BSS00047.sub_VS_Uniform_BKG_CONTROL_36_50000000.pval.signal.bedgraph.gz.bigWig")
max_atac_len = int(bw.chroms("chr1"))
print(max_atac_len)
atac_between = bw.values("chr1", 0, 3)
print(atac_between)