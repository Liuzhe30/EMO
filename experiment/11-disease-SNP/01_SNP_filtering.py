import requests
import pandas as pd
import json
from tqdm import tqdm

import warnings
warnings.filterwarnings('ignore')

# step 1: fetch rsids
snp_list = []
data = pd.read_csv('raw_data/RAriskSNPs_20131120.csv')
for i in range(len(data)):
    if(':' not in data['rsid'][i]):
        snp_list.append(data['rsid'][i])

print(snp_list)

# step 2: mapping SNP positions
snp_positions = {}
for snp in tqdm(snp_list):
    url = f'https://rest.ensembl.org/variation/human/{snp}?content-type=application/json'
    response = requests.get(url,verify=False)
    if response.status_code == 200:
        data = response.json()
        if 'mappings' in data:
            for mapping in data['mappings']:
                if mapping['assembly_name'] == 'GRCh38':
                    snp_positions[snp] = [mapping['seq_region_name'], mapping['start']]
                    break

with open('data/SNP_positions.json','w') as f:
    json.dump(snp_positions,f,indent=4)

with open('data/SNP_positions.json','r') as f:
    snp_positions = json.load(f)

# step 3: fetch regions
data = pd.read_csv('raw_data/GSM4118993_T0_ATAC_head.narrowPeak',sep='\t')
#print(data.head())
regions = []
for i in range(len(data)):
    regions.append((data['CHR'][i].split('chr')[-1],int(data['start'][i])-1000,int(data['end'][i])+1000))
#print(regions)

snp_in_region = []

for snp, (chrom, pos) in snp_positions.items():
    for region in regions:
        region_chrom, start, end = region
        if chrom == region_chrom and start <= pos <= end:
            snp_in_region.append((snp, chrom, pos, region))
            break

print("SNPs in Regions:")
for snp_info in snp_in_region:
    print(snp_info)

'''
('rs2476601', '1', 113834946, ('1', 113834897, 113837129))
('rs3806624', '3', 27723132, ('3', 27720631, 27723397))   
('rs7731626', '5', 56148856, ('5', 56147311, 56149821))   
('rs2234067', '6', 36387877, ('6', 36386351, 36388498))   
('rs2233424', '6', 44266184, ('6', 44263999, 44266839))
('rs947474', '10', 6348488, ('10', 6346729, 6349669))
('rs3824660', '10', 8062759, ('10', 8060482, 8062884))
('rs968567', '11', 61828092, ('11', 61826701, 61829325))
('rs3218251', '22', 37149465, ('22', 37147423, 37149644))
'''