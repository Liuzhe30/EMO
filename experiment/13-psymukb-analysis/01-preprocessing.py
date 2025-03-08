import pandas as pd
from pyliftover import LiftOver

data1 = pd.read_csv('datasets/20190514_MasterFile-allDNMs_ncRNA_v1.5.txt',sep='\t')[['Chr','Start','End','Ref','Alt','PrimaryPhenotype','DisorderCategory','Gene.refGene']]
data2 = pd.read_csv('datasets/20190514_MasterFile_allDNMs-intergenic_v1.5.txt',sep='\t')[['Chr','Start','End','Ref','Alt','PrimaryPhenotype','DisorderCategory','Gene.refGene']]
data3 = pd.read_csv('datasets/20190514_MasterFile-allDNMs_ncRNA_v1.5.txt',sep='\t')[['Chr','Start','End','Ref','Alt','PrimaryPhenotype','DisorderCategory','Gene.refGene']]

data = pd.concat([data1,data2,data3],axis=0)

print(data.head())

'''
  Chr      Start        End Ref Alt PrimaryPhenotype      DisorderCategory  Gene.refGene
0   1  104112584  104112584   G   A     Autism (ASD)  psychiatric disorder       ACTG1P4
1   1   11839486   11839486   C   G     Autism (ASD)  psychiatric disorder  LOC102724659
2   1  224415486  224415486   T   C     Autism (ASD)  psychiatric disorder  LOC101927164
3   1  232173135  232173135   A   C     Autism (ASD)  psychiatric disorder   TSNAX-DISC1
4   1  145515623  145515623   G   C     Autism (ASD)  psychiatric disorder        GNRHR2
'''

data = data[data['Start'] == data['End']]
lo = LiftOver('hg19', 'hg38')

def convert_to_hg38(chrom, pos):
    chrom = f'chr{chrom}' 
    result = lo.convert_coordinate(chrom, pos)
    return result[0][1] if result else None 

data['hg38_Pos'] = data.apply(lambda row: convert_to_hg38(row['Chr'], row['Start']), axis=1)
data = data.dropna(subset=['hg38_Pos'])
data['hg38_Pos'] = data['hg38_Pos'].astype(int)
data = data.drop(columns=['Start', 'End']).rename(columns={'hg38_Pos': 'position'})
print(data.head())

'''
  Chr Ref Alt PrimaryPhenotype      DisorderCategory  position
0   1   G   A     Autism (ASD)  psychiatric disorder   103569962
1   1   C   G     Autism (ASD)  psychiatric disorder    11779429
2   1   T   C     Autism (ASD)  psychiatric disorder   224227784
3   1   A   C     Autism (ASD)  psychiatric disorder   232037389
4   1   G   C     Autism (ASD)  psychiatric disorder   145919464
'''

refFlat_data = pd.read_csv("datasets/ucsc_refseq.txt", sep="\t")
refFlat_data["TSS"] = refFlat_data.apply(
    lambda row: row["txStart"] if row["strand"] == "+" else row["txEnd"], axis=1
)

# Select the farthest TSS for each gene (minimum for positive chain, maximum for negative chain)
tss_mapping = refFlat_data.groupby("name2")["TSS"].agg(
    lambda x: min(x) if refFlat_data.loc[x.index[0], "strand"] == "+" else max(x)
).reset_index()

data = data.merge(
    tss_mapping, left_on="Gene.refGene", right_on="name2", how="left"
).rename(columns={"TSS": "TSS_position"}).drop(columns=["name2"])
data = data.dropna(subset=['TSS_position'])
data['TSS_position'] = data['TSS_position'].astype(int)
print(data.head())

'''
  Chr Ref Alt PrimaryPhenotype      DisorderCategory  Gene.refGene   position  TSS_position
0   1   G   A     Autism (ASD)  psychiatric disorder       ACTG1P4  103569962     103569403
2   1   T   C     Autism (ASD)  psychiatric disorder  LOC101927164  224227784     224219612
3   1   A   C     Autism (ASD)  psychiatric disorder   TSNAX-DISC1  232037389     231528652
4   1   G   C     Autism (ASD)  psychiatric disorder        GNRHR2  145919464     145919012
5   1   C   T     Autism (ASD)  psychiatric disorder     SSBP3-AS1   54236883      54236442

'''

# post-processing
data = data.drop_duplicates()
data['TSS_distance'] = data['position'] - data['TSS_position']
data = data[abs(data['TSS_distance']) <= 100000]
data = data[~data['Ref'].str.contains('-', regex=False)]
data = data[~data['Alt'].str.contains('-', regex=False)]
data = data[(data['Ref'].str.len() == 1) & (data['Alt'].str.len() == 1)]
data = data[~data['Chr'].isin(['X', 'Y'])]

data.to_csv('datasets/psymukb_case.csv',index=False)

grouped_data = data.groupby('PrimaryPhenotype')
for phenotype, df in grouped_data:
    if len(df) >= 10:  
        if len(df) > 1000:
            df = df.sample(n=1000, random_state=42)  
        filename = f"datasets/{phenotype.replace(' ', '_').replace('(', '').replace(')', '')}.csv"
        df.to_csv(filename, index=False)

