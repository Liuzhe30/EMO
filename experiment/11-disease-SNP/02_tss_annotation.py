import pandas as pd

gene_annotations = pd.read_csv("raw_data/hg38.refGene.gtf/hg38.refGene.gtf", sep="\t", comment="#", header=None, 
        names=["chrom", "source", "feature", "start", "end", "score", "strand", "frame", "attribute"])
print(gene_annotations.head())

# Filter for TSS locations (considering the start of genes)
tss_annotations = gene_annotations[gene_annotations['feature'] == 'transcript'].copy()

# Extract gene name and TSS
tss_annotations['gene_name'] = tss_annotations['attribute'].str.extract(r'gene_name "([^"]+)"')

# Calculate TSS based on strand information
tss_annotations['tss'] = tss_annotations.apply(lambda row: row['start'] if row['strand'] == '+' else row['end'], axis=1)

# Simplify the data to necessary columns: chrom, tss, gene_name
tss_annotations = tss_annotations[['chrom', 'tss', 'gene_name']]

# Display the first few rows of the TSS annotation data
tss_annotations.drop_duplicates(inplace=True)
tss_annotations = tss_annotations.reset_index(drop=True)
print(tss_annotations.head())

tss_annotations.to_csv("data/tss_annotations.csv",index=False)

'''
   chrom    tss  gene_name
0   chr1  11874    DDX11L1
4   chr1  29370     WASH7P
16  chr1  17436  MIR6859-1
18  chr1  17436  MIR6859-2
20  chr1  17436  MIR6859-3
'''