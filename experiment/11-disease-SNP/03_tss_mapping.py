import pandas as pd

tss_df = pd.read_csv('data/tss_annotations.csv')
snp_df = pd.read_csv('data/filtered_SNP_hg38.csv')

for i in range(len(snp_df)):
    snp_small = pd.DataFrame()
    snp_middle = pd.DataFrame()
    snp_large = pd.DataFrame()
    snp_huge = pd.DataFrame()

    rsid = snp_df['RSID'][i]
    chr = str(snp_df['CHR'][i])
    pos = int(snp_df['POS'][i])
    ref = snp_df['Ref'][i]
    alt = snp_df['Alt'][i]

    tss_chr_df = tss_df[tss_df['chrom'] == "chr" + chr].reset_index(drop=True)
    for j in range(len(tss_chr_df)):
        tss_pos = tss_chr_df['tss'][j]
        gene_name = tss_chr_df['gene_name'][j]
        distance = pos - tss_pos
        if(pos-1_000 < tss_pos < pos+1_000):
            snp_small = snp_small._append({'TSS_POS':tss_pos, 'gene':gene_name,'variant_id':rsid,'slope':0,'Ref':ref,'Alt':alt,'CHR':'chr'+chr,
                                            'SNP_POS':pos,'label':0,'tss_distance':distance},ignore_index=True)
        elif(pos-10_000 < tss_pos < pos-1_000 or pos+1_000 < tss_pos < pos+10_000):
            snp_middle = snp_middle._append({'TSS_POS':tss_pos, 'gene':gene_name,'variant_id':rsid,'slope':0,'Ref':ref,'Alt':alt,'CHR':'chr'+chr,
                                            'SNP_POS':pos,'label':0,'tss_distance':distance},ignore_index=True)         
        elif(pos-100_000 < tss_pos < pos-10_000 or pos+10_000 < tss_pos < pos+100_000):
            snp_large = snp_large._append({'TSS_POS':tss_pos, 'gene':gene_name,'variant_id':rsid,'slope':0,'Ref':ref,'Alt':alt,'CHR':'chr'+chr,
                                            'SNP_POS':pos,'label':0,'tss_distance':distance},ignore_index=True)                           
        elif(pos-1_000_000 < tss_pos < pos-100_000 or pos+100_000 < tss_pos < pos+1_000_000):
            snp_huge = snp_huge._append({'TSS_POS':tss_pos, 'gene':gene_name,'variant_id':rsid,'slope':0,'Ref':ref,'Alt':alt,'CHR':'chr'+chr,
                                            'SNP_POS':pos,'label':0,'tss_distance':distance},ignore_index=True)

    print(snp_small.head())
    print(snp_middle.head())
    print(snp_large.head())
    print(snp_huge.head())
    print()
    snp_small.to_pickle('data/snp_tss/' + rsid + '_small.pkl')
    snp_middle.to_pickle('data/snp_tss/' + rsid + '_middle.pkl')
    snp_large.to_pickle('data/snp_tss/' + rsid + '_large.pkl')
    snp_huge.to_pickle('data/snp_tss/' + rsid + '_huge.pkl')

'''
    TSS_POS   gene variant_id  slope Ref Alt    CHR   SNP_POS  label  tss_distance
0  37149916  IL2RB  rs3218251      0   T   A  chr22  37149465      0           451
Empty DataFrame
Columns: []
Index: []
    TSS_POS     gene variant_id  slope Ref Alt    CHR   SNP_POS  label  tss_distance
0  37210689    SSTR3  rs3218251      0   T   A  chr22  37149465      0         61224
1  37199423  C1QTNF6  rs3218251      0   T   A  chr22  37149465      0         49958
2  37051736   KCTD17  rs3218251      0   T   A  chr22  37149465      0        -97729
3  37188247  C1QTNF6  rs3218251      0   T   A  chr22  37149465      0         38782
4  37109713  TMPRSS6  rs3218251      0   T   A  chr22  37149465      0        -39752
    TSS_POS      gene variant_id  slope Ref Alt    CHR   SNP_POS  label  tss_distance
0  37953601  C22orf23  rs3218251      0   T   A  chr22  37149465      0        804136
1  38110684  BAIAP2L2  rs3218251      0   T   A  chr22  37149465      0        961219
2  36481640      TXN2  rs3218251      0   T   A  chr22  37149465      0       -667825
3  37953699    POLR2F  rs3218251      0   T   A  chr22  37149465      0        804234
4  37007851     TEX33  rs3218251      0   T   A  chr22  37149465      0       -141614
'''