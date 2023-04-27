# mapping sequences
import pandas as pd
from pandas import read_parquet
pd.set_option('display.max_columns', None)
#pd.set_option('display.max_rows', None)

fasta_path = '../../datasets/chr_fasta_hg38/'
file_path = '../../datasets/1_ppc_0.9_single_mutation/'
            
with open("../0_gtex_list.txt") as r:
    lines = r.readlines()
    gtex_bulk_list = []
    for line in lines:
        gtex_bulk_list.append(line.strip())
    print(gtex_bulk_list)

# gtex_bulk_list = ['Adrenal_Gland'] # split for multithreading working
  
for bulk in gtex_bulk_list:
    for chr_no in range(1, 23):
        data = pd.read_pickle(file_path + bulk + '_' + str(chr_no) + '.pkl')
        #print(len(data))
        #print(data)
        data['variant_51_seq'] = 0
        data['tss_51_seq'] = 0
        data['seq_between_variant_tss'] = 0
        data['variant_51_seq_after_mutation'] = 0
        data['seq_between_variant_tss_after_mutation'] = 0
        for i in range(len(data)):        
            variant_id = data['variant_id'].values[i]
            chr_str = variant_id.split('_')[0]
            position = int(variant_id.split('_')[1])
            before_mutation = variant_id.split('_')[2]
            after_mutation = variant_id.split('_')[3]  
            tss_distance = int(data['tss_distance'].values[i])
            tss_position = position - tss_distance
            
            with open(fasta_path + chr_str + '_new.fasta') as fa:
                line = fa.readline()                
                variant_51_seq = line[position - 26:position + 25]  
                #print(variant_51_seq)
                tss_51_seq = line[tss_position - 26:tss_position + 25]
                #print(tss_51_seq)
                if(tss_distance > 0):
                    seq_between_variant_tss = line[tss_position - 1:position]
                else:
                    seq_between_variant_tss = line[position -1: tss_position]
                #print(seq_between_variant_tss)
                if(line[position - 1] >= 'a' and line[position - 1] <= 'z'):
                    variant_51_seq_after_mutation = line[position - 26: position - 1] + after_mutation.lower() + line[position: position + 25]
                    if(tss_distance > 0):
                        seq_between_variant_tss_after_mutation = seq_between_variant_tss[0:-2] + after_mutation.lower()
                    else:
                        seq_between_variant_tss_after_mutation = after_mutation.lower() + seq_between_variant_tss[1:-1]
                else:
                    variant_51_seq_after_mutation = line[position - 26: position - 1] + after_mutation + line[position: position + 25]
                    if(tss_distance > 0):
                        seq_between_variant_tss_after_mutation = seq_between_variant_tss[0:-2] + after_mutation
                    else:
                        seq_between_variant_tss_after_mutation = after_mutation + seq_between_variant_tss[1:-1]                    
                #print(variant_51_seq_after_mutation)
                
                data.loc[i, 'variant_51_seq'] = variant_51_seq
                data.loc[i, 'tss_51_seq'] = tss_51_seq
                data.loc[i, 'seq_between_variant_tss'] = seq_between_variant_tss
                data.loc[i, 'variant_51_seq_after_mutation'] = variant_51_seq_after_mutation
                data.loc[i, 'seq_between_variant_tss_after_mutation'] = seq_between_variant_tss_after_mutation
                #print(data)
        new_file_down = '../../dataset/2_ppc_0.9_seq_mapping/' + bulk + '_' + str(chr_no) + '.pkl'
        #print(data)
        data.to_pickle(new_file_down)


'''
# count
file_path_3 = 'dataset/2_ppc_0.9_seq_mapping/'

for bulk in gtex_bulk_list:
    for chr_no in range(1, 2):
        data = pd.read_pickle(file_path_3 + bulk + '_' + str(chr_no) + '.pkl')
        print(data)
'''
                
                
'''
def get_ATGC_position_51(chr_str, position):
    with open(fasta_path + chr_str + '.fa') as fa:
        line = fa.readline()
        z_no = int(float(position) / 50)
        x_no = int(position) % 50
        print(z_no)
        print(x_no)
        for i in range(0, z_no):
            line = fa.readline()        
        if(x_no == 0):
            #pos_fasta_upper = line.strip()[-1].upper()
            #print(pos_fasta_upper)
            pos_fasta = line.strip()[-1]
            print(pos_fasta)
            result_fasta_51 = line[-26:-2] + pos_fasta
            line = fa.readline()
            result_fasta_51 += line[0:25]
            print(result_fasta_51)
        elif(x_no >= 26):
            line = fa.readline()
            pos_fasta = line[x_no - 1]
            print(pos_fasta)
            result_fasta_51 = line[26 - x_no:-1]
            line = fa.readline()
            result_fasta_51 += line[0:x_no - 25]
            print(result_fasta_51)
        elif(x_no <= 25 and x_no > 0):
            result_fasta_51 = line[x_no - 26:-1]
            line = fa.readline()
            result_fasta_51 += line[0:x_no + 25]
            print(result_fasta_51)
    
    return result_fasta_51      
    '''  