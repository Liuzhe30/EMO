import os
import pandas as pd
import numpy as np

cell_list = []
with open('../onek1k_cell_type.txt',encoding='utf8') as r:
    line = r.readline()
    while line:
        cell_list.append(line.strip().replace('/','_'))
        if(os.path.exists('E:/onek1k_eqtl_dataset/cell_type/' + line.strip().replace('/','_') + '.tsv')):
            os.remove('E:/onek1k_eqtl_dataset/cell_type/' + line.strip().replace('/','_')+ '.tsv')
        line = r.readline()
print(cell_list)

# read and split dataset
with open('E:/onek1k_eqtl_dataset/onek1k_eqtl_dataset.tsv',encoding='utf8') as r2:
    line = r2.readline()
    # process the first error line
    line_head = line.split('ROUNDbin')[0]
    line2 = line.split('ROUNDbin')[1]
    line_head = line_head + 'ROUND'
    line2 = 'bin' + line2
    print(line_head)
    with open('../onek1k_head.txt','w+') as wh:
        for item in line_head.split('\t'):
            wh.write(item + '\n')
    line = r2.readline()
    with open('E:/onek1k_eqtl_dataset/cell_type/' + line2.split('\t')[1].replace('/','_') + '.tsv', 'a+',encoding='utf8') as w2:
        w2.write(line2)
    while line:
        with open('E:/onek1k_eqtl_dataset/cell_type/' + line.split('\t')[1].replace('/','_') + '.tsv', 'a+',encoding='utf8') as w:
            w.write(line)
        line = r2.readline()
    




'''
# check cell_type
cell_list = []
with open('E:/onek1k_eqtl_dataset/onek1k_eqtl_dataset.tsv',encoding='utf8') as r:
    line = r.readline()
    while line:
        cell_type = line.split('\t')[1]
        if(cell_type not in cell_list and cell_type != 'CELL_TYPE'):
            print(cell_type)
            cell_list.append(cell_type)
        line = r.readline()
print(cell_list)

with open('../onek1k_cell_type.txt','w+',encoding='utf8') as w:
    for item in cell_list:
        w.write(item + '\n')
        '''