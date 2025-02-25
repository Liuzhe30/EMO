import pandas as pd

compare_tissue_list = ['Adipose_Subcutaneous','Artery_Tibial','Breast_Mammary_Tissue','Colon_Transverse','Nerve_Tibial','Thyroid']

with open('cmd.txt','w+') as w:
    for tissue in compare_tissue_list:
        for model_size in ['small','middle']:
            for splittype in ['test']:
                w.write('python chromatin.py ' + model_size + '/' + splittype + '_' + model_size + '_' + tissue + '.vcf\n')
                w.write('python predict.py --coorFile ' + model_size + '/' + splittype + '_' + model_size + '_' + tissue + '.vcf --geneFile ' + model_size + '/' + splittype + '_' + model_size + '_' + tissue + '.closestgene --snpEffectFilePattern ./example/example.vcf.shift_SHIFT.diff.h5 --modelList ./resources/modellist --output expecto_results/' + splittype + '_' + model_size + '_' + tissue + '.csv\n')
            
