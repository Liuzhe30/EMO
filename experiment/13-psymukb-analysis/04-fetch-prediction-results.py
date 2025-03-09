import pandas as pd
pd.set_option('display.max_columns', None)

disease_list = ['Autism_ASD','Developmental_Delay_DD','Fetal_non-Preterm_birth_non-PTB','Fetal_preterm_birth_PTB','Mix_Autism_or_Schizophrenia','Sibling_Control']
# 1000,24,863,497,198,1000
tissue_list = ['hippocampus','cerebellum','prefrontal_cortex','temporal_lobe','substantia_nigra']

slope_columns = [tissue + '_slope' for tissue in tissue_list]
sign_columns = [tissue + '_sign' for tissue in tissue_list]

for disease in disease_list:
    file_path = f'datasets/{disease}_pred.pkl'
    output_path = f'datasets/{disease}_pred_final.csv'

    data = pd.read_pickle(file_path)[['Chr', 'Ref', 'Alt', 'position'] + slope_columns + sign_columns]
    data.to_csv(output_path, index=False)

'''
Chr	Ref	Alt	position	hippocampus_slope	cerebellum_slope	prefrontal_cortex_slope	temporal_lobe_slope	substantia_nigra_slope
9	T	C	92819845	0.09405224	0.040140826	0.109988414	0.099657565	0.067360893
10	T	A	78996445	0.129093304	0.151506886	0.163972959	0.084876031	0.127745315
10	A	G	120824851	0.186338976	0.14761126	0.251977026	0.338231802	0.232195571
9	A	G	2614615	0.87506336	0.954618394	0.900415778	0.858251691	1.099137068
14	G	T	97970375	0.473214	0.479877979	0.484405607	0.482635468	0.486697853
'''