####################################### Pan-cancer multiomic data ##########################################
## https://data.humantumoratlas.org/explore?selectedFilters=%5B%7B%22group%22%3A%22AtlasName%22%2C%22value%22%3A%22HTAN+WUSTL%22%7D%5D&tab=file

#Atlas: HTAN WUSTL
#Level: Level 3
#Assay: scATAC-seq
#File: Formatgzip

# pip install synapseclient

# python download_data_from_HTAN.py

import synapseclient
import pandas as pd
import os

syn = synapseclient.Synapse()
syn.login('Luz','57888282')

ids = pd.read_csv('synapse_id.txt', header=None)
cnt = 0
for id in ids[0].values:
    meta = syn.get(entity=id, downloadFile=False)
    if meta.name.split('-')[-1] in ['barcodes.tsv.gz', 'features.tsv.gz', 'matrix.mtx.gz']:
        dir_name = '-'.join(meta.name.split('-')[:-1])
        if os.path.exists(dir_name):
            file = syn.get(entity=id, downloadLocation=dir_name)
        else:
            os.mkdir(dir_name)
            file = syn.get(entity=id, downloadLocation=dir_name)
        os.rename(dir_name+'/'+meta.name, dir_name+'/'+(meta.name.split('-')[-1]))
        cnt += 1
        print('file '+str(cnt)+' '+meta.name+' done')
    else:
        cnt += 1
        print('file '+str(cnt)+' '+meta.name+' skip')


## cell count stats for each sample
#for sample in `cat files.txt`; do
#    echo $sample `zcat $sample/barcodes.tsv.gz | wc -l` >> files_cell_cnt.txt
#done


## paper associated github: https://github.com/ding-lab/PanCan_snATAC_publication/tree/main

# python filter_cells.py
import scanpy as sc

dat = sc.read_10x_mtx('HT307C1-Th1K1', gex_only=False)




peaks_info = dat.var
peaks_info.reset_index(drop=True, inplace=True)
peaks_info = peaks_info[peaks_info['feature_types']=='Peaks']

(peaks_info['gene_ids'].map(lambda x: x.split(':')[0])).isin(['chr'+str(i) for i in range(1, 23)])






