#### pan-cancer dataset
## python download_rds.py
import synapseclient
import pandas as pd
import os

syn = synapseclient.Synapse()
syn.login('Luz','jiluzhang_57888282')

ids = pd.read_table('rds_synapse_id.txt', header=None)
for id in ids[1].values:
    meta = syn.get(entity=id, downloadFile=True, downloadLocation='.')
    print(id, 'done')

#### generate info files (barcode & cell_anno)
# processbar: https://blog.csdn.net/ZaoJewin/article/details/133894474
# library(Seurat)
# library(Signac)

samples <- read.table('rds_samples_id.txt')  # remove normal samples (CPT704DU-M1  GBML018G1-M1N2  GBML019G1-M1N1
                                             #                        HT291C1-M1A3 SN001H1-Ms1  SP369H1-Mc1  SP819H1-Mc1)
pb <- txtProgressBar(min=0, max=122, style=3)

for (i in 1:nrow(samples)){
    s_1 <- samples[i, 1]
    rds <- readRDS(paste0(s_1, '.rds'))
    out <- rds@meta.data[c('Original_barcode', 'cell_type')]
    out <- out[(out$cell_type %in% c('Endothelial', 'B-cells', 'Plasma',
                                     'Fibroblasts', 'T-cells', 'Macrophages')), ]

    s_2 <- samples[i, 2]
    dir.create(s_2)
    write.table(out[c('Original_barcode', 'cell_type')], paste0(s_2, '/', s_2, '_info.txt'), row.names=FALSE, col.names=FALSE, quote=FALSE, sep='\t')
    setTxtProgressBar(pb, i) 
}
close(pb)


########## plot stacked bars ##########
from plotnine import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

samples = pd.read_table('rds_samples_cancer_type.txt', header=None) 
stat_df = pd.DataFrame({'Cancer_type':['CEAD']*5+['CESC']*5+['CRC']*5+['UCEC']*5+['PDAC']*5+['BRCA']*5+['HNSCC']*5+['SKCM']*5+['OV']*5,
                        'Cell_type': ['Endothelial', 'B-cells', 'Fibroblasts', 'T-cells', 'Macrophages']*9,
                        'Cell_count':0})

for i in range(samples.shape[0]):
    s = samples[0][i]
    dat = pd.read_table(s+'/'+s+'_info.txt', header=None)
    cancer_type = samples[1][i]
    stat_dict = dict(dat[1].value_counts())
    for cell_type in stat_dict.keys():  # BRCA & BRCA_Basal  B-cells & Plasma
        if cancer_type=='BRCA_Basal':
            if cell_type=='Plasma':
                stat_df.loc[(stat_df['Cancer_type']=='BRCA')&(stat_df['Cell_type']=='B-cells'), 'Cell_count'] += stat_dict[cell_type]
            else:
                stat_df.loc[(stat_df['Cancer_type']=='BRCA')&(stat_df['Cell_type']==cell_type), 'Cell_count'] += stat_dict[cell_type]
        else:
            if cell_type=='Plasma':
                stat_df.loc[(stat_df['Cancer_type']==cancer_type)&(stat_df['Cell_type']=='B-cells'), 'Cell_count'] += stat_dict[cell_type]
            else:
                stat_df.loc[(stat_df['Cancer_type']==cancer_type)&(stat_df['Cell_type']==cell_type), 'Cell_count'] += stat_dict[cell_type]

cancer_type_cnt = stat_df.groupby('Cancer_type', as_index=False).agg('sum')[['Cancer_type', 'Cell_count']]
#   Cancer_type  Cell_count
# 0        BRCA       19729
# 1        CEAD         401
# 2        CESC        4660
# 3         CRC       28641
# 4       HNSCC       20124
# 5          OV        6478
# 6        PDAC       36575
# 7        SKCM       26561
# 8        UCEC        1240

plt.rcParams['pdf.fonttype'] = 42
p = ggplot(cancer_type_cnt, aes(x='Cancer_type', y='Cell_count', fill='Cancer_type')) + geom_bar(stat='identity', position='stack') + \
                                                                            xlab('Cancer type') + ylab('Cell count') + scale_fill_brewer(palette="Paired", type='qual', direction=-1) +\
                                                                            scale_y_continuous(limits=[0, 40000], breaks=np.arange(0, 40000+1, 10000)) + theme_bw()
p.save(filename='cancer_type_cell_count.pdf', dpi=600, height=4, width=5)


stat_df['Cell_ptg'] = stat_df['Cell_count'] / stat_df.groupby('Cancer_type')['Cell_count'].transform('sum')  # calculate ptg for each cancer type
cell_type_lst = ['Endothelial', 'B-cells', 'Fibroblasts', 'T-cells', 'Macrophages']
stat_df['Cell_type'] = pd.Categorical(stat_df['Cell_type'], categories=cell_type_lst)

# color map: https://www.jianshu.com/p/b88a77a609da
plt.rcParams['pdf.fonttype'] = 42
p = ggplot(stat_df, aes(x='Cancer_type', y='Cell_ptg', fill='Cell_type')) + geom_bar(stat='identity', position='stack') + coord_flip() + \
                                                                            xlab('Cancer type') + ylab('Proportion') + scale_fill_brewer(palette="Set2", type='qual', direction=-1) +\
                                                                            scale_y_continuous(limits=[0, 1.01], breaks=np.arange(0, 1.1, 0.2)) + theme_bw()
p.save(filename='cancer_type_cell_type_ptg.pdf', dpi=600, height=4, width=5)


## python filter_cells_with_cell_anno.py  # 30 min
# something wrong in rds files
# VF027V1-S1_1N1   VF027V1-S1
# VF027V1-S1Y1     VF027V1-S2
# ->
# VF027V1-S1_1N1   VF027V1-S2
# VF027V1-S1Y1     VF027V1-S1
# i=122 & i=123

import scanpy as sc
import pandas as pd
from tqdm import tqdm

samples = pd.read_table('rds_samples_id.txt', header=None)
for i in tqdm(range(samples.shape[0]), ncols=80):
    s_2= samples[1][i]
    info = pd.read_table(s_2+'/'+s_2+'_info.txt', header=None)
    info.columns = ['barcode', 'cell_anno']
    dat = sc.read_10x_mtx('/fs/home/jiluzhang/2023_nature_LD/'+s_2, gex_only=False)
    out = dat[info['barcode'], :].copy()
    out.obs['cell_anno'] = info['cell_anno'].values
    out.write(s_2+'/'+s_2+'_filtered.h5ad')


## python rna_atac_align.py
import scanpy as sc
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
import pybedtools
from tqdm import tqdm

samples = pd.read_table('rds_samples_id.txt', header=None)
for s in tqdm(samples[1], ncols=80):
    dat = sc.read_h5ad(s+'/'+s+'_filtered.h5ad')
    
    ########## rna ##########
    rna = dat[:, dat.var['feature_types']=='Gene Expression'].copy()
    rna.X = rna.X.toarray()
    genes = pd.read_table('human_genes.txt', names=['gene_id', 'gene_name'])
    
    rna_genes = rna.var.copy()
    rna_genes['gene_id'] = rna_genes['gene_ids'].map(lambda x: x.split('.')[0])  # delete ensembl id with version number
    rna_exp = pd.concat([rna_genes, pd.DataFrame(rna.X.T, index=rna_genes.index)], axis=1)
    
    X_new = pd.merge(genes, rna_exp, how='left', on='gene_id').iloc[:, 4:].T
    X_new.fillna(value=0, inplace=True)
    
    rna_new = sc.AnnData(X_new.values, obs=rna.obs, var=pd.DataFrame({'gene_ids': genes['gene_id'], 'feature_types': 'Gene Expression'}))
    rna_new.var.index = genes['gene_name'].values
    rna_new.X = csr_matrix(rna_new.X)
    rna_new.write(s+'/'+s+'_rna.h5ad')
    
    ########## atac ##########
    atac = dat[:, dat.var['feature_types']=='Peaks'].copy()
    atac.X = atac.X.toarray()
    atac.X[atac.X>0] = 1  # binarization
    
    peaks = pd.DataFrame({'id': atac.var_names})
    peaks['chr'] = peaks['id'].map(lambda x: x.split(':')[0])
    peaks['start'] = peaks['id'].map(lambda x: x.split(':')[1].split('-')[0])
    peaks['end'] = peaks['id'].map(lambda x: x.split(':')[1].split('-')[1])
    peaks.drop(columns='id', inplace=True)
    peaks['idx'] = range(peaks.shape[0])
    
    cCREs = pd.read_table('human_cCREs.bed', names=['chr', 'start', 'end'])
    cCREs['idx'] = range(cCREs.shape[0])
    cCREs_bed = pybedtools.BedTool.from_dataframe(cCREs)
    peaks_bed = pybedtools.BedTool.from_dataframe(peaks)
    idx_map = peaks_bed.intersect(cCREs_bed, wa=True, wb=True).to_dataframe().iloc[:, [3, 7]]
    idx_map.columns = ['peaks_idx', 'cCREs_idx']
    
    m = np.zeros([atac.n_obs, cCREs_bed.to_dataframe().shape[0]], dtype='float32')
    for i in tqdm(range(atac.X.shape[0]), ncols=80, desc='Aligning ATAC peaks'):
        m[i][idx_map[idx_map['peaks_idx'].isin(peaks.iloc[np.nonzero(atac.X[i])]['idx'])]['cCREs_idx']] = 1
    
    atac_new = sc.AnnData(m, obs=atac.obs, var=pd.DataFrame({'gene_ids': cCREs['chr']+':'+cCREs['start'].map(str)+'-'+cCREs['end'].map(str), 'feature_types': 'Peaks'}))
    atac_new.var.index = atac_new.var['gene_ids'].values
    atac_new.X = csr_matrix(atac_new.X)
    atac_out.write(s+'/'+s+'_atac.h5ad')

    # for chrom in tqdm(['chr'+str(i) for i in range(1, 23)], ncols=80, desc='Writing chrom atac h5ad'):
    #     atac_out = atac_new[:, atac_new.var.index.map(lambda x: x.split(':')[0]==chrom)].copy()
    #     atac_out.X = csr_matrix(atac_out.X)
    #     atac_out.write(s+'/'+s+'_atac_'+chrom+'.h5ad')


#### merge same h5ad of cancer type
import pandas as pd
import scanpy as sc
import anndata as ad

samples = pd.read_table('rds_samples_cancer_type.txt', header=None) 
cancer_type_lst = list(samples[1].drop_duplicates().values)
for cancer_type in cancer_type_lst:
    s_lst = list(samples[samples[1]==cancer_type][0])
    for i in range(len(s_lst)):
        s = s_lst[i]
        if i==0:
            rna_out = sc.read_h5ad(s+'/'+s+'_rna.h5ad')
            rna_out.obs.index = s+'_'+rna_out.obs.index
            atac_out = sc.read_h5ad(s+'/'+s+'_atac.h5ad')
            atac_out.obs.index = s+'_'+atac_out.obs.index
        else:
            rna = sc.read_h5ad(s+'/'+s+'_rna.h5ad')
            rna.obs.index = s+'_'+rna.obs.index
            rna_out = ad.concat([rna_out, rna])
            atac = sc.read_h5ad(s+'/'+s+'_atac.h5ad')
            atac.obs.index = s+'_'+atac.obs.index
            atac_out = ad.concat([atac_out, atac])
        
        print(cancer_type, s)

    rna_out.var = rna.var
    atac_out.var = atac.var
    rna_out.write(cancer_type+'_rna_raw.h5ad')
    atac_out.write(cancer_type+'_atac_raw.h5ad')


#### merge & filter raw h5ad
import scanpy as sc
import anndata as ad
import pandas as pd
from tqdm import tqdm

samples = pd.read_table('rds_samples_cancer_type.txt', header=None) 
cancer_type_lst = list(samples[1].drop_duplicates().values)

## rna
for cancer_type in tqdm(cancer_type_lst, ncols=80, desc='Merge rna h5ad'):
    if cancer_type==cancer_type_lst[0]:
        rna = sc.read_h5ad(cancer_type+'_rna_raw.h5ad')
        rna.obs['cancer_type'] = cancer_type
    else:
        rna_tmp = sc.read_h5ad(cancer_type+'_rna_raw.h5ad')
        rna_tmp.obs['cancer_type'] = cancer_type
        rna = ad.concat([rna, rna_tmp])

rna.var = rna_tmp.var                         # 144409 × 38244
rna.write('all_cancer_type_rna_raw.h5ad')

sc.pp.filter_genes(rna, min_cells=100)        # 144409 × 19160
sc.pp.filter_cells(rna, min_genes=200)        # 144409 × 19160
sc.pp.filter_cells(rna, max_genes=20000)      # 144409 × 19160
rna.write('rna.h5ad')

## atac
for cancer_type in tqdm(cancer_type_lst, ncols=80, desc='Merge atac h5ad'):
    if cancer_type==cancer_type_lst[0]:
        atac = sc.read_h5ad(cancer_type+'_atac_raw.h5ad')
    else:
        atac_tmp = sc.read_h5ad(cancer_type+'_atac_raw.h5ad')
        atac = ad.concat([atac, atac_tmp])

atac.var = atac_tmp.var                       # 144409 × 1033239
atac.obs['cancer_type'] = rna.obs['cancer_type']
atac.write('all_cancer_type_atac_raw.h5ad')

sc.pp.filter_genes(atac, min_cells=100)       # 144409 × 389267
sc.pp.filter_cells(atac, min_genes=500)       # 144409 × 389267
sc.pp.filter_cells(atac, max_genes=50000)     # 144409 × 389267
atac.write('atac.h5ad')


#### split dataset (ov for testing)
import scanpy as sc
import random

rna = sc.read_h5ad('rna.h5ad')
rna_test = rna[rna.obs['cancer_type']=='OV', :].copy()        # 6478 × 19160
rna_train_val = rna[rna.obs['cancer_type']!='OV', :].copy()   # 137931 × 19160

random.seed(0)
idx = list(range(rna_train_val.n_obs))
random.shuffle(idx)
train_idx = idx[:int(len(idx)*0.9)]
val_idx = idx[int(len(idx)*0.9):]

rna[train_idx, :].write('rna_train.h5ad')
rna[val_idx, :].write('rna_val.h5ad')
rna_test.write('rna_test.h5ad')

atac = sc.read_h5ad('atac.h5ad')
atac[train_idx, :].write('atac_train.h5ad')
atac[val_idx, :].write('atac_val.h5ad')
atac[rna_test.obs.index, :].write('atac_test.h5ad')

## max gene count
import scanpy as sc
sc.read_h5ad('rna_test.h5ad').obs.n_genes.max()  # 8779


#### cisformer
python data_preprocess.py -r rna_train.h5ad -a atac_train.h5ad -s train_pt --dt train -n train --config rna2atac_config_train.yaml   # ~2 h
python data_preprocess.py -r rna_val.h5ad -a atac_val.h5ad -s val_pt --dt val -n val --config rna2atac_config_val.yaml 
python data_preprocess.py -r rna_test.h5ad -a atac_test.h5ad -s test_pt --dt test -n test --config rna2atac_config_test.yaml  # 30 min
nohup accelerate launch --config_file accelerator_config_train.yaml --main_process_port 29823 rna2atac_train.py --config_file rna2atac_config_train.yaml \
                        --train_data_dir train_pt --val_data_dir val_pt -s save -n rna2atac_pan_cancer > rna2atac_train_20241025.log &   # gamma_step:1  gamma:0.5
accelerate launch --config_file accelerator_config_test.yaml --main_process_port 29822 rna2atac_predict.py -d test_pt \
                  -l save/2024-10-25_rna2atac_pan_cancer_5/pytorch_model.bin --config_file rna2atac_config_test.yaml  # 30 min

python npy2h5ad.py
python cal_cluster.py --file atac_cisformer_umap.h5ad
python plot_save_umap.py --pred atac_cisformer.h5ad --true atac_test.h5ad  # 4 min

python cal_auroc_auprc.py --pred atac_cisformer.h5ad --true atac_test.h5ad
# Cell-wise AUROC: 0.8860
# Cell-wise AUPRC: 0.1629
# Peak-wise AUROC: 0.6459
# Peak-wise AUPRC: 0.0405
python cal_cluster.py --file atac_cisformer_umap.h5ad
# AMI: 0.5631
# ARI: 0.3901
# HOM: 0.7717
# NMI: 0.564


#### plot umap for true atac
import snapatac2 as snap
import scanpy as sc

true = snap.read('atac_test.h5ad', backed=None)  # 6478 × 389267
# Plasma         1839
# Fibroblasts    1706
# Macrophages    1500
# T-cells        1042
# B-cells         232
# Endothelial     159
snap.pp.select_features(true)
snap.tl.spectral(true)
snap.tl.umap(true)

true.obs['cell_anno'].replace({'Plasma': 'B-cells'}, inplace=True)
sc.pl.umap(true, color='cell_anno', legend_fontsize='7', legend_loc='right margin', size=5,
           title='', frameon=True, save='_atac_true.pdf')

true.obs['batch'] = true.obs.index.map(lambda x: x.split('_')[0])
# VF027V1-S1    2379
# VF034V1-T1    2076
# VF032V1-S1     754
# VF044V1-S1     533
# VF050V1-S2     392
# VF035V1-S1     205
# VF026V1-S1      70
# VF027V1-S2      69
sc.pl.umap(true, color='batch', legend_fontsize='7', legend_loc='right margin', size=5,
           title='', frameon=True, save='_atac_batch_true.pdf')




#### scbutterfly
## python scbt_b.py
## nohup python scbt_b.py > scbt_b_20241030.log &  # 167474  (large memory requirement)
# import os
# from scButterfly.butterfly import Butterfly
# from scButterfly.split_datasets import *
# import scanpy as sc
# import anndata as ad
# import random
# import numpy as np
# import gc

# os.environ["CUDA_VISIBLE_DEVICES"] = "5"

# RNA_data = sc.read_h5ad('rna.h5ad')
# ATAC_data = sc.read_h5ad('atac.h5ad')

# RNA_data_train = sc.read_h5ad('rna_train.h5ad')
# RNA_data_val = sc.read_h5ad('rna_val.h5ad')
# RNA_data_test = sc.read_h5ad('rna_test.h5ad')

# train_id = list(np.argwhere(RNA_data.obs.index.isin(sc.read_h5ad('rna_train.h5ad').obs.index)).flatten())
# validation_id = list(np.argwhere(RNA_data.obs.index.isin(sc.read_h5ad('rna_val.h5ad').obs.index)).flatten())
# test_id = list(np.argwhere(RNA_data.obs.index.isin(sc.read_h5ad('rna_test.h5ad').obs.index)).flatten())

# butterfly = Butterfly()
# butterfly.load_data(RNA_data, ATAC_data, train_id, test_id, validation_id)

# del RNA_data, ATAC_data, train_id, validation_id, test_id
# gc.collect()

# butterfly.data_preprocessing(normalize_total=False, log1p=False, use_hvg=False, n_top_genes=None, binary_data=False, 
#                              filter_features=False, fpeaks=None, tfidf=False, normalize=False) 

# butterfly.ATAC_data_p.var['chrom'] = butterfly.ATAC_data_p.var['gene_ids'].map(lambda x: x.split(':')[0])
# chrom_list = []
# last_one = ''
# for i in range(len(butterfly.ATAC_data_p.var.chrom)):
#     temp = butterfly.ATAC_data_p.var.chrom[i]
#     if temp[0 : 3] == 'chr':
#         if not temp == last_one:
#             chrom_list.append(1)
#             last_one = temp
#         else:
#             chrom_list[-1] += 1
#     else:
#         chrom_list[-1] += 1

# butterfly.augmentation(aug_type=None)
# butterfly.construct_model(chrom_list=chrom_list)
# butterfly.train_model(batch_size=16)
# A2R_predict, R2A_predict = butterfly.test_model(test_cluster=False, test_figure=False, output_data=True)


## nohup python scbt_b_0.py > scbt_b_0_20241031.log &  # 748469  7h
import os
from scButterfly.butterfly import Butterfly
from scButterfly.split_datasets import *
import scanpy as sc
import anndata as ad
import random
import numpy as np
import gc

os.environ["CUDA_VISIBLE_DEVICES"] = "5"

RNA_data_train_val = sc.read_h5ad('rna_train_val_0.h5ad')
ATAC_data_train_val = sc.read_h5ad('atac_train_val_0.h5ad')
RNA_data_test = sc.read_h5ad('rna_test.h5ad')
ATAC_data_test = sc.read_h5ad('atac_test.h5ad')

RNA_data = ad.concat([RNA_data_train_val, RNA_data_test])
RNA_data.var = RNA_data_train_val.var
ATAC_data = ad.concat([ATAC_data_train_val, ATAC_data_test])
ATAC_data.var = ATAC_data_train_val.var

RNA_data_val = sc.read_h5ad('rna_val.h5ad')

train_id = list(np.argwhere(~RNA_data.obs.index.isin(RNA_data_val.obs.index.append(RNA_data_test.obs.index))).flatten())
validation_id = list(np.argwhere(RNA_data.obs.index.isin(RNA_data_val.obs.index)).flatten())
test_id = list(np.argwhere(RNA_data.obs.index.isin(RNA_data_test.obs.index)).flatten())

butterfly = Butterfly()
butterfly.load_data(RNA_data, ATAC_data, train_id, test_id, validation_id)

del RNA_data_train_val, ATAC_data_train_val, RNA_data_test, ATAC_data_test, RNA_data, ATAC_data, RNA_data_val, train_id, validation_id, test_id
gc.collect()

butterfly.data_preprocessing(normalize_total=False, log1p=False, use_hvg=False, n_top_genes=None, binary_data=False, 
                             filter_features=False, fpeaks=None, tfidf=False, normalize=False) 

butterfly.ATAC_data_p.var['chrom'] = butterfly.ATAC_data_p.var['gene_ids'].map(lambda x: x.split(':')[0])
chrom_list = []
last_one = ''
for i in range(len(butterfly.ATAC_data_p.var.chrom)):
    temp = butterfly.ATAC_data_p.var.chrom[i]
    if temp[0 : 3] == 'chr':
        if not temp == last_one:
            chrom_list.append(1)
            last_one = temp
        else:
            chrom_list[-1] += 1
    else:
        chrom_list[-1] += 1

butterfly.augmentation(aug_type=None)
butterfly.construct_model(chrom_list=chrom_list)
butterfly.train_model(batch_size=16)
A2R_predict, R2A_predict = butterfly.test_model(test_cluster=False, test_figure=False, output_data=True)


## nohup python scbt_b_1.py > scbt_b_1_20241101.log &  # 1121654
import os
from scButterfly.butterfly import Butterfly
from scButterfly.split_datasets import *
import scanpy as sc
import anndata as ad
import random
import numpy as np
import gc

os.environ["CUDA_VISIBLE_DEVICES"] = "5"

RNA_data_train_val = sc.read_h5ad('rna_train_val_1.h5ad')
ATAC_data_train_val = sc.read_h5ad('atac_train_val_1.h5ad')
RNA_data_test = sc.read_h5ad('rna_test.h5ad')
ATAC_data_test = sc.read_h5ad('atac_test.h5ad')

RNA_data = ad.concat([RNA_data_train_val, RNA_data_test])
RNA_data.var = RNA_data_train_val.var
ATAC_data = ad.concat([ATAC_data_train_val, ATAC_data_test])
ATAC_data.var = ATAC_data_train_val.var

RNA_data_val = sc.read_h5ad('rna_val.h5ad')

train_id = list(np.argwhere(~RNA_data.obs.index.isin(RNA_data_val.obs.index.append(RNA_data_test.obs.index))).flatten())
validation_id = list(np.argwhere(RNA_data.obs.index.isin(RNA_data_val.obs.index)).flatten())
test_id = list(np.argwhere(RNA_data.obs.index.isin(RNA_data_test.obs.index)).flatten())

butterfly = Butterfly()
butterfly.load_data(RNA_data, ATAC_data, train_id, test_id, validation_id)

del RNA_data_train_val, ATAC_data_train_val, RNA_data_test, ATAC_data_test, RNA_data, ATAC_data, RNA_data_val, train_id, validation_id, test_id
gc.collect()

butterfly.data_preprocessing(normalize_total=False, log1p=False, use_hvg=False, n_top_genes=None, binary_data=False, 
                             filter_features=False, fpeaks=None, tfidf=False, normalize=False) 

butterfly.ATAC_data_p.var['chrom'] = butterfly.ATAC_data_p.var['gene_ids'].map(lambda x: x.split(':')[0])
chrom_list = []
last_one = ''
for i in range(len(butterfly.ATAC_data_p.var.chrom)):
    temp = butterfly.ATAC_data_p.var.chrom[i]
    if temp[0 : 3] == 'chr':
        if not temp == last_one:
            chrom_list.append(1)
            last_one = temp
        else:
            chrom_list[-1] += 1
    else:
        chrom_list[-1] += 1

butterfly.augmentation(aug_type=None)
butterfly.construct_model(chrom_list=chrom_list)
butterfly.train_model(batch_size=16, load_model='/fs/home/jiluzhang/Nature_methods/Figure_2/pan_cancer/scbt/step_0')
A2R_predict, R2A_predict = butterfly.test_model(test_cluster=False, test_figure=False, output_data=True)


## nohup python scbt_b_2.py > scbt_b_2_20241101.log &  # 1575618
import os
from scButterfly.butterfly import Butterfly
from scButterfly.split_datasets import *
import scanpy as sc
import anndata as ad
import random
import numpy as np
import gc

os.environ["CUDA_VISIBLE_DEVICES"] = "5"

RNA_data_train_val = sc.read_h5ad('rna_train_val_2.h5ad')
ATAC_data_train_val = sc.read_h5ad('atac_train_val_2.h5ad')
RNA_data_test = sc.read_h5ad('rna_test.h5ad')
ATAC_data_test = sc.read_h5ad('atac_test.h5ad')

RNA_data = ad.concat([RNA_data_train_val, RNA_data_test])
RNA_data.var = RNA_data_train_val.var
ATAC_data = ad.concat([ATAC_data_train_val, ATAC_data_test])
ATAC_data.var = ATAC_data_train_val.var

RNA_data_val = sc.read_h5ad('rna_val.h5ad')

train_id = list(np.argwhere(~RNA_data.obs.index.isin(RNA_data_val.obs.index.append(RNA_data_test.obs.index))).flatten())
validation_id = list(np.argwhere(RNA_data.obs.index.isin(RNA_data_val.obs.index)).flatten())
test_id = list(np.argwhere(RNA_data.obs.index.isin(RNA_data_test.obs.index)).flatten())

butterfly = Butterfly()
butterfly.load_data(RNA_data, ATAC_data, train_id, test_id, validation_id)

del RNA_data_train_val, ATAC_data_train_val, RNA_data_test, ATAC_data_test, RNA_data, ATAC_data, RNA_data_val, train_id, validation_id, test_id
gc.collect()

butterfly.data_preprocessing(normalize_total=False, log1p=False, use_hvg=False, n_top_genes=None, binary_data=False, 
                             filter_features=False, fpeaks=None, tfidf=False, normalize=False) 

butterfly.ATAC_data_p.var['chrom'] = butterfly.ATAC_data_p.var['gene_ids'].map(lambda x: x.split(':')[0])
chrom_list = []
last_one = ''
for i in range(len(butterfly.ATAC_data_p.var.chrom)):
    temp = butterfly.ATAC_data_p.var.chrom[i]
    if temp[0 : 3] == 'chr':
        if not temp == last_one:
            chrom_list.append(1)
            last_one = temp
        else:
            chrom_list[-1] += 1
    else:
        chrom_list[-1] += 1

butterfly.augmentation(aug_type=None)
butterfly.construct_model(chrom_list=chrom_list)
butterfly.train_model(batch_size=16, load_model='/fs/home/jiluzhang/Nature_methods/Figure_2/pan_cancer/scbt/step_1')
A2R_predict, R2A_predict = butterfly.test_model(test_cluster=False, test_figure=False, output_data=True)


cp predict/R2A.h5ad atac_scbt.h5ad
python cal_auroc_auprc.py --pred atac_scbt.h5ad --true atac_test.h5ad
# Cell-wise AUROC: 0.8831
# Cell-wise AUPRC: 0.1625
# Peak-wise AUROC: 0.6480
# Peak-wise AUPRC: 0.0398

python plot_save_umap.py --pred atac_scbt.h5ad --true atac_test.h5ad  # 8 min
python cal_cluster.py --file atac_scbt_umap.h5ad
# AMI: 0.5018
# ARI: 0.2915
# HOM: 0.7154
# NMI: 0.5029


#### BABEL
## concat train & valid dataset
## python concat_train_val.py

## split h5ad files
import scanpy as sc
import anndata as ad
import random
from tqdm import tqdm

rna_train = sc.read_h5ad('rna_train.h5ad')
rna_val = sc.read_h5ad('rna_val.h5ad')
atac_train = sc.read_h5ad('atac_train.h5ad')
atac_val = sc.read_h5ad('atac_val.h5ad')

train_cell_idx = list(rna_train.obs.index)
random.seed(0)
random.shuffle(train_cell_idx)

p = int(len(train_cell_idx)/3)
for i in tqdm(range(3), ncols=80, desc='split h5ad'):
    rna_train_val = ad.concat([rna_train[train_cell_idx[(i*p):(i*p+p)]], rna_val])
    rna_train_val.var = rna_train.var
    rna_train_val.write('rna_train_val_'+str(i)+'.h5ad')
    
    atac_train_val = ad.concat([atac_train[train_cell_idx[(i*p):(i*p+p)]], atac_val])
    atac_train_val.var = atac_train.var
    atac_train_val.write('atac_train_val_'+str(i)+'.h5ad')

## python h5ad2h5.py -n train_val -i 0
## python h5ad2h5.py -n train_val -i 1
## python h5ad2h5.py -n train_val -i 2
import h5py
import numpy as np
from scipy.sparse import csr_matrix, hstack
import argparse

parser = argparse.ArgumentParser(description='concat RNA and ATAC h5ad to h5 format')
parser.add_argument('-n', '--name', type=str, help='sample type or name')
parser.add_argument('-i', '--idx', type=str, help='splitting index')

args = parser.parse_args()
sn = args.name

i_raw = args.idx
if i_raw.isdigit():
    i = i_raw
else:
    i = ''

rna  = h5py.File('rna_'+sn+'_'+i+'.h5ad', 'r')
atac = h5py.File('atac_'+sn+'_'+i+'.h5ad', 'r')
out  = h5py.File(sn+'_'+i+'.h5', 'w')

g = out.create_group('matrix')
g.create_dataset('barcodes', data=rna['obs']['_index'][:])
rna_atac_csr_mat = hstack((csr_matrix((rna['X']['data'], rna['X']['indices'],  rna['X']['indptr']), shape=[rna['obs']['_index'].shape[0], rna['var']['_index'].shape[0]]),
                           csr_matrix((atac['X']['data'], atac['X']['indices'],  atac['X']['indptr']), shape=[atac['obs']['_index'].shape[0], atac['var']['_index'].shape[0]])))
rna_atac_csr_mat = rna_atac_csr_mat.tocsr()
g.create_dataset('data', data=rna_atac_csr_mat.data)
g.create_dataset('indices', data=rna_atac_csr_mat.indices)
g.create_dataset('indptr',  data=rna_atac_csr_mat.indptr)
l = list(rna_atac_csr_mat.shape)
l.reverse()
g.create_dataset('shape', data=l)

g_2 = g.create_group('features')
g_2.create_dataset('_all_tag_keys', data=np.array([b'genome', b'interval']))
g_2.create_dataset('feature_type', data=np.append([b'Gene Expression']*rna['var']['gene_ids'].shape[0], [b'Peaks']*atac['var']['gene_ids'].shape[0]))
g_2.create_dataset('genome', data=np.array([b'GRCh38'] * (rna['var']['gene_ids'].shape[0]+atac['var']['gene_ids'].shape[0])))
g_2.create_dataset('id', data=np.append(rna['var']['gene_ids'][:], atac['var']['gene_ids'][:]))
g_2.create_dataset('interval', data=np.append(rna['var']['gene_ids'][:], atac['var']['gene_ids'][:]))
g_2.create_dataset('name', data=np.append(rna['var']['_index'][:], atac['var']['_index'][:]))      

out.close()


# python h5ad2h5.py -n test -i None

nohup python /fs/home/jiluzhang/BABEL/bin/train_model.py --data train_val_0.h5 --outdir train_out_0 --batchsize 512 --earlystop 25 --device 3 --nofilter > train_20241031_0.log &
# error: ValueError: could not convert integer scalar (too many values)
# 367839

# add 'spliced_net_.initialize()' to train_model.py
nohup python /fs/home/jiluzhang/BABEL/bin/train_model.py --data train_val_1.h5 --pretrain train_out_0/net_params.pt --outdir train_out_1 --batchsize 512 --earlystop 25 --device 3 --nofilter > train_20241031_1.log &
# 622886
nohup python /fs/home/jiluzhang/BABEL/bin/train_model.py --data train_val_2.h5 --pretrain train_out_1/net_params.pt --outdir train_out_2 --batchsize 512 --earlystop 25 --device 3 --nofilter > train_20241031_2.log &
# 662283

python /fs/home/jiluzhang/BABEL/bin/predict_model.py --checkpoint train_out_2 --data test.h5 --outdir test_out --device 3 --nofilter --noplot --transonly   # 9 min

## match cells bw pred & true
## python match_cell.py

python cal_auroc_auprc.py --pred atac_babel.h5ad --true atac_test.h5ad
# Cell-wise AUROC: 0.8865
# Cell-wise AUPRC: 0.1631
# Peak-wise AUROC: 0.6583
# Peak-wise AUPRC: 0.0444

python plot_save_umap.py --pred atac_babel.h5ad --true atac_test.h5ad  # 5 min
python cal_cluster.py --file atac_babel_umap.h5ad
# AMI: 0.5387
# ARI: 0.3473
# HOM: 0.7428
# NMI: 0.5396



#### differential peak overlap


































