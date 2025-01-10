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
python plot_save_umap.py --pred atac_cisformer.h5ad --true atac_test.h5ad  # 4 min

python cal_auroc_auprc.py --pred atac_cisformer.h5ad --true atac_test.h5ad
# Cell-wise AUROC: 0.8860
# Cell-wise AUPRC: 0.1629
# Peak-wise AUROC: 0.6459
# Peak-wise AUPRC: 0.0405
python cal_cluster.py --file atac_cisformer_umap.h5ad
# AMI: 0.563
# ARI: 0.3648
# HOM: 0.7824
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


#### plot metrics comparison
from plotnine import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dat = pd.DataFrame({'val': [0.5387, 0.3473, 0.7428, 0.5396, 0.5018, 0.2915, 0.7154, 0.5029, 0.5630, 0.3648, 0.7824, 0.5640]})
dat['alg'] = ['babel']*4 + ['scbt']*4 + ['cifm']*4
dat['alg'] = pd.Categorical(dat['alg'], categories=['babel', 'scbt', 'cifm'])
dat['evl'] = ['ARI', 'AMI', 'NMI', 'HOM']*3
dat['evl'] = pd.Categorical(dat['evl'], categories=['AMI', 'NMI', 'ARI', 'HOM'])

plt.rcParams['pdf.fonttype'] = 42
p = ggplot(dat, aes(x='factor(evl)', y='val', fill='alg')) + geom_bar(stat='identity', position=position_dodge()) + xlab('') + ylab('') +\
                                                             scale_y_continuous(limits=[0, 0.8], breaks=np.arange(0, 0.8+0.1, 0.2)) + theme_bw()
p.save(filename='babel_scbt_cifm_ari_ami_nmi_hom.pdf', dpi=600, height=4, width=5)



#### marker peak overlap
## add 'zmin=zmin, zmax=zmax' & 'zmin: int = 0, zmax: int = 5' to '/fs/home/jiluzhang/softwares/miniconda3/envs/snapatac2/lib/python3.8/site-packages/snapatac2/plotting/__init__.py'
## 'return render_plot(fig, width, height, interactive, show, out_file)' -> 'return mat, render_plot(fig, width, height, interactive, show, out_file)'
import snapatac2 as snap
import numpy as np
from scipy import stats

true = snap.read('atac_test.h5ad', backed=None)
true.obs['cell_anno'].replace({'Plasma': 'B-cells'}, inplace=True)
babel = snap.read('atac_babel_umap.h5ad', backed=None)
babel.var.index = true.var.index
scbt = snap.read('atac_scbt_umap.h5ad', backed=None)
scbt.var.index = true.var.index
cifm = snap.read('atac_cisformer_umap.h5ad', backed=None)
cifm.var.index = true.var.index

true.obs['cell_anno'].value_counts()
# B-cells        2071
# Fibroblasts    1706
# Macrophages    1500
# T-cells        1042
# Endothelial     159

## p=0.025
true_marker_peaks = snap.tl.marker_regions(true, groupby='cell_anno', pvalue=0.025)
babel_marker_peaks = snap.tl.marker_regions(babel, groupby='cell_anno', pvalue=0.025)
scbt_marker_peaks = snap.tl.marker_regions(scbt, groupby='cell_anno', pvalue=0.025)
cifm_marker_peaks = snap.tl.marker_regions(cifm, groupby='cell_anno', pvalue=0.025)

true_mat = snap.pl.regions(true, groupby='cell_anno', peaks=true_marker_peaks, show=False, width=500, height=500, zmin=0, zmax=2.5, out_file='marker_peak_heatmap_true_true.pdf')[0]
babel_mat = snap.pl.regions(babel, groupby='cell_anno', peaks=true_marker_peaks, show=False, width=500, height=500, zmin=0, zmax=2.5, out_file='marker_peak_heatmap_true_babel.pdf')[0]
scbt_mat = snap.pl.regions(scbt, groupby='cell_anno', peaks=true_marker_peaks, show=False, width=500, height=500, zmin=0, zmax=2.5, out_file='marker_peak_heatmap_true_scbt.pdf')[0]
cifm_mat = snap.pl.regions(cifm, groupby='cell_anno', peaks=true_marker_peaks, show=False, width=500, height=500, zmin=0, zmax=2.5, out_file='marker_peak_heatmap_true_cisformer.pdf')[0]

stats.pearsonr(babel_mat.flatten(), true_mat.flatten())[0]   # 0.6588884017204633
stats.pearsonr(scbt_mat.flatten(), true_mat.flatten())[0]    # 0.6605551608029702
stats.pearsonr(cifm_mat.flatten(), true_mat.flatten())[0]    # 0.6739921318072876

# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt

# true_bcell = true[true.obs['cell_anno']=='B-cells', :].X.toarray().sum(axis=0)
# true_bcell_norm = true_bcell/true_bcell.sum()
# true_fibro = true[true.obs['cell_anno']=='Fibroblasts', :].X.toarray().sum(axis=0)
# true_fibro_norm = true_fibro/true_fibro.sum()
# true_macro = true[true.obs['cell_anno']=='Macrophages', :].X.toarray().sum(axis=0)
# true_macro_norm = true_macro/true_macro.sum()
# true_tcell = true[true.obs['cell_anno']=='T-cells', :].X.toarray().sum(axis=0)
# true_tcell_norm = true_tcell/true_tcell.sum()
# true_endo = true[true.obs['cell_anno']=='Endothelial', :].X.toarray().sum(axis=0)
# true_endo_norm = true_endo/true_endo.sum()
# true_df = pd.DataFrame({'B cells':true_bcell_norm, 'Fibroblasts':true_fibro_norm, 'Macrophages':true_macro_norm,
#                         'T cells':true_tcell_norm, 'Endothelial':true_endo_norm})
# true_df.index = true.var.index
# idx = list(true_marker_peaks['B-cells']) + list(true_marker_peaks['Fibroblasts']) + list(true_marker_peaks['Macrophages']) +\
#       list(true_marker_peaks['T-cells']) + list(true_marker_peaks['Endothelial'])
# df = true_df.loc[idx, :]

# plt.figure(figsize=(5, 8))
# #sns.heatmap(df, cmap='RdBu_r', yticklabels=False)  viridis
# sns.clustermap(df, cmap='RdBu_r', z_score=0, vmin=-4, vmax=4, row_cluster=False, col_cluster=False, yticklabels=False) 
# #plt.savefig('marker_peak_heatmap_true_true.pdf')
# sns.clustermap(np.log10(df*1000000+1), row_cluster=False, col_cluster=False, yticklabels=False) 
# plt.savefig('tmp.pdf')
# plt.close()

# true_adata = snap.tl.aggregate_X(true, groupby='cell_anno', normalize="RPKM")
# true_df = pd.DataFrame(true_adata.X.T)
# true_df.index = true_adata.var.index
# true_df.columns = true_adata.obs.index


cnt_dict = {}
for cell_type in true.obs['cell_anno'].value_counts().index:
    cnt_dict[cell_type] = [len(true_marker_peaks[cell_type]),
                           len(babel_marker_peaks[cell_type]),
                           len(scbt_marker_peaks[cell_type]),
                           len(cifm_marker_peaks[cell_type])]
# {'B-cells': [3123, 11509, 36808, 9383],
#  'Fibroblasts': [6538, 28443, 45277, 28179],
#  'Macrophages': [1591, 13408, 15889, 20692],
#  'T-cells': [936, 6491, 2963, 8066],
#  'Endothelial': [1136, 2282, 9900, 1654]}

res_dict = {}
for cell_type in true.obs['cell_anno'].value_counts().index:
    res_dict[cell_type] = [len(np.intersect1d(babel_marker_peaks[cell_type], true_marker_peaks[cell_type])),
                           len(np.intersect1d(scbt_marker_peaks[cell_type], true_marker_peaks[cell_type])),
                           len(np.intersect1d(cifm_marker_peaks[cell_type], true_marker_peaks[cell_type]))]
# {'B-cells': [325, 845, 301],
#  'Fibroblasts': [1462, 1835, 1400],
#  'Macrophages': [335, 464, 555],
#  'T-cells': [262, 152, 299],
#  'Endothelial': [80, 213, 25]}


#### motif enrichment
## true
true_motifs = snap.tl.motif_enrichment(motifs=snap.datasets.cis_bp(unique=True),
                                       regions=true_marker_peaks,
                                       genome_fasta=snap.genome.hg38,
                                       background=None,
                                       method='hypergeometric')
# snap.pl.motif_enrichment(true_motifs, min_log_fc=0.5, max_fdr=0.00001, height=1600, interactive=False, show=False, out_file='marker_peak_enrichment_true.pdf')

for c in set(true.obs.cell_anno.values):
    true_motifs_sp = true_motifs[c].to_pandas()
    del true_motifs_sp['family']
    true_motifs_sp.to_csv('motif_enrichment_true_'+c+'.txt', sep='\t', index=False)

## babel
babel_motifs = snap.tl.motif_enrichment(motifs=snap.datasets.cis_bp(unique=True),
                                        regions=babel_marker_peaks,
                                        genome_fasta=snap.genome.hg38,
                                        background=None,
                                        method='hypergeometric')

for c in set(true.obs.cell_anno.values):
    babel_motifs_sp = babel_motifs[c].to_pandas()
    del babel_motifs_sp['family']
    babel_motifs_sp.to_csv('motif_enrichment_babel_'+c+'.txt', sep='\t', index=False)

## scbt
scbt_motifs = snap.tl.motif_enrichment(motifs=snap.datasets.cis_bp(unique=True),
                                       regions=scbt_marker_peaks,
                                       genome_fasta=snap.genome.hg38,
                                       background=None,
                                       method='hypergeometric')

for c in set(true.obs.cell_anno.values):
    scbt_motifs_sp = scbt_motifs[c].to_pandas()
    del scbt_motifs_sp['family']
    scbt_motifs_sp.to_csv('motif_enrichment_scbt_'+c+'.txt', sep='\t', index=False)

## cisformer
cifm_motifs = snap.tl.motif_enrichment(motifs=snap.datasets.cis_bp(unique=True),
                                       regions=cifm_marker_peaks,
                                       genome_fasta=snap.genome.hg38,
                                       background=None,
                                       method='hypergeometric')

for c in set(true.obs.cell_anno.values):
    cifm_motifs_sp = cifm_motifs[c].to_pandas()
    del cifm_motifs_sp['family']
    cifm_motifs_sp.to_csv('motif_enrichment_cifm_'+c+'.txt', sep='\t', index=False)


#### Plot top enriched motifs
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

## true
true_df = pd.DataFrame()
for c in ['B-cells', 'Endothelial', 'Fibroblasts', 'Macrophages', 'T-cells']:
    raw = pd.read_table('motif_enrichment_true_'+c+'.txt')[['name', 'log2(fold change)', 'adjusted p-value']]
    raw.rename(columns={'log2(fold change)':c+'_logfc', 
                        'adjusted p-value': c+'_fdr'}, inplace=True)
    true_df = pd.concat([true_df, raw.iloc[:, 1:]], axis=1)

true_df.index = raw['name'].values

bcell_tf = list(true_df[(true_df['B-cells_logfc']>np.log2(1.2)) & 
                        (true_df['B-cells_fdr']<0.05) & 
                        (true_df['B-cells_logfc']>true_df['Endothelial_logfc']) & 
                        (true_df['B-cells_logfc']>true_df['Fibroblasts_logfc']) &
                        (true_df['B-cells_logfc']>true_df['Macrophages_logfc']) &
                        (true_df['B-cells_logfc']>true_df['T-cells_logfc'])].index)  # 71
endo_tf = list(true_df[(true_df['Endothelial_logfc']>np.log2(1.2)) & 
                       (true_df['Endothelial_fdr']<0.05) & 
                       (true_df['Endothelial_logfc']>true_df['B-cells_logfc']) & 
                       (true_df['Endothelial_logfc']>true_df['Fibroblasts_logfc']) &
                       (true_df['Endothelial_logfc']>true_df['Macrophages_logfc']) &
                       (true_df['Endothelial_logfc']>true_df['T-cells_logfc'])].index)  # 83
fibro_tf = list(true_df[(true_df['Fibroblasts_logfc']>np.log2(1.2)) & 
                        (true_df['Fibroblasts_fdr']<0.05) & 
                        (true_df['Fibroblasts_logfc']>true_df['B-cells_logfc']) & 
                        (true_df['Fibroblasts_logfc']>true_df['Endothelial_logfc']) &
                        (true_df['Fibroblasts_logfc']>true_df['Macrophages_logfc']) &
                        (true_df['Fibroblasts_logfc']>true_df['T-cells_logfc'])].index)  # 85
macro_tf = list(true_df[(true_df['Macrophages_logfc']>np.log2(1.2)) & 
                        (true_df['Macrophages_fdr']<0.05) & 
                        (true_df['Macrophages_logfc']>true_df['B-cells_logfc']) & 
                        (true_df['Macrophages_logfc']>true_df['Endothelial_logfc']) &
                        (true_df['Macrophages_logfc']>true_df['Fibroblasts_logfc']) &
                        (true_df['Macrophages_logfc']>true_df['T-cells_logfc'])].index)  # 11
tcell_tf = list(true_df[(true_df['T-cells_logfc']>np.log2(1.2)) & 
                        (true_df['T-cells_fdr']<0.05) & 
                        (true_df['T-cells_logfc']>true_df['Endothelial_logfc']) & 
                        (true_df['T-cells_logfc']>true_df['Fibroblasts_logfc']) &
                        (true_df['T-cells_logfc']>true_df['Macrophages_logfc']) &
                        (true_df['T-cells_logfc']>true_df['B-cells_logfc'])].index)   # 46
tf_lst = bcell_tf + endo_tf + fibro_tf + macro_tf + tcell_tf   # 296

true_dat = true_df.loc[tf_lst].iloc[:, [0,2,4,6,8]]
true_dat.replace(-np.inf, -4, inplace=True)
true_dat.replace(np.inf, 4, inplace=True)

sns.clustermap(true_dat, cmap='viridis', z_score=0, vmin=-2, vmax=2, row_cluster=False, col_cluster=False, yticklabels=False, figsize=(4,15)) 
plt.savefig('motif_diff_true_heatmap.pdf')
plt.close()

## babel
babel_df = pd.DataFrame()
for c in ['B-cells', 'Endothelial', 'Fibroblasts', 'Macrophages', 'T-cells']:
    raw = pd.read_table('motif_enrichment_babel_'+c+'.txt')[['name', 'log2(fold change)', 'adjusted p-value']]
    raw.rename(columns={'log2(fold change)':c+'_logfc', 
                        'adjusted p-value': c+'_fdr'}, inplace=True)
    babel_df = pd.concat([babel_df, raw.iloc[:, 1:]], axis=1)

babel_df.index = raw['name'].values
babel_dat = babel_df.loc[tf_lst].iloc[:, [0,2,4,6,8]]
babel_dat.replace(-np.inf, -4, inplace=True)
babel_dat.replace(np.inf, 4, inplace=True)

sns.clustermap(babel_dat, cmap='viridis', z_score=0, vmin=-2, vmax=2, row_cluster=False, col_cluster=False, yticklabels=False, figsize=(4,15)) 
plt.savefig('motif_diff_babel_heatmap.pdf')
plt.close()

## scbt
scbt_df = pd.DataFrame()
for c in ['B-cells', 'Endothelial', 'Fibroblasts', 'Macrophages', 'T-cells']:
    raw = pd.read_table('motif_enrichment_scbt_'+c+'.txt')[['name', 'log2(fold change)', 'adjusted p-value']]
    raw.rename(columns={'log2(fold change)':c+'_logfc', 
                        'adjusted p-value': c+'_fdr'}, inplace=True)
    scbt_df = pd.concat([scbt_df, raw.iloc[:, 1:]], axis=1)

scbt_df.index = raw['name'].values
scbt_dat = scbt_df.loc[tf_lst].iloc[:, [0,2,4,6,8]]
scbt_dat.replace(-np.inf, -4, inplace=True)
scbt_dat.replace(np.inf, 4, inplace=True)

sns.clustermap(scbt_dat, cmap='viridis', z_score=0, vmin=-2, vmax=2, row_cluster=False, col_cluster=False, yticklabels=False, figsize=(4,15)) 
plt.savefig('motif_diff_scbt_heatmap.pdf')
plt.close()

## cifm
cifm_df = pd.DataFrame()
for c in ['B-cells', 'Endothelial', 'Fibroblasts', 'Macrophages', 'T-cells']:
    raw = pd.read_table('motif_enrichment_cifm_'+c+'.txt')[['name', 'log2(fold change)', 'adjusted p-value']]
    raw.rename(columns={'log2(fold change)':c+'_logfc', 
                        'adjusted p-value': c+'_fdr'}, inplace=True)
    cifm_df = pd.concat([cifm_df, raw.iloc[:, 1:]], axis=1)

cifm_df.index = raw['name'].values
cifm_dat = cifm_df.loc[tf_lst].iloc[:, [0,2,4,6,8]]
cifm_dat.replace(-np.inf, -4, inplace=True)
cifm_dat.replace(np.inf, 4, inplace=True)

sns.clustermap(cifm_dat, cmap='viridis', z_score=0, vmin=-2, vmax=2, row_cluster=False, col_cluster=False, yticklabels=False, figsize=(4,15)) 
plt.savefig('motif_diff_cifm_heatmap.pdf')
plt.close()


stats.pearsonr(babel_dat.values.flatten(), true_dat.values.flatten())[0]   # 0.45424571198089597
stats.pearsonr(scbt_dat.values.flatten(), true_dat.values.flatten())[0]    # 0.46659538870807743
stats.pearsonr(cifm_dat.values.flatten(), true_dat.values.flatten())[0]    # 0.4777760875672916


#### motif overlapping
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

true_df = pd.DataFrame()
for c in ['B-cells', 'Endothelial', 'Fibroblasts', 'Macrophages', 'T-cells']:
    raw = pd.read_table('motif_enrichment_true_'+c+'.txt')[['name', 'log2(fold change)', 'adjusted p-value']]
    raw.rename(columns={'log2(fold change)':c+'_logfc', 
                        'adjusted p-value': c+'_fdr'}, inplace=True)
    true_df = pd.concat([true_df, raw.iloc[:, 1:]], axis=1)

true_df.index = raw['name'].values

cifm_df = pd.DataFrame()
for c in ['B-cells', 'Endothelial', 'Fibroblasts', 'Macrophages', 'T-cells']:
    raw = pd.read_table('motif_enrichment_cifm_'+c+'.txt')[['name', 'log2(fold change)', 'adjusted p-value']]
    raw.rename(columns={'log2(fold change)':c+'_logfc', 
                        'adjusted p-value': c+'_fdr'}, inplace=True)
    cifm_df = pd.concat([cifm_df, raw.iloc[:, 1:]], axis=1)

cifm_df.index = raw['name'].values

babel_df = pd.DataFrame()
for c in ['B-cells', 'Endothelial', 'Fibroblasts', 'Macrophages', 'T-cells']:
    raw = pd.read_table('motif_enrichment_babel_'+c+'.txt')[['name', 'log2(fold change)', 'adjusted p-value']]
    raw.rename(columns={'log2(fold change)':c+'_logfc', 
                        'adjusted p-value': c+'_fdr'}, inplace=True)
    babel_df = pd.concat([babel_df, raw.iloc[:, 1:]], axis=1)

babel_df.index = raw['name'].values

scbt_df = pd.DataFrame()
for c in ['B-cells', 'Endothelial', 'Fibroblasts', 'Macrophages', 'T-cells']:
    raw = pd.read_table('motif_enrichment_scbt_'+c+'.txt')[['name', 'log2(fold change)', 'adjusted p-value']]
    raw.rename(columns={'log2(fold change)':c+'_logfc', 
                        'adjusted p-value': c+'_fdr'}, inplace=True)
    scbt_df = pd.concat([scbt_df, raw.iloc[:, 1:]], axis=1)

scbt_df.index = raw['name'].values

## B-cells
bcell_tf_true = list(true_df[(true_df['B-cells_logfc']>np.log2(1.2)) & 
                             (true_df['B-cells_fdr']<0.05) & 
                             (true_df['B-cells_logfc']>true_df['Endothelial_logfc']) & 
                             (true_df['B-cells_logfc']>true_df['Fibroblasts_logfc']) &
                             (true_df['B-cells_logfc']>true_df['Macrophages_logfc']) &
                             (true_df['B-cells_logfc']>true_df['T-cells_logfc'])].index)  # 71
bcell_tf_cifm = list(true_df[(cifm_df['B-cells_logfc']>np.log2(1.2)) & 
                             (cifm_df['B-cells_fdr']<0.05) & 
                             (cifm_df['B-cells_logfc']>cifm_df['Endothelial_logfc']) & 
                             (cifm_df['B-cells_logfc']>cifm_df['Fibroblasts_logfc']) &
                             (cifm_df['B-cells_logfc']>cifm_df['Macrophages_logfc']) &
                             (cifm_df['B-cells_logfc']>cifm_df['T-cells_logfc'])].index)  # 77
bcell_tf_babel = list(true_df[(babel_df['B-cells_logfc']>np.log2(1.2)) & 
                              (babel_df['B-cells_fdr']<0.05) & 
                              (babel_df['B-cells_logfc']>babel_df['Endothelial_logfc']) & 
                              (babel_df['B-cells_logfc']>babel_df['Fibroblasts_logfc']) &
                              (babel_df['B-cells_logfc']>babel_df['Macrophages_logfc']) &
                              (babel_df['B-cells_logfc']>babel_df['T-cells_logfc'])].index)  # 169
bcell_tf_scbt = list(true_df[(scbt_df['B-cells_logfc']>np.log2(1.2)) & 
                             (scbt_df['B-cells_fdr']<0.05) & 
                             (scbt_df['B-cells_logfc']>scbt_df['Endothelial_logfc']) & 
                             (scbt_df['B-cells_logfc']>scbt_df['Fibroblasts_logfc']) &
                             (scbt_df['B-cells_logfc']>scbt_df['Macrophages_logfc']) &
                             (scbt_df['B-cells_logfc']>scbt_df['T-cells_logfc'])].index)  # 57

from matplotlib_venn import venn2
plt.rcParams['pdf.fonttype'] = 42

## bcell true vs. cifm
bcell_tf_true_cnt = len(bcell_tf_true)
bcell_tf_cifm_cnt = len(bcell_tf_cifm)
bcell_tf_true_cifm_cnt = len([i for i in bcell_tf_true if i in bcell_tf_cifm])

plt.figure(figsize=(4, 4))
venn2(subsets=(bcell_tf_true_cnt-bcell_tf_true_cifm_cnt, 
               bcell_tf_cifm_cnt-bcell_tf_true_cifm_cnt,
               bcell_tf_true_cifm_cnt), set_labels=('True', 'Cisformer'))
plt.savefig('bcell_tf_true_cifm_venn.pdf')
plt.close()

[i for i in bcell_tf_true if i in bcell_tf_cifm]

## bcell true vs. babel
bcell_tf_true_cnt = len(bcell_tf_true)
bcell_tf_babel_cnt = len(bcell_tf_babel)
bcell_tf_true_babel_cnt = len([i for i in bcell_tf_true if i in bcell_tf_babel])

plt.figure(figsize=(4, 4))
venn2(subsets=(bcell_tf_true_cnt-bcell_tf_true_babel_cnt, 
               bcell_tf_babel_cnt-bcell_tf_true_babel_cnt,
               bcell_tf_true_babel_cnt), set_labels=('True', 'BABEL'))
plt.savefig('bcell_tf_true_babel_venn.pdf')
plt.close()

## bcell true vs. scbt
bcell_tf_true_cnt = len(bcell_tf_true)
bcell_tf_scbt_cnt = len(bcell_tf_scbt)
bcell_tf_true_scbt_cnt = len([i for i in bcell_tf_true if i in bcell_tf_scbt])

plt.figure(figsize=(4, 4))
venn2(subsets=(bcell_tf_true_cnt-bcell_tf_true_scbt_cnt, 
               bcell_tf_scbt_cnt-bcell_tf_true_scbt_cnt,
               bcell_tf_true_scbt_cnt), set_labels=('True', 'scbt'))
plt.savefig('bcell_tf_true_scbt_venn.pdf')
plt.close()

## endo
endo_tf_true = list(true_df[(true_df['Endothelial_logfc']>np.log2(1.2)) & 
                            (true_df['Endothelial_fdr']<0.05) & 
                            (true_df['Endothelial_logfc']>true_df['B-cells_logfc']) & 
                            (true_df['Endothelial_logfc']>true_df['Fibroblasts_logfc']) &
                            (true_df['Endothelial_logfc']>true_df['Macrophages_logfc']) &
                            (true_df['Endothelial_logfc']>true_df['T-cells_logfc'])].index)  # 83
endo_tf_cifm = list(cifm_df[(cifm_df['Endothelial_logfc']>np.log2(1.2)) & 
                            (cifm_df['Endothelial_fdr']<0.05) & 
                            (cifm_df['Endothelial_logfc']>cifm_df['B-cells_logfc']) & 
                            (cifm_df['Endothelial_logfc']>cifm_df['Fibroblasts_logfc']) &
                            (cifm_df['Endothelial_logfc']>cifm_df['Macrophages_logfc']) &
                            (cifm_df['Endothelial_logfc']>cifm_df['T-cells_logfc'])].index) 
endo_tf_babel = list(babel_df[(babel_df['Endothelial_logfc']>np.log2(1.2)) & 
                            (babel_df['Endothelial_fdr']<0.05) & 
                            (babel_df['Endothelial_logfc']>babel_df['B-cells_logfc']) & 
                            (babel_df['Endothelial_logfc']>babel_df['Fibroblasts_logfc']) &
                            (babel_df['Endothelial_logfc']>babel_df['Macrophages_logfc']) &
                            (babel_df['Endothelial_logfc']>babel_df['T-cells_logfc'])].index)
endo_tf_scbt = list(scbt_df[(scbt_df['Endothelial_logfc']>np.log2(1.2)) & 
                            (scbt_df['Endothelial_fdr']<0.05) & 
                            (scbt_df['Endothelial_logfc']>scbt_df['B-cells_logfc']) & 
                            (scbt_df['Endothelial_logfc']>scbt_df['Fibroblasts_logfc']) &
                            (scbt_df['Endothelial_logfc']>scbt_df['Macrophages_logfc']) &
                            (scbt_df['Endothelial_logfc']>scbt_df['T-cells_logfc'])].index) 

## endo true vs. cifm
endo_tf_true_cnt = len(endo_tf_true)
endo_tf_cifm_cnt = len(endo_tf_cifm)
endo_tf_true_cifm_cnt = len([i for i in endo_tf_true if i in endo_tf_cifm])

plt.figure(figsize=(4, 4))
venn2(subsets=(endo_tf_true_cnt-endo_tf_true_cifm_cnt, 
               endo_tf_cifm_cnt-endo_tf_true_cifm_cnt,
               endo_tf_true_cifm_cnt), set_labels=('True', 'Cisformer'))
plt.savefig('endo_tf_true_cifm_venn.pdf')
plt.close()

[i for i in endo_tf_true if i in endo_tf_cifm]

## endo true vs. babel
endo_tf_true_cnt = len(endo_tf_true)
endo_tf_babel_cnt = len(endo_tf_babel)
endo_tf_true_babel_cnt = len([i for i in endo_tf_true if i in endo_tf_babel])

plt.figure(figsize=(4, 4))
venn2(subsets=(endo_tf_true_cnt-endo_tf_true_babel_cnt, 
               endo_tf_babel_cnt-endo_tf_true_babel_cnt,
               endo_tf_true_babel_cnt), set_labels=('True', 'BABEL'))
plt.savefig('endo_tf_true_babel_venn.pdf')
plt.close()

## endo true vs. scbt
endo_tf_true_cnt = len(endo_tf_true)
endo_tf_scbt_cnt = len(endo_tf_scbt)
endo_tf_true_scbt_cnt = len([i for i in endo_tf_true if i in endo_tf_scbt])

plt.figure(figsize=(4, 4))
venn2(subsets=(endo_tf_true_cnt-endo_tf_true_scbt_cnt, 
               endo_tf_scbt_cnt-endo_tf_true_scbt_cnt,
               endo_tf_true_scbt_cnt), set_labels=('True', 'scbt'))
plt.savefig('endo_tf_true_scbt_venn.pdf')
plt.close()


## fibro
fibro_tf_true = list(true_df[(true_df['Fibroblasts_logfc']>np.log2(1.2)) & 
                             (true_df['Fibroblasts_fdr']<0.05) & 
                             (true_df['Fibroblasts_logfc']>true_df['B-cells_logfc']) & 
                             (true_df['Fibroblasts_logfc']>true_df['Endothelial_logfc']) &
                             (true_df['Fibroblasts_logfc']>true_df['Macrophages_logfc']) &
                             (true_df['Fibroblasts_logfc']>true_df['T-cells_logfc'])].index)  # 85
fibro_tf_cifm = list(cifm_df[(cifm_df['Fibroblasts_logfc']>np.log2(1.2)) & 
                             (cifm_df['Fibroblasts_fdr']<0.05) & 
                             (cifm_df['Fibroblasts_logfc']>cifm_df['B-cells_logfc']) & 
                             (cifm_df['Fibroblasts_logfc']>cifm_df['Endothelial_logfc']) &
                             (cifm_df['Fibroblasts_logfc']>cifm_df['Macrophages_logfc']) &
                             (cifm_df['Fibroblasts_logfc']>cifm_df['T-cells_logfc'])].index)
fibro_tf_babel = list(babel_df[(babel_df['Fibroblasts_logfc']>np.log2(1.2)) & 
                             (babel_df['Fibroblasts_fdr']<0.05) & 
                             (babel_df['Fibroblasts_logfc']>babel_df['B-cells_logfc']) & 
                             (babel_df['Fibroblasts_logfc']>babel_df['Endothelial_logfc']) &
                             (babel_df['Fibroblasts_logfc']>babel_df['Macrophages_logfc']) &
                             (babel_df['Fibroblasts_logfc']>babel_df['T-cells_logfc'])].index)
fibro_tf_scbt = list(scbt_df[(scbt_df['Fibroblasts_logfc']>np.log2(1.2)) & 
                             (scbt_df['Fibroblasts_fdr']<0.05) & 
                             (scbt_df['Fibroblasts_logfc']>scbt_df['B-cells_logfc']) & 
                             (scbt_df['Fibroblasts_logfc']>scbt_df['Endothelial_logfc']) &
                             (scbt_df['Fibroblasts_logfc']>scbt_df['Macrophages_logfc']) &
                             (scbt_df['Fibroblasts_logfc']>scbt_df['T-cells_logfc'])].index)
## fibro true vs. cifm
fibro_tf_true_cnt = len(fibro_tf_true)
fibro_tf_cifm_cnt = len(fibro_tf_cifm)
fibro_tf_true_cifm_cnt = len([i for i in fibro_tf_true if i in fibro_tf_cifm])

plt.figure(figsize=(4, 4))
venn2(subsets=(fibro_tf_true_cnt-fibro_tf_true_cifm_cnt, 
               fibro_tf_cifm_cnt-fibro_tf_true_cifm_cnt,
               fibro_tf_true_cifm_cnt), set_labels=('True', 'Cisformer'))
plt.savefig('fibro_tf_true_cifm_venn.pdf')
plt.close()

[i for i in fibro_tf_true if i in fibro_tf_cifm]

## fibro true vs. babel
fibro_tf_true_cnt = len(fibro_tf_true)
fibro_tf_babel_cnt = len(fibro_tf_babel)
fibro_tf_true_babel_cnt = len([i for i in fibro_tf_true if i in fibro_tf_babel])

plt.figure(figsize=(4, 4))
venn2(subsets=(fibro_tf_true_cnt-fibro_tf_true_babel_cnt, 
               fibro_tf_babel_cnt-fibro_tf_true_babel_cnt,
               fibro_tf_true_babel_cnt), set_labels=('True', 'BABEL'))
plt.savefig('fibro_tf_true_babel_venn.pdf')
plt.close()

## fibro true vs. scbt
fibro_tf_true_cnt = len(fibro_tf_true)
fibro_tf_scbt_cnt = len(fibro_tf_scbt)
fibro_tf_true_scbt_cnt = len([i for i in fibro_tf_true if i in fibro_tf_scbt])

plt.figure(figsize=(4, 4))
venn2(subsets=(fibro_tf_true_cnt-fibro_tf_true_scbt_cnt, 
               fibro_tf_scbt_cnt-fibro_tf_true_scbt_cnt,
               fibro_tf_true_scbt_cnt), set_labels=('True', 'scbt'))
plt.savefig('fibro_tf_true_scbt_venn.pdf')
plt.close()

## macro
macro_tf_true = list(true_df[(true_df['Macrophages_logfc']>np.log2(1.2)) & 
                             (true_df['Macrophages_fdr']<0.05) & 
                             (true_df['Macrophages_logfc']>true_df['B-cells_logfc']) & 
                             (true_df['Macrophages_logfc']>true_df['Endothelial_logfc']) &
                             (true_df['Macrophages_logfc']>true_df['Fibroblasts_logfc']) &
                             (true_df['Macrophages_logfc']>true_df['T-cells_logfc'])].index)  # 11
macro_tf_cifm = list(cifm_df[(cifm_df['Macrophages_logfc']>np.log2(1.2)) & 
                             (cifm_df['Macrophages_fdr']<0.05) & 
                             (cifm_df['Macrophages_logfc']>cifm_df['B-cells_logfc']) & 
                             (cifm_df['Macrophages_logfc']>cifm_df['Endothelial_logfc']) &
                             (cifm_df['Macrophages_logfc']>cifm_df['Fibroblasts_logfc']) &
                             (cifm_df['Macrophages_logfc']>cifm_df['T-cells_logfc'])].index)
macro_tf_babel = list(babel_df[(babel_df['Macrophages_logfc']>np.log2(1.2)) & 
                             (babel_df['Macrophages_fdr']<0.05) & 
                             (babel_df['Macrophages_logfc']>babel_df['B-cells_logfc']) & 
                             (babel_df['Macrophages_logfc']>babel_df['Endothelial_logfc']) &
                             (babel_df['Macrophages_logfc']>babel_df['Fibroblasts_logfc']) &
                             (babel_df['Macrophages_logfc']>babel_df['T-cells_logfc'])].index)
macro_tf_scbt = list(scbt_df[(scbt_df['Macrophages_logfc']>np.log2(1.2)) & 
                             (scbt_df['Macrophages_fdr']<0.05) & 
                             (scbt_df['Macrophages_logfc']>scbt_df['B-cells_logfc']) & 
                             (scbt_df['Macrophages_logfc']>scbt_df['Endothelial_logfc']) &
                             (scbt_df['Macrophages_logfc']>scbt_df['Fibroblasts_logfc']) &
                             (scbt_df['Macrophages_logfc']>scbt_df['T-cells_logfc'])].index)
## macro true vs. cifm
macro_tf_true_cnt = len(macro_tf_true)
macro_tf_cifm_cnt = len(macro_tf_cifm)
macro_tf_true_cifm_cnt = len([i for i in macro_tf_true if i in macro_tf_cifm])

plt.figure(figsize=(4, 4))
venn2(subsets=(macro_tf_true_cnt-macro_tf_true_cifm_cnt, 
               macro_tf_cifm_cnt-macro_tf_true_cifm_cnt,
               macro_tf_true_cifm_cnt), set_labels=('True', 'Cisformer'))
plt.savefig('macro_tf_true_cifm_venn.pdf')
plt.close()

[i for i in macro_tf_true if i in macro_tf_cifm]

## macro true vs. babel
macro_tf_true_cnt = len(macro_tf_true)
macro_tf_babel_cnt = len(macro_tf_babel)
macro_tf_true_babel_cnt = len([i for i in macro_tf_true if i in macro_tf_babel])

plt.figure(figsize=(4, 4))
venn2(subsets=(macro_tf_true_cnt-macro_tf_true_babel_cnt, 
               macro_tf_babel_cnt-macro_tf_true_babel_cnt,
               macro_tf_true_babel_cnt), set_labels=('True', 'BABEL'))
plt.savefig('macro_tf_true_babel_venn.pdf')
plt.close()

## macro true vs. scbt
macro_tf_true_cnt = len(macro_tf_true)
macro_tf_scbt_cnt = len(macro_tf_scbt)
macro_tf_true_scbt_cnt = len([i for i in macro_tf_true if i in macro_tf_scbt])

plt.figure(figsize=(4, 4))
venn2(subsets=(macro_tf_true_cnt-macro_tf_true_scbt_cnt, 
               macro_tf_scbt_cnt-macro_tf_true_scbt_cnt,
               macro_tf_true_scbt_cnt), set_labels=('True', 'scbt'))
plt.savefig('macro_tf_true_scbt_venn.pdf')
plt.close()

## tcell
tcell_tf_true = list(true_df[(true_df['T-cells_logfc']>np.log2(1.2)) & 
                             (true_df['T-cells_fdr']<0.05) & 
                             (true_df['T-cells_logfc']>true_df['Endothelial_logfc']) & 
                             (true_df['T-cells_logfc']>true_df['Fibroblasts_logfc']) &
                             (true_df['T-cells_logfc']>true_df['Macrophages_logfc']) &
                             (true_df['T-cells_logfc']>true_df['B-cells_logfc'])].index)   # 46
tcell_tf_cifm = list(cifm_df[(cifm_df['T-cells_logfc']>np.log2(1.2)) & 
                             (cifm_df['T-cells_fdr']<0.05) & 
                             (cifm_df['T-cells_logfc']>cifm_df['Endothelial_logfc']) & 
                             (cifm_df['T-cells_logfc']>cifm_df['Fibroblasts_logfc']) &
                             (cifm_df['T-cells_logfc']>cifm_df['Macrophages_logfc']) &
                             (cifm_df['T-cells_logfc']>cifm_df['B-cells_logfc'])].index)
tcell_tf_babel = list(babel_df[(babel_df['T-cells_logfc']>np.log2(1.2)) & 
                             (babel_df['T-cells_fdr']<0.05) & 
                             (babel_df['T-cells_logfc']>babel_df['Endothelial_logfc']) & 
                             (babel_df['T-cells_logfc']>babel_df['Fibroblasts_logfc']) &
                             (babel_df['T-cells_logfc']>babel_df['Macrophages_logfc']) &
                             (babel_df['T-cells_logfc']>babel_df['B-cells_logfc'])].index)
tcell_tf_scbt = list(scbt_df[(scbt_df['T-cells_logfc']>np.log2(1.2)) & 
                             (scbt_df['T-cells_fdr']<0.05) & 
                             (scbt_df['T-cells_logfc']>scbt_df['Endothelial_logfc']) & 
                             (scbt_df['T-cells_logfc']>scbt_df['Fibroblasts_logfc']) &
                             (scbt_df['T-cells_logfc']>scbt_df['Macrophages_logfc']) &
                             (scbt_df['T-cells_logfc']>scbt_df['B-cells_logfc'])].index)
## tcell true vs. cifm
tcell_tf_true_cnt = len(tcell_tf_true)
tcell_tf_cifm_cnt = len(tcell_tf_cifm)
tcell_tf_true_cifm_cnt = len([i for i in tcell_tf_true if i in tcell_tf_cifm])

plt.figure(figsize=(4, 4))
venn2(subsets=(tcell_tf_true_cnt-tcell_tf_true_cifm_cnt, 
               tcell_tf_cifm_cnt-tcell_tf_true_cifm_cnt,
               tcell_tf_true_cifm_cnt), set_labels=('True', 'Cisformer'))
plt.savefig('tcell_tf_true_cifm_venn.pdf')
plt.close()

[i for i in tcell_tf_true if i in tcell_tf_cifm]

## tcell true vs. babel
tcell_tf_true_cnt = len(tcell_tf_true)
tcell_tf_babel_cnt = len(tcell_tf_babel)
tcell_tf_true_babel_cnt = len([i for i in tcell_tf_true if i in tcell_tf_babel])

plt.figure(figsize=(4, 4))
venn2(subsets=(tcell_tf_true_cnt-tcell_tf_true_babel_cnt, 
               tcell_tf_babel_cnt-tcell_tf_true_babel_cnt,
               tcell_tf_true_babel_cnt), set_labels=('True', 'BABEL'))
plt.savefig('tcell_tf_true_babel_venn.pdf')
plt.close()

## tcell true vs. scbt
tcell_tf_true_cnt = len(tcell_tf_true)
tcell_tf_scbt_cnt = len(tcell_tf_scbt)
tcell_tf_true_scbt_cnt = len([i for i in tcell_tf_true if i in tcell_tf_scbt])

plt.figure(figsize=(4, 4))
venn2(subsets=(tcell_tf_true_cnt-tcell_tf_true_scbt_cnt, 
               tcell_tf_scbt_cnt-tcell_tf_true_scbt_cnt,
               tcell_tf_true_scbt_cnt), set_labels=('True', 'scbt'))
plt.savefig('tcell_tf_true_scbt_venn.pdf')
plt.close()


## Attention based chromatin accessibility regulator screening & ranking
import pandas as pd
import numpy as np
import random
from functools import partial
import sys
sys.path.append("M2Mmodel")
from utils import PairDataset
import scanpy as sc
import torch
from M2Mmodel.utils import *
from collections import Counter
import pickle as pkl
import yaml
from M2Mmodel.M2M import M2M_rna2atac

import h5py
from tqdm import tqdm

from multiprocessing import Pool
import pickle

# from config
with open("rna2atac_config_test.yaml", "r") as f:
    config = yaml.safe_load(f)

model = M2M_rna2atac(
            dim = config["model"].get("dim"),

            enc_num_gene_tokens = config["model"].get("total_gene") + 1, # +1 for <PAD>
            enc_num_value_tokens = config["model"].get('max_express') + 1, # +1 for <PAD>
            enc_depth = config["model"].get("enc_depth"),
            enc_heads = config["model"].get("enc_heads"),
            enc_ff_mult = config["model"].get("enc_ff_mult"),
            enc_dim_head = config["model"].get("enc_dim_head"),
            enc_emb_dropout = config["model"].get("enc_emb_dropout"),
            enc_ff_dropout = config["model"].get("enc_ff_dropout"),
            enc_attn_dropout = config["model"].get("enc_attn_dropout"),

            dec_depth = config["model"].get("dec_depth"),
            dec_heads = config["model"].get("dec_heads"),
            dec_ff_mult = config["model"].get("dec_ff_mult"),
            dec_dim_head = config["model"].get("dec_dim_head"),
            dec_emb_dropout = config["model"].get("dec_emb_dropout"),
            dec_ff_dropout = config["model"].get("dec_ff_dropout"),
            dec_attn_dropout = config["model"].get("dec_attn_dropout")
        )
model = model.half()
model.load_state_dict(torch.load('./save/2024-10-25_rna2atac_pan_cancer_5/pytorch_model.bin'))
device = torch.device('cuda:4')
model.to(device)
model.eval()

################################ B cells ################################
# rna  = sc.read_h5ad('rna_test.h5ad')
# random.seed(0)
# bcell_idx = list(np.argwhere(rna.obs['cell_anno']=='B-cells').flatten())
# bcell_idx_20 = random.sample(list(bcell_idx), 20)
# rna[bcell_idx_20].write('rna_test_bcell_20.h5ad')

# atac = sc.read_h5ad('atac_test.h5ad')
# atac[bcell_idx_20].write('atac_test_bcell_20.h5ad')

# python data_preprocess.py -r rna_test_bcell_20.h5ad -a atac_test_bcell_20.h5ad -s test_pt_20 --dt test -n test_bcell --config rna2atac_config_test.yaml

bcell_data = torch.load("./test_pt_20/test_bcell_0.pt")
dataset = PreDataset(bcell_data)
dataloader_kwargs = {'batch_size': 1, 'shuffle': False}
loader = torch.utils.data.DataLoader(dataset, **dataloader_kwargs)

# time: 7 min
i=0
for inputs in tqdm(loader, ncols=80, desc='output attention matrix'):
    rna_sequence, rna_value, atac_sequence, _, enc_pad_mask = [each.to(device) for each in inputs]
    attn = model.generate_attn_weight(rna_sequence, atac_sequence, rna_value, enc_mask=enc_pad_mask, which='decoder')
    attn = attn[0]  # attn = attn[0].to(torch.float16)
    with h5py.File('attn/attn_bcell_'+str(i)+'.h5', 'w') as f:
        f.create_dataset('attn', data=attn)
    torch.cuda.empty_cache()
    i += 1

rna_bcell_20 = sc.read_h5ad('rna_test_bcell_20.h5ad')
nonzero = np.count_nonzero(rna_bcell_20.X.toarray(), axis=0)
# gene_lst = rna_bcell_20.var.index[nonzero>0]  # 8679 (maybe the cutoff is too loose)
gene_lst = rna_bcell_20.var.index[nonzero>=3]  # 3333

atac_bcell_20 = sc.read_h5ad('atac_test_bcell_20.h5ad')

gene_attn = {}
for gene in gene_lst:
    gene_attn[gene] = np.zeros([atac_bcell_20.shape[1], 1], dtype='float32')  # not float16

for i in range(20):
    rna_sequence = bcell_data[0][i].flatten()
    with h5py.File('attn/attn_bcell_'+str(i)+'.h5', 'r') as f:
        attn = f['attn'][:]
        for gene in tqdm(gene_lst, ncols=80, desc='cell '+str(i)):
            idx = torch.argwhere((rna_sequence==(np.argwhere(rna_bcell_20.var.index==gene))[0][0]+1).flatten())
            if len(idx)!=0:
                gene_attn[gene] += attn[:, [idx.item()]]

with open('./attn/attn_20_bcell_3cell.pkl', 'wb') as file:
    pickle.dump(gene_attn, file)

gene_attn_sum = [gene_attn[gene].sum() for gene in gene_lst]
df = pd.DataFrame({'gene':gene_lst, 'cnt':gene_attn_sum})
df.to_csv('./attn/attn_no_norm_cnt_20_bcell_3cell.txt', sep='\t', header=None, index=None)

################################ Endothelial ################################
# rna  = sc.read_h5ad('rna_test.h5ad')
# random.seed(0)
# endo_idx = list(np.argwhere(rna.obs['cell_anno']=='Endothelial').flatten())
# endo_idx_20 = random.sample(list(endo_idx), 20)
# rna[endo_idx_20].write('rna_test_endo_20.h5ad')

# atac = sc.read_h5ad('atac_test.h5ad')
# atac[endo_idx_20].write('atac_test_endo_20.h5ad')

# python data_preprocess.py -r rna_test_endo_20.h5ad -a atac_test_endo_20.h5ad -s test_pt_20 --dt test -n test_endo --config rna2atac_config_test.yaml

endo_data = torch.load("./test_pt_20/test_endo_0.pt")
dataset = PreDataset(endo_data)
dataloader_kwargs = {'batch_size': 1, 'shuffle': False}
loader = torch.utils.data.DataLoader(dataset, **dataloader_kwargs)

i=0
for inputs in tqdm(loader, ncols=80, desc='output attention matrix'):
    rna_sequence, rna_value, atac_sequence, _, enc_pad_mask = [each.to(device) for each in inputs]
    attn = model.generate_attn_weight(rna_sequence, atac_sequence, rna_value, enc_mask=enc_pad_mask, which='decoder')
    attn = attn[0]  # attn = attn[0].to(torch.float16)
    with h5py.File('attn/attn_endo_'+str(i)+'.h5', 'w') as f:
        f.create_dataset('attn', data=attn)
    torch.cuda.empty_cache()
    i += 1

rna_endo_20 = sc.read_h5ad('rna_test_endo_20.h5ad')
nonzero = np.count_nonzero(rna_endo_20.X.toarray(), axis=0)
gene_lst = rna_endo_20.var.index[nonzero>=3]  # 5437

atac_endo_20 = sc.read_h5ad('atac_test_endo_20.h5ad')

gene_attn = {}
for gene in gene_lst:
    gene_attn[gene] = np.zeros([atac_endo_20.shape[1], 1], dtype='float32')  # not float16

for i in range(20):
    rna_sequence = endo_data[0][i].flatten()
    with h5py.File('attn/attn_endo_'+str(i)+'.h5', 'r') as f:
        attn = f['attn'][:]
        for gene in tqdm(gene_lst, ncols=80, desc='cell '+str(i)):
            idx = torch.argwhere((rna_sequence==(np.argwhere(rna_endo_20.var.index==gene))[0][0]+1).flatten())
            if len(idx)!=0:
                gene_attn[gene] += attn[:, [idx.item()]]

with open('./attn/attn_20_endo_3cell.pkl', 'wb') as file:
    pickle.dump(gene_attn, file)

gene_attn_sum = [gene_attn[gene].sum() for gene in gene_lst]
df = pd.DataFrame({'gene':gene_lst, 'cnt':gene_attn_sum})
df.to_csv('./attn/attn_no_norm_cnt_20_endo_3cell.txt', sep='\t', header=None, index=None)

################################ Fibroblasts ################################
# rna  = sc.read_h5ad('rna_test.h5ad')
# random.seed(0)
# fibro_idx = list(np.argwhere(rna.obs['cell_anno']=='Fibroblasts').flatten())
# fibro_idx_20 = random.sample(list(fibro_idx), 20)
# rna[fibro_idx_20].write('rna_test_fibro_20.h5ad')

# atac = sc.read_h5ad('atac_test.h5ad')
# atac[fibro_idx_20].write('atac_test_fibro_20.h5ad')

# python data_preprocess.py -r rna_test_fibro_20.h5ad -a atac_test_fibro_20.h5ad -s test_pt_20 --dt test -n test_fibro --config rna2atac_config_test.yaml

fibro_data = torch.load("./test_pt_20/test_fibro_0.pt")
dataset = PreDataset(fibro_data)
dataloader_kwargs = {'batch_size': 1, 'shuffle': False}
loader = torch.utils.data.DataLoader(dataset, **dataloader_kwargs)

i=0
for inputs in tqdm(loader, ncols=80, desc='output attention matrix'):
    rna_sequence, rna_value, atac_sequence, _, enc_pad_mask = [each.to(device) for each in inputs]
    attn = model.generate_attn_weight(rna_sequence, atac_sequence, rna_value, enc_mask=enc_pad_mask, which='decoder')
    attn = attn[0]  # attn = attn[0].to(torch.float16)
    with h5py.File('attn/attn_fibro_'+str(i)+'.h5', 'w') as f:
        f.create_dataset('attn', data=attn)
    torch.cuda.empty_cache()
    i += 1

rna_fibro_20 = sc.read_h5ad('rna_test_fibro_20.h5ad')
nonzero = np.count_nonzero(rna_fibro_20.X.toarray(), axis=0)
gene_lst = rna_fibro_20.var.index[nonzero>=3]  # 6750

atac_fibro_20 = sc.read_h5ad('atac_test_fibro_20.h5ad')

gene_attn = {}
for gene in gene_lst:
    gene_attn[gene] = np.zeros([atac_fibro_20.shape[1], 1], dtype='float32')  # not float16

for i in range(20):
    rna_sequence = fibro_data[0][i].flatten()
    with h5py.File('attn/attn_fibro_'+str(i)+'.h5', 'r') as f:
        attn = f['attn'][:]
        for gene in tqdm(gene_lst, ncols=80, desc='cell '+str(i)):
            idx = torch.argwhere((rna_sequence==(np.argwhere(rna_fibro_20.var.index==gene))[0][0]+1).flatten())
            if len(idx)!=0:
                gene_attn[gene] += attn[:, [idx.item()]]

with open('./attn/attn_20_fibro_3cell.pkl', 'wb') as file:
    pickle.dump(gene_attn, file)

gene_attn_sum = [gene_attn[gene].sum() for gene in gene_lst]
df = pd.DataFrame({'gene':gene_lst, 'cnt':gene_attn_sum})
df.to_csv('./attn/attn_no_norm_cnt_20_fibro_3cell.txt', sep='\t', header=None, index=None)

################################ Macrophages ################################
# rna  = sc.read_h5ad('rna_test.h5ad')
# random.seed(0)
# macro_idx = list(np.argwhere(rna.obs['cell_anno']=='Macrophages').flatten())
# macro_idx_20 = random.sample(list(macro_idx), 20)
# rna[macro_idx_20].write('rna_test_macro_20.h5ad')

# atac = sc.read_h5ad('atac_test.h5ad')
# atac[macro_idx_20].write('atac_test_macro_20.h5ad')

# python data_preprocess.py -r rna_test_macro_20.h5ad -a atac_test_macro_20.h5ad -s test_pt_20 --dt test -n test_macro --config rna2atac_config_test.yaml

macro_data = torch.load("./test_pt_20/test_macro_0.pt")
dataset = PreDataset(macro_data)
dataloader_kwargs = {'batch_size': 1, 'shuffle': False}
loader = torch.utils.data.DataLoader(dataset, **dataloader_kwargs)

i=0
for inputs in tqdm(loader, ncols=80, desc='output attention matrix'):
    rna_sequence, rna_value, atac_sequence, _, enc_pad_mask = [each.to(device) for each in inputs]
    attn = model.generate_attn_weight(rna_sequence, atac_sequence, rna_value, enc_mask=enc_pad_mask, which='decoder')
    attn = attn[0]  # attn = attn[0].to(torch.float16)
    with h5py.File('attn/attn_macro_'+str(i)+'.h5', 'w') as f:
        f.create_dataset('attn', data=attn)
    torch.cuda.empty_cache()
    i += 1

rna_macro_20 = sc.read_h5ad('rna_test_macro_20.h5ad')
nonzero = np.count_nonzero(rna_macro_20.X.toarray(), axis=0)
gene_lst = rna_macro_20.var.index[nonzero>=3]  # 6750

atac_macro_20 = sc.read_h5ad('atac_test_macro_20.h5ad')

gene_attn = {}
for gene in gene_lst:
    gene_attn[gene] = np.zeros([atac_macro_20.shape[1], 1], dtype='float32')  # not float16

for i in range(20):
    rna_sequence = macro_data[0][i].flatten()
    with h5py.File('attn/attn_macro_'+str(i)+'.h5', 'r') as f:
        attn = f['attn'][:]
        for gene in tqdm(gene_lst, ncols=80, desc='cell '+str(i)):
            idx = torch.argwhere((rna_sequence==(np.argwhere(rna_macro_20.var.index==gene))[0][0]+1).flatten())
            if len(idx)!=0:
                gene_attn[gene] += attn[:, [idx.item()]]

with open('./attn/attn_20_macro_3cell.pkl', 'wb') as file:
    pickle.dump(gene_attn, file)

gene_attn_sum = [gene_attn[gene].sum() for gene in gene_lst]
df = pd.DataFrame({'gene':gene_lst, 'cnt':gene_attn_sum})
df.to_csv('./attn/attn_no_norm_cnt_20_macro_3cell.txt', sep='\t', header=None, index=None)

################################ T-cells ################################
# rna  = sc.read_h5ad('rna_test.h5ad')
# random.seed(0)
# tcell_idx = list(np.argwhere(rna.obs['cell_anno']=='T-cells').flatten())
# tcell_idx_20 = random.sample(list(tcell_idx), 20)
# rna[tcell_idx_20].write('rna_test_tcell_20.h5ad')

# atac = sc.read_h5ad('atac_test.h5ad')
# atac[tcell_idx_20].write('atac_test_tcell_20.h5ad')

# python data_preprocess.py -r rna_test_tcell_20.h5ad -a atac_test_tcell_20.h5ad -s test_pt_20 --dt test -n test_tcell --config rna2atac_config_test.yaml

tcell_data = torch.load("./test_pt_20/test_tcell_0.pt")
dataset = PreDataset(tcell_data)
dataloader_kwargs = {'batch_size': 1, 'shuffle': False}
loader = torch.utils.data.DataLoader(dataset, **dataloader_kwargs)

i=0
for inputs in tqdm(loader, ncols=80, desc='output attention matrix'):
    rna_sequence, rna_value, atac_sequence, _, enc_pad_mask = [each.to(device) for each in inputs]
    attn = model.generate_attn_weight(rna_sequence, atac_sequence, rna_value, enc_mask=enc_pad_mask, which='decoder')
    attn = attn[0]  # attn = attn[0].to(torch.float16)
    with h5py.File('attn/attn_tcell_'+str(i)+'.h5', 'w') as f:
        f.create_dataset('attn', data=attn)
    torch.cuda.empty_cache()
    i += 1

rna_tcell_20 = sc.read_h5ad('rna_test_tcell_20.h5ad')
nonzero = np.count_nonzero(rna_tcell_20.X.toarray(), axis=0)
gene_lst = rna_tcell_20.var.index[nonzero>=3]  # 6750

atac_tcell_20 = sc.read_h5ad('atac_test_tcell_20.h5ad')

gene_attn = {}
for gene in gene_lst:
    gene_attn[gene] = np.zeros([atac_tcell_20.shape[1], 1], dtype='float32')  # not float16

for i in range(20):
    rna_sequence = tcell_data[0][i].flatten()
    with h5py.File('attn/attn_tcell_'+str(i)+'.h5', 'r') as f:
        attn = f['attn'][:]
        for gene in tqdm(gene_lst, ncols=80, desc='cell '+str(i)):
            idx = torch.argwhere((rna_sequence==(np.argwhere(rna_tcell_20.var.index==gene))[0][0]+1).flatten())
            if len(idx)!=0:
                gene_attn[gene] += attn[:, [idx.item()]]

with open('./attn/attn_20_tcell_3cell.pkl', 'wb') as file:
    pickle.dump(gene_attn, file)

gene_attn_sum = [gene_attn[gene].sum() for gene in gene_lst]
df = pd.DataFrame({'gene':gene_lst, 'cnt':gene_attn_sum})
df.to_csv('./attn/attn_no_norm_cnt_20_tcell_3cell.txt', sep='\t', header=None, index=None)


## factor enrichment
import scanpy as sc
import pandas as pd
import numpy as np
from plotnine import *
from scipy import stats
import matplotlib.pyplot as plt

rna = sc.read_h5ad('../rna_test_bcell_20.h5ad')

# wget -c https://jaspar.elixir.no/download/data/2024/CORE/JASPAR2024_CORE_vertebrates_non-redundant_pfms_jaspar.txt
# grep ">" JASPAR2024_CORE_vertebrates_non-redundant_pfms_jaspar.txt | grep -v "::" | grep -v "-" | awk '{print toupper($2)}' | sort | uniq > TF_jaspar.txt

cr = ['SMARCA4', 'SMARCA2', 'ARID1A', 'ARID1B', 'SMARCB1',
      'CHD1', 'CHD2', 'CHD3', 'CHD4', 'CHD5', 'CHD6', 'CHD7', 'CHD8', 'CHD9',
      'BRD2', 'BRD3', 'BRD4', 'BRDT',
      'SMARCA5', 'SMARCA1', 'ACTL6A', 'ACTL6B',
      'SSRP1', 'SUPT16H',
      'EP400',
      'SMARCD1', 'SMARCD2', 'SMARCD3']

tf_lst = pd.read_table('TF_jaspar.txt', header=None)
tf = list(tf_lst[0].values)

## B cells
df = pd.read_csv('attn_no_norm_cnt_20_bcell_3cell.txt', header=None, sep='\t')
df.columns = ['gene', 'attn']
df = df.sort_values('attn', ascending=False)
df.index = range(df.shape[0])

df_cr = pd.DataFrame({'idx': 'CR', 'attn': df[df['gene'].isin(cr)]['attn'].values})               # 12
df_tf = pd.DataFrame({'idx': 'TF', 'attn': df[df['gene'].isin(tf)]['attn'].values})               # 108
df_others = pd.DataFrame({'idx': 'Others', 'attn': df[~df['gene'].isin(cr+tf)]['attn'].values})   # 3213
df_cr_tf_others = pd.concat([df_cr, df_tf, df_others])
df_cr_tf_others['idx'] = pd.Categorical(df_cr_tf_others['idx'], categories=['CR', 'TF', 'Others'])
df_cr_tf_others['Avg_attn'] = df_cr_tf_others['attn']/20  # 20 cells
df_cr_tf_others['Avg_attn_norm'] = np.log10(df_cr_tf_others['Avg_attn']/min(df_cr_tf_others['Avg_attn']))
df_cr_tf_others['Avg_attn_norm'] = df_cr_tf_others['Avg_attn_norm']/(df_cr_tf_others['Avg_attn_norm'].max())  # the same as min-max normalization

plt.rcParams['pdf.fonttype'] = 42
p = ggplot(df_cr_tf_others, aes(x='idx', y='Avg_attn_norm', fill='idx')) + geom_boxplot(width=0.5, show_legend=False, outlier_shape='') + xlab('') +\
                                                                           scale_y_continuous(limits=[0, 1], breaks=np.arange(0, 1+0.1, 0.2)) + theme_bw()
p.save(filename='cr_tf_others_box_cnt_20_bcell_3cell.pdf', dpi=300, height=4, width=4)

np.median(df_cr_tf_others[df_cr_tf_others['idx']=='CR']['Avg_attn_norm'])       # 0.6955349610735737
np.median(df_cr_tf_others[df_cr_tf_others['idx']=='TF']['Avg_attn_norm'])       # 0.6640555687558144
np.median(df_cr_tf_others[df_cr_tf_others['idx']=='Others']['Avg_attn_norm'])   # 0.6271017256046776

stats.ttest_ind(df_cr_tf_others[df_cr_tf_others['idx']=='CR']['Avg_attn_norm'],
                df_cr_tf_others[df_cr_tf_others['idx']=='Others']['Avg_attn_norm'])[1]  # 0.3200225406455772
stats.ttest_ind(df_cr_tf_others[df_cr_tf_others['idx']=='TF']['Avg_attn_norm'],
                df_cr_tf_others[df_cr_tf_others['idx']=='Others']['Avg_attn_norm'])[1]  # 0.02460240862081797
stats.ttest_ind(df_cr_tf_others[df_cr_tf_others['idx']=='CR']['Avg_attn_norm'],
                df_cr_tf_others[df_cr_tf_others['idx']=='TF']['Avg_attn_norm'])[1]      # 0.8168420631826141

stats.ttest_ind(df_cr_tf_others[df_cr_tf_others['idx']!='Others']['Avg_attn_norm'],
                df_cr_tf_others[df_cr_tf_others['idx']=='Others']['Avg_attn_norm'])[1]  # 0.014787446940820742

## Endothelial
df = pd.read_csv('attn_no_norm_cnt_20_endo_3cell.txt', header=None, sep='\t')
df.columns = ['gene', 'attn']
df = df.sort_values('attn', ascending=False)
df.index = range(df.shape[0])

df_cr = pd.DataFrame({'idx': 'CR', 'attn': df[df['gene'].isin(cr)]['attn'].values})               # 21
df_tf = pd.DataFrame({'idx': 'TF', 'attn': df[df['gene'].isin(tf)]['attn'].values})               # 181
df_others = pd.DataFrame({'idx': 'Others', 'attn': df[~df['gene'].isin(cr+tf)]['attn'].values})   # 5235
df_cr_tf_others = pd.concat([df_cr, df_tf, df_others])
df_cr_tf_others['idx'] = pd.Categorical(df_cr_tf_others['idx'], categories=['CR', 'TF', 'Others'])
df_cr_tf_others['Avg_attn'] =  df_cr_tf_others['attn']/20  # 20 cells
df_cr_tf_others['Avg_attn_norm'] = np.log10(df_cr_tf_others['Avg_attn']/min(df_cr_tf_others['Avg_attn']))
df_cr_tf_others['Avg_attn_norm'] = df_cr_tf_others['Avg_attn_norm']/(df_cr_tf_others['Avg_attn_norm'].max())  # the same as min-max normalization

plt.rcParams['pdf.fonttype'] = 42
p = ggplot(df_cr_tf_others, aes(x='idx', y='Avg_attn_norm', fill='idx')) + geom_boxplot(width=0.5, show_legend=False, outlier_shape='') + xlab('') +\
                                                                           scale_y_continuous(limits=[0, 1], breaks=np.arange(0, 1+0.1, 0.2)) + theme_bw()
p.save(filename='cr_tf_others_box_cnt_20_endo_3cell.pdf', dpi=300, height=4, width=4)

np.median(df_cr_tf_others[df_cr_tf_others['idx']=='CR']['Avg_attn_norm'])       # 0.7058555261215231
np.median(df_cr_tf_others[df_cr_tf_others['idx']=='TF']['Avg_attn_norm'])       # 0.5623256433322571
np.median(df_cr_tf_others[df_cr_tf_others['idx']=='Others']['Avg_attn_norm'])   # 0.5714279115236991

stats.ttest_ind(df_cr_tf_others[df_cr_tf_others['idx']=='CR']['Avg_attn_norm'],
                df_cr_tf_others[df_cr_tf_others['idx']=='Others']['Avg_attn_norm'])[1]  # 0.44836968765627316
stats.ttest_ind(df_cr_tf_others[df_cr_tf_others['idx']=='TF']['Avg_attn_norm'],
                df_cr_tf_others[df_cr_tf_others['idx']=='Others']['Avg_attn_norm'])[1]  # 0.661982133524154
stats.ttest_ind(df_cr_tf_others[df_cr_tf_others['idx']=='CR']['Avg_attn_norm'],
                df_cr_tf_others[df_cr_tf_others['idx']=='TF']['Avg_attn_norm'])[1]      # 0.5880467672943193

## Fibroblasts
df = pd.read_csv('attn_no_norm_cnt_20_fibro_3cell.txt', header=None, sep='\t')
df.columns = ['gene', 'attn']
df = df.sort_values('attn', ascending=False)
df.index = range(df.shape[0])

df_cr = pd.DataFrame({'idx': 'CR', 'attn': df[df['gene'].isin(cr)]['attn'].values})               # 19
df_tf = pd.DataFrame({'idx': 'TF', 'attn': df[df['gene'].isin(tf)]['attn'].values})               # 233
df_others = pd.DataFrame({'idx': 'Others', 'attn': df[~df['gene'].isin(cr+tf)]['attn'].values})   # 6498
df_cr_tf_others = pd.concat([df_cr, df_tf, df_others])
df_cr_tf_others['idx'] = pd.Categorical(df_cr_tf_others['idx'], categories=['CR', 'TF', 'Others'])
df_cr_tf_others['Avg_attn'] =  df_cr_tf_others['attn']/20  # 20 cells
df_cr_tf_others['Avg_attn_norm'] = np.log10(df_cr_tf_others['Avg_attn']/min(df_cr_tf_others['Avg_attn']))
df_cr_tf_others['Avg_attn_norm'] = df_cr_tf_others['Avg_attn_norm']/(df_cr_tf_others['Avg_attn_norm'].max())  # the same as min-max normalization

plt.rcParams['pdf.fonttype'] = 42
p = ggplot(df_cr_tf_others, aes(x='idx', y='Avg_attn_norm', fill='idx')) + geom_boxplot(width=0.5, show_legend=False, outlier_shape='') + xlab('') +\
                                                                           scale_y_continuous(limits=[0, 1], breaks=np.arange(0, 1+0.1, 0.2)) + theme_bw()
p.save(filename='cr_tf_others_box_cnt_20_fibro_3cell.pdf', dpi=300, height=4, width=4)

np.median(df_cr_tf_others[df_cr_tf_others['idx']=='CR']['Avg_attn_norm'])       # 0.6655984950690559
np.median(df_cr_tf_others[df_cr_tf_others['idx']=='TF']['Avg_attn_norm'])       # 0.565262057878596
np.median(df_cr_tf_others[df_cr_tf_others['idx']=='Others']['Avg_attn_norm'])   # 0.5331135601223107

stats.ttest_ind(df_cr_tf_others[df_cr_tf_others['idx']=='CR']['Avg_attn_norm'],
                df_cr_tf_others[df_cr_tf_others['idx']=='Others']['Avg_attn_norm'])[1]  # 0.4161597254044507
stats.ttest_ind(df_cr_tf_others[df_cr_tf_others['idx']=='TF']['Avg_attn_norm'],
                df_cr_tf_others[df_cr_tf_others['idx']=='Others']['Avg_attn_norm'])[1]  # 0.2156659261463783
stats.ttest_ind(df_cr_tf_others[df_cr_tf_others['idx']=='CR']['Avg_attn_norm'],
                df_cr_tf_others[df_cr_tf_others['idx']=='TF']['Avg_attn_norm'])[1]      # 0.6851610027662458

stats.ttest_ind(df_cr_tf_others[df_cr_tf_others['idx']!='Others']['Avg_attn_norm'],
                df_cr_tf_others[df_cr_tf_others['idx']=='Others']['Avg_attn_norm'])[1]  # 0.15954336841596595

## Macrophages
df = pd.read_csv('attn_no_norm_cnt_20_macro_3cell.txt', header=None, sep='\t')
df.columns = ['gene', 'attn']
df = df.sort_values('attn', ascending=False)
df.index = range(df.shape[0])

df_cr = pd.DataFrame({'idx': 'CR', 'attn': df[df['gene'].isin(cr)]['attn'].values})               # 16
df_tf = pd.DataFrame({'idx': 'TF', 'attn': df[df['gene'].isin(tf)]['attn'].values})               # 137
df_others = pd.DataFrame({'idx': 'Others', 'attn': df[~df['gene'].isin(cr+tf)]['attn'].values})   # 4083
df_cr_tf_others = pd.concat([df_cr, df_tf, df_others])
df_cr_tf_others['idx'] = pd.Categorical(df_cr_tf_others['idx'], categories=['CR', 'TF', 'Others'])
df_cr_tf_others['Avg_attn'] =  df_cr_tf_others['attn']/20  # 20 cells
df_cr_tf_others['Avg_attn_norm'] = np.log10(df_cr_tf_others['Avg_attn']/min(df_cr_tf_others['Avg_attn']))
df_cr_tf_others['Avg_attn_norm'] = df_cr_tf_others['Avg_attn_norm']/(df_cr_tf_others['Avg_attn_norm'].max())  # the same as min-max normalization

plt.rcParams['pdf.fonttype'] = 42
p = ggplot(df_cr_tf_others, aes(x='idx', y='Avg_attn_norm', fill='idx')) + geom_boxplot(width=0.5, show_legend=False, outlier_shape='') + xlab('') +\
                                                                           scale_y_continuous(limits=[0, 1], breaks=np.arange(0, 1+0.1, 0.2)) + theme_bw()
p.save(filename='cr_tf_others_box_cnt_20_macro_3cell.pdf', dpi=300, height=4, width=4)

np.median(df_cr_tf_others[df_cr_tf_others['idx']=='CR']['Avg_attn_norm'])       # 0.7665129389451164
np.median(df_cr_tf_others[df_cr_tf_others['idx']=='TF']['Avg_attn_norm'])       # 0.6918338217334038
np.median(df_cr_tf_others[df_cr_tf_others['idx']=='Others']['Avg_attn_norm'])   # 0.6315503715085543

stats.ttest_ind(df_cr_tf_others[df_cr_tf_others['idx']=='CR']['Avg_attn_norm'],
                df_cr_tf_others[df_cr_tf_others['idx']=='Others']['Avg_attn_norm'])[1]  # 0.4005924770661283
stats.ttest_ind(df_cr_tf_others[df_cr_tf_others['idx']=='TF']['Avg_attn_norm'],
                df_cr_tf_others[df_cr_tf_others['idx']=='Others']['Avg_attn_norm'])[1]  # 0.03105355289359248
stats.ttest_ind(df_cr_tf_others[df_cr_tf_others['idx']=='CR']['Avg_attn_norm'],
                df_cr_tf_others[df_cr_tf_others['idx']=='TF']['Avg_attn_norm'])[1]      # 0.9334925984335126

stats.ttest_ind(df_cr_tf_others[df_cr_tf_others['idx']!='Others']['Avg_attn_norm'],
                df_cr_tf_others[df_cr_tf_others['idx']=='Others']['Avg_attn_norm'])[1]  # 0.021326851530636147

## T-cells
df = pd.read_csv('attn_no_norm_cnt_20_tcell_3cell.txt', header=None, sep='\t')
df.columns = ['gene', 'attn']
df = df.sort_values('attn', ascending=False)
df.index = range(df.shape[0])

df_cr = pd.DataFrame({'idx': 'CR', 'attn': df[df['gene'].isin(cr)]['attn'].values})               # 15
df_tf = pd.DataFrame({'idx': 'TF', 'attn': df[df['gene'].isin(tf)]['attn'].values})               # 105
df_others = pd.DataFrame({'idx': 'Others', 'attn': df[~df['gene'].isin(cr+tf)]['attn'].values})   # 3105
df_cr_tf_others = pd.concat([df_cr, df_tf, df_others])
df_cr_tf_others['idx'] = pd.Categorical(df_cr_tf_others['idx'], categories=['CR', 'TF', 'Others'])
df_cr_tf_others['Avg_attn'] =  df_cr_tf_others['attn']/20  # 20 cells
df_cr_tf_others['Avg_attn_norm'] = np.log10(df_cr_tf_others['Avg_attn']/min(df_cr_tf_others['Avg_attn']))
df_cr_tf_others['Avg_attn_norm'] = df_cr_tf_others['Avg_attn_norm']/(df_cr_tf_others['Avg_attn_norm'].max())  # the same as min-max normalization

plt.rcParams['pdf.fonttype'] = 42
p = ggplot(df_cr_tf_others, aes(x='idx', y='Avg_attn_norm', fill='idx')) + geom_boxplot(width=0.5, show_legend=False, outlier_shape='') + xlab('') +\
                                                                           scale_y_continuous(limits=[0, 1], breaks=np.arange(0, 1+0.1, 0.2)) + theme_bw()
p.save(filename='cr_tf_others_box_cnt_20_tcell_3cell.pdf', dpi=300, height=4, width=4)

np.median(df_cr_tf_others[df_cr_tf_others['idx']=='CR']['Avg_attn_norm'])       # 0.6866661752571018
np.median(df_cr_tf_others[df_cr_tf_others['idx']=='TF']['Avg_attn_norm'])       # 0.6766970426257376
np.median(df_cr_tf_others[df_cr_tf_others['idx']=='Others']['Avg_attn_norm'])   # 0.6453717776836605

stats.ttest_ind(df_cr_tf_others[df_cr_tf_others['idx']=='CR']['Avg_attn_norm'],
                df_cr_tf_others[df_cr_tf_others['idx']=='Others']['Avg_attn_norm'])[1]  # 0.7907526703266188
stats.ttest_ind(df_cr_tf_others[df_cr_tf_others['idx']=='TF']['Avg_attn_norm'],
                df_cr_tf_others[df_cr_tf_others['idx']=='Others']['Avg_attn_norm'])[1]  # 0.05588771897861935
stats.ttest_ind(df_cr_tf_others[df_cr_tf_others['idx']=='CR']['Avg_attn_norm'],
                df_cr_tf_others[df_cr_tf_others['idx']=='TF']['Avg_attn_norm'])[1]      # 0.6684072216866288

stats.ttest_ind(df_cr_tf_others[df_cr_tf_others['idx']!='Others']['Avg_attn_norm'],
                df_cr_tf_others[df_cr_tf_others['idx']=='Others']['Avg_attn_norm'])[1]  # 0.060765003505965906


#### specific tf families
# tf_family is the subclass of tf_class
# wget -c https://jaspar.elixir.no/download/data/2024/CORE/JASPAR2024_CORE_non-redundant_pfms_transfac.txt
# grep "ID " JASPAR2024_CORE_non-redundant_pfms_transfac.txt | sed 's/ID //g' | awk '{print toupper($0)}' > tf_ref_id.txt
# grep "tf_family" JASPAR2024_CORE_non-redundant_pfms_transfac.txt | sed 's/CC tf_family://g' > tf_ref_family.txt
# grep "tf_class" JASPAR2024_CORE_non-redundant_pfms_transfac.txt | sed 's/CC tf_class://g' > tf_ref_class.txt
# grep "CC tax_group" JASPAR2024_CORE_non-redundant_pfms_transfac.txt | sed 's/CC tax_group://g' > tf_ref_tax.txt
# tf_class: 65
# tf_family: 161

import scanpy as sc
import pandas as pd
import numpy as np
from plotnine import *
from scipy import stats
import matplotlib.pyplot as plt

tf = pd.read_table('TF_jaspar.txt', header=None)
tf.rename(columns={0: 'id'}, inplace=True)

tf_ref_id = pd.read_table('tf_ref_id.txt', header=None)
tf_ref_id.rename(columns={0: 'id'}, inplace=True)

tf_ref_class = pd.read_table('tf_ref_class.txt', header=None, skip_blank_lines=False)
tf_ref_class.rename(columns={0: 'class'}, inplace=True)

tf_ref_family = pd.read_table('tf_ref_family.txt', header=None, skip_blank_lines=False)
tf_ref_family.rename(columns={0: 'family'}, inplace=True)

tf_ref_tax = pd.read_table('tf_ref_tax.txt', header=None)
tf_ref_tax.rename(columns={0: 'tax'}, inplace=True)

tf_ref = pd.concat([tf_ref_tax, tf_ref_id, tf_ref_class, tf_ref_family], axis=1)
tf_ref = tf_ref[tf_ref['tax']=='vertebrates'].drop_duplicates().dropna()
tf_dict = pd.merge(tf, tf_ref, how='left', on='id')[['id', 'class', 'family']]
tf_dict.columns = ['gene', 'class', 'family']

## B cells
# E2A & EBF & PAX & IRF
# b_cell_tf_family = ['E2A',
#                     'Early B-Cell Factor-related factors',
#                     'Paired domain only', 'Paired plus homeo domain',
#                     'Runt-related factors',
#                     'Interferon-regulatory factors']
# not all tfs in the same family are all specific

df_attn = pd.read_csv('attn_no_norm_cnt_20_bcell_3cell.txt', header=None, sep='\t')
df_attn.columns = ['gene', 'attn']

df_attn['Avg_attn'] = df_attn['attn']/20  # 20 cells
df_attn['Avg_attn_norm'] = np.log10(df_attn['Avg_attn']/min(df_attn['Avg_attn']))
df_attn['Avg_attn_norm'] = df_attn['Avg_attn_norm']/(df_attn['Avg_attn_norm'].max())

tf_attn = pd.merge(tf_dict, df_attn)

tf_class = tf_dict['class'].value_counts().index.values[:7]  # top 7 count tf

df = pd.DataFrame()
for i in tf_class:
    df = pd.concat([df, pd.DataFrame({'idx':i, 'val':tf_attn[tf_attn['class']==i]['Avg_attn_norm'].values})])

df['idx'] = pd.Categorical(df['idx'], categories=df.groupby('idx')['val'].median().sort_values(ascending=False).index)

plt.rcParams['pdf.fonttype'] = 42
p = ggplot(df, aes(x='idx', y='val', fill='idx')) + geom_boxplot(width=0.5, show_legend=False, outlier_shape='') + xlab('') +\
                                                    scale_y_continuous(limits=[0, 1], breaks=np.arange(0, 1+0.1, 0.2)) +\
                                                    theme_bw() + theme(axis_text_x=element_text(angle=45, hjust=1))
p.save(filename='tf_class_box_cnt_20_bcell_3cell.pdf', dpi=300, height=4, width=4)


## endothelial
df_attn = pd.read_csv('attn_no_norm_cnt_20_endo_3cell.txt', header=None, sep='\t')
df_attn.columns = ['gene', 'attn']

df_attn['Avg_attn'] = df_attn['attn']/20  # 20 cells
df_attn['Avg_attn_norm'] = np.log10(df_attn['Avg_attn']/min(df_attn['Avg_attn']))
df_attn['Avg_attn_norm'] = df_attn['Avg_attn_norm']/(df_attn['Avg_attn_norm'].max())

tf_attn = pd.merge(tf_dict, df_attn)

tf_class = tf_dict['class'].value_counts().index.values[:7]  # top 7 count tf

df = pd.DataFrame()
for i in tf_class:
    df = pd.concat([df, pd.DataFrame({'idx':i, 'val':tf_attn[tf_attn['class']==i]['Avg_attn_norm'].values})])

df['idx'] = pd.Categorical(df['idx'], categories=df.groupby('idx')['val'].median().sort_values(ascending=False).index)

plt.rcParams['pdf.fonttype'] = 42
p = ggplot(df, aes(x='idx', y='val', fill='idx')) + geom_boxplot(width=0.5, show_legend=False, outlier_shape='') + xlab('') +\
                                                    scale_y_continuous(limits=[0, 1], breaks=np.arange(0, 1+0.1, 0.2)) +\
                                                    theme_bw() + theme(axis_text_x=element_text(angle=45, hjust=1))
p.save(filename='tf_class_box_cnt_20_endo_3cell.pdf', dpi=300, height=4, width=4)

## fibroblast
df_attn = pd.read_csv('attn_no_norm_cnt_20_fibro_3cell.txt', header=None, sep='\t')
df_attn.columns = ['gene', 'attn']

df_attn['Avg_attn'] = df_attn['attn']/20  # 20 cells
df_attn['Avg_attn_norm'] = np.log10(df_attn['Avg_attn']/min(df_attn['Avg_attn']))
df_attn['Avg_attn_norm'] = df_attn['Avg_attn_norm']/(df_attn['Avg_attn_norm'].max())

tf_attn = pd.merge(tf_dict, df_attn)

tf_class = tf_dict['class'].value_counts().index.values[:7]  # top 7 count tf

df = pd.DataFrame()
for i in tf_class:
    df = pd.concat([df, pd.DataFrame({'idx':i, 'val':tf_attn[tf_attn['class']==i]['Avg_attn_norm'].values})])

df['idx'] = pd.Categorical(df['idx'], categories=df.groupby('idx')['val'].median().sort_values(ascending=False).index)

plt.rcParams['pdf.fonttype'] = 42
p = ggplot(df, aes(x='idx', y='val', fill='idx')) + geom_boxplot(width=0.5, show_legend=False, outlier_shape='') + xlab('') +\
                                                    scale_y_continuous(limits=[0, 1], breaks=np.arange(0, 1+0.1, 0.2)) +\
                                                    theme_bw() + theme(axis_text_x=element_text(angle=45, hjust=1))
p.save(filename='tf_class_box_cnt_20_fibro_3cell.pdf', dpi=300, height=4, width=4)

## macrophages
df_attn = pd.read_csv('attn_no_norm_cnt_20_macro_3cell.txt', header=None, sep='\t')
df_attn.columns = ['gene', 'attn']

df_attn['Avg_attn'] = df_attn['attn']/20  # 20 cells
df_attn['Avg_attn_norm'] = np.log10(df_attn['Avg_attn']/min(df_attn['Avg_attn']))
df_attn['Avg_attn_norm'] = df_attn['Avg_attn_norm']/(df_attn['Avg_attn_norm'].max())

tf_attn = pd.merge(tf_dict, df_attn)

tf_class = tf_dict['class'].value_counts().index.values[:7]  # top 7 count tf

df = pd.DataFrame()
for i in tf_class:
    df = pd.concat([df, pd.DataFrame({'idx':i, 'val':tf_attn[tf_attn['class']==i]['Avg_attn_norm'].values})])

df['idx'] = pd.Categorical(df['idx'], categories=df.groupby('idx')['val'].median().sort_values(ascending=False).index)

plt.rcParams['pdf.fonttype'] = 42
p = ggplot(df, aes(x='idx', y='val', fill='idx')) + geom_boxplot(width=0.5, show_legend=False, outlier_shape='') + xlab('') +\
                                                    scale_y_continuous(limits=[0, 1], breaks=np.arange(0, 1+0.1, 0.2)) +\
                                                    theme_bw() + theme(axis_text_x=element_text(angle=45, hjust=1))
p.save(filename='tf_class_box_cnt_20_macro_3cell.pdf', dpi=300, height=4, width=4)

## T cells
df_attn = pd.read_csv('attn_no_norm_cnt_20_tcell_3cell.txt', header=None, sep='\t')
df_attn.columns = ['gene', 'attn']

df_attn['Avg_attn'] = df_attn['attn']/20  # 20 cells
df_attn['Avg_attn_norm'] = np.log10(df_attn['Avg_attn']/min(df_attn['Avg_attn']))
df_attn['Avg_attn_norm'] = df_attn['Avg_attn_norm']/(df_attn['Avg_attn_norm'].max())

tf_attn = pd.merge(tf_dict, df_attn)

tf_class = tf_dict['class'].value_counts().index.values[:7]  # top 7 count tf

df = pd.DataFrame()
for i in tf_class:
    df = pd.concat([df, pd.DataFrame({'idx':i, 'val':tf_attn[tf_attn['class']==i]['Avg_attn_norm'].values})])

df['idx'] = pd.Categorical(df['idx'], categories=df.groupby('idx')['val'].median().sort_values(ascending=False).index)

plt.rcParams['pdf.fonttype'] = 42
p = ggplot(df, aes(x='idx', y='val', fill='idx')) + geom_boxplot(width=0.5, show_legend=False, outlier_shape='') + xlab('') +\
                                                    scale_y_continuous(limits=[0, 1], breaks=np.arange(0, 1+0.1, 0.2)) +\
                                                    theme_bw() + theme(axis_text_x=element_text(angle=45, hjust=1))
p.save(filename='tf_class_box_cnt_20_tcell_3cell.pdf', dpi=300, height=4, width=4)


###### rank genes based on attention score
import pandas as pd
import numpy as np
from plotnine import *
import matplotlib.pyplot as plt

def plot_output_top(ct='bcell'):
    df = pd.read_csv('attn_no_norm_cnt_20_'+ct+'_3cell.txt', header=None, sep='\t')
    df.columns = ['gene', 'attn']
    df = df.sort_values('attn', ascending=False)
    df.index = range(df.shape[0])
    df['Avg_attn'] =  df['attn']/20
    df['Avg_attn_norm'] = np.log10(df['Avg_attn']/min(df['Avg_attn']))
    df['Avg_attn_norm'] = df['Avg_attn_norm']/(df['Avg_attn_norm'].max())
    df_top10 = df[:10]
    gene_top10 = list(df_top10['gene'])
    gene_top10.reverse()
    df_top10['gene'] = pd.Categorical(df_top10['gene'], categories=gene_top10)

    plt.rcParams['pdf.fonttype'] = 42
    p = ggplot(df_top10) + aes(x='gene', y='Avg_attn_norm') + geom_segment(aes(x='gene', xend='gene', y=0.95, yend='Avg_attn_norm'), color='red', size=1) + \
                                                              geom_point(color='red', size=3) + coord_flip() + \
                                                              xlab('Genes') + ylab('Attn') + labs(title=ct) + \
                                                              scale_y_continuous(limits=[0.95, 1], breaks=np.arange(0.95, 1+0.01, 0.01)) + \
                                                              theme_bw() + theme(plot_title=element_text(hjust=0.5)) 
    p.save(filename='top10_lollipp_'+ct+'.pdf', dpi=300, height=4, width=5)
    
    df[:100]['gene'].to_csv('gene_top100_'+ct+'.txt', header=None, index=None)

plot_output_top(ct='bcell')
plot_output_top(ct='endo')
plot_output_top(ct='fibro')
plot_output_top(ct='macro')
plot_output_top(ct='tcell')

cr = ['SMARCA4', 'SMARCA2', 'ARID1A', 'ARID1B', 'SMARCB1',
      'CHD1', 'CHD2', 'CHD3', 'CHD4', 'CHD5', 'CHD6', 'CHD7', 'CHD8', 'CHD9',
      'BRD2', 'BRD3', 'BRD4', 'BRDT',
      'SMARCA5', 'SMARCA1', 'ACTL6A', 'ACTL6B',
      'SSRP1', 'SUPT16H',
      'EP400',
      'SMARCD1', 'SMARCD2', 'SMARCD3']   # 28
tf_lst = pd.read_table('TF_jaspar.txt', header=None)
tf = list(tf_lst[0].values)   # 735
cr_tf = cr + tf   # 763

def ext_cr_tf(ct='bcell'):
    df = pd.read_csv('attn_no_norm_cnt_20_'+ct+'_3cell.txt', header=None, sep='\t')
    df.columns = ['gene', 'attn']
    df = df.sort_values('attn', ascending=False)
    df.index = range(df.shape[0])
    df['Avg_attn'] =  df['attn']/20
    df['Avg_attn_norm'] = np.log10(df['Avg_attn']/min(df['Avg_attn']))
    df['Avg_attn_norm'] = df['Avg_attn_norm']/(df['Avg_attn_norm'].max())
    df = df[df['gene'].isin(cr_tf)]
    df.index = range(df.shape[0])
    del df['attn']
    del df['Avg_attn']
    del df['Avg_attn_norm']
    #df[ct+'_rank'] = list(reversed(range(1, df.shape[0]+1)))
    df[ct+'_rank'] = range(1, df.shape[0]+1)
    return df

bcell = ext_cr_tf(ct='bcell')
endo = ext_cr_tf(ct='endo')
fibro = ext_cr_tf(ct='fibro')
macro = ext_cr_tf(ct='macro')
tcell = ext_cr_tf(ct='tcell')

bcell_endo = pd.merge(bcell, endo, how='outer')
fibro_macro = pd.merge(fibro, macro, how='outer')
tcell_not = pd.merge(bcell_endo, fibro_macro, how='outer')
cells = pd.merge(tcell_not, tcell, how='outer')
cells.fillna(500, inplace=True)

bcell_top100_lst = list(cells.sort_values('bcell_rank')['gene'])[:100]
endo_top100_lst = list(cells.sort_values('endo_rank')['gene'])[:100]
fibro_top100_lst = list(cells.sort_values('fibro_rank')['gene'])[:100]
macro_top100_lst = list(cells.sort_values('macro_rank')['gene'])[:100]
tcell_top100_lst = list(cells.sort_values('tcell_rank')['gene'])[:100]

bcell_spe_lst = [i for i in bcell_top100_lst if i not in (endo_top100_lst+fibro_top100_lst+macro_top100_lst+tcell_top100_lst)]   # 13
endo_spe_lst = [i for i in endo_top100_lst if i not in (bcell_top100_lst+fibro_top100_lst+macro_top100_lst+tcell_top100_lst)]    # 10
fibro_spe_lst = [i for i in fibro_top100_lst if i not in (bcell_top100_lst+endo_top100_lst+macro_top100_lst+tcell_top100_lst)]   # 18
macro_spe_lst = [i for i in macro_top100_lst if i not in (bcell_top100_lst+endo_top100_lst+fibro_top100_lst+tcell_top100_lst)]   # 14
tcell_spe_lst = [i for i in tcell_top100_lst if i not in (bcell_top100_lst+endo_top100_lst+fibro_top100_lst+macro_top100_lst)]   # 14

common_lst = [i for i in bcell_top100_lst if ((i in endo_top100_lst) & (i in fibro_top100_lst) & (i in macro_top100_lst) & (i in tcell_top100_lst))]  # 28

lst = bcell_spe_lst + endo_spe_lst + fibro_spe_lst + macro_spe_lst + tcell_spe_lst  # exclude common factors
cells.index = cells['gene'].values
df = cells.loc[lst]

import seaborn as sns

plt.rcParams['pdf.fonttype'] = 42
sns.clustermap(1/(df.iloc[:, 1:]), cmap='viridis', row_cluster=False, col_cluster=False, z_score=0, vmin=-1, vmax=1, figsize=(5, 20)) 
plt.savefig('cell_specific_factor_heatmap.pdf')
plt.close()



# sns.clustermap(df, cmap='RdBu_r', z_score=0, vmin=-4, vmax=4, row_cluster=False, col_cluster=False, yticklabels=False) 
# #plt.savefig('marker_peak_heatmap_true_true.pdf')
# sns.clustermap(np.log10(df*1000000+1), row_cluster=False, col_cluster=False, yticklabels=False) 
# plt.savefig('tmp.pdf')
# plt.close()


# def plot_output_top_cr_tf(ct='bcell'):
#     df = pd.read_csv('attn_no_norm_cnt_20_'+ct+'_3cell.txt', header=None, sep='\t')
#     df.columns = ['gene', 'attn']
#     df = df.sort_values('attn', ascending=False)
#     df.index = range(df.shape[0])
#     df['Avg_attn'] =  df['attn']/20
#     df['Avg_attn_norm'] = np.log10(df['Avg_attn']/min(df['Avg_attn']))
#     df['Avg_attn_norm'] = df['Avg_attn_norm']/(df['Avg_attn_norm'].max())
#     df_top10 = df[df['gene'].isin(cr_tf)][:10]
#     gene_top10 = list(df_top10['gene'])
#     gene_top10.reverse()
#     df_top10['gene'] = pd.Categorical(df_top10['gene'], categories=gene_top10)

#     plt.rcParams['pdf.fonttype'] = 42
#     p = ggplot(df_top10) + aes(x='gene', y='Avg_attn_norm') + geom_segment(aes(x='gene', xend='gene', y=0.95, yend='Avg_attn_norm'), color='red', size=1) + \
#                                                               geom_point(color='red', size=3) + coord_flip() + \
#                                                               xlab('Genes') + ylab('Attn') + labs(title=ct) + \
#                                                               scale_y_continuous(limits=[0.95, 1], breaks=np.arange(0.95, 1+0.01, 0.01)) + \
#                                                               theme_bw() + theme(plot_title=element_text(hjust=0.5)) 
#     p.save(filename='top10_lollipp_'+ct+'.pdf', dpi=300, height=4, width=5)
    
#     df[:100]['gene'].to_csv('gene_top100_'+ct+'.txt', header=None, index=None)

# plot_output_top_cr_tf(ct='bcell')
# plot_output_top_cr_tf(ct='endo')
# plot_output_top_cr_tf(ct='fibro')
# plot_output_top_cr_tf(ct='macro')
# plot_output_top_cr_tf(ct='tcell')


###### rank all genes based on attention score
import pandas as pd
import numpy as np
from plotnine import *
import matplotlib.pyplot as plt
import seaborn as sns

cr = ['SMARCA4', 'SMARCA2', 'ARID1A', 'ARID1B', 'SMARCB1',
      'CHD1', 'CHD2', 'CHD3', 'CHD4', 'CHD5', 'CHD6', 'CHD7', 'CHD8', 'CHD9',
      'BRD2', 'BRD3', 'BRD4', 'BRDT',
      'SMARCA5', 'SMARCA1', 'ACTL6A', 'ACTL6B',
      'SSRP1', 'SUPT16H',
      'EP400',
      'SMARCD1', 'SMARCD2', 'SMARCD3']   # 28
tf_lst = pd.read_table('TF_jaspar.txt', header=None)
tf = list(tf_lst[0].values)   # 735
cr_tf = cr + tf   # 763

def out_norm_attn(ct='bcell'):
    df = pd.read_csv('attn_no_norm_cnt_20_'+ct+'_3cell.txt', header=None, sep='\t')
    df.columns = ['gene', 'attn']
    df = df.sort_values('attn', ascending=False)
    df.index = range(df.shape[0])
    df['Avg_attn'] =  df['attn']/20
    df['Avg_attn_norm'] = np.log10(df['Avg_attn']/min(df['Avg_attn']))
    df['Avg_attn_norm'] = df['Avg_attn_norm']/(df['Avg_attn_norm'].max())
    df = df[df['gene'].isin(cr_tf)][['gene', 'Avg_attn_norm']]
    df.columns = ['gene', ct+'_avg_attn_norm']
    return df
    
bcell = out_norm_attn(ct='bcell')
endo = out_norm_attn(ct='endo')
fibro = out_norm_attn(ct='fibro')
macro = out_norm_attn(ct='macro')
tcell = out_norm_attn(ct='tcell')

bcell_endo = pd.merge(bcell, endo, how='outer')
fibro_macro = pd.merge(fibro, macro, how='outer')
tcell_not = pd.merge(bcell_endo, fibro_macro, how='outer')
cells = pd.merge(tcell_not, tcell, how='outer')
cells.fillna(0, inplace=True)

# cells[(cells['bcell_avg_attn_norm']>cells['endo_avg_attn_norm']) &
#       (cells['bcell_avg_attn_norm']>cells['fibro_avg_attn_norm']) &
#       (cells['bcell_avg_attn_norm']>cells['macro_avg_attn_norm']) &
#       (cells['bcell_avg_attn_norm']>cells['tcell_avg_attn_norm'])]

plt.rcParams['pdf.fonttype'] = 42
sns.clustermap(cells.iloc[:, 1:], cmap='viridis', row_cluster=True, col_cluster=True, figsize=(5, 20)) 
plt.savefig('tf_cr_attn_heatmap.pdf')
plt.close()
