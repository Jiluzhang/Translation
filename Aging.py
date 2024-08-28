## mouse kidney
## https://cf.10xgenomics.com/samples/cell-arc/2.0.2/M_Kidney_Chromium_Nuc_Isolation_vs_SaltyEZ_vs_ComplexTissueDP/\
##                            M_Kidney_Chromium_Nuc_Isolation_vs_SaltyEZ_vs_ComplexTissueDP_filtered_feature_bc_matrix.h5
## kidney.h5

import scanpy as sc

dat = sc.read_10x_h5('kidney.h5', gex_only=False)                  # 14652 × 97637
dat.var_names_make_unique()

rna = dat[:, dat.var['feature_types']=='Gene Expression'].copy()   # 14652 × 32285
rna.write('rna.h5ad')

atac = dat[:, dat.var['feature_types']=='Peaks'].copy()            # 14652 × 65352
atac.X[atac.X!=0] = 1
atac.write('atac.h5ad')


python split_train_val.py --RNA rna.h5ad --ATAC atac.h5ad --train_pct 0.9
python data_preprocess.py -r rna_train.h5ad -a atac_train.h5ad -s preprocessed_data_train --dt train --config rna2atac_config_train.yaml
python data_preprocess.py -r rna_val.h5ad -a atac_val.h5ad -s preprocessed_data_val --dt val --config rna2atac_config_val.yaml
nohup accelerate launch --config_file accelerator_config_train.yaml --main_process_port 29823 rna2atac_train.py --config_file rna2atac_config_train.yaml \
                        --train_data_dir ./preprocessed_data_train --val_data_dir ./preprocessed_data_val -n rna2atac_train > 20240821.log &   
# accuracy improves until about 10 epoch
# 3199536


import scanpy as sc
import numpy as np
rna = sc.read_h5ad('rna_val.h5ad')
np.count_nonzero(rna.X.toarray(), axis=1).max()  # 13845

import snapatac2 as snap
import scanpy as sc
atac = sc.read_h5ad('atac_val.h5ad')
snap.pp.select_features(atac)
snap.tl.spectral(atac)
snap.tl.umap(atac)
snap.pp.knn(atac)
snap.tl.leiden(atac)
sc.pl.umap(atac, color='leiden', legend_fontsize='7', legend_loc='right margin', size=5,
           title='', frameon=True, save='_atac_val_true.pdf')
atac.write('atac_val_with_leiden.h5ad')


python data_preprocess.py -r rna_val.h5ad -a atac_val.h5ad -s preprocessed_data_test --dt test --config rna2atac_config_test.yaml
accelerate launch --config_file accelerator_config_test.yaml --main_process_port 29822 rna2atac_test.py \
                  -d ./preprocessed_data_test \
                  -l save/2024-08-21_rna2atac_train_56/pytorch_model.bin --config_file rna2atac_config_test.yaml
python npy2h5ad.py
python csr2array.py --pred rna2atac_scm2m_raw.h5ad --true atac_val_with_leiden.h5ad
python cal_auroc_auprc.py --pred rna2atac_scm2m.h5ad --true atac_val_with_leiden.h5ad
python cal_cluster_plot.py --pred rna2atac_scm2m.h5ad --true atac_val_with_leiden.h5ad


## Tabula Muris Senis: https://cellxgene.cziscience.com/collections/0b9d8a04-bb9d-44da-aa27-705bb65b54eb
# kidney dataset: wget -c https://datasets.cellxgene.cziscience.com/e896f2b3-e22b-42c0-81e8-9dd1b6e80d64.h5ad
# -> kidney_aging.h5ad

import scanpy as sc
import pandas as pd
from scipy.sparse import csr_matrix
import numpy as np

rna = sc.read_h5ad('kidney_aging.h5ad')

## map rna
ref_rna = sc.read_h5ad('rna.h5ad')
genes = ref_rna.var[['gene_ids']]
genes['gene_name'] = genes.index.values
genes.reset_index(drop=True, inplace=True)

rna_exp = pd.concat([rna.var['feature_name'], pd.DataFrame(rna.raw.X.toarray().T, index=rna.var.index.values)], axis=1)
rna_exp['gene_ids'] = rna_exp.index.values
rna_exp.rename(columns={'feature_name': 'gene_name'}, inplace=True)
X_new = pd.merge(genes, rna_exp, how='left', on='gene_ids').iloc[:, 3:].T
X_new.fillna(value=0, inplace=True)

rna_new = sc.AnnData(X_new.values, obs=rna.obs, var=pd.DataFrame({'gene_ids': genes['gene_ids'], 'feature_types': 'Gene Expression'}))
rna_new.var.index = genes['gene_name'].values
rna_new.X = csr_matrix(rna_new.X)
rna_new.write('rna_unpaired.h5ad')

## pseudo atac
ref_atac = sc.read_h5ad('atac.h5ad')

atac_new = sc.AnnData(np.zeros([rna.n_obs, ref_atac.n_vars]),
                      obs=rna.obs, 
                      var=ref_atac.var)
atac_new.X[:, 0] = 1
atac_new.X = csr_matrix(atac_new.X)
atac_new.write('atac_unpaired.h5ad')


import scanpy as sc
import numpy as np
rna = sc.read_h5ad('rna_unpaired.h5ad')
np.count_nonzero(rna.X.toarray(), axis=1).max()  # 7617



python data_preprocess_unpaired.py -r rna_unpaired.h5ad -a atac_unpaired.h5ad -s preprocessed_data_unpaired --dt test --config rna2atac_config_unpaired.yaml
accelerate launch --config_file accelerator_config_test.yaml --main_process_port 29822 rna2atac_test.py \
                  -d ./preprocessed_data_unpaired \
                  -l save/2024-08-21_rna2atac_train_55/pytorch_model.bin --config_file rna2atac_config_unpaired.yaml   # ~ 0.5h

import scanpy as sc
import numpy as np
from scipy.sparse import csr_matrix
import snapatac2 as snap
import pandas as pd
from tqdm import tqdm

dat = np.load('predict_unpaired.npy')
dat[dat>0.5] = 1
dat[dat<=0.5] = 0
atac = sc.read_h5ad('atac_unpaired.h5ad')
atac.X = csr_matrix(dat)

snap.pp.select_features(atac)
snap.tl.spectral(atac, n_comps=100)  # snap.tl.spectral(atac, n_comps=100, weighted_by_sd=False)
snap.tl.umap(atac)

# sc.pl.umap(atac[atac.obs['cell_type'].isin(atac.obs.cell_type.value_counts()[:5].index), :], color='cell_type', legend_fontsize='7', legend_loc='right margin', size=10,
#            title='', frameon=True, save='_unpaired_tmp.pdf')

sc.pl.umap(atac[atac.obs['cell_type'].isin(['fenestrated cell', 'B cell']), :], color='cell_type', legend_fontsize='7', legend_loc='right margin', size=10,
            title='', frameon=True, save='_unpaired_luz.pdf')

sc.pl.umap(atac, color='cell_type', legend_fontsize='7', legend_loc='right margin', size=10,
           title='', frameon=True, save='_unpaired_tmp.pdf')

sc.pl.umap(atac, color='cell_type', legend_fontsize='7', legend_loc='right margin', size=10,
           title='', frameon=True, save='_unpaired.pdf')

sc.pl.umap(atac, color='cell_type', legend_fontsize='7', legend_loc='right margin', size=10,
           title='', frameon=True, save='_unpaired_n_comps_200.pdf')


def out_pred_bedgraph(cell_type='T cell'):
    atac_X = atac[atac.obs['cell_type']==cell_type, :].X.toarray().mean(axis=0)
    df = pd.DataFrame({'chr': atac.var['gene_ids'].map(lambda x: x.split(':')[0]).values,
                       'start': atac.var['gene_ids'].map(lambda x: x.split(':')[1].split('-')[0]).values,
                       'end': atac.var['gene_ids'].map(lambda x: x.split(':')[1].split('-')[1]).values,
                       'val': atac_X})
    df.to_csv(cell_type.replace(' ', '_')+'_atac_pred.bedgraph', index=False, header=False, sep='\t')

for cell_type in set(atac.obs.cell_type.values):
    out_pred_bedgraph(cell_type=cell_type)


def out_pred_norm_bedgraph(cell_type='T cell'):
    # atac_X_raw = atac[atac.obs['cell_type']==cell_type, :].X.toarray()
    # atac_X = (atac_X_raw/(atac_X_raw.sum(axis=1)[:, np.newaxis])).mean(axis=0)*10000
    atac_X_raw = atac[atac.obs['cell_type']==cell_type, :].X.toarray().sum(axis=0)
    atac_X = atac_X_raw/atac_X_raw.sum()*10000
    df = pd.DataFrame({'chr': atac.var['gene_ids'].map(lambda x: x.split(':')[0]).values,
                       'start': atac.var['gene_ids'].map(lambda x: x.split(':')[1].split('-')[0]).values,
                       'end': atac.var['gene_ids'].map(lambda x: x.split(':')[1].split('-')[1]).values,
                       'val': atac_X})
    df.to_csv(cell_type.replace(' ', '_')+'_atac_pred_norm.bedgraph', index=False, header=False, sep='\t')

for cell_type in tqdm(set(atac.obs.cell_type.values), ncols=80):
    out_pred_norm_bedgraph(cell_type=cell_type)


m_18 = atac[(atac.obs['cell_type']=='T cell') & (atac.obs['development_stage']=='18 month-old stage')].X.toarray()  # 65
m_18 = (m_18.sum(axis=0))/(m_18.shape[0])
m_20 = atac[(atac.obs['cell_type']=='T cell') & (atac.obs['development_stage']=='20 month-old stage and over')].X.toarray()  # 1263
m_20 = (m_20.sum(axis=0))/(m_20.shape[0])
m_03 = atac[(atac.obs['cell_type']=='T cell') & (atac.obs['development_stage']=='3 month-old stage')].X.toarray()  # 18
m_03 = (m_03.sum(axis=0))/(m_03.shape[0])
m_01 = atac[(atac.obs['cell_type']=='T cell') & (atac.obs['development_stage']=='4 weeks')].X.toarray()  # 13
m_01 = (m_01.sum(axis=0))/(m_01.shape[0])

m_all = np.vstack((m_01, m_03, m_18, m_20))



# marker genes: https://drive.google.com/drive/u/0/folders/1JZgFDmdVlw-UN8Gb_lRxSAdCZfhlzdrH
# Single cell regulatory landscape of the mouse kidney highlights cellular differentiation programs and disease targets
# Differentail Expression: https://cellxgene.cziscience.com/differential-expression

## prob average seems almost same!!!
# dat = np.load('predict_unpaired.npy')
# atac = sc.read_h5ad('atac_unpaired.h5ad')
# atac.X = dat

# def out_pred_prob_bedgraph(cell_type='T cell'):
#     atac_X = atac[atac.obs['cell_type']==cell_type, :].X.mean(axis=0)
#     df = pd.DataFrame({'chr': atac.var['gene_ids'].map(lambda x: x.split(':')[0]).values,
#                        'start': atac.var['gene_ids'].map(lambda x: x.split(':')[1].split('-')[0]).values,
#                        'end': atac.var['gene_ids'].map(lambda x: x.split(':')[1].split('-')[1]).values,
#                        'val': atac_X})
#     df.to_csv(cell_type.replace(' ', '_')+'_atac_pred_prob.bedgraph', index=False, header=False, sep='\t')

# for cell_type in set(atac.obs.cell_type.values):
#     out_pred_prob_bedgraph(cell_type=cell_type)


