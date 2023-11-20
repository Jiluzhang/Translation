import scanpy as sc
import pandas as pd
import numpy as np
import pybedtools
from tqdm import tqdm

dat = sc.read_10x_h5('/fs/home/jiluzhang/scMOG_raw/scMOG_modified/dataset/DM_rep4.h5', gex_only=False)

########## rna ##########
rna = dat[:, dat.var['feature_types']=='Gene Expression'].copy()
rna.X = np.array(rna.X.todense()) 
genes = pd.read_table('human_genes.txt', names=['gene_id', 'gene_name'])

rna_genes = rna.var.copy()
rna_genes['gene_id'] = rna_genes['gene_ids'].map(lambda x: x.split('.')[0])  # delete ensembl id with version number
rna_exp = pd.concat([rna_genes, pd.DataFrame(rna.X.T, index=rna_genes.index)], axis=1)

X_new = pd.merge(genes, rna_exp, how='left', on='gene_id').iloc[:, 5:].T
X_new = np.array(X_new, dtype='float32')
X_new[np.isnan(X_new)] = 0

rna_new_var = pd.DataFrame({'gene_ids': genes['gene_id'], 'feature_types': 'Gene Expression', 'my_Id': range(genes.shape[0])})
rna_new = sc.AnnData(X_new, obs=rna.obs, var=rna_new_var)  # 5517*38244

rna_new.var.index = genes['gene_name']  # set index for var

rna_new[:1000, :].write('rna_train_dm_1000.h5ad')
rna_new[1000:2000, :].write('rna_train_dm_1000_2000.h5ad')
rna_new[2000:3000, :].write('rna_train_dm_2000_3000.h5ad')
rna_new[3000:4000, :].write('rna_train_dm_3000_4000.h5ad')
rna_new[4000:5000, :].write('rna_train_dm_4000_5000.h5ad')
rna_new[5417:, :].write('rna_test_dm_100.h5ad')


########## atac ##########
atac = dat[:, dat.var['feature_types']=='Peaks'].copy()
atac.X = np.array(atac.X.todense())
atac.X[atac.X>1] = 1
atac.X[atac.X<1] = 0

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
for i in tqdm(range(atac.X.shape[0])):
    m[i][idx_map[idx_map['peaks_idx'].isin(peaks.iloc[np.nonzero(atac.X[i])]['idx'])]['cCREs_idx']] = 1

atac_new_var = pd.DataFrame({'gene_ids': cCREs['chr'] + ':' + cCREs['start'].map(str) + '-' + cCREs['end'].map(str), 'feature_types': 'Peaks', 'my_Id': range(cCREs.shape[0])})
atac_new = sc.AnnData(m, obs=atac.obs, var=atac_new_var)

atac_new.var.index = atac_new.var['gene_ids']  # set index

atac_new[:1000, :].write('atac_train_dm_1000.h5ad')
atac_new[1000:2000, :].write('atac_train_dm_1000_2000.h5ad')
atac_new[2000:3000, :].write('atac_train_dm_2000_3000.h5ad')
atac_new[3000:4000, :].write('atac_train_dm_3000_4000.h5ad')
atac_new[4000:5000, :].write('atac_train_dm_4000_5000.h5ad')
atac_new[5417:, :].write('atac_test_dm_100.h5ad')



import scanpy as sc
rna_dm = sc.read_h5ad('rna_train_dm_1000.h5ad')
rna_dm.var.index = rna_dm.var['gene_ids']
rna_hsr = sc.read_h5ad('rna_train_hsr_1000.h5ad')
rna_hsr.var.index = rna_hsr.var['gene_ids']
sc.AnnData.concatenate(rna_dm[:500, :], rna_hsr[:500, :]).write('rna_train_dm_500_hsr_500_1.h5ad')
sc.AnnData.concatenate(rna_dm[500:1000, :], rna_hsr[500:1000, :]).write('rna_train_dm_500_hsr_500_2.h5ad')

atac_dm = sc.read_h5ad('atac_train_dm_1000.h5ad')
atac_dm.var.index = atac_dm.var['gene_ids']
atac_hsr = sc.read_h5ad('atac_train_hsr_1000.h5ad')
atac_hsr.var.index = atac_hsr.var['gene_ids']
sc.AnnData.concatenate(atac_dm[:500, :], atac_hsr[:500, :]).write('atac_train_dm_500_hsr_500_1.h5ad')
sc.AnnData.concatenate(atac_dm[500:1000, :], atac_hsr[500:1000, :]).write('atac_train_dm_500_hsr_500_2.h5ad')


import scanpy as sc
dat = sc.read_10x_h5('train_dm_1000.h5', gex_only=False)
atac = dat[:, dat.var['feature_types']=='Peaks']
m = atac.X.toarray()
for i in range(20):
    print(sum(m[i]!=0))

cnt = 0
for i in range(1000):
    if (sum(m[i])<5000):
        cnt += 1

