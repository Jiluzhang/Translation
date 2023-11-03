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
rna_new = sc.AnnData(X_new, obs=rna.obs, var=rna_new_var)
rna_new.write('rna_scTranslator_new_2.h5ad')

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
atac_new.write('atac_scTranslator_new_2.h5ad')
