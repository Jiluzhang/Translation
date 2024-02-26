import scanpy as sc
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix

########## rna ##########
rna_raw = pd.read_csv('GSM6206885_HumanBrain_50um_matrix.tsv.gz', sep='\t')
rna_raw = rna_raw.T  # cell * gene

obs = pd.DataFrame(index=rna_raw.index)
var = pd.DataFrame({'gene_ids': rna_raw.columns, 'feature_types': 'Gene Expression'})
var.index = var['gene_ids'].values
rna = sc.AnnData(rna_raw, obs=obs, var=var)
rna.X = csr_matrix(rna.X)
rna.write('spatial_rna_raw.h5ad')

genes = pd.read_table('human_genes.txt', names=['gene_id', 'gene_name'])

rna_genes = rna.var.copy()
rna_exp = pd.concat([rna_genes, pd.DataFrame(rna.X.T.todense(), index=rna_genes.index)], axis=1)
rna_exp.rename(columns={'gene_ids': 'gene_name'}, inplace=True)

X_new = pd.merge(genes, rna_exp, how='left', on='gene_name').iloc[:, 3:].T
X_new = np.array(X_new, dtype='float32')
X_new[np.isnan(X_new)] = 0

rna_new_var = pd.DataFrame({'gene_ids': genes['gene_id'], 'feature_types': 'Gene Expression'})
rna_new = sc.AnnData(X_new, obs=rna.obs, var=rna_new_var)  
rna_new.var.index = genes['gene_name'].values  # set index for var
rna_new.X = csr_matrix(rna_new.X)

rna_new.write('spatial_rna.h5ad')


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

cCREs = pd.read_table('cCRE_gene/human_cCREs.bed', names=['chr', 'start', 'end'])
cCREs['idx'] = range(cCREs.shape[0])
cCREs_bed = pybedtools.BedTool.from_dataframe(cCREs)
peaks_bed = pybedtools.BedTool.from_dataframe(peaks)
idx_map = peaks_bed.intersect(cCREs_bed, wa=True, wb=True).to_dataframe().iloc[:, [3, 7]]
idx_map.columns = ['peaks_idx', 'cCREs_idx']

m = np.zeros([atac.n_obs, cCREs_bed.to_dataframe().shape[0]], dtype='float32')
for j in tqdm(range(atac.X.shape[0])):
  m[j][idx_map[idx_map['peaks_idx'].isin(peaks.iloc[np.nonzero(atac.X[j])]['idx'])]['cCREs_idx']] = 1

atac_new_var = pd.DataFrame({'gene_ids': cCREs['chr'] + ':' + cCREs['start'].map(str) + '-' + cCREs['end'].map(str), 'feature_types': 'Peaks'})
atac_new = sc.AnnData(m, obs=atac.obs, var=atac_new_var)
atac_new.var.index = atac_new.var['gene_ids'].values  # set index
atac_new.X = csr_matrix(atac_new.X)

atac_new.write('normal_h5ad/'+ids[0][i]+'_atac.h5ad')
