import scanpy as sc
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
import pybedtools
from tqdm import tqdm

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
atac_raw = pd.read_csv('GSM6206884_HumanBrain_50um_fragments.tsv.gz', comment='#', header=None, sep='\t')  # 44,940,844
atac_raw.columns = ['chr', 'start', 'end', 'barcode', 'count']
atac_raw = atac_raw[atac_raw['chr'].isin(['chr'+str(i) for i in range(1, 23)])]  # chr1-chr22  43,668,969
atac_raw['peak_id'] = atac_raw['chr']+':'+atac_raw['start'].astype(str)+'-'+atac_raw['end'].astype(str)

cCREs = pd.read_table('human_cCREs.bed', names=['chr', 'start', 'end'])
cCREs['idx'] = range(cCREs.shape[0])
cCREs_bed = pybedtools.BedTool.from_dataframe(cCREs)

m = np.zeros([2500, cCREs_bed.to_dataframe().shape[0]], dtype='float32')
## ~1.5h
for i in tqdm(range(2500)):
  atac_tmp = atac_raw[atac_raw['barcode']==rna_new.obs.index[i]+'-1']
  
  peaks = atac_tmp[['chr', 'start', 'end']]
  peaks['idx'] = range(peaks.shape[0])
  
  peaks_bed = pybedtools.BedTool.from_dataframe(peaks)
  idx_map = peaks_bed.intersect(cCREs_bed, wa=True, wb=True).to_dataframe().iloc[:, [3, 7]]
  idx_map.columns = ['peaks_idx', 'cCREs_idx']
  
  m[i][idx_map['cCREs_idx'].values] = 1

atac_new_var = pd.DataFrame({'gene_ids': cCREs['chr'] + ':' + cCREs['start'].map(str) + '-' + cCREs['end'].map(str), 'feature_types': 'Peaks'})
atac_new = sc.AnnData(m, obs=rna_new.obs, var=atac_new_var)
atac_new.var.index = atac_new.var['gene_ids'].values  # set index
atac_new.X = csr_matrix(m)

atac_new.write('spatial_atac.h5ad')
