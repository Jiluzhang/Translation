#### https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi
#### Single-cell genomics and regulatory networks for 388 human brains

## RT00372N
## wget -c https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM8155nnn/GSM8155483/suppl/GSM8155483_RT00372N_RNA_barcodes.txt.gz
## wget -c https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM8155nnn/GSM8155483/suppl/GSM8155483_RT00372N_RNA_matrix.mtx.gz
## wget -c https://ftp.ncbi.nlm.nih.gov/geo/series/GSE261nnn/GSE261983/suppl/GSE261983_RNA_gene_list.txt.gz
## wget -c https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM8155nnn/GSM8155503/suppl/GSM8155503_RT00372N_ATAC_barcodes.txt.gz
## wget -c https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM8155nnn/GSM8155503/suppl/GSM8155503_RT00372N_ATAC_matrix.mtx.gz
## wget -c https://ftp.ncbi.nlm.nih.gov/geo/series/GSE261nnn/GSE261983/suppl/GSE261983_ATAC_peakset.bed.gz

#zcat GSM8155503_RT00372N_ATAC_matrix.mtx.gz | head -n 2 >> ATAC_matrix.mtx
#zcat GSM8155503_RT00372N_ATAC_matrix.mtx.gz | tail -n +3 | awk '{print $2 "\t" $1 "\t" $3}' >> ATAC_matrix.mtx

import scanpy as sc
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix


rna_raw = sc.read_mtx('GSM8155483_RT00372N_RNA_matrix.mtx.gz')  # 8369 × 35649
rna_raw_barcode = pd.read_table('GSM8155483_RT00372N_RNA_barcodes.txt.gz', header=None)
rna_raw.obs.index = rna_raw_barcode[0].values
rna_raw_feature = pd.read_table('GSE261983_RNA_gene_list.txt.gz', header=None)
rna_raw.var.index = rna_raw_feature[0].values
rna_raw.X = rna_raw.X.T

atac_raw = sc.read_mtx('ATAC_matrix.mtx')  # 8984 × 423443
atac_raw_barcode = pd.read_table('GSM8155503_RT00372N_ATAC_barcodes.txt.gz', header=None)
atac_raw.obs.index = atac_raw_barcode[0].values
atac_raw_feature = pd.read_table('GSE261983_ATAC_peakset.bed.gz', header=None)
atac_raw_feature['idx'] = atac_raw_feature[0]+':'+atac_raw_feature[1].astype('str')+'-'+atac_raw_feature[2].astype('str')
atac_raw.var.index = atac_raw_feature['idx'].values

idx = np.intersect1d(rna_raw.obs.index, atac_raw.obs.index)
rna = rna_raw[idx, :].copy()
atac = atac_raw[idx, :].copy()

rna.X = rna.X.toarray()
genes = pd.read_table('human_genes.txt', names=['gene_ids', 'gene_name'])
rna_exp = pd.concat([rna.var, pd.DataFrame(rna.X.T, index=rna.var.index)], axis=1)
rna_exp['gene_name'] = rna_exp.index.values
X_new = pd.merge(genes, rna_exp, how='left', on='gene_name').iloc[:, 2:].T
X_new.fillna(value=0, inplace=True)
rna_new = sc.AnnData(X_new.values, obs=rna.obs, var=pd.DataFrame({'gene_ids': genes['gene_ids'], 'feature_types': 'Gene Expression'}))
rna_new.var.index = genes['gene_name'].values
rna_new.X = csr_matrix(rna_new.X)   # 8369 × 38244


