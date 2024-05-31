## pbmc dataset

## alignment -> filtering -> clustering -> modeling -> evaluation

## raw data process
# human_genes.txt  38244
# human_cCREs.bed  1033239
# python align_filter_rna_atac_10x.py -i pbmc_granulocyte_sorted_10k_filtered_feature_bc_matrix.h5
import argparse
import warnings
import scanpy as sc
import pandas as pd
import numpy as np
import pybedtools
from tqdm import tqdm
from scipy.sparse import csr_matrix

warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(description='Alignment for h5 file from 10x')
parser.add_argument('-i', '--input', type=str, help='h5 file')

args = parser.parse_args()
h5_file = args.input

dat = sc.read_10x_h5(h5_file, gex_only=False)
dat.var_names_make_unique()

## RNA
rna = dat[:, dat.var['feature_types']=='Gene Expression'].copy()
rna.X = rna.X.toarray()
genes = pd.read_table('human_genes.txt', names=['gene_ids', 'gene_name'])

rna_exp = pd.concat([rna.var, pd.DataFrame(rna.X.T, index=rna.var.index)], axis=1)
X_new = pd.merge(genes, rna_exp, how='left', on='gene_ids').iloc[:, 4:].T
X_new.fillna(value=0, inplace=True)

rna_new = sc.AnnData(X_new.values, obs=rna.obs, var=pd.DataFrame({'gene_ids': genes['gene_ids'], 'feature_types': 'Gene Expression'}))
rna_new.var.index = genes['gene_name'].values
rna_new.X = csr_matrix(rna_new.X)
rna_new.write('rna_aligned.h5ad')
print('######## RNA alignment done ########')

## ATAC
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
atac_new.write('atac_aligned.h5ad')
print('######## ATAC alignment done ########')

## filter genes & peaks & cells
sc.pp.filter_genes(rna_new, min_cells=10)
sc.pp.filter_cells(rna_new, min_genes=500)
sc.pp.filter_cells(rna_new, max_genes=5000)

sc.pp.filter_genes(atac_new, min_cells=10)
sc.pp.filter_cells(atac_new, min_genes=1000)
sc.pp.filter_cells(atac_new, max_genes=50000)

idx = np.intersect1d(rna_new.obs.index, atac_new.obs.index)
rna_new[idx, :].copy().write('rna.h5ad')
atac_new[idx, :].copy().write('atac.h5ad')
print(f'Select {rna_new.n_obs} / {rna.n_obs} cells')
print(f'Select {rna_new.n_vars} / 38244 genes')
print(f'Select {atac_new.n_vars} / 1033239 peaks')





sc.pp.normalize_total(rna, target_sum=1e4)
sc.pp.log1p(rna)

sc.pp.highly_variable_genes(rna)
rna.raw = rna
rna = rna[:, rna.var.highly_variable]
sc.tl.pca(rna)
sc.pp.neighbors(rna)
sc.tl.umap(rna, min_dist=0.4)
sc.tl.leiden(rna, resolution=0.1)
sc.pl.umap(rna, color='leiden', legend_fontsize='5', legend_loc='on data',
           title='', frameon=True, save='.pdf')

sc.tl.rank_genes_groups(adata, 'leiden', method='wilcoxon')

pd.DataFrame(adata.uns['rank_genes_groups']['names']).head(10)
#            0          1       2        3        4
# 0      PRKCH       TCF4  SLC8A1    BANK1    FBXL7
# 1        FYN       PAX5   RBM47  RALGPS2     SVIL
# 2     THEMIS   ARHGAP24   LRMDA   CCSER1     NFIB
# 3     BCL11B       AFF3    TYMP    MEF2C  PPFIBP1
# 4     INPP4B      FCRL5   DMXL2    FCRL1      APP
# 5      CELF2      GPM6A    DPYD   FCHSD2    ZFPM2
# 6      CD247  LINC01320    TNS3    MS4A1    DOCK1
# 7      MBNL1     NIBAN3   MCTP1     AFF3    MAGI1
# 8       CBLB      FOXP1  PLXDC2    CDK14   SPTBN1
# 9  LINC01934     RUBCNL   FMNL2   RIPOR2    RBPMS

# cluster_0:PRKCH,FYN,THEMIS,BCL11B,INPP4B,CELF2,CD247,MBNL1,CBLB,LINC01934
# cluster_1:TCF4,PAX5,ARHGAP24,AFF3,FCRL5,GPM6A,LINC01320,NIBAN3,FOXP1,RUBCNL
# cluster_2:SLC8A1,RBM47,LRMDA,TYMP,DMXL2,DPYD,TNS3,MCTP1,PLXDC2,FMNL2
# cluster_3:BANK1,RALGPS2,CCSER1,MEF2C,FCRL1,FCHSD2,MS4A1,AFF3,CDK14,RIPOR2
# cluster_4:FBXL7,SVIL,NFIB,PPFIBP1,APP,ZFPM2,DOCK1,MAGI1,SPTBN1,RBPMS

# cluster_0: T cell
# cluster_1: Tumor B cell
# cluster_2: Monocyte
# cluster_3: Normal B cell
# cluster_4: Other

## cell type annotation tool: http://xteam.xbio.top/ACT
## ref: https://www.nature.com/articles/s41467-023-36559-0/figures/5
cell_anno_dict = {'0':'T cell', '1':'Tumor B cell', '2':'Monocyte', '3':'Normal B cell', '4':'Other'}
adata.obs['cell_anno'] = adata.obs['leiden']
adata.obs['cell_anno'].replace(cell_anno_dict, inplace=True)

sc.pl.umap(adata, color='cell_anno', legend_fontsize='7', legend_loc='on data', size=3,
           title='', frameon=True, save='_cell_annotation.pdf')

adata.write('rna_tumor_B_res.h5ad')
