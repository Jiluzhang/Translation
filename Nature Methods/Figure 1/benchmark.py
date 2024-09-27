#### Dataset 1: PBMCs (https://cf.10xgenomics.com/samples/cell-arc/2.0.0/pbmc_granulocyte_sorted_10k/pbmc_granulocyte_sorted_10k_filtered_feature_bc_matrix.h5)
####            meta info (https://mailustceducn-my.sharepoint.com/personal/hyl2016_mail_ustc_edu_cn/_layouts/15/download.aspx?UniqueId=e86f4bba%2D8560%2D4689%2Dbb93%2D9ea5c859ba28)
#### Dataset 2: BMMCs
#### Dataset 3: Adult brain

#### celltype (not predicted.id)
# CD4 TCM (Central memory T cells)
# CD4 Naive
# CD8 Naive
# CD16 Mono
# NK
# Treg (Regulatory T cells)
# CD14 Mono
# CD4 intermediate
# CD4 TEM (Effector memory T cells)
# RPL/S Leukocytes (discard)
# B memory
# MAIT (mucosal-associated invariant T cells)
# IL1B+ Mono -> CD14 Mono
# B naive
# CD8 TEM
# pDC (Plasmacytoid dendritic cells)
# cDC2 (type-2 conventional dendritic cells)


#### h5 to h5ad
import scanpy as sc
import pandas as pd

dat = sc.read_10x_h5('pbmc_granulocyte_sorted_10k_filtered_feature_bc_matrix.h5', gex_only=False)  # 11898 × 180488
dat.var_names_make_unique()

## add cell annotation
meta_info = pd.read_table('meta_data.txt')
cell_barcode = pd.DataFrame({'barcode': dat.obs.index.values})
dat.obs['cell_anno'] = pd.merge(cell_barcode, meta_info, how='left')['celltype'].values
dat = dat[(~dat.obs['cell_anno'].isna() & (dat.obs['cell_anno']!='RPL/S Leukocytes')), :].copy()   # 9964 × 180488
dat.obs['cell_anno'].replace({'IL1B+ Mono': 'CD14 Mono'}, inplace=True)

## rna
rna = dat[:, dat.var['feature_types']=='Gene Expression'].copy()  # 9964 × 36601
sc.pp.filter_genes(rna, min_cells=10)
sc.pp.filter_cells(rna, min_genes=200)
sc.pp.filter_cells(rna, max_genes=20000)    # 9964 × 21000

## atac
atac = dat[:, dat.var['feature_types']=='Peaks'].copy()  # 9964 × 143887
atac.X[atac.X>0] = 1
sc.pp.filter_genes(atac, min_cells=10)
sc.pp.filter_cells(atac, min_genes=500)
sc.pp.filter_cells(atac, max_genes=50000)  # 9964 × 143856

## output rna & atac (no cells are filtered)
rna.write('rna.h5ad')
atac.write('atac.h5ad')









