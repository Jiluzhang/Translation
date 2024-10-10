#### brain datasets from scenario_4
import scanpy as sc
import snapatac2 as snap
import pandas as pd
from tqdm import tqdm

true = sc.read_h5ad('atac_test.h5ad')
snap.pp.select_features(true)
snap.tl.spectral(true)
snap.tl.umap(true)

true.obs['cell_anno'].value_counts()
# Oligodendrocyte    1595
# Astrocyte           402
# OPC                 174
# VIP                 115
# Microglia           109
# SST                 104
# IT                   83
# PVALB                15


def out_norm_bedgraph(atac, cell_type='Oligodendrocyte', type='pred'):
    atac_X_raw = atac[atac.obs['cell_anno']==cell_type, :].X.toarray().sum(axis=0)
    atac_X = atac_X_raw/atac_X_raw.sum()*10000
    df = pd.DataFrame({'chr': atac.var['gene_ids'].map(lambda x: x.split(':')[0]).values,
                       'start': atac.var['gene_ids'].map(lambda x: x.split(':')[1].split('-')[0]).values,
                       'end': atac.var['gene_ids'].map(lambda x: x.split(':')[1].split('-')[1]).values,
                       'val': atac_X})
    df.to_csv(cell_type.replace(' ', '_')+'_atac_'+type+'_norm.bedgraph', index=False, header=False, sep='\t')

for cell_type in tqdm(set(true.obs['cell_anno'].values), ncols=80):
    out_norm_bedgraph(atac=true, cell_type=cell_type, type='true')

for cell_type in tqdm(set(true.obs['cell_anno'].values), ncols=80):
    out_norm_bedgraph(atac=pred, cell_type=cell_type, type='cisformer')

## GFAP
true.var.iloc[198675]   # chr17:44915696-44915866    GFAP
true.var.iloc[51853]    # chr3:126197879-126198225   ALDH1L1

true.obs['gene_0_1'] = true.X.toarray()[:, 51853]
true.obs['gene_0_1'] = true.obs['gene_0_1'].astype('category')
# sc.pl.umap(true, color='gene_0_1', palette=['darkgrey', 'orangered'], legend_fontsize='7', legend_loc='right margin', size=5,
#            title='', frameon=True, save='_GFAP_true.pdf')
true.obs[true.obs['gene_0_1']==1]['cell_anno'].value_counts()
# Astrocyte          148
# OPC                 37
# Oligodendrocyte     10
# Microglia            8
# VIP                  5
# IT                   4
# SST                  3
# PVALB                0


pred = sc.read_h5ad('atac_cisformer_umap.h5ad')
pred.obs['gene_0_1'] = pred.X.toarray()[:, 51855]
pred.obs['gene_0_1'] = pred.obs['gene_0_1'].astype('category')
# sc.pl.umap(pred, color='chr17:44915696-44915866', palette=['darkgrey', 'orangered'], legend_fontsize='7', legend_loc='right margin', size=5,
#            title='', frameon=True, save='_GFAP_cisformer.pdf')
pred.obs[pred.obs['gene_0_1']==1]['cell_anno'].value_counts()
# Oligodendrocyte    540
# Microglia           84
# Astrocyte           59
# OPC                  5
# SST                  4
# VIP                  2
# IT                   1
# PVALB                0


scbt = sc.read_h5ad('atac_scbt_umap.h5ad')
scbt.obs['gene_0_1'] = scbt.X.toarray()[:, 198675]
scbt.obs['gene_0_1'] = scbt.obs['gene_0_1'].astype('category')
sc.pl.umap(scbt, color='gene_0_1', palette=['darkgrey', 'orangered'], legend_fontsize='7', legend_loc='right margin', size=5,
           title='', frameon=True, save='_GFAP_scbt.pdf')

scbt.obs[scbt.obs['gene_0_1']==1]['cell_anno'].value_counts()


cifm = sc.read_h5ad('atac_cisformer.h5ad')
cifm.obs['gene_0_1'] = cifm.X[:, 198675]
scbt.obs['gene_0_1'] = scbt.obs['gene_0_1'].astype('category')
sc.pl.umap(scbt, color='gene_0_1', palette=['darkgrey', 'orangered'], legend_fontsize='7', legend_loc='right margin', size=5,
           title='', frameon=True, save='_GFAP_scbt.pdf')

scbt.obs[scbt.obs['gene_0_1']==1]['cell_anno'].value_counts()


