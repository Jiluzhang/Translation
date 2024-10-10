#### brain datasets from scenario_4
import scanpy as sc
import snapatac2 as snap
import pandas as pd

true = sc.read_h5ad('atac_test.h5ad')
snap.pp.select_features(true)
snap.tl.spectral(true)
snap.tl.umap(true)

## GFAP
true.var.iloc[198675]   # chr17:44915696-44915866
true.obs['gene_0_1'] = true.X.toarray()[:, 198675]
true.obs['gene_0_1'] = true.obs['gene_0_1'].astype('category')
sc.pl.umap(true, color='gene_0_1', palette=['darkgrey', 'orangered'], legend_fontsize='7', legend_loc='right margin', size=5,
           title='', frameon=True, save='_GFAP_true.pdf')
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
pred.obs['gene_0_1'] = pred.X.toarray()[:, 198675]
pred.obs['gene_0_1'] = pred.obs['gene_0_1'].astype('category')
sc.pl.umap(pred, color='gene_0_1', palette=['darkgrey', 'orangered'], legend_fontsize='7', legend_loc='right margin', size=5,
           title='', frameon=True, save='_GFAP_cisformer.pdf')

pred.obs[pred.obs['gene_0_1']==1]['cell_anno'].value_counts()
# Oligodendrocyte    540
# Microglia           84
# Astrocyte           59
# OPC                  5
# SST                  4
# VIP                  2
# IT                   1
# PVALB                0
