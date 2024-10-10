#### brain datasets from scenario_4
import scanpy as sc
import snapatac2 as snap
import seaborn as sns

true = sc.read_h5ad('atac_test.h5ad')
snap.pp.select_features(true)
snap.tl.spectral(true)
snap.tl.umap(true)
sc.pl.umap(true, color=['chr17:44915696-44915866'], legend_fontsize='7', legend_loc='right margin', size=5,
           title='', frameon=True, save='_true.pdf')
