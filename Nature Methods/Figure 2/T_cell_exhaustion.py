## tractory analysis: https://scanpy.readthedocs.io/en/latest/tutorials/trajectories/paga-paul15.html
## workdir: /fs/home/jiluzhang/Nature_methods/Figure_2/pan_cancer/cisformer/T_cell

import scanpy as sc
import harmonypy as hm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

plt.rcParams['pdf.fonttype'] = 42

rna = sc.read_h5ad('../rna.h5ad')   # 144409 × 19160
tcell = rna[rna.obs['cell_anno']=='T-cells'].copy()  # 36898 × 19160

sc.pp.normalize_total(tcell, target_sum=1e4)
sc.pp.log1p(tcell)
# fibro.obs['batch'] = [x.split('_')[0] for x in fibro.obs.index]  # exist samples with only one fibroblast
sc.pp.highly_variable_genes(tcell, batch_key='cancer_type')
tcell.raw = tcell
tcell = tcell[:, tcell.var.highly_variable]   # 36898 × 1868
sc.tl.pca(tcell)
ho = hm.run_harmony(tcell.obsm['X_pca'], tcell.obs, 'cancer_type')  # Converged after 2 iterations

tcell.obsm['X_pca_harmony'] = ho.Z_corr.T
sc.pp.neighbors(tcell, use_rep='X_pca_harmony')
sc.tl.umap(tcell)
sc.tl.leiden(tcell, resolution=0.25)
tcell.obs['leiden'].value_counts()
# 0    14374
# 1    11712
# 2     5107
# 3     3941
# 4     1545
# 5      219

sc.tl.rank_genes_groups(tcell, 'leiden')
pd.DataFrame(tcell.uns['rank_genes_groups']['names']).head(20)

sc.pl.matrixplot(tcell, ['CD4', 'CD8A', 'CD8B'], groupby='leiden', cmap='coolwarm', standard_scale='var', save='tcell_leiden_heatmap.pdf')
# cluster 1 highly express CD8A & CD8B
# maybe more clusters

tcell.write('res_tcell.h5ad')

cd8_t = tcell[tcell.obs['leiden']=='1'].copy()  # 11712 × 1868
sc.pp.neighbors(cd8_t, use_rep='X_pca_harmony')
sc.tl.umap(cd8_t)
sc.tl.leiden(cd8_t, resolution=0.90)
cd8_t.obs['leiden'].value_counts()
# 0    4145
# 1    2683
# 2    2500
# 3    1631
# 4     753
sc.pl.matrixplot(cd8_t, ['CCR7', 'TCF7',
                         'EOMES', 'IFNG',
                         'PDCD1', 'CTLA4', 'HAVCR2', 'CD101', 'CD38'], groupby='leiden', cmap='coolwarm', standard_scale='var', save='cd8_t_leiden_heatmap.pdf')
sc.pl.umap(cd8_t, color='leiden', legend_fontsize='5', legend_loc='right margin', size=5,
           title='', frameon=True, save='_leiden_cd8_t.pdf')

# naive T  &  effector T  &  memory T  &  exhausted T

sc.tl.paga(tcell, groups='leiden')
sc.pl.paga(tcell, layout='fa', arrowsize=200, color=['leiden', 'PDCD1'], save='_tcell.pdf')

sc.tl.draw_graph(tcell, init_pos="paga")
sc.pl.draw_graph(tcell, color=['leiden', 'PDCD1'], legend_loc='on data', save='_tcell_sc.pdf')

tcell.uns["iroot"] = np.flatnonzero(tcell.obs["leiden"]=='0')[0]
sc.tl.dpt(tcell)
sc.pl.draw_graph(tcell, color=['leiden', 'dpt_pseudotime'], arrows=True, legend_loc='on data', save='_tcell_pseudo.pdf')

