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

sc.tl.leiden(tcell, resolution=0.75)
tcell.obs['leiden'].value_counts()
# 0    10811
# 1     6304
# 2     5198
# 3     4721
# 4     2767
# 5     2570
# 6     1588
# 7     1587
# 8     1352

sc.pl.matrixplot(tcell, ['CD4', 'CD8A', 'CD8B'], groupby='leiden', cmap='coolwarm', standard_scale='var', save='leiden_heatmap_tcell.pdf')
# cluster 1 & 3 highly express CD8A & CD8B

sc.pl.umap(tcell, color='leiden', legend_fontsize='5', legend_loc='right margin', size=5,
           title='', frameon=True, save='_leiden_tcell.pdf')

tcell.write('res_tcell.h5ad')

## cd8 t
cd8_t = tcell[tcell.obs['leiden'].isin(['1', '3'])].copy()  # 11025 × 1868

sc.tl.leiden(cd8_t, resolution=0.90)
cd8_t.obs['leiden'].value_counts()
# 0    2937
# 1    2697
# 2    2010
# 3    1393
# 4     960
# 5     529
# 6     499

# naive T  &  effector T  &  memory T  &  exhausted T
sc.pl.matrixplot(cd8_t, ['CCR7', 'LEF1', 'SELL', 'CD3E', 'CD3D',
                         'GNLY', 'GZMB', 'NKG7', 'CD3D', 'PRF1',
                         'CCR7', 'CD3G', 'IL7R', 'SELL', 'CD3D',
                         'PDCD1', 'CTLA4', 'HAVCR2', 'CD101', 'CD38'], groupby='leiden', cmap='coolwarm', standard_scale='var', dendrogram=True, save='leiden_heatmap_cd8_t.pdf')

sc.pl.umap(cd8_t, color='leiden', legend_fontsize='5', legend_loc='right margin', size=5,
           title='', frameon=True, save='_leiden_cd8_t.pdf')

# marker genes from NBT
# sc.pl.matrixplot(cd8_t, ['CCR7', 'TCF7',
#                          'EOMES', 'IFNG',
#                          'PDCD1', 'CTLA4', 'HAVCR2', 'CD101', 'CD38',
#                          'CD69', 'CD7', 'CRTAM', 'CD3G', 'IL7R', 'SELL'], groupby='leiden', cmap='coolwarm', standard_scale='var', save='cd8_t_leiden_heatmap.pdf')

cd8_t.obs['cell_anno'] = cd8_t.obs['leiden']
cd8_t.obs['cell_anno'].replace({'0':'naive', '1':'ex', '2':'memory', '3':'effect', '4':'effect', '5':'effect', '6':'naive'}, inplace=True)

sc.pl.umap(cd8_t, color='cell_anno', legend_fontsize='5', legend_loc='right margin', size=5,
           title='', frameon=True, save='_cell_anno_cd8_t.pdf')

cd8_t.write('res_cd8_t.h5ad')

## cd8_t_naive_effect_ex
cd8_t_naive_effect_ex = cd8_t[cd8_t.obs['cell_anno'].isin(['naive', 'effect', 'ex'])].copy()  # 9015 × 1868

sc.tl.paga(cd8_t_naive_effect_ex, groups='cell_anno')
sc.pl.paga(cd8_t_naive_effect_ex, layout='fa', color=['cell_anno'], save='_cd8_t_naive_effect_ex.pdf')

sc.tl.draw_graph(cd8_t_naive_effect_ex, init_pos='paga')
sc.pl.draw_graph(cd8_t_naive_effect_ex, color=['cell_anno'], legend_loc='on data', save='_cd8_t_naive_effect_ex_cell_anno.pdf')

cd8_t_naive_effect_ex.uns["iroot"] = np.flatnonzero(cd8_t_naive_effect_ex.obs["cell_anno"]=='naive')[0]
sc.tl.dpt(cd8_t_naive_effect_ex)
sc.pl.draw_graph(cd8_t_naive_effect_ex, color=['cell_anno', 'dpt_pseudotime'], vmax=0.5, legend_loc='on data', save='_cd8_t_naive_effect_ex_pseudo.pdf')

sc.pl.umap(cd8_t_naive_effect_ex, color='dpt_pseudotime', legend_fontsize='5', legend_loc='right margin', size=5, vmax=0.25,
           title='', frameon=True, save='_cell_anno_cd8_t_naive_effect_ex_pseudo.pdf')

cd8_t_naive_effect_ex.write('res_cd8_t_naive_effect_ex.h5ad')


