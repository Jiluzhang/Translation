## tractory analysis: https://scanpy.readthedocs.io/en/latest/tutorials/trajectories/paga-paul15.html
## workdir: /fs/home/jiluzhang/Nature_methods/Figure_2/pan_cancer/cisformer/T_cell

import scanpy as sc
import harmonypy as hm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

plt.rcParams['pdf.fonttype'] = 42

rna = sc.read_h5ad('../rna_test.h5ad')   # 6478 × 19160
tcell = rna[rna.obs['cell_anno']=='T-cells'].copy()  # 1042 × 19160

sc.pp.normalize_total(tcell, target_sum=1e4)
sc.pp.log1p(tcell)
# fibro.obs['batch'] = [x.split('_')[0] for x in fibro.obs.index]  # exist samples with only one fibroblast
sc.pp.highly_variable_genes(tcell, batch_key='cancer_type')
tcell.raw = tcell
tcell = tcell[:, tcell.var.highly_variable]   # 1042 × 3174
sc.tl.pca(tcell)
ho = hm.run_harmony(tcell.obsm['X_pca'], tcell.obs, 'cancer_type')  # Converged after 2 iterations

tcell.obsm['X_pca_harmony'] = ho.Z_corr.T
sc.pp.neighbors(tcell, use_rep='X_pca_harmony')

sc.tl.leiden(tcell, resolution=0.25)
tcell.obs['leiden'].value_counts()
# 0    695
# 1    161
# 2     92
# 3     40
# 4     38
# 5     16
sc.tl.paga(tcell, groups='leiden')
sc.pl.paga(tcell, layout='fa', arrowsize=200, color=['leiden', 'PDCD1'], save='_tcell.pdf')

sc.tl.draw_graph(tcell, init_pos="paga")
sc.pl.draw_graph(tcell, color=['leiden', 'PDCD1'], legend_loc='on data', save='_tcell_sc.pdf')

tcell.uns["iroot"] = np.flatnonzero(tcell.obs["leiden"]=='0')[0]
sc.tl.dpt(tcell)
sc.pl.draw_graph(tcell, color=['leiden', 'dpt_pseudotime'], arrows=True, legend_loc='on data', save='_tcell_pseudo.pdf')

