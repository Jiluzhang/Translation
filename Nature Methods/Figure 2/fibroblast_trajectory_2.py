## workdir: /fs/home/jiluzhang/Nature_methods/Figure_2/pan_cancer/cisformer/Fibroblast/subtype

import scanpy as sc
import harmonypy as hm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import rcParams

plt.rcParams['pdf.fonttype'] = 42

rna_raw = sc.read_h5ad('cancer_fibro_annotation.h5ad')           # 19138 × 37982
rna = rna_raw[rna_raw.obs['cell_type']!='iFibro_FBLN1'].copy()   # 14738 × 37982
sc.pp.filter_genes(rna, min_cells=10)                            # 14738 × 19400
rna.obs['cell_type'].value_counts()
# eFibro_COL11A1    8675
# apFibro_CCL5      2736
# qFibro_SAA1       1304
# eFibro_CTHRC1     1260
# MyoFibro_MYH11     763

np.percentile(np.count_nonzero(rna.X.toarray(), axis=1), 80)   # 2455.0
np.percentile(np.count_nonzero(rna.X.toarray(), axis=1), 95)   # 3580.0
np.percentile(np.count_nonzero(rna.X.toarray(), axis=1), 100)  # 8406.0

rna.write('rna.h5ad')

atac = sc.read_h5ad('cancer_fibro_atac_annotation.h5ad')       # 19138 × 1033239
atac = atac[rna_raw.obs['cell_type']!='iFibro_FBLN1'].copy()   # 14738 × 1033239
sc.pp.filter_genes(atac, min_cells=10)                         # 14738 × 368527
atac.write('atac.h5ad')

sc.pp.normalize_total(rna, target_sum=1e4)
sc.pp.log1p(rna)
sc.pp.highly_variable_genes(rna, batch_key='cancer_type')
rna.raw = rna
rna = rna[:, rna.var.highly_variable]   # 19138 × 2239
sc.tl.pca(rna)
ho = hm.run_harmony(rna.obsm['X_pca'], rna.obs, 'cancer_type')  # Converged after 2 iterations

rna.obsm['X_pca_harmony'] = ho.Z_corr.T
sc.pp.neighbors(rna, use_rep='X_pca_harmony')
sc.tl.umap(rna)

# sc.tl.leiden(rna, resolution=0.75)
# rna.obs['leiden'].value_counts()
# # 0     3575
# # 1     2887
# # 2     2293
# # 3     2238
# # 4     1914
# # 5     1461
# # 6     1164
# # 7      988
# # 8      824
# # 9      747
# # 10     537
# # 11     357
# # 12     153

# sc.pl.matrixplot(rna, ['MYH11', 'RGS5', 'NDUFA4L2', 'ADIRF', 
#                        'CCL5', 'HLA-DRA', 'LYZ', 'TYROBP', 'HLA-DRB1', 'HLA-DPB1',
#                        'COL11A1',
#                        'CTHRC1', 'MMP11', 'COL1A1',
#                        'FBLN1', 'FBLN2', 'CFD', 'GSN',
#                        'SAA1'],
#                  dendrogram=True, groupby='leiden', cmap='coolwarm', standard_scale='var', save='leiden_heatmap.pdf')

rna.obs['cell_type'] = pd.Categorical(rna.obs['cell_type'], categories=['qFibro_SAA1', 'eFibro_CTHRC1', 'eFibro_COL11A1', 'apFibro_CCL5', 'MyoFibro_MYH11'])

sc.settings.set_figure_params(dpi_save=600)
rcParams["figure.figsize"] = (2, 2)

sc.pl.umap(rna, color='cell_type', legend_fontsize='5', legend_loc='right margin', size=5,
           title='', frameon=True, save='_cell_type_fibro.pdf')

sc.pl.matrixplot(rna, ['SAA1',
                       'CTHRC1', 'MMP11', 'COL1A1',
                       'COL11A1',
                       'CCL5', 'HLA-DRA', 'LYZ',
                       'MYH11', 'RGS5', 'ADIRF'],
                 dendrogram=False, groupby='cell_type', cmap='coolwarm', standard_scale='var', save='cell_type_heatmap.pdf')

rna.write('rna_res.h5ad')

## train Cisformer
python split_train_val.py --RNA rna.h5ad --ATAC atac.h5ad --train_pct 0.9
python data_preprocess.py -r rna_train.h5ad -a atac_train.h5ad -s train_pt --dt train -n train --config rna2atac_config_train.yaml
python data_preprocess.py -r rna_val.h5ad -a atac_val.h5ad -s val_pt --dt val -n val --config rna2atac_config_val.yaml
nohup accelerate launch --config_file accelerator_config_train.yaml --main_process_port 29823 rna2atac_train.py --config_file rna2atac_config_train.yaml \
                        --train_data_dir train_pt --val_data_dir val_pt -s save -n rna2atac > 20250402.log &    # 2426617

























