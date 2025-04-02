## workdir: /fs/home/jiluzhang/Nature_methods/Figure_2/pan_cancer/cisformer/Macrophage/subtype

import scanpy as sc
import harmonypy as hm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import rcParams

plt.rcParams['pdf.fonttype'] = 42

rna = sc.read_h5ad('cancer_macro_annotation.h5ad')           # 26772 × 37982
sc.pp.filter_genes(rna, min_cells=30)                        # 26772 × 17203
rna.obs['cell_type'].value_counts()
# Macro_C1QC      9073
# Macro_APOC1     5324
# Macro_SPP1      2751
# Macro_IL32      2287
# Mono_FCN1       2270
# Macro_THBS1     1949
# Macro_SLPI      1216
# Macro_CDC20      992
# Macro_COL1A1     910

np.percentile(np.count_nonzero(rna.X.toarray(), axis=1), 80)   # 1983.0
np.percentile(np.count_nonzero(rna.X.toarray(), axis=1), 90)   # 2356.0
np.percentile(np.count_nonzero(rna.X.toarray(), axis=1), 95)   # 2726.449999999997
np.percentile(np.count_nonzero(rna.X.toarray(), axis=1), 100)  # 7068.0

rna.write('rna.h5ad')

atac = sc.read_h5ad('cancer_macro_atac.h5ad')       # 28262 × 1033239
sc.pp.filter_genes(atac, min_cells=30)              # 28262 × 315670

atac.obs.index = [atac.obs.index[i]+'_'+atac.obs['sample'][i] for i in range(atac.n_obs)]  
atac[[rna.obs.index[i]+'_'+rna.obs['sample'][i] for i in range(rna.n_obs)]].write('atac.h5ad')  # 26772 × 315670

sc.pp.normalize_total(rna, target_sum=1e4)
sc.pp.log1p(rna)
sc.pp.highly_variable_genes(rna, batch_key='cancer_type')
rna.raw = rna
rna = rna[:, rna.var.highly_variable]   # 26772 × 2060
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

rna.obs['cell_type'] = pd.Categorical(rna.obs['cell_type'], categories=['Mono_FCN1', 'Macro_THBS1', 'Macro_IL32', 'Macro_C1QC',
                                                                        'Macro_SPP1', 'Macro_CDC20', 'Macro_SLPI', 'Macro_APOC1',
                                                                        'Macro_COL1A1'])

sc.settings.set_figure_params(dpi_save=600)
rcParams["figure.figsize"] = (2, 2)

sc.pl.umap(rna, color='cell_type', legend_fontsize='5', legend_loc='right margin', size=2,
           title='', frameon=True, save='_cell_type_macro.pdf')

sc.pl.matrixplot(rna, ['FCN1', 'LYZ', 
                       'THBS1',
                       'IL32',
                       'C1QC', 'C1QA', 'C1QB',
                       'SPP1', 'MMP9',
                       'CDC20', 'STMN1',
                       'SLPI',
                       'APOC1',
                       'COL1A1', 'COL1A2',],
                 dendrogram=False, groupby='cell_type', cmap='coolwarm', standard_scale='var', save='cell_type_heatmap.pdf')

rna.write('rna_res.h5ad')

## train Cisformer
python split_train_val.py --RNA rna.h5ad --ATAC atac.h5ad --train_pct 0.9
python data_preprocess.py -r rna_train.h5ad -a atac_train.h5ad -s train_pt --dt train -n train --config rna2atac_config_train.yaml
python data_preprocess.py -r rna_val.h5ad -a atac_val.h5ad -s val_pt --dt val -n val --config rna2atac_config_val.yaml
nohup accelerate launch --config_file accelerator_config_train.yaml --main_process_port 29824 rna2atac_train.py --config_file rna2atac_config_train.yaml \
                        --train_data_dir train_pt --val_data_dir val_pt -s save -n rna2atac > 20250402.log &    # 2880898






