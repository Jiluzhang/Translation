## workdir: /fs/home/jiluzhang/Nature_methods/Spatial
# cp /fs/home/jiluzhang/Nature_methods/Figure_1/scenario_4/rna.h5ad rna.h5ad     # 2597 × 17699
# cp /fs/home/jiluzhang/Nature_methods/Figure_1/scenario_4/atac.h5ad atac.h5ad   # 2597 × 217893

python split_train_val.py --RNA rna.h5ad --ATAC atac.h5ad --train_pct 0.9
python data_preprocess.py -r rna_train.h5ad -a atac_train.h5ad -s train_pt --dt train -n train --config rna2atac_config_train.yaml
python data_preprocess.py -r rna_val.h5ad -a atac_val.h5ad -s val_pt --dt val -n val --config rna2atac_config_val.yaml
nohup accelerate launch --config_file accelerator_config_train.yaml --main_process_port 29824 rna2atac_train.py --config_file rna2atac_config_train.yaml \
                        --train_data_dir train_pt --val_data_dir val_pt -s save -n rna2atac_brain > 20241219.log &   # dec_depth:1  dec_heads:1  lr:2e-4  length:2048  mlt:40  gamma_step:5  gamma:0.5
# 1917494


# cp /fs/home/jiluzhang/2023_nature_RF/human_brain/spatial_rna.h5ad .
# cp /fs/home/jiluzhang/2023_nature_RF/human_brain/spatial_atac.h5ad .

######################################################## filter & map rna & atac h5ad ########################################################
import scanpy as sc
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

rna = sc.read_h5ad('spatial_rna.h5ad')  #  2500 × 38244

sc.pp.filter_genes(rna, min_cells=10)       # 2500 × 13997
sc.pp.filter_cells(rna, min_genes=200)      # 2196 × 13997
sc.pp.filter_cells(rna, max_genes=20000)    # 2196 × 13997
rna.obs['n_genes'].max()                    # 6589

atac = sc.read_h5ad('spatial_atac.h5ad')  #  2500 × 1033239

sc.pp.filter_genes(atac, min_cells=10)       # 2500 × 657953
sc.pp.filter_cells(atac, min_genes=500)      # 2429 × 657953
sc.pp.filter_cells(atac, max_genes=50000)    # 2424 × 657953
atac.obs['n_genes'].max()                    # 48867

idx = np.intersect1d(rna.obs.index, atac.obs.index)
del rna.obs['n_genes']
del rna.var['n_cells']
rna[idx, :].copy().write('spatial_rna_filtered.h5ad')     # 2172 × 13997
del atac.obs['n_genes']
del atac.var['n_cells']
atac[idx, :].copy().write('spatial_atac_filtered.h5ad')   # 2172 × 657953

rna = rna[idx, :].copy()
atac = atac[idx, :].copy()

rna_ref = sc.read_h5ad('./rna.h5ad')   # 2597 × 17699
genes = rna_ref.var[['gene_ids']].copy()
genes['gene_name'] = genes.index.values
genes.reset_index(drop=True, inplace=True)
rna_exp = pd.concat([rna.var, pd.DataFrame(rna.X.T, index=rna.var.index)], axis=1)
X_new = pd.merge(genes, rna_exp, how='left', on='gene_ids').iloc[:, 3:].T
X_new.fillna(value=0, inplace=True)
rna_new = sc.AnnData(X_new.values, obs=rna.obs, var=pd.DataFrame({'gene_ids': genes['gene_ids'], 'feature_types': 'Gene Expression'})) 
# rna_new = sc.AnnData(X_new.values, obs=rna.obs[['cell_anno']], var=pd.DataFrame({'gene_ids': genes['gene_ids'], 'feature_types': 'Gene Expression'}))
rna_new.var.index = genes['gene_name'].values
rna_new.X = csr_matrix(rna_new.X)   # 2172 × 17699
rna_new.write('spatial_rna_test.h5ad')
np.count_nonzero(rna_new.X.toarray(), axis=1).max()  # 6287

## atac
atac_ref = sc.read_h5ad('./atac.h5ad')   # 2597 × 217893

peaks = atac_ref.var[['gene_ids']].copy()
peaks['gene_name'] = peaks.index.values
peaks.reset_index(drop=True, inplace=True)
atac.X = atac.X.toarray()
atac_exp = pd.concat([atac.var, pd.DataFrame(atac.X.T, index=atac.var.index)], axis=1)
X_new = pd.merge(peaks, atac_exp, how='left', on='gene_ids').iloc[:, 3:].T
X_new.fillna(value=0, inplace=True)
atac_new = sc.AnnData(X_new.values, obs=atac.obs, var=pd.DataFrame({'gene_ids': peaks['gene_ids'], 'feature_types': 'Peaks'}))
# atac_new = sc.AnnData(X_new.values, obs=atac.obs[['cell_anno']], var=pd.DataFrame({'gene_ids': peaks['gene_ids'], 'feature_types': 'Peaks'}))
atac_new.var.index = peaks['gene_name'].values
atac_new.X = csr_matrix(atac_new.X)   # 2172 × 217893
atac_new.write('spatial_atac_test.h5ad')
########################################################################################################################################################################

#### plot true umap for val
import snapatac2 as snap
import scanpy as sc

atac = sc.read_h5ad('atac_val.h5ad')
snap.pp.select_features(atac)
snap.tl.spectral(atac)
snap.tl.umap(atac)
sc.pl.umap(atac, color='cell_anno', legend_fontsize='7', legend_loc='right margin', size=10,
           title='', frameon=True, save='_atac_val_true.pdf')

#### val prediction
python data_preprocess.py -r rna_val.h5ad -a atac_val.h5ad -s test_pt --dt test --config rna2atac_config_test.yaml
accelerate launch --config_file accelerator_config_train.yaml --main_process_port 29822 rna2atac_predict.py \
                  -d ./test_pt -l save/2024-12-19_rna2atac_brain_50/pytorch_model.bin --config_file rna2atac_config_test.yaml
python npy2h5ad.py
python plot_save_umap.py --pred atac_cisformer.h5ad --true atac_test.h5ad
python cal_auroc_auprc.py --pred atac_cisformer.h5ad --true atac_test.h5ad
python cal_cluster.py --file atac_cisformer_umap.h5ad


#### plot true umap for test
import snapatac2 as snap
import scanpy as sc

atac = sc.read_h5ad('spatial_atac_test.h5ad')
snap.pp.select_features(atac)
snap.tl.spectral(atac)
snap.tl.umap(atac)
snap.pp.knn(atac)
snap.tl.leiden(atac)
atac.obs['cell_anno'] = atac.obs['leiden']
sc.pl.umap(atac, color='cell_anno', legend_fontsize='7', legend_loc='right margin', size=10,
           title='', frameon=True, save='_atac_test_true.pdf')
atac.write('spatial_atac_test_leiden.h5ad')


#### spatial prediction
python data_preprocess.py -r spatial_rna_test.h5ad -a spatial_atac_test.h5ad -s spatial_test_pt --dt test --config rna2atac_config_test.yaml
accelerate launch --config_file accelerator_config_train.yaml --main_process_port 29822 rna2atac_predict.py \
                  -d ./spatial_test_pt -l save/2024-12-19_rna2atac_brain_50/pytorch_model.bin --config_file rna2atac_config_test.yaml
python npy2h5ad.py
python plot_save_umap.py --pred atac_cisformer.h5ad --true atac_test.h5ad
python cal_auroc_auprc.py --pred atac_cisformer.h5ad --true atac_test.h5ad
python cal_cluster.py --file atac_cisformer_umap.h5ad


#### plot spatial pattern for ground truth
# cp /fs/home/jiluzhang/2023_nature_RF/human_brain/HumanBrain_50um_spatial/tissue_positions_list.csv .
# barcode    in_tissue    array_row    array_column    pxl_col_in_fullres    pxl_row_in_fullres

import scanpy as sc
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

atac = sc.read_h5ad('atac_test.h5ad')

atac.var['chrom'] = atac.var.index.map(lambda x: x.split(':')[0])
atac.var['start'] = atac.var.index.map(lambda x: x.split(':')[1].split('-')[0]).astype('int')
atac.var['end'] = atac.var.index.map(lambda x: x.split(':')[1].split('-')[1]).astype('int')

pos = pd.read_csv('tissue_positions_list.csv', header=None)
pos = pos.iloc[:, [0, 2, 3]]
pos.rename(columns={0:'cell_idx', 2:'X', 3:'Y'}, inplace=True)

## THY1
df_thy1 = pd.DataFrame({'cell_idx':atac.obs.index, 
                        'peak':atac.X[:, ((atac.var['chrom']=='chr11') & ((abs((atac.var['start']+atac.var['end'])*0.5-119424985)<50000)))].toarray().sum(axis=1)})
df_thy1_pos = pd.merge(df_thy1, pos)

mat_thy1 = np.zeros([50, 50])
for i in range(df_thy1_pos.shape[0]):
    if df_thy1_pos['peak'][i]!=0:
        mat_thy1[df_thy1_pos['X'][i], (49-df_thy1_pos['Y'][i])] = df_thy1_pos['peak'][i]  # start from upper right

plt.figure(figsize=(6, 5))
sns.heatmap(mat_thy1, cmap='Reds', xticklabels=False, yticklabels=False)  # cbar=False, vmin=0, vmax=10, linewidths=0.1, linecolor='grey'
plt.savefig('ATAC_THY1_true.pdf')
plt.close()

## BCL11B
df_bcl11b = pd.DataFrame({'cell_idx':atac.obs.index, 
                          'peak':atac.X[:, ((atac.var['chrom']=='chr14') & ((abs((atac.var['start']+atac.var['end'])*0.5-99272197)<50000)))].toarray().sum(axis=1)})
df_bcl11b_pos = pd.merge(df_bcl11b, pos)

mat_bcl11b = np.zeros([50, 50])
for i in range(df_bcl11b_pos.shape[0]):
    if df_bcl11b_pos['peak'][i]!=0:
        mat_bcl11b[df_bcl11b_pos['X'][i], (49-df_bcl11b_pos['Y'][i])] = df_bcl11b_pos['peak'][i]  # start from upper right

plt.figure(figsize=(6, 5))
sns.heatmap(mat_bcl11b, cmap='Reds', xticklabels=False, yticklabels=False)  # cbar=False, vmin=0, vmax=10, linewidths=0.1, linecolor='grey'
plt.savefig('ATAC_BCL11B_true.pdf')
plt.close()


#### plot spatial pattern for cisformer prediction
import scanpy as sc
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

atac = sc.read_h5ad('atac_cisformer_umap.h5ad')

atac.var['chrom'] = atac.var.index.map(lambda x: x.split(':')[0])
atac.var['start'] = atac.var.index.map(lambda x: x.split(':')[1].split('-')[0]).astype('int')
atac.var['end'] = atac.var.index.map(lambda x: x.split(':')[1].split('-')[1]).astype('int')

pos = pd.read_csv('tissue_positions_list.csv', header=None)
pos = pos.iloc[:, [0, 2, 3]]
pos.rename(columns={0:'cell_idx', 2:'X', 3:'Y'}, inplace=True)

## THY1
df_thy1 = pd.DataFrame({'cell_idx':atac.obs.index, 
                        'peak':atac.X[:, ((atac.var['chrom']=='chr11') & ((abs((atac.var['start']+atac.var['end'])*0.5-119424985)<50000)))].toarray().sum(axis=1)})
df_thy1_pos = pd.merge(df_thy1, pos)

mat_thy1 = np.zeros([50, 50])
for i in range(df_thy1_pos.shape[0]):
    if df_thy1_pos['peak'][i]!=0:
        mat_thy1[df_thy1_pos['X'][i], (49-df_thy1_pos['Y'][i])] = df_thy1_pos['peak'][i]  # start from upper right

plt.figure(figsize=(6, 5))
sns.heatmap(mat_thy1, cmap='Reds', xticklabels=False, yticklabels=False)  # cbar=False, vmin=0, vmax=10, linewidths=0.1, linecolor='grey'
plt.savefig('ATAC_THY1_cisformer.pdf')
plt.close()

## BCL11B
df_bcl11b = pd.DataFrame({'cell_idx':atac.obs.index, 
                          'peak':atac.X[:, ((atac.var['chrom']=='chr14') & ((abs((atac.var['start']+atac.var['end'])*0.5-99272197)<50000)))].toarray().sum(axis=1)})
df_bcl11b_pos = pd.merge(df_bcl11b, pos)

mat_bcl11b = np.zeros([50, 50])
for i in range(df_bcl11b_pos.shape[0]):
    if df_bcl11b_pos['peak'][i]!=0:
        mat_bcl11b[df_bcl11b_pos['X'][i], (49-df_bcl11b_pos['Y'][i])] = df_bcl11b_pos['peak'][i]  # start from upper right

plt.figure(figsize=(6, 5))
sns.heatmap(mat_bcl11b, cmap='Reds', xticklabels=False, yticklabels=False)  # cbar=False, vmin=0, vmax=10, linewidths=0.1, linecolor='grey'
plt.savefig('ATAC_BCL11B_cisformer.pdf')
plt.close()


#### cell spatial distribution
import scanpy as sc
import snapatac2 as snap
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

pos = pd.read_csv('tissue_positions_list.csv', header=None)
pos = pos.iloc[:, [0, 2, 3]]
pos.rename(columns={0:'cell_idx', 2:'X', 3:'Y'}, inplace=True)

## plot for true
atac_true = sc.read_h5ad('atac_test.h5ad')

df_true = pd.DataFrame({'cell_idx':atac_true.obs.index, 'cell_anno':atac_true.obs.cell_anno.astype('int')+1})
df_pos_true = pd.merge(df_true, pos)

mat_true = np.zeros([50, 50])
for i in range(df_pos_true.shape[0]):
    if df_pos_true['cell_anno'][i]!=0:
        mat_true[df_pos_true['X'][i], (49-df_pos_true['Y'][i])] = df_pos_true['cell_anno'][i]  # start from upper right

plt.figure(figsize=(6, 5))
sns.heatmap(mat_true, cmap=sns.color_palette(['grey', 'cyan', 'green', 'purple', 'orange', 'yellow', 'red']), xticklabels=False, yticklabels=False)
plt.savefig('ATAC_spatial_true.pdf')
plt.close()

## plot for cisformer
atac_cisformer = sc.read_h5ad('atac_cisformer_umap.h5ad')

snap.pp.knn(atac_cisformer)
snap.tl.leiden(atac_cisformer, resolution=1)  # keep cluster number to 6 (resolution=0.7)

df_cisformer = pd.DataFrame({'cell_idx':atac_cisformer.obs.index, 'cell_anno':atac_cisformer.obs.leiden.astype('int')+1})
df_pos_cisformer = pd.merge(df_cisformer, pos)

mat_cisformer = np.zeros([50, 50])
for i in range(df_pos_cisformer.shape[0]):
    if df_pos_cisformer['cell_anno'][i]!=0:
        mat_cisformer[df_pos_cisformer['X'][i], (49-df_pos_cisformer['Y'][i])] = df_pos_cisformer['cell_anno'][i]  # start from upper right

plt.figure(figsize=(6, 5))
sns.heatmap(mat_cisformer, cmap=sns.color_palette(['grey', 'pink', 'red', 'green', 'cyan', 'yellow', 'orange', 'black', 'brown']), xticklabels=False, yticklabels=False)
plt.savefig('ATAC_spatial_cisformer.pdf')
plt.close()













