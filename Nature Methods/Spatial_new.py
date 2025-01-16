## workdir: /fs/home/jiluzhang/Nature_methods/Spatial_new
# cp /fs/home/jiluzhang/Nature_methods/Figure_1/scenario_4/rna.h5ad rna.h5ad     # 2597 × 17699
# cp /fs/home/jiluzhang/Nature_methods/Figure_1/scenario_4/atac.h5ad atac.h5ad   # 2597 × 217893

python split_train_val.py --RNA rna.h5ad --ATAC atac.h5ad --train_pct 0.9
python data_preprocess.py -r rna_train.h5ad -a atac_train.h5ad -s train_pt --dt train -n train --config rna2atac_config_train.yaml
python data_preprocess.py -r rna_val.h5ad -a atac_val.h5ad -s val_pt --dt val -n val --config rna2atac_config_val.yaml

nohup accelerate launch --config_file accelerator_config_train.yaml --main_process_port 29824 rna2atac_train.py --config_file rna2atac_config_train.yaml \
                        --train_data_dir train_pt --val_data_dir val_pt -s save -n rna2atac_brain > 20250115.log & 

# cp /fs/home/jiluzhang/2023_nature_RF/human_brain/spatial_rna.h5ad .
# cp /fs/home/jiluzhang/2023_nature_RF/human_brain/spatial_atac.h5ad .


######################################################## compare no_norm and with_norm ####################################################################################
import scanpy as sc

rna = sc.read_h5ad('rna_train.h5ad')
rna[:100, :].write('rna_train_100_no_norm.h5ad')
sc.pp.log1p(rna)
sc.pp.highly_variable_genes(rna)
rna[:100, :].write('rna_train_100_with_norm.h5ad')

atac = sc.read_h5ad('atac_train.h5ad')
atac[:100, :].write('atac_train_100.h5ad')

rna = sc.read_h5ad('rna_val.h5ad')
rna.write('rna_val_no_norm.h5ad')
sc.pp.log1p(rna)
sc.pp.highly_variable_genes(rna)
rna.write('rna_val_with_norm.h5ad')


python data_preprocess.py -r rna_train_100_no_norm.h5ad -a atac_train_100.h5ad -s train_pt_no_norm --dt train -n train --config rna2atac_config_train_no_norm.yaml
python data_preprocess.py -r rna_val_no_norm.h5ad -a atac_val.h5ad -s val_pt_no_norm --dt val -n val --config rna2atac_config_val_no_norm.yaml
nohup accelerate launch --config_file accelerator_config_train.yaml --main_process_port 29824 rna2atac_train.py --config_file rna2atac_config_train_no_norm.yaml \
                        --train_data_dir train_pt_no_norm --val_data_dir val_pt_no_norm -s save_no_norm -n rna2atac_brain > no_norm.log &  # 344226

python data_preprocess.py -r rna_train_100_with_norm.h5ad -a atac_train_100.h5ad -s train_pt_with_norm --dt train -n train --config rna2atac_config_train_with_norm.yaml
python data_preprocess.py -r rna_val_with_norm.h5ad -a atac_val.h5ad -s val_pt_with_norm --dt val -n val --config rna2atac_config_val_with_norm.yaml
nohup accelerate launch --config_file accelerator_config_train.yaml --main_process_port 29825 rna2atac_train.py --config_file rna2atac_config_train_with_norm.yaml \
                        --train_data_dir train_pt_with_norm --val_data_dir val_pt_with_norm -s save_with_norm -n rna2atac_brain > with_norm.log &  # 358890

nohup accelerate launch --config_file accelerator_config_train.yaml --main_process_port 29825 rna2atac_train.py --config_file rna2atac_config_train_with_norm_128.yaml \
                        --train_data_dir train_pt_with_norm --val_data_dir val_pt_with_norm -s save_with_norm_128 -n rna2atac_brain > with_norm_128.log &  # 547440
#################################################################################################################################################################################




######################################################## filter & map rna & atac h5ad ########################################################
import scanpy as sc
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

rna = sc.read_h5ad('spatial_rna.h5ad')  #  2500 × 38244

sc.pp.filter_genes(rna, min_cells=10)       # 2500 × 13997
# sc.pp.filter_cells(rna, min_genes=200)      # 2196 × 13997  (do not filter cells)
# sc.pp.filter_cells(rna, max_genes=20000)    # 2196 × 13997
np.count_nonzero(rna.X.toarray(), axis=1).max()   # 6589

atac = sc.read_h5ad('spatial_atac.h5ad')  #  2500 × 1033239

sc.pp.filter_genes(atac, min_cells=10)       # 2500 × 657953
# sc.pp.filter_cells(atac, min_genes=500)      # 2429 × 657953
# sc.pp.filter_cells(atac, max_genes=50000)    # 2424 × 657953
np.count_nonzero(atac.X.toarray(), axis=1).max()   # 60239

idx = np.intersect1d(rna.obs.index, atac.obs.index)
del rna.var['n_cells']
rna[idx, :].copy().write('spatial_rna_filtered.h5ad')     # 2500 × 13997
del atac.var['n_cells']
atac[idx, :].copy().write('spatial_atac_filtered.h5ad')   # 2500 × 657953

rna = rna[idx, :].copy()
atac = atac[idx, :].copy()

rna_ref = sc.read_h5ad('./rna.h5ad')   # 2597 × 17699
genes = rna_ref.var[['gene_ids']].copy()
genes['gene_name'] = genes.index.values
genes.reset_index(drop=True, inplace=True)
rna_exp = pd.concat([rna.var, pd.DataFrame(rna.X.toarray().T, index=rna.var.index)], axis=1)
X_new = pd.merge(genes, rna_exp, how='left', on='gene_ids').iloc[:, 3:].T
X_new.fillna(value=0, inplace=True)
rna_new = sc.AnnData(X_new.values, obs=rna.obs, var=pd.DataFrame({'gene_ids': genes['gene_ids'], 'feature_types': 'Gene Expression'})) 
# rna_new = sc.AnnData(X_new.values, obs=rna.obs[['cell_anno']], var=pd.DataFrame({'gene_ids': genes['gene_ids'], 'feature_types': 'Gene Expression'}))
rna_new.var.index = genes['gene_name'].values
rna_new.X = csr_matrix(rna_new.X)   # 2500 × 17699
rna_new.write('spatial_rna_test.h5ad')
np.count_nonzero(rna_new.X.toarray(), axis=1).max()  # 6423

## atac
atac_ref = sc.read_h5ad('./atac.h5ad')   # 2597 × 217893

peaks = atac_ref.var[['gene_ids']].copy()
peaks['gene_name'] = peaks.index.values
peaks.reset_index(drop=True, inplace=True)
atac_exp = pd.concat([atac.var, pd.DataFrame(atac.X.T, index=atac.var.index)], axis=1)
X_new = pd.merge(peaks, atac_exp, how='left', on='gene_ids').iloc[:, 3:].T
X_new.fillna(value=0, inplace=True)
atac_new = sc.AnnData(X_new.values, obs=atac.obs, var=pd.DataFrame({'gene_ids': peaks['gene_ids'], 'feature_types': 'Peaks'}))
# atac_new = sc.AnnData(X_new.values, obs=atac.obs[['cell_anno']], var=pd.DataFrame({'gene_ids': peaks['gene_ids'], 'feature_types': 'Peaks'}))
atac_new.var.index = peaks['gene_name'].values
atac_new.X = csr_matrix(atac_new.X)   # 2500 × 217893
atac_new.write('spatial_atac_test.h5ad')
########################################################################################################################################################################

# #### clustering for true with atac
# import snapatac2 as snap
# import scanpy as sc

# atac = sc.read_h5ad('../spatial_atac_test.h5ad')
# snap.pp.select_features(atac)
# snap.tl.spectral(atac)
# snap.tl.umap(atac)
# snap.pp.knn(atac)
# snap.tl.leiden(atac)
# atac.obs['cell_anno'] = atac.obs['leiden']
# # sc.pl.umap(atac, color='cell_anno', legend_fontsize='7', legend_loc='right margin', size=10,
# #            title='', frameon=True, save='_atac_test_true.pdf')
# atac.write('atac_test.h5ad')


#### clustering for true with rna
import scanpy as sc

rna = sc.read_h5ad('spatial_rna_test.h5ad')
sc.pp.normalize_total(rna, target_sum=1e4)
sc.pp.log1p(rna)
sc.pp.highly_variable_genes(rna)
rna = rna[:, rna.var.highly_variable]
sc.tl.pca(rna)
sc.pp.neighbors(rna)
sc.tl.umap(rna)
sc.tl.leiden(rna, resolution=0.9)
rna.obs['leiden'].value_counts()
# 0    710
# 1    518
# 2    367
# 3    326
# 4    294
# 5    155
# 6     68
# 7     62

rna.obs['cell_anno'] = rna.obs['leiden']
rna.write('rna_test.h5ad')

atac = sc.read_h5ad('spatial_atac_test.h5ad')
atac.obs['cell_anno'] = rna.obs['leiden']
atac.write('atac_test.h5ad')

## spatial evaluation
python data_preprocess.py -r spatial_rna_test.h5ad -a spatial_atac_test.h5ad -s spatial_test_pt --dt test --config rna2atac_config_test.yaml
accelerate launch --config_file accelerator_config_test.yaml --main_process_port 29822 rna2atac_predict.py \
                  -d ./spatial_test_pt -l ./save/2025-01-15_rna2atac_brain_15/pytorch_model.bin --config_file rna2atac_config_test.yaml
python npy2h5ad.py
python cal_auroc_auprc.py --pred atac_cisformer.h5ad --true atac_test.h5ad
# Cell-wise AUROC: 0.6665
# Cell-wise AUPRC: 0.0285
# Peak-wise AUROC: 0.5597
# Peak-wise AUPRC: 0.0169
python plot_save_umap.py --pred atac_cisformer.h5ad --true atac_test.h5ad
python cal_cluster.py --file atac_cisformer_umap.h5ad
# AMI: 0.3189
# ARI: 0.1971
# HOM: 0.3489
# NMI: 0.3230

# model = M2M_rna2atac(
#             dim = 210,

#             enc_num_gene_tokens = 17699 + 1, # +1 for <PAD>
#             enc_num_value_tokens = 64 + 1, # +1 for <PAD>
#             enc_depth = 0,
#             enc_heads = 1,
#             enc_ff_mult = 4,
#             enc_dim_head = 128,
#             enc_emb_dropout = 0.1,
#             enc_ff_dropout = 0.1,
#             enc_attn_dropout = 0.1,

#             dec_depth = 10,
#             dec_heads = 10,
#             dec_ff_mult = 10,
#             dec_dim_head = 128,
#             dec_emb_dropout = 0.1,
#             dec_ff_dropout = 0.1,
#             dec_attn_dropout = 0.1
#         )

# param_sum = 0
# for name, param in model.named_parameters():
#     if param.requires_grad:
#         param_sum += np.prod(list(param.shape))

# print(param_sum)

#### plot spatial pattern for ground truth
# cp /fs/home/jiluzhang/2023_nature_RF/human_brain/HumanBrain_50um_spatial/tissue_positions_list.csv .
# barcode    in_tissue    array_row    array_column    pxl_col_in_fullres    pxl_row_in_fullres

import scanpy as sc
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

plt.rcParams['pdf.fonttype'] = 42

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
sns.heatmap(mat_thy1, cmap='Reds', xticklabels=False, yticklabels=False, vmin=0, vmax=10)  # cbar=False, vmin=0, vmax=10, linewidths=0.1, linecolor='grey'
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
sns.heatmap(mat_bcl11b, cmap='Reds', xticklabels=False, yticklabels=False, vmin=0, vmax=9)  # cbar=False, vmin=0, vmax=10, linewidths=0.1, linecolor='grey'
plt.savefig('ATAC_BCL11B_true.pdf')
plt.close()


#### plot spatial pattern for cisformer prediction
import scanpy as sc
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

plt.rcParams['pdf.fonttype'] = 42

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
sns.heatmap(mat_thy1, cmap='Reds', xticklabels=False, yticklabels=False, vmin=0, vmax=25)  # cbar=False, vmin=0, vmax=10, linewidths=0.1, linecolor='grey'
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
sns.heatmap(mat_bcl11b, cmap='Reds', xticklabels=False, yticklabels=False, vmin=0, vmax=27)  # cbar=False, vmin=0, vmax=10, linewidths=0.1, linecolor='grey'
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

df_true = pd.DataFrame({'cell_idx':atac_true.obs.index, 'cell_anno':atac_true.obs.cell_anno.astype('int')})
df_pos_true = pd.merge(df_true, pos)

mat_true = np.zeros([50, 50])
for i in range(df_pos_true.shape[0]):
    if df_pos_true['cell_anno'][i]!=0:
        mat_true[df_pos_true['X'][i], (49-df_pos_true['Y'][i])] = df_pos_true['cell_anno'][i]  # start from upper right

plt.figure(figsize=(6, 5))
sns.heatmap(mat_true, cmap=sns.color_palette(['grey', 'cornflowerblue', 'green', 'purple', 'orange', 'red', 'yellow', 'blue']), xticklabels=False, yticklabels=False)
plt.savefig('ATAC_spatial_true.pdf')
plt.close()

## plot for cisformer
atac_cisformer = sc.read_h5ad('atac_cisformer_umap.h5ad')

snap.pp.knn(atac_cisformer)
snap.tl.leiden(atac_cisformer, resolution=0.85)  # keep cluster number to 8
atac_cisformer.obs['leiden'].value_counts()
# 0    425
# 1    388
# 2    376
# 3    375
# 4    325
# 5    267
# 6    252
# 7     92

df_cisformer = pd.DataFrame({'cell_idx':atac_cisformer.obs.index, 'cell_anno':atac_cisformer.obs.leiden.astype('int')})
df_pos_cisformer = pd.merge(df_cisformer, pos)

mat_cisformer = np.zeros([50, 50])
for i in range(df_pos_cisformer.shape[0]):
    if df_pos_cisformer['cell_anno'][i]!=0:
        mat_cisformer[df_pos_cisformer['X'][i], (49-df_pos_cisformer['Y'][i])] = df_pos_cisformer['cell_anno'][i]  # start from upper right

plt.figure(figsize=(6, 5))
sns.heatmap(mat_cisformer, cmap=sns.color_palette(['grey', 'cornflowerblue', 'green', 'red', 'orange', 'blue', 'yellow', 'purple']), xticklabels=False, yticklabels=False)
plt.savefig('ATAC_spatial_cisformer.pdf')
plt.close()

from sklearn.metrics import normalized_mutual_info_score

normalized_mutual_info_score(atac_cisformer.obs['cell_anno'], atac_cisformer.obs['leiden']  # 0.31030679154742474


#### scbutterfly
nohup python scbt_b.py > scbt_b_20241226.log &   # 3280602

# cp predict/R2A.h5ad atac_scbt.h5ad
# python cal_auroc_auprc.py --pred atac_scbt.h5ad --true atac_test.h5ad
# python plot_save_umap.py --pred atac_scbt.h5ad --true atac_test.h5ad
# python cal_cluster.py --file atac_scbt_umap.h5ad

nohup python scbt_b_predict.py > scbt_b_predict.log &  # 559057
cp predict/R2A.h5ad atac_scbt.h5ad
python cal_auroc_auprc.py --pred atac_scbt.h5ad --true atac_test.h5ad
# Cell-wise AUROC: 0.6888
# Cell-wise AUPRC: 0.0296
# Peak-wise AUROC: 0.5998
# Peak-wise AUPRC: 0.026
python plot_save_umap.py --pred atac_scbt.h5ad --true atac_test.h5ad
python cal_cluster.py --file atac_scbt_umap.h5ad
# AMI: 0.3577
# ARI: 0.2096
# HOM: 0.4093
# NMI: 0.3621

#### plot spatial pattern for scbt prediction
import scanpy as sc
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import snapatac2 as snap

plt.rcParams['pdf.fonttype'] = 42

atac = sc.read_h5ad('atac_scbt_umap.h5ad')
atac_true = sc.read_h5ad('atac_test.h5ad')
atac.var.index = atac_true.var.index.values

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
sns.heatmap(mat_thy1, cmap='Reds', xticklabels=False, yticklabels=False, vmin=0, vmax=20)  # cbar=False, vmin=0, vmax=10, linewidths=0.1, linecolor='grey'
plt.savefig('ATAC_THY1_scbt.pdf')
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
sns.heatmap(mat_bcl11b, cmap='Reds', xticklabels=False, yticklabels=False, vmin=0, vmax=24)  # cbar=False, vmin=0, vmax=10, linewidths=0.1, linecolor='grey'
plt.savefig('ATAC_BCL11B_scbt.pdf')
plt.close()

## spatial
snap.pp.knn(atac)
snap.tl.leiden(atac, resolution=0.65)
atac.obs['leiden'].value_counts()
# 0    510
# 1    478
# 2    375
# 3    361
# 4    317
# 5    221
# 6    173
# 7     65

from sklearn.metrics import normalized_mutual_info_score

normalized_mutual_info_score(atac.obs['cell_anno'], atac.obs['leiden']  # 0.3760858556890463

df_scbt = pd.DataFrame({'cell_idx':atac.obs.index, 'cell_anno':atac.obs.leiden.astype('int')})
df_pos_scbt = pd.merge(df_scbt, pos)

mat_scbt = np.zeros([50, 50])
for i in range(df_pos_scbt.shape[0]):
    if df_pos_scbt['cell_anno'][i]!=0:
        mat_scbt[df_pos_scbt['X'][i], (49-df_pos_scbt['Y'][i])] = df_pos_scbt['cell_anno'][i]  # start from upper right

plt.figure(figsize=(6, 5))
sns.heatmap(mat_scbt, cmap=sns.color_palette(['grey', 'cornflowerblue', 'green', 'orange', 'red', 'purple', 'yellow', 'blue']), xticklabels=False, yticklabels=False)
plt.savefig('ATAC_spatial_scbt.pdf')
plt.close()


#### BABEL
python h5ad2h5.py -n train_val
nohup python /fs/home/jiluzhang/BABEL/bin/train_model.py --data train_val.h5 --outdir train_out --batchsize 512 --earlystop 25 --device 6 --nofilter > train_20241226.log &  # 3384749

# python h5ad2h5.py -n test
# nohup python /fs/home/jiluzhang/BABEL/bin/predict_model.py --checkpoint train_out --data test.h5 --outdir test_out --device 1 --nofilter --noplot --transonly > predict_20241226.log &
# # 3582789
# python match_cell.py
# python cal_auroc_auprc.py --pred atac_babel.h5ad --true atac_test.h5ad
# python plot_save_umap.py --pred atac_babel.h5ad --true atac_test.h5ad
# python cal_cluster.py --file atac_babel_umap.h5ad

python h5ad2h5.py -n test
nohup python /fs/home/jiluzhang/BABEL/bin/predict_model.py --checkpoint /fs/home/jiluzhang/Nature_methods/Spatial/babel/train_out --data test.h5 --outdir test_out \
                                                           --device 1 --nofilter --noplot --transonly > predict_20250114.log &   # 569614
python match_cell.py
python cal_auroc_auprc.py --pred atac_babel.h5ad --true atac_test.h5ad
# Cell-wise AUROC: 0.6889
# Cell-wise AUPRC: 0.0296
# Peak-wise AUROC: 0.6082
# Peak-wise AUPRC: 0.0261
python plot_save_umap.py --pred atac_babel.h5ad --true atac_test.h5ad
python cal_cluster.py --file atac_babel_umap.h5ad
# AMI: 0.3448
# ARI: 0.2020
# HOM: 0.3893
# NMI: 0.3490


#### plot spatial pattern for babel prediction
import scanpy as sc
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import snapatac2 as snap

atac = sc.read_h5ad('atac_babel_umap.h5ad')
atac_true = sc.read_h5ad('atac_test.h5ad')
atac.var.index = atac_true.var.index.values

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
sns.heatmap(mat_thy1, cmap='Reds', xticklabels=False, yticklabels=False, vmin=0, vmax=14)  # cbar=False, vmin=0, vmax=10, linewidths=0.1, linecolor='grey'
plt.savefig('ATAC_THY1_babel.pdf')
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
sns.heatmap(mat_bcl11b, cmap='Reds', xticklabels=False, yticklabels=False, vmin=0, vmax=3)  # cbar=False, vmin=0, vmax=10, linewidths=0.1, linecolor='grey'
plt.savefig('ATAC_BCL11B_babel.pdf')
plt.close()


from scipy import stats
stats.pearsonr(mat_thy1_scbt.flatten(), mat_thy1_true.flatten())[0]       # 0.3287928872046775
stats.pearsonr(mat_thy1_cifm.flatten(), mat_thy1_true.flatten())[0]       # 0.23474590003006277
stats.pearsonr(mat_thy1_babel.flatten(), mat_thy1_true.flatten())[0]       # 0.20658652164925642
stats.pearsonr(mat_bcl11b_scbt.flatten(), mat_bcl11b_true.flatten())[0]   # 0.17057192864978218
stats.pearsonr(mat_bcl11b_cifm.flatten(), mat_bcl11b_true.flatten())[0]   # 0.2621234653897852
stats.pearsonr(mat_bcl11b_babel.flatten(), mat_bcl11b_true.flatten())[0]   # 0.40318798708760156

## spatial
snap.pp.knn(atac)
snap.tl.leiden(atac, resolution=0.75)
atac.obs['leiden'].value_counts()
# 0    468
# 1    431
# 2    412
# 3    405
# 4    324
# 5    238
# 6    157
# 7     65

df_babel = pd.DataFrame({'cell_idx':atac.obs.index, 'cell_anno':atac.obs.leiden.astype('int')})
df_pos_babel = pd.merge(df_babel, pos)

mat_babel = np.zeros([50, 50])
for i in range(df_pos_babel.shape[0]):
    if df_pos_babel['cell_anno'][i]!=0:
        mat_babel[df_pos_babel['X'][i], (49-df_pos_babel['Y'][i])] = df_pos_babel['cell_anno'][i]  # start from upper right

plt.figure(figsize=(6, 5))
sns.heatmap(mat_babel, cmap=sns.color_palette(['grey', 'red', 'green', 'orange', 'cornflowerblue', 'purple', 'yellow', 'blue']), xticklabels=False, yticklabels=False)
plt.savefig('ATAC_spatial_babel.pdf')
plt.close()

from sklearn.metrics import normalized_mutual_info_score

normalized_mutual_info_score(atac.obs['cell_anno'], atac.obs['leiden']  # 0.3444513808689984
