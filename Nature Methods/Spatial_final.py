## workdir: /fs/home/jiluzhang/Nature_methods/Spatial_final
# cp /fs/home/jiluzhang/Nature_methods/Figure_1/scenario_4/rna.h5ad rna.h5ad     # 2597 × 17699
# cp /fs/home/jiluzhang/Nature_methods/Figure_1/scenario_4/atac.h5ad atac.h5ad   # 2597 × 217893

python split_train_val.py --RNA rna.h5ad --ATAC atac.h5ad --train_pct 0.9
python data_preprocess.py -r rna_train.h5ad -a atac_train.h5ad -s train_pt --dt train -n train --config rna2atac_config_train.yaml  # multiple=100
python data_preprocess.py -r rna_val.h5ad -a atac_val.h5ad -s val_pt --dt val -n val --config rna2atac_config_val.yaml
nohup accelerate launch --config_file accelerator_config_train.yaml --main_process_port 29824 rna2atac_train.py --config_file rna2atac_config_train.yaml \
                        --train_data_dir train_pt --val_data_dir val_pt -s save -n rna2atac_brain > training_20250121.log &   # 1835055


# cp /fs/home/jiluzhang/2023_nature_RF/human_brain/spatial_rna.h5ad .
# cp /fs/home/jiluzhang/2023_nature_RF/human_brain/spatial_atac.h5ad .

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

# idx = np.intersect1d(rna.obs.index, atac.obs.index)
# del rna.var['n_cells']
# rna[idx, :].copy().write('spatial_rna_filtered.h5ad')     # 2500 × 13997
# del atac.var['n_cells']
# atac[idx, :].copy().write('spatial_atac_filtered.h5ad')   # 2500 × 657953

# rna = rna[idx, :].copy()
# atac = atac[idx, :].copy()

rna_ref = sc.read_h5ad('./rna.h5ad')   # 2597 × 17699
genes = rna_ref.var[['gene_ids']].copy()
genes['gene_name'] = genes.index.values
genes.reset_index(drop=True, inplace=True)
rna_exp = pd.concat([rna.var, pd.DataFrame(rna.X.toarray().T, index=rna.var.index)], axis=1)
X_new = pd.merge(genes, rna_exp, how='left', on='gene_ids').iloc[:, 4:].T
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
atac_exp = pd.concat([atac.var, pd.DataFrame(atac.X.toarray().T, index=atac.var.index)], axis=1)
X_new = pd.merge(peaks, atac_exp, how='left', on='gene_ids').iloc[:, 4:].T
X_new.fillna(value=0, inplace=True)
atac_new = sc.AnnData(X_new.values, obs=atac.obs, var=pd.DataFrame({'gene_ids': peaks['gene_ids'], 'feature_types': 'Peaks'}))
atac_new.var.index = peaks['gene_name'].values
atac_new.X = csr_matrix(atac_new.X)   # 2500 × 217893
atac_new.write('spatial_atac_test.h5ad')
########################################################################################################################################################################

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
sc.tl.leiden(rna, resolution=0.85)
rna.obs['leiden'].value_counts()
# 0    685
# 1    511
# 2    344
# 3    330
# 4    279
# 5    157
# 6    133
# 7     61

rna.obs['cell_anno'] = rna.obs['leiden']
rna.write('rna_test.h5ad')

atac = sc.read_h5ad('spatial_atac_test.h5ad')
atac.obs['cell_anno'] = rna.obs['leiden']
atac.write('atac_test.h5ad')


## spatial evaluation
## epoch=3
python data_preprocess.py -r spatial_rna_test.h5ad -a spatial_atac_test.h5ad -s spatial_test_pt --dt test --config rna2atac_config_test.yaml
accelerate launch --config_file accelerator_config_test.yaml --main_process_port 29822 rna2atac_predict.py \
                  -d ./spatial_test_pt -l ./save/2025-01-21_rna2atac_brain_3/pytorch_model.bin --config_file rna2atac_config_test.yaml
mv ../predict.npy .
python npy2h5ad.py
python cal_auroc_auprc.py --pred atac_cisformer.h5ad --true atac_test.h5ad
# Cell-wise AUROC: 0.6799
# Cell-wise AUPRC: 0.0289
# Peak-wise AUROC: 0.5678
# Peak-wise AUPRC: 0.0167
python plot_save_umap.py --pred atac_cisformer.h5ad --true atac_test.h5ad
python cal_cluster.py --file atac_cisformer_umap.h5ad
# AMI: 0.2687
# ARI: 0.1735
# HOM: 0.2833
# NMI: 0.2725

## epoch=5
#python data_preprocess.py -r spatial_rna_test.h5ad -a spatial_atac_test.h5ad -s spatial_test_pt --dt test --config rna2atac_config_test.yaml
accelerate launch --config_file accelerator_config_test.yaml --main_process_port 29822 rna2atac_predict.py \
                  -d ./spatial_test_pt -l ./save/2025-01-21_rna2atac_brain_5/pytorch_model.bin --config_file rna2atac_config_test.yaml
mv ../predict.npy .
python npy2h5ad.py
python cal_auroc_auprc.py --pred atac_cisformer.h5ad --true atac_test.h5ad
# Cell-wise AUROC: 0.6820
# Cell-wise AUPRC: 0.0292
# Peak-wise AUROC: 0.5736
# Peak-wise AUPRC: 0.0170
python plot_save_umap.py --pred atac_cisformer.h5ad --true atac_test.h5ad
python cal_cluster.py --file atac_cisformer_umap.h5ad
# AMI: 0.2656
# ARI: 0.1688
# HOM: 0.2794
# NMI: 0.2693

## epoch=10
#python data_preprocess.py -r spatial_rna_test.h5ad -a spatial_atac_test.h5ad -s spatial_test_pt --dt test --config rna2atac_config_test.yaml
accelerate launch --config_file accelerator_config_test.yaml --main_process_port 29822 rna2atac_predict.py \
                  -d ./spatial_test_pt -l ./save/2025-01-21_rna2atac_brain_10/pytorch_model.bin --config_file rna2atac_config_test.yaml
mv ../predict.npy .
python npy2h5ad.py
python cal_auroc_auprc.py --pred atac_cisformer.h5ad --true atac_test.h5ad
# Cell-wise AUROC: 0.6826
# Cell-wise AUPRC: 0.0293
# Peak-wise AUROC: 0.5729
# Peak-wise AUPRC: 0.0170
python plot_save_umap.py --pred atac_cisformer.h5ad --true atac_test.h5ad
python cal_cluster.py --file atac_cisformer_umap.h5ad
# AMI: 0.2664
# ARI: 0.1516
# HOM: 0.29
# NMI: 0.2707

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
sns.heatmap(mat_thy1, cmap='Reds', xticklabels=False, yticklabels=False, vmin=0, vmax=mat_thy1.flatten().max())  # cbar=False, vmin=0, vmax=10, linewidths=0.1, linecolor='grey'
plt.savefig('ATAC_THY1_true.pdf')
plt.close()

np.save('mat_thy1_true', mat_thy1)

## BCL11B
df_bcl11b = pd.DataFrame({'cell_idx':atac.obs.index, 
                          'peak':atac.X[:, ((atac.var['chrom']=='chr14') & ((abs((atac.var['start']+atac.var['end'])*0.5-99272197)<50000)))].toarray().sum(axis=1)})
df_bcl11b_pos = pd.merge(df_bcl11b, pos)

mat_bcl11b = np.zeros([50, 50])
for i in range(df_bcl11b_pos.shape[0]):
    if df_bcl11b_pos['peak'][i]!=0:
        mat_bcl11b[df_bcl11b_pos['X'][i], (49-df_bcl11b_pos['Y'][i])] = df_bcl11b_pos['peak'][i]  # start from upper right

plt.figure(figsize=(6, 5))
sns.heatmap(mat_bcl11b, cmap='Reds', xticklabels=False, yticklabels=False, vmin=0, vmax=mat_bcl11b.flatten().max())  # cbar=False, vmin=0, vmax=10, linewidths=0.1, linecolor='grey'
plt.savefig('ATAC_BCL11B_true.pdf')
plt.close()

np.save('mat_bcl11b_true', mat_bcl11b)

## spatial
df_true = pd.DataFrame({'cell_idx':atac.obs.index, 'cell_anno':atac.obs.cell_anno.astype('int')})
df_pos_true = pd.merge(df_true, pos)

mat_true = np.zeros([50, 50])
for i in range(df_pos_true.shape[0]):
    if df_pos_true['cell_anno'][i]!=0:
        mat_true[df_pos_true['X'][i], (49-df_pos_true['Y'][i])] = df_pos_true['cell_anno'][i]  # start from upper right

plt.figure(figsize=(6, 5))
sns.heatmap(mat_true, cmap=sns.color_palette(['grey', 'cornflowerblue', 'green', 'purple', 'orange', 'red', 'yellow', 'blue']), xticklabels=False, yticklabels=False)
plt.savefig('ATAC_spatial_true.pdf')
plt.close()

#### plot spatial pattern for cisformer prediction
import scanpy as sc
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import snapatac2 as snap
from sklearn.metrics import normalized_mutual_info_score

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
sns.heatmap(mat_thy1, cmap='Reds', xticklabels=False, yticklabels=False, vmin=0, vmax=mat_thy1.flatten().max())  # cbar=False, vmin=0, vmax=10, linewidths=0.1, linecolor='grey'
plt.savefig('ATAC_THY1_cisformer.pdf')
plt.close()

np.save('mat_thy1_cisformer', mat_thy1)

## BCL11B
df_bcl11b = pd.DataFrame({'cell_idx':atac.obs.index, 
                          'peak':atac.X[:, ((atac.var['chrom']=='chr14') & ((abs((atac.var['start']+atac.var['end'])*0.5-99272197)<50000)))].toarray().sum(axis=1)})
df_bcl11b_pos = pd.merge(df_bcl11b, pos)

mat_bcl11b = np.zeros([50, 50])
for i in range(df_bcl11b_pos.shape[0]):
    if df_bcl11b_pos['peak'][i]!=0:
        mat_bcl11b[df_bcl11b_pos['X'][i], (49-df_bcl11b_pos['Y'][i])] = df_bcl11b_pos['peak'][i]  # start from upper right

plt.figure(figsize=(6, 5))
sns.heatmap(mat_bcl11b, cmap='Reds', xticklabels=False, yticklabels=False, vmin=0, vmax=mat_bcl11b.flatten().max())  # cbar=False, vmin=0, vmax=10, linewidths=0.1, linecolor='grey'
plt.savefig('ATAC_BCL11B_cisformer.pdf')
plt.close()

np.save('mat_bcl11b_cisformer', mat_bcl11b)

## spatial
snap.pp.knn(atac)
snap.tl.leiden(atac, resolution=0.9)
atac.obs['leiden'].value_counts()
# 0    480
# 1    391
# 2    332
# 3    281
# 4    278
# 5    262
# 6    255
# 7    221

df_cifm = pd.DataFrame({'cell_idx':atac.obs.index, 'cell_anno':atac.obs.leiden.astype('int')})
df_pos_cifm = pd.merge(df_cifm, pos)

mat_cifm = np.zeros([50, 50])
for i in range(df_pos_cifm.shape[0]):
    if df_pos_cifm['cell_anno'][i]!=0:
        mat_cifm[df_pos_cifm['X'][i], (49-df_pos_cifm['Y'][i])] = df_pos_cifm['cell_anno'][i]  # start from upper right

plt.figure(figsize=(6, 5))
sns.heatmap(mat_cifm, cmap=sns.color_palette(['grey', 'yellow', 'green', 'red', 'blue', 'purple', 'orange', 'cornflowerblue']), xticklabels=False, yticklabels=False)
plt.savefig('ATAC_spatial_cisformer.pdf')
plt.close()

normalized_mutual_info_score(atac.obs['cell_anno'], atac.obs['leiden'])  # 0.28876717756132614

## correlation
from scipy import stats
import numpy as np

stats.pearsonr(np.load('mat_thy1_cisformer.npy').flatten(), np.load('mat_thy1_true.npy').flatten())[0]       # 0.08258100908394941
stats.pearsonr(np.load('mat_bcl11b_cisformer.npy').flatten(), np.load('mat_bcl11b_true.npy').flatten())[0]   # 0.1365499864406945


#### scbutterfly
# /data/home/jiluzhang/miniconda3/envs/scButterfly/lib/python3.9/site-packages/scButterfly/train_model.py
# line 329: "reconstruct_loss = r_loss_w * (lossR2R + lossA2R) + a_loss_w * (lossR2A + lossA2A)" -> "reconstruct_loss = lossR2A"
# line 331: "kl_div = kl_div_r + kl_div_a" -> "kl_div = kl_div_r"
# line 374: "loss2 = d_loss(predict_rna.reshape(batch_size), torch.tensor(temp).cuda().float())" -> "loss2 = 0"
# butterfly.train_model() parameters: R2R_pretrain_epoch=0, A2A_pretrain_epoch=0, translation_kl_warmup=0, patience=10

# /data/home/jiluzhang/miniconda3/envs/scButterfly/lib/python3.9/site-packages/scButterfly/butterfly.py
# line 338: "R_encoder_dim_list = [256, 128]," -> "R_encoder_dim_list = [64, 16]," 
# line 339: "A_encoder_dim_list = [32, 128]," -> "A_encoder_dim_list = [32, 16],"
# line 340: "R_decoder_dim_list = [128, 256]," -> "R_decoder_dim_list = [16, 64]," 
# line 341: "A_decoder_dim_list = [128, 32]," ->  "A_decoder_dim_list = [16, 32],"
# line 346: "translator_embed_dim = 128, " -> "translator_embed_dim = 16,"
# line 347: "translator_input_dim_r = 128," -> "translator_input_dim_r = 16,"
# line 348: "translator_input_dim_a = 128," -> "translator_input_dim_a = 16,"
# line 351: "discriminator_dim_list_R = [128]," -> "discriminator_dim_list_R = [16],"
# line 352: "discriminator_dim_list_A = [128]," -> "discriminator_dim_list_A = [16],"

nohup python scbt_b.py > scbt_b_20250121.log &   # 1633652
mv predict predict_val  # avoid no output
python scbt_b_predict.py
cp predict/R2A.h5ad atac_scbt.h5ad
python cal_auroc_auprc.py --pred atac_scbt.h5ad --true atac_test.h5ad
# Cell-wise AUROC: 0.6743
# Cell-wise AUPRC: 0.0284
# Peak-wise AUROC: 0.6434
# Peak-wise AUPRC: 0.0335
python plot_save_umap.py --pred atac_scbt.h5ad --true atac_test.h5ad
python cal_cluster.py --file atac_scbt_umap.h5ad
# AMI: 0.2656
# ARI: 0.1217
# HOM: 0.3202
# NMI: 0.2716

#### plot spatial pattern for scbt prediction
import scanpy as sc
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import snapatac2 as snap
from sklearn.metrics import normalized_mutual_info_score

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
sns.heatmap(mat_thy1, cmap='Reds', xticklabels=False, yticklabels=False, vmin=0, vmax=mat_thy1.flatten().max())  # cbar=False, vmin=0, vmax=10, linewidths=0.1, linecolor='grey'
plt.savefig('ATAC_THY1_scbt.pdf')
plt.close()

np.save('mat_thy1_scbt', mat_thy1)

## BCL11B
df_bcl11b = pd.DataFrame({'cell_idx':atac.obs.index, 
                          'peak':atac.X[:, ((atac.var['chrom']=='chr14') & ((abs((atac.var['start']+atac.var['end'])*0.5-99272197)<50000)))].toarray().sum(axis=1)})
df_bcl11b_pos = pd.merge(df_bcl11b, pos)

mat_bcl11b = np.zeros([50, 50])
for i in range(df_bcl11b_pos.shape[0]):
    if df_bcl11b_pos['peak'][i]!=0:
        mat_bcl11b[df_bcl11b_pos['X'][i], (49-df_bcl11b_pos['Y'][i])] = df_bcl11b_pos['peak'][i]  # start from upper right

plt.figure(figsize=(6, 5))
sns.heatmap(mat_bcl11b, cmap='Reds', xticklabels=False, yticklabels=False, vmin=0, vmax=mat_bcl11b.flatten().max())  # cbar=False, vmin=0, vmax=10, linewidths=0.1, linecolor='grey'
plt.savefig('ATAC_BCL11B_scbt.pdf')
plt.close()

np.save('mat_bcl11b_scbt', mat_bcl11b)

## spatial
snap.pp.knn(atac)
snap.tl.leiden(atac, resolution=0.37)
atac.obs['leiden'].value_counts()
# 0    449
# 1    393
# 2    344
# 3    317
# 4    315
# 5    278
# 6    261
# 7    143

df_scbt = pd.DataFrame({'cell_idx':atac.obs.index, 'cell_anno':atac.obs.leiden.astype('int')})
df_pos_scbt = pd.merge(df_scbt, pos)

mat_scbt = np.zeros([50, 50])
for i in range(df_pos_scbt.shape[0]):
    if df_pos_scbt['cell_anno'][i]!=0:
        mat_scbt[df_pos_scbt['X'][i], (49-df_pos_scbt['Y'][i])] = df_pos_scbt['cell_anno'][i]  # start from upper right

plt.figure(figsize=(6, 5))
sns.heatmap(mat_scbt, cmap=sns.color_palette(['blue', 'red', 'green', 'orange', 'grey', 'purple', 'yellow', 'cornflowerblue']), xticklabels=False, yticklabels=False)
plt.savefig('ATAC_spatial_scbt.pdf')
plt.close()

normalized_mutual_info_score(atac.obs['cell_anno'], atac.obs['leiden'])  # 0.2517963872767232

## correlation
from scipy import stats
import numpy as np

stats.pearsonr(np.load('mat_thy1_scbt.npy').flatten(), np.load('mat_thy1_true.npy').flatten())[0]       # 0.33269823405427934
stats.pearsonr(np.load('mat_bcl11b_scbt.npy').flatten(), np.load('mat_bcl11b_true.npy').flatten())[0]   # 0.26719161941600833


#### babel
## /fs/home/jiluzhang/BABEL/babel/loss_functions.py
## line 430: "loss = loss11 + self.loss2_weight * loss22" -> "loss = 0*loss11 + 0*self.loss2_weight * loss22"
## line 431: "loss += next(self.cross_warmup) * (loss21 + self.loss2_weight * loss12)" -> "loss += next(self.cross_warmup) * (0*loss21 + 0*self.loss2_weight * loss12 + loss12)"

cp ../rna.h5ad rna_train_val.h5ad
cp ../atac.h5ad atac_train_val.h5ad
python h5ad2h5.py -n train_val   # train + val -> train_val
nohup python /fs/home/jiluzhang/BABEL/bin/train_model.py --data train_val.h5 --outdir train_out --batchsize 512 --earlystop 10 --device 0 --nofilter > train_20250121.log &  # 1655108

python h5ad2h5.py -n test
python /fs/home/jiluzhang/BABEL/bin/predict_model.py --checkpoint ./train_out --data test.h5 --outdir test_out --device 0 --nofilter --noplot --transonly
python match_cell.py
python cal_auroc_auprc.py --pred atac_babel.h5ad --true atac_test.h5ad
# Cell-wise AUROC: 0.6820
# Cell-wise AUPRC: 0.0289
# Peak-wise AUROC: 0.6591
# Peak-wise AUPRC: 0.0350
python plot_save_umap.py --pred atac_babel.h5ad --true atac_test.h5ad
python cal_cluster.py --file atac_babel_umap.h5ad
# AMI: 0.1952
# ARI: 0.0916
# HOM: 0.2239
# NMI: 0.2005


#### plot spatial pattern for babel prediction
import scanpy as sc
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import snapatac2 as snap
from sklearn.metrics import normalized_mutual_info_score

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
sns.heatmap(mat_thy1, cmap='Reds', xticklabels=False, yticklabels=False, vmin=0, vmax=mat_thy1.flatten().max())  # cbar=False, vmin=0, vmax=10, linewidths=0.1, linecolor='grey'
plt.savefig('ATAC_THY1_babel.pdf')
plt.close()

np.save('mat_thy1_babel', mat_thy1)

## BCL11B
df_bcl11b = pd.DataFrame({'cell_idx':atac.obs.index, 
                          'peak':atac.X[:, ((atac.var['chrom']=='chr14') & ((abs((atac.var['start']+atac.var['end'])*0.5-99272197)<50000)))].toarray().sum(axis=1)})
df_bcl11b_pos = pd.merge(df_bcl11b, pos)

mat_bcl11b = np.zeros([50, 50])
for i in range(df_bcl11b_pos.shape[0]):
    if df_bcl11b_pos['peak'][i]!=0:
        mat_bcl11b[df_bcl11b_pos['X'][i], (49-df_bcl11b_pos['Y'][i])] = df_bcl11b_pos['peak'][i]  # start from upper right

plt.figure(figsize=(6, 5))
sns.heatmap(mat_bcl11b, cmap='Reds', xticklabels=False, yticklabels=False, vmin=0, vmax=mat_bcl11b.flatten().max())  # cbar=False, vmin=0, vmax=10, linewidths=0.1, linecolor='grey'
plt.savefig('ATAC_BCL11B_babel.pdf')
plt.close()

np.save('mat_bcl11b_babel', mat_bcl11b)

## spatial
snap.pp.knn(atac)
snap.tl.leiden(atac, resolution=0.55)
atac.obs['leiden'].value_counts()
# 0    422
# 1    399
# 2    368
# 3    368
# 4    291
# 5    240
# 6    233
# 7    179

df_babel = pd.DataFrame({'cell_idx':atac.obs.index, 'cell_anno':atac.obs.leiden.astype('int')})
df_pos_babel = pd.merge(df_babel, pos)

mat_babel = np.zeros([50, 50])
for i in range(df_pos_babel.shape[0]):
    if df_pos_babel['cell_anno'][i]!=0:
        mat_babel[df_pos_babel['X'][i], (49-df_pos_babel['Y'][i])] = df_pos_babel['cell_anno'][i]  # start from upper right

plt.figure(figsize=(6, 5))
sns.heatmap(mat_babel, cmap=sns.color_palette(['grey', 'yellow', 'green', 'orange', 'cornflowerblue', 'blue', 'red', 'purple']), xticklabels=False, yticklabels=False)
plt.savefig('ATAC_spatial_babel.pdf')
plt.close()

normalized_mutual_info_score(atac.obs['cell_anno'], atac.obs['leiden'])  # 0.20731315571838918

## correlation
from scipy import stats
import numpy as np

stats.pearsonr(np.load('mat_thy1_babel.npy').flatten(), np.load('mat_thy1_true.npy').flatten())[0]       # 0.1862186531052791
stats.pearsonr(np.load('mat_bcl11b_babel.npy').flatten(), np.load('mat_bcl11b_true.npy').flatten())[0]   # 0.29335627674715303


# cisformer: 0.31889501366384665    0.10369363444090407  0.19583682142803596  (large epoch 8)
# cisformer: 0.35816329600401065   0.22958961973311048   0.23425539546358495  (large epoch 2)
# cisformer: 0.3664339864145293    0.2260101020231856    0.18265064492508273  (large epoch 3)
# cisformer: 0.3362813762904754    0.21873923282570173    0.3351487391940996  (small epoch 10)
# cisformer: 0.352699532832734    0.17687696813899462    0.2141702425676784   (small epoch 20)
# scbt: 0.3136268806448111     0.29894978188093657     0.31026103815849715
# scbt: 0.34248733645998697    0.3793128531619313    0.3565337192047811  (modify loss again)
# scbt: 0.2741391818663338    0.3467068488459373    0.3684382758281502  (reduce embedding dimension from 128 to 16)
# babel: 0.23355932140479316   0.05806806333751863    0.05780821367095134
