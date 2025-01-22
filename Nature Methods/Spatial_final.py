#### GSE261983
#### Single-cell genomics and regulatory networks for 388 human brains

## wget -c https://ftp.ncbi.nlm.nih.gov/geo/series/GSE261nnn/GSE261983/suppl/GSE261983_RNA_gene_list.txt.gz
## wget -c https://ftp.ncbi.nlm.nih.gov/geo/series/GSE261nnn/GSE261983/suppl/GSE261983_ATAC_peakset.bed.gz

## RT00372N
## wget -c https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM8155nnn/GSM8155483/suppl/GSM8155483_RT00372N_RNA_barcodes.txt.gz
## wget -c https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM8155nnn/GSM8155483/suppl/GSM8155483_RT00372N_RNA_matrix.mtx.gz
## wget -c https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM8155nnn/GSM8155503/suppl/GSM8155503_RT00372N_ATAC_barcodes.txt.gz
## wget -c https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM8155nnn/GSM8155503/suppl/GSM8155503_RT00372N_ATAC_matrix.mtx.gz

## RT00374N
## wget -c https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM8155nnn/GSM8155485/suppl/GSM8155485_RT00374N_RNA_barcodes.txt.gz
## wget -c https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM8155nnn/GSM8155485/suppl/GSM8155485_RT00374N_RNA_matrix.mtx.gz
## wget -c https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM8155nnn/GSM8155505/suppl/GSM8155505_RT00374N_ATAC_barcodes.txt.gz
## wget -c https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM8155nnn/GSM8155505/suppl/GSM8155505_RT00374N_ATAC_matrix.mtx.gz

# zcat GSM8155503_RT00372N_ATAC_matrix.mtx.gz | head -n 2 >> RT00372N_ATAC_matrix.mtx
# zcat GSM8155503_RT00372N_ATAC_matrix.mtx.gz | tail -n +3 | awk '{print $2 "\t" $1 "\t" $3}' >> RT00372N_ATAC_matrix.mtx
# zcat GSM8155505_RT00374N_ATAC_matrix.mtx.gz | head -n 2 >> RT00374N_ATAC_matrix.mtx
# zcat GSM8155505_RT00374N_ATAC_matrix.mtx.gz | tail -n +3 | awk '{print $2 "\t" $1 "\t" $3}' >> RT00374N_ATAC_matrix.mtx

# zcat GSM8155483_RT00372N_RNA_matrix.mtx.gz | head -n 2 >> RT00372N_RNA_matrix.mtx
# zcat GSM8155483_RT00372N_RNA_matrix.mtx.gz | tail -n +3 | awk '{print $2 "\t" $1 "\t" $3}' >> RT00372N_RNA_matrix.mtx
# zcat GSM8155485_RT00374N_RNA_matrix.mtx.gz | head -n 2 >> RT00374N_RNA_matrix.mtx
# zcat GSM8155485_RT00374N_RNA_matrix.mtx.gz | tail -n +3 | awk '{print $2 "\t" $1 "\t" $3}' >> RT00374N_RNA_matrix.mtx

import scanpy as sc
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
import pybedtools
from tqdm import tqdm
import anndata as ad

rna_raw_rt00372n = sc.read_mtx('RT00372N_RNA_matrix.mtx')  # 8369 × 35649
rna_raw_rt00372n_barcode = pd.read_table('GSM8155483_RT00372N_RNA_barcodes.txt.gz', header=None)
rna_raw_rt00372n.obs.index = rna_raw_rt00372n_barcode[0].values
rna_raw_rt00372n_feature = pd.read_table('GSE261983_RNA_gene_list.txt.gz', header=None)
rna_raw_rt00372n.var.index = rna_raw_rt00372n_feature[0].values

rna_raw_rt00374n = sc.read_mtx('RT00374N_RNA_matrix.mtx')  # 7032 × 35649
rna_raw_rt00374n_barcode = pd.read_table('GSM8155485_RT00374N_RNA_barcodes.txt.gz', header=None)
rna_raw_rt00374n.obs.index = rna_raw_rt00374n_barcode[0].values
rna_raw_rt00374n_feature = pd.read_table('GSE261983_RNA_gene_list.txt.gz', header=None)
rna_raw_rt00374n.var.index = rna_raw_rt00374n_feature[0].values

rna_raw = ad.concat([rna_raw_rt00372n, rna_raw_rt00374n], axis=0)  # concat rna  15401 × 35649

atac_raw_rt00372n = sc.read_mtx('RT00372N_ATAC_matrix.mtx')  # 8984 × 423443
atac_raw_rt00372n_barcode = pd.read_table('GSM8155503_RT00372N_ATAC_barcodes.txt.gz', header=None)
atac_raw_rt00372n.obs.index = atac_raw_rt00372n_barcode[0].values
atac_raw_rt00372n_feature = pd.read_table('GSE261983_ATAC_peakset.bed.gz', header=None)
atac_raw_rt00372n_feature['idx'] = atac_raw_rt00372n_feature[0]+':'+atac_raw_rt00372n_feature[1].astype('str')+'-'+atac_raw_rt00372n_feature[2].astype('str')
atac_raw_rt00372n.var.index = atac_raw_rt00372n_feature['idx'].values

atac_raw_rt00374n = sc.read_mtx('RT00374N_ATAC_matrix.mtx')  # 7496 × 423443
atac_raw_rt00374n_barcode = pd.read_table('GSM8155505_RT00374N_ATAC_barcodes.txt.gz', header=None)
atac_raw_rt00374n.obs.index = atac_raw_rt00374n_barcode[0].values
atac_raw_rt00374n_feature = pd.read_table('GSE261983_ATAC_peakset.bed.gz', header=None)
atac_raw_rt00374n_feature['idx'] = atac_raw_rt00374n_feature[0]+':'+atac_raw_rt00374n_feature[1].astype('str')+'-'+atac_raw_rt00374n_feature[2].astype('str')
atac_raw_rt00374n.var.index = atac_raw_rt00374n_feature['idx'].values

atac_raw = ad.concat([atac_raw_rt00372n, atac_raw_rt00374n], axis=0)  # concat atac  16480 × 423443

idx = np.intersect1d(rna_raw.obs.index, atac_raw.obs.index)  # 15401
rna_raw[idx, :].write('rna_raw.h5ad')
atac_raw[idx, :].write('atac_raw.h5ad')

## rna
rna = sc.read_h5ad('rna_raw.h5ad')  # 15401 × 35649
rna.X = rna.X.toarray()
genes = pd.read_table('human_genes.txt', names=['gene_ids', 'gene_name'])
rna_exp = pd.concat([rna.var, pd.DataFrame(rna.X.T, index=rna.var.index)], axis=1)
rna_exp['gene_name'] = rna_exp.index.values
X_new = pd.merge(genes, rna_exp, how='left', on='gene_name').iloc[:, 2:].T
X_new.fillna(value=0, inplace=True)
rna_new = sc.AnnData(X_new.values, obs=rna.obs, var=pd.DataFrame({'gene_ids': genes['gene_ids'], 'feature_types': 'Gene Expression'}))
rna_new.var.index = genes['gene_name'].values
rna_new.X = csr_matrix(rna_new.X)   # 15401 × 38244

## atac
atac = sc.read_h5ad('atac_raw.h5ad')  # 15401 × 423443
atac.X[atac.X>0] = 1  # binarization
atac.X = atac.X.toarray()
  
peaks = pd.DataFrame({'id': atac.var_names})
peaks['chr'] = peaks['id'].map(lambda x: x.split(':')[0])
peaks['start'] = peaks['id'].map(lambda x: x.split(':')[1].split('-')[0])
peaks['end'] = peaks['id'].map(lambda x: x.split(':')[1].split('-')[1])
peaks.drop(columns='id', inplace=True)
peaks['idx'] = range(peaks.shape[0])

cCREs = pd.read_table('human_cCREs.bed', names=['chr', 'start', 'end'])
cCREs['idx'] = range(cCREs.shape[0])
cCREs_bed = pybedtools.BedTool.from_dataframe(cCREs)
peaks_bed = pybedtools.BedTool.from_dataframe(peaks)
idx_map = peaks_bed.intersect(cCREs_bed, wa=True, wb=True).to_dataframe().iloc[:, [3, 7]]
idx_map.columns = ['peaks_idx', 'cCREs_idx']

m = np.zeros([atac.n_obs, cCREs_bed.to_dataframe().shape[0]], dtype='float32')
for i in tqdm(range(atac.X.shape[0]), ncols=80, desc='Aligning ATAC peaks'):
    m[i][idx_map[idx_map['peaks_idx'].isin(peaks.iloc[np.nonzero(atac.X[i])]['idx'])]['cCREs_idx']] = 1

atac_new = sc.AnnData(m, obs=atac.obs, var=pd.DataFrame({'gene_ids': cCREs['chr']+':'+cCREs['start'].map(str)+'-'+cCREs['end'].map(str), 'feature_types': 'Peaks'}))
atac_new.var.index = atac_new.var['gene_ids'].values
atac_new.X = csr_matrix(atac_new.X)  # 15401 × 1033239

## filtering
sc.pp.filter_genes(rna_new, min_cells=100)      # 15401 × 15559
sc.pp.filter_cells(rna_new, min_genes=500)      # 15329 × 15559
sc.pp.filter_cells(rna_new, max_genes=10000)    # 15329 × 15559
rna_new.obs['n_genes'].max()                    # 8041

sc.pp.filter_genes(atac_new, min_cells=100)     # 15401 × 221055
sc.pp.filter_cells(atac_new, min_genes=1000)    # 15237 × 221055
sc.pp.filter_cells(atac_new, max_genes=50000)   # 15237 × 221055
atac_new.obs['n_genes'].max()                   # 41065

idx = np.intersect1d(rna_new.obs.index, atac_new.obs.index)  # 15166
rna_new[idx, :].write('rna.h5ad')     # 15166 × 15559
atac_new[idx, :].write('atac.h5ad')   # 15166 × 221055


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

rna_ref = sc.read_h5ad('./rna.h5ad')   # 15166 × 15559
genes = rna_ref.var[['gene_ids']].copy()
genes['gene_name'] = genes.index.values
genes.reset_index(drop=True, inplace=True)
rna_exp = pd.concat([rna.var, pd.DataFrame(rna.X.toarray().T, index=rna.var.index)], axis=1)
X_new = pd.merge(genes, rna_exp, how='left', on='gene_ids').iloc[:, 4:].T
X_new.fillna(value=0, inplace=True)
rna_new = sc.AnnData(X_new.values, obs=rna.obs, var=pd.DataFrame({'gene_ids': genes['gene_ids'], 'feature_types': 'Gene Expression'})) 
# rna_new = sc.AnnData(X_new.values, obs=rna.obs[['cell_anno']], var=pd.DataFrame({'gene_ids': genes['gene_ids'], 'feature_types': 'Gene Expression'}))
rna_new.var.index = genes['gene_name'].values
rna_new.X = csr_matrix(rna_new.X)   # 2500 × 15559
rna_new.write('spatial_rna_test.h5ad')
np.count_nonzero(rna_new.X.toarray(), axis=1).max()  # 6347

## atac
atac_ref = sc.read_h5ad('./atac.h5ad')   # 15166 × 221055

peaks = atac_ref.var[['gene_ids']].copy()
peaks['gene_name'] = peaks.index.values
peaks.reset_index(drop=True, inplace=True)
atac_exp = pd.concat([atac.var, pd.DataFrame(atac.X.toarray().T, index=atac.var.index)], axis=1)
X_new = pd.merge(peaks, atac_exp, how='left', on='gene_ids').iloc[:, 4:].T
X_new.fillna(value=0, inplace=True)
atac_new = sc.AnnData(X_new.values, obs=atac.obs, var=pd.DataFrame({'gene_ids': peaks['gene_ids'], 'feature_types': 'Peaks'}))
atac_new.var.index = peaks['gene_name'].values
atac_new.X = csr_matrix(atac_new.X)   # 2500 × 221055
atac_new.write('spatial_atac_test.h5ad')
########################################################################################################################

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
# 0    516
# 1    486
# 2    401
# 3    319
# 4    313
# 5    256
# 6    148
# 7     61

rna.obs['cell_anno'] = rna.obs['leiden']
rna.write('rna_test.h5ad')

atac = sc.read_h5ad('spatial_atac_test.h5ad')
atac.obs['cell_anno'] = rna.obs['leiden']
atac.write('atac_test.h5ad')

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
sns.heatmap(mat_true, cmap=sns.color_palette(['grey', 'cornflowerblue', 'green', 'purple', 'orange', 'yellow', 'red', 'blue']), xticklabels=False, yticklabels=False)
plt.savefig('ATAC_spatial_true.pdf')
plt.close()


############################################################################ Cisformer ############################################################################
python split_train_val.py --RNA rna.h5ad --ATAC atac.h5ad --train_pct 0.9
nohup python data_preprocess.py -r rna_train.h5ad -a atac_train.h5ad -s train_pt --dt train -n train --config rna2atac_config_train.yaml > data_preprocess_train_20250122.log & 
# 2604477
python data_preprocess.py -r rna_val.h5ad -a atac_val.h5ad -s val_pt --dt val -n val --config rna2atac_config_val.yaml
nohup accelerate launch --config_file accelerator_config_train.yaml --main_process_port 29824 rna2atac_train.py --config_file rna2atac_config_train.yaml \
                        --train_data_dir train_pt --val_data_dir val_pt -s save -n rna2atac_brain > 20250122.log &   # 2649393

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


############################################################################ scButterfly ############################################################################
# /data/home/jiluzhang/miniconda3/envs/scButterfly/lib/python3.9/site-packages/scButterfly/train_model.py
# line 329: "reconstruct_loss = r_loss_w * (lossR2R + lossA2R) + a_loss_w * (lossR2A + lossA2A)" -> "reconstruct_loss = lossR2A"
# line 331: "kl_div = kl_div_r + kl_div_a" -> "kl_div = np.array(0)"
# line 751: "loss_d.backward()" (add # Luz)  not train for discriminator
# butterfly.train_model() parameters: R2R_pretrain_epoch=0, A2A_pretrain_epoch=0, translation_kl_warmup=0, patience=10
# line 974-977 (add # Luz) not pca

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

nohup python scbt_b.py > scbt_b_20250122.log &   # 2696403
mv predict predict_val  # avoid no output
python scbt_b_predict.py
cp predict/R2A.h5ad atac_scbt.h5ad
python cal_auroc_auprc.py --pred atac_scbt.h5ad --true atac_test.h5ad
# Cell-wise AUROC: 0.6667
# Cell-wise AUPRC: 0.0305
# Peak-wise AUROC: 0.6474
# Peak-wise AUPRC: 0.0471
python plot_save_umap.py --pred atac_scbt.h5ad --true atac_test.h5ad
python cal_cluster.py --file atac_scbt_umap.h5ad
# AMI: 0.2920
# ARI: 0.1439
# HOM: 0.3373
# NMI: 0.2972

## scbt_new
nohup python scbt_b.py > scbt_b_20250122.log &   # 2689377
mv predict predict_val  # avoid no output
python scbt_b_predict.py
cp predict/R2A.h5ad atac_scbt.h5ad
python cal_auroc_auprc.py --pred atac_scbt.h5ad --true atac_test.h5ad
# Cell-wise AUROC: 
# Cell-wise AUPRC: 
# Peak-wise AUROC: 
# Peak-wise AUPRC: 
python plot_save_umap.py --pred atac_scbt.h5ad --true atac_test.h5ad
python cal_cluster.py --file atac_scbt_umap.h5ad
# AMI: 
# ARI: 
# HOM: 
# NMI: 

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
# 0    534
# 1    456
# 2    381
# 3    354
# 4    274
# 5    212
# 6    158
# 7    131

df_scbt = pd.DataFrame({'cell_idx':atac.obs.index, 'cell_anno':atac.obs.leiden.astype('int')})
df_pos_scbt = pd.merge(df_scbt, pos)

mat_scbt = np.zeros([50, 50])
for i in range(df_pos_scbt.shape[0]):
    if df_pos_scbt['cell_anno'][i]!=0:
        mat_scbt[df_pos_scbt['X'][i], (49-df_pos_scbt['Y'][i])] = df_pos_scbt['cell_anno'][i]  # start from upper right

plt.figure(figsize=(6, 5))
sns.heatmap(mat_scbt, cmap=sns.color_palette(['blue', 'purple', 'green', 'orange', 'grey', 'red', 'yellow', 'cornflowerblue']), xticklabels=False, yticklabels=False)
plt.savefig('ATAC_spatial_scbt.pdf')
plt.close()

normalized_mutual_info_score(atac.obs['cell_anno'], atac.obs['leiden'])  # 0.3000233047724948

## correlation
from scipy import stats
import numpy as np

stats.pearsonr(np.load('mat_thy1_scbt.npy').flatten(), np.load('mat_thy1_true.npy').flatten())[0]       # 0.3748678172271672
stats.pearsonr(np.load('mat_bcl11b_scbt.npy').flatten(), np.load('mat_bcl11b_true.npy').flatten())[0]   # 0.3101401754841971


############################################################################ BABEL ############################################################################
## /fs/home/jiluzhang/BABEL/babel/loss_functions.py
## line 430: "loss = loss11 + self.loss2_weight * loss22" -> "loss = 0*loss11 + 0*self.loss2_weight * loss22"
## line 431: "loss += next(self.cross_warmup) * (loss21 + self.loss2_weight * loss12)" -> "loss += next(self.cross_warmup) * (0*loss21 + 0*self.loss2_weight * loss12 + loss12)"

cp ../rna.h5ad rna_train_val.h5ad
cp ../atac.h5ad atac_train_val.h5ad
python h5ad2h5.py -n train_val   # train + val -> train_val
nohup python /fs/home/jiluzhang/BABEL/bin/train_model.py --data train_val.h5 --outdir train_out --batchsize 512 --earlystop 10 --device 0 --nofilter > train_20250122.log &  # 2511418

python h5ad2h5.py -n test
python /fs/home/jiluzhang/BABEL/bin/predict_model.py --checkpoint ./train_out --data test.h5 --outdir test_out --device 0 --nofilter --noplot --transonly
python match_cell.py
python cal_auroc_auprc.py --pred atac_babel.h5ad --true atac_test.h5ad
# Cell-wise AUROC: 0.6384
# Cell-wise AUPRC: 0.0272
# Peak-wise AUROC: 0.4341
# Peak-wise AUPRC: 0.0169
python plot_save_umap.py --pred atac_babel.h5ad --true atac_test.h5ad
python cal_cluster.py --file atac_babel_umap.h5ad
# AMI: 0.2087
# ARI: 0.1082
# HOM: 0.2362
# NMI: 0.2138


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
# 0    426
# 1    403
# 2    396
# 3    334
# 4    320
# 5    222
# 6    209
# 7    190

df_babel = pd.DataFrame({'cell_idx':atac.obs.index, 'cell_anno':atac.obs.leiden.astype('int')})
df_pos_babel = pd.merge(df_babel, pos)

mat_babel = np.zeros([50, 50])
for i in range(df_pos_babel.shape[0]):
    if df_pos_babel['cell_anno'][i]!=0:
        mat_babel[df_pos_babel['X'][i], (49-df_pos_babel['Y'][i])] = df_pos_babel['cell_anno'][i]  # start from upper right

plt.figure(figsize=(6, 5))
sns.heatmap(mat_babel, cmap=sns.color_palette(['red', 'yellow', 'green', 'orange', 'blue', 'cornflowerblue', 'grey', 'purple']), xticklabels=False, yticklabels=False)
plt.savefig('ATAC_spatial_babel.pdf')
plt.close()

normalized_mutual_info_score(atac.obs['cell_anno'], atac.obs['leiden'])  # 0.21223661767544588

## correlation
from scipy import stats
import numpy as np

stats.pearsonr(np.load('mat_thy1_babel.npy').flatten(), np.load('mat_thy1_true.npy').flatten())[0]       # 0.09844612591162669
stats.pearsonr(np.load('mat_bcl11b_babel.npy').flatten(), np.load('mat_bcl11b_true.npy').flatten())[0]   # -0.29728841021479196
