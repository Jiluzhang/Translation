#### https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi
#### Single-cell genomics and regulatory networks for 388 human brains

## RT00372N
## wget -c https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM8155nnn/GSM8155483/suppl/GSM8155483_RT00372N_RNA_barcodes.txt.gz
## wget -c https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM8155nnn/GSM8155483/suppl/GSM8155483_RT00372N_RNA_matrix.mtx.gz
## wget -c https://ftp.ncbi.nlm.nih.gov/geo/series/GSE261nnn/GSE261983/suppl/GSE261983_RNA_gene_list.txt.gz
## wget -c https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM8155nnn/GSM8155503/suppl/GSM8155503_RT00372N_ATAC_barcodes.txt.gz
## wget -c https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM8155nnn/GSM8155503/suppl/GSM8155503_RT00372N_ATAC_matrix.mtx.gz
## wget -c https://ftp.ncbi.nlm.nih.gov/geo/series/GSE261nnn/GSE261983/suppl/GSE261983_ATAC_peakset.bed.gz

#zcat GSM8155503_RT00372N_ATAC_matrix.mtx.gz | head -n 2 >> ATAC_matrix.mtx
#zcat GSM8155503_RT00372N_ATAC_matrix.mtx.gz | tail -n +3 | awk '{print $2 "\t" $1 "\t" $3}' >> ATAC_matrix.mtx

import scanpy as sc
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
import pybedtools
from tqdm import tqdm

rna_raw = sc.read_mtx('GSM8155483_RT00372N_RNA_matrix.mtx.gz')  # 8369 × 35649
rna_raw_barcode = pd.read_table('GSM8155483_RT00372N_RNA_barcodes.txt.gz', header=None)
rna_raw.obs.index = rna_raw_barcode[0].values
rna_raw_feature = pd.read_table('GSE261983_RNA_gene_list.txt.gz', header=None)
rna_raw.var.index = rna_raw_feature[0].values
rna_raw.X = rna_raw.X.T

atac_raw = sc.read_mtx('ATAC_matrix.mtx')  # 8984 × 423443
atac_raw_barcode = pd.read_table('GSM8155503_RT00372N_ATAC_barcodes.txt.gz', header=None)
atac_raw.obs.index = atac_raw_barcode[0].values
atac_raw_feature = pd.read_table('GSE261983_ATAC_peakset.bed.gz', header=None)
atac_raw_feature['idx'] = atac_raw_feature[0]+':'+atac_raw_feature[1].astype('str')+'-'+atac_raw_feature[2].astype('str')
atac_raw.var.index = atac_raw_feature['idx'].values

idx = np.intersect1d(rna_raw.obs.index, atac_raw.obs.index)
rna_raw[idx, :].write('rna_raw.h5ad')
atac_raw[idx, :].write('atac_raw.h5ad')

## rna
rna = sc.read_h5ad('rna_raw.h5ad')  # 8369 × 35649
rna.X = rna.X.toarray()
genes = pd.read_table('human_genes.txt', names=['gene_ids', 'gene_name'])
rna_exp = pd.concat([rna.var, pd.DataFrame(rna.X.T, index=rna.var.index)], axis=1)
rna_exp['gene_name'] = rna_exp.index.values
X_new = pd.merge(genes, rna_exp, how='left', on='gene_name').iloc[:, 2:].T
X_new.fillna(value=0, inplace=True)
rna_new = sc.AnnData(X_new.values, obs=rna.obs, var=pd.DataFrame({'gene_ids': genes['gene_ids'], 'feature_types': 'Gene Expression'}))
rna_new.var.index = genes['gene_name'].values
rna_new.X = csr_matrix(rna_new.X)   # 8369 × 38244

## atac
atac = sc.read_h5ad('atac_raw.h5ad')  # 8369 × 423443
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
atac_new.X = csr_matrix(atac_new.X)  # 8369 × 1033239

## filtering
sc.pp.filter_genes(rna_new, min_cells=100)      # 8369 × 13390
sc.pp.filter_cells(rna_new, min_genes=200)      # 8369 × 13390
sc.pp.filter_cells(rna_new, max_genes=20000)    # 8369 × 13390
rna_new.obs['n_genes'].max()                    # 7851

sc.pp.filter_genes(atac_new, min_cells=100)     # 8369 × 125853
sc.pp.filter_cells(atac_new, min_genes=500)     # 8369 × 125853
sc.pp.filter_cells(atac_new, max_genes=50000)   # 8369 × 125853
atac_new.obs['n_genes'].max()                   # 33969

rna_new.write('rna.h5ad')
atac_new.write('atac.h5ad')

## train model
python split_train_val.py --RNA rna.h5ad --ATAC atac.h5ad --train_pct 0.9
python data_preprocess.py -r rna_train.h5ad -a atac_train.h5ad -s train_pt --dt train -n train --config rna2atac_config_train.yaml   # 20 min
python data_preprocess.py -r rna_val.h5ad -a atac_val.h5ad -s val_pt --dt val -n val --config rna2atac_config_val.yaml
nohup accelerate launch --config_file accelerator_config_train.yaml --main_process_port 29824 rna2atac_train.py --config_file rna2atac_config_train.yaml \
                        --train_data_dir train_pt --val_data_dir val_pt -s save -n rna2atac_brain > 20250120.log &   # 2789948

## train model (small)  depth=3 & heads=3
nohup accelerate launch --config_file accelerator_config_train.yaml --main_process_port 29824 rna2atac_train.py --config_file rna2atac_config_train_small.yaml \
                        --train_data_dir train_pt --val_data_dir val_pt -s save_small -n rna2atac_brain > 20250120_small.log &   # 3517099

## train model (little)  depth=1 & heads=1
nohup accelerate launch --config_file accelerator_config_train.yaml --main_process_port 29824 rna2atac_train.py --config_file rna2atac_config_train_little.yaml \
                        --train_data_dir train_pt --val_data_dir val_pt -s save_little -n rna2atac_brain > 20250121_little.log &   # 609309

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

rna_ref = sc.read_h5ad('./rna.h5ad')   # 8369 × 13390
genes = rna_ref.var[['gene_ids']].copy()
genes['gene_name'] = genes.index.values
genes.reset_index(drop=True, inplace=True)
rna_exp = pd.concat([rna.var, pd.DataFrame(rna.X.toarray().T, index=rna.var.index)], axis=1)
X_new = pd.merge(genes, rna_exp, how='left', on='gene_ids').iloc[:, 4:].T
X_new.fillna(value=0, inplace=True)
rna_new = sc.AnnData(X_new.values, obs=rna.obs, var=pd.DataFrame({'gene_ids': genes['gene_ids'], 'feature_types': 'Gene Expression'})) 
# rna_new = sc.AnnData(X_new.values, obs=rna.obs[['cell_anno']], var=pd.DataFrame({'gene_ids': genes['gene_ids'], 'feature_types': 'Gene Expression'}))
rna_new.var.index = genes['gene_name'].values
rna_new.X = csr_matrix(rna_new.X)   # 2500 × 17532
rna_new.write('spatial_rna_test.h5ad')
np.count_nonzero(rna_new.X.toarray(), axis=1).max()  # 6228

## atac
atac_ref = sc.read_h5ad('./atac.h5ad')   # 8369 × 125853

peaks = atac_ref.var[['gene_ids']].copy()
peaks['gene_name'] = peaks.index.values
peaks.reset_index(drop=True, inplace=True)
atac_exp = pd.concat([atac.var, pd.DataFrame(atac.X.toarray().T, index=atac.var.index)], axis=1)
X_new = pd.merge(peaks, atac_exp, how='left', on='gene_ids').iloc[:, 4:].T
X_new.fillna(value=0, inplace=True)
atac_new = sc.AnnData(X_new.values, obs=atac.obs, var=pd.DataFrame({'gene_ids': peaks['gene_ids'], 'feature_types': 'Peaks'}))
# atac_new = sc.AnnData(X_new.values, obs=atac.obs[['cell_anno']], var=pd.DataFrame({'gene_ids': peaks['gene_ids'], 'feature_types': 'Peaks'}))
atac_new.var.index = peaks['gene_name'].values
atac_new.X = csr_matrix(atac_new.X)   # 2500 × 125853
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
sc.tl.leiden(rna, resolution=0.9)
rna.obs['leiden'].value_counts()
# 0    712
# 1    429
# 2    407
# 3    310
# 4    250
# 5    172
# 6    152
# 7     68

rna.obs['cell_anno'] = rna.obs['leiden']
rna.write('rna_test.h5ad')

atac = sc.read_h5ad('spatial_atac_test.h5ad')
atac.obs['cell_anno'] = rna.obs['leiden']
atac.write('atac_test.h5ad')


## epoch=3
python data_preprocess.py -r spatial_rna_test.h5ad -a spatial_atac_test.h5ad -s spatial_test_pt --dt test --config rna2atac_config_test.yaml
accelerate launch --config_file accelerator_config_test.yaml --main_process_port 29822 rna2atac_predict.py \
                  -d ./spatial_test_pt -l ./save/2025-01-20_rna2atac_brain_3/pytorch_model.bin --config_file rna2atac_config_test.yaml
python npy2h5ad.py
python cal_auroc_auprc.py --pred atac_cisformer.h5ad --true atac_test.h5ad
# Cell-wise AUROC: 0.6424
# Cell-wise AUPRC: 0.0339
# Peak-wise AUROC: 0.6712
# Peak-wise AUPRC: 0.0352
python plot_save_umap.py --pred atac_cisformer.h5ad --true atac_test.h5ad
python cal_cluster.py --file atac_cisformer_umap.h5ad
# AMI: 0.3581
# ARI: 0.2608
# HOM: 0.3848
# NMI: 0.3620

## epoch=2
accelerate launch --config_file accelerator_config_test.yaml --main_process_port 29822 rna2atac_predict.py \
                  -d ./spatial_test_pt -l ./save/2025-01-20_rna2atac_brain_2/pytorch_model.bin --config_file rna2atac_config_test.yaml
python npy2h5ad.py
python cal_auroc_auprc.py --pred atac_cisformer.h5ad --true atac_test.h5ad
# Cell-wise AUROC: 0.6395
# Cell-wise AUPRC: 0.0333
# Peak-wise AUROC: 0.6718
# Peak-wise AUPRC: 0.0362
python plot_save_umap.py --pred atac_cisformer.h5ad --true atac_test.h5ad
python cal_cluster.py --file atac_cisformer_umap.h5ad
# AMI: 0.3564
# ARI: 0.2489
# HOM: 0.3714
# NMI: 0.3597

## epoch=5
accelerate launch --config_file accelerator_config_test.yaml --main_process_port 29822 rna2atac_predict.py \
                  -d ./spatial_test_pt -l ./save/2025-01-20_rna2atac_brain_5/pytorch_model.bin --config_file rna2atac_config_test.yaml
python npy2h5ad.py
python cal_auroc_auprc.py --pred atac_cisformer.h5ad --true atac_test.h5ad
# Cell-wise AUROC: 0.6399
# Cell-wise AUPRC: 0.0340
# Peak-wise AUROC: 0.6504
# Peak-wise AUPRC: 0.0306
python plot_save_umap.py --pred atac_cisformer.h5ad --true atac_test.h5ad
python cal_cluster.py --file atac_cisformer_umap.h5ad
# AMI: 0.3262
# ARI: 0.2223
# HOM: 0.3365
# NMI: 0.3294

## epoch=8
accelerate launch --config_file accelerator_config_test.yaml --main_process_port 29822 rna2atac_predict.py \
                  -d ./spatial_test_pt -l ./save/2025-01-20_rna2atac_brain_8/pytorch_model.bin --config_file rna2atac_config_test.yaml
python npy2h5ad.py
python cal_auroc_auprc.py --pred atac_cisformer.h5ad --true atac_test.h5ad
# Cell-wise AUROC: 0.6387
# Cell-wise AUPRC: 0.0342
# Peak-wise AUROC: 0.6298
# Peak-wise AUPRC: 0.0275
python plot_save_umap.py --pred atac_cisformer.h5ad --true atac_test.h5ad
python cal_cluster.py --file atac_cisformer_umap.h5ad
# AMI: 0.3294
# ARI: 0.2461
# HOM: 0.3326
# NMI: 0.3324

## evaluation for validataion dataset
python data_preprocess.py -r rna_val.h5ad -a atac_val.h5ad -s test_val_pt --dt test --config rna2atac_config_test_val.yaml
accelerate launch --config_file accelerator_config_test_val.yaml --main_process_port 29822 rna2atac_predict.py \
                  -d ./test_val_pt -l ./save/2025-01-20_rna2atac_brain_3/pytorch_model.bin --config_file rna2atac_config_test_val.yaml
python npy2h5ad.py
python cal_auroc_auprc.py --pred atac_cisformer.h5ad --true atac_test.h5ad
# Cell-wise AUROC: 0.7318
# Cell-wise AUPRC: 0.1007
# Peak-wise AUROC: 0.6206
# Peak-wise AUPRC: 0.0539
python plot_save_umap.py --pred atac_cisformer.h5ad --true atac_test.h5ad
python cal_cluster.py --file atac_cisformer_umap.h5ad
# AMI: 0.7358
# ARI: 0.5605
# HOM: 0.7411
# NMI: 0.7393

#### evaluation for small model
## epoch=20
accelerate launch --config_file accelerator_config_test.yaml --main_process_port 29822 rna2atac_predict.py \
                  -d ./spatial_test_pt -l ./save_small/2025-01-20_rna2atac_brain_20/pytorch_model.bin --config_file rna2atac_config_test_small.yaml
python npy2h5ad.py
python cal_auroc_auprc.py --pred atac_cisformer.h5ad --true atac_test.h5ad
# Cell-wise AUROC: 0.6399
# Cell-wise AUPRC: 0.0342
# Peak-wise AUROC: 0.6527
# Peak-wise AUPRC: 0.0304
python plot_save_umap.py --pred atac_cisformer.h5ad --true atac_test.h5ad
python cal_cluster.py --file atac_cisformer_umap.h5ad
# AMI: 0.3369
# ARI: 0.2351
# HOM: 0.3389
# NMI: 0.3400

## epoch=30
accelerate launch --config_file accelerator_config_test.yaml --main_process_port 29822 rna2atac_predict.py \
                  -d ./spatial_test_pt -l ./save_small/2025-01-20_rna2atac_brain_30/pytorch_model.bin --config_file rna2atac_config_test_small.yaml
python npy2h5ad.py
python cal_auroc_auprc.py --pred atac_cisformer.h5ad --true atac_test.h5ad
# Cell-wise AUROC: 0.6396
# Cell-wise AUPRC: 0.0342
# Peak-wise AUROC: 0.6445
# Peak-wise AUPRC: 0.0290
python plot_save_umap.py --pred atac_cisformer.h5ad --true atac_test.h5ad
python cal_cluster.py --file atac_cisformer_umap.h5ad
# AMI: 0.3347
# ARI: 0.2282
# HOM: 0.3409
# NMI: 0.3380

## epoch=10
accelerate launch --config_file accelerator_config_test.yaml --main_process_port 29822 rna2atac_predict.py \
                  -d ./spatial_test_pt -l ./save_small/2025-01-20_rna2atac_brain_10/pytorch_model.bin --config_file rna2atac_config_test_small.yaml
python npy2h5ad.py
python cal_auroc_auprc.py --pred atac_cisformer.h5ad --true atac_test.h5ad
# Cell-wise AUROC: 0.6419
# Cell-wise AUPRC: 0.0344
# Peak-wise AUROC: 0.6762
# Peak-wise AUPRC: 0.0358
python plot_save_umap.py --pred atac_cisformer.h5ad --true atac_test.h5ad
python cal_cluster.py --file atac_cisformer_umap.h5ad
# AMI: 0.3320
# ARI: 0.2326
# HOM: 0.3435
# NMI: 0.3354

## epoch=5
accelerate launch --config_file accelerator_config_test.yaml --main_process_port 29822 rna2atac_predict.py \
                  -d ./spatial_test_pt -l ./save_small/2025-01-20_rna2atac_brain_5/pytorch_model.bin --config_file rna2atac_config_test_small.yaml
python npy2h5ad.py
python cal_auroc_auprc.py --pred atac_cisformer.h5ad --true atac_test.h5ad
# Cell-wise AUROC: 0.6455
# Cell-wise AUPRC: 0.0344
# Peak-wise AUROC: 0.6846
# Peak-wise AUPRC: 0.0394
python plot_save_umap.py --pred atac_cisformer.h5ad --true atac_test.h5ad
python cal_cluster.py --file atac_cisformer_umap.h5ad
# AMI: 0.3400
# ARI: 0.2109
# HOM: 0.3584
# NMI: 0.3437

## epoch=2
accelerate launch --config_file accelerator_config_test.yaml --main_process_port 29822 rna2atac_predict.py \
                  -d ./spatial_test_pt -l ./save_small/2025-01-20_rna2atac_brain_2/pytorch_model.bin --config_file rna2atac_config_test_small.yaml
python npy2h5ad.py
python cal_auroc_auprc.py --pred atac_cisformer.h5ad --true atac_test.h5ad
# Cell-wise AUROC: 0.6442
# Cell-wise AUPRC: 0.0338
# Peak-wise AUROC: 0.6870
# Peak-wise AUPRC: 0.0407
python plot_save_umap.py --pred atac_cisformer.h5ad --true atac_test.h5ad
python cal_cluster.py --file atac_cisformer_umap.h5ad
# AMI: 0.3507
# ARI: 0.2007
# HOM: 0.3928
# NMI: 0.3549

#### evaluation for little model
## epoch=1
accelerate launch --config_file accelerator_config_test.yaml --main_process_port 29822 rna2atac_predict.py \
                  -d ./spatial_test_pt -l ./save_little/2025-01-21_rna2atac_brain_1/pytorch_model.bin --config_file rna2atac_config_test_little.yaml
python npy2h5ad.py
python cal_auroc_auprc.py --pred atac_cisformer.h5ad --true atac_test.h5ad
# Cell-wise AUROC: 0.6169
# Cell-wise AUPRC: 0.0301
# Peak-wise AUROC: 0.5220
# Peak-wise AUPRC: 0.0210
python plot_save_umap.py --pred atac_cisformer.h5ad --true atac_test.h5ad
python cal_cluster.py --file atac_cisformer_umap.h5ad
# AMI: 0.3513
# ARI: 0.225
# HOM: 0.3562
# NMI: 0.3545

## epoch=5
accelerate launch --config_file accelerator_config_test.yaml --main_process_port 29822 rna2atac_predict.py \
                  -d ./spatial_test_pt -l ./save_little/2025-01-21_rna2atac_brain_5/pytorch_model.bin --config_file rna2atac_config_test_little.yaml
python npy2h5ad.py
python cal_auroc_auprc.py --pred atac_cisformer.h5ad --true atac_test.h5ad
# Cell-wise AUROC: 0.6409
# Cell-wise AUPRC: 0.0331
# Peak-wise AUROC: 0.5338
# Peak-wise AUPRC: 0.0209
python plot_save_umap.py --pred atac_cisformer.h5ad --true atac_test.h5ad
python cal_cluster.py --file atac_cisformer_umap.h5ad
# AMI: 0.3980
# ARI: 0.3077
# HOM: 0.3976
# NMI: 0.4010

## epoch=7
accelerate launch --config_file accelerator_config_test.yaml --main_process_port 29822 rna2atac_predict.py \
                  -d ./spatial_test_pt -l ./save_little/2025-01-21_rna2atac_brain_7/pytorch_model.bin --config_file rna2atac_config_test_little.yaml
python npy2h5ad.py
python cal_auroc_auprc.py --pred atac_cisformer.h5ad --true atac_test.h5ad
# Cell-wise AUROC: 0.6426
# Cell-wise AUPRC: 0.0335
# Peak-wise AUROC: 0.5325
# Peak-wise AUPRC: 0.0208
python plot_save_umap.py --pred atac_cisformer.h5ad --true atac_test.h5ad
python cal_cluster.py --file atac_cisformer_umap.h5ad
# AMI: 0.3904
# ARI: 0.2734
# HOM: 0.3992
# NMI: 0.3935

## epoch=10
accelerate launch --config_file accelerator_config_test.yaml --main_process_port 29822 rna2atac_predict.py \
                  -d ./spatial_test_pt -l ./save_little/2025-01-21_rna2atac_brain_10/pytorch_model.bin --config_file rna2atac_config_test_little.yaml
python npy2h5ad.py
python cal_auroc_auprc.py --pred atac_cisformer.h5ad --true atac_test.h5ad
# Cell-wise AUROC: 0.6414
# Cell-wise AUPRC: 0.0334
# Peak-wise AUROC: 0.5310
# Peak-wise AUPRC: 0.0205
python plot_save_umap.py --pred atac_cisformer.h5ad --true atac_test.h5ad
python cal_cluster.py --file atac_cisformer_umap.h5ad
# AMI: 0.3941
# ARI: 0.2784
# HOM: 0.3900
# NMI: 0.3969


#### clustering for true with rna
import scanpy as sc

rna = sc.read_h5ad('../rna_val.h5ad')
sc.pp.normalize_total(rna, target_sum=1e4)
sc.pp.log1p(rna)
sc.pp.highly_variable_genes(rna)
rna = rna[:, rna.var.highly_variable]
sc.tl.pca(rna)
sc.pp.neighbors(rna)
sc.tl.umap(rna)
sc.tl.leiden(rna, resolution=0.5)
rna.obs['leiden'].value_counts()
# 0    223
# 1    180
# 2    124
# 3     88
# 4     86
# 5     48
# 6     48
# 7     40

rna.obs['cell_anno'] = rna.obs['leiden']
rna.write('rna_test.h5ad')

atac = sc.read_h5ad('../atac_val.h5ad')
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
snap.tl.leiden(atac, resolution=0.8)
atac.obs['leiden'].value_counts()
# 0    481
# 1    450
# 2    378
# 3    332
# 4    275
# 5    272
# 6    180
# 7    132

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

normalized_mutual_info_score(atac.obs['cell_anno'], atac.obs['leiden'])  # 0.3599611011166151

## correlation
from scipy import stats
import numpy as np

stats.pearsonr(np.load('mat_thy1_cisformer.npy').flatten(), np.load('mat_thy1_true.npy').flatten())[0]       # 0.2260101020231856
stats.pearsonr(np.load('mat_bcl11b_cisformer.npy').flatten(), np.load('mat_bcl11b_true.npy').flatten())[0]   # 0.18265064492508273


#### scbutterfly
# /data/home/jiluzhang/miniconda3/envs/scButterfly/lib/python3.9/site-packages/scButterfly/train_model.py
# line 329: "reconstruct_loss = r_loss_w * (lossR2R + lossA2R) + a_loss_w * (lossR2A + lossA2A)" -> "reconstruct_loss = lossR2A"
# line 374: "loss2 = d_loss(predict_rna.reshape(batch_size), torch.tensor(temp).cuda().float())" -> "loss2 = 0"
# butterfly.train_model() parameters: R2R_pretrain_epoch=0, A2A_pretrain_epoch=0, translation_kl_warmup=0, patience=10

nohup python scbt_b.py > scbt_b_20250121.log &   # 1527522
mv predict predict_val  # avoid no output
python scbt_b_predict.py
cp predict/R2A.h5ad atac_scbt.h5ad
python cal_auroc_auprc.py --pred atac_scbt.h5ad --true atac_test.h5ad
# Cell-wise AUROC: 0.6596
# Cell-wise AUPRC: 0.0356
# Peak-wise AUROC: 0.6979
# Peak-wise AUPRC: 0.0607
python plot_save_umap.py --pred atac_scbt.h5ad --true atac_test.h5ad
python cal_cluster.py --file atac_scbt_umap.h5ad
# AMI: 0.3091
# ARI: 0.1725
# HOM: 0.3444
# NMI: 0.3135



nohup python scbt_b.py > scbt_b_20250120.log &   # 2667825

# cp predict/R2A.h5ad atac_scbt.h5ad
# python cal_auroc_auprc.py --pred atac_scbt.h5ad --true atac_test.h5ad
# python plot_save_umap.py --pred atac_scbt.h5ad --true atac_test.h5ad
# python cal_cluster.py --file atac_scbt_umap.h5ad

mv predict predict_val  # avoid no output
python scbt_b_predict.py
cp predict/R2A.h5ad atac_scbt.h5ad
python cal_auroc_auprc.py --pred atac_scbt.h5ad --true atac_test.h5ad
# Cell-wise AUROC: 0.6545
# Cell-wise AUPRC: 0.0349
# Peak-wise AUROC: 0.6881
# Peak-wise AUPRC: 0.0608
python plot_save_umap.py --pred atac_scbt.h5ad --true atac_test.h5ad
python cal_cluster.py --file atac_scbt_umap.h5ad
# AMI: 0.3723
# ARI: 0.2013
# HOM: 0.4161
# NMI: 0.3764

## evaluation for val
cp predict_val/R2A.h5ad atac_scbt.h5ad
python cal_auroc_auprc.py --pred atac_scbt.h5ad --true atac_test.h5ad
# Cell-wise AUROC: 0.7603
# Cell-wise AUPRC: 0.1118
# Peak-wise AUROC: 0.6759
# Peak-wise AUPRC: 0.0741
python plot_save_umap.py --pred atac_scbt.h5ad --true atac_test.h5ad
python cal_cluster.py --file atac_scbt_umap.h5ad
# AMI: 0.7424
# ARI: 0.5610
# HOM: 0.7327
# NMI: 0.7457


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
snap.tl.leiden(atac, resolution=0.7)
atac.obs['leiden'].value_counts()
# 0    451
# 1    399
# 2    353
# 3    351
# 4    290
# 5    243
# 6    240
# 7    173

df_scbt = pd.DataFrame({'cell_idx':atac.obs.index, 'cell_anno':atac.obs.leiden.astype('int')})
df_pos_scbt = pd.merge(df_scbt, pos)

mat_scbt = np.zeros([50, 50])
for i in range(df_pos_scbt.shape[0]):
    if df_pos_scbt['cell_anno'][i]!=0:
        mat_scbt[df_pos_scbt['X'][i], (49-df_pos_scbt['Y'][i])] = df_pos_scbt['cell_anno'][i]  # start from upper right

plt.figure(figsize=(6, 5))
sns.heatmap(mat_scbt, cmap=sns.color_palette(['grey', 'orange', 'green', 'red', 'blue', 'purple', 'yellow', 'cornflowerblue']), xticklabels=False, yticklabels=False)
plt.savefig('ATAC_spatial_scbt.pdf')
plt.close()

normalized_mutual_info_score(atac.obs['cell_anno'], atac.obs['leiden'])  # 0.36542319219235764

## correlation
from scipy import stats
import numpy as np

stats.pearsonr(np.load('mat_thy1_scbt.npy').flatten(), np.load('mat_thy1_true.npy').flatten())[0]       # 0.3049408763956322
stats.pearsonr(np.load('mat_bcl11b_scbt.npy').flatten(), np.load('mat_bcl11b_true.npy').flatten())[0]   # 0.1493882939065321


#### BABEL
## /fs/home/jiluzhang/BABEL/babel/loss_functions.py
## line 430: "loss = loss11 + self.loss2_weight * loss22" -> "loss = 0*loss11 + 0*self.loss2_weight * loss22"
## line 431: "loss += next(self.cross_warmup) * (loss21 + self.loss2_weight * loss12)" -> "loss += next(self.cross_warmup) * (0*loss21 + 0*self.loss2_weight * loss12 + loss12)"

python h5ad2h5.py -n train_val   # train + val -> train_val
nohup python /fs/home/jiluzhang/BABEL/bin/train_model.py --data train_val.h5 --outdir train_out --batchsize 512 --earlystop 25 --device 6 --nofilter > train_20250120.log &  # 2726382


####################################################################################################################################################################################################
## earlystop=25
nohup python /fs/home/jiluzhang/BABEL/bin/train_model.py --data train_val.h5 --outdir train_out --batchsize 512 --earlystop 25 --device 6 --nofilter > train_20250121.log &  # 1321827

# python h5ad2h5.py -n test
python /fs/home/jiluzhang/BABEL/bin/predict_model.py --checkpoint ./train_out --data test.h5 --outdir test_out --device 6 --nofilter --noplot --transonly
python match_cell.py
python cal_auroc_auprc.py --pred atac_babel.h5ad --true atac_test.h5ad
# Cell-wise AUROC: 0.6600
# Cell-wise AUPRC: 0.0348
# Peak-wise AUROC: 0.5563
# Peak-wise AUPRC: 0.0284
python plot_save_umap.py --pred atac_babel.h5ad --true atac_test.h5ad
python cal_cluster.py --file atac_babel_umap.h5ad
# AMI: 0.2473
# ARI: 0.1439
# HOM: 0.256
# NMI: 0.2509

## earlystop=10
nohup python /fs/home/jiluzhang/BABEL/bin/train_model.py --data train_val.h5 --outdir train_out --batchsize 512 --earlystop 10 --device 6 --nofilter > train_20250121.log &  # 1465297

# python h5ad2h5.py -n test
python /fs/home/jiluzhang/BABEL/bin/predict_model.py --checkpoint ./train_out --data test.h5 --outdir test_out --device 6 --nofilter --noplot --transonly
python match_cell.py
python cal_auroc_auprc.py --pred atac_babel.h5ad --true atac_test.h5ad
# Cell-wise AUROC: 0.6650
# Cell-wise AUPRC: 0.0358
# Peak-wise AUROC: 0.5606
# Peak-wise AUPRC: 0.0264
python plot_save_umap.py --pred atac_babel.h5ad --true atac_test.h5ad
python cal_cluster.py --file atac_babel_umap.h5ad
# AMI: 0.2090
# ARI: 0.0886
# HOM: 0.2352
# NMI: 0.2140
####################################################################################################################################################################################################





# python h5ad2h5.py -n test
# nohup python /fs/home/jiluzhang/BABEL/bin/predict_model.py --checkpoint train_out --data test.h5 --outdir test_out --device 1 --nofilter --noplot --transonly > predict_20241226.log &
# # 3582789
# python match_cell.py
# python cal_auroc_auprc.py --pred atac_babel.h5ad --true atac_test.h5ad
# python plot_save_umap.py --pred atac_babel.h5ad --true atac_test.h5ad
# python cal_cluster.py --file atac_babel_umap.h5ad

python h5ad2h5.py -n test
nohup python /fs/home/jiluzhang/BABEL/bin/predict_model.py --checkpoint ./train_out --data test.h5 --outdir test_out --device 6 --nofilter --noplot --transonly > predict_20250120.log &
# 2841761
python match_cell.py
python cal_auroc_auprc.py --pred atac_babel.h5ad --true atac_test.h5ad
# Cell-wise AUROC: 0.6608
# Cell-wise AUPRC: 0.0349
# Peak-wise AUROC: 0.5982
# Peak-wise AUPRC: 0.0356
python plot_save_umap.py --pred atac_babel.h5ad --true atac_test.h5ad
python cal_cluster.py --file atac_babel_umap.h5ad
# AMI: 0.3785
# ARI: 0.2003
# HOM: 0.4224
# NMI: 0.3826

## evaluation for val
python h5ad2h5.py -n test
python /fs/home/jiluzhang/BABEL/bin/predict_model.py --checkpoint ../train_out --data test.h5 --outdir test_out --device 6 --nofilter --noplot --transonly 
python match_cell.py
python cal_auroc_auprc.py --pred atac_babel.h5ad --true atac_test.h5ad
# Cell-wise AUROC: 0.7097
# Cell-wise AUPRC: 0.099
# Peak-wise AUROC: 0.5887
# Peak-wise AUPRC: 0.0605
python plot_save_umap.py --pred atac_babel.h5ad --true atac_test.h5ad
python cal_cluster.py --file atac_babel_umap.h5ad
# AMI: 0.5713
# ARI: 0.3985
# HOM: 0.5753
# NMI: 0.5771


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
snap.tl.leiden(atac, resolution=0.5)
atac.obs['leiden'].value_counts()
# 0    553
# 1    415
# 2    337
# 3    305
# 4    295
# 5    273
# 6    194
# 7    128

df_babel = pd.DataFrame({'cell_idx':atac.obs.index, 'cell_anno':atac.obs.leiden.astype('int')})
df_pos_babel = pd.merge(df_babel, pos)

mat_babel = np.zeros([50, 50])
for i in range(df_pos_babel.shape[0]):
    if df_pos_babel['cell_anno'][i]!=0:
        mat_babel[df_pos_babel['X'][i], (49-df_pos_babel['Y'][i])] = df_pos_babel['cell_anno'][i]  # start from upper right

plt.figure(figsize=(6, 5))
sns.heatmap(mat_babel, cmap=sns.color_palette(['grey', 'yellow', 'green', 'orange', 'cornflowerblue', 'purple', 'red', 'blue']), xticklabels=False, yticklabels=False)
plt.savefig('ATAC_spatial_babel.pdf')
plt.close()

normalized_mutual_info_score(atac.obs['cell_anno'], atac.obs['leiden'])  # 0.3808095016094799

## correlation
from scipy import stats
import numpy as np

stats.pearsonr(np.load('mat_thy1_babel.npy').flatten(), np.load('mat_thy1_true.npy').flatten())[0]       # 0.27912440524970544
stats.pearsonr(np.load('mat_bcl11b_babel.npy').flatten(), np.load('mat_bcl11b_true.npy').flatten())[0]   # 0.2524620672510995
