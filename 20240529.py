#################### HNSCC -> HNSCC ####################
import scanpy as sc
import snapatac2 as snap

## RNA
rna = sc.read_h5ad('/fs/home/jiluzhang/2023_nature_LD/normal_h5ad/P5504-N1_rna.h5ad')       # 2336 × 38244
sc.pp.filter_genes(rna, min_cells=10)            # 2336 × 13982
rna.write('hnscc_rna.h5ad')

## ATAC
atac = sc.read_h5ad('/fs/home/jiluzhang/2023_nature_LD/normal_h5ad/P5504-N1_atac.h5ad')     # 2336 × 1033239
sc.pp.filter_genes(atac, min_cells=10)              # 2336 × 166487
atac.write('hnscc_atac.h5ad')


python split_train_val_test.py --RNA hnscc_rna.h5ad --ATAC hnscc_atac.h5ad
python rna2atac_data_preprocess.py --config_file rna2atac_config.yaml --dataset_type train
python rna2atac_data_preprocess.py --config_file rna2atac_config_whole.yaml --dataset_type val
# python rna2atac_data_preprocess.py --config_file rna2atac_config_whole.yaml --dataset_type test
accelerate launch --config_file accelerator_config.yaml rna2atac_pretrain.py --config_file rna2atac_config.yaml \
                  --train_data_dir ./preprocessed_data_train --val_data_dir ./preprocessed_data_val -n rna2atac_train
accelerate launch --config_file accelerator_config.yaml --main_process_port 29821 rna2atac_evaluate.py -d ./preprocessed_data_val \
                  -l save/pytorch_model_epoch_3.bin --config_file rna2atac_config_whole.yaml && mv hnscc_predict.npy hnscc_val_predict.npy

atac_train = snap.read('hnscc_atac_train_0.h5ad', backed=None)
snap.pp.select_features(atac_train)
snap.tl.spectral(atac_train)
snap.tl.umap(atac_train)
snap.pp.knn(atac_train)
snap.tl.leiden(atac_train, resolution=1)
snap.pl.umap(atac_train, color='leiden', show=False, out_file='umap_hnscc_train_true.pdf', marker_size=2.0, height=500)
atac_train.write('hnscc_atac_train_0_res.h5ad')

atac_val = snap.read('hnscc_atac_val_0.h5ad', backed=None)
snap.pp.select_features(atac_val)
snap.tl.spectral(atac_val)
snap.tl.umap(atac_val)
snap.pp.knn(atac_val)
snap.tl.leiden(atac_val, resolution=1)
snap.pl.umap(atac_val, color='leiden', show=False, out_file='umap_hnscc_val_true.pdf', marker_size=2.0, height=500)
atac_val.write('hnscc_atac_val_0_res.h5ad')



## plot umap for train & valid & test
import snapatac2 as snap
import numpy as np
from scipy.sparse import csr_matrix

m_raw = np.load('hnscc_val_predict.npy')
m = m_raw.copy()
# [sum(m_raw[i]>0.7) for i in range(5)]
m[m>0.7]=1
m[m<=0.7]=0

atac_true = snap.read('hnscc_atac_val_0_res.h5ad', backed=None)
atac_pred = atac_true.copy()
atac_pred.X = csr_matrix(m)

snap.pp.select_features(atac_pred)
snap.tl.spectral(atac_pred)
snap.tl.umap(atac_pred)
snap.pl.umap(atac_pred, color='leiden', show=False, out_file='umap_hnscc_val_predict.pdf', marker_size=2.0, height=500)



m_raw = np.load('hnscc_train_predict.npy')
m = m_raw.copy()
# [sum(m_raw[i]>0.7) for i in range(5)]
m[m>0.7]=1
m[m<=0.7]=0

atac_true = snap.read('hnscc_atac_train_0_res.h5ad', backed=None)
atac_pred = atac_true.copy()
atac_pred.X = csr_matrix(m)

snap.pp.select_features(atac_pred)
snap.tl.spectral(atac_pred)
snap.tl.umap(atac_pred)
snap.pl.umap(atac_pred, color='leiden', show=False, out_file='umap_hnscc_train_predict.pdf', marker_size=2.0, height=500)

