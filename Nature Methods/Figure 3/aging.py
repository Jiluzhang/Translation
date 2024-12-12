## mouse kidney
## https://cf.10xgenomics.com/samples/cell-arc/2.0.2/M_Kidney_Chromium_Nuc_Isolation_vs_SaltyEZ_vs_ComplexTissueDP/\
##                            M_Kidney_Chromium_Nuc_Isolation_vs_SaltyEZ_vs_ComplexTissueDP_filtered_feature_bc_matrix.h5
## kidney.h5

import scanpy as sc

dat = sc.read_10x_h5('kidney.h5', gex_only=False)                  # 14652 × 97637
dat.var_names_make_unique()

rna = dat[:, dat.var['feature_types']=='Gene Expression'].copy()   # 14652 × 32285
rna.write('rna.h5ad')

atac = dat[:, dat.var['feature_types']=='Peaks'].copy()            # 14652 × 65352
atac.X[atac.X!=0] = 1
atac.write('atac.h5ad')

conda create -n cisformer_Luz --clone /data/home/zouqihang/miniconda3/envs/cisformer
pip install torcheval
conda activate cisformer_Luz

python split_train_val.py --RNA rna.h5ad --ATAC atac.h5ad --train_pct 0.9

#### 2048_40
python data_preprocess.py -r rna_train.h5ad -a atac_train.h5ad -s train_pt --dt train -n train --config rna2atac_config_train.yaml
python data_preprocess.py -r rna_val.h5ad -a atac_val.h5ad -s val_pt --dt val -n val --config rna2atac_config_val.yaml
nohup accelerate launch --config_file accelerator_config_train.yaml --main_process_port 29823 rna2atac_train.py --config_file rna2atac_config_train.yaml \
                        --train_data_dir train_pt --val_data_dir val_pt -s save -n rna2atac_aging > 20241212.log &   # dec_depth:1  dec_heads:1  lr:2e-4  length:2048  mlt:40  gamma_step:5  gamma:0.5

import scanpy as sc
import numpy as np
rna = sc.read_h5ad('rna_val.h5ad')
np.count_nonzero(rna.X.toarray(), axis=1).max()  # 13845

import snapatac2 as snap
import scanpy as sc
atac = sc.read_h5ad('atac_val.h5ad')
snap.pp.select_features(atac)
snap.tl.spectral(atac)
snap.tl.umap(atac)
snap.pp.knn(atac)
snap.tl.leiden(atac)
atac.obs['cell_anno'] = atac.obs['leiden']
sc.pl.umap(atac, color='cell_anno', legend_fontsize='7', legend_loc='right margin', size=5,
           title='', frameon=True, save='_atac_val_true.pdf')
atac.write('atac_test.h5ad')


python data_preprocess.py -r rna_val.h5ad -a atac_val.h5ad -s test_pt --dt test --config rna2atac_config_test.yaml

#### 6_6
accelerate launch --config_file accelerator_config_train.yaml --main_process_port 29822 rna2atac_predict.py \
                  -d ./test_pt -l save_6_6/2024-12-11_rna2atac_aging_30/pytorch_model.bin --config_file rna2atac_config_test_6_6.yaml
python npy2h5ad.py
python plot_save_umap.py --pred atac_cisformer.h5ad --true atac_test.h5ad
python cal_auroc_auprc.py --pred atac_cisformer.h5ad --true atac_test.h5ad
python cal_cluster.py --file atac_cisformer_umap.h5ad

#### 1_1_epoch_10
accelerate launch --config_file accelerator_config_train.yaml --main_process_port 29822 rna2atac_predict.py \
                  -d ./test_pt -l save/2024-12-12_rna2atac_aging_10/pytorch_model.bin --config_file rna2atac_config_test_1_1.yaml
python npy2h5ad.py
python plot_save_umap.py --pred atac_cisformer.h5ad --true atac_test.h5ad
python cal_auroc_auprc.py --pred atac_cisformer.h5ad --true atac_test.h5ad
python cal_cluster.py --file atac_cisformer_umap.h5ad

#### 1_1_epoch_5
accelerate launch --config_file accelerator_config_train.yaml --main_process_port 29822 rna2atac_predict.py \
                  -d ./test_pt -l save/2024-12-12_rna2atac_aging_5/pytorch_model.bin --config_file rna2atac_config_test_1_1.yaml
python npy2h5ad.py
python plot_save_umap.py --pred atac_cisformer.h5ad --true atac_test.h5ad
python cal_auroc_auprc.py --pred atac_cisformer.h5ad --true atac_test.h5ad
python cal_cluster.py --file atac_cisformer_umap.h5ad


#### 1_1_epoch_15
accelerate launch --config_file accelerator_config_train.yaml --main_process_port 29822 rna2atac_predict.py \
                  -d ./test_pt -l save/2024-12-12_rna2atac_aging_15/pytorch_model.bin --config_file rna2atac_config_test_1_1.yaml
python npy2h5ad.py
python plot_save_umap.py --pred atac_cisformer.h5ad --true atac_test.h5ad
python cal_auroc_auprc.py --pred atac_cisformer.h5ad --true atac_test.h5ad
python cal_cluster.py --file atac_cisformer_umap.h5ad

