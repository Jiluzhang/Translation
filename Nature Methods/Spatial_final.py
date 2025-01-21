## workdir: /fs/home/jiluzhang/Nature_methods/Spatial_final
# cp /fs/home/jiluzhang/Nature_methods/Figure_1/scenario_4/rna.h5ad rna.h5ad     # 2597 × 17699
# cp /fs/home/jiluzhang/Nature_methods/Figure_1/scenario_4/atac.h5ad atac.h5ad   # 2597 × 217893

python split_train_val.py --RNA rna.h5ad --ATAC atac.h5ad --train_pct 0.9
python data_preprocess.py -r rna_train.h5ad -a atac_train.h5ad -s train_pt --dt train -n train --config rna2atac_config_train.yaml
python data_preprocess.py -r rna_val.h5ad -a atac_val.h5ad -s val_pt --dt val -n val --config rna2atac_config_val.yaml
nohup accelerate launch --config_file accelerator_config_train.yaml --main_process_port 29824 rna2atac_train.py --config_file rna2atac_config_train.yaml \
                        --train_data_dir train_pt --val_data_dir val_pt -s save -n rna2atac_brain > training_20250121.log &   # 1589507


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
python data_preprocess.py -r spatial_rna_test.h5ad -a spatial_atac_test.h5ad -s spatial_test_pt --dt test --config rna2atac_config_test.yaml
accelerate launch --config_file accelerator_config_test.yaml --main_process_port 29822 rna2atac_predict.py \
                  -d ./spatial_test_pt -l ./save/2025-01-21_rna2atac_brain_10/pytorch_model.bin --config_file rna2atac_config_test.yaml
python npy2h5ad.py
python cal_auroc_auprc.py --pred atac_cisformer.h5ad --true atac_test.h5ad
# Cell-wise AUROC: 0.6825
# Cell-wise AUPRC: 0.0293
# Peak-wise AUROC: 0.5662
# Peak-wise AUPRC: 0.0166
python plot_save_umap.py --pred atac_cisformer.h5ad --true atac_test.h5ad
python cal_cluster.py --file atac_cisformer_umap.h5ad
# AMI: 0.2928
# ARI: 0.1986
# HOM: 0.3144
# NMI: 0.2970



#### scbutterfly
# /data/home/jiluzhang/miniconda3/envs/scButterfly/lib/python3.9/site-packages/scButterfly/train_model.py
# line 329: "reconstruct_loss = r_loss_w * (lossR2R + lossA2R) + a_loss_w * (lossR2A + lossA2A)" -> "reconstruct_loss = lossR2A"
# line 374: "loss2 = d_loss(predict_rna.reshape(batch_size), torch.tensor(temp).cuda().float())" -> "loss2 = 0"
# butterfly.train_model() parameters: R2R_pretrain_epoch=0, A2A_pretrain_epoch=0, translation_kl_warmup=0, patience=10

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































