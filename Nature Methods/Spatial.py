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




