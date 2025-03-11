## mouse heart
## https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE281085
## Beads_1: https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM8610nnn/GSM8610658/suppl/GSM8610658_Beads_1_filtered_feature_bc_matrix.h5
## Beads_2: https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM8610nnn/GSM8610659/suppl/GSM8610659_Beads_2_filtered_feature_bc_matrix.h5
## Beads_3: https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM8610nnn/GSM8610660/suppl/GSM8610660_Beads_3_filtered_feature_bc_matrix.h5

import scanpy as sc
import numpy as np

beads_1 = sc.read_10x_h5('GSM8610658_Beads_1_filtered_feature_bc_matrix.h5', gex_only=False)  # 9541 × 147252
beads_1.var_names_make_unique()
beads_2 = sc.read_10x_h5('GSM8610659_Beads_2_filtered_feature_bc_matrix.h5', gex_only=False)  # 12076 × 149979
beads_2.var_names_make_unique()
beads_3 = sc.read_10x_h5('GSM8610660_Beads_3_filtered_feature_bc_matrix.h5', gex_only=False)  # 9653 × 169993
beads_3.var_names_make_unique()

# select data of beads_2
## rna
rna = beads_2[:, beads_2.var['feature_types']=='Gene Expression'].copy()   # 12076 × 32285
sc.pp.filter_genes(rna, min_cells=10)       # 12076 × 18249
sc.pp.filter_cells(rna, min_genes=200)      # 11994 × 18249
sc.pp.filter_cells(rna, max_genes=20000)    # 11994 × 18249
rna.obs['n_genes'].quantile(0.7)            # 2003.0
rna.obs['n_genes'].quantile(0.8)            # 2330.0
rna.obs['n_genes'].quantile(0.9)            # 2945.0

## atac
atac = beads_2[:, beads_2.var['feature_types']=='Peaks'].copy()    # 12076 × 117694
atac.X[atac.X!=0] = 1

sc.pp.filter_genes(atac, min_cells=10)     # 12076 × 117694
sc.pp.filter_cells(atac, min_genes=500)    # 11026 × 117694
sc.pp.filter_cells(atac, max_genes=50000)  # 11026 × 117694
atac.obs['n_genes'].quantile(0.7)          # 4912.0
atac.obs['n_genes'].quantile(0.8)          # 6017.0
atac.obs['n_genes'].quantile(0.9)          # 8551.0

idx = np.intersect1d(rna.obs.index, atac.obs.index)
del rna.obs['n_genes']
del rna.var['n_cells']
rna[idx, :].copy().write('rna.h5ad')     # 10944 × 18249
del atac.obs['n_genes']
del atac.var['n_cells']
atac[idx, :].copy().write('atac.h5ad')   # 10944 × 117694


#################################### HERE ####################################
#################################### HERE ####################################
#################################### HERE ####################################
#################################### HERE ####################################
#################################### HERE ####################################


python split_train_val.py --RNA rna.h5ad --ATAC atac.h5ad --train_pct 0.9

#### 4096_2048_40
python data_preprocess.py -r rna_train.h5ad -a atac_train.h5ad -s train_pt --dt train -n train --config rna2atac_config_train.yaml  # 22 min
python data_preprocess.py -r rna_val.h5ad -a atac_val.h5ad -s val_pt --dt val -n val --config rna2atac_config_val.yaml
nohup accelerate launch --config_file accelerator_config_train.yaml --main_process_port 29823 rna2atac_train.py --config_file rna2atac_config_train.yaml \
                        --train_data_dir train_pt --val_data_dir val_pt -s save -n rna2atac_aging > 20250311.log &   # 2888722


# import scanpy as sc
# import numpy as np
# rna = sc.read_h5ad('rna_val.h5ad')
# np.count_nonzero(rna.X.toarray(), axis=1).max()  # 13845

# import snapatac2 as snap
# import scanpy as sc
# atac = sc.read_h5ad('atac_val.h5ad')
# snap.pp.select_features(atac)
# snap.tl.spectral(atac)
# snap.tl.umap(atac)
# snap.pp.knn(atac)
# snap.tl.leiden(atac)
# atac.obs['cell_anno'] = atac.obs['leiden']
# sc.pl.umap(atac, color='cell_anno', legend_fontsize='7', legend_loc='right margin', size=5,
#            title='', frameon=True, save='_atac_val_true.pdf')
# atac.write('atac_test.h5ad')


# python data_preprocess.py -r rna_val.h5ad -a atac_val.h5ad -s test_pt --dt test --config rna2atac_config_test.yaml




# kidney dataset: wget -c https://datasets.cellxgene.cziscience.com/b6fff8a8-b297-4715-b041-43d23a4e0474.h5ad
# -> heart_aging.h5ad

## map rna & atac to reference
import scanpy as sc
import pandas as pd
from scipy.sparse import csr_matrix
import numpy as np

rna = sc.read_h5ad('kidney_aging.h5ad')  # 21647 × 17984

## map rna
ref_rna = sc.read_h5ad('rna.h5ad')
genes = ref_rna.var[['gene_ids']]
genes['gene_name'] = genes.index.values
genes.reset_index(drop=True, inplace=True)

rna_exp = pd.concat([rna.var['feature_name'], pd.DataFrame(rna.raw.X.toarray().T, index=rna.var.index.values)], axis=1)
rna_exp['gene_ids'] = rna_exp.index.values
rna_exp.rename(columns={'feature_name': 'gene_name'}, inplace=True)
X_new = pd.merge(genes, rna_exp, how='left', on='gene_ids').iloc[:, 3:].T
X_new.fillna(value=0, inplace=True)

rna_new = sc.AnnData(X_new.values, obs=rna.obs, var=pd.DataFrame({'gene_ids': genes['gene_ids'], 'feature_types': 'Gene Expression'}))
rna_new.var.index = genes['gene_name'].values
rna_new.X = csr_matrix(rna_new.X)  # 21647 × 20856
rna_new.write('rna_unpaired.h5ad')
np.count_nonzero(rna_new.X.toarray(), axis=1).max()  # 7577

# kidney proximal convoluted tubule epithelial cell            4460
# B cell                                                       3124
# epithelial cell of proximal tubule                           3053
# kidney loop of Henle thick ascending limb epithelial cell    1554
# lymphocyte                                                   1536
# macrophage                                                   1407
# T cell                                                       1359
# fenestrated cell                                             1027
# kidney collecting duct principal cell                         789
# kidney distal convoluted tubule epithelial cell               744
# plasmatocyte                                                  496
# brush cell                                                    414
# kidney cortex artery cell                                     381
# plasma cell                                                   325
# mesangial cell                                                261
# kidney loop of Henle ascending limb epithelial cell           201
# kidney capillary endothelial cell                             161
# fibroblast                                                    161
# kidney proximal straight tubule epithelial cell                95
# natural killer cell                                            66
# kidney collecting duct epithelial cell                         25
# leukocyte                                                       5
# kidney cell                                                     3

## pseudo atac
ref_atac = sc.read_h5ad('atac.h5ad')
atac_new = sc.AnnData(np.zeros([rna.n_obs, ref_atac.n_vars]), obs=rna.obs, var=ref_atac.var)
atac_new.X[:, 0] = 1
atac_new.X = csr_matrix(atac_new.X)
atac_new.write('atac_unpaired.h5ad')


python data_preprocess_unpaired.py -r rna_unpaired.h5ad -a atac_unpaired.h5ad -s preprocessed_data_unpaired --dt test --config rna2atac_config_unpaired.yaml
accelerate launch --config_file accelerator_config_test.yaml --main_process_port 29822 rna2atac_test.py \
                  -d ./preprocessed_data_unpaired \
                  -l save/2024-08-21_rna2atac_train_55/pytorch_model.bin --config_file rna2atac_config_unpaired.yaml   # ~ 0.5h


#### 1_1_epoch_30 (not better than epoch_15)
accelerate launch --config_file accelerator_config_train.yaml --main_process_port 29822 rna2atac_predict.py \
                  -d ./test_pt -l save/2024-12-12_rna2atac_aging_30/pytorch_model.bin --config_file rna2atac_config_test_1_1.yaml
python npy2h5ad.py
python plot_save_umap.py --pred atac_cisformer.h5ad --true atac_test.h5ad
python cal_auroc_auprc.py --pred atac_cisformer.h5ad --true atac_test.h5ad
python cal_cluster.py --file atac_cisformer_umap.h5ad
