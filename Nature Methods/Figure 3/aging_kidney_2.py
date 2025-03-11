## mouse kidney
## https://cf.10xgenomics.com/samples/cell-arc/2.0.2/M_Kidney_Chromium_Nuc_Isolation_vs_SaltyEZ_vs_ComplexTissueDP/\
##                            M_Kidney_Chromium_Nuc_Isolation_vs_SaltyEZ_vs_ComplexTissueDP_filtered_feature_bc_matrix.h5
## kidney.h5

import scanpy as sc
import numpy as np

dat = sc.read_10x_h5('kidney.h5', gex_only=False)                  # 14652 × 97637
dat.var_names_make_unique()

## rna
rna = dat[:, dat.var['feature_types']=='Gene Expression'].copy()   # 14652 × 32285
sc.pp.filter_genes(rna, min_cells=10)       # 14652 × 20856
sc.pp.filter_cells(rna, min_genes=200)      # 14613 × 20856
sc.pp.filter_cells(rna, max_genes=20000)    # 14613 × 20856
rna.obs['n_genes'].quantile(0.8)            # 3403.0
rna.obs['n_genes'].quantile(0.9)            # 4438.8

## atac
atac = dat[:, dat.var['feature_types']=='Peaks'].copy()            # 14652 × 65352
atac.X[atac.X!=0] = 1

sc.pp.filter_genes(atac, min_cells=10)     # 14652 × 65352
sc.pp.filter_cells(atac, min_genes=500)    # 11679 × 65352
sc.pp.filter_cells(atac, max_genes=50000)  # 11679 × 65352
atac.obs['n_genes'].quantile(0.7)          # 3951.0
atac.obs['n_genes'].quantile(0.8)          # 4947.0
atac.obs['n_genes'].quantile(0.9)          # 6805.4

idx = np.intersect1d(rna.obs.index, atac.obs.index)
del rna.obs['n_genes']
del rna.var['n_cells']
rna[idx, :].copy().write('rna.h5ad')     # 11662 × 20856
del atac.obs['n_genes']
del atac.var['n_cells']
atac[idx, :].copy().write('atac.h5ad')   # 11662 × 65352


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


## Tabula Muris Senis: https://cellxgene.cziscience.com/collections/0b9d8a04-bb9d-44da-aa27-705bb65b54eb
# kidney dataset: wget -c https://datasets.cellxgene.cziscience.com/e896f2b3-e22b-42c0-81e8-9dd1b6e80d64.h5ad
# gene expression: https://cellxgene.cziscience.com/gene-expression
# -> kidney_aging.h5ad

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


python data_preprocess_unpaired.py -r rna_unpaired.h5ad -a atac_unpaired.h5ad -s preprocessed_data_unpaired --dt test --config rna2atac_config_unpaired.yaml  # 45 min
accelerate launch --config_file accelerator_config_test.yaml --main_process_port 29822 rna2atac_predict.py \
                  -d ./preprocessed_data_unpaired -l save/2025-03-11_rna2atac_aging_15/pytorch_model.bin --config_file rna2atac_config_test.yaml  # 15 min
# python npy2h5ad.py
# python plot_save_umap.py --pred atac_cisformer.h5ad --true atac_test.h5ad
# python cal_auroc_auprc.py --pred atac_cisformer.h5ad --true atac_test.h5ad
# python cal_cluster.py --file atac_cisformer_umap.h5ad

## plot umap for atac
import scanpy as sc
import numpy as np
from scipy.sparse import csr_matrix
import snapatac2 as snap
import pandas as pd
from tqdm import tqdm

dat = np.load('predict.npy')
dat[dat>0.5] = 1
dat[dat<=0.5] = 0
atac = sc.read_h5ad('atac_unpaired.h5ad')
atac.X = csr_matrix(dat)  # 21647 × 65352

snap.pp.select_features(atac)
snap.tl.spectral(atac)  # snap.tl.spectral(atac, n_comps=100, weighted_by_sd=False)
snap.tl.umap(atac)

kidney_cell_anno_dict = {
  'kidney proximal convoluted tubule epithelial cell':'tubule epithelial cell',
  'B cell':'B cell',
  'epithelial cell of proximal tubule':'tubule epithelial cell',
  'kidney loop of Henle thick ascending limb epithelial cell':'limb epithelial cell',
  'lymphocyte':'others',
  'macrophage':'macrophage',
  'T cell':'T cell',
  'fenestrated cell':'endothelial cell',
  'kidney collecting duct principal cell':'epithelial cell',
  'kidney distal convoluted tubule epithelial cell':'tubule epithelial cell',
  'plasmatocyte':'others',
  'brush cell':'epithelial cell',
  'kidney cortex artery cell':'others',
  'plasma cell':'B cell',
  'mesangial cell':'others',
  'kidney loop of Henle ascending limb epithelial cell':'limb epithelial cell',
  'kidney capillary endothelial cell':'endothelial cell',
  'fibroblast':'fibroblast',
  'kidney proximal straight tubule epithelial cell':'tubule epithelial cell',
  'natural killer cell':'NK',
  'kidney collecting duct epithelial cell': 'epithelial cell',
  'leukocyte':'others',
  'kidney cell':'others'
}

atac.obs['cell_anno'] = atac.obs['cell_type'].replace(kidney_cell_anno_dict)
atac.obs['cell_anno'].value_counts()
# epithelial cell     11335
# B cell               3449
# others               2682
# macrophage           1407
# T cell               1359
# endothelial cell     1188
# fibroblast            161
# NK                     66

sc.pl.umap(atac, color='cell_type', legend_fontsize='7', legend_loc='right margin', size=10,
           title='', frameon=True, save='_predict_atac.pdf')
sc.pl.umap(atac, color='cell_anno', legend_fontsize='7', legend_loc='right margin', size=10,
           title='', frameon=True, save='_predict_atac_big_cluster.pdf')

atac.write('atac_predict.h5ad')

# sc.pl.umap(atac[atac.obs['cell_type'].isin(atac.obs.cell_type.value_counts()[:5].index), :], color='cell_type', legend_fontsize='7', legend_loc='right margin', size=10,
#            title='', frameon=True, save='_unpaired_tmp.pdf')

sc.pl.umap(atac[atac.obs['cell_type'].isin(['fenestrated cell', 'B cell']), :], color='cell_type', legend_fontsize='7', legend_loc='right margin', size=10,
            title='', frameon=True, save='_unpaired_luz.pdf')

sc.pl.umap(atac, color='cell_type', legend_fontsize='7', legend_loc='right margin', size=10,
           title='', frameon=True, save='_unpaired_tmp.pdf')

sc.pl.umap(atac, color='cell_type', legend_fontsize='7', legend_loc='right margin', size=10,
           title='', frameon=True, save='_unpaired.pdf')

sc.pl.umap(atac, color='cell_type', legend_fontsize='7', legend_loc='right margin', size=10,
           title='', frameon=True, save='_unpaired_n_comps_200.pdf')
