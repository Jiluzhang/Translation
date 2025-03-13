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

#### 4096_2048_40_3_3
python data_preprocess.py -r rna_train.h5ad -a atac_train.h5ad -s train_pt --dt train -n train --config rna2atac_config_train.yaml  # 22 min
python data_preprocess.py -r rna_val.h5ad -a atac_val.h5ad -s val_pt --dt val -n val --config rna2atac_config_val.yaml
nohup accelerate launch --config_file accelerator_config_train.yaml --main_process_port 29823 rna2atac_train.py --config_file rna2atac_config_train.yaml \
                        --train_data_dir train_pt --val_data_dir val_pt -s save -n rna2atac_aging > 20250311.log &   # 2888722

#### 4096_2048_40_6_6
nohup accelerate launch --config_file accelerator_config_train.yaml --main_process_port 29823 rna2atac_train.py --config_file rna2atac_config_train_6_6.yaml \
                        --train_data_dir train_pt --val_data_dir val_pt -s save_6_6 -n rna2atac_aging > 20250313.log &   # 2812281


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
import matplotlib.pyplot as plt

plt.rcParams['pdf.fonttype'] = 42

dat = np.load('predict.npy')
dat[dat>0.5] = 1
dat[dat<=0.5] = 0
atac = sc.read_h5ad('atac_unpaired.h5ad')
atac.X = csr_matrix(dat)  # 21647 × 65352

atac = atac[atac.obs['cell_type'].isin(['kidney proximal convoluted tubule epithelial cell',
                                        'B cell',
                                        'epithelial cell of proximal tubule',
                                        'kidney loop of Henle thick ascending limb epithelial cell',
                                        'macrophage', 
                                        'T cell',
                                        'fenestrated cell',
                                        'kidney collecting duct principal cell',
                                        'kidney distal convoluted tubule epithelial cell'])].copy()  
# 17517 × 65352 (delete too common cell annotation & too small cell number)

# atac_ref = sc.read_h5ad('atac.h5ad')
# snap.pp.select_features(atac_ref)
# snap.tl.spectral(atac_ref)
# snap.tl.umap(atac_ref)
# snap.pp.knn(atac_ref)
# snap.tl.leiden(atac_ref)
# sc.pl.umap(atac_ref, color='leiden', legend_fontsize='7', legend_loc='right margin', size=5,
#            title='', frameon=True, save='_ref_atac.pdf')

snap.pp.select_features(atac)
snap.tl.spectral(atac)  # snap.tl.spectral(atac, n_comps=100, weighted_by_sd=False)
snap.tl.umap(atac)

atac.write('atac_predict.h5ad')

# atac = sc.read_h5ad('atac_predict.h5ad')

sc.pl.umap(atac, color='cell_type', legend_fontsize='7', legend_loc='right margin', size=5,
           title='', frameon=True, save='_predict_atac.pdf')

# sc.pl.umap(atac, color='cell_type', groups=['kidney loop of Henle ascending limb epithelial cell',
#                                             'kidney distal convoluted tubule epithelial cell',
#                                             'kidney collecting duct principal cell',
#                                             'kidney loop of Henle thick ascending limb epithelial cell'],
#            legend_fontsize='7', legend_loc='right margin', size=5, title='', frameon=True, save='_predict_atac_1.pdf')

# sc.pl.umap(atac, color='cell_type', groups=['epithelial cell of proximal tubule',
#                                             'kidney proximal convoluted tubule epithelial cell',
#                                             'brush cell'],
#            legend_fontsize='7', legend_loc='right margin', size=5, title='', frameon=True, save='_predict_atac_2.pdf')

# sc.pl.umap(atac, color='cell_type', groups=['T cell', 'B cell', 'macrophage', 'mesangial cell',
#                                             'plasma cell', 'fibroblast'],
#            legend_fontsize='7', legend_loc='right margin', size=5, title='', frameon=True, save='_predict_atac_3.pdf')

# sc.pl.umap(atac, color='cell_type', groups=['fenestrated cell', 'kidney cortex artery cell', 'kidney capillary endothelial cell'],
#            legend_fontsize='7', legend_loc='right margin', size=5, title='', frameon=True, save='_predict_atac_4.pdf')


## cell annotation umap for rna
rna = sc.read_h5ad('rna_unpaired.h5ad')   # 21647 × 20856
rna = rna[rna.obs['cell_type'].isin(['kidney proximal convoluted tubule epithelial cell',
                                     'B cell',
                                     'epithelial cell of proximal tubule',
                                     'kidney loop of Henle thick ascending limb epithelial cell',
                                     'macrophage', 
                                     'T cell',
                                     'fenestrated cell',
                                     'kidney collecting duct principal cell',
                                     'kidney distal convoluted tubule epithelial cell'])].copy()
# 17517 × 20856

sc.pp.normalize_total(rna, target_sum=1e4)
sc.pp.log1p(rna)
sc.pp.highly_variable_genes(rna)
rna.raw = rna
rna = rna[:, rna.var.highly_variable]  # 17517 × 1922

sc.tl.pca(rna)
sc.pp.neighbors(rna)
sc.tl.umap(rna)
rna.uns['cell_type_colors'] = atac.uns['cell_type_colors']  # keep consistent color bw rna & atac
rna.write('rna_unpaired_cell_anno.h5ad')
sc.pl.umap(rna, color='cell_type', legend_fontsize='7', legend_loc='right margin', size=5, title='', frameon=True, save='_rna.pdf')

## reduce resolution
sc.tl.pca(rna, n_comps=6)
sc.pp.neighbors(rna)
sc.tl.umap(rna)
rna.uns['cell_type_colors'] = atac.uns['cell_type_colors']  # keep consistent color bw rna & atac
sc.pl.umap(rna, color='cell_type', legend_fontsize='7', legend_loc='right margin', size=5, title='', frameon=True, save='_rna_ncomps_6.pdf')
rna.write('rna_unpaired_cell_anno.h5ad')

# sc.pl.umap(rna, color='cell_type', groups=['kidney loop of Henle ascending limb epithelial cell',
#                                             'kidney distal convoluted tubule epithelial cell',
#                                             'kidney collecting duct principal cell',
#                                             'kidney loop of Henle thick ascending limb epithelial cell'],
#            legend_fontsize='7', legend_loc='right margin', size=5, title='', frameon=True, save='_rna_1.pdf')

# sc.pl.umap(rna, color='cell_type', groups=['epithelial cell of proximal tubule',
#                                             'kidney proximal convoluted tubule epithelial cell',
#                                             'brush cell'],
#            legend_fontsize='7', legend_loc='right margin', size=5, title='', frameon=True, save='_rna_2.pdf')

# sc.pl.umap(atac, color='cell_type', groups=['T cell', 'B cell', 'macrophage', 'mesangial cell',
#                                             'plasma cell', 'fibroblast'],
#            legend_fontsize='7', legend_loc='right margin', size=5, title='', frameon=True, save='_rna_3.pdf')

# sc.pl.umap(rna, color='cell_type', groups=['fenestrated cell', 'kidney cortex artery cell', 'kidney capillary endothelial cell'],
#            legend_fontsize='7', legend_loc='right margin', size=5, title='', frameon=True, save='_rna_4.pdf')


## plot box for cdkn2a expression 
import scanpy as sc
from plotnine import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['pdf.fonttype'] = 42

rna = sc.read_h5ad('rna_unpaired_cell_anno.h5ad')
rna.obs['cell_type'].value_counts()
# kidney proximal convoluted tubule epithelial cell            4460
# B cell                                                       3124
# epithelial cell of proximal tubule                           3053
# kidney loop of Henle thick ascending limb epithelial cell    1554
# macrophage                                                   1407
# T cell                                                       1359
# fenestrated cell                                             1027
# kidney collecting duct principal cell                         789
# kidney distal convoluted tubule epithelial cell               744

rna_kp = rna[rna.obs['cell_type']=='kidney proximal convoluted tubule epithelial cell'].copy()
df = pd.DataFrame({'age':rna_kp.obs['age'],
                   'exp':rna_kp.X[:, np.argwhere(rna_kp.var.index=='Cdkn1a').flatten().item()].toarray().flatten()})
df['age'] = pd.Categorical(df['age'], categories=['1m', '3m', '18m', '21m', '30m'])
p = ggplot(df, aes(x='age', y='exp', fill='age')) + geom_jitter(size=1, width=0.2, height=0, show_legend=False) + xlab('Age') + ylab('Cdkn1a expression level') +\
                                                    scale_y_continuous(limits=[0, 4], breaks=np.arange(0, 4+0.1, 1.0)) +\
                                                    stat_summary(fun_y=np.mean, geom='point', color='red', size=3, shape=0, stroke=1, show_legend=False) +\
                                                    stat_summary(fun_y=np.mean, geom='point', color='red', size=3, shape=1, stroke=1, show_legend=False) + theme_bw()
p.save(filename='Cdkn1a_exp_dotplot_kp.pdf', dpi=600, height=4, width=4)

# rna_kp.obs['cdkn1a_exp'] = rna_kp.X[:, np.argwhere(rna_kp.var.index=='Cdkn1a').flatten().item()].toarray().flatten()
# sc.pl.violin(rna_kp, keys='cdkn1a_exp', groupby='age', rotation=90, stripplot=False, save='_cdkn1a_exp_kp.pdf')

# p = ggplot(df, aes(x='age', y='exp', fill='age')) + geom_boxplot(width=0.5, show_legend=False, outlier_shape='') +\
#                                                     xlab('Age') + ylab('Cdkn1a expression') +\
#                                                     scale_y_continuous(limits=[0, 1.5], breaks=np.arange(0, 1.5+0.1, 0.3)) + theme_bw()
# p.save(filename='Cdkn1a_exp_boxplot.pdf', dpi=600, height=4, width=4)









