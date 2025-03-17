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

accelerate launch --config_file accelerator_config_test.yaml --main_process_port 29822 rna2atac_predict.py \
                  -d ./preprocessed_data_unpaired -l save_6_6/2025-03-13_rna2atac_aging_30/pytorch_model.bin --config_file rna2atac_config_test_6_6.yaml  #  min

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


## plot box for cdkn1a expression 
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

# 近端PCT上皮细胞主要参与大规模的物质重吸收，具有微绒毛和丰富的线粒体。
# 远端DCT上皮细胞则主要调节钠、钙和酸碱平衡，细胞表面较为光滑，且受激素调节。

## epithelial cells
rna_epi = rna[rna.obs['cell_type'].isin(['kidney proximal convoluted tubule epithelial cell',
                                         'epithelial cell of proximal tubule',
                                         'kidney loop of Henle thick ascending limb epithelial cell',
                                         'kidney collecting duct principal cell',
                                         'kidney distal convoluted tubule epithelial cell'])].copy()
df = pd.DataFrame({'age':rna_epi.obs['age'],
                   'exp':rna_epi.X[:, np.argwhere(rna_epi.var.index=='Cdkn1a').flatten().item()].toarray().flatten()})
df['age'] = pd.Categorical(df['age'], categories=['1m', '3m', '18m', '21m', '24m', '30m'])
p = ggplot(df, aes(x='age', y='exp', fill='age')) + geom_jitter(size=1, width=0.2, height=0, show_legend=False) + xlab('Age') + ylab('Cdkn1a expression level') +\
                                                    scale_y_continuous(limits=[0, 4], breaks=np.arange(0, 4+0.1, 1.0)) +\
                                                    stat_summary(fun_y=np.mean, geom='point', color='red', size=3, shape=0, stroke=1, show_legend=False) +\
                                                    stat_summary(fun_y=np.mean, geom='point', color='red', size=3, shape=1, stroke=1, show_legend=False) + theme_bw()
p.save(filename='Cdkn1a_exp_dotplot_epi.pdf', dpi=600, height=4, width=4)
df['age'].value_counts()
# 30m    3929
# 18m    2092
# 1m     1750
# 3m     1424
# 21m    1405
# 24m       0

## fenestrated cell
rna_fc = rna[rna.obs['cell_type'].isin(['fenestrated cell'])].copy()
df = pd.DataFrame({'age':rna_fc.obs['age'],
                   'exp':rna_fc.X[:, np.argwhere(rna_fc.var.index=='Cdkn1a').flatten().item()].toarray().flatten()})
df['age'] = pd.Categorical(df['age'], categories=['1m', '3m', '18m', '21m', '24m', '30m'])
p = ggplot(df, aes(x='age', y='exp', fill='age')) + geom_jitter(size=1, width=0.2, height=0, show_legend=False) + xlab('Age') + ylab('Cdkn1a expression level') +\
                                                    scale_y_continuous(limits=[0, 4], breaks=np.arange(0, 4+0.1, 1.0)) +\
                                                    stat_summary(fun_y=np.mean, geom='point', color='red', size=3, shape=0, stroke=1, show_legend=False) +\
                                                    stat_summary(fun_y=np.mean, geom='point', color='red', size=3, shape=1, stroke=1, show_legend=False) + theme_bw()
p.save(filename='Cdkn1a_exp_dotplot_fc.pdf', dpi=600, height=4, width=4)
df['age'].value_counts()
# 30m    281
# 1m     206
# 18m    190
# 3m     181
# 21m    169
# 24m      0

## immune cells
rna_ic = rna[rna.obs['cell_type'].isin(['macrophage', 'T cell', 'B cell'])].copy()
df = pd.DataFrame({'age':rna_ic.obs['age'],
                   'exp':rna_ic.X[:, np.argwhere(rna_ic.var.index=='Cdkn1a').flatten().item()].toarray().flatten()})
df['age'] = pd.Categorical(df['age'], categories=['1m', '3m', '18m', '21m', '24m', '30m'])
p = ggplot(df, aes(x='age', y='exp', fill='age')) + geom_jitter(size=1, width=0.2, height=0, show_legend=False) + xlab('Age') + ylab('Cdkn1a expression level') +\
                                                    scale_y_continuous(limits=[0, 4], breaks=np.arange(0, 4+0.1, 1.0)) +\
                                                    stat_summary(fun_y=np.mean, geom='point', color='red', size=3, shape=0, stroke=1, show_legend=False) +\
                                                    stat_summary(fun_y=np.mean, geom='point', color='red', size=3, shape=1, stroke=1, show_legend=False) + theme_bw()
p.save(filename='Cdkn1a_exp_dotplot_ic.pdf', dpi=600, height=4, width=4)
df['age'].value_counts()
# 24m    4193
# 30m     861
# 18m     380
# 21m     205
# 3m      166
# 1m       85

## kidney proximal convoluted tubule epithelial cell
rna_kp = rna[rna.obs['cell_type']=='kidney proximal convoluted tubule epithelial cell'].copy()
df = pd.DataFrame({'age':rna_kp.obs['age'],
                   'exp':rna_kp.X[:, np.argwhere(rna_kp.var.index=='Cdkn1a').flatten().item()].toarray().flatten()})
df['age'] = pd.Categorical(df['age'], categories=['1m', '3m', '18m', '21m', '30m'])
p = ggplot(df, aes(x='age', y='exp', fill='age')) + geom_jitter(size=1, width=0.2, height=0, show_legend=False) + xlab('Age') + ylab('Cdkn1a expression level') +\
                                                    scale_y_continuous(limits=[0, 4], breaks=np.arange(0, 4+0.1, 1.0)) +\
                                                    stat_summary(fun_y=np.mean, geom='point', color='red', size=3, shape=0, stroke=1, show_legend=False) +\
                                                    stat_summary(fun_y=np.mean, geom='point', color='red', size=3, shape=1, stroke=1, show_legend=False) + theme_bw()
p.save(filename='Cdkn1a_exp_dotplot_kp.pdf', dpi=600, height=4, width=4)
df['age'].value_counts()
# 18m    1117
# 1m      938
# 30m     864
# 21m     860
# 3m      681

# rna_kp.obs['cdkn1a_exp'] = rna_kp.X[:, np.argwhere(rna_kp.var.index=='Cdkn1a').flatten().item()].toarray().flatten()
# sc.pl.violin(rna_kp, keys='cdkn1a_exp', groupby='age', rotation=90, stripplot=False, save='_cdkn1a_exp_kp.pdf')

# p = ggplot(df, aes(x='age', y='exp', fill='age')) + geom_boxplot(width=0.5, show_legend=False, outlier_shape='') +\
#                                                     xlab('Age') + ylab('Cdkn1a expression') +\
#                                                     scale_y_continuous(limits=[0, 1.5], breaks=np.arange(0, 1.5+0.1, 0.3)) + theme_bw()
# p.save(filename='Cdkn1a_exp_boxplot.pdf', dpi=600, height=4, width=4)

## B cell
rna_b = rna[rna.obs['cell_type']=='B cell'].copy()
df = pd.DataFrame({'age':rna_b.obs['age'],
                   'exp':rna_b.raw.X[:, np.argwhere(rna_b.raw.var.index=='Cdkn1a').flatten().item()].toarray().flatten()})
df['age'] = df['age'].replace({'1m':'1,3m', '3m':'1,3m', '18m':'18,21m', '21m':'18,21m', '24m':'24m', '30m':'30m'})
df['age'] = pd.Categorical(df['age'], categories=['1,3m', '18,21m', '24m', '30m'])
p = ggplot(df, aes(x='age', y='exp', fill='age')) + geom_jitter(size=1, width=0.2, height=0, show_legend=False) + xlab('Age') + ylab('Cdkn1a expression level') +\
                                                    scale_y_continuous(limits=[0, 4], breaks=np.arange(0, 4+0.1, 1.0)) +\
                                                    stat_summary(fun_y=np.mean, geom='point', color='red', size=3, shape=0, stroke=1, show_legend=False) +\
                                                    stat_summary(fun_y=np.mean, geom='point', color='red', size=3, shape=1, stroke=1, show_legend=False) + theme_bw()
p.save(filename='Cdkn1a_exp_dotplot_b.pdf', dpi=600, height=4, width=4)

## epithelial cell of proximal tubule
rna_ec = rna[rna.obs['cell_type']=='epithelial cell of proximal tubule'].copy()
df = pd.DataFrame({'age':rna_ec.obs['age'],
                   'exp':rna_ec.raw.X[:, np.argwhere(rna_ec.raw.var.index=='Cdkn2a').flatten().item()].toarray().flatten()})
df['age'] = pd.Categorical(df['age'], categories=['1m', '3m', '18m', '21m', '30m'])
p = ggplot(df, aes(x='age', y='exp', fill='age')) + geom_jitter(size=1, width=0.2, height=0, show_legend=False) + xlab('Age') + ylab('Cdkn1a expression level') +\
                                                    scale_y_continuous(limits=[0, 4], breaks=np.arange(0, 4+0.1, 1.0)) +\
                                                    stat_summary(fun_y=np.mean, geom='point', color='red', size=3, shape=0, stroke=1, show_legend=False) +\
                                                    stat_summary(fun_y=np.mean, geom='point', color='red', size=3, shape=1, stroke=1, show_legend=False) + theme_bw()
p.save(filename='Cdkn1a_exp_dotplot_ec.pdf', dpi=600, height=4, width=4)

## kidney loop of Henle thick ascending limb epithelial cell
rna_ec = rna[rna.obs['cell_type']=='kidney loop of Henle thick ascending limb epithelial cell'].copy()
df = pd.DataFrame({'age':rna_ec.obs['age'],
                   'exp':rna_ec.raw.X[:, np.argwhere(rna_ec.raw.var.index=='Cdkn1a').flatten().item()].toarray().flatten()})
df['age'] = pd.Categorical(df['age'], categories=['1m', '3m', '18m', '21m', '24m', '30m'])
p = ggplot(df, aes(x='age', y='exp', fill='age')) + geom_jitter(size=1, width=0.2, height=0, show_legend=False) + xlab('Age') + ylab('Cdkn1a expression level') +\
                                                    scale_y_continuous(limits=[0, 4], breaks=np.arange(0, 4+0.1, 1.0)) +\
                                                    stat_summary(fun_y=np.mean, geom='point', color='red', size=3, shape=0, stroke=1, show_legend=False) +\
                                                    stat_summary(fun_y=np.mean, geom='point', color='red', size=3, shape=1, stroke=1, show_legend=False) + theme_bw()
p.save(filename='Cdkn1a_exp_dotplot_kl.pdf', dpi=600, height=4, width=4)
df['age'].value_counts()
# 1m     460
# 3m     310
# 30m    309
# 18m    247
# 21m    228
# 24m      0

# rna_ec = rna[rna.obs['cell_type']=='epithelial cell of proximal tubule'].copy()
# df = pd.DataFrame({'age':rna_ec.obs['age'],
#                    'exp_1':rna_ec.raw.X[:, np.argwhere(rna_ec.raw.var.index=='Cdkn2a').flatten().item()].toarray().flatten(),
#                    'exp_2':rna_ec.raw.X[:, np.argwhere(rna_ec.raw.var.index=='E2f2').flatten().item()].toarray().flatten(),
#                    'exp_3':rna_ec.raw.X[:, np.argwhere(rna_ec.raw.var.index=='Lmnb1').flatten().item()].toarray().flatten(),
#                    'exp_4':rna_ec.raw.X[:, np.argwhere(rna_ec.raw.var.index=='Tnf').flatten().item()].toarray().flatten(),
#                    'exp_5':rna_ec.raw.X[:, np.argwhere(rna_ec.raw.var.index=='Itgax').flatten().item()].toarray().flatten()})
# df['exp'] = (df['exp_1']+df['exp_2']+df['exp_3']+df['exp_4']+df['exp_5'])/5
# df['age'] = pd.Categorical(df['age'], categories=['1m', '3m', '18m', '21m', '30m'])
# p = ggplot(df, aes(x='age', y='exp', fill='age')) + geom_jitter(size=1, width=0.2, height=0, show_legend=False) + xlab('Age') + ylab('Cdkn1a expression level') +\
#                                                     scale_y_continuous(limits=[0, 4], breaks=np.arange(0, 4+0.1, 1.0)) +\
#                                                     stat_summary(fun_y=np.mean, geom='point', color='red', size=3, shape=0, stroke=1, show_legend=False) +\
#                                                     stat_summary(fun_y=np.mean, geom='point', color='red', size=3, shape=1, stroke=1, show_legend=False) + theme_bw()
# p.save(filename='Cdkn1a_exp_dotplot_ec.pdf', dpi=600, height=4, width=4)



#################################### Epitrace ####################################
###### transform h5ad to 10x
import numpy as np
import scanpy as sc
import scipy.io as sio
import pandas as pd
from scipy.sparse import csr_matrix

atac = sc.read_h5ad('atac_predict.h5ad')  # 17517 × 65352

## output peaks
peaks_df = pd.DataFrame({'chr': atac.var['gene_ids'].map(lambda x: x.split(':')[0]).values,
                         'start': atac.var['gene_ids'].map(lambda x: x.split(':')[1].split('-')[0]).values,
                         'end': atac.var['gene_ids'].map(lambda x: x.split(':')[1].split('-')[1]).values})
peaks_df.to_csv('scATAC_peaks.bed', sep='\t', header=False, index=False)

## output barcodes
pd.DataFrame(atac.obs.index).to_csv('scATAC_barcodes.tsv', sep='\t', header=False, index=False)

## output matrix
sio.mmwrite("scATAC_matrix.mtx", atac.X.T)  # peak*cell (not cell*peak)

## output meta
pd.DataFrame(atac.obs[['age', 'cell_type']]).to_csv('scATAC_meta.tsv', sep='\t', header=False, index=True)


###### R code ######
library(readr)
library(EpiTrace)
library(dplyr)
library(GenomicRanges)
library(ggplot2)
library(Seurat)


## load the  “mouse clocks”
# https://github.com/MagpiePKU/EpiTrace/blob/master/data/mouse_clock_mm285_design_clock347_mm10.rds
mouse_clock_by_MM285 <- readRDS('mouse_clock_mm285_design_clock347_mm10.rds')

## prepare the scATAC data
peaks_df <- read_tsv('scATAC_peaks.bed',col_names = c('chr','start','end'))
cells <- read_tsv('scATAC_barcodes.tsv',col_names=c('cell'))
mm <- Matrix::readMM('scATAC_matrix.mtx')
peaks_df$peaks <- paste0(peaks_df$chr, '-', peaks_df$start, '-', peaks_df$end)
init_gr <- EpiTrace::Init_Peakset(peaks_df)
init_mm <- EpiTrace::Init_Matrix(peakname=peaks_df$peaks, cellname=cells$cell, matrix=mm)

## EpiTrace analysis
# trace(EpiTraceAge_Convergence, edit=TRUE)  ref_genome='hg19'???
epitrace_obj_age_conv_estimated_by_mouse_clock <- EpiTraceAge_Convergence(peakSet=init_gr, matrix=init_mm, ref_genome='mm10',
                                                                          clock_gr=mouse_clock_by_MM285, 
                                                                          iterative_time=5, min.cutoff=0, non_standard_clock=TRUE,
                                                                          qualnum=10, ncore_lim=48, mean_error_limit=0.1)
## add meta info of cell types
meta <- as.data.frame(epitrace_obj_age_conv_estimated_by_mouse_clock@meta.data)
write.table(meta, 'epitrace_meta.txt', sep='\t', quote=FALSE)
cell_info <- read.table('scATAC_meta.tsv', sep='\t', col.names=c('cell', 'age', 'cell_type'))
meta_cell_info <- left_join(meta, cell_info) 
epitrace_obj_age_conv_estimated_by_mouse_clock <- AddMetaData(epitrace_obj_age_conv_estimated_by_mouse_clock, metadata=meta_cell_info[['cell_type']], col.name='cell_type')

epitrace_obj_age_conv_estimated_by_mouse_clock[['all']] <- Seurat::CreateAssayObject(counts=init_mm[,c(epitrace_obj_age_conv_estimated_by_mouse_clock$cell)], min.cells=0, min.features=0, check.matrix=F)
DefaultAssay(epitrace_obj_age_conv_estimated_by_mouse_clock) <- 'all'
saveRDS(epitrace_obj_age_conv_estimated_by_mouse_clock, file='epitrace_obj_age_conv_estimated_by_mouse_clock.rds')
epitrace_obj_age_conv_estimated_by_mouse_clock <- readRDS(file='epitrace_obj_age_conv_estimated_by_mouse_clock.rds')


~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
## peak-age association (consider all cell types for comparison)
# asso_res_mouse_clock <- AssociationOfPeaksToAge(epitrace_obj_age_conv_estimated_by_mouse_clock,
#                                                 epitrace_age_name="EpiTraceAge_iterative", parallel=TRUE, peakSetName='all')
# asso_res_mouse_clock[order(asso_res_mouse_clock$correlation_of_EpiTraceAge, decreasing=TRUE), ][1:10, 1:2]
# asso_res_mouse_clock[order(asso_res_mouse_clock$scaled_correlation_of_EpiTraceAge, decreasing=TRUE), ][1:10, c(1,3)]

## plot top 20 up & down peaks
## kidney proximal convoluted tubule epithelial cell
asso_res_mouse_clock_01 <- AssociationOfPeaksToAge(subset(epitrace_obj_age_conv_estimated_by_mouse_clock, cell_type %in% c('kidney proximal convoluted tubule epithelial cell')),
                                                   epitrace_age_name="EpiTraceAge_iterative", parallel=T, peakSetName='all')
up <- asso_res_mouse_clock_01[order(asso_res_mouse_clock_01$scaled_correlation_of_EpiTraceAge, decreasing=TRUE), ][1:20, c(1,3)]
dw <- asso_res_mouse_clock_01[order(asso_res_mouse_clock_01$scaled_correlation_of_EpiTraceAge, decreasing=FALSE), ][1:20, c(1,3)]
dat <- rbind(up, dw)
rownames(dat) <- seq(1, nrow(dat))
colnames(dat) <- c('Peaks', 'Scaled_correlation')
dat <- dat[order(dat$Scaled_correlation, decreasing=FALSE), ]
dat$Peaks <- factor(dat$Peaks, levels=dat$Peaks)
p <- ggplot(dat, aes(x=Peaks, y=Scaled_correlation)) + geom_bar(stat='identity', fill='darkred', position=position_dodge()) + coord_flip() +
                                                       scale_y_continuous(limits=c(-2.5, 2.5), breaks=seq(-2.5, 2.5, 0.5)) + theme_bw() + 
                                                       ggtitle('Kidney proximal convoluted tubule epithelial cells') + theme(plot.title=element_text(hjust=0.5))
ggsave(p, filename='peaks_epitrace_corr_kp.pdf', dpi=600, height=10, width=6)

## annotate peaks
peaks <- as.data.frame(do.call(rbind, strsplit(as.vector(rev(dat$Peaks)), split='-'))) # use rev() to match ranking
colnames(peaks) <- c('chrom', 'start', 'end')
write.table(peaks, 'peaks_epitrace_top20_bot20_kp.bed',
            sep='\t', quote=FALSE, row.names=FALSE, col.names=FALSE)

library(TxDb.Mmusculus.UCSC.mm10.knownGene)
library(ChIPseeker)
peaks <- readPeakFile('peaks_epitrace_top20_bot20_kidney_proximal_convoluted_tubule_epithelial_cell.bed')
peaks_anno <- annotatePeak(peak=peaks, tssRegion=c(-3000, 3000), TxDb=TxDb.Mmusculus.UCSC.mm10.knownGene, annoDb='org.Mm.eg.db')
as.data.frame(peaks_anno)['SYMBOL']

epitrace_obj_age_conv_estimated_by_mouse_clock <- readRDS(file='epitrace_obj_age_conv_estimated_by_mouse_clock.rds')
epitrace_obj_age_conv_estimated_by_mouse_clock@assays$all@data

enriched.motifs <- FindMotifs(object=epitrace_obj_age_conv_estimated_by_mouse_clock,
                              features=rownames(asso_res_mouse_clock_01[order(asso_res_mouse_clock_01$scaled_correlation_of_EpiTraceAge, decreasing=TRUE), ][1:20, ]['locus']))
# Selecting background regions to match input sequence characteristics
# Error in match.arg(arg = layer) : 
#   'arg' should be one of “data”, “scale.data”, “counts”


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
true_motifs = snap.tl.motif_enrichment(motifs=snap.datasets.cis_bp(unique=True),
                                       regions=true_marker_peaks,
                                       genome_fasta=snap.genome.hg38,
                                       background=None,
                                       method='hypergeometric')
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~




# idx <- rownames(asso_res_mouse_clock_01[order(asso_res_mouse_clock_01$scaled_correlation_of_EpiTraceAge, decreasing=TRUE), ][1:100, ])
# peaks <- as.data.frame(do.call(rbind, strsplit(idx, split='-')))
# colnames(peaks) <- c('chrom', 'start', 'end')
# write.table(peaks, 'peaks_epitrace_top100_kidney_proximal_convoluted_tubule_epithelial_cell.bed',
#             sep='\t', quote=FALSE, row.names=FALSE, col.names=FALSE)

# # BiocManager::install("TxDb.Mmusculus.UCSC.mm10.knownGene")
# # BiocManager::install("org.Mm.eg.db")
# library(TxDb.Mmusculus.UCSC.mm10.knownGene)
# library(ChIPseeker)
# peaks <- readPeakFile('peaks_epitrace_top100_kidney_proximal_convoluted_tubule_epithelial_cell.bed')
# peaks_anno <- annotatePeak(peak=peaks, tssRegion=c(-3000, 3000), TxDb=TxDb.Mmusculus.UCSC.mm10.knownGene, annoDb='org.Mm.eg.db')
# as.data.frame(peaks_anno)['SYMBOL']


## B cell
asso_res_mouse_clock_01 <- AssociationOfPeaksToAge(subset(epitrace_obj_age_conv_estimated_by_mouse_clock, cell_type %in% c('B cell')),
                                                   epitrace_age_name="EpiTraceAge_iterative", parallel=T, peakSetName='all')
up <- asso_res_mouse_clock_01[order(asso_res_mouse_clock_01$scaled_correlation_of_EpiTraceAge, decreasing=TRUE), ][1:20, c(1,3)]
dw <- asso_res_mouse_clock_01[order(asso_res_mouse_clock_01$scaled_correlation_of_EpiTraceAge, decreasing=FALSE), ][1:20, c(1,3)]
dat <- rbind(up, dw)
rownames(dat) <- seq(1, nrow(dat))
colnames(dat) <- c('Peaks', 'Scaled_correlation')
dat <- dat[order(dat$Scaled_correlation, decreasing=FALSE), ]
dat$Peaks <- factor(dat$Peaks, levels=dat$Peaks)
p <- ggplot(dat, aes(x=Peaks, y=Scaled_correlation)) + geom_bar(stat='identity', fill='darkred', position=position_dodge()) + coord_flip() +
                                                       scale_y_continuous(limits=c(-2.5, 3.0), breaks=seq(-2.5, 3.0, 0.5)) + theme_bw() + 
                                                       ggtitle('B cells') + theme(plot.title=element_text(hjust=0.5))
ggsave(p, filename='peaks_epitrace_corr_B_cell.pdf', dpi=300, height=10, width=6)

## annotate peaks
peaks <- as.data.frame(do.call(rbind, strsplit(as.vector(rev(dat$Peaks)), split='-')))  # use rev() to match ranking
colnames(peaks) <- c('chrom', 'start', 'end')
write.table(peaks, 'peaks_epitrace_top20_bot20_B_cell.bed',
            sep='\t', quote=FALSE, row.names=FALSE, col.names=FALSE)
peaks <- readPeakFile('peaks_epitrace_top20_bot20_B_cell.bed')
peaks_anno <- annotatePeak(peak=peaks, tssRegion=c(-3000, 3000), TxDb=TxDb.Mmusculus.UCSC.mm10.knownGene, annoDb='org.Mm.eg.db')
as.data.frame(peaks_anno)['SYMBOL']



## epithelial cell of proximal tubule
asso_res_mouse_clock_01 <- AssociationOfPeaksToAge(subset(epitrace_obj_age_conv_estimated_by_mouse_clock, cell_type %in% c('epithelial cell of proximal tubule')),
                                                   epitrace_age_name="EpiTraceAge_iterative", parallel=T, peakSetName='all')
up <- asso_res_mouse_clock_01[order(asso_res_mouse_clock_01$scaled_correlation_of_EpiTraceAge, decreasing=TRUE), ][1:20, c(1,3)]
dw <- asso_res_mouse_clock_01[order(asso_res_mouse_clock_01$scaled_correlation_of_EpiTraceAge, decreasing=FALSE), ][1:20, c(1,3)]
dat <- rbind(up, dw)
rownames(dat) <- seq(1, nrow(dat))
colnames(dat) <- c('Peaks', 'Scaled_correlation')
dat <- dat[order(dat$Scaled_correlation, decreasing=FALSE), ]
dat$Peaks <- factor(dat$Peaks, levels=dat$Peaks)
p <- ggplot(dat, aes(x=Peaks, y=Scaled_correlation)) + geom_bar(stat='identity', fill='darkred', position=position_dodge()) + coord_flip() +
                                                       scale_y_continuous(limits=c(-2.5, 2.5), breaks=seq(-2.5, 2.5, 0.5)) + theme_bw() + 
                                                       ggtitle('Epithelial cell of proximal tubule') + theme(plot.title=element_text(hjust=0.5))
ggsave(p, filename='peaks_epitrace_corr_epithelial_cell_of_proximal_tubule.pdf', dpi=300, height=10, width=6)

## annotate peaks
peaks <- as.data.frame(do.call(rbind, strsplit(as.vector(rev(dat$Peaks)), split='-')))  # use rev() to match ranking
colnames(peaks) <- c('chrom', 'start', 'end')
write.table(peaks, 'peaks_epitrace_top20_bot20_epithelial_cell_of_proximal_tubule.bed',
            sep='\t', quote=FALSE, row.names=FALSE, col.names=FALSE)
peaks <- readPeakFile('peaks_epitrace_top20_bot20_epithelial_cell_of_proximal_tubule.bed')
peaks_anno <- annotatePeak(peak=peaks, tssRegion=c(-3000, 3000), TxDb=TxDb.Mmusculus.UCSC.mm10.knownGene, annoDb='org.Mm.eg.db')
as.data.frame(peaks_anno)['SYMBOL']



## T cell
asso_res_mouse_clock_01 <- AssociationOfPeaksToAge(subset(epitrace_obj_age_conv_estimated_by_mouse_clock, cell_type %in% c('T cell')),
                                                   epitrace_age_name="EpiTraceAge_iterative", parallel=T, peakSetName='all')
up <- asso_res_mouse_clock_01[order(asso_res_mouse_clock_01$scaled_correlation_of_EpiTraceAge, decreasing=TRUE), ][1:20, c(1,3)]
dw <- asso_res_mouse_clock_01[order(asso_res_mouse_clock_01$scaled_correlation_of_EpiTraceAge, decreasing=FALSE), ][1:20, c(1,3)]
dat <- rbind(up, dw)
rownames(dat) <- seq(1, nrow(dat))
colnames(dat) <- c('Peaks', 'Scaled_correlation')
dat <- dat[order(dat$Scaled_correlation, decreasing=FALSE), ]
dat$Peaks <- factor(dat$Peaks, levels=dat$Peaks)
p <- ggplot(dat, aes(x=Peaks, y=Scaled_correlation)) + geom_bar(stat='identity', fill='darkred', position=position_dodge()) + coord_flip() +
                                                       scale_y_continuous(limits=c(-2.5, 3.0), breaks=seq(-2.5, 3.0, 0.5)) + theme_bw() + 
                                                       ggtitle('T cells') + theme(plot.title=element_text(hjust=0.5))
ggsave(p, filename='peaks_epitrace_corr_T_cell.pdf', dpi=300, height=10, width=6)

## annotate peaks
peaks <- as.data.frame(do.call(rbind, strsplit(as.vector(rev(dat$Peaks)), split='-')))  # use rev() to match ranking
colnames(peaks) <- c('chrom', 'start', 'end')
write.table(peaks, 'peaks_epitrace_top20_bot20_T_cell.bed',
            sep='\t', quote=FALSE, row.names=FALSE, col.names=FALSE)
peaks <- readPeakFile('peaks_epitrace_top20_bot20_T_cell.bed')
peaks_anno <- annotatePeak(peak=peaks, tssRegion=c(-3000, 3000), TxDb=TxDb.Mmusculus.UCSC.mm10.knownGene, annoDb='org.Mm.eg.db')
as.data.frame(peaks_anno)['SYMBOL']

## macrophage
asso_res_mouse_clock_01 <- AssociationOfPeaksToAge(subset(epitrace_obj_age_conv_estimated_by_mouse_clock, cell_type %in% c('macrophage')),
                                                   epitrace_age_name="EpiTraceAge_iterative", parallel=T, peakSetName='all')
up <- asso_res_mouse_clock_01[order(asso_res_mouse_clock_01$scaled_correlation_of_EpiTraceAge, decreasing=TRUE), ][1:20, c(1,3)]
dw <- asso_res_mouse_clock_01[order(asso_res_mouse_clock_01$scaled_correlation_of_EpiTraceAge, decreasing=FALSE), ][1:20, c(1,3)]
dat <- rbind(up, dw)
rownames(dat) <- seq(1, nrow(dat))
colnames(dat) <- c('Peaks', 'Scaled_correlation')
dat <- dat[order(dat$Scaled_correlation, decreasing=FALSE), ]
dat$Peaks <- factor(dat$Peaks, levels=dat$Peaks)
p <- ggplot(dat, aes(x=Peaks, y=Scaled_correlation)) + geom_bar(stat='identity', fill='darkred', position=position_dodge()) + coord_flip() +
                                                       scale_y_continuous(limits=c(-2.5, 3.0), breaks=seq(-2.5, 3.0, 0.5)) + theme_bw() + 
                                                       ggtitle('Macrophages') + theme(plot.title=element_text(hjust=0.5))
ggsave(p, filename='peaks_epitrace_corr_macrophage.pdf', dpi=300, height=10, width=6)

## annotate peaks
peaks <- as.data.frame(do.call(rbind, strsplit(as.vector(rev(dat$Peaks)), split='-')))  # use rev() to match ranking
colnames(peaks) <- c('chrom', 'start', 'end')
write.table(peaks, 'peaks_epitrace_top20_bot20_macrophage.bed',
            sep='\t', quote=FALSE, row.names=FALSE, col.names=FALSE)
peaks <- readPeakFile('peaks_epitrace_top20_bot20_macrophage.bed')
peaks_anno <- annotatePeak(peak=peaks, tssRegion=c(-3000, 3000), TxDb=TxDb.Mmusculus.UCSC.mm10.knownGene, annoDb='org.Mm.eg.db')
as.data.frame(peaks_anno)['SYMBOL']

## kidney loop of Henle thick ascending limb epithelial cell
asso_res_mouse_clock_01 <- AssociationOfPeaksToAge(subset(epitrace_obj_age_conv_estimated_by_mouse_clock, cell_type %in% c('kidney loop of Henle thick ascending limb epithelial cell')),
                                                   epitrace_age_name="EpiTraceAge_iterative", parallel=T, peakSetName='all')
up <- asso_res_mouse_clock_01[order(asso_res_mouse_clock_01$scaled_correlation_of_EpiTraceAge, decreasing=TRUE), ][1:20, c(1,3)]
dw <- asso_res_mouse_clock_01[order(asso_res_mouse_clock_01$scaled_correlation_of_EpiTraceAge, decreasing=FALSE), ][1:20, c(1,3)]
dat <- rbind(up, dw)
rownames(dat) <- seq(1, nrow(dat))
colnames(dat) <- c('Peaks', 'Scaled_correlation')
dat <- dat[order(dat$Scaled_correlation, decreasing=FALSE), ]
dat$Peaks <- factor(dat$Peaks, levels=dat$Peaks)
p <- ggplot(dat, aes(x=Peaks, y=Scaled_correlation)) + geom_bar(stat='identity', fill='darkred', position=position_dodge()) + coord_flip() +
                                                       scale_y_continuous(limits=c(-2.0, 3.0), breaks=seq(-2.0, 3.0, 0.5)) + theme_bw() + 
                                                       ggtitle('Kidney loop of Henle thick ascending limb epithelial cell') + theme(plot.title=element_text(hjust=0.5))
ggsave(p, filename='peaks_epitrace_corr_kidney_loop_of_Henle_thick_ascending_limb_epithelial_cell.pdf', dpi=300, height=10, width=6)

## annotate peaks
peaks <- as.data.frame(do.call(rbind, strsplit(as.vector(rev(dat$Peaks)), split='-')))  # use rev() to match ranking
colnames(peaks) <- c('chrom', 'start', 'end')
write.table(peaks, 'peaks_epitrace_top20_bot20_kidney_loop_of_Henle_thick_ascending_limb_epithelial_cell.bed',
            sep='\t', quote=FALSE, row.names=FALSE, col.names=FALSE)
peaks <- readPeakFile('peaks_epitrace_top20_bot20_kidney_loop_of_Henle_thick_ascending_limb_epithelial_cell.bed')
peaks_anno <- annotatePeak(peak=peaks, tssRegion=c(-3000, 3000), TxDb=TxDb.Mmusculus.UCSC.mm10.knownGene, annoDb='org.Mm.eg.db')
as.data.frame(peaks_anno)['SYMBOL']
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


#### plot for each age
from plotnine import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['pdf.fonttype'] = 42

meta = pd.read_table('epitrace_meta.txt', sep='\t')
cell_info = pd.read_table('scATAC_meta.tsv', sep='\t', names=['cell', 'age', 'cell_type'])
meta_cell_info = pd.merge(meta, cell_info) 

## epithelial cell
df = meta_cell_info[meta_cell_info['cell_type'].isin(['kidney proximal convoluted tubule epithelial cell',
                                                      'epithelial cell of proximal tubule',
                                                      'kidney loop of Henle thick ascending limb epithelial cell',
                                                      'kidney collecting duct principal cell',
                                                      'kidney distal convoluted tubule epithelial cell'])][['EpiTraceAge_iterative', 'age']]
df.rename(columns={'EpiTraceAge_iterative': 'epitrace_age'}, inplace=True)
df['age'] = pd.Categorical(df['age'], categories=['1m', '3m', '18m', '21m', '24m', '30m'])
p = ggplot(df, aes(x='age', y='epitrace_age', fill='age')) + geom_jitter(size=1, width=0.2, height=0, show_legend=False) + xlab('Age') + ylab('epitrace_age') +\
                                                             scale_y_continuous(limits=[0, 1], breaks=np.arange(0, 1+0.1, 0.2)) +\
                                                             stat_summary(fun_y=np.mean, geom='point', color='red', size=3, shape=0, stroke=1, show_legend=False) +\
                                                             stat_summary(fun_y=np.mean, geom='point', color='red', size=3, shape=1, stroke=1, show_legend=False) + theme_bw()
p.save(filename='epitrace_age_dotplot_epi.pdf', dpi=600, height=4, width=4)
df['age'].value_counts()
# 30m    3929
# 18m    2092
# 1m     1750
# 3m     1424
# 21m    1405
# 24m       0

## fenestrated cell
df = meta_cell_info[meta_cell_info['cell_type'].isin(['fenestrated cell'])][['EpiTraceAge_iterative', 'age']]
df.rename(columns={'EpiTraceAge_iterative': 'epitrace_age'}, inplace=True)
df['age'] = pd.Categorical(df['age'], categories=['1m', '3m', '18m', '21m', '24m', '30m'])
p = ggplot(df, aes(x='age', y='epitrace_age', fill='age')) + geom_jitter(size=1, width=0.2, height=0, show_legend=False) + xlab('Age') + ylab('epitrace_age') +\
                                                             scale_y_continuous(limits=[0, 1], breaks=np.arange(0, 1+0.1, 0.2)) +\
                                                             stat_summary(fun_y=np.mean, geom='point', color='red', size=3, shape=0, stroke=1, show_legend=False) +\
                                                             stat_summary(fun_y=np.mean, geom='point', color='red', size=3, shape=1, stroke=1, show_legend=False) + theme_bw()
p.save(filename='epitrace_age_dotplot_fc.pdf', dpi=600, height=4, width=4)
df['age'].value_counts()
# 30m    3929
# 18m    2092
# 1m     1750
# 3m     1424
# 21m    1405
# 24m       0

## immune cell
df = meta_cell_info[meta_cell_info['cell_type'].isin(['macrophage', 'T cell', 'B cell'])][['EpiTraceAge_iterative', 'age']]
df.rename(columns={'EpiTraceAge_iterative': 'epitrace_age'}, inplace=True)
df['age'] = pd.Categorical(df['age'], categories=['1m', '3m', '18m', '21m', '24m', '30m'])
p = ggplot(df, aes(x='age', y='epitrace_age', fill='age')) + geom_jitter(size=1, width=0.2, height=0, show_legend=False) + xlab('Age') + ylab('epitrace_age') +\
                                                             scale_y_continuous(limits=[0, 1], breaks=np.arange(0, 1+0.1, 0.2)) +\
                                                             stat_summary(fun_y=np.mean, geom='point', color='red', size=3, shape=0, stroke=1, show_legend=False) +\
                                                             stat_summary(fun_y=np.mean, geom='point', color='red', size=3, shape=1, stroke=1, show_legend=False) + theme_bw()
p.save(filename='epitrace_age_dotplot_ic.pdf', dpi=600, height=4, width=4)
df['age'].value_counts()
# 30m    3929
# 18m    2092
# 1m     1750
# 3m     1424
# 21m    1405
# 24m       0

## kidney proximal convoluted tubule epithelial cell
df = meta_cell_info[meta_cell_info['cell_type']=='kidney proximal convoluted tubule epithelial cell'][['EpiTraceAge_iterative', 'age']]
df.rename(columns={'EpiTraceAge_iterative': 'epitrace_age'}, inplace=True)
df['age'] = pd.Categorical(df['age'], categories=['1m', '3m', '18m', '21m', '30m'])
p = ggplot(df, aes(x='age', y='epitrace_age', fill='age')) + geom_jitter(size=1, width=0.2, height=0, show_legend=False) + xlab('Age') + ylab('epitrace_age') +\
                                                             scale_y_continuous(limits=[0, 1], breaks=np.arange(0, 1+0.1, 0.2)) +\
                                                             stat_summary(fun_y=np.mean, geom='point', color='red', size=3, shape=0, stroke=1, show_legend=False) +\
                                                             stat_summary(fun_y=np.mean, geom='point', color='red', size=3, shape=1, stroke=1, show_legend=False) + theme_bw()
p.save(filename='epitrace_age_dotplot_kp.pdf', dpi=600, height=4, width=4)

## B cell
df = meta_cell_info[meta_cell_info['cell_type']=='B cell'][['EpiTraceAge_iterative', 'age']]
df.rename(columns={'EpiTraceAge_iterative': 'epitrace_age'}, inplace=True)
df['age'].value_counts()
# 24m    2922
# 30m      81
# 18m      51
# 21m      51
# 1m       10
# 3m        9
# df['age'] = df['age'].replace({'1m':'1,3m', '3m':'1,3m', '18m':'18,21m', '21m':'18,21m', '24m':'24m', '30m':'30m'})
# df['age'].value_counts()
# # 24m       2922
# # 18,21m     102
# # 30m         81
# # 1,3m        19
df['age'] = pd.Categorical(df['age'], categories=['1m', '3m', '18m', '21m', '24m', '30m'])
p = ggplot(df, aes(x='age', y='epitrace_age', fill='age')) + geom_jitter(size=1, width=0.2, height=0, show_legend=False) + xlab('Age') + ylab('epitrace_age') +\
                                                             scale_y_continuous(limits=[0, 1], breaks=np.arange(0, 1+0.1, 0.2)) +\
                                                             stat_summary(fun_y=np.mean, geom='point', color='red', size=3, shape=0, stroke=1, show_legend=False) +\
                                                             stat_summary(fun_y=np.mean, geom='point', color='red', size=3, shape=1, stroke=1, show_legend=False) + theme_bw()
p.save(filename='epitrace_age_dotplot_b.pdf', dpi=600, height=4, width=4)

## epithelial cell of proximal tubule
df = meta_cell_info[meta_cell_info['cell_type']=='epithelial cell of proximal tubule'][['EpiTraceAge_iterative', 'age']]
df.rename(columns={'EpiTraceAge_iterative': 'epitrace_age'}, inplace=True)
df['age'].value_counts()
# 30m    2211
# 18m     400
# 3m      195
# 1m      169
# 21m      78
# df['age'] = df['age'].replace({'1m':'1,3m', '3m':'1,3m', '18m':'18m', '21m':'21m', '30m':'30m'})
# df['age'] = pd.Categorical(df['age'], categories=['1,3m', '18m', '21m', '30m'])
df['age'] = pd.Categorical(df['age'], categories=['1m', '3m', '18m', '21m', '30m'])
p = ggplot(df, aes(x='age', y='epitrace_age', fill='age')) + geom_jitter(size=1, width=0.2, height=0, show_legend=False) + xlab('Age') + ylab('epitrace_age') +\
                                                             scale_y_continuous(limits=[0, 1], breaks=np.arange(0, 1+0.1, 0.2)) +\
                                                             stat_summary(fun_y=np.mean, geom='point', color='red', size=3, shape=0, stroke=1, show_legend=False) +\
                                                             stat_summary(fun_y=np.mean, geom='point', color='red', size=3, shape=1, stroke=1, show_legend=False) + theme_bw()
p.save(filename='epitrace_age_dotplot_ec.pdf', dpi=600, height=4, width=4)

## kidney loop of Henle thick ascending limb epithelial cell
df = meta_cell_info[meta_cell_info['cell_type']=='kidney loop of Henle thick ascending limb epithelial cell'][['EpiTraceAge_iterative', 'age']]
df.rename(columns={'EpiTraceAge_iterative': 'epitrace_age'}, inplace=True)
df['age'].value_counts()
# 1m     460
# 3m     310
# 30m    309
# 18m    247
# 21m    228
df['age'] = pd.Categorical(df['age'], categories=['1m', '3m', '18m', '21m', '30m'])
p = ggplot(df, aes(x='age', y='epitrace_age', fill='age')) + geom_jitter(size=1, width=0.2, height=0, show_legend=False) + xlab('Age') + ylab('epitrace_age') +\
                                                             scale_y_continuous(limits=[0, 1], breaks=np.arange(0, 1+0.1, 0.2)) +\
                                                             stat_summary(fun_y=np.mean, geom='point', color='red', size=3, shape=0, stroke=1, show_legend=False) +\
                                                             stat_summary(fun_y=np.mean, geom='point', color='red', size=3, shape=1, stroke=1, show_legend=False) + theme_bw()
p.save(filename='epitrace_age_dotplot_kl.pdf', dpi=600, height=4, width=4)

## macrophage
df = meta_cell_info[meta_cell_info['cell_type']=='macrophage'][['EpiTraceAge_iterative', 'age']]
df.rename(columns={'EpiTraceAge_iterative': 'epitrace_age'}, inplace=True)
df['age'].value_counts()
# 30m    553
# 24m    284
# 18m    264
# 3m     139
# 21m    105
# 1m      62
df['age'] = pd.Categorical(df['age'], categories=['1m', '3m', '18m', '21m', '24m', '30m'])
p = ggplot(df, aes(x='age', y='epitrace_age', fill='age')) + geom_jitter(size=1, width=0.2, height=0, show_legend=False) + xlab('Age') + ylab('epitrace_age') +\
                                                             scale_y_continuous(limits=[0, 1], breaks=np.arange(0, 1+0.1, 0.2)) +\
                                                             stat_summary(fun_y=np.mean, geom='point', color='red', size=3, shape=0, stroke=1, show_legend=False) +\
                                                             stat_summary(fun_y=np.mean, geom='point', color='red', size=3, shape=1, stroke=1, show_legend=False) + theme_bw()
p.save(filename='epitrace_age_dotplot_macro.pdf', dpi=600, height=4, width=4)

## T cell
df = meta_cell_info[meta_cell_info['cell_type']=='T cell'][['EpiTraceAge_iterative', 'age']]
df.rename(columns={'EpiTraceAge_iterative': 'epitrace_age'}, inplace=True)
df['age'].value_counts()
# 24m    987
# 30m    227
# 18m     65
# 21m     49
# 3m      18
# 1m      13
df['age'] = pd.Categorical(df['age'], categories=['1m', '3m', '18m', '21m', '24m', '30m'])
p = ggplot(df, aes(x='age', y='epitrace_age', fill='age')) + geom_jitter(size=1, width=0.2, height=0, show_legend=False) + xlab('Age') + ylab('epitrace_age') +\
                                                             scale_y_continuous(limits=[0, 1], breaks=np.arange(0, 1+0.1, 0.2)) +\
                                                             stat_summary(fun_y=np.mean, geom='point', color='red', size=3, shape=0, stroke=1, show_legend=False) +\
                                                             stat_summary(fun_y=np.mean, geom='point', color='red', size=3, shape=1, stroke=1, show_legend=False) + theme_bw()
p.save(filename='epitrace_age_dotplot_t.pdf', dpi=600, height=4, width=4)

## fenestrated cell
df = meta_cell_info[meta_cell_info['cell_type']=='fenestrated cell'][['EpiTraceAge_iterative', 'age']]
df.rename(columns={'EpiTraceAge_iterative': 'epitrace_age'}, inplace=True)
df['age'].value_counts()
# 30m    281
# 1m     206
# 18m    190
# 3m     181
# 21m    169
df['age'] = pd.Categorical(df['age'], categories=['1m', '3m', '18m', '21m', '30m'])
p = ggplot(df, aes(x='age', y='epitrace_age', fill='age')) + geom_jitter(size=1, width=0.2, height=0, show_legend=False) + xlab('Age') + ylab('epitrace_age') +\
                                                             scale_y_continuous(limits=[0, 1], breaks=np.arange(0, 1+0.1, 0.2)) +\
                                                             stat_summary(fun_y=np.mean, geom='point', color='red', size=3, shape=0, stroke=1, show_legend=False) +\
                                                             stat_summary(fun_y=np.mean, geom='point', color='red', size=3, shape=1, stroke=1, show_legend=False) + theme_bw()
p.save(filename='epitrace_age_dotplot_fc.pdf', dpi=600, height=4, width=4)

## kidney collecting duct principal cell
df = meta_cell_info[meta_cell_info['cell_type']=='kidney collecting duct principal cell'][['EpiTraceAge_iterative', 'age']]
df.rename(columns={'EpiTraceAge_iterative': 'epitrace_age'}, inplace=True)
df['age'].value_counts()
# 30m    398
# 18m    130
# 3m     113
# 1m      83
# 21m     65
df['age'] = pd.Categorical(df['age'], categories=['1m', '3m', '18m', '21m', '30m'])
p = ggplot(df, aes(x='age', y='epitrace_age', fill='age')) + geom_jitter(size=1, width=0.2, height=0, show_legend=False) + xlab('Age') + ylab('epitrace_age') +\
                                                             scale_y_continuous(limits=[0, 1], breaks=np.arange(0, 1+0.1, 0.2)) +\
                                                             stat_summary(fun_y=np.mean, geom='point', color='red', size=3, shape=0, stroke=1, show_legend=False) +\
                                                             stat_summary(fun_y=np.mean, geom='point', color='red', size=3, shape=1, stroke=1, show_legend=False) + theme_bw()
p.save(filename='epitrace_age_dotplot_kc.pdf', dpi=600, height=4, width=4)

## kidney distal convoluted tubule epithelial cell
df = meta_cell_info[meta_cell_info['cell_type']=='kidney distal convoluted tubule epithelial cell'][['EpiTraceAge_iterative', 'age']]
df.rename(columns={'EpiTraceAge_iterative': 'epitrace_age'}, inplace=True)
df['age'].value_counts()
# 18m    198
# 21m    174
# 30m    147
# 3m     125
# 1m     100
df['age'] = pd.Categorical(df['age'], categories=['1m', '3m', '18m', '21m', '30m'])
p = ggplot(df, aes(x='age', y='epitrace_age', fill='age')) + geom_jitter(size=1, width=0.2, height=0, show_legend=False) + xlab('Age') + ylab('epitrace_age') +\
                                                             scale_y_continuous(limits=[0, 1], breaks=np.arange(0, 1+0.1, 0.2)) +\
                                                             stat_summary(fun_y=np.mean, geom='point', color='red', size=3, shape=0, stroke=1, show_legend=False) +\
                                                             stat_summary(fun_y=np.mean, geom='point', color='red', size=3, shape=1, stroke=1, show_legend=False) + theme_bw()
p.save(filename='epitrace_age_dotplot_kd.pdf', dpi=600, height=4, width=4)

# rna_ec = rna[rna.obs['cell_type']=='epithelial cell of proximal tubule'].copy()
# df_atac = meta_cell_info[meta_cell_info['cell_type']=='epithelial cell of proximal tubule'][['EpiTraceAge_iterative', 'age']]
# df_atac.rename(columns={'EpiTraceAge_iterative': 'epitrace_age'}, inplace=True)
# df = pd.DataFrame({'exp':rna_ec.raw.X[:, np.argwhere(rna_ec.raw.var.index=='Cdkn1a').flatten().item()].toarray().flatten(),
#                    'epi':df_atac['epitrace_age']})
# df = df[df['exp']!=0]
# from scipy.stats import pearsonr
# pearsonr(df['exp'], df['epi'])


## plot expression level of certain genes for epitrace
import scanpy as sc
from plotnine import *
import pandas as pd
import numpy as np

rna = sc.read_h5ad('../rna_unpaired.h5ad')
rna.layers['counts'] = rna.X.copy()
sc.pp.normalize_total(rna)
sc.pp.log1p(rna)

meta = pd.read_table('epitrace_meta.txt')
rna_meta = rna[rna.obs.index.isin(meta.index)]

dat = rna_meta[rna_meta.obs['cell_type']=='kidney proximal convoluted tubule epithelial cell']

df = pd.DataFrame({'age': dat.obs['age'], 'exp': dat[:, dat.var.index=='Cdkn1a'].X.toarray().flatten()})

# df = pd.DataFrame({'age': dat.obs['age'], 'exp': dat[:, dat.var.index=='Pantr2'].X.toarray().flatten()})
# [df[df['age']=='1m']['exp'].mean(), df[df['age']=='3m']['exp'].mean(), df[df['age']=='18m']['exp'].mean(), df[df['age']=='21m']['exp'].mean(), df[df['age']=='30m']['exp'].mean()]

df['age'] = pd.Categorical(df['age'], categories=['1m', '3m', '18m', '21m', '30m'])
p = ggplot(df, aes(x='age', y='exp', fill='age')) + geom_jitter(width=0.2, height=0, show_legend=False) + xlab('Age') + ylab('Cdkn1a expression level') +\
                                                    scale_y_continuous(limits=[0, 3], breaks=np.arange(0, 3+0.1, 0.5)) +\
                                                    stat_summary(fun_y=np.mean, geom='point', color='red', size=3, shape=0, stroke=1, show_legend=False) +\
                                                    stat_summary(fun_y=np.mean, geom='point', color='red', size=3, shape=1, stroke=1, show_legend=False) + theme_bw()
p.save(filename='gene_exp_cdkn1a_kidney_proximal_convoluted_tubule_epithelial_cell.pdf', dpi=300, height=4, width=4)

df = pd.DataFrame({'age': dat.obs['age'], 'exp': dat[:, dat.var.index=='Kcns3'].X.toarray().flatten()})
df['age'] = pd.Categorical(df['age'], categories=['1m', '3m', '18m', '21m', '30m'])
p = ggplot(df, aes(x='age', y='exp', fill='age')) + geom_jitter(width=0.2, height=0, show_legend=False) + xlab('Age') + ylab('Kcns3 expression level') +\
                                                    scale_y_continuous(limits=[0, 3], breaks=np.arange(0, 3+0.1, 0.5)) +\
                                                    stat_summary(fun_y=np.mean, geom='point', color='red', size=3, shape=0, stroke=1, show_legend=False) +\
                                                    stat_summary(fun_y=np.mean, geom='point', color='red', size=3, shape=1, stroke=1, show_legend=False) + theme_bw()
p.save(filename='gene_exp_orc1_kidney_proximal_convoluted_tubule_epithelial_cell.pdf', dpi=300, height=4, width=4)




