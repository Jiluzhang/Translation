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

######## BABEL
cp ../rna.h5ad rna_train_val.h5ad
cp ../atac.h5ad atac_train_val.h5ad
python h5ad2h5.py -n train_val   # train + val -> train_val

# /fs/home/jiluzhang/BABEL/babel/sc_data_loaders.py
# HG38_GTF -> MM10_GTF (line 174)
nohup python /fs/home/jiluzhang/BABEL/bin/train_model.py --data train_val.h5 --outdir train_out --batchsize 512 --earlystop 10 --device 7 --nofilter > train_20250420.log &  # 2132323

cp ../rna_unpaired.h5ad rna_test.h5ad
cp ../atac_unpaired.h5ad atac_test.h5ad

# ## reset obs index
# import scanpy as sc

# rna = sc.read_h5ad('rna_test.h5ad')
# rna.obs.index = rna.obs.index.values
# rna.write('rna_test.h5ad')

# atac = sc.read_h5ad('atac_test.h5ad')
# atac.obs.index = atac.obs.index.values
# atac.write('atac_test.h5ad')

python h5ad2h5.py -n test
nohup python /fs/home/jiluzhang/BABEL/bin/predict_model.py --checkpoint ./train_out --data test.h5 --outdir test_out --device 7 --nofilter --noplot --transonly > predict_20240420.log &  # 2224825
python match_cell.py

## plot umap for atac & calculate metrics
import scanpy as sc
import numpy as np
from scipy.sparse import csr_matrix
import snapatac2 as snap
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score, normalized_mutual_info_score, homogeneity_score

plt.rcParams['pdf.fonttype'] = 42

atac = sc.read_h5ad('atac_babel.h5ad')
m = atac.X.toarray()
m = ((m.T>m.T.mean(axis=0)).T) & (m>m.mean(axis=0)).astype(int)
atac.X = csr_matrix(m)

true = sc.read_h5ad('atac_test.h5ad')
atac.obs['cell_type'] = true.obs['cell_type']

atac = atac[atac.obs['cell_type'].isin(['kidney proximal convoluted tubule epithelial cell',
                                        'B cell',
                                        'epithelial cell of proximal tubule',
                                        'kidney loop of Henle thick ascending limb epithelial cell',
                                        'macrophage', 
                                        'T cell',
                                        'fenestrated cell',
                                        'kidney collecting duct principal cell',
                                        'kidney distal convoluted tubule epithelial cell'])].copy()  
# 17517 × 63910 (delete too common cell annotation & too small cell number)

snap.pp.select_features(atac)
snap.tl.spectral(atac)  # snap.tl.spectral(atac, n_comps=100, weighted_by_sd=False)
snap.tl.umap(atac)

atac.write('atac_babel_binary.h5ad')

sc.pl.umap(atac, color='cell_type', legend_fontsize='7', legend_loc='right margin', size=5,
           title='', frameon=True, save='_babel_atac.pdf')

AMI_lst = []
ARI_lst = []
HOM_lst = []
NMI_lst = []

for _ in range(10):
    snap.pp.knn(atac)
    snap.tl.leiden(atac)
    AMI_lst.append(adjusted_mutual_info_score(atac.obs['cell_type'], atac.obs['leiden']))
    ARI_lst.append(adjusted_rand_score(atac.obs['cell_type'], atac.obs['leiden']))
    HOM_lst.append(homogeneity_score(atac.obs['cell_type'], atac.obs['leiden']))
    NMI_lst.append(normalized_mutual_info_score(atac.obs['cell_type'], atac.obs['leiden']))

np.mean(AMI_lst)  # 0.534028949324214
np.mean(ARI_lst)  # 0.24628838812540313
np.mean(HOM_lst)  # 0.6442652309203132
np.mean(NMI_lst)  # 0.5347980413779834


#### scbutterfly
## nohup python scbt_b.py > scbt_b_20250420.log &  # 2235081
import os
from scButterfly.butterfly import Butterfly
from scButterfly.split_datasets import *
import scanpy as sc
import anndata as ad
import random

os.environ["CUDA_VISIBLE_DEVICES"] = "5"

RNA_data = sc.read_h5ad('rna.h5ad')
ATAC_data = sc.read_h5ad('atac.h5ad')

random.seed(0)
idx = list(range(RNA_data.n_obs))
random.shuffle(idx)
train_id = idx[:int(len(idx)*0.9)]
validation_id = idx[int(len(idx)*0.9):]
test_id = validation_id

butterfly = Butterfly()
butterfly.load_data(RNA_data, ATAC_data, train_id, test_id, validation_id)
butterfly.data_preprocessing(normalize_total=False, log1p=False, use_hvg=False, n_top_genes=None, binary_data=False, 
                             filter_features=False, fpeaks=None, tfidf=False, normalize=False) 

butterfly.ATAC_data_p.var['chrom'] = butterfly.ATAC_data_p.var['gene_ids'].map(lambda x: x.split(':')[0])
chrom_list = []
last_one = ''
for i in range(len(butterfly.ATAC_data_p.var.chrom)):
    temp = butterfly.ATAC_data_p.var.chrom[i]
    if temp[0 : 3] == 'chr':
        if not temp == last_one:
            chrom_list.append(1)
            last_one = temp
        else:
            chrom_list[-1] += 1
    else:
        chrom_list[-1] += 1

butterfly.augmentation(aug_type=None)
butterfly.construct_model(chrom_list=chrom_list)
butterfly.train_model(batch_size=16)
A2R_predict, R2A_predict = butterfly.test_model(test_cluster=False, test_figure=False, output_data=True)

# rm -r predict  # ensure output unpaired prediction
# nohup python scbt_b_predict.py > predict_20250420.log &  # 2539530
import os
from scButterfly.butterfly import Butterfly
from scButterfly.split_datasets import *
import scanpy as sc
import anndata as ad

os.environ["CUDA_VISIBLE_DEVICES"] = "5"

RNA_data = sc.read_h5ad('rna_test.h5ad')
ATAC_data = sc.read_h5ad('atac_test.h5ad')

train_id = list(range(RNA_data.n_obs))[:10]       # pseudo
validation_id = list(range(RNA_data.n_obs))[:10]  # pseudo
test_id = list(range(RNA_data.n_obs))             # true

butterfly = Butterfly()
butterfly.load_data(RNA_data, ATAC_data, train_id, test_id, validation_id)
butterfly.data_preprocessing(normalize_total=False, log1p=False, use_hvg=False, n_top_genes=None, binary_data=False, 
                             filter_features=False, fpeaks=None, tfidf=False, normalize=False) 

butterfly.ATAC_data_p.var['chrom'] = butterfly.ATAC_data_p.var['gene_ids'].map(lambda x: x.split(':')[0])
chrom_list = []
last_one = ''
for i in range(len(butterfly.ATAC_data_p.var.chrom)):
    temp = butterfly.ATAC_data_p.var.chrom[i]
    if temp[0 : 3] == 'chr':
        if not temp == last_one:
            chrom_list.append(1)
            last_one = temp
        else:
            chrom_list[-1] += 1
    else:
        chrom_list[-1] += 1

butterfly.augmentation(aug_type=None)
butterfly.construct_model(chrom_list=chrom_list)
# butterfly.train_model(batch_size=16)
A2R_predict, R2A_predict = butterfly.test_model(load_model=True, model_path='/fs/home/jiluzhang/Nature_methods/Figure_3/aging_kidney_2/scbt',
                                                test_cluster=False, test_figure=False, output_data=True)

cp predict/R2A.h5ad atac_scbt.h5ad

## plot umap for atac & calculate metrics
import scanpy as sc
import numpy as np
from scipy.sparse import csr_matrix
import snapatac2 as snap
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score, normalized_mutual_info_score, homogeneity_score

plt.rcParams['pdf.fonttype'] = 42

atac = sc.read_h5ad('atac_scbt.h5ad')
m = atac.X.toarray()
m = ((m.T>m.T.mean(axis=0)).T) & (m>m.mean(axis=0)).astype(int)
atac.X = csr_matrix(m)

# atac = sc.read_h5ad('atac_scbt.h5ad')
# m = atac.X.toarray()
# m[m>0.5] = 1
# m[m<=0.5] = 0
# atac.X = csr_matrix(m)

true = sc.read_h5ad('atac_test.h5ad')
atac.obs['cell_type'] = true.obs['cell_type']

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

snap.pp.select_features(atac)  # n_features=20000
snap.tl.spectral(atac)  # snap.tl.spectral(atac, n_comps=100, weighted_by_sd=False)
snap.tl.umap(atac)

atac.write('atac_scbt_binary.h5ad')

sc.pl.umap(atac, color='cell_type', legend_fontsize='7', legend_loc='right margin', size=5,
           title='', frameon=True, save='_scbt_atac.pdf')

AMI_lst = []
ARI_lst = []
HOM_lst = []
NMI_lst = []

for _ in range(10):
    snap.pp.knn(atac)
    snap.tl.leiden(atac)
    AMI_lst.append(adjusted_mutual_info_score(atac.obs['cell_type'], atac.obs['leiden']))
    ARI_lst.append(adjusted_rand_score(atac.obs['cell_type'], atac.obs['leiden']))
    HOM_lst.append(homogeneity_score(atac.obs['cell_type'], atac.obs['leiden']))
    NMI_lst.append(normalized_mutual_info_score(atac.obs['cell_type'], atac.obs['leiden']))

np.mean(AMI_lst)  # 0.6232300501249661
np.mean(ARI_lst)  # 0.3134447495431582
np.mean(HOM_lst)  # 0.7585169797275532
np.mean(NMI_lst)  # 0.6239047230244502



## plot umap for atac (cisformer)
import scanpy as sc
import numpy as np
from scipy.sparse import csr_matrix
import snapatac2 as snap
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score, normalized_mutual_info_score, homogeneity_score

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

AMI_lst = []
ARI_lst = []
HOM_lst = []
NMI_lst = []

for _ in range(10):
    snap.pp.knn(atac)
    snap.tl.leiden(atac)
    AMI_lst.append(adjusted_mutual_info_score(atac.obs['cell_type'], atac.obs['leiden']))
    ARI_lst.append(adjusted_rand_score(atac.obs['cell_type'], atac.obs['leiden']))
    HOM_lst.append(homogeneity_score(atac.obs['cell_type'], atac.obs['leiden']))
    NMI_lst.append(normalized_mutual_info_score(atac.obs['cell_type'], atac.obs['leiden']))

np.mean(AMI_lst)  # 0.6446262671191171
np.mean(ARI_lst)  # 0.39339774822746815
np.mean(HOM_lst)  # 0.7028368728814347
np.mean(NMI_lst)  # 0.6450953102689998


## AMI & NMI & ARI & HOM
from plotnine import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['pdf.fonttype'] = 42

dat = pd.DataFrame({'val': [0.5340, 0.2463, 0.6443, 0.5348,
                            0.6232, 0.3134, 0.7585, 0.6239,
                            0.6446, 0.3934, 0.7028, 0.6451]})
dat['Method'] = ['BABEL']*4 + ['scButterfly']*4 + ['Cisformer']*4
dat['Metrics'] = ['AMI', 'ARI', 'HOM', 'NMI']*3
dat['Method'] = pd.Categorical(dat['Method'], categories=['BABEL', 'scButterfly', 'Cisformer'])
dat['Metrics'] = pd.Categorical(dat['Metrics'], categories=['AMI', 'NMI', 'ARI', 'HOM'])

p = ggplot(dat, aes(x='Metrics', y='val', fill='Method')) + geom_bar(stat='identity', position=position_dodge(), width=0.75) + ylab('') +\
                                                            scale_y_continuous(limits=[0, 0.8], breaks=np.arange(0, 0.8+0.1, 0.2)) + theme_bw()
p.save(filename='ami_nmi_ari_hom.pdf', dpi=600, height=4, width=6)



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

rna = sc.read_h5ad('rna_unpaired_cell_anno.h5ad')
sc.tl.pca(rna, n_comps=20)
sc.pp.neighbors(rna)
sc.tl.umap(rna)
#rna.uns['cell_type_colors'] = atac.uns['cell_type_colors']  # keep consistent color bw rna & atac
sc.pl.umap(rna, color='cell_type', legend_fontsize='7', legend_loc='right margin', size=5, title='', frameon=True, save='_rna_ncomps_50.pdf')




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


asso_res_mouse_clock_kp <- AssociationOfPeaksToAge(subset(epitrace_obj_age_conv_estimated_by_mouse_clock, cell_type %in% c('kidney proximal convoluted tubule epithelial cell')),
                                                   epitrace_age_name="EpiTraceAge_iterative", parallel=T, peakSetName='all')
asso_res_mouse_clock_kl <- AssociationOfPeaksToAge(subset(epitrace_obj_age_conv_estimated_by_mouse_clock, cell_type %in% c('kidney loop of Henle thick ascending limb epithelial cell')),
                                                   epitrace_age_name="EpiTraceAge_iterative", parallel=T, peakSetName='all')
kp_kl <- cbind(asso_res_mouse_clock_kp['correlation_of_EpiTraceAge'], asso_res_mouse_clock_kl['correlation_of_EpiTraceAge'])
kp_kl = na.omit(kp_kl)
# kp_kl[is.na(kp_kl)] = 0
# kp_kl = kp_kl[(kp_kl[, 1]!=0) | (kp_kl[, 2]!=0), ]
# kp_kl <- na.omit(cbind(asso_res_mouse_clock_kp['scaled_correlation_of_EpiTraceAge'], asso_res_mouse_clock_kl['scaled_correlation_of_EpiTraceAge']))
kp_kl['peak_id'] <- rownames(kp_kl)
colnames(kp_kl) <- c('cor_kp', 'cor_kl', 'peak_id')
write.table(kp_kl[, c('peak_id', 'cor_kp', 'cor_kl')], 'correlation_of_EpiTraceAge_kp_kl.txt', sep='\t', quote=FALSE, col.names=FALSE, row.names=FALSE)

## correlation comparison (kp vs. kl) not good to show
from plotnine import *
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

plt.rcParams['pdf.fonttype'] = 42

df = pd.read_table('correlation_of_EpiTraceAge_kp_kl.txt', header=0, names=['peak_id', 'cor_kp', 'cor_kl'])
df['idx'] = 'others'
df.loc[(df['cor_kp']>0.30)  &  (df['cor_kl']<0.05),  'idx'] = 'kp_pos'
df.loc[(df['cor_kp']<-0.05) &  (df['cor_kl']>0.05),  'idx'] = 'kp_neg'
df.loc[(df['cor_kl']>0.30)  &  (df['cor_kp']>0.30),  'idx'] = 'co_pos'
df.loc[(df['cor_kl']<-0.05) &  (df['cor_kp']<-0.05), 'idx'] = 'co_neg'
df['idx'].value_counts()
# others    9048
# kp_pos    2431
# kl_pos     903
# kp_neg     596
# kl_neg      17
df['idx'] = pd.Categorical(df['idx'], categories=['kp_pos', 'kp_neg', 'co_pos', 'co_neg', 'others'])

p = ggplot(df, aes(x='cor_kp', y='cor_kl', color='idx')) + geom_point(size=0.1) +\
                                                           scale_x_continuous(limits=[-0.4, 0.7], breaks=np.arange(-0.4, 0.7+0.1, 0.1)) +\
                                                           scale_y_continuous(limits=[-0.1, 0.6], breaks=np.arange(-0.1, 0.6+0.1, 0.1)) + theme_bw()
p.save(filename='correlation_of_EpiTraceAge_kp_kl.pdf', dpi=600, height=4, width=6)


from sklearn.cluster import KMeans
import seaborn as sns

plt.rcParams['pdf.fonttype'] = 42

dat = df[['cor_kp', 'cor_kl']]
kmeans = KMeans(n_clusters=4, random_state=0)
kmeans.fit(dat)

dat['cluster'] = kmeans.labels_   # 0:high_high  1:low_high  2:low_low  3:high_low
dat['cluster'].value_counts()
# 3    4975   high_low
# 0    3263   high_high
# 2    2485   low_low
# 1    2272   low_high
dat.sort_values('cluster', inplace=True)

sns.heatmap(dat[['cor_kp', 'cor_kl']], cmap='viridis', vmin=-0.6, vmax=0.6) 
plt.savefig('correlation_of_EpiTraceAge_kp_kl_heatmap.pdf')
plt.close()

sns.heatmap(dat.loc[dat['cluster']==0, ['cor_kp', 'cor_kl']], cmap='viridis', vmin=-0.6, vmax=0.6) 
plt.savefig('correlation_of_EpiTraceAge_kp_kl_heatmap_high_high.pdf')
plt.close()

sns.heatmap(dat.loc[dat['cluster']==1, ['cor_kp', 'cor_kl']], cmap='viridis', vmin=-0.6, vmax=0.6) 
plt.savefig('correlation_of_EpiTraceAge_kp_kl_heatmap_low_high.pdf')
plt.close()

sns.heatmap(dat.loc[dat['cluster']==2, ['cor_kp', 'cor_kl']], cmap='viridis', vmin=-0.6, vmax=0.6) 
plt.savefig('correlation_of_EpiTraceAge_kp_kl_heatmap_low_low.pdf')
plt.close()

sns.heatmap(dat.loc[dat['cluster']==3, ['cor_kp', 'cor_kl']], cmap='viridis', vmin=-0.6, vmax=0.6) 
plt.savefig('correlation_of_EpiTraceAge_kp_kl_heatmap_high_low.pdf')
plt.close()

dat['peak_id'] = df.loc[dat.index]['peak_id']
dat.to_csv('correlation_of_EpiTraceAge_kp_kl_heatmap_cluster.txt', sep='\t', index=False)


## calculate motif enrichment
import pandas as pd
import snapatac2 as snap

dat = pd.read_table('correlation_of_EpiTraceAge_kp_kl_heatmap_cluster.txt')
dat['peak_id'] = dat['peak_id'].str.replace('-', ':', n=1)
peaks_dict = {}
peaks_dict['high_low'] = list(dat[dat['cluster']==3]['peak_id'].values)
peaks_dict['low_high'] = list(dat[dat['cluster']==1]['peak_id'].values)
peaks_dict['high_high'] = list(dat[dat['cluster']==0]['peak_id'].values)
peaks_dict['low_low'] = list(dat[dat['cluster']==2]['peak_id'].values)

motifs = snap.tl.motif_enrichment(motifs=snap.datasets.cis_bp(unique=True),
                                  regions=peaks_dict,
                                  genome_fasta=snap.genome.mm10,
                                  background=None,
                                  method='hypergeometric')

motifs['high_low'].to_pandas().to_csv('motif_enrichment_high_low.txt', sep='\t', index=False)
motifs['low_high'].to_pandas().to_csv('motif_enrichment_low_high.txt', sep='\t', index=False)
motifs['high_high'].to_pandas().to_csv('motif_enrichment_high_high.txt', sep='\t', index=False)
motifs['low_low'].to_pandas().to_csv('motif_enrichment_low_low.txt', sep='\t', index=False)


######################################## QH ########################################
import pandas as pd
import snapatac2 as snap

# /data/home/zouqihang/desktop/project/M2M/version1.0.0/output/devel_fibro/bulk_subtype_enhancers
peaks_dict = {}
for ct in ['apFibro_CCL5', 'CAF_EndMT', 'CAF_PN', 'eFibro_COL11A1', 'eFibro_CTHRC1', 'Fibro_common', 'iFibro_FBLN1', 'MyoFibro_MYH11', 'qFibro_SAA1']:
    peaks_dict[ct] = list(pd.read_table(ct+'.txt', header=None)[0].values)

motifs = snap.tl.motif_enrichment(motifs=snap.datasets.cis_bp(unique=True),
                                  regions=peaks_dict,
                                  genome_fasta=snap.genome.hg38,
                                  background=None,
                                  method='hypergeometric')

for ct in ['apFibro_CCL5', 'CAF_EndMT', 'CAF_PN', 'eFibro_COL11A1', 'eFibro_CTHRC1', 'Fibro_common', 'iFibro_FBLN1', 'MyoFibro_MYH11', 'qFibro_SAA1']:
    motifs[ct].to_pandas().to_csv('/fs/home/jiluzhang/To_QH/motif_enrichment/'+ct+'_motif.txt', sep='\t', index=False)

# /data/home/zouqihang/desktop/project/M2M/version1.0.0/output/devel_macro/bulk_subtype_enhancers
peaks_dict = {}
for ct in ['Macro_APOC1', 'Macro_C1QC', 'Macro_CDC20', 'Macro_COL1A1', 'Macro_common', 'Macro_CXCL10', 'Macro_FABP4', 'Macro_IL32', 'Macro_SLPI', 'Macro_SPP1', 'Macro_THBS1']:
    peaks_dict[ct] = list(pd.read_table(ct+'.txt', header=None)[0].values)

motifs = snap.tl.motif_enrichment(motifs=snap.datasets.cis_bp(unique=True),
                                  regions=peaks_dict,
                                  genome_fasta=snap.genome.hg38,
                                  background=None,
                                  method='hypergeometric')

for ct in ['Macro_APOC1', 'Macro_C1QC', 'Macro_CDC20', 'Macro_COL1A1', 'Macro_common', 'Macro_CXCL10', 'Macro_FABP4', 'Macro_IL32', 'Macro_SLPI', 'Macro_SPP1', 'Macro_THBS1']:
    motifs[ct].to_pandas().to_csv('/fs/home/jiluzhang/To_QH/motif_enrichment/'+ct+'_motif.txt', sep='\t', index=False)
######################################## QH ########################################




## plot motif enrichment
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from plotnine import *

plt.rcParams['pdf.fonttype'] = 42

## high_low
df_high_low = pd.read_table('motif_enrichment_high_low.txt')
df_high_low['-log10(FDR)'] = -np.log10(df_high_low['adjusted p-value'])

val_max = df_high_low[df_high_low['-log10(FDR)']!=np.inf]['-log10(FDR)'].max()
np.random.seed(0)
df_high_low.loc[df_high_low['-log10(FDR)']==np.inf, '-log10(FDR)'] = val_max + np.random.uniform(0, 1, size=6)

df_high_low['idx'] = '0'
df_high_low.loc[(df_high_low['-log10(FDR)']>(-np.log10(0.05))) & (df_high_low['log2(fold change)']>(np.log2(1.25))), 'idx'] = '1'

p = ggplot(df_high_low, aes(x='log2(fold change)', y='-log10(FDR)', color='idx')) + geom_point(size=0.25) + scale_color_manual(values=['grey', 'red']) +\
                                                                                    scale_x_continuous(limits=[-1, 1], breaks=np.arange(-1, 1+0.1, 0.5)) +\
                                                                                    scale_y_continuous(limits=[0, 20], breaks=np.arange(0, 20+0.1, 5)) +\
                                                                                    geom_text(aes(label='name'), data=df_high_low[df_high_low['idx']=='1'], size=3, nudge_y=0.2) + theme_bw()
                                                                                    # geom_hline(yintercept=(-np.log10(0.05)), color="black", linetype="dashed") +\
                                                                                    # geom_vline(xintercept=(np.log2(1.25)), color='black', linetype='dashed')
p.save(filename='motif_enrichment_high_low_vocano.pdf', dpi=600, height=4, width=4)

## low_high
df_low_high = pd.read_table('motif_enrichment_low_high.txt')
df_low_high['-log10(FDR)'] = -np.log10(df_low_high['adjusted p-value'])

#val_max = df_low_high[df_low_high['-log10(FDR)']!=np.inf]['-log10(FDR)'].max()
#np.random.seed(0)
#df_low_high.loc[df_low_high['-log10(FDR)']==np.inf, '-log10(FDR)'] = val_max + np.random.uniform(0, 1, size=6)

df_low_high['idx'] = '0'
df_low_high.loc[(df_low_high['-log10(FDR)']>(-np.log10(0.05))) & (df_low_high['log2(fold change)']>(np.log2(1.25))), 'idx'] = '1'

p = ggplot(df_low_high, aes(x='log2(fold change)', y='-log10(FDR)', color='idx')) + geom_point(size=0.25) + scale_color_manual(values=['grey', 'red']) +\
                                                                                    scale_x_continuous(limits=[-2, 2], breaks=np.arange(-2, 2+0.1, 0.5)) +\
                                                                                    scale_y_continuous(limits=[0, 5], breaks=np.arange(0, 5+0.1, 1)) +\
                                                                                    geom_text(aes(label='name'), data=df_low_high[df_low_high['idx']=='1'], size=3, nudge_y=0.2) + theme_bw()
p.save(filename='motif_enrichment_low_high_vocano.pdf', dpi=600, height=4, width=4)

## high_high
df_high_high = pd.read_table('motif_enrichment_high_high.txt')
df_high_high['-log10(FDR)'] = -np.log10(df_high_high['adjusted p-value'])

# val_max = df_high_high[df_high_high['-log10(FDR)']!=np.inf]['-log10(FDR)'].max()
# np.random.seed(0)
# df_high_high.loc[df_high_high['-log10(FDR)']==np.inf, '-log10(FDR)'] = val_max + np.random.uniform(0, 1, size=6)

df_high_high['idx'] = '0'
df_high_high.loc[(df_high_high['-log10(FDR)']>(-np.log10(0.05))) & (df_high_high['log2(fold change)']>(np.log2(1.25))), 'idx'] = '1'

p = ggplot(df_high_high, aes(x='log2(fold change)', y='-log10(FDR)', color='idx')) + geom_point(size=0.25) + scale_color_manual(values=['grey', 'red']) +\
                                                                                     scale_x_continuous(limits=[-1, 1.5], breaks=np.arange(-1, 1.5+0.1, 0.5)) +\
                                                                                     scale_y_continuous(limits=[0, 8], breaks=np.arange(0, 8+0.1, 2)) +\
                                                                                     geom_text(aes(label='name'), data=df_high_high[df_high_high['idx']=='1'], size=3, nudge_y=0.2) + theme_bw()
p.save(filename='motif_enrichment_high_high_vocano.pdf', dpi=600, height=4, width=4)

## low_low
df_low_low = pd.read_table('motif_enrichment_low_low.txt')
df_low_low['-log10(FDR)'] = -np.log10(df_low_low['adjusted p-value'])

#val_max = df_low_low[df_low_low['-log10(FDR)']!=np.inf]['-log10(FDR)'].max()
#np.random.seed(0)
#df_low_low.loc[df_low_low['-log10(FDR)']==np.inf, '-log10(FDR)'] = val_max + np.random.uniform(0, 1, size=6)

df_low_low['idx'] = '0'
df_low_low.loc[(df_low_low['-log10(FDR)']>(-np.log10(0.05))) & (df_low_low['log2(fold change)']>(np.log2(1.25))), 'idx'] = '1'

p = ggplot(df_low_low, aes(x='log2(fold change)', y='-log10(FDR)', color='idx')) + geom_point(size=0.25) + scale_color_manual(values=['grey', 'red']) +\
                                                                                   scale_x_continuous(limits=[-1.5, 1], breaks=np.arange(-1.5, 1+0.1, 0.5)) +\
                                                                                   scale_y_continuous(limits=[0, 8], breaks=np.arange(0, 8+0.1, 2)) +\
                                                                                   geom_text(aes(label='name'), data=df_low_low[df_low_low['idx']=='1'], size=3, nudge_y=0.2) + theme_bw()
p.save(filename='motif_enrichment_low_low_vocano.pdf', dpi=600, height=4, width=4)






## pos & neg seperately
# df['cor_kp_kl'] = df['cor_kp'] - df['cor_kl']

# df_pos = df[df['cor_kp']>0.1]  # 8833
# df_pos.sort_values('cor_kp_kl', ascending=False, inplace=True)
# df_pos['idx'] = range(df_pos.shape[0])
# p = ggplot(df_pos, aes(x='idx', y='cor_kp_kl')) + geom_point(size=0.25) +\
#                                                   scale_x_continuous(limits=[0, 9000], breaks=np.arange(0, 9000+1, 3000)) +\
#                                                   scale_y_continuous(limits=[-0.5, 0.7], breaks=np.arange(-0.5, 0.7+0.1, 0.2)) + theme_bw()
# p.save(filename='correlation_of_EpiTraceAge_kp_kl_pos.pdf', dpi=600, height=4, width=4)

# df_neg = df[df['cor_kp']<-0.1]  # 1769
# df_neg.sort_values('cor_kp_kl', ascending=True, inplace=True)
# df_neg['idx'] = range(df_neg.shape[0])
# p = ggplot(df_neg, aes(x='idx', y='cor_kp_kl')) + geom_point(size=0.25) +\
#                                                   scale_x_continuous(limits=[0, 1800], breaks=np.arange(0, 1800+1, 300)) +\
#                                                   scale_y_continuous(limits=[-0.8, -0.2], breaks=np.arange(-0.8, -0.2+0.1, 0.2)) + theme_bw()
# p.save(filename='correlation_of_EpiTraceAge_kp_kl_neg.pdf', dpi=600, height=4, width=4)


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




