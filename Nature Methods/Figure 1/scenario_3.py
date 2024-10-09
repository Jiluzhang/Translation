#### map to cCREs
import scanpy as sc
import pandas as pd
from scipy.sparse import csr_matrix
import pybedtools
import numpy as np
from tqdm import tqdm
import random

dat_raw = sc.read_h5ad('GSE194122_openproblems_neurips2021_multiome_BMMC_processed.h5ad')  # 69249 × 129921
dat_raw.obs['cell_anno'] = dat_raw.obs['cell_type']

## random selection (2000)
idx = dat_raw.obs.index.values.copy()
random.seed(0)
random.shuffle(idx)
dat = dat_raw[idx[:2000], :].copy()  # 2000 × 129921

## rna
rna = dat[:, dat.var['feature_types']=='GEX'].copy()  # 2000 × 13431
rna.X = rna.layers['counts'].toarray()  # use raw counts
genes = pd.read_table('human_genes.txt', names=['gene_ids', 'gene_name'])
rna_exp = pd.concat([rna.var, pd.DataFrame(rna.X.T, index=rna.var.index)], axis=1)
rna_exp.rename(columns={'gene_id': 'gene_ids'}, inplace=True)
X_new = pd.merge(genes, rna_exp, how='left', on='gene_ids').iloc[:, 3:].T
X_new.fillna(value=0, inplace=True)
rna_new = sc.AnnData(X_new.values, obs=rna.obs[['cell_anno']], var=pd.DataFrame({'gene_ids': genes['gene_ids'], 'feature_types': 'Gene Expression'}))
rna_new.var.index = genes['gene_name'].values
rna_new.X = csr_matrix(rna_new.X)   # 2000 × 38244

sc.pp.filter_genes(rna_new, min_cells=10)       # 2000 × 11895
sc.pp.filter_cells(rna_new, min_genes=200)      # 1997 × 11895
sc.pp.filter_cells(rna_new, max_genes=20000)    # 1997 × 11895
rna_new.obs['n_genes'].max()                    # 5573

## atac
atac = dat[:, dat.var['feature_types']=='ATAC'].copy()  # 2000 × 116490
# atac.X[atac.X>0] = 1  # binarization
atac.X = atac.X.toarray()  # with binarization
  
peaks = pd.DataFrame({'id': atac.var_names})
peaks['chr'] = peaks['id'].map(lambda x: x.split('-')[0])
peaks['start'] = peaks['id'].map(lambda x: x.split('-')[1])
peaks['end'] = peaks['id'].map(lambda x: x.split('-')[2])
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

atac_new = sc.AnnData(m, obs=atac.obs[['cell_anno']], var=pd.DataFrame({'gene_ids': cCREs['chr']+':'+cCREs['start'].map(str)+'-'+cCREs['end'].map(str), 'feature_types': 'Peaks'}))
atac_new.var.index = atac_new.var['gene_ids'].values
atac_new.X = csr_matrix(atac_new.X)

sc.pp.filter_genes(atac_new, min_cells=10)     # 2000 × 198782
sc.pp.filter_cells(atac_new, min_genes=500)    # 1995 × 198782
sc.pp.filter_cells(atac_new, max_genes=50000)  # 1995 × 198782

## output rna & atac
idx = np.intersect1d(rna_new.obs.index, atac_new.obs.index)
del rna_new.obs['n_genes']
del rna_new.var['n_cells']
rna_new[idx, :].copy().write('rna_2000.h5ad')     # 1992 × 11895
del atac_new.obs['n_genes']
del atac_new.var['n_cells']
atac_new[idx, :].copy().write('atac_2000.h5ad')   # 1992 × 198782


#### map to reference
import scanpy as sc
import pandas as pd
from scipy.sparse import csr_matrix
import numpy as np

## rna
rna = sc.read_h5ad('rna_2000.h5ad')                                                        # 1992 × 11895
rna_ref = sc.read_h5ad('/fs/home/jiluzhang/Nature_methods/Figure_1/scenario_1/rna.h5ad')   # 9964 × 16706
genes = rna_ref.var[['gene_ids']].copy()
genes['gene_name'] = genes.index.values
genes.reset_index(drop=True, inplace=True)
rna.X = rna.X.toarray()
rna_exp = pd.concat([rna.var, pd.DataFrame(rna.X.T, index=rna.var.index)], axis=1)
X_new = pd.merge(genes, rna_exp, how='left', on='gene_ids').iloc[:, 3:].T
X_new.fillna(value=0, inplace=True)
rna_new = sc.AnnData(X_new.values, obs=rna.obs[['cell_anno']], var=pd.DataFrame({'gene_ids': genes['gene_ids'], 'feature_types': 'Gene Expression'}))
rna_new.var.index = genes['gene_name'].values
rna_new.X = csr_matrix(rna_new.X)   # 1992 × 16706
rna_new.write('rna_test.h5ad')
np.count_nonzero(rna_new.X.toarray(), axis=1).max()  # 5563

## atac
atac = sc.read_h5ad('atac_2000.h5ad')                                                        # 1992 × 198782
atac_ref = sc.read_h5ad('/fs/home/jiluzhang/Nature_methods/Figure_1/scenario_1/atac.h5ad')   # 9964 × 236295

peaks = atac_ref.var[['gene_ids']].copy()
peaks['gene_name'] = peaks.index.values
peaks.reset_index(drop=True, inplace=True)
atac.X = atac.X.toarray()
atac_exp = pd.concat([atac.var, pd.DataFrame(atac.X.T, index=atac.var.index)], axis=1)
X_new = pd.merge(peaks, atac_exp, how='left', on='gene_ids').iloc[:, 3:].T
X_new.fillna(value=0, inplace=True)
atac_new = sc.AnnData(X_new.values, obs=atac.obs[['cell_anno']], var=pd.DataFrame({'gene_ids': peaks['gene_ids'], 'feature_types': 'Peaks'}))
atac_new.var.index = peaks['gene_name'].values
atac_new.X = csr_matrix(atac_new.X)   # 1992 × 236295
atac_new.write('atac_test.h5ad')


#### cisformer
python data_preprocess.py -r rna_test.h5ad -a atac_test.h5ad -s test_pt --dt test -n test --config rna2atac_config_test.yaml  # 4 min
accelerate launch --config_file accelerator_config_test.yaml --main_process_port 29822 rna2atac_predict.py -d test_pt \
                  -l /fs/home/jiluzhang/Nature_methods/Figure_1/scenario_1/cisformer/save_mlt_40_large/2024-09-30_rna2atac_pbmc_34/pytorch_model.bin \
                  --config_file rna2atac_config_test.yaml  # 4.5 min
python npy2h5ad.py
python cal_auroc_auprc.py --pred atac_cisformer.h5ad --true atac_test.h5ad
# Cell-wise AUROC: 0.8682
# Cell-wise AUPRC: 0.2125
# Peak-wise AUROC: 0.6392
# Peak-wise AUPRC: 0.0704
python plot_save_umap.py --pred atac_cisformer.h5ad --true atac_test.h5ad  # 2 min
python cal_cluster.py --file atac_cisformer_umap.h5ad
# AMI: 0.6123
# ARI: 0.4621
# HOM: 0.5484
# NMI: 0.6195



#### scbutterfly
## python scbt_b_predict.py  # 10 min
import os
from scButterfly.butterfly import Butterfly
from scButterfly.split_datasets import *
import scanpy as sc
import anndata as ad

os.environ["CUDA_VISIBLE_DEVICES"] = "4"

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
A2R_predict, R2A_predict = butterfly.test_model(load_model=True, model_path='/fs/home/jiluzhang/Nature_methods/Figure_1/scenario_1/scbt',
                                                test_cluster=False, test_figure=False, output_data=True)

cp predict/R2A.h5ad atac_scbt.h5ad
python cal_auroc_auprc.py --pred atac_scbt.h5ad --true atac_test.h5ad
# Cell-wise AUROC: 0.8726
# Cell-wise AUPRC: 0.2168
# Peak-wise AUROC: 0.6374
# Peak-wise AUPRC: 0.0730

python plot_save_umap.py --pred atac_scbt.h5ad --true atac_test.h5ad  # 8 min
python cal_cluster.py --file atac_scbt_umap.h5ad
# AMI: 0.5214
# ARI: 0.3538
# HOM: 0.4947
# NMI: 0.5319


#### BABEL
python h5ad2h5.py -n test
python /fs/home/jiluzhang/BABEL/bin/predict_model.py --checkpoint /fs/home/jiluzhang/Nature_methods/Figure_1/scenario_1/babel/train_out --data test.h5 --outdir test_out --device 2 --nofilter --noplot --transonly   # 9 min

python match_cell.py

python cal_auroc_auprc.py --pred atac_babel.h5ad --true atac_test.h5ad
# Cell-wise AUROC: 0.8727
# Cell-wise AUPRC: 0.2161
# Peak-wise AUROC: 0.6292
# Peak-wise AUPRC: 0.0679

python plot_save_umap.py --pred atac_babel.h5ad --true atac_test.h5ad  # 8 min
python cal_cluster.py --file atac_babel_umap.h5ad
# AMI: 0.5312
# ARI: 0.3402
# HOM: 0.5086
# NMI: 0.5416


#### plot metrics
## python plot_metrics.py
#    ami_nmi_ari_hom.txt & auroc_auprc.txt
# -> ami_nmi_ari_hom.pdf & auroc_auprc.pdf

#### plot true atac umap
## python plot_true_umap.py
