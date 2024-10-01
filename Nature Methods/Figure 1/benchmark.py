#### Dataset 1: PBMCs (https://cf.10xgenomics.com/samples/cell-arc/2.0.0/pbmc_granulocyte_sorted_10k/pbmc_granulocyte_sorted_10k_filtered_feature_bc_matrix.h5)
####            meta info (https://mailustceducn-my.sharepoint.com/personal/hyl2016_mail_ustc_edu_cn/_layouts/15/download.aspx?UniqueId=e86f4bba%2D8560%2D4689%2Dbb93%2D9ea5c859ba28)
#### Dataset 2: BMMCs
#### Dataset 3: Adult brain

#### celltype (not predicted.id)
# CD4 TCM (Central memory T cells)
# CD4 Naive
# CD8 Naive
# CD16 Mono
# NK
# Treg (Regulatory T cells)
# CD14 Mono
# CD4 intermediate
# CD4 TEM (Effector memory T cells)
# RPL/S Leukocytes (discard)
# B memory
# MAIT (mucosal-associated invariant T cells)
# IL1B+ Mono -> CD14 Mono
# B naive
# CD8 TEM
# pDC (Plasmacytoid dendritic cells)
# cDC2 (type-2 conventional dendritic cells)


#### h5 to h5ad
import scanpy as sc
import pandas as pd
from scipy.sparse import csr_matrix
import pybedtools
import numpy as np
from tqdm import tqdm

dat = sc.read_10x_h5('pbmc_granulocyte_sorted_10k_filtered_feature_bc_matrix.h5', gex_only=False)  # 11898 × 180488
dat.var_names_make_unique()

## add cell annotation
meta_info = pd.read_table('meta_data.txt')
cell_barcode = pd.DataFrame({'barcode': dat.obs.index.values})
dat.obs['cell_anno'] = pd.merge(cell_barcode, meta_info, how='left')['celltype'].values
dat = dat[(~dat.obs['cell_anno'].isna() & (dat.obs['cell_anno']!='RPL/S Leukocytes')), :].copy()   # 9964 × 180488
dat.obs['cell_anno'].replace({'IL1B+ Mono': 'CD14 Mono'}, inplace=True)

## rna
rna = dat[:, dat.var['feature_types']=='Gene Expression'].copy()  # 9964 × 36601
rna.X = rna.X.toarray()
genes = pd.read_table('human_genes.txt', names=['gene_ids', 'gene_name'])
rna_exp = pd.concat([rna.var, pd.DataFrame(rna.X.T, index=rna.var.index)], axis=1)
X_new = pd.merge(genes, rna_exp, how='left', on='gene_ids').iloc[:, 4:].T
X_new.fillna(value=0, inplace=True)
rna_new = sc.AnnData(X_new.values, obs=rna.obs, var=pd.DataFrame({'gene_ids': genes['gene_ids'], 'feature_types': 'Gene Expression'}))
rna_new.var.index = genes['gene_name'].values
rna_new.X = csr_matrix(rna_new.X)   # 9964 × 38244

sc.pp.filter_genes(rna_new, min_cells=10)       # 9964 × 16706
sc.pp.filter_cells(rna_new, min_genes=200)      # 9964 × 16706
sc.pp.filter_cells(rna_new, max_genes=20000)    # 9964 × 16706
rna_new.obs['n_genes'].max()      # 3529

## atac
atac = dat[:, dat.var['feature_types']=='Peaks'].copy()  # 9964 × 143887
atac.X[atac.X>0] = 1  # binarization
atac.X = atac.X.toarray()
  
peaks = pd.DataFrame({'id': atac.var_names})
peaks['chr'] = peaks['id'].map(lambda x: x.split(':')[0])
peaks['start'] = peaks['id'].map(lambda x: x.split(':')[1].split('-')[0])
peaks['end'] = peaks['id'].map(lambda x: x.split(':')[1].split('-')[1])
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

atac_new = sc.AnnData(m, obs=atac.obs, var=pd.DataFrame({'gene_ids': cCREs['chr']+':'+cCREs['start'].map(str)+'-'+cCREs['end'].map(str), 'feature_types': 'Peaks'}))
atac_new.var.index = atac_new.var['gene_ids'].values
atac_new.X = csr_matrix(atac_new.X)

sc.pp.filter_genes(atac_new, min_cells=10)     # 9964 × 236295
sc.pp.filter_cells(atac_new, min_genes=500)    # 9964 × 236295
sc.pp.filter_cells(atac_new, max_genes=50000)  # 9964 × 236295

## output rna & atac (no cells are filtered)
rna_new.write('rna.h5ad')
atac_new.write('atac.h5ad')


#### train model & prediction
python split_train_val_test.py --RNA rna.h5ad --ATAC atac.h5ad --train_pct 0.7 --valid_pct 0.1  # 15 s
python data_preprocess.py -r rna_val.h5ad -a atac_val.h5ad -s val_pt --dt val -n val --config rna2atac_config_train_mlt_1.yaml             # 20 s
python data_preprocess.py -r rna_test.h5ad -a atac_test.h5ad -s test_pt --dt test -n test --config rna2atac_config_test.yaml                     # 17 min

### early stop: loss -> auroc (delta:0)  use round to avoid float issue 
### patience: 25 -> 10
### mult=1
python data_preprocess.py -r rna_train.h5ad -a atac_train.h5ad -s train_pt_mlt_1 --dt train -n train --config rna2atac_config_train_mlt_1.yaml   # 1.5 min
nohup accelerate launch --config_file accelerator_config_train.yaml --main_process_port 29823 rna2atac_train.py --config_file rna2atac_config_train_mlt_1.yaml \
                        --train_data_dir train_pt_mlt_1 --val_data_dir val_pt -s save_mlt_1 -n rna2atac_pbmc > rna2atac_train_20240929_mlt_1.log &
accelerate launch --config_file accelerator_config_test.yaml --main_process_port 29822 rna2atac_predict.py -d test_pt \
                  -l save_mlt_1/2024-09-29_rna2atac_pbmc_228/pytorch_model.bin --config_file rna2atac_config_test.yaml  # 3 min
python npy2h5ad.py
python cal_auroc_auprc.py --pred atac_cisformer.h5ad --true atac_test.h5ad
# Cell-wise AUROC: 0.8445
# Cell-wise AUPRC: 0.4068
# Peak-wise AUROC: 0.5643
# Peak-wise AUPRC: 0.1058
python plot_save_umap.py --pred atac_cisformer.h5ad --true atac_test.h5ad  # 2 min
python cal_cluster.py --file atac_cisformer_umap.h5ad
# AMI: 0.5806
# ARI: 0.4146
# HOM: 0.5477
# NMI: 0.5859

# ### mult=2
# python data_preprocess.py -r rna_train.h5ad -a atac_train.h5ad -s train_pt_mlt_2 --dt train -n train --config rna2atac_config_train_mlt_2.yaml   # 1.5 min
# nohup accelerate launch --config_file accelerator_config_train.yaml --main_process_port 29823 rna2atac_train.py --config_file rna2atac_config_train_mlt_2.yaml \
#                         --train_data_dir train_pt_mlt_2 --val_data_dir val_pt -s save_mlt_2 -n rna2atac_pbmc > rna2atac_train_20240929_mlt_2.log &
# accelerate launch --config_file accelerator_config_test.yaml --main_process_port 29822 rna2atac_predict.py -d test_pt \
#                       -l save_mlt_2/2024-09-29_rna2atac_pbmc_263/pytorch_model.bin --config_file rna2atac_config_test.yaml  # 3 min
# python npy2h5ad.py
# python cal_auroc_auprc.py --pred atac_cisformer.h5ad --true atac_test.h5ad
# # Cell-wise AUROC: 0.8531
# # Cell-wise AUPRC: 0.4203
# # Peak-wise AUROC: 0.5799
# # Peak-wise AUPRC: 0.1097
# python plot_save_umap.py --pred atac_cisformer.h5ad --true atac_test.h5ad  # 4 min
# python cal_cluster.py --file atac_cisformer_umap.h5ad
# # AMI: 0.5660
# # ARI: 0.4035
# # HOM: 0.5329
# # NMI: 0.5709

### mult=5
python data_preprocess.py -r rna_train.h5ad -a atac_train.h5ad -s train_pt_mlt_5 --dt train -n train --config rna2atac_config_train_mlt_5.yaml   # 3 min
nohup accelerate launch --config_file accelerator_config_train.yaml --main_process_port 29824 rna2atac_train.py --config_file rna2atac_config_train_mlt_5.yaml \
                        --train_data_dir train_pt_mlt_5 --val_data_dir val_pt -s save_mlt_5 -n rna2atac_pbmc > rna2atac_train_20240929_mlt_5.log &   # 1748857
accelerate launch --config_file accelerator_config_test.yaml --main_process_port 29822 rna2atac_predict.py -d test_pt \
                      -l save_mlt_5/2024-09-29_rna2atac_pbmc_283/pytorch_model.bin --config_file rna2atac_config_test.yaml  # 3 min
python npy2h5ad.py
python cal_auroc_auprc.py --pred atac_cisformer.h5ad --true atac_test.h5ad
# Cell-wise AUROC: 0.8612
# Cell-wise AUPRC: 0.4334
# Peak-wise AUROC: 0.5927
# Peak-wise AUPRC: 0.1130
python plot_save_umap.py --pred atac_cisformer.h5ad --true atac_test.h5ad  # 8 min
python cal_cluster.py --file atac_cisformer_umap.h5ad
# AMI: 0.6426
# ARI: 0.4658
# HOM: 0.6138
# NMI: 0.6468

### mult=10
python data_preprocess.py -r rna_train.h5ad -a atac_train.h5ad -s train_pt_mlt_10 --dt train -n train --config rna2atac_config_train_mlt_10.yaml   # 5.5 min
nohup accelerate launch --config_file accelerator_config_train.yaml --main_process_port 29825 rna2atac_train.py --config_file rna2atac_config_train_mlt_10.yaml \
                        --train_data_dir train_pt_mlt_10 --val_data_dir val_pt -s save_mlt_10 -n rna2atac_pbmc > rna2atac_train_20240929_mlt_10.log &   # 1766032
accelerate launch --config_file accelerator_config_test.yaml --main_process_port 29822 rna2atac_predict.py -d test_pt \
                      -l save_mlt_10/2024-09-29_rna2atac_pbmc_299/pytorch_model.bin --config_file rna2atac_config_test.yaml  # 3 min
python npy2h5ad.py
python cal_auroc_auprc.py --pred atac_cisformer.h5ad --true atac_test.h5ad
# Cell-wise AUROC: 0.8667
# Cell-wise AUPRC: 0.4406
# Peak-wise AUROC: 0.6022
# Peak-wise AUPRC: 0.1152
python plot_save_umap.py --pred atac_cisformer.h5ad --true atac_test.h5ad  # 8 min
python cal_cluster.py --file atac_cisformer_umap.h5ad
# AMI: 0.6514
# ARI: 0.4821
# HOM: 0.6206
# NMI: 0.6556

### mult=20
python data_preprocess.py -r rna_train.h5ad -a atac_train.h5ad -s train_pt_mlt_20 --dt train -n train --config rna2atac_config_train_mlt_20.yaml   #  10 min
nohup accelerate launch --config_file accelerator_config_train.yaml --main_process_port 29826 rna2atac_train.py --config_file rna2atac_config_train_mlt_20.yaml \
                        --train_data_dir train_pt_mlt_20 --val_data_dir val_pt -s save_mlt_20 -n rna2atac_pbmc > rna2atac_train_20240929_mlt_20.log &   # 1771075
accelerate launch --config_file accelerator_config_test.yaml --main_process_port 29822 rna2atac_predict.py -d test_pt \
                      -l save_mlt_20/2024-09-29_rna2atac_pbmc_184/pytorch_model.bin --config_file rna2atac_config_test.yaml  # 3 min
python npy2h5ad.py
python cal_auroc_auprc.py --pred atac_cisformer.h5ad --true atac_test.h5ad
# Cell-wise AUROC: 0.8690
# Cell-wise AUPRC: 0.4437
# Peak-wise AUROC: 0.6053
# Peak-wise AUPRC: 0.1160
python plot_save_umap.py --pred atac_cisformer.h5ad --true atac_test.h5ad  # 8 min
python cal_cluster.py --file atac_cisformer_umap.h5ad
# AMI: 0.6564
# ARI: 0.4976
# HOM: 0.6292
# NMI: 0.6608

### mult=40
python data_preprocess.py -r rna_train.h5ad -a atac_train.h5ad -s train_pt_mlt_40 --dt train -n train --config rna2atac_config_train_mlt_40.yaml   
nohup accelerate launch --config_file accelerator_config_train.yaml --main_process_port 29827 rna2atac_train.py --config_file rna2atac_config_train_mlt_40.yaml \
                        --train_data_dir train_pt_mlt_40 --val_data_dir val_pt -s save_mlt_40 -n rna2atac_pbmc > rna2atac_train_20240929_mlt_40.log &   # 1776670
accelerate launch --config_file accelerator_config_test.yaml --main_process_port 29822 rna2atac_predict.py -d test_pt \
                      -l save_mlt_40/2024-09-29_rna2atac_pbmc_92/pytorch_model.bin --config_file rna2atac_config_test.yaml  # 3 min
python npy2h5ad.py
python cal_auroc_auprc.py --pred atac_cisformer.h5ad --true atac_test.h5ad
# Cell-wise AUROC: 0.8702
# Cell-wise AUPRC: 0.4440
# Peak-wise AUROC: 0.6100
# Peak-wise AUPRC: 0.1172
python plot_save_umap.py --pred atac_cisformer.h5ad --true atac_test.h5ad  # 8 min
python cal_cluster.py --file atac_cisformer_umap.h5ad
# AMI: 0.6797
# ARI: 0.5109
# HOM: 0.6691
# NMI: 0.6845

### mult=100
python data_preprocess.py -r rna_train.h5ad -a atac_train.h5ad -s train_pt_mlt_100 --dt train -n train --config rna2atac_config_train_mlt_100.yaml   
nohup accelerate launch --config_file accelerator_config_train.yaml --main_process_port 29827 rna2atac_train.py --config_file rna2atac_config_train_mlt_100.yaml \
                        --train_data_dir train_pt_mlt_100 --val_data_dir val_pt -s save_mlt_100 -n rna2atac_pbmc > rna2atac_train_20240930_mlt_100.log &   # 3851526
accelerate launch --config_file accelerator_config_test.yaml --main_process_port 29822 rna2atac_predict.py -d test_pt \
                      -l save_mlt_40/2024-09-29_rna2atac_pbmc_92/pytorch_model.bin --config_file rna2atac_config_test.yaml  # 3 min
python npy2h5ad.py
python cal_auroc_auprc.py --pred atac_cisformer.h5ad --true atac_test.h5ad
# Cell-wise AUROC: 
# Cell-wise AUPRC: 
# Peak-wise AUROC: 
# Peak-wise AUPRC: 
python plot_save_umap.py --pred atac_cisformer.h5ad --true atac_test.h5ad  # 8 min
python cal_cluster.py --file atac_cisformer_umap.h5ad
# AMI: 
# ARI: 
# HOM: 
# NMI: 

### mult=20 (more parameters)
nohup accelerate launch --config_file accelerator_config_train.yaml --main_process_port 29826 rna2atac_train.py --config_file rna2atac_config_train_mlt_20_large.yaml \
                        --train_data_dir train_pt_mlt_20 --val_data_dir val_pt -s save_mlt_20_large -n rna2atac_pbmc > rna2atac_train_20240930_mlt_20_large.log &   # 3856717
accelerate launch --config_file accelerator_config_test.yaml --main_process_port 29822 rna2atac_predict.py -d test_pt \
                      -l save_mlt_20_large/2024-09-30_rna2atac_pbmc_20/pytorch_model.bin --config_file rna2atac_config_test_large.yaml  # 3 min
python npy2h5ad.py
python cal_auroc_auprc.py --pred atac_cisformer.h5ad --true atac_test.h5ad
# Cell-wise AUROC: 0.8928
# Cell-wise AUPRC: 0.4817
# Peak-wise AUROC: 0.6721
# Peak-wise AUPRC: 0.1342
python plot_save_umap.py --pred atac_cisformer.h5ad --true atac_test.h5ad  # 8 min
python cal_cluster.py --file atac_cisformer_umap.h5ad
# AMI: 0.6606
# ARI: 0.458
# HOM: 0.6526
# NMI: 0.6654

### mult=40 (more parameters)
nohup accelerate launch --config_file accelerator_config_train.yaml --main_process_port 29826 rna2atac_train.py --config_file rna2atac_config_train_mlt_40_large.yaml \
                        --train_data_dir train_pt_mlt_40 --val_data_dir val_pt -s save_mlt_40_large -n rna2atac_pbmc > rna2atac_train_20240930_mlt_40_large.log &   # 472737
accelerate launch --config_file accelerator_config_test.yaml --main_process_port 29822 rna2atac_predict.py -d test_pt \
                      -l save_mlt_40_large/2024-09-30_rna2atac_pbmc_34/pytorch_model.bin --config_file rna2atac_config_test_large.yaml  # 6 min
python npy2h5ad.py
python cal_auroc_auprc.py --pred atac_cisformer.h5ad --true atac_test.h5ad
# Cell-wise AUROC: 0.8944
# Cell-wise AUPRC: 0.4843
# Peak-wise AUROC: 0.6802
# Peak-wise AUPRC: 0.1365
python plot_save_umap.py --pred atac_cisformer.h5ad --true atac_test.h5ad  # 8 min
python cal_cluster.py --file atac_cisformer_umap.h5ad
# AMI: 0.7179
# ARI: 0.5415
# HOM: 0.7177
# NMI: 0.722

### mult=40 (more and more parameters)
nohup accelerate launch --config_file accelerator_config_train.yaml --main_process_port 29826 rna2atac_train.py --config_file rna2atac_config_train_mlt_40_huge.yaml \
                        --train_data_dir train_pt_mlt_40 --val_data_dir val_pt -s save_mlt_40_huge -n rna2atac_pbmc > rna2atac_train_20241001_mlt_40_huge.log &   # 2368119
accelerate launch --config_file accelerator_config_test.yaml --main_process_port 29822 rna2atac_predict.py -d test_pt \
                      -l save_mlt_40_huge/2024-10-01_rna2atac_pbmc_11/pytorch_model.bin --config_file rna2atac_config_test_huge.yaml  # 6 min
python npy2h5ad.py
python cal_auroc_auprc.py --pred atac_cisformer.h5ad --true atac_test.h5ad
# Cell-wise AUROC: 0.8943
# Cell-wise AUPRC: 0.4858
# Peak-wise AUROC: 0.6815
# Peak-wise AUPRC: 0.1365
python plot_save_umap.py --pred atac_cisformer.h5ad --true atac_test.h5ad  # 8 min
python cal_cluster.py --file atac_cisformer_umap.h5ad
# AMI: 0.7016
# ARI: 0.5636
# HOM: 0.6807
# NMI: 0.7057


#### scbutterfly
## python scbt_b.py
## nohup python scbt_b.py > scbt_b_20240929.log &
import os
from scButterfly.butterfly import Butterfly
from scButterfly.split_datasets import *
import scanpy as sc
import anndata as ad
import random

os.environ["CUDA_VISIBLE_DEVICES"] = "4"

RNA_data = sc.read_h5ad('rna.h5ad')
ATAC_data = sc.read_h5ad('atac.h5ad')

random.seed(0)
idx = list(range(RNA_data.n_obs))
random.shuffle(idx)
train_id = idx[:int(len(idx)*0.7)]
validation_id = idx[int(len(idx)*0.7):(int(len(idx)*0.7)+int(len(idx)*0.1))]
test_id = idx[(int(len(idx)*0.7)+int(len(idx)*0.1)):]

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

cp predict/R2A.h5ad atac_scbt.h5ad
python cal_auroc_auprc.py --pred atac_scbt.h5ad --true atac_test.h5ad
# Cell-wise AUROC: 0.8975
# Cell-wise AUPRC: 0.4910
# Peak-wise AUROC: 0.6996
# Peak-wise AUPRC: 0.1475

python plot_save_umap.py --pred atac_scbt.h5ad --true atac_test.h5ad  # 8 min
python cal_cluster.py --file atac_scbt_umap.h5ad
# AMI: 0.6982
# ARI: 0.4801
# HOM: 0.7259
# NMI: 0.7036



#### BABEL
## concat train & valid dataset
## python concat_train_val.py
import scanpy as sc
import anndata as ad

rna_train = sc.read_h5ad('rna_train.h5ad')
rna_val = sc.read_h5ad('rna_val.h5ad')
rna_train_val = ad.concat([rna_train, rna_val])
rna_train_val.var = rna_train.var
rna_train_val.write('rna_train_val.h5ad')

atac_train = sc.read_h5ad('atac_train.h5ad')
atac_val = sc.read_h5ad('atac_val.h5ad')
atac_train_val = ad.concat([atac_train, atac_val])
atac_train_val.var = atac_train.var
atac_train_val.write('atac_train_val.h5ad')


# python h5ad2h5.py -n train_val
# python h5ad2h5.py -n test
import h5py
import numpy as np
from scipy.sparse import csr_matrix, hstack
import argparse

parser = argparse.ArgumentParser(description='concat RNA and ATAC h5ad to h5 format')
parser.add_argument('-n', '--name', type=str, help='sample type or name')

args = parser.parse_args()
sn = args.name
rna  = h5py.File('rna_'+sn+'.h5ad', 'r')
atac = h5py.File('atac_'+sn+'.h5ad', 'r')
out  = h5py.File(sn+'.h5', 'w')

g = out.create_group('matrix')
g.create_dataset('barcodes', data=rna['obs']['_index'][:])
rna_atac_csr_mat = hstack((csr_matrix((rna['X']['data'], rna['X']['indices'],  rna['X']['indptr']), shape=[rna['obs']['_index'].shape[0], rna['var']['_index'].shape[0]]),
                           csr_matrix((atac['X']['data'], atac['X']['indices'],  atac['X']['indptr']), shape=[atac['obs']['_index'].shape[0], atac['var']['_index'].shape[0]])))
rna_atac_csr_mat = rna_atac_csr_mat.tocsr()
g.create_dataset('data', data=rna_atac_csr_mat.data)
g.create_dataset('indices', data=rna_atac_csr_mat.indices)
g.create_dataset('indptr',  data=rna_atac_csr_mat.indptr)
l = list(rna_atac_csr_mat.shape)
l.reverse()
g.create_dataset('shape', data=l)

g_2 = g.create_group('features')
g_2.create_dataset('_all_tag_keys', data=np.array([b'genome', b'interval']))
g_2.create_dataset('feature_type', data=np.append([b'Gene Expression']*rna['var']['gene_ids'].shape[0], [b'Peaks']*atac['var']['gene_ids'].shape[0]))
g_2.create_dataset('genome', data=np.array([b'GRCh38'] * (rna['var']['gene_ids'].shape[0]+atac['var']['gene_ids'].shape[0])))
g_2.create_dataset('id', data=np.append(rna['var']['gene_ids'][:], atac['var']['gene_ids'][:]))
g_2.create_dataset('interval', data=np.append(rna['var']['gene_ids'][:], atac['var']['gene_ids'][:]))
g_2.create_dataset('name', data=np.append(rna['var']['_index'][:], atac['var']['_index'][:]))      

out.close()


nohup python /fs/home/jiluzhang/BABEL/bin/train_model.py --data train_val.h5 --outdir train_out --batchsize 512 --earlystop 25 --device 3 --nofilter > train_20240929.log &
python /fs/home/jiluzhang/BABEL/bin/predict_model.py --checkpoint train_out --data test.h5 --outdir test_out --device 3 --nofilter --noplot --transonly   # 9 min

## match cells bw pred & true
## python match_cell.py
import scanpy as sc

pred = sc.read_h5ad('test_out/rna_atac_adata.h5ad')
true = sc.read_h5ad('atac_test.h5ad')
pred[true.obs.index, :].write('atac_babel.h5ad')

python cal_auroc_auprc.py --pred atac_babel.h5ad --true atac_test.h5ad
# Cell-wise AUROC: 0.8962
# Cell-wise AUPRC: 0.4886
# Peak-wise AUROC: 0.6846
# Peak-wise AUPRC: 0.1377

python plot_save_umap.py --pred atac_babel.h5ad --true atac_test.h5ad  # 8 min
python cal_cluster.py --file atac_babel_umap.h5ad
# AMI: 0.6151
# ARI: 0.3919
# HOM: 0.6176
# NMI: 0.6208





################################ Scenario_2 ################################
#### train model & prediction
## python split_train_val_test.py  # 15 s
import scanpy as sc
import random
import numpy as np

rna = sc.read_h5ad('rna.h5ad')
atac = sc.read_h5ad('atac.h5ad')

# CD4 Naive  1029
# NK         448
# Treg       123
# cDC2       120
# pDC        96
# toal       1816

test_idx = np.argwhere(rna.obs['cell_anno'].isin(['CD4 Naive', 'NK', 'Treg', 'cDC2', 'pDC'])).flatten()
train_val_idx = np.setdiff1d(np.arange(rna.n_obs), test_idx)

random.seed(0)
random.shuffle(train_val_idx)
train_idx = train_val_idx[:int(len(train_val_idx)*0.9)]
val_idx = train_val_idx[int(len(train_val_idx)*0.9):]

rna[train_idx, :].write('rna_train.h5ad')
rna[val_idx, :].write('rna_val.h5ad')
rna[test_idx, :].write('rna_test.h5ad')
atac[train_idx, :].write('atac_train.h5ad')
atac[val_idx, :].write('atac_val.h5ad')
atac[test_idx, :].write('atac_test.h5ad')



nohup accelerate launch --config_file accelerator_config_train.yaml --main_process_port 29823 rna2atac_train.py --config_file rna2atac_config_train_mlt_1_test.yaml \
                        --train_data_dir train_pt_mlt_1 --val_data_dir val_pt -l save_mlt_1_true/2024-09-29_rna2atac_pbmc_238/pytorch_model.bin -s save_mlt_1 \
                        -n rna2atac_pbmc > rna2atac_train_20240929_mlt_1.log &




python data_preprocess.py -r rna_val.h5ad -a atac_val.h5ad -s val_pt --dt val -n val --config rna2atac_config_train_mlt_1.yaml             # 20 s
python data_preprocess.py -r rna_test.h5ad -a atac_test.h5ad -s test_pt --dt test -n test --config rna2atac_config_test.yaml                     # 17 min

### early stop: loss -> auroc (delta:0)  use round to avoid float issue 
### patience: 25 -> 10
### mult=1
python data_preprocess.py -r rna_train.h5ad -a atac_train.h5ad -s train_pt_mlt_1 --dt train -n train --config rna2atac_config_train_mlt_1.yaml   # 1.5 min
nohup accelerate launch --config_file accelerator_config_train.yaml --main_process_port 29823 rna2atac_train.py --config_file rna2atac_config_train_mlt_1.yaml \
                        --train_data_dir train_pt_mlt_1 --val_data_dir val_pt -s save_mlt_1 -n rna2atac_pbmc > rna2atac_train_20240929_mlt_1.log &
accelerate launch --config_file accelerator_config_test.yaml --main_process_port 29822 rna2atac_predict.py -d test_pt \
                  -l save_mlt_1/2024-09-29_rna2atac_pbmc_230/pytorch_model.bin --config_file rna2atac_config_test.yaml  # 3 min
python npy2h5ad.py
python cal_auroc_auprc.py --pred atac_cisformer.h5ad --true atac_test.h5ad
# Cell-wise AUROC: 0.8422
# Cell-wise AUPRC: 0.4101
# Peak-wise AUROC: 0.5694
# Peak-wise AUPRC: 0.1073
python cal_cluster_plot.py --pred atac_cisformer.h5ad --true atac_test.h5ad
# AMI: [0.6077, 0.5867, 0.6388, 0.5979, 0.6018]
# ARI: [0.4406, 0.4068, 0.6307, 0.4496, 0.4594]
# HOM: [0.5801, 0.5577, 0.5834, 0.555, 0.5592]
# NMI: [0.6124, 0.5916, 0.6427, 0.6021, 0.606]













