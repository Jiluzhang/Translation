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
sc.pp.filter_genes(rna, min_cells=10)
sc.pp.filter_cells(rna, min_genes=200)
sc.pp.filter_cells(rna, max_genes=20000)    # 9964 × 21000
rna.obs['n_genes'].max()      # 3766

## atac
atac = dat[:, dat.var['feature_types']=='Peaks'].copy()  # 9964 × 143887
atac.X[atac.X>0] = 1
sc.pp.filter_genes(atac, min_cells=10)
sc.pp.filter_cells(atac, min_genes=500)
sc.pp.filter_cells(atac, max_genes=50000)  # 9964 × 143856

## output rna & atac (no cells are filtered)
rna.write('rna.h5ad')
atac.write('atac.h5ad')

#### train model & prediction
python split_train_val_test.py --RNA rna.h5ad --ATAC atac.h5ad --train_pct 0.7 --valid_pct 0.1
python data_preprocess.py -r rna_train.h5ad -a atac_train.h5ad -s train_pt --dt train -n train --config rna2atac_config_train.yaml   # 3.5 min
python data_preprocess.py -r rna_val.h5ad -a atac_val.h5ad -s val_pt --dt val -n val --config rna2atac_config_train.yaml             # 30 s
python data_preprocess.py -r rna_test.h5ad -a atac_test.h5ad -s test_pt --dt test -n test --config rna2atac_config_test.yaml         # 3 min
nohup accelerate launch --config_file accelerator_config_train.yaml --main_process_port 29823 rna2atac_train.py --config_file rna2atac_config_train.yaml \
                        --train_data_dir train_pt --val_data_dir val_pt -n rna2atac_pbmc > rna2atac_train_20240927.log &
accelerate launch --config_file accelerator_config_test.yaml --main_process_port 29822 rna2atac_predict.py -d test_pt \
                      -l save/2024-09-27_rna2atac_pbmc_50/pytorch_model.bin --config_file rna2atac_config_test.yaml
python npy2h5ad.py

python cal_auroc_auprc.py --pred atac_cisformer.h5ad --true atac_test.h5ad
# Cell-wise AUROC: 0.88
# Cell-wise AUPRC: 0.4111
# Peak-wise AUROC: 0.6238
# Peak-wise AUPRC: 0.0807

python cal_cluster_plot.py --pred atac_cisformer.h5ad --true atac_test.h5ad
# AMI: [0.6753, 0.6777, 0.6769, 0.6701, 0.674]
# ARI: [0.4908, 0.5001, 0.4993, 0.4984, 0.5277]
# HOM: [0.6665, 0.669, 0.6685, 0.6478, 0.6505]
# NMI: [0.6802, 0.6826, 0.6818, 0.6745, 0.6784]



#### scbutterfly
## python scbt_b.py
## nohup python scbt_b.py > scbt_b_20240927.log &
import os
from scButterfly.butterfly import Butterfly
from scButterfly.split_datasets import *
import scanpy as sc
import anndata as ad
import random

os.environ["CUDA_VISIBLE_DEVICES"] = "7"

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


#### BABEL
## concat train & valid dataset
## concat_train_val.py
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


nohup python /fs/home/jiluzhang/BABEL/bin/train_model.py --data train_val.h5 --outdir train_out --batchsize 512 --earlystop 25 --device 2 --nofilter > train_20240927.log &
python /fs/home/jiluzhang/BABEL/bin/predict_model.py --checkpoint train_out --data test.h5 --outdir test_out --device 2 --nofilter --noplot --transonly

## match cells bw pred & true
## match_cell.py
import scanpy as sc

pred = sc.read_h5ad('test_out/rna_atac_adata.h5ad')
true = sc.read_h5ad('atac_test.h5ad')
pred[true.obs.index, :].write('atac_babel.h5ad')


####################################### gene and peak number decrease!!!! ####################################################

python cal_auroc_auprc.py --pred atac_babel.h5ad --true atac_test.h5ad
# Cell-wise AUROC: 0.88
# Cell-wise AUPRC: 0.4111
# Peak-wise AUROC: 0.6238
# Peak-wise AUPRC: 0.0807

python cal_cluster_plot.py --pred atac_cisformer.h5ad --true atac_test.h5ad
# AMI: [0.6753, 0.6777, 0.6769, 0.6701, 0.674]
# ARI: [0.4908, 0.5001, 0.4993, 0.4984, 0.5277]
# HOM: [0.6665, 0.669, 0.6685, 0.6478, 0.6505]
# NMI: [0.6802, 0.6826, 0.6818, 0.6745, 0.6784]



















