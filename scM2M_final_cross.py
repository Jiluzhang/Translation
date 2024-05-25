# CM3542C2-T1  CRC  6,831

# Tumor B         ->  Tumor B
# Tumor B + CRC  ->  Tumor B

# /fs/home/jiluzhang/2023_nature_LD/strict_filtered_out/CM354C2-T1_rna_aligned.h5ad
# /fs/home/jiluzhang/2023_nature_LD/strict_filtered_out/CM354C2-T1_atac_aligned.h5ad

######## select genes & peaks ########
import scanpy as sc

tumor_B_train_rna = sc.read_h5ad('rna_tumor_B_train_filtered_0.h5ad')
crc_rna = sc.read_h5ad('/fs/home/jiluzhang/2023_nature_LD/strict_filtered_out/CM354C2-T1_rna_aligned.h5ad')
crc_rna[:, tumor_B_train_rna.var.index].write('crc_rna_0.h5ad')

tumor_B_train_atac = sc.read_h5ad('atac_tumor_B_train_filtered_0.h5ad')
crc_atac = sc.read_h5ad('/fs/home/jiluzhang/2023_nature_LD/strict_filtered_out/CM354C2-T1_atac_aligned.h5ad')
crc_atac[:, tumor_B_train_atac.var.index].write('crc_atac_0.h5ad')


######## preprocess & training & predicting ########
python rna2atac_data_preprocess.py --config_file rna2atac_config.yaml 
accelerate launch --config_file accelerator_config.yaml rna2atac_pretrain.py --config_file rna2atac_config.yaml -d ./preprocessed_data_tumor_B_crc -n rna2atac_tumor_B_crc

# python rna2atac_data_preprocess_whole.py --config_file rna2atac_config_whole.yaml
accelerate launch --config_file accelerator_config.yaml --main_process_port 29821 rna2atac_evaluate.py -d ./preprocessed_data_whole_test_all -l save/2024-05-21_rna2atac_tumor_B_crc_15/pytorch_model.bin --config_file rna2atac_config_whole.yaml

conda create -n snapatac2 python=3.8.13
conda activate snapatac2
#conda install -c conda-forge scanpy python-igraph leidenalg
pip install scanpy -i https://pypi.tuna.tsinghua.edu.cn/simple
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
# restart shell
~/.cargo/config
# [source.crates-io]
# #registry = "https://github.com/rust-lang/crates.io-index"
# replace-with = 'tuna'
# [source.tuna]
# registry = "https://mirrors.tuna.tsinghua.edu.cn/git/crates.io-index.git"
pip install snapatac2 -i https://pypi.tuna.tsinghua.edu.cn/simple --verbose




import numpy as np
import scanpy as sc
import snapatac2 as snap
from scipy.sparse import csr_matrix
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve, auc
from tqdm import tqdm
import random
from sklearn import metrics

m_raw = np.load('precict_from_tumor_B_crc.npy')

m = m_raw.copy()
# [sum(m_raw[i]>0.7) for i in range(5)]   [10467, 12286, 12222, 10728, 10362]
m[m>0.7]=1
m[m<=0.7]=0

atac_true = snap.read('atac_tumor_B_test_filtered_0.h5ad', backed=None)
del atac_true.obsm['X_spectral']
del atac_true.obsm['X_umap']

atac_pred = atac_true.copy()
atac_pred.X = csr_matrix(m)

snap.pp.select_features(atac_pred)
snap.tl.spectral(atac_pred)
snap.tl.umap(atac_pred)
snap.pl.umap(atac_pred, color='cell_anno', show=False, out_file='umap_tumor_B_crc_test_predict_0.7.pdf', marker_size=2.0, height=500)
snap.pp.knn(atac_pred)
snap.tl.leiden(atac_pred)
ARI = metrics.adjusted_rand_score(atac_pred.obs['cell_anno'], atac_pred.obs['leiden'])
AMI = metrics.adjusted_mutual_info_score(atac_pred.obs['cell_anno'], atac_pred.obs['leiden'])
NMI = metrics.normalized_mutual_info_score(atac_pred.obs['cell_anno'], atac_pred.obs['leiden'])
HOM = metrics.homogeneity_score(atac_pred.obs['cell_anno'], atac_pred.obs['leiden'])
print(ARI, AMI, NMI, HOM)
# 0.12570701571524157 0.2603323443875042 0.26147564931848244 0.41597569847177773
atac_pred.write('atac_tumor_B_test_crc_filtered_0_scm2m.h5ad')

atac_pred = np.load('precict_from_tumor_B_crc.npy')
atac_true = snap.read('atac_tumor_B_test_filtered_0.h5ad', backed=None)

random.seed(0)
idx = list(range(atac_true.n_obs))
random.shuffle(idx)
idx = idx[:500]

auprc_lst = []
for i in tqdm(range(500)):
    true_0 = atac_true.X[idx[i]].toarray()[0]
    pred_0 = atac_pred[idx[i]]
    precision, recall, thresholds = precision_recall_curve(true_0, pred_0)
    auprc_lst.append(auc(recall, precision))

np.mean(auprc_lst)  # 0.1815552462891631

auroc_lst = []
for i in tqdm(range(500)):
    true_0 = atac_true.X[idx[i]].toarray()[0]
    pred_0 = atac_pred[idx[i]]
    fpr, tpr, _ = roc_curve(true_0, pred_0)
    auroc_lst.append(auc(fpr, tpr))

np.mean(auroc_lst)  # 0.7667124840123922



## Barplot for benchmark
from plotnine import *
import pandas as pd
import numpy as np

## ARI & AMI & NMI & HOM
raw = pd.read_table('ari_ami_nmi_hom_crc.txt', index_col=0)
dat = pd.DataFrame(raw.values.flatten(), columns=['val'])
dat['dataset'] = ['tumor_B']*4 + ['tumor_B_CRC']*4
dat['evl'] = ['ARI', 'AMI', 'NMI', 'HOM']*2

p = ggplot(dat, aes(x='factor(evl)', y='val', fill='dataset')) + geom_bar(stat='identity', position=position_dodge()) + xlab('') + ylab('') +\
                                                                 scale_y_continuous(limits=[0, 1], breaks=np.arange(0, 1.1, 0.2)) + theme_bw()
p.save(filename='ari_ami_nmi_hom_crc.pdf', dpi=600, height=4, width=5)

## AUROC & AUPRC
raw = pd.read_table('auroc_auprc_crc.txt', index_col=0)
dat = pd.DataFrame(raw.values.flatten(), columns=['val'])
dat['dataset'] = ['tumor_B']*2 + ['tumor_B_CRC']*2
dat['evl'] = ['AUROC', 'AUPRC']*2

p = ggplot(dat, aes(x='factor(evl)', y='val', fill='dataset')) + geom_bar(width=0.5, stat='identity', position=position_dodge()) + xlab('') + ylab('') +\
                                                                 geom_text(aes(label='val'), stat='identity', position=position_dodge2(width=0.5), size=8, va='bottom') +\
                                                                 scale_y_continuous(limits=[0, 1], breaks=np.arange(0, 1.1, 0.2)) + theme_bw()
p.save(filename='auroc_auprc_crc.pdf', dpi=600, height=4, width=5)




## TFDP1 as a modulator of global chromatin accessibility
import scanpy as sc
import numpy as np

rna = sc.read_h5ad('rna_tumor_B_test_filtered_0.h5ad')
max(rna[:, 'TFDP1'].X.toarray())  # 13
np.where(rna[:, 'TFDP1'].X.toarray()==13)  # 2508
rna[2508, :].write('TFDP1_13_rna_0.h5ad')

atac = sc.read_h5ad('atac_tumor_B_test_filtered_0.h5ad')
atac[2508, :].write('TFDP1_13_atac_0.h5ad')

rna_tfdp1 = sc.read_h5ad('TFDP1_13_rna_0.h5ad')
rna_tfdp1[:, 'TFDP1'].X = 0
rna_tfdp1.write('TFDP1_13_rna_ko_0.h5ad')

## for wt
python rna2atac_data_preprocess_whole.py --config_file rna2atac_config_whole.yaml
accelerate launch --config_file accelerator_config.yaml --main_process_port 29821 rna2atac_evaluate.py -d ./preprocessed_data_whole_TFDP1 -l save_tumor_B/2024-05-16_rna2atac_tumor_B_20/pytorch_model.bin --config_file rna2atac_config_whole.yaml

## for ko
python rna2atac_data_preprocess_whole.py --config_file rna2atac_config_whole.yaml
accelerate launch --config_file accelerator_config.yaml --main_process_port 29821 rna2atac_evaluate.py -d ./preprocessed_data_whole_TFDP1_ko -l save_tumor_B/2024-05-16_rna2atac_tumor_B_20/pytorch_model.bin --config_file rna2atac_config_whole.yaml


import scanpy as sc
import numpy as np
import random
from sklearn.metrics import precision_recall_curve, roc_curve, auc

## wt vs. true
atac_pred_wt = np.load('TFDP1_13_predict.npy')
atac_true = sc.read_h5ad('TFDP1_13_atac_0.h5ad')

true_0 = atac_true.X.toarray()[0]
pred_0 = atac_pred_wt[0]
precision, recall, thresholds = precision_recall_curve(true_0, pred_0)
print(auc(recall, precision))   # 0.6888713079340169

true_0 = atac_true.X.toarray()[0]
pred_0 = atac_pred_wt[0]
fpr, tpr, _ = roc_curve(true_0, pred_0)
print(auc(fpr, tpr))  # 0.7706938643616125


## ko vs. true
atac_pred_ko = np.load('TFDP1_13_ko_predict.npy')

true_0 = atac_true.X.toarray()[0]
pred_0 = atac_pred_ko[0]
precision, recall, thresholds = precision_recall_curve(true_0, pred_0)
print(auc(recall, precision))   # 0.6877390955590554

true_0 = atac_true.X.toarray()[0]
pred_0 = atac_pred_ko[0]
fpr, tpr, _ = roc_curve(true_0, pred_0)
print(auc(fpr, tpr))  # 0.7693186238770752


## wt vs. ko
from plotnine import *
import scanpy as sc
import pandas as pd
import numpy as np

atac_pred_wt = np.load('TFDP1_13_predict.npy')
atac_pred_ko = np.load('TFDP1_13_ko_predict.npy')

dat = pd.DataFrame({'wt': atac_pred_wt[0], 'ko': atac_pred_ko[0]})

p = ggplot(dat, aes(x='wt', y='ko')) + geom_point(size=0.02) + theme_bw()
p.save(filename='TFDP1_wt_ko_scatter.png', dpi=100, height=4, width=4)


## plot box
df = pd.DataFrame(dat.values.flatten(), columns=['val'])
df['type'] = ['wt', 'ko']*dat.shape[0]

p = ggplot(df,aes(x='type', y='val', fill='type')) + geom_boxplot(show_legend=False) + theme_bw()
p.save(filename='TFDP1_more_1_wt_ko_box.png', dpi=100, height=4, width=4)

stats.ttest_rel(dat['wt'], dat['ko'])
dat['wt'].mean()  # 0.37247285
dat['ko'].mean()  # 0.36820325




##### select more cell to do in-silico perturbation
import scanpy as sc
import numpy as np

rna = sc.read_h5ad('rna_tumor_B_test_filtered_0.h5ad')
np.unique(rna[:, 'TFDP1'].X.toarray(), return_counts=True)
# array([ 0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 12., 13.])
# array([6202,  808,  181,   46,   21,   10,    5,    2,    3,    3,    1, 1])
# 7283-6202-808=273
idx = np.where(rna[:, 'TFDP1'].X.toarray()>1)[0]
rna[idx, :].write('TFDP1_more_1_rna_0.h5ad')

atac = sc.read_h5ad('atac_tumor_B_test_filtered_0.h5ad')
atac[idx, :].write('TFDP1_more_1_atac_0.h5ad')

rna_tfdp1 = sc.read_h5ad('TFDP1_more_1_rna_0.h5ad')
rna_tfdp1[:, 'TFDP1'].X = 0
rna_tfdp1.write('TFDP1_more_1_rna_ko_0.h5ad')


## for wt
python rna2atac_data_preprocess_whole.py --config_file rna2atac_config_whole.yaml
accelerate launch --config_file accelerator_config.yaml --main_process_port 29821 rna2atac_evaluate.py -d ./preprocessed_data_whole_TFDP1_more_1 -l save_tumor_B/2024-05-16_rna2atac_tumor_B_20/pytorch_model.bin --config_file rna2atac_config_whole.yaml

## for ko
python rna2atac_data_preprocess_whole.py --config_file rna2atac_config_whole.yaml
accelerate launch --config_file accelerator_config.yaml --main_process_port 29821 rna2atac_evaluate.py -d ./preprocessed_data_whole_TFDP1_more_1_ko -l save_tumor_B/2024-05-16_rna2atac_tumor_B_20/pytorch_model.bin --config_file rna2atac_config_whole.yaml


## wt vs. ko
from plotnine import *
import scanpy as sc
import pandas as pd
import numpy as np
from scipy import stats

atac_pred_wt = np.load('TFDP1_more_1_predict.npy')
atac_pred_ko = np.load('TFDP1_more_1_ko_predict.npy')

dat = pd.DataFrame({'wt': atac_pred_wt.mean(axis=0), 'ko': atac_pred_ko.mean(axis=0)})
# dat = pd.DataFrame({'wt': atac_pred_wt[13], 'ko': atac_pred_ko[13]})

p = ggplot(dat, aes(x='wt', y='ko')) + geom_point(size=0.02) + theme_bw()
p.save(filename='TFDP1_more_1_wt_ko_scatter.png', dpi=100, height=4, width=4)


## plot box
dat = pd.DataFrame({'wt': atac_pred_wt.mean(axis=0), 'ko': atac_pred_ko.mean(axis=0)})
df = pd.DataFrame(dat.values.flatten(), columns=['val'])
df['type'] = ['wt', 'ko']*dat.shape[0]

p = ggplot(df,aes(x='type', y='val', fill='type')) + geom_boxplot(show_legend=False) + theme_bw()
p.save(filename='TFDP1_more_1_wt_ko_more_1_box.png', dpi=100, height=4, width=4)

stats.ttest_rel(dat['wt'], dat['ko'])
dat['wt'].mean()  # 0.33196306
dat['ko'].mean()  # 0.33137012



## ReMap2022: https://remap.univ-amu.fr/



############ scButterfly ############
## Github: https://github.com/BioX-NKU/scButterfly
## Tutorial: https://scbutterfly.readthedocs.io/en/latest/
conda create -n scButterfly python==3.9
conda activate scButterfly
pip install scButterfly -i https://pypi.tuna.tsinghua.edu.cn/simple
conda install pytorch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 pytorch-cuda=12.1 -c pytorch -c nvidia




########## pan-cancer datasets ##########
# 1. check attention setting                      √
# 2. modify RNA encoder (similar to ATAC)         √
# 3. use validation dataset to tune parameters    √


## split train & val dataset
# python split_train_val.py

import scanpy as sc
import random

rna = sc.read_h5ad('TFDP1_more_1_rna_0.h5ad')
atac = sc.read_h5ad('TFDP1_more_1_atac_0.h5ad')

random.seed(0)
idx = list(range(rna.n_obs))
random.shuffle(idx)
train_idx = idx[:int(len(idx)*0.8)]
val_idx = idx[int(len(idx)*0.8):]

rna[train_idx, :].write('TFDP1_more_1_rna_train_0.h5ad')
rna[val_idx, :].write('TFDP1_more_1_rna_val_0.h5ad')
atac[train_idx, :].write('TFDP1_more_1_atac_train_0.h5ad')
atac[val_idx, :].write('TFDP1_more_1_atac_val_0.h5ad')


python rna2atac_data_preprocess.py --config_file rna2atac_config.yaml --dataset_type train
python rna2atac_data_preprocess.py --config_file rna2atac_config.yaml --dataset_type val

# Early stopping: https://blog.csdn.net/qq_37430422/article/details/103638681
# https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py  copy this py file to the current fold
#Luz torch.save(model.state_dict(), self.path)


# https://github.com/huggingface/accelerate/blob/main/src/accelerate/accelerator.py
conda create -n scM2M_clone --clone /data/home/zouqihang/miniconda3/envs/m2m
# import accelerate
# accelerate.__file__ 
# '/fs/home/jiluzhang/softwares/miniconda3/envs/scM2M_clone/lib/python3.10/site-packages/accelerate/__init__.py'
# add code to accelerate.py
# # Set a flag tensor for early stopping and other breakpoints
# self.flag_tensor = None
# def set_trigger(self): ...
# def check_trigger(self): ...

accelerate launch --config_file accelerator_config.yaml rna2atac_pretrain.py --config_file rna2atac_config.yaml --train_data_dir ./preprocessed_data_train --val_data_dir ./preprocessed_data_val -n rna2atac_train


python rna2atac_data_preprocess_whole.py --config_file rna2atac_config_whole.yaml
accelerate launch --config_file accelerator_config.yaml --main_process_port 29821 rna2atac_evaluate.py -d ./preprocessed_data_tmp_whole -l save/2024-05-24_rna2atac_tmp_31/pytorch_model.bin --config_file rna2atac_config_whole.yaml



############ pan-cancer datasets testing (only normal cells) ############
# ML123M1-Ty1   SKCM    2,442
# P5504-N1      HNSCC   2,336
# PM581P1-T1N1  PDAC    2,567

# cp /fs/home/jiluzhang/2023_nature_LD/normal_h5ad/ML123M1-Ty1_*.h5ad .
# cp /fs/home/jiluzhang/2023_nature_LD/normal_h5ad/P5504-N1_*.h5ad .
# cp /fs/home/jiluzhang/2023_nature_LD/normal_h5ad/PM581P1-T1_*.h5ad .

## filter genes & peaks
import scanpy as sc
import anndata as ad

## RNA
rna_skcm = sc.read_h5ad('ML123M1-Ty1_rna.h5ad')     # 2442 × 38244
rna_skcm.obs.index = 'ML123M1-Ty1_' + rna_skcm.obs.index
rna_hnscc = sc.read_h5ad('P5504-N1_rna.h5ad')       # 2336 × 38244
rna_hnscc.obs.index = 'P5504-N1_' + rna_hnscc.obs.index

rna = ad.concat([rna_skcm, rna_hnscc], axis=0)   # 4778 × 38244
rna.var = rna_skcm.var
sc.pp.filter_genes(rna, min_cells=10)            # 4778 × 15730
rna.write('skcm_hnscc_rna.h5ad')

rna_pdac = sc.read_h5ad('PM581P1-T1_rna.h5ad')      # 2567 × 38244
rna_pdac.obs.index = 'PM581P1-T1_' + rna_pdac.obs.index
rna_pdac[:, rna.var.index].write('pdac_rna.h5ad')

## ATAC
atac_skcm = sc.read_h5ad('ML123M1-Ty1_atac.h5ad')     # 2442 × 1033239
atac_skcm.obs.index = 'ML123M1-Ty1_' + atac_skcm.obs.index
atac_hnscc = sc.read_h5ad('P5504-N1_atac.h5ad')       # 2336 × 1033239
atac_hnscc.obs.index = 'P5504-N1_' + atac_hnscc.obs.index

atac = ad.concat([atac_skcm, atac_hnscc], axis=0)   # 4778 × 1033239
atac.var = atac_skcm.var
sc.pp.filter_genes(atac, min_cells=10)              # 4778 × 229254
atac.write('skcm_hnscc_atac.h5ad')

atac_pdac = sc.read_h5ad('PM581P1-T1_atac.h5ad')      # 2567 × 1033239
atac_pdac.obs.index = 'PM581P1-T1_' + atac_pdac.obs.index
atac_pdac[:, atac.var.index].write('pdac_atac.h5ad')


python split_train_val.py --RNA skcm_hnscc_rna.h5ad --ATAC skcm_hnscc_atac.h5ad

python rna2atac_data_preprocess.py --config_file rna2atac_config.yaml --dataset_type train  # ~7 min
python rna2atac_data_preprocess.py --config_file rna2atac_config.yaml --dataset_type val    # ~7 min

nohup accelerate launch --config_file accelerator_config.yaml rna2atac_pretrain.py --config_file rna2atac_config.yaml --train_data_dir ./preprocessed_data_train --val_data_dir ./preprocessed_data_val -n rna2atac_train > training_20240525.log &
# 2760530


## split testing dataset (modify 10000 to 1000 in 'rna2atac_data_preprocess_whole.py'!!!!!!!!!!)
# python split_testing.py --RNA pdac_rna.h5ad --ATAC pdac_atac.h5ad
# import scanpy as sc
# import argparse

# parser = argparse.ArgumentParser(description='Split testing dataset every 1000 cells')
# parser.add_argument('--RNA', type=str, help='RNA h5ad file')
# parser.add_argument('--ATAC', type=str, help='ATAC h5ad file')

# args = parser.parse_args()
# rna_file = args.RNA
# atac_file = args.ATAC

# rna = sc.read_h5ad(rna_file)
# atac = sc.read_h5ad(atac_file)

# for i in range(rna.n_obs//1000+1):
#     rna[i*1000:(i+1)*1000, :].write(rna_file.replace('.h5ad', '')+'_'+str(i)+'_0.h5ad')
#     atac[i*1000:(i+1)*1000, :].write(atac_file.replace('.h5ad', '')+'_'+str(i)+'_0.h5ad')


mv pdac_atac.h5ad pdac_atac_0.h5ad
mv pdac_rna.h5ad pdac_rna_0.h5ad

python rna2atac_data_preprocess_whole.py --config_file rna2atac_config_whole.yaml  # ~14 min
accelerate launch --config_file accelerator_config.yaml --main_process_port 29821 rna2atac_evaluate.py -d ./preprocessed_data_test -l save/2024-05-25_rna2atac_train_8/pytorch_model.bin --config_file rna2atac_config_whole.yaml



## calculate metrics
import numpy as np
import scanpy as sc
import snapatac2 as snap
from scipy.sparse import csr_matrix
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve, auc
from tqdm import tqdm
import random
from sklearn import metrics

m_raw = np.load('pdac_predict.npy')

m = m_raw.copy()
# [sum(m_raw[i]>0.7) for i in range(5)]   [10467, 12286, 12222, 10728, 10362]
m[m>0.7]=1
m[m<=0.7]=0

atac_true = snap.read('pdac_atac_0.h5ad', backed=None)
snap.pp.select_features(atac_true)
snap.tl.spectral(atac_true)
snap.tl.umap(atac_true)
snap.pp.knn(atac_true)
snap.tl.leiden(atac_true)
snap.pl.umap(atac_true, color='leiden', show=False, out_file='umap_pdac_true.pdf', marker_size=2.0, height=500)

# del atac_true.obsm['X_spectral']
# del atac_true.obsm['X_umap']

atac_pred = atac_true.copy()
atac_pred.X = csr_matrix(m)

snap.pp.select_features(atac_pred)
snap.tl.spectral(atac_pred)
snap.tl.umap(atac_pred)
snap.pp.knn(atac_pred)
snap.tl.leiden(atac_pred)
snap.pl.umap(atac_pred, color='leiden', show=False, out_file='umap_tumor_B_crc_test_predict_0.7.pdf', marker_size=2.0, height=500)
ARI = metrics.adjusted_rand_score(atac_pred.obs['cell_anno'], atac_pred.obs['leiden'])
AMI = metrics.adjusted_mutual_info_score(atac_pred.obs['cell_anno'], atac_pred.obs['leiden'])
NMI = metrics.normalized_mutual_info_score(atac_pred.obs['cell_anno'], atac_pred.obs['leiden'])
HOM = metrics.homogeneity_score(atac_pred.obs['cell_anno'], atac_pred.obs['leiden'])
print(ARI, AMI, NMI, HOM)
# 0.12570701571524157 0.2603323443875042 0.26147564931848244 0.41597569847177773
atac_pred.write('atac_tumor_B_test_crc_filtered_0_scm2m.h5ad')

atac_pred = np.load('precict_from_tumor_B_crc.npy')
atac_true = snap.read('atac_tumor_B_test_filtered_0.h5ad', backed=None)

random.seed(0)
idx = list(range(atac_true.n_obs))
random.shuffle(idx)
idx = idx[:500]

auprc_lst = []
for i in tqdm(range(500)):
    true_0 = atac_true.X[idx[i]].toarray()[0]
    pred_0 = atac_pred[idx[i]]
    precision, recall, thresholds = precision_recall_curve(true_0, pred_0)
    auprc_lst.append(auc(recall, precision))

np.mean(auprc_lst)  # 0.1815552462891631

auroc_lst = []
for i in tqdm(range(500)):
    true_0 = atac_true.X[idx[i]].toarray()[0]
    pred_0 = atac_pred[idx[i]]
    fpr, tpr, _ = roc_curve(true_0, pred_0)
    auroc_lst.append(auc(fpr, tpr))

np.mean(auroc_lst)  # 0.7667124840123922




