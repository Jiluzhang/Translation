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





