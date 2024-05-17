## rna2atac_config.yaml  rna2atac_data_preprocess.py  rna2atac_evaluate.py  rna2atac_pretrain.py

# pan_cancer_atac_dataset_0.h5ad  head 1000
# pan_cancer_rna_dataset_0.h5ad   head 1000

python rna2atac_data_preprocess.py  # ~6.5 min
accelerate launch --config_file accelerator_config.yaml rna2atac_pretrain.py -d ./preprocessed_data -n rna2atac_test

python rna2atac_data_preprocess.py --config_file rna2atac_config_whole.yaml
accelerate launch --config_file accelerator_config.yaml rna2atac_evaluate.py -d ./preprocessed_data_whole -n rna2atac_test -l save_01/2024-05-06_rna2atac_test/pytorch_model.bin

accelerate launch --config_file accelerator_config.yaml rna2atac_evaluate.py -d ./tmp -n rna2atac_test -l save_01/2024-05-06_rna2atac_test/pytorch_model.bin --config_file rna2atac_config_whole.yaml


rna_tumor_B_train_2_cell_types.h5ad   # 200 × 16428   rna_tumor_B_train_2_cell_types_0.h5ad 
atac_tumor_B_train_2_cell_types.h5ad  # 200 × 181038  atac_tumor_B_train_2_cell_types_0.h5ad 

python rna2atac_data_preprocess.py --config_file rna2atac_config.yaml # ~4.5 min

accelerate launch --config_file accelerator_config.yaml rna2atac_pretrain.py --config_file rna2atac_config.yaml -d ./preprocessed_data_10 -n rna2atac_test 
# accelerate launch --config_file accelerator_config.yaml rna2atac_pretrain.py --config_file rna2atac_config.yaml -d ./preprocessed_data_10 -n rna2atac_test -l save_01/2024-05-09_rna2atac_test_100/pytorch_model.bin


# batch_size=16  epoch: 30     loss: 0.6794     auroc: 0.5917     auprc: 0.5875
# batch_size=8   epoch: 30     loss: 0.68     auroc: 0.5887     auprc: 0.5856
# batch_size=8   epoch: 100     loss: 0.642     auroc: 0.6763     auprc: 0.6718

python rna2atac_data_preprocess_whole.py --config_file rna2atac_config_whole.yaml
accelerate launch --config_file accelerator_config.yaml rna2atac_evaluate.py -d ./preprocessed_data_whole -l save_01/2024-05-09_rna2atac_test_30/pytorch_model.bin --config_file rna2atac_config_whole.yaml
accelerate launch --config_file accelerator_config.yaml rna2atac_evaluate.py -d ./preprocessed_data_whole -l save/2024-05-09_rna2atac_test_30/pytorch_model.bin --config_file rna2atac_config_whole.yaml







import numpy as np
import scanpy as sc
import snapatac2 as snap
from scipy.sparse import csr_matrix
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve, auc

m_raw = np.load('predict_epoch_90.npy')
#m = ((m_raw.T > m_raw.T.mean(axis=0)).T) & (m_raw>m_raw.mean(axis=0)).astype(int)

m = m_raw.copy()
m[m>0.6]=1
m[m<=0.6]=0

atac_true = snap.read('atac_tumor_B_train_2_cell_types_0.h5ad', backed=None)
del atac_true.obsm['X_spectral']
del atac_true.obsm['X_umap']

atac_pred = atac_true.copy()
atac_pred.X = csr_matrix(m)
#atac_pred = atac_pred[atac_pred.obs['cell_anno'].isin(['T cell', 'Tumor B cell']), :].copy()

snap.pp.select_features(atac_pred)#, n_features=50000)
snap.tl.spectral(atac_pred) #snap.tl.spectral(atac_pred, n_comps=50)
snap.tl.umap(atac_pred)
snap.pl.umap(atac_pred, color='cell_anno', show=False, out_file='umap_tumor_B_predict_epoch_90.pdf', marker_size=2.5, height=500)

# snap.pp.select_features(atac_true)#, n_features=50000)
# snap.tl.spectral(atac_true) #snap.tl.spectral(atac_pred, n_comps=50)
# snap.tl.umap(atac_true)
# snap.pl.umap(atac_true, color='cell_anno', show=False, out_file='umap_tumor_B_true.pdf', marker_size=2.5, height=500)



atac_pred = np.load('predict_epoch_90.npy')
atac_true = snap.read('atac_tumor_B_train_2_cell_types_0.h5ad', backed=None)

auprc_lst = []
for i in range(200):
    true_0 = atac_true.X[i].toarray()[0]
    pred_0 = atac_pred[i]
    precision, recall, thresholds = precision_recall_curve(true_0, pred_0)
    auprc_lst.append(auc(recall, precision))
    #print(i, atac_pred.obs['cell_anno'][i], auc(recall, precision))

np.mean(auprc_lst)

# epoch=20    0.10788303370478967
# epoch=40    0.11283707229267012
# epoch=60    0.12317800018187781
# epoch=80    0.1420146195666489
# epoch=100   0.15332182973640157

auroc_lst = []
for i in range(200):
    true_0 = atac_true.X[i].toarray()[0]
    pred_0 = atac_pred[i]
    fpr, tpr, _ = roc_curve(true_0, pred_0)
    auroc_lst.append(auc(fpr, tpr))

np.mean(auroc_lst)
# epoch=20    0.5880639196985943
# epoch=40    0.6012055913199156
# epoch=60    0.6270619083846645
# epoch=80    0.6702170733195197
# epoch=100   0.6885955191454858


############# Tumor B datasets ###############
import scanpy as sc

rna_train = sc.read_h5ad('rna_tumor_B_train.h5ad')  # 7283 × 38244
sc.pp.filter_genes(rna_train, min_cells=10)         # 7283 × 16428
rna_train.write('rna_tumor_B_train_filtered.h5ad')
rna_test = sc.read_h5ad('rna_tumor_B_test.h5ad')
rna_test[:, rna_train.var.index].write('rna_tumor_B_test_filtered.h5ad')

atac_train = sc.read_h5ad('atac_tumor_B_train.h5ad')  # 7283 × 1033239
sc.pp.filter_genes(atac_train, min_cells=10)          # 7283 × 181038
atac_train.write('atac_tumor_B_train_filtered.h5ad')
atac_test = sc.read_h5ad('atac_tumor_B_test.h5ad')
atac_test[:, atac_train.var.index].write('atac_tumor_B_test_filtered.h5ad')



python rna2atac_data_preprocess.py --config_file rna2atac_config.yaml # ~ 30 min

accelerate launch --config_file accelerator_config.yaml rna2atac_pretrain.py --config_file rna2atac_config.yaml -d ./preprocessed_data -n rna2atac_tumor_B

python rna2atac_data_preprocess_whole.py --config_file rna2atac_config_whole.yaml  # 500 cells in 3 min
accelerate launch --config_file accelerator_config.yaml --main_process_port 29821 rna2atac_evaluate.py -d ./preprocessed_data_whole -l save/2024-05-15_rna2atac_tumor_B_1/pytorch_model.bin --config_file rna2atac_config_whole.yaml
accelerate launch --config_file accelerator_config.yaml --main_process_port 29821 rna2atac_evaluate.py -d ./preprocessed_data_whole -l save/2024-05-15_rna2atac_tumor_B_2/pytorch_model.bin --config_file rna2atac_config_whole.yaml

# accelerator.gather() need same batch size for each GPU!!!

accelerate launch --config_file accelerator_config.yaml --main_process_port 29821 rna2atac_evaluate.py -d ./preprocessed_data_whole_test -l save/2024-05-15_rna2atac_tumor_B_6/pytorch_model.bin --config_file rna2atac_config_whole.yaml



import numpy as np
import scanpy as sc
import snapatac2 as snap
from scipy.sparse import csr_matrix
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve, auc
from tqdm import tqdm

m_raw = np.load('predict.npy')

m = m_raw.copy()
m[m>0.5]=1
m[m<=0.5]=0

atac_true = snap.read('atac_tumor_B_test_filtered_0.h5ad', backed=None)  # atac_true = snap.read('atac_tumor_B_test_filtered_500_0.h5ad', backed=None)
del atac_true.obsm['X_spectral']
del atac_true.obsm['X_umap']

atac_pred = atac_true.copy()
atac_pred.X = csr_matrix(m)
#atac_pred = atac_pred[atac_pred.obs['cell_anno'].isin(['T cell', 'Tumor B cell']), :].copy()

snap.pp.select_features(atac_pred)#, n_features=50000)
snap.tl.spectral(atac_pred) #snap.tl.spectral(atac_pred, n_comps=50)
snap.tl.umap(atac_pred)
snap.pl.umap(atac_pred, color='cell_anno', show=False, out_file='umap_tumor_B_test_predict.pdf', marker_size=2.5, height=500) # snap.pl.umap(atac_pred, color='cell_anno', show=False, out_file='umap_tumor_B_predict_epoch_10.pdf', marker_size=2.5, height=500)

# snap.pp.select_features(atac_true)#, n_features=50000)
# snap.tl.spectral(atac_true) #snap.tl.spectral(atac_pred, n_comps=50)
# snap.tl.umap(atac_true)
# snap.pl.umap(atac_true, color='cell_anno', show=False, out_file='umap_tumor_B_test_true.pdf', marker_size=2.5, height=500)



atac_pred = np.load('predict.npy')
atac_true = snap.read('atac_tumor_B_test_filtered_0.h5ad', backed=None) # atac_true = snap.read('atac_tumor_B_test_filtered_500_0.h5ad', backed=None)

## maybe sampling is a better choice
import random

random.seed(0)
idx = list(range(atac_true.n_obs))
random.shuffle(idx)
idx = idx[:500]

auprc_lst = []
for i in tqdm(range(500)):
    true_0 = atac_true.X[i].toarray()[0]
    pred_0 = atac_pred[i]
    precision, recall, thresholds = precision_recall_curve(true_0, pred_0)
    auprc_lst.append(auc(recall, precision))

np.mean(auprc_lst)  # 0.19324982160086668

auroc_lst = []
for i in tqdm(range(500)):
    true_0 = atac_true.X[i].toarray()[0]
    pred_0 = atac_pred[i]
    fpr, tpr, _ = roc_curve(true_0, pred_0)
    auroc_lst.append(auc(fpr, tpr))

np.mean(auroc_lst)  # 0.7992866943383289



############### enc_len_2048_dec_len_2048 ###############
# enc_max_len: 2048
# dec_max_len: 2048
# mutiple_rate: 10

python rna2atac_data_preprocess.py --config_file rna2atac_config.yaml
accelerate launch --config_file accelerator_config.yaml rna2atac_pretrain.py --config_file rna2atac_config.yaml -d ./preprocessed_data -n rna2atac_tumor_B


python rna2atac_data_preprocess_whole.py --config_file rna2atac_config_whole.yaml
accelerate launch --config_file accelerator_config.yaml --main_process_port 29821 rna2atac_evaluate.py -d ./preprocessed_data_whole -l save/2024-05-16_rna2atac_tumor_B_9/pytorch_model.bin --config_file rna2atac_config_whole.yaml

accelerate launch --config_file accelerator_config.yaml --main_process_port 29821 rna2atac_evaluate.py -d ./preprocessed_data_whole -l save/2024-05-16_rna2atac_tumor_B_39/pytorch_model.bin --config_file rna2atac_config_whole.yaml


# np.delete(atac_pred.obsm['X_umap'], -3, axis=1)


########## seperate h5ad files ##########
import scanpy as sc

rna = sc.read_h5ad('rna_tumor_B_test_filtered_0.h5ad')
atac = sc.read_h5ad('atac_tumor_B_test_filtered_0.h5ad')

cnt = rna.n_obs//1000+1
for i in range(cnt):
    if i!=cnt:
        rna[i*1000:(i+1)*1000, :].write('rna_tumor_B_test_filtered_'+str(i)+'_0.h5ad')
        atac[i*1000:(i+1)*1000, :].write('atac_tumor_B_test_filtered_'+str(i)+'_0.h5ad')
    else:
        rna[i*1000:, :].write('rna_tumor_B_test_filtered_'+str(i)+'_0.h5ad')
        atac[i*1000:, :].write('atac_tumor_B_test_filtered_'+str(i)+'_0.h5ad')
    print((i+1)*1000, 'cells done')








