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
import random
from sklearn import metrics

m_raw = np.load('predict.npy')

m = m_raw.copy()
# [sum(m_raw[i]>0.7) for i in range(5)]   [13446, 9937, 8868, 14362, 11070]
m[m>0.7]=1
m[m<=0.7]=0

atac_true = snap.read('atac_tumor_B_test_filtered_0.h5ad', backed=None)  # atac_true = snap.read('atac_tumor_B_test_filtered_500_0.h5ad', backed=None)
del atac_true.obsm['X_spectral']
del atac_true.obsm['X_umap']

atac_pred = atac_true.copy()
atac_pred.X = csr_matrix(m)
#atac_pred = atac_pred[atac_pred.obs['cell_anno'].isin(['T cell', 'Tumor B cell']), :].copy()

snap.pp.select_features(atac_pred)#, n_features=50000)
snap.tl.spectral(atac_pred) #snap.tl.spectral(atac_pred, n_comps=50)
snap.tl.umap(atac_pred)
snap.pl.umap(atac_pred, color='cell_anno', show=False, out_file='umap_tumor_B_test_predict_0.7.pdf', marker_size=2.0, height=500) # snap.pl.umap(atac_pred, color='cell_anno', show=False, out_file='umap_tumor_B_predict_epoch_10.pdf', marker_size=2.5, height=500)
snap.pp.knn(atac_pred)
snap.tl.leiden(atac_pred)
ARI = metrics.adjusted_rand_score(atac_pred.obs['cell_anno'], atac_pred.obs['leiden'])
AMI = metrics.adjusted_mutual_info_score(atac_pred.obs['cell_anno'], atac_pred.obs['leiden'])
NMI = metrics.normalized_mutual_info_score(atac_pred.obs['cell_anno'], atac_pred.obs['leiden'])
HOM = metrics.homogeneity_score(atac_pred.obs['cell_anno'], atac_pred.obs['leiden'])
print(ARI, AMI, NMI, HOM)
# 0.17911352210172168 0.4008356259524111 0.4017551710754636 0.6430320591974056
atac_pred.write('atac_tumor_B_test_filtered_0_scm2m.h5ad')

snap.pp.select_features(atac_true)#, n_features=50000)
snap.tl.spectral(atac_true) #snap.tl.spectral(atac_pred, n_comps=50)
snap.tl.umap(atac_true)
snap.pl.umap(atac_true, color='cell_anno', show=False, out_file='umap_tumor_B_test_true.pdf', marker_size=2.0, height=500)
snap.pp.knn(atac_true)
snap.tl.leiden(atac_true)
ARI = metrics.adjusted_rand_score(atac_true.obs['cell_anno'], atac_true.obs['leiden'])
AMI = metrics.adjusted_mutual_info_score(atac_true.obs['cell_anno'], atac_true.obs['leiden'])
NMI = metrics.normalized_mutual_info_score(atac_true.obs['cell_anno'], atac_true.obs['leiden'])
HOM = metrics.homogeneity_score(atac_true.obs['cell_anno'], atac_true.obs['leiden'])
print(ARI, AMI, NMI, HOM)
# 0.31085714396263286 0.5311110081584016 0.5317297992626187 0.770295425885568
atac_true.write('atac_tumor_B_test_filtered_0_processed.h5ad')


atac_pred = np.load('predict.npy')
atac_true = snap.read('atac_tumor_B_test_filtered_0.h5ad', backed=None) # atac_true = snap.read('atac_tumor_B_test_filtered_500_0.h5ad', backed=None)

## maybe sampling is a better choice
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

np.mean(auprc_lst)  # 0.19416483373492507

auroc_lst = []
for i in tqdm(range(500)):
    true_0 = atac_true.X[idx[i]].toarray()[0]
    pred_0 = atac_pred[idx[i]]
    fpr, tpr, _ = roc_curve(true_0, pred_0)
    auroc_lst.append(auc(fpr, tpr))

np.mean(auroc_lst)  # 0.7954613133117091



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



##################### h5ad to h5 (for both scRNA-seq and scATAC-seq)#############################
## h5ad_to_h5.py
import h5py
import numpy as np
from scipy.sparse import csr_matrix, hstack

rna  = h5py.File('rna_tumor_B_train_filtered_0.h5ad', 'r')
atac = h5py.File('atac_tumor_B_train_filtered_0.h5ad', 'r')
out  = h5py.File('rna_atac_tumor_B_train_filtered_0.h5', 'w')

g = out.create_group('matrix')
g.create_dataset('barcodes', data=rna['obs']['_index'][:])
g.create_dataset('data', data=np.append(rna['X']['data'][:], atac['X']['data'][:]))

g_2 = g.create_group('features')
g_2.create_dataset('_all_tag_keys', data=np.array([b'genome', b'interval']))
g_2.create_dataset('feature_type', data=np.append([b'Gene Expression']*rna['var']['_index'].shape[0], [b'Peaks']*atac['var']['_index'].shape[0]))
g_2.create_dataset('genome', data=np.array([b'GRCh38'] * (rna['var']['_index'].shape[0]+atac['var']['_index'].shape[0])))
g_2.create_dataset('id', data=np.append(rna['var']['_index'][:], atac['var']['_index'][:]))        # gene names for ENSMBLE ID????
g_2.create_dataset('interval', data=np.append(rna['var']['_index'][:], atac['var']['_index'][:]))  # genes for scRNA-seq????
g_2.create_dataset('name', data=np.append(rna['var']['_index'][:], atac['var']['_index'][:]))      

## https://blog.csdn.net/m0_64204369/article/details/123035598
## https://blog.csdn.net/HHTNAN/article/details/79790370
## matrix: cell*feature  shape: [feature, cell]  # confused!!!
rna_atac_csr_mat = hstack((csr_matrix((rna['X']['data'], rna['X']['indices'],  rna['X']['indptr']), shape=[rna['obs']['_index'].shape[0], rna['var']['_index'].shape[0]]),
                           csr_matrix((atac['X']['data'], atac['X']['indices'],  atac['X']['indptr']), shape=[atac['obs']['index'].shape[0], atac['var']['_index'].shape[0]])))

g.create_dataset('indices', data=rna_atac_csr_mat.indices)
g.create_dataset('indptr',  data=rna_atac_csr_mat.indptr)

l = list(rna_atac_csr_mat.shape)
l.reverse()
g.create_dataset('shape', data=l)

out.close()


time python /fs/home/jiluzhang/scMOG/scMOG_code/bin/Preprocessing.py --data rna_atac_tumor_B_train_filtered_0.h5 --outdir scMOG_tumor_B_train --nofilter  # ~4.5 min
time python /fs/home/jiluzhang/scMOG/scMOG_code/bin/train.py --outdir training_out  # ~6.5 min   # modify the train.py to not plot
cp ../../rna_tumor_B_test_filtered_0.h5ad truth_rna.h5ad
cp ../../atac_tumor_B_test_filtered_0.h5ad truth_atac.h5ad
time python /fs/home/jiluzhang/scMOG/scMOG_code/bin/predict-atac.py --outdir predict_atac_out  # ~30 s   ## not save sparse matrix to reduce h5ad size (8.5min -> 30s)



import numpy as np
import scanpy as sc
import snapatac2 as snap
from scipy.sparse import csr_matrix
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve, auc
from tqdm import tqdm
import random
from sklearn import metrics

atac_pred = snap.read('rna_atac_adata_final.h5ad', backed=None)
atac_true = snap.read('../truth_atac.h5ad', backed=None) 

random.seed(0)
idx = list(range(atac_true.n_obs))
random.shuffle(idx)
idx = idx[:500]

auprc_lst = []
for i in tqdm(range(500)):
    true_0 = atac_true.X[idx[i]].toarray()[0]
    pred_0 = atac_pred.X[idx[i]]
    precision, recall, thresholds = precision_recall_curve(true_0, pred_0)
    auprc_lst.append(auc(recall, precision))

np.mean(auprc_lst)  # 0.20521846330200294

auroc_lst = []
for i in tqdm(range(500)):
    true_0 = atac_true.X[idx[i]].toarray()[0]
    pred_0 = atac_pred.X[idx[i]]
    fpr, tpr, _ = roc_curve(true_0, pred_0)
    auroc_lst.append(auc(fpr, tpr))

np.mean(auroc_lst)  # 0.7951432043697237


m_raw = snap.read('rna_atac_adata_final.h5ad', backed=None).X
# [sum(m_raw[i]>0.7) for i in range(5)]   [10, 114627, 26879, 8333, 31838]
m = m_raw.copy()
m[m>0.7]=1
m[m<=0.7]=0

atac_true = snap.read('../truth_atac.h5ad', backed=None)
del atac_true.obsm['X_spectral']
del atac_true.obsm['X_umap']

atac_pred = atac_true.copy()
atac_pred.X = csr_matrix(m)

snap.pp.select_features(atac_pred)#, n_features=50000)
snap.tl.spectral(atac_pred) #snap.tl.spectral(atac_pred, n_comps=50)  # time-consuming
snap.tl.umap(atac_pred)
snap.pl.umap(atac_pred, color='cell_anno', show=False, out_file='umap_tumor_B_test_predict_scmog.pdf', marker_size=2.0, height=500)
snap.pp.knn(atac_pred)
snap.tl.leiden(atac_pred)
ARI = metrics.adjusted_rand_score(atac_pred.obs['cell_anno'], atac_pred.obs['leiden'])
AMI = metrics.adjusted_mutual_info_score(atac_pred.obs['cell_anno'], atac_pred.obs['leiden'])
NMI = metrics.normalized_mutual_info_score(atac_pred.obs['cell_anno'], atac_pred.obs['leiden'])
HOM = metrics.homogeneity_score(atac_pred.obs['cell_anno'], atac_pred.obs['leiden'])
print(ARI, AMI, NMI, HOM)
# 0.0992703358136857 0.12174162903285979 0.12411650867559945 0.21776862222843424
atac_pred.write('atac_tumor_B_test_filtered_0_scmog.h5ad')




# atac.X.astype(np.float16)

## Barplot for benchmark
from plotnine import *
import pandas as pd
import numpy as np

## RAW
raw = pd.read_table('ari_ami_nmi_hom.txt', index_col=0)
dat = pd.DataFrame(raw.values.flatten(), columns=['val'])
dat['alg'] = ['Raw']*4 + ['scM2M']*4 + ['scMOG']*4
dat['evl'] = ['ARI', 'AMI', 'NMI', 'HOM']*3

dat.columns = ['count']
dat['idx'] = 'RAW'
np.median(dat['count'])  # 710663.0
p = ggplot(dat, aes(x='idx', y='count')) + geom_bar(width=0.5, color='black', fill='brown', outlier_shape='') + xlab('') + \
                                           scale_y_continuous(limits=[600000, 800000], breaks=range(600000, 800000+1, 50000)) + theme_bw()
p.save(filename='cells_all.pdf', dpi=600, height=5, width=4)

q5 = (ggplot(ToothGrowth_summary, aes(x='factor(dose)', y='len', fill='supp')) + geom_bar(stat="identity", position=position_dodge()) +theme_classic()+
geom_errorbar(ToothGrowth_summary,aes(ymin='ymin', ymax='ymax'), width=.2, position=position_dodge(.9))+
theme(legend_position=(0.3,0.8))+theme(axis_text=element_text(size=14,color='black',family='sans-serif'),axis_title=element_text(size=14,color='black',family='sans-serif'),legend_text= element_text(size=14,color='black',family='sans-serif'))+labs(x='dose (mg)',y='length (cm)',fill='')+theme(legend_background=element_rect(alpha=0))+scale_y_continuous(expand=(0, 0, .1, 0))+labs(fill='')+scale_fill_manual(values=['#D45E1A', '#0F73B0'])
)
q5.save('Tooth_boxploterrorbar.pdf',width=4, height=5.5)






























