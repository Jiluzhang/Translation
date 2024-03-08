## Flash-Frozen Lymph Node with B Cell Lymphoma

# wget -c https://cf.10xgenomics.com/samples/cell-arc/2.0.0/lymph_node_lymphoma_14k/lymph_node_lymphoma_14k_filtered_feature_bc_matrix.h5

#################### Preprossing ####################
import scanpy as sc
import pandas as pd
import numpy as np
import pybedtools
from tqdm import tqdm
from scipy.sparse import csr_matrix

dat = sc.read_10x_h5('/fs/home/jiluzhang/10x_tumor_B/lymph_node_lymphoma_14k_filtered_feature_bc_matrix.h5', gex_only=False)

########## rna ##########
rna = dat[:, dat.var['feature_types']=='Gene Expression'].copy()
rna.X = rna.X.toarray()
genes = pd.read_table('human_genes.txt', names=['gene_id', 'gene_name'])

rna_genes = rna.var.copy()
rna_genes['gene_id'] = rna_genes['gene_ids'].map(lambda x: x.split('.')[0])  # delete ensembl id with version number
rna_exp = pd.concat([rna_genes, pd.DataFrame(rna.X.T, index=rna_genes.index)], axis=1)

X_new = pd.merge(genes, rna_exp, how='left', on='gene_id').iloc[:, 5:].T  # X_new = pd.merge(genes, rna_exp, how='left', left_on='gene_name', right_on='gene_id').iloc[:, 6:].T
X_new = np.array(X_new, dtype='float32')
X_new[np.isnan(X_new)] = 0

rna_new_var = pd.DataFrame({'gene_ids': genes['gene_id'], 'feature_types': 'Gene Expression'})
rna_new = sc.AnnData(csr_matrix(X_new), obs=rna.obs, var=rna_new_var)

rna_new.var.index = genes['gene_name'].values  # set index for var

rna_new.write('rna_tumor_B.h5ad')


########## atac ##########
atac = dat[:, dat.var['feature_types']=='Peaks'].copy()
atac.X = atac.X.toarray()
atac.X[atac.X>1] = 1
atac.X[atac.X<1] = 0

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
for i in tqdm(range(atac.X.shape[0])):
    m[i][idx_map[idx_map['peaks_idx'].isin(peaks.iloc[np.nonzero(atac.X[i])]['idx'])]['cCREs_idx']] = 1

atac_new_var = pd.DataFrame({'gene_ids': cCREs['chr'] + ':' + cCREs['start'].map(str) + '-' + cCREs['end'].map(str), 'feature_types': 'Peaks'})
atac_new = sc.AnnData(csr_matrix(m), obs=atac.obs, var=atac_new_var)

atac_new.var.index = atac_new.var['gene_ids'].values

atac_new.write('atac_tumor_B.h5ad')


#################### Prediction ####################
import sys
import os
sys.path.append("M2Mmodel")
import torch
from M2M import M2M_rna2atac
from utils import *
import scanpy as sc
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix

test_atac_file = '/fs/home/jiluzhang/10x_tumor_B/atac_tumor_B.h5ad'
test_rna_file = '/fs/home/jiluzhang/10x_tumor_B/rna_tumor_B.h5ad'

enc_max_len = 38244
dec_max_len = 1033239

batch_size = 1
rna_max_value = 255
attn_window_size = 2*2048

device = torch.device("cuda:0")

test_atac = sc.read_h5ad(test_atac_file)
test_atac.X = test_atac.X.toarray()
test_rna = sc.read_h5ad(test_rna_file)
test_rna.X = test_rna.X.toarray()
test_rna.X = np.round(test_rna.X/(test_rna.X.sum(axis=1, keepdims=True))*10000)
test_rna.X[test_rna.X>255] = 255

# parameters for dataloader construction
test_kwargs = {'batch_size': batch_size, 'shuffle': False}  # test_kwargs = {'batch_size': batch_size, 'shuffle': True}
cuda_kwargs = {
    # 'pin_memory': True,  # 将加载的数据张量复制到 CUDA 设备的固定内存中，提高效率但会消耗更多内存
    "num_workers": 4,
    'prefetch_factor': 2
}  
test_kwargs.update(cuda_kwargs)

test_dataset = FullLenPairDataset(test_rna.X, test_atac.X, rna_max_value)
test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)

model = M2M_rna2atac(
            dim = 128, #Luz 16,
            enc_num_tokens = rna_max_value + 1,
            enc_seq_len = enc_max_len,
            enc_depth = 2,
            enc_heads = 4,
            enc_attn_window_size = attn_window_size,
            enc_dilation_growth_rate = 2,
            enc_ff_mult = 4,

            latent_len = 256, #Luz 256, 
            num_middle_MLP = 1, #Luz 2, 
            latent_dropout = 0.1, #Luz 0.1,

            dec_seq_len = dec_max_len,
            dec_depth = 2,
            dec_heads = 8,
            dec_attn_window_size = attn_window_size,
            dec_dilation_growth_rate = 5,
            dec_ff_mult = 4
)

model.load_state_dict(torch.load('/fs/home/jiluzhang/scM2M_v2/models_32124_lr_0.00001_epoch_3/32124_cells_10000_epoch_1/pytorch_model.bin'))
model.to(device)

model.eval()
y_hat_out = torch.Tensor().cuda()
with torch.no_grad():
  data_iter = enumerate(test_loader)
  for i, (src, tgt) in data_iter:
    src = src.long().to(device)
    logist = model(src)
    y_hat_out = torch.cat((y_hat_out, torch.sigmoid(logist)))
    
    if 2*(i+1)%1000==0 or 2*(i+1)==test_atac.n_obs:
      np.save('atac_tumor_B_pred_'+str(2*(i+1))+'.npy', y_hat_out.to('cpu').numpy())
      y_hat_out = torch.Tensor().cuda()
    print(2*(i+1), 'cells done', flush=True)


#atac_res = sc.AnnData(np.zeros((test_atac.n_obs, test_atac.n_vars)), obs=test_atac.obs, var=test_atac.var)
#atac_res.X = y_hat_out.to('cpu').numpy()
#atac_res.write('atac_tumor_B_pred.h5ad')


################ True & Pred atac files ################
import numpy as np
import scanpy as sc

# need large memory
# out = np.load('atac_tumor_B_pred_1000.npy')
# for i in (list(range(2000, 14566, 1000)) + [14566]):
#     m = np.load('atac_tumor_B_pred_'+str(i)+'.npy')
#     out = np.vstack((out, m))
#     print('matrix', i, 'done')

# randomly select 100 cells (head 100)
true_atac = sc.read_h5ad('atac_tumor_B.h5ad')
true_atac[:100, :].write('atac_tumor_B_true_100.h5ad')
pred_atac_X = np.load('atac_tumor_B_pred_1000.npy')[:100, ]
pred_atac = sc.AnnData(pred_atac_X, obs=true_atac[:100, :].obs, var=true_atac[:100, :].var)
pred_atac.write('atac_tumor_B_pred_100.h5ad')

# randomly select 100 cells (tail 100)
true_atac = sc.read_h5ad('atac_tumor_B.h5ad')
true_atac[-100:, :].write('atac_tumor_B_true_tail_100.h5ad')
pred_atac_X = np.load('atac_tumor_B_pred_14566.npy')[-100:, ]
pred_atac = sc.AnnData(pred_atac_X, obs=true_atac[-100:, :].obs, var=true_atac[-100:, :].var)
pred_atac.write('atac_tumor_B_pred_tail_100.h5ad')


########## AUROC & AUPRC ##########
# python plot_atac_roc_pr.py --pred atac_tumor_B_pred_100.h5ad --true atac_tumor_B_true_100.h5ad

import argparse
import pandas as pd
import sklearn.metrics as metrics
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Evaluate predicted atac peaks')
parser.add_argument('-p', '--pred', type=str, help='prediction')
parser.add_argument('-t', '--true', type=str, help='ground truth')
args = parser.parse_args()
pred_file = args.pred
true_file = args.true

pred_atac = sc.read_h5ad(pred_file)
print('read pred file done')
true_atac = sc.read_h5ad(true_file)
print('read true file done')

def get_array(X):
  if type(X)!=np.ndarray:
    return X.toarray()
  else:
    return X

pred = get_array(pred_atac.X)
true = get_array(true_atac.X)

## auroc
fpr, tpr, _ = metrics.roc_curve(true.flatten(), pred.flatten())
auroc = metrics.auc(fpr, tpr)
fig, ax = plt.subplots(dpi=300, figsize=(5, 5))
ax.plot(fpr, tpr)
ax.set(xlim=(0, 1.0), ylim=(0.0, 1.05),
       xlabel="False positive rate", ylabel="True positive rate",
       title=f"RNA->ATAC (AUROC={auroc:.3f})")
fig.savefig('rna2atac_auroc.pdf', dpi=600, bbox_inches="tight")
print('plot roc curve done')

## auprc
precision, recall, _thresholds = metrics.precision_recall_curve(true.flatten(), pred.flatten())
auprc = metrics.auc(recall, precision)
fig, ax = plt.subplots(dpi=300, figsize=(5, 5))
ax.plot(recall, precision)
ax.set(xlim=(0, 1.0), ylim=(0.0, 1.05),
       xlabel="Recall", ylabel="Precision",
       title=f"RNA->ATAC (AUPRC={auprc:.3f})")
fig.savefig('rna2atac_auprc.pdf', dpi=600, bbox_inches="tight")
print('plot pr curve done')


############ Cell annotation ############
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import matplotlib.pyplot as plt

adata = sc.read_h5ad('rna_tumor_B.h5ad')  # 14566 × 38244

sc.pp.filter_genes(adata, min_cells=3)  # 14566 × 20198
#adata.var['mt'] = adata.var_names.str.startswith('MT-')
#sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
adata.raw = adata
adata = adata[:, adata.var.highly_variable]
#sc.pp.regress_out(adata, ['total_counts', 'pct_counts_mt'])
#sc.pp.scale(adata, max_value=10)

sc.tl.pca(adata, svd_solver='arpack')
#sc.pl.pca_variance_ratio(adata, log=True)

sc.pp.neighbors(adata)
sc.tl.umap(adata, min_dist=0.2)
sc.tl.leiden(adata, resolution=0.2)

sc.tl.rank_genes_groups(adata, 'leiden', method='wilcoxon')

sc.pl.umap(adata, color='leiden', legend_fontsize='5', legend_loc='on data',
           title='', frameon=True, save='_min_dist_0.2_resolution_0.2.pdf')

pd.DataFrame(adata.uns['rank_genes_groups']['names']).head(10)
#            0          1       2        3        4
# 0      PRKCH       TCF4  SLC8A1    BANK1    FBXL7
# 1        FYN       PAX5   RBM47  RALGPS2     SVIL
# 2     THEMIS   ARHGAP24   LRMDA   CCSER1     NFIB
# 3     BCL11B       AFF3    TYMP    MEF2C  PPFIBP1
# 4     INPP4B      FCRL5   DMXL2    FCRL1      APP
# 5      CELF2      GPM6A    DPYD   FCHSD2    ZFPM2
# 6      CD247  LINC01320    TNS3    MS4A1    DOCK1
# 7      MBNL1     NIBAN3   MCTP1     AFF3    MAGI1
# 8       CBLB      FOXP1  PLXDC2    CDK14   SPTBN1
# 9  LINC01934     RUBCNL   FMNL2   RIPOR2    RBPMS

# cluster_0:PRKCH,FYN,THEMIS,BCL11B,INPP4B,CELF2,CD247,MBNL1,CBLB,LINC01934
# cluster_1:TCF4,PAX5,ARHGAP24,AFF3,FCRL5,GPM6A,LINC01320,NIBAN3,FOXP1,RUBCNL
# cluster_2:SLC8A1,RBM47,LRMDA,TYMP,DMXL2,DPYD,TNS3,MCTP1,PLXDC2,FMNL2
# cluster_3:BANK1,RALGPS2,CCSER1,MEF2C,FCRL1,FCHSD2,MS4A1,AFF3,CDK14,RIPOR2
# cluster_4:FBXL7,SVIL,NFIB,PPFIBP1,APP,ZFPM2,DOCK1,MAGI1,SPTBN1,RBPMS

# cluster_0: T cell
# cluster_1: Tumor B cell
# cluster_2: Monocyte
# cluster_3: Normal B cell
# cluster_4: Other

## cell type annotation tool: http://xteam.xbio.top/ACT
## ref: https://www.nature.com/articles/s41467-023-36559-0/figures/5
cell_anno_dict = {'0':'T cell', '1':'Tumor B cell', '2':'Monocyte', '3':'Normal B cell', '4':'Other'}
adata.obs['cell_anno'] = adata.obs['leiden']
adata.obs['cell_anno'].replace(cell_anno_dict, inplace=True)

sc.pl.umap(adata, color='cell_anno', legend_fontsize='7', legend_loc='on data', size=3,
           title='', frameon=True, save='_cell_annotation.pdf')

adata.write('rna_tumor_B_res.h5ad')


##################### scATAC-seq data processing #####################
## https://github.com/kaizhang/SnapATAC2
# pip install snapatac2
# pip install ipython

import snapatac2 as snap
import polars as pl

data = snap.read('atac_tumor_B.h5ad')
data.to_memory()

snap.pp.select_features(data, n_features=250000)
snap.tl.spectral(data)
snap.tl.umap(data)
#snap.pl.umap(data, color='leiden', interactive=False, height=500)
data.obs['cell_anno'] = pl.from_pandas(adata.obs['cell_anno'])
#snap.pl.umap(data, color='cell_anno', show=False, out_file='umap_atac_true.pdf', height=500)

data.write('atac_true_tumor_B_res.h5ad')

atac_true = sc.read_h5ad('atac_true_tumor_B_res.h5ad')
atac_true.obs.cell_anno = atac_true.obs.cell_anno.cat.set_categories(['T cell', 'Tumor B cell', 'Monocyte', 'Normal B cell', 'Other'])  # keep the color consistent

sc.pl.umap(atac_true, color='cell_anno', legend_fontsize='7', legend_loc='on data', size=3,
           title='', frameon=True, save='_atac_true_cell_annotation.pdf')


## prediction result evaulation
import numpy as np
import scanpy as sc
import snapatac2 as snap
from scipy.sparse import csr_matrix

out = np.zeros([14566, 1033239])
for i in (list(range(1000, 14566, 1000)) + [14566]):
    m = np.load('atac_tumor_B_pred_'+str(i)+'.npy')
    m[m>0.95]=1
    m[m<=0.95]=0
    if i!=14566:
        out[(i-1000):i] = m
    else:
        out[(i-566):i] = m
    print('matrix', i, 'done')


atac_true = snap.read('atac_tumor_B.h5ad', backed=None)
atac_pred = atac_true.copy()

#out = out.astype(np.float32)
atac_pred.X = csr_matrix(out)

snap.pp.select_features(atac_pred) #snap.pp.select_features(atac_pred, n_features=250000)
snap.tl.spectral(atac_pred) #snap.tl.spectral(atac_pred, n_comps=50)
snap.tl.umap(atac_pred)
snap.pl.umap(atac_pred, color='cell_anno', show=False, out_file='umap_atac_pred_0.95.pdf', height=500)


# i = 14566
# m = np.load('atac_tumor_B_pred_'+str(i)+'.npy')
# np.setdiff1d(np.where(m[10]>0.95), np.where(m[9]>0.95))
# array([802275])


atac_pred.write('atac_pred_tumor_B_res_cutoff_0.95.h5ad')

atac_true_res = sc.read_h5ad('atac_true_tumor_B_res.h5ad')
atac_pred.obs.cell_anno = atac_true_res.obs.cell_anno.cat.set_categories(['T cell', 'Tumor B cell', 'Monocyte', 'Normal B cell', 'Other'])  # keep the color consistent

sc.pl.umap(atac_pred, color='cell_anno', legend_fontsize='7', legend_loc='right margin', size=3,
           title='', frameon=True, save='_atac_pred_cell_annotation.pdf')


#### prediction is not ranked!!!!



import pandas as pd

def out_pred_bedgraph(cell_type='T cell'):
    t_cell_pred = atac_pred[atac_pred.obs['cell_anno']==cell_type, :].X.toarray().mean(axis=0)
    t_cell_df = pd.DataFrame({'chr': atac_pred.var['gene_ids'].map(lambda x: x.split(':')[0]).values,
                              'start': atac_pred.var['gene_ids'].map(lambda x: x.split(':')[1].split('-')[0]).values,
                              'end': atac_pred.var['gene_ids'].map(lambda x: x.split(':')[1].split('-')[1]).values,
                              'val': t_cell_pred})
    t_cell_df.to_csv(cell_type.replace(' ', '_')+'_atac_pred.bedgraph', index=False, header=False, sep='\t')
    print(cell_type, 'pred done')

def out_true_bedgraph(cell_type='T cell'):
    t_cell_true = atac_true[atac_true.obs['cell_anno']==cell_type, :].X.toarray().mean(axis=0)
    t_cell_true_df = pd.DataFrame({'chr': atac_true.var['gene_ids'].map(lambda x: x.split(':')[0]).values,
                                   'start': atac_true.var['gene_ids'].map(lambda x: x.split(':')[1].split('-')[0]).values,
                                   'end': atac_true.var['gene_ids'].map(lambda x: x.split(':')[1].split('-')[1]).values,
                                   'val': t_cell_true})
    t_cell_true_df.to_csv(cell_type.replace(' ', '_')+'_atac_true.bedgraph', index=False, header=False, sep='\t')
    print(cell_type, 'true done')

for cell_type in set(atac_pred.obs['cell_anno'].values):
    out_pred_bedgraph(cell_type)
    out_true_bedgraph(cell_type)









