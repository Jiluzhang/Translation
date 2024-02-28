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

batch_size = 2
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
test_kwargs = {'batch_size': batch_size, 'shuffle': True}
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
    print(2*(i+1), 'cells done')
    
    if 2*(i+1)%1000==0 or 2*(i+1)==test_atac.n_obs:
      np.save('atac_tumor_B_pred_'+str(2*(i+1))+'.npy', y_hat_out.to('cpu').numpy())
      y_hat_out = torch.Tensor().cuda()
    print(2*(i+1), 'cells done', flush=True)


#atac_res = sc.AnnData(np.zeros((test_atac.n_obs, test_atac.n_vars)), obs=test_atac.obs, var=test_atac.var)
#atac_res.X = y_hat_out.to('cpu').numpy()
#atac_res.write('atac_tumor_B_pred.h5ad')



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
adata.var['mt'] = adata.var_names.str.startswith('MT-')
sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
adata.raw = adata
adata = adata[:, adata.var.highly_variable]
sc.pp.regress_out(adata, ['total_counts', 'pct_counts_mt'])
sc.pp.scale(adata, max_value=10)

sc.tl.pca(adata, svd_solver='arpack')
sc.pl.pca_variance_ratio(adata, log=True)

sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40)
sc.tl.umap(adata)
sc.tl.leiden(adata)

sc.tl.rank_genes_groups(adata, 'leiden', method='wilcoxon')
adata.write('rna_tumor_B_res.h5ad')

## umap plot
# sc.pl.umap(adata, color='leiden', legend_fontsize='3.5', legend_loc='on data',
#            title='', frameon=True, save='umap.pdf')
# sc.pl.umap(adata, color='sample_groups', legend_fontsize='5', legend_loc='right margin', size=2, 
#            title='', frameon=True, save='batch.pdf')

#marker_genes = ['Dcn', 'Col1a1', 'Col1a2', 'Lum',                               'Pdgfra',
#                'Cdh5', 'Pecam1', 'Vwf',                 'Npr3', 'Fabp4',
#                'Plp1',
#                'Tagln', 'Mylk', 'Myh11',
#                                                         'Pdgfrb',              'Rgs5', 'Abcc9',
#                'Lyz2', 'Cd14', 'Mrc1', 'Cd68',          'Adgre1',              'C1qa',
#                'Cd3d', 'Cd2', 'Klrd1', 'Nkg7', 'Il7r',  'Cd8a',
#                'Cd79a', 'Ms4a1', 'Ighd',                'Cd19',
#                'S100a9',
#                'Mki67', 'Top2a']
#sc.pl.dotplot(adata, marker_genes, groupby='leiden', dendrogram=True,
#              save='marker_genes.pdf')

# cluster_0:Dcn,Cst3,Mfap4,Lum,Igfbp4
# cluster_1:Cd74,mt-Rnr1,Apoe,Gm42418,Lyz2
# cluster_2:Mgp,Eln,Col1a1,Fn1,Tm4sf1
# cluster_3:C1qb,C1qc,C1qa,Lyz2,Ms4a7
# cluster_4:Dcn,Mfap4,Sparc,Col3a1,Serpinf1
# cluster_5:Ctss,Ms4a4c,Fcer1g,Fcgr1,Ly6e
# cluster_6:Tuba1b,Mki67,Tubb5,Ube2c,Spp1
# cluster_7:Ms4a4b,Trbc2,H2-K1,AW112020,H2-Q7
# cluster_8:Cfh,Serping1,Gsn,Dcn,Cst3
# cluster_9:Acta2,Tagln,Myl9,Igfbp7,Tpm2
# cluster_10:Fth1,Ftl1,Ctsd,Prdx1,Lyz2
# cluster_11:Mgp,Bgn,Lox,Ppic,Sparc
# cluster_12:Pecam1,Ly6c1,Fabp4,Cdh5,Cav1
# cluster_13:Rps24,Rps27,Trbc2,Rplp2,Rplp1
# cluster_14:H2-Ab1,H2-Aa,Cd74,H2-Eb1,Ifi30
# cluster_15:Il1b,Cxcl2,Srgn,S100a8,S100a9
# cluster_16:Igkc,Ighm,Cd74,Cd79a,Rps27
# cluster_17:Cldn5,Cavin2,Plvap,Mmrn1,Lrg1
# cluster_18:Stmn2,Npy,Gnas,Uchl1,Tubb3
# cluster_19:Plp1,Dbi,Prnp,Matn2,Gpm6b
# cluster_20:Car3,Cfd,Adipoq,Gpx3,Fabp4
# cluster_21:AW112010,Il2rb,Nkg7,Ms4a4b,Ccl5
# cluster_22:Igkc,Jchain,Igha,Iglv1,Cd74
# cluster_23:Sparcl1,Tpm1,Tinagl1,Myh11,Crip1
# cluster_24:Trdc,Tcrg-C1,Cxcr6,Cd163l1,Cd3g
# cluster_25:S100a8,S100a9,Lcn2,Camp,Ngp
# cluster_26:Mpz,Pmp22,Mbp,Cnp,Mal
# cluster_27:Fcer1a,Tpsb2,Mcpt4,Mrgprx2,Cpa3

# cluster_0: Fibroblast
# cluster_1: Macrophage
# cluster_2: Fibroblast
# cluster_3: Macrophage
# cluster_4: Fibroblast
# cluster_5: Monocyte
# cluster_6: T cell
# cluster_7: T cell
# cluster_8: Fibroblast
# cluster_9: Smooth muscle cell
# cluster_10: Macrophage
# cluster_11: Fibroblast
# cluster_12: Endothelial cell
# cluster_13: T cell
# cluster_14: Dendritic cell
# cluster_15: Macrophage
# cluster_16: B cell
# cluster_17: Endothelial cell
# cluster_18: Neuron
# cluster_19: Schwann cell
# cluster_20: Fat cell
# cluster_21: NK cell
# cluster_22: Plasma cell
# cluster_23: Smooth muscle cell
# cluster_24: T cell
# cluster_25: Neutrophil
# cluster_26: Schwann cell
# cluster_27: Mast cell

## cell type annotation tool: http://xteam.xbio.top/ACT
cell_anno_dict = {'0':'Fibroblast', '1':'Macrophage', '2':'Fibroblast', '3':'Macrophage', '4':'Fibroblast', '5':'Monocyte',
                  '6':'T cell', '7':'T cell', '8':'Fibroblast', '9':'Smooth muscle cell', '10':'Macrophage', '11':'Fibroblast',
                  '12':'Endothelial cell', '13':'T cell', '14':'Dendritic cell', '15':'Macrophage', '16':'B cell',
                  '17':'Endothelial cell', '18':'Neuron', '19':'Schwann cell', '20':'Fat cell', '21':'NK cell', '22':'Plasma cell',
                  '23':'Smooth muscle cell', '24':'T cell', '25':'Neutrophil', '26':'Schwann cell', '27':'Mast cell'}
adata.obs['cell_anno'] = adata.obs['leiden']
adata.obs['cell_anno'].replace(cell_anno_dict, inplace=True)

sc.pl.umap(adata, color='cell_anno', legend_fontsize='5', legend_loc='on data',
           title='', frameon=True, save='umap.pdf')
sc.pl.umap(adata, color='cell_anno', legend_fontsize='5', legend_loc='right margin', size=1,
           title='', frameon=True, save='umap2.pdf')
sc.pl.umap(adata, color='sample_groups', legend_fontsize='5', legend_loc='right margin', size=2, 
           title='', frameon=True, save='batch.pdf')

## output marker genes for each cluster
pd.DataFrame(adata.uns['rank_genes_groups']['names']).head(5).to_csv('cluster_marker_genes.txt', index=False, sep='\t')

#marker_genes = np.array(pd.DataFrame(adata.uns['rank_genes_groups']['names']).head(5)[['0', '1', '5', '6', '9', '12', '14', '16', '18', '19', '20', '21', '22', '25', '27']].T).flatten()
marker_genes_top5 = pd.DataFrame(adata.uns['rank_genes_groups']['names']).head(5)
marker_genes_dict = {'Fibroblast': marker_genes_top5['0'], 'Macrophage': marker_genes_top5['1'], 'Monocyte': marker_genes_top5['5'], 'T cell': marker_genes_top5['6'],
                     'Smooth muscle cell': marker_genes_top5['9'], 'Endothelial cell': marker_genes_top5['12'], 'Dendritic cell': marker_genes_top5['14'], 'B cell': marker_genes_top5['16'],
                     'Neuron': marker_genes_top5['18'], 'Schwann cell': marker_genes_top5['19'], 'Fat cell': marker_genes_top5['20'], 'NK cell': marker_genes_top5['21'],
                     'Plasma cell': marker_genes_top5['22'], 'Neutrophil': marker_genes_top5['25'], 'Mast cell': marker_genes_top5['27']}
sc.pl.dotplot(adata, marker_genes_dict, groupby='cell_anno', 
              categories_order=['Fibroblast', 'Macrophage', 'Monocyte', 'T cell', 'Smooth muscle cell', 'Endothelial cell', 'Dendritic cell',
                                'B cell', 'Neuron', 'Schwann cell', 'Fat cell', 'NK cell', 'Plasma cell', 'Neutrophil', 'Mast cell'],
              save='marker_genes.pdf')

adata[adata.obs['sample_groups']=='wt-1'].obs['cell_anno'].value_counts()
adata[adata.obs['sample_groups']=='wt-2'].obs['cell_anno'].value_counts()
adata[adata.obs['sample_groups']=='ko-1'].obs['cell_anno'].value_counts()
adata[adata.obs['sample_groups']=='ko-2'].obs['cell_anno'].value_counts()



