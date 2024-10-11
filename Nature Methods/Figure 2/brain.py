#### Scenario_1
import scanpy as sc
import snapatac2 as snap
import pandas as pd
from tqdm import tqdm

true = sc.read_h5ad('atac_test.h5ad')
snap.pp.select_features(true)
snap.tl.spectral(true)
snap.tl.umap(true)

true.obs['cell_anno'].value_counts()
# CD14 Mono           572
# CD8 Naive           313
# CD4 TCM             219
# CD4 Naive           201
# CD8 TEM             109
# CD4 intermediate    100
# NK                   91
# B memory             84
# CD16 Mono            72
# B naive              70
# CD4 TEM              68
# Treg                 28
# MAIT                 23
# pDC                  23
# cDC2                 21

pred = sc.read_h5ad('atac_cisformer_umap.h5ad')

def out_norm_bedgraph(atac, cell_type='Oligodendrocyte', type='pred'):
    atac_X_raw = atac[atac.obs['cell_anno']==cell_type, :].X.toarray().sum(axis=0)
    atac_X = atac_X_raw/atac_X_raw.sum()*10000
    df = pd.DataFrame({'chr': atac.var['gene_ids'].map(lambda x: x.split(':')[0]).values,
                       'start': atac.var['gene_ids'].map(lambda x: x.split(':')[1].split('-')[0]).values,
                       'end': atac.var['gene_ids'].map(lambda x: x.split(':')[1].split('-')[1]).values,
                       'val': atac_X})
    df.to_csv(cell_type.replace(' ', '_')+'_atac_'+type+'_norm.bedgraph', index=False, header=False, sep='\t')

for cell_type in tqdm(set(true.obs['cell_anno'].values), ncols=80):
    out_norm_bedgraph(atac=true, cell_type=cell_type, type='true')
    out_norm_bedgraph(atac=pred, cell_type=cell_type, type='cisformer')


#### highlight cell type for true
true.obs['cell_or_not'] = (true.obs['cell_anno']=='CD14 Mono').astype('category')
true.obs['cell_or_not'].replace({False:'Not CD14 Mono', True:'CD14 Mono'}, inplace=True)
true.obs['cell_or_not'] = pd.Categorical(true.obs['cell_or_not'], categories=['CD14 Mono', 'Not CD14 Mono'])
sc.pl.umap(true, color='cell_or_not', palette=['orangered', 'darkgrey'], legend_fontsize='7', legend_loc='right margin', size=5,
           title='', frameon=True, save='_CD14_Mono_true.pdf')

#### plot peak for true
## CD14 for CD14 Mono
true.var.iloc[75191]['gene_ids']   # chr5:140630821-140630986

true.obs['peak_0_1'] = true.X.toarray()[:, 75191]
true.obs['peak_0_1'] = true.obs['peak_0_1'].astype('category')
true.obs['peak_0_1'].replace({1:'Open', 0:'Closed'}, inplace=True)
true.obs['peak_0_1'] = pd.Categorical(true.obs['peak_0_1'], categories=['Open', 'Closed'])
sc.pl.umap(true, color='peak_0_1', palette=['orangered', 'darkgrey'], legend_fontsize='7', legend_loc='right margin', size=5,
           title='', frameon=True, save='_CD14_75191_true.pdf')

#### highlight cell type for pred
pred.obs['cell_or_not'] = (pred.obs['cell_anno']=='CD14 Mono').astype('category')
pred.obs['cell_or_not'].replace({False:'Not CD14 Mono', True:'CD14 Mono'}, inplace=True)
pred.obs['cell_or_not'] = pd.Categorical(pred.obs['cell_or_not'], categories=['CD14 Mono', 'Not CD14 Mono'])
sc.pl.umap(pred, color='cell_or_not', palette=['orangered', 'darkgrey'], legend_fontsize='7', legend_loc='right margin', size=5,
           title='', frameon=True, save='_CD14_Mono_pred.pdf')

#### plot peak for pred
pred.obs['peak_0_1'] = pred.X.toarray()[:, 75191]
pred.obs['peak_0_1'] = pred.obs['peak_0_1'].astype('category')
pred.obs['peak_0_1'].replace({1:'Open', 0:'Closed'}, inplace=True)
pred.obs['peak_0_1'] = pd.Categorical(pred.obs['peak_0_1'], categories=['Open', 'Closed'])
sc.pl.umap(pred, color='peak_0_1', palette=['orangered', 'darkgrey'], legend_fontsize='7', legend_loc='right margin', size=5,
           title='', frameon=True, save='_CD14_75191_cisformer.pdf')





## GFAP
true.var.iloc[198675]   # chr17:44915696-44915866    GFAP
true.var.iloc[51853]    # chr3:126197879-126198225   ALDH1L1

true.obs['gene_0_1'] = true.X.toarray()[:, 75195]
true.obs['gene_0_1'] = true.obs['gene_0_1'].astype('category')
# sc.pl.umap(true, color='gene_0_1', palette=['darkgrey', 'orangered'], legend_fontsize='7', legend_loc='right margin', size=5,
#            title='', frameon=True, save='_GFAP_true.pdf')
true.obs[true.obs['gene_0_1']==1]['cell_anno'].value_counts()
# Astrocyte          148
# OPC                 37
# Oligodendrocyte     10
# Microglia            8
# VIP                  5
# IT                   4
# SST                  3
# PVALB                0

true.obs['gene_0_1'] = true.X.toarray()[:, 75197]
res = (true.obs[true.obs['gene_0_1']==1]['cell_anno'].value_counts())/(true.obs['cell_anno'].value_counts())
res.sort_values(ascending=False)


pred.obs['gene_0_1'] = pred.X.toarray()[:, 75197]
res = (pred.obs[pred.obs['gene_0_1']==1]['cell_anno'].value_counts())/(pred.obs['cell_anno'].value_counts())
res.sort_values(ascending=False)

                                                                      
pred_raw = sc.read_h5ad('atac_cisformer.h5ad')
pred = pred_raw.copy()
pred.X[:, 198675][pred.X[:, 198675]>0.4] = 1
pred.X[:, 198675][pred.X[:, 198675]<=0.4] = 0
pred.obs['gene_0_1'] = pred.X[:, 198675]
pred.obs[pred.obs['gene_0_1']==1]['cell_anno'].value_counts()




(pred.obs[pred.obs['gene_0_1']==1]['cell_anno'].value_counts())/(pred.obs['cell_anno'].value_counts())

scbt = sc.read_h5ad('atac_scbt_umap.h5ad')
scbt.obs['gene_0_1'] = scbt.X.toarray()[:, 198675]
scbt.obs['gene_0_1'] = scbt.obs['gene_0_1'].astype('category')
sc.pl.umap(scbt, color='gene_0_1', palette=['darkgrey', 'orangered'], legend_fontsize='7', legend_loc='right margin', size=5,
           title='', frameon=True, save='_GFAP_scbt.pdf')

scbt.obs[scbt.obs['gene_0_1']==1]['cell_anno'].value_counts()


cifm = sc.read_h5ad('atac_cisformer.h5ad')
cifm.obs['gene_0_1'] = cifm.X[:, 198675]
scbt.obs['gene_0_1'] = scbt.obs['gene_0_1'].astype('category')
sc.pl.umap(scbt, color='gene_0_1', palette=['darkgrey', 'orangered'], legend_fontsize='7', legend_loc='right margin', size=5,
           title='', frameon=True, save='_GFAP_scbt.pdf')

scbt.obs[scbt.obs['gene_0_1']==1]['cell_anno'].value_counts()


