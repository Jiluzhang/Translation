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


############################################################################################################################################################
import scanpy as sc
import snapatac2 as snap
import pandas as pd
from tqdm import tqdm
import numpy as np

true = sc.read_h5ad('atac_test.h5ad')
true.var['chrom'] = true.var['gene_ids'].map(lambda x: x.split(':')[0]).values
true.var['start'] = true.var['gene_ids'].map(lambda x: x.split(':')[1].split('-')[0]).values
true.var['end'] = true.var['gene_ids'].map(lambda x: x.split(':')[1].split('-')[1]).values

olig_cell_idx = np.argwhere(true.obs['cell_anno']=='Oligodendrocyte').flatten()  # 1595

pd.DataFrame({'chr':true.var['chrom'].values, 'start':true.var['start'].values, 'end':true.var['end'].values,
              'val': true.X[olig_cell_idx[0]].toarray()[0]}).to_csv('Oligodendrocyte_00_atac_true.bedgraph', index=False, header=False, sep='\t')
pd.DataFrame({'chr':true.var['chrom'].values, 'start':true.var['start'].values, 'end':true.var['end'].values,
              'val': true.X[olig_cell_idx[100]].toarray()[0]}).to_csv('Oligodendrocyte_01_atac_true.bedgraph', index=False, header=False, sep='\t')
pd.DataFrame({'chr':true.var['chrom'].values, 'start':true.var['start'].values, 'end':true.var['end'].values,
              'val': true.X[olig_cell_idx[1000]].toarray()[0]}).to_csv('Oligodendrocyte_02_atac_true.bedgraph', index=False, header=False, sep='\t')

cifm = sc.read_h5ad('atac_cisformer_umap.h5ad')
pd.DataFrame({'chr':true.var['chrom'].values, 'start':true.var['start'].values, 'end':true.var['end'].values,
              'val': cifm.X[olig_cell_idx[0]].toarray()[0]}).to_csv('Oligodendrocyte_00_atac_cifm.bedgraph', index=False, header=False, sep='\t')
pd.DataFrame({'chr':true.var['chrom'].values, 'start':true.var['start'].values, 'end':true.var['end'].values,
              'val': cifm.X[olig_cell_idx[100]].toarray()[0]}).to_csv('Oligodendrocyte_01_atac_cifm.bedgraph', index=False, header=False, sep='\t')
pd.DataFrame({'chr':true.var['chrom'].values, 'start':true.var['start'].values, 'end':true.var['end'].values,
              'val': cifm.X[olig_cell_idx[1000]].toarray()[0]}).to_csv('Oligodendrocyte_02_atac_cifm.bedgraph', index=False, header=False, sep='\t')

scbt = sc.read_h5ad('atac_scbt_umap.h5ad')
pd.DataFrame({'chr':true.var['chrom'].values, 'start':true.var['start'].values, 'end':true.var['end'].values,
              'val': scbt.X[olig_cell_idx[0]].toarray()[0]}).to_csv('Oligodendrocyte_00_atac_scbt.bedgraph', index=False, header=False, sep='\t')
pd.DataFrame({'chr':true.var['chrom'].values, 'start':true.var['start'].values, 'end':true.var['end'].values,
              'val': scbt.X[olig_cell_idx[100]].toarray()[0]}).to_csv('Oligodendrocyte_01_atac_scbt.bedgraph', index=False, header=False, sep='\t')
pd.DataFrame({'chr':true.var['chrom'].values, 'start':true.var['start'].values, 'end':true.var['end'].values,
              'val': scbt.X[olig_cell_idx[1000]].toarray()[0]}).to_csv('Oligodendrocyte_02_atac_scbt.bedgraph', index=False, header=False, sep='\t')

babel = sc.read_h5ad('atac_babel_umap.h5ad')
pd.DataFrame({'chr':true.var['chrom'].values, 'start':true.var['start'].values, 'end':true.var['end'].values,
              'val': babel.X[olig_cell_idx[0]].toarray()[0]}).to_csv('Oligodendrocyte_00_atac_babel.bedgraph', index=False, header=False, sep='\t')
pd.DataFrame({'chr':true.var['chrom'].values, 'start':true.var['start'].values, 'end':true.var['end'].values,
              'val': babel.X[olig_cell_idx[100]].toarray()[0]}).to_csv('Oligodendrocyte_01_atac_babel.bedgraph', index=False, header=False, sep='\t')
pd.DataFrame({'chr':true.var['chrom'].values, 'start':true.var['start'].values, 'end':true.var['end'].values,
              'val': babel.X[olig_cell_idx[1000]].toarray()[0]}).to_csv('Oligodendrocyte_02_atac_babel.bedgraph', index=False, header=False, sep='\t')

babel_f1 = []
scbt_f1 = []
cifm_f1 = []
for i in np.argwhere(true.obs['cell_anno']=='Astrocyte').flatten()[:20]:
    babel_f1.append(f1_score(babel.X[i].toarray()[0], true.X[i].toarray()[0]))
    scbt_f1.append(f1_score(scbt.X[i].toarray()[0], true.X[i].toarray()[0]))
    cifm_f1.append(f1_score(cifm.X[i].toarray()[0], true.X[i].toarray()[0]))
    # print([f1_score(babel.X[i].toarray()[0], true.X[i].toarray()[0]), 
    #        f1_score(scbt.X[i].toarray()[0], true.X[i].toarray()[0]),
    #        f1_score(cifm.X[i].toarray()[0], true.X[i].toarray()[0])])

[np.mean(babel_f1), np.mean(scbt_f1), np.mean(cifm_f1)]


import snapatac2 as snap
import numpy as np

true = snap.read('atac_test.h5ad', backed=None)
babel = snap.read('atac_babel_umap.h5ad', backed=None)
babel.var.index = true.var.index
scbt = snap.read('atac_scbt_umap.h5ad', backed=None)
scbt.var.index = true.var.index
cifm = snap.read('atac_cisformer_umap.h5ad', backed=None)
cifm.var.index = true.var.index

babel.var['selected'].sum()  # 119123
scbt.var['selected'].sum()   # 159760
cifm.var['selected'].sum()   # 165188


## p=0.05
true_marker_peaks = snap.tl.marker_regions(true, groupby='cell_anno', pvalue=0.05)
babel_marker_peaks = snap.tl.marker_regions(babel, groupby='cell_anno', pvalue=0.05)
scbt_marker_peaks = snap.tl.marker_regions(scbt, groupby='cell_anno', pvalue=0.05)
cifm_marker_peaks = snap.tl.marker_regions(cifm, groupby='cell_anno', pvalue=0.05)

cnt_dict = {}
for cell_type in true.obs['cell_anno'].value_counts().index:
    cnt_dict[cell_type] = [len(true_marker_peaks[cell_type]),
                           len(babel_marker_peaks[cell_type]),
                           len(scbt_marker_peaks[cell_type]),
                           len(cifm_marker_peaks[cell_type])]
# {'Oligodendrocyte': [6251, 10022, 1794, 11692],
#  'Astrocyte': [2055, 2893, 2053, 14595],
#  'OPC': [1288, 3354, 2871, 2581],
#  'VIP': [1176, 739, 10002, 2675],
#  'Microglia': [5670, 36101, 27034, 40042],
#  'SST': [1224, 969, 1890, 1729],
#  'IT': [3286, 1157, 21641, 518],
#  'PVALB': [8870, 718, 4802, 845]}

res_dict = {}
for cell_type in true.obs['cell_anno'].value_counts().index:
    res_dict[cell_type] = [len(np.intersect1d(babel_marker_peaks[cell_type], true_marker_peaks[cell_type])),
                           len(np.intersect1d(scbt_marker_peaks[cell_type], true_marker_peaks[cell_type])),
                           len(np.intersect1d(cifm_marker_peaks[cell_type], true_marker_peaks[cell_type]))]
# {'Oligodendrocyte': [163, 21, 115],
#  'Astrocyte': [29, 14, 117],
#  'OPC': [28, 10, 14],
#  'VIP': [2, 52, 16],
#  'Microglia': [2499, 1119, 1453],
#  'SST': [4, 9, 7],
#  'IT': [21, 366, 10],
#  'PVALB': [19, 89, 31]}

## p=0.1
true_marker_peaks = snap.tl.marker_regions(true, groupby='cell_anno', pvalue=0.1)
babel_marker_peaks = snap.tl.marker_regions(babel, groupby='cell_anno', pvalue=0.1)
scbt_marker_peaks = snap.tl.marker_regions(scbt, groupby='cell_anno', pvalue=0.1)
cifm_marker_peaks = snap.tl.marker_regions(cifm, groupby='cell_anno', pvalue=0.1)

cnt_dict = {}
for cell_type in true.obs['cell_anno'].value_counts().index:
    cnt_dict[cell_type] = [len(true_marker_peaks[cell_type]),
                           len(babel_marker_peaks[cell_type]),
                           len(scbt_marker_peaks[cell_type]),
                           len(cifm_marker_peaks[cell_type])]
# {'Oligodendrocyte': [13396, 14245, 4223, 18141],
#  'Astrocyte': [6680, 3701, 3417, 25777],
#  'OPC': [4832, 6100, 7106, 6242],
#  'VIP': [5595, 1284, 16761, 4224],
#  'Microglia': [11826, 39347, 32201, 52347],
#  'SST': [5370, 1651, 4502, 3666],
#  'IT': [9346, 3192, 37058, 1459],
#  'PVALB': [18028, 1393, 8862, 3173]}

res_dict = {}
for cell_type in true.obs['cell_anno'].value_counts().index:
    res_dict[cell_type] = [len(np.intersect1d(babel_marker_peaks[cell_type], true_marker_peaks[cell_type])),
                           len(np.intersect1d(scbt_marker_peaks[cell_type], true_marker_peaks[cell_type])),
                           len(np.intersect1d(cifm_marker_peaks[cell_type], true_marker_peaks[cell_type]))]
# {'Oligodendrocyte': [658, 221, 712],
#  'Astrocyte': [101, 105, 689],
#  'OPC': [128, 165, 123],
#  'VIP': [34, 511, 91],
#  'Microglia': [4741, 2625, 4261],
#  'SST': [38, 74, 73],
#  'IT': [165, 1652, 81],
#  'PVALB': [120, 431, 378]}
############################################################################################################################################################






import pandas as pd
from sklearn.metrics import f1_score

dat = pd.read_table('Oligodendrocyte_01_atac_true.bedgraph', header=None)
dat.columns = ['chrom', 'start', 'end', 'true']
dat['babel'] = pd.read_table('Oligodendrocyte_01_atac_babel.bedgraph', header=None)[3]
dat['scbt'] = pd.read_table('Oligodendrocyte_01_atac_scbt.bedgraph', header=None)[3]
dat['cifm'] = pd.read_table('Oligodendrocyte_01_atac_cifm.bedgraph', header=None)[3]
[f1_score(dat['babel'], dat['true']), f1_score(dat['scbt'], dat['true']), f1_score(dat['cifm'], dat['true'])]

[((dat['babel']==1) & (dat['true']==1)).sum() / ((dat['babel']==1).sum()),
 ((dat['scbt']==1) & (dat['true']==1)).sum() / ((dat['scbt']==1).sum()),
 ((dat['cifm']==1) & (dat['true']==1)).sum() / ((dat['cifm']==1).sum())]





#### highlight cell type for true
# true.obs['cell_or_not'] = (true.obs['cell_anno']=='CD14 Mono').astype('category')
# true.obs['cell_or_not'].replace({False:'Not CD14 Mono', True:'CD14 Mono'}, inplace=True)
# true.obs['cell_or_not'] = pd.Categorical(true.obs['cell_or_not'], categories=['CD14 Mono', 'Not CD14 Mono'])
# sc.pl.umap(true, color='cell_or_not', palette=['orangered', 'darkgrey'], legend_fontsize='7', legend_loc='right margin', size=5,
#            title='', frameon=True, save='_CD14_Mono_true.pdf')

def hlt_cell_type(atac=true, label_type='true', cell_type='CD14 Mono', color='orangered'):
    atac.obs['cell_or_not'] = (atac.obs['cell_anno']==cell_type).astype('category')
    atac.obs['cell_or_not'].replace({False:'Not '+cell_type, True:cell_type}, inplace=True)
    atac.obs['cell_or_not'] = pd.Categorical(atac.obs['cell_or_not'], categories=[cell_type, 'Not '+cell_type])
    sc.pl.umap(atac, color='cell_or_not', palette=[color, 'darkgrey'], legend_fontsize='7', legend_loc='right margin', size=5,
               title='', frameon=True, save=('_'+cell_type.replace(' ', '_')+'_'+label_type+'.pdf'))

#### plot peak for true
def hlt_peak(atac=true, label_type='true', idx=75191, gene='CD14', color='orangered'):
    atac.obs['peak_0_1'] = atac.X.toarray()[:, idx]
    atac.obs['peak_0_1'] = atac.obs['peak_0_1'].astype('category')
    atac.obs['peak_0_1'].replace({1:'Open', 0:'Closed'}, inplace=True)
    atac.obs['peak_0_1'] = pd.Categorical(atac.obs['peak_0_1'], categories=['Open', 'Closed'])
    sc.pl.umap(atac, color='peak_0_1', palette=[color, 'darkgrey'], legend_fontsize='7', legend_loc='right margin', size=5,
               title='', frameon=True, save=('_'+gene+'_'+str(idx)+'_'+label_type+'.pdf'))

#### split chrom info
chrom_info = pd.DataFrame({'info':true.var.gene_ids.values})
chrom_info['chrom'] = chrom_info['info'].map(lambda x: x.split(':')[0])
chrom_info['start'] = chrom_info['info'].map(lambda x: x.split(':')[1].split('-')[0])
chrom_info['end'] = chrom_info['info'].map(lambda x: x.split(':')[1].split('-')[1])
chrom_info['start'] = chrom_info['start'].astype(int)
chrom_info['end'] = chrom_info['end'].astype(int)

## CD14 for CD14 Mono
true.var.iloc[75191]['gene_ids']   # chr5:140630821-140630986
hlt_cell_type(atac=true, label_type='true', cell_type='CD14 Mono', color='orangered')
hlt_cell_type(atac=pred, label_type='pred', cell_type='CD14 Mono', color='orangered')
hlt_peak(atac=true, label_type='true', idx=75191, gene='CD14', color='orangered')
hlt_peak(atac=pred, label_type='pred', idx=75191, gene='CD14', color='orangered')

## GNLY for NK
chrom_info[(chrom_info['chrom']=='chr2') & (chrom_info['start']<85694100) & (chrom_info['end']>85694400)]
# 30946  chr2:85694079-85694424  chr2  85694079  85694424
hlt_cell_type(atac=true, label_type='true', cell_type='NK', color='royalblue')
hlt_cell_type(atac=pred, label_type='pred', cell_type='NK', color='royalblue')
hlt_peak(atac=true, label_type='true', idx=30946, gene='GNLY', color='royalblue')
hlt_peak(atac=pred, label_type='pred', idx=30946, gene='GNLY', color='royalblue')

## MS4A1 for B cell
chrom_info[(chrom_info['chrom']=='chr11') & (chrom_info['start']<60455600) & (chrom_info['end']>60455800)]
# 141682  chr11:60455504-60455853  chr11  60455504  60455853

true.obs['cell_or_not'] = (true.obs['cell_anno'].isin(['B memory', 'B naive'])).astype('category')
true.obs['cell_or_not'].replace({False:'Not B cell', True:'B cell'}, inplace=True)
true.obs['cell_or_not'] = pd.Categorical(true.obs['cell_or_not'], categories=['B cell', 'Not B cell'])
sc.pl.umap(true, color='cell_or_not', palette=['limegreen', 'darkgrey'], legend_fontsize='7', legend_loc='right margin', size=5,
           title='', frameon=True, save='_B_cell_true.pdf')

pred.obs['cell_or_not'] = (pred.obs['cell_anno'].isin(['B memory', 'B naive'])).astype('category')
pred.obs['cell_or_not'].replace({False:'Not B cell', True:'B cell'}, inplace=True)
pred.obs['cell_or_not'] = pd.Categorical(pred.obs['cell_or_not'], categories=['B cell', 'Not B cell'])
sc.pl.umap(pred, color='cell_or_not', palette=['limegreen', 'darkgrey'], legend_fontsize='7', legend_loc='right margin', size=5,
           title='', frameon=True, save='_B_cell_pred.pdf')

hlt_peak(atac=true, label_type='true', idx=141682, gene='MS4A1', color='limegreen')
hlt_peak(atac=pred, label_type='pred', idx=141682, gene='MS4A1', color='limegreen')


