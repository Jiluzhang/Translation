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


