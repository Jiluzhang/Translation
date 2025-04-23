## workdir: /fs/home/jiluzhang/Nature_methods/Figure_2/pan_cancer/cisformer/atac2rna
import scanpy as sc
import pandas as pd
import numpy as np
from plotnine import *
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from tqdm import tqdm
from scipy.stats import pearsonr, spearmanr

plt.rcParams['pdf.fonttype'] = 42

## plot prediction vs. true
pred = sc.read_h5ad('pancancer_whole_predicted_rna.h5ad')  # 6478 × 38244
true = sc.read_h5ad('mapped_rna_test.h5ad')                # 6478 × 38244

sc.pp.log1p(pred)
sc.pp.log1p(true)

pred_X = pred.X.toarray()
true_X = true.X.toarray()

pearsonr_lst = []
spearmanr_lst = []
for i in tqdm(range(38244), ncols=80):
    if ((len(set(true_X[:, i]))!=1) & (len(set(pred_X[:, i]))!=1)):
        pearsonr_lst.append(pearsonr(true_X[:, i], pred_X[:, i])[0])
        spearmanr_lst.append(spearmanr(true_X[:, i], pred_X[:, i])[0])

np.mean(pearsonr_lst)   # 0.8775330835211348
np.mean(spearmanr_lst)  # 0.9469449249037732

df = pd.DataFrame({'idx': ['pearsonr']*len(pearsonr_lst)+['spearmanr']*len(spearmanr_lst),
                   'val': pearsonr_lst+spearmanr_lst})

p = ggplot(df, aes(x='idx', y='val', fill='idx')) + geom_boxplot(width=0.5, show_legend=False, outlier_shape='') + xlab('') +\
                                                    coord_cartesian(ylim=(0.7, 1.0)) +\
                                                    scale_y_continuous(breaks=np.arange(0.7, 1.0+0.1, 0.1)) + theme_bw()
p.save(filename='pancancer_atac2rna_pearsonr_spearmanr.pdf', dpi=600, height=4, width=4)


## plot specific enhancer heatmap
df = pd.read_table('pancancer_specific_enhancers_raw_data.tsv', index_col=0)
df = df.loc[['T-cells', 'B-cells', 'Macrophages', 'Fibroblasts', 'Endothelial']]   # 184103

t_cell_peaks = df.loc['T-cells'][df.loc['T-cells']==1].index.values          # 24591
b_cell_peaks = df.loc['B-cells'][df.loc['B-cells']==1].index.values          # 58770
macro_peaks = df.loc['Macrophages'][df.loc['Macrophages']==1].index.values   # 16491
fibro_peaks = df.loc['Fibroblasts'][df.loc['Fibroblasts']==1].index.values   # 14657
endo_peaks = df.loc['Endothelial'][df.loc['Endothelial']==1].index.values    # 69594

df = df.loc[:, np.hstack([t_cell_peaks, b_cell_peaks, macro_peaks, fibro_peaks, endo_peaks])]

sns.heatmap(df, cmap='coolwarm', vmin=-1, vmax=1, xticklabels=False)  # figsize=(5, 150)
plt.savefig('pancancer_specific_enhancers.pdf')
plt.close()

sns.heatmap(df, cmap='coolwarm', vmin=-1, vmax=1, xticklabels=False)  # figsize=(5, 150)
plt.savefig('pancancer_specific_enhancers.png')
plt.close()

## plot specific tf heatmap
df = pd.read_table('pancancer_specific_tfs_raw_data.tsv', index_col=0)   # 255

with open('motif_enrichment.pkl', 'rb') as file:
    motif_info = pickle.load(file)



## motif enrichment for pancancer
import snapatac2 as snap
import pandas as pd
import numpy as np
import pickle
import random

df = pd.read_table('pancancer_specific_enhancers_raw_data.tsv', index_col=0)
t_cell_peaks = df.loc['T-cells'][df.loc['T-cells']==1].index.values          # 24591
b_cell_peaks = df.loc['B-cells'][df.loc['B-cells']==1].index.values          # 58770
macro_peaks = df.loc['Macrophages'][df.loc['Macrophages']==1].index.values   # 16491
fibro_peaks = df.loc['Fibroblasts'][df.loc['Fibroblasts']==1].index.values   # 14657
endo_peaks = df.loc['Endothelial'][df.loc['Endothelial']==1].index.values    # 69594

random.seed(0)
peaks_dict = {'t_cell': random.sample(list(t_cell_peaks), 10000),
              'b_cell': random.sample(list(b_cell_peaks), 10000),
              'macro': random.sample(list(macro_peaks), 10000),
              'fibro': random.sample(list(fibro_peaks), 10000),
              'endo': random.sample(list(endo_peaks), 10000)}
motifs_enrich = snap.tl.motif_enrichment(motifs=snap.datasets.cis_bp(unique=True),
                                         regions=peaks_dict,
                                         genome_fasta=snap.genome.hg38,
                                         background=None,
                                         method='hypergeometric')
with open('motifs_enrich_luz.pkl', 'wb') as file:
    pickle.dump(motifs_enrich, file)

# with open('motifs_enrich_luz.pkl', 'rb') as file:
#     motifs_enrich = pickle.load(file)

t_cell = motifs_enrich['t_cell'].to_pandas()
b_cell = motifs_enrich['b_cell'].to_pandas()
macro = motifs_enrich['macro'].to_pandas()
fibro = motifs_enrich['fibro'].to_pandas()
endo = motifs_enrich['endo'].to_pandas()

df = pd.DataFrame({'tf':t_cell['name'],
                   't_cell_log2fc':t_cell['log2(fold change)'],
                   't_cell_qval':t_cell['adjusted p-value'],
                   'b_cell_log2fc':b_cell['log2(fold change)'],
                   'b_cell_qval':b_cell['adjusted p-value'],
                   'macro_log2fc':macro['log2(fold change)'],
                   'macro_qval':macro['adjusted p-value'],
                   'fibro_log2fc':fibro['log2(fold change)'],
                   'fibro_qval':fibro['adjusted p-value'],
                   'endo_log2fc':endo['log2(fold change)'],
                   'endo_qval':endo['adjusted p-value']})

t_cell_spe = df[(df['t_cell_log2fc']>np.log2(1.2)) & (df['t_cell_qval']<0.05) &
                (df['t_cell_log2fc']>(df['b_cell_log2fc']+0.1)) &
                (df['t_cell_log2fc']>(df['macro_log2fc']+0.1)) &
                (df['t_cell_log2fc']>(df['fibro_log2fc']+0.1)) &
                (df['t_cell_log2fc']>(df['endo_log2fc']+0.1))]   # 38
t_cell_spe = t_cell_spe.sort_values('t_cell_log2fc', ascending=False)
b_cell_spe = df[(df['b_cell_log2fc']>np.log2(1.2)) & (df['b_cell_qval']<0.05) &
                (df['b_cell_log2fc']>(df['t_cell_log2fc']+0.1)) &
                (df['b_cell_log2fc']>(df['macro_log2fc']+0.1)) &
                (df['b_cell_log2fc']>(df['fibro_log2fc']+0.1)) &
                (df['b_cell_log2fc']>(df['endo_log2fc']+0.1))]   # 14
b_cell_spe = b_cell_spe.sort_values('b_cell_log2fc', ascending=False)
macro_spe = df[(df['macro_log2fc']>np.log2(1.2)) & (df['macro_qval']<0.05) &
                (df['macro_log2fc']>(df['t_cell_log2fc']+0.1)) &
                (df['macro_log2fc']>(df['b_cell_log2fc']+0.1)) &
                (df['macro_log2fc']>(df['fibro_log2fc']+0.1)) &
                (df['macro_log2fc']>(df['endo_log2fc']+0.1))]   # 29
macro_spe = macro_spe.sort_values('macro_log2fc', ascending=False)
fibro_spe = df[(df['fibro_log2fc']>np.log2(1.2)) & (df['fibro_qval']<0.05) &
                (df['fibro_log2fc']>(df['t_cell_log2fc']+0.1)) &
                (df['fibro_log2fc']>(df['b_cell_log2fc']+0.1)) &
                (df['fibro_log2fc']>(df['macro_log2fc']+0.1)) &
                (df['fibro_log2fc']>(df['endo_log2fc']+0.1))]   # 27
fibro_spe = fibro_spe.sort_values('fibro_log2fc', ascending=False)
endo_spe = df[(df['endo_log2fc']>np.log2(1.2)) & (df['endo_qval']<0.05) &
                (df['endo_log2fc']>(df['t_cell_log2fc']+0.4)) &
                (df['endo_log2fc']>(df['b_cell_log2fc']+0.4)) &
                (df['endo_log2fc']>(df['macro_log2fc']+0.4)) &
                (df['endo_log2fc']>(df['fibro_log2fc']+0.4))]   # 33
endo_spe = endo_spe.sort_values('endo_log2fc', ascending=False)

t_cell_spe['tf'].values
# 'HSF5', 'DMRT3', 'DMRT1', 'ZNF653', 'PBX4', 'DMRT2', 'VSX2', 'BCL6', 'HESX1', 'HSFY1', 'SEBOX', 'GSX1', 'SHOX', 'PRRX2',
# 'PRRX1', 'ALX3', 'ISX', 'LBX1', 'NKX2-3', 'POU4F2', 'DMRTA1', 'MLXIP', 'DBX1', 'TBX2', 'LIN54', 'SOX7', 'TEAD2', 'FOS', 'NFE2L1',
# 'TEAD4', 'NFE2L2', 'ZNF324', 'SIX2', 'BACH1', 'NFE2', 'SIX1', 'TEAD1', 'HLX'
# [BCL6, FOS, NFE2L1, NFE2L2]

b_cell_spe['tf'].values
# 'ETV3', 'E2F5', 'PROX1', 'PROX2', 'ZNF454', 'FLI1', 'RUNX2', 'ZNF343', 'RUNX3', 'ZNF213', 'ZBTB11', 'GLI2', 'THAP1', 'ZNF816'
# [RUNX1, RUNX3, FLI1, PROX1]

macro_spe['tf'].values
# 'SMAD3', 'BHLHE22', 'OLIG1', 'BBX', 'NEUROG2', 'DMRTB1', 'BARX2', 'MTERF1', 'POU2AF1', 'ZNF154', 'EOMES', 'NFE2L3', 'FOXD4L5',
# 'FOXD4L6', 'FOXD4L4', 'FOXD4L3', 'FOXD4L1', 'FOXD4', 'TBX3', 'HOXC4', 'FERD3L', 'ZBTB3', 'STAT5A', 'ZNF329', 'TCF7L2', 'FOXR1',
# 'HAND2', 'FOXA2', 'ZNF224'
# [SMAD3, STAT5A, TCF7L2, FOXA2]

fibro_spe['tf'].values
# 'SIX6', 'GCM2', 'PROP1', 'POU3F4', 'POU3F1', 'ZNF655', 'NKX3-1', 'HBP1', 'TBP', 'GFI1B', 'SOX6', 'SIX3', 'IRX4', 'ZNF296', 'TGIF1',
# 'FOXR2', 'IRF9', 'ZFP82', 'ZNF35', 'FOXQ1', 'ZFP69', 'IRF8', 'IRF1', 'SMAD4', 'CPEB1', 'STAT4', 'SOX11'
# [SMAD4, STAT4, IRF1, SOX11]

endo_spe['tf'].values
# 'ZNF385D', 'TIGD1', 'FBXL19', 'HINFP', 'LIN28A', 'LIN28B', 'YY1', 'ZNF610', 'PHF1', 'SP4', 'ETV3L', 'YY2', 'SPDEF', 'GSC', 'ZNF501',
# 'ARNTL', 'CREB3', 'ZBTB37', 'ETV2', 'ELF1', 'SIX5', 'ELF4', 'EHF', 'MLX', 'TFEC', 'USF1', 'ATF4', 'ETV6', 'ELK1', 'TFEB', 'MITF',
# 'SALL1', 'ZNF619'
[ETV2, ELK1, ELF1, ELF4] & [TFEB, MITF, ARNTL, YY1, SPDEF, ATF4, TFEC]

df_spe = pd.concat([t_cell_spe, b_cell_spe, macro_spe, fibro_spe, endo_spe])
df_spe.to_csv('motifs_enrich_luz.txt', sep='\t', index=False)

## plot heatmap for enriched TFs in pan-cancer
import pandas as pd
import numpy as np
from plotnine import *
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['pdf.fonttype'] = 42

df_raw = pd.read_table('motifs_enrich_luz.txt')
df_raw.index = df_raw['tf'].values
df_raw = df_raw[df_raw['tf']!='TIGD1']  # remove inf
df = df_raw[['t_cell_log2fc', 'b_cell_log2fc', 'macro_log2fc', 'fibro_log2fc', 'endo_log2fc']]

sns.clustermap(df.T, cmap='coolwarm', row_cluster=False, col_cluster=False, z_score=1, vmin=-1.5, vmax=1.5, figsize=(40, 8))  # standard_scale=1, z_score=1
plt.savefig('motifs_enrich_luz_heatmap.pdf')
plt.close()


#### cd8 t cell & macro & fibro umap
import scanpy as sc
import matplotlib.pyplot as plt
from matplotlib import rcParams

plt.rcParams['pdf.fonttype'] = 42
sc.settings.set_figure_params(dpi_save=1200)
rcParams["figure.figsize"] = (2, 2)

t_cell = sc.read_h5ad('res_cd8_t_naive_effect_ex.h5ad')
t_cell.uns['cell_anno_colors'] = ['#ff7f0e', '#1f77b4', '#d62728']
sc.pl.umap(t_cell, color='cell_anno', legend_fontsize='5', legend_loc='right margin', size=3,
           title='', frameon=True, save='_cell_anno_t_cell.pdf')

macro = sc.read_h5ad('res_macro.h5ad')
sc.pl.umap(macro, color='cell_type', legend_fontsize='5', legend_loc='right margin', size=3,
           title='', frameon=True, save='_cell_anno_macro.pdf')

fibro = sc.read_h5ad('res_fibro.h5ad')
sc.pl.umap(fibro, color='cell_type', legend_fontsize='5', legend_loc='right margin', size=3,
           title='', frameon=True, save='_cell_anno_fibro.pdf')

#### cd8 t cell & macro & fibro enhancer & tf heatmap
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import snapatac2 as snap
import pandas as pd
import random
import pickle
import numpy as np

plt.rcParams['pdf.fonttype'] = 42

## t cell
t_cell = pd.read_table('t_cell_specific_enhancers_attention_raw_data.tsv', header=None)
t_cell.index = t_cell[0].values
t_cell = t_cell.iloc[:, 1:]
t_cell_ref = pd.read_table('t_cell_specific_enhancers_accessibility_raw_data.tsv', index_col=0)
t_cell.columns = t_cell_ref.columns

sns.heatmap(t_cell, cmap='coolwarm', vmin=-1, vmax=1, xticklabels=False)
plt.savefig('t_cell_specific_enhancers.png')
plt.close()

naive_peaks = t_cell.columns[t_cell.loc['Naive CD8 T']==1].values        # 23944
effect_peaks = t_cell.columns[t_cell.loc['Effector CD8 T']==1].values    # 53128
ex_peaks = t_cell.columns[t_cell.loc['Exhausted CD8 T']==1].values       # 24414

# random.seed(0)
# peaks_dict = {'naive': random.sample(list(naive_peaks), 10000),
#               'effect': random.sample(list(effect_peaks), 10000),
#               'ex': random.sample(list(ex_peaks), 10000)}
peaks_dict = {'naive': naive_peaks,
              'effect': effect_peaks,
              'ex': ex_peaks}
motifs_enrich = snap.tl.motif_enrichment(motifs=snap.datasets.cis_bp(unique=True),
                                         regions=peaks_dict,
                                         genome_fasta=snap.genome.hg38,
                                         background=None,
                                         method='hypergeometric')
with open('t_cell_motifs_enrich_luz.pkl', 'wb') as file:
    pickle.dump(motifs_enrich, file)

# with open('t_cell_motifs_enrich_luz.pkl', 'rb') as file:
#     motifs_enrich = pickle.load(file)

naive = motifs_enrich['naive'].to_pandas()
effect = motifs_enrich['effect'].to_pandas()
ex = motifs_enrich['ex'].to_pandas()

df = pd.DataFrame({'tf':naive['name'],
                   'naive_log2fc':naive['log2(fold change)'],
                   'naive_qval':naive['adjusted p-value'],
                   'effect_log2fc':effect['log2(fold change)'],
                   'effect_qval':effect['adjusted p-value'],
                   'ex_log2fc':ex['log2(fold change)'],
                   'ex_qval':ex['adjusted p-value']})

########################################################################
########################################################################
########################################################################
########################################################################
########################################################################
########################################################################

ex_spe = df[(df['ex_log2fc']>np.log2(1.2)) & (df['ex_qval']<0.05) &
                (df['t_cell_log2fc']>(df['b_cell_log2fc']+0.1)) &
                (df['t_cell_log2fc']>(df['macro_log2fc']+0.1)) &
                (df['t_cell_log2fc']>(df['fibro_log2fc']+0.1)) &
                (df['t_cell_log2fc']>(df['endo_log2fc']+0.1))]   # 38
ex_spe = ex_spe.sort_values('ex_log2fc', ascending=False)
b_cell_spe = df[(df['b_cell_log2fc']>np.log2(1.2)) & (df['b_cell_qval']<0.05) &
                (df['b_cell_log2fc']>(df['t_cell_log2fc']+0.1)) &
                (df['b_cell_log2fc']>(df['macro_log2fc']+0.1)) &
                (df['b_cell_log2fc']>(df['fibro_log2fc']+0.1)) &
                (df['b_cell_log2fc']>(df['endo_log2fc']+0.1))]   # 14
b_cell_spe = b_cell_spe.sort_values('b_cell_log2fc', ascending=False)
macro_spe = df[(df['macro_log2fc']>np.log2(1.2)) & (df['macro_qval']<0.05) &
                (df['macro_log2fc']>(df['t_cell_log2fc']+0.1)) &
                (df['macro_log2fc']>(df['b_cell_log2fc']+0.1)) &
                (df['macro_log2fc']>(df['fibro_log2fc']+0.1)) &
                (df['macro_log2fc']>(df['endo_log2fc']+0.1))]   # 29
macro_spe = macro_spe.sort_values('macro_log2fc', ascending=False)
fibro_spe = df[(df['fibro_log2fc']>np.log2(1.2)) & (df['fibro_qval']<0.05) &
                (df['fibro_log2fc']>(df['t_cell_log2fc']+0.1)) &
                (df['fibro_log2fc']>(df['b_cell_log2fc']+0.1)) &
                (df['fibro_log2fc']>(df['macro_log2fc']+0.1)) &
                (df['fibro_log2fc']>(df['endo_log2fc']+0.1))]   # 27
fibro_spe = fibro_spe.sort_values('fibro_log2fc', ascending=False)
endo_spe = df[(df['endo_log2fc']>np.log2(1.2)) & (df['endo_qval']<0.05) &
                (df['endo_log2fc']>(df['t_cell_log2fc']+0.4)) &
                (df['endo_log2fc']>(df['b_cell_log2fc']+0.4)) &
                (df['endo_log2fc']>(df['macro_log2fc']+0.4)) &
                (df['endo_log2fc']>(df['fibro_log2fc']+0.4))]   # 33
endo_spe = endo_spe.sort_values('endo_log2fc', ascending=False)

t_cell_spe['tf'].values
# 'HSF5', 'DMRT3', 'DMRT1', 'ZNF653', 'PBX4', 'DMRT2', 'VSX2', 'BCL6', 'HESX1', 'HSFY1', 'SEBOX', 'GSX1', 'SHOX', 'PRRX2',
# 'PRRX1', 'ALX3', 'ISX', 'LBX1', 'NKX2-3', 'POU4F2', 'DMRTA1', 'MLXIP', 'DBX1', 'TBX2', 'LIN54', 'SOX7', 'TEAD2', 'FOS', 'NFE2L1',
# 'TEAD4', 'NFE2L2', 'ZNF324', 'SIX2', 'BACH1', 'NFE2', 'SIX1', 'TEAD1', 'HLX'
# [BCL6, FOS, NFE2L1, NFE2L2]

b_cell_spe['tf'].values
# 'ETV3', 'E2F5', 'PROX1', 'PROX2', 'ZNF454', 'FLI1', 'RUNX2', 'ZNF343', 'RUNX3', 'ZNF213', 'ZBTB11', 'GLI2', 'THAP1', 'ZNF816'
# [RUNX1, RUNX3, FLI1, PROX1]

macro_spe['tf'].values
# 'SMAD3', 'BHLHE22', 'OLIG1', 'BBX', 'NEUROG2', 'DMRTB1', 'BARX2', 'MTERF1', 'POU2AF1', 'ZNF154', 'EOMES', 'NFE2L3', 'FOXD4L5',
# 'FOXD4L6', 'FOXD4L4', 'FOXD4L3', 'FOXD4L1', 'FOXD4', 'TBX3', 'HOXC4', 'FERD3L', 'ZBTB3', 'STAT5A', 'ZNF329', 'TCF7L2', 'FOXR1',
# 'HAND2', 'FOXA2', 'ZNF224'
# [SMAD3, STAT5A, TCF7L2, FOXA2]

fibro_spe['tf'].values
# 'SIX6', 'GCM2', 'PROP1', 'POU3F4', 'POU3F1', 'ZNF655', 'NKX3-1', 'HBP1', 'TBP', 'GFI1B', 'SOX6', 'SIX3', 'IRX4', 'ZNF296', 'TGIF1',
# 'FOXR2', 'IRF9', 'ZFP82', 'ZNF35', 'FOXQ1', 'ZFP69', 'IRF8', 'IRF1', 'SMAD4', 'CPEB1', 'STAT4', 'SOX11'
# [SMAD4, STAT4, IRF1, SOX11]

endo_spe['tf'].values
# 'ZNF385D', 'TIGD1', 'FBXL19', 'HINFP', 'LIN28A', 'LIN28B', 'YY1', 'ZNF610', 'PHF1', 'SP4', 'ETV3L', 'YY2', 'SPDEF', 'GSC', 'ZNF501',
# 'ARNTL', 'CREB3', 'ZBTB37', 'ETV2', 'ELF1', 'SIX5', 'ELF4', 'EHF', 'MLX', 'TFEC', 'USF1', 'ATF4', 'ETV6', 'ELK1', 'TFEB', 'MITF',
# 'SALL1', 'ZNF619'
[ETV2, ELK1, ELF1, ELF4] & [TFEB, MITF, ARNTL, YY1, SPDEF, ATF4, TFEC]

df_spe = pd.concat([t_cell_spe, b_cell_spe, macro_spe, fibro_spe, endo_spe])
df_spe.to_csv('motifs_enrich_luz.txt', sep='\t', index=False)

## plot heatmap for enriched TFs in pan-cancer
import pandas as pd
import numpy as np
from plotnine import *
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['pdf.fonttype'] = 42

df_raw = pd.read_table('motifs_enrich_luz.txt')
df_raw.index = df_raw['tf'].values
df_raw = df_raw[df_raw['tf']!='TIGD1']  # remove inf
df = df_raw[['t_cell_log2fc', 'b_cell_log2fc', 'macro_log2fc', 'fibro_log2fc', 'endo_log2fc']]

sns.clustermap(df.T, cmap='coolwarm', row_cluster=False, col_cluster=False, z_score=1, vmin=-1.5, vmax=1.5, figsize=(40, 8))  # standard_scale=1, z_score=1
plt.savefig('motifs_enrich_luz_heatmap.pdf')
plt.close()


## fibro
fibro = pd.read_table('fibro_specific_enhancers_attention_raw_data.tsv', header=None)
fibro.index = fibro.iloc[:, 0].values
fibro = fibro.iloc[:, 1:]
fibro_ref = pd.read_table('fibro_specific_enhancers_accessibility_raw_data.tsv', index_col=0)
fibro.columns = fibro_ref.columns

sns.heatmap(fibro, cmap='coolwarm', vmin=-1, vmax=1, xticklabels=False)
plt.savefig('fibro_specific_enhancers.png')
plt.close()

## macro
macro = pd.read_table('macro_specific_enhancers_attention_raw_data.tsv', header=None)
macro.index = macro.iloc[:, 0].values
macro = macro.iloc[:, 1:]
macro_ref = pd.read_table('macro_specific_enhancers_accessibility_raw_data.tsv', index_col=0)
macro.columns = macro_ref.columns

sns.heatmap(macro, cmap='coolwarm', vmin=-1, vmax=1, xticklabels=False)
plt.savefig('macro_specific_enhancers.png')
plt.close()























