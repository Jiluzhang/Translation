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
# [ETV2, ELK1, ELF1, ELF4] & [TFEB, MITF, ARNTL, YY1, SPDEF, ATF4, TFEC]

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

naive_spe = df[(df['naive_log2fc']>np.log2(1.2)) & (df['naive_qval']<0.05) &
               (df['naive_log2fc']>df['effect_log2fc']) &
               (df['naive_log2fc']>df['ex_log2fc'])]
naive_spe = naive_spe.sort_values('naive_log2fc', ascending=False)

effect_spe = df[(df['effect_log2fc']>np.log2(1.2)) & (df['effect_qval']<0.05) &
                (df['effect_log2fc']>df['naive_log2fc']) &
                (df['effect_log2fc']>df['ex_log2fc'])]
effect_spe = effect_spe.sort_values('effect_log2fc', ascending=False)

ex_spe = df[(df['ex_log2fc']>np.log2(1.2)) & (df['ex_qval']<0.05) &
            (df['ex_log2fc']>df['naive_log2fc']) &
            (df['ex_log2fc']>df['effect_log2fc'])]
ex_spe = ex_spe.sort_values('ex_log2fc', ascending=False)

naive_spe['tf'].values
# 'MECOM', 'PRDM16', 'ZBTB45', 'PAX3', 'HMGA2', 'ZBTB32', 'MEF2B', 'TBX21', 'MEIS2', 'ZNF433', 'ZNF285', 'TGIF2LY', 'MEOX1', 'PKNOX1',
# 'FOXA3', 'NFIL3', 'ZNF716', 'SOX7', 'NKX2-1', 'LEF1', 'ARID5B', 'SOX21', 'TGIF2', 'OTX1', 'TBX15', 'ZNF250', 'MYF6', 'OTX2',
# 'FOXL2', 'FOXD2', 'FOXB2', 'ZNF274', 'ZNF264', 'ZNF236', 'ZFP82','NFIA', 'MEIS3', 'NFIX'
# [LEF1]

effect_spe['tf'].values
# 'GMEB1', 'NRF1', 'ZSCAN10', 'ATF7', 'AC023509.3', 'YY2', 'CREB1', 'YY1', 'ZBED1', 'OVOL1', 'ETV4', 'ZBTB33', 'HINFP', 'FOSB',
# 'ZNF385D', 'FEV', 'ATF2', 'GABPA', 'ERF', 'FOSL1', 'ZNF200', 'CREB3', 'ELK4', 'OVOL2', 'NFYC', 'CREBL2', 'E4F1', 'ZNF597',
# 'KLF14', 'JDP2', 'LIN28A', 'LIN28B', 'TCFL5', 'ATF1', 'SP4', 'ELF1', 'ZNF441', 'ETV3L', 'SP1', 'SP9', 'EGR1', 'ETV1', 'NFYA',
# 'TIGD1', 'MTF2', 'KLF15', 'CREM', 'KLF7', 'TIGD2', 'JRKL', 'SP8', 'E2F5', 'AHR', 'HES4', 'BATF3', 'ZNF100', 'SP5', 'KMT2A', 'DNMT1',
# 'MBD2', 'ZNF501', 'ZNF541', 'ZNF783', 'ETV3', 'KLF13', 'KLF12'
[BATF3, CREB1, YY1, EGR1]

ex_spe['tf'].values
# 'NEUROG1', 'NEUROG2', 'CEBPG', 'SOX1', 'DPF1', 'NFE2', 'HSFY1', 'NEUROG3', 'NFE2L1', 'FOS', 'DUX1', 'PROP1', 'MEF2A',
# 'FOXO6', 'HOXD4', 'HOXB4', 'SOX14', 'FOSL2', 'BACH1', 'IRX3', 'JUNB', 'NFATC2', 'ZNF177', 'SOX15', 'TBX1', 'POU6F1', 'CDC5L',
# 'ZNF382', 'FOXR2', 'GSX1', 'HNF1B', 'POU4F1', 'DNTTIP1', 'NFE2L2', 'MIXL1', 'FOXD1', 'DRGX', 'LHX3', 'BACH2', 'POU3F2', 'FOXO1',
# 'ZNF35', 'ZNF45', 'LMX1B', 'MAFK', 'POU5F1B', 'DUX3', 'POU2F2', 'NFAT5', 'CEBPA', 'ZNF157', 'ZNF490', 'ZNF260'
[NFATC2, NFE2L2]

df_spe = pd.concat([naive_spe, effect_spe, ex_spe])
df_spe.to_csv('t_cell_motifs_enrich_luz.txt', sep='\t', index=False)

## fibro
fibro = pd.read_table('fibro_specific_enhancers_attention_raw_data.tsv', header=None)
fibro.index = fibro[0].values
fibro = fibro.iloc[:, 1:]
fibro_ref = pd.read_table('fibro_specific_enhancers_accessibility_raw_data.tsv', index_col=0)
fibro.columns = fibro_ref.columns

sns.heatmap(fibro, cmap='coolwarm', vmin=-1, vmax=1, xticklabels=False)
plt.savefig('fibro_specific_enhancers.png')
plt.close()

saa1_peaks = fibro.columns[fibro.loc['qFibro_SAA1']==1].values          # 70901
cthrc1_peaks = fibro.columns[fibro.loc['eFibro_CTHRC1']==1].values      # 50929
col11a1_peaks = fibro.columns[fibro.loc['eFibro_COL11A1']==1].values    # 14392
ccl5_peaks = fibro.columns[fibro.loc['apFibro_CCL5']==1].values         # 35405
myh11_peaks = fibro.columns[fibro.loc['MyoFibro_MYH11']==1].values      # 46977

peaks_dict = {'saa1': saa1_peaks, 'cthrc1': cthrc1_peaks, 'col11a1': col11a1_peaks,
              'ccl5': ccl5_peaks, 'myh11': myh11_peaks}
motifs_enrich = snap.tl.motif_enrichment(motifs=snap.datasets.cis_bp(unique=True),
                                         regions=peaks_dict,
                                         genome_fasta=snap.genome.hg38,
                                         background=None,
                                         method='hypergeometric')
with open('fibro_motifs_enrich_luz.pkl', 'wb') as file:
    pickle.dump(motifs_enrich, file)

# with open('fibro_motifs_enrich_luz.pkl', 'rb') as file:
#     motifs_enrich = pickle.load(file)

saa1 = motifs_enrich['saa1'].to_pandas()
cthrc1 = motifs_enrich['cthrc1'].to_pandas()
col11a1 = motifs_enrich['col11a1'].to_pandas()
ccl5 = motifs_enrich['ccl5'].to_pandas()
myh11 = motifs_enrich['myh11'].to_pandas()

df = pd.DataFrame({'tf':saa1['name'],
                   'saa1_log2fc':saa1['log2(fold change)'], 'saa1_qval':saa1['adjusted p-value'],
                   'cthrc1_log2fc':cthrc1['log2(fold change)'], 'cthrc1_qval':cthrc1['adjusted p-value'],
                   'col11a1_log2fc':col11a1['log2(fold change)'], 'col11a1_qval':col11a1['adjusted p-value'],
                   'ccl5_log2fc':ccl5['log2(fold change)'], 'ccl5_qval':ccl5['adjusted p-value'],
                   'myh11_log2fc':myh11['log2(fold change)'], 'myh11_qval':myh11['adjusted p-value'],})

saa1_spe = df[(df['saa1_log2fc']>np.log2(1.1))         & 
              (df['saa1_qval']<0.05)                   &
              (df['saa1_log2fc']>df['cthrc1_log2fc'])  &
              (df['saa1_log2fc']>df['col11a1_log2fc']) &
              (df['saa1_log2fc']>df['ccl5_log2fc'])    &
              (df['saa1_log2fc']>df['myh11_log2fc'])]
saa1_spe = saa1_spe.sort_values('saa1_log2fc', ascending=False)  # 30

cthrc1_spe = df[(df['cthrc1_log2fc']>np.log2(1.1))         & 
              (df['cthrc1_qval']<0.05)                   &
              (df['cthrc1_log2fc']>df['saa1_log2fc'])  &
              (df['cthrc1_log2fc']>df['col11a1_log2fc']) &
              (df['cthrc1_log2fc']>df['ccl5_log2fc'])    &
              (df['cthrc1_log2fc']>df['myh11_log2fc'])]
cthrc1_spe = cthrc1_spe.sort_values('cthrc1_log2fc', ascending=False)  # 10

col11a1_spe = df[(df['col11a1_log2fc']>np.log2(1.5))         & 
              (df['col11a1_qval']<0.05)                   &
              (df['col11a1_log2fc']>df['cthrc1_log2fc'])  &
              (df['col11a1_log2fc']>df['saa1_log2fc']) &
              (df['col11a1_log2fc']>df['ccl5_log2fc'])    &
              (df['col11a1_log2fc']>df['myh11_log2fc'])]
col11a1_spe = col11a1_spe.sort_values('col11a1_log2fc', ascending=False)  # 42

ccl5_spe = df[(df['ccl5_log2fc']>np.log2(1.2))         & 
              (df['ccl5_qval']<0.05)                   &
              (df['ccl5_log2fc']>df['cthrc1_log2fc'])  &
              (df['ccl5_log2fc']>df['col11a1_log2fc']) &
              (df['ccl5_log2fc']>df['saa1_log2fc'])    &
              (df['ccl5_log2fc']>df['myh11_log2fc'])]
ccl5_spe = ccl5_spe.sort_values('ccl5_log2fc', ascending=False)  # 43

myh11_spe = df[(df['myh11_log2fc']>np.log2(1.4))         & 
              (df['myh11_qval']<0.05)                   &
              (df['myh11_log2fc']>df['cthrc1_log2fc'])  &
              (df['myh11_log2fc']>df['col11a1_log2fc']) &
              (df['myh11_log2fc']>df['ccl5_log2fc'])    &
              (df['myh11_log2fc']>df['saa1_log2fc'])]
myh11_spe = myh11_spe.sort_values('myh11_log2fc', ascending=False)  # 19

cthrc1_spe['tf'].values
# 'HOXA11', 'HOXD10', 'HOXC11', 'HOXA10', 'HOXD11', 'LHX6', 'LHX8', 'MSC', 'TCF7L2', 'ZNF547'
# [TCF7L2]

df_spe = pd.concat([saa1_spe, cthrc1_spe, col11a1_spe, ccl5_spe, myh11_spe])
df_spe.to_csv('fibro_motifs_enrich_luz.txt', sep='\t', index=False)

## macro
macro = pd.read_table('macro_specific_enhancers_attention_raw_data.tsv', header=None)
macro.index = macro[0].values
macro = macro.iloc[:, 1:]
macro_ref = pd.read_table('macro_specific_enhancers_accessibility_raw_data.tsv', index_col=0)
macro.columns = macro_ref.columns

sns.heatmap(macro, cmap='coolwarm', vmin=-1, vmax=1, xticklabels=False)
plt.savefig('macro_specific_enhancers.png')
plt.close()

fcn1_peaks = macro.columns[macro.loc['Mono_FCN1']==1].values            # 11154
thbs1_peaks = macro.columns[macro.loc['Macro_THBS1']==1].values         # 14879
il32_peaks = macro.columns[macro.loc['Macro_IL32']==1].values           # 13056
c1qc_peaks = macro.columns[macro.loc['Macro_C1QC']==1].values           # 4613
spp1_peaks = macro.columns[macro.loc['Macro_SPP1']==1].values           # 8695
cdc20_peaks = macro.columns[macro.loc['Macro_CDC20']==1].values         # 52746
slpi_peaks = macro.columns[macro.loc['Macro_SLPI']==1].values           # 38076
apoc1_peaks = macro.columns[macro.loc['Macro_APOC1']==1].values         # 6089
col1a1_peaks = macro.columns[macro.loc['Macro_COL1A1']==1].values       # 54227

peaks_dict = {'fcn1': fcn1_peaks, 'thbs1': thbs1_peaks, 'il32': il32_peaks,
              'c1qc': c1qc_peaks, 'spp1': spp1_peaks, 'cdc20': cdc20_peaks,
              'slpi': slpi_peaks, 'apoc1': apoc1_peaks, 'col1a1': col1a1_peaks}
motifs_enrich = snap.tl.motif_enrichment(motifs=snap.datasets.cis_bp(unique=True),
                                         regions=peaks_dict,
                                         genome_fasta=snap.genome.hg38,
                                         background=None,
                                         method='hypergeometric')

with open('macro_motifs_enrich_luz.pkl', 'wb') as file:
    pickle.dump(motifs_enrich, file)

# with open('macro_motifs_enrich_luz.pkl', 'rb') as file:
#     motifs_enrich = pickle.load(file)

fcn1 = motifs_enrich['fcn1'].to_pandas()
thbs1 = motifs_enrich['thbs1'].to_pandas()
il32 = motifs_enrich['il32'].to_pandas()
c1qc = motifs_enrich['c1qc'].to_pandas()
spp1 = motifs_enrich['spp1'].to_pandas()
cdc20 = motifs_enrich['cdc20'].to_pandas()
slpi = motifs_enrich['slpi'].to_pandas()
apoc1 = motifs_enrich['apoc1'].to_pandas()
col1a1 = motifs_enrich['col1a1'].to_pandas()

df = pd.DataFrame({'tf':saa1['name'],
                   'fcn1_log2fc':fcn1['log2(fold change)'], 'fcn1_qval':fcn1['adjusted p-value'],
                   'thbs1_log2fc':thbs1['log2(fold change)'], 'thbs1_qval':thbs1['adjusted p-value'],
                   'il32_log2fc':il32['log2(fold change)'], 'il32_qval':il32['adjusted p-value'],
                   'c1qc_log2fc':c1qc['log2(fold change)'], 'c1qc_qval':c1qc['adjusted p-value'],
                   'spp1_log2fc':spp1['log2(fold change)'], 'spp1_qval':spp1['adjusted p-value'],
                   'cdc20_log2fc':cdc20['log2(fold change)'], 'cdc20_qval':cdc20['adjusted p-value'],
                   'slpi_log2fc':slpi['log2(fold change)'], 'slpi_qval':slpi['adjusted p-value'],
                   'apoc1_log2fc':apoc1['log2(fold change)'], 'apoc1_qval':apoc1['adjusted p-value'],
                   'col1a1_log2fc':col1a1['log2(fold change)'], 'col1a1_qval':col1a1['adjusted p-value']})

fcn1_spe = df[(df['fcn1_log2fc']>np.log2(1.15))         & 
              (df['fcn1_qval']<0.05)                   &
              (df['fcn1_log2fc']>df['thbs1_log2fc'])   &
              (df['fcn1_log2fc']>df['il32_log2fc'])    &
              (df['fcn1_log2fc']>df['c1qc_log2fc'])    &
              (df['fcn1_log2fc']>df['spp1_log2fc'])    &
              (df['fcn1_log2fc']>df['cdc20_log2fc'])   &
              (df['fcn1_log2fc']>df['slpi_log2fc'])    &
              (df['fcn1_log2fc']>df['apoc1_log2fc'])   &
              (df['fcn1_log2fc']>df['col1a1_log2fc'])]
fcn1_spe = fcn1_spe.sort_values('fcn1_log2fc', ascending=False)  # 65

thbs1_spe = df[(df['thbs1_log2fc']>np.log2(1.15))         & 
              (df['thbs1_qval']<0.05)                   &
              (df['thbs1_log2fc']>df['fcn1_log2fc'])   &
              (df['thbs1_log2fc']>df['il32_log2fc'])    &
              (df['thbs1_log2fc']>df['c1qc_log2fc'])    &
              (df['thbs1_log2fc']>df['spp1_log2fc'])    &
              (df['thbs1_log2fc']>df['cdc20_log2fc'])   &
              (df['thbs1_log2fc']>df['slpi_log2fc'])    &
              (df['thbs1_log2fc']>df['apoc1_log2fc'])   &
              (df['thbs1_log2fc']>df['col1a1_log2fc'])]
thbs1_spe = thbs1_spe.sort_values('thbs1_log2fc', ascending=False)  # 55

il32_spe = df[(df['il32_log2fc']>np.log2(1.15))         & 
              (df['il32_qval']<0.05)                   &
              (df['il32_log2fc']>df['thbs1_log2fc'])   &
              (df['il32_log2fc']>df['fcn1_log2fc'])    &
              (df['il32_log2fc']>df['c1qc_log2fc'])    &
              (df['il32_log2fc']>df['spp1_log2fc'])    &
              (df['il32_log2fc']>df['cdc20_log2fc'])   &
              (df['il32_log2fc']>df['slpi_log2fc'])    &
              (df['il32_log2fc']>df['apoc1_log2fc'])   &
              (df['il32_log2fc']>df['col1a1_log2fc'])]
il32_spe = il32_spe.sort_values('il32_log2fc', ascending=False)  # 59

c1qc_spe = df[(df['c1qc_log2fc']>np.log2(1.15))         & 
              (df['c1qc_qval']<0.05)                   &
              (df['c1qc_log2fc']>df['thbs1_log2fc'])   &
              (df['c1qc_log2fc']>df['il32_log2fc'])    &
              (df['c1qc_log2fc']>df['fcn1_log2fc'])    &
              (df['c1qc_log2fc']>df['spp1_log2fc'])    &
              (df['c1qc_log2fc']>df['cdc20_log2fc'])   &
              (df['c1qc_log2fc']>df['slpi_log2fc'])    &
              (df['c1qc_log2fc']>df['apoc1_log2fc'])   &
              (df['c1qc_log2fc']>df['col1a1_log2fc'])]
c1qc_spe = c1qc_spe.sort_values('c1qc_log2fc', ascending=False)  # 71

spp1_spe = df[(df['spp1_log2fc']>np.log2(1.15))         & 
              (df['spp1_qval']<0.05)                   &
              (df['spp1_log2fc']>df['thbs1_log2fc'])   &
              (df['spp1_log2fc']>df['il32_log2fc'])    &
              (df['spp1_log2fc']>df['c1qc_log2fc'])    &
              (df['spp1_log2fc']>df['fcn1_log2fc'])    &
              (df['spp1_log2fc']>df['cdc20_log2fc'])   &
              (df['spp1_log2fc']>df['slpi_log2fc'])    &
              (df['spp1_log2fc']>df['apoc1_log2fc'])   &
              (df['spp1_log2fc']>df['col1a1_log2fc'])]
spp1_spe = spp1_spe.sort_values('spp1_log2fc', ascending=False)  # 88

cdc20_spe = df[(df['cdc20_log2fc']>np.log2(1.15))         & 
              (df['cdc20_qval']<0.05)                   &
              (df['cdc20_log2fc']>df['thbs1_log2fc'])   &
              (df['cdc20_log2fc']>df['il32_log2fc'])    &
              (df['cdc20_log2fc']>df['c1qc_log2fc'])    &
              (df['cdc20_log2fc']>df['spp1_log2fc'])    &
              (df['cdc20_log2fc']>df['fcn1_log2fc'])   &
              (df['cdc20_log2fc']>df['slpi_log2fc'])    &
              (df['cdc20_log2fc']>df['apoc1_log2fc'])   &
              (df['cdc20_log2fc']>df['col1a1_log2fc'])]
cdc20_spe = cdc20_spe.sort_values('cdc20_log2fc', ascending=False)  # 84

slpi_spe = df[(df['slpi_log2fc']>np.log2(1.15))         & 
              (df['slpi_qval']<0.05)                   &
              (df['slpi_log2fc']>df['thbs1_log2fc'])   &
              (df['slpi_log2fc']>df['il32_log2fc'])    &
              (df['slpi_log2fc']>df['c1qc_log2fc'])    &
              (df['slpi_log2fc']>df['spp1_log2fc'])    &
              (df['slpi_log2fc']>df['cdc20_log2fc'])   &
              (df['slpi_log2fc']>df['fcn1_log2fc'])    &
              (df['slpi_log2fc']>df['apoc1_log2fc'])   &
              (df['slpi_log2fc']>df['col1a1_log2fc'])]
slpi_spe = slpi_spe.sort_values('slpi_log2fc', ascending=False)  # 18

apoc1_spe = df[(df['apoc1_log2fc']>np.log2(1.15))         & 
              (df['apoc1_qval']<0.05)                   &
              (df['apoc1_log2fc']>df['thbs1_log2fc'])   &
              (df['apoc1_log2fc']>df['il32_log2fc'])    &
              (df['apoc1_log2fc']>df['c1qc_log2fc'])    &
              (df['apoc1_log2fc']>df['spp1_log2fc'])    &
              (df['apoc1_log2fc']>df['cdc20_log2fc'])   &
              (df['apoc1_log2fc']>df['slpi_log2fc'])    &
              (df['apoc1_log2fc']>df['fcn1_log2fc'])   &
              (df['apoc1_log2fc']>df['col1a1_log2fc'])]
apoc1_spe = apoc1_spe.sort_values('apoc1_log2fc', ascending=False)  # 89

col1a1_spe = df[(df['col1a1_log2fc']>np.log2(1.15))         & 
              (df['col1a1_qval']<0.05)                   &
              (df['col1a1_log2fc']>df['thbs1_log2fc'])   &
              (df['col1a1_log2fc']>df['il32_log2fc'])    &
              (df['col1a1_log2fc']>df['c1qc_log2fc'])    &
              (df['col1a1_log2fc']>df['spp1_log2fc'])    &
              (df['col1a1_log2fc']>df['cdc20_log2fc'])   &
              (df['col1a1_log2fc']>df['slpi_log2fc'])    &
              (df['col1a1_log2fc']>df['apoc1_log2fc'])   &
              (df['col1a1_log2fc']>df['fcn1_log2fc'])]
col1a1_spe = col1a1_spe.sort_values('col1a1_log2fc', ascending=False)  # 56

slpi_spe['tf'].values
# 'TP73', 'ZEB1', 'ZNF284', 'ZBTB1', 'TCF4', 'KLF6', 'HES5', 'TFCP2L1', 'HES7',
# 'TFAP2B', 'ZEB2', 'MNT', 'SNAI1', 'TP53', 'SP3', 'TFAP2A', 'MYF5', 'HSF1'

df_spe = pd.concat([fcn1_spe, thbs1_spe, il32_spe, c1qc_spe, spp1_spe, cdc20_spe, slpi_spe, apoc1_spe, col1a1_spe])
df_spe.to_csv('macro_motifs_enrich_luz.txt', sep='\t', index=False)


## plot heatmap for enriched TFs in t cell & fibro & macro
import pandas as pd
import numpy as np
from plotnine import *
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['pdf.fonttype'] = 42

## t cell
df_raw = pd.read_table('t_cell_motifs_enrich_luz.txt')
df_raw.index = df_raw['tf'].values
df = df_raw[['naive_log2fc', 'effect_log2fc', 'ex_log2fc']]

sns.clustermap(df.T, cmap='coolwarm', row_cluster=False, col_cluster=False, z_score=1, vmin=-1.5, vmax=1.5, figsize=(40, 8))  # standard_scale=1, z_score=1
plt.savefig('t_cell_motifs_enrich_luz_heatmap.pdf')
plt.close()

## fibro
df_raw = pd.read_table('fibro_motifs_enrich_luz.txt')
df_raw.index = df_raw['tf'].values
df = df_raw[['saa1_log2fc', 'cthrc1_log2fc', 'col11a1_log2fc', 'ccl5_log2fc', 'myh11_log2fc']]

sns.clustermap(df.T, cmap='coolwarm', row_cluster=False, col_cluster=False, z_score=1, vmin=-1.5, vmax=1.5, figsize=(40, 8))  # standard_scale=1, z_score=1
plt.savefig('fibro_motifs_enrich_luz_heatmap.pdf')
plt.close()

## macro
df_raw = pd.read_table('macro_motifs_enrich_luz.txt')
df_raw.index = df_raw['tf'].values
df = df_raw[['fcn1_log2fc', 'thbs1_log2fc', 'il32_log2fc', 'c1qc_log2fc', 'spp1_log2fc', 'cdc20_log2fc', 'slpi_log2fc', 'apoc1_log2fc', 'col1a1_log2fc']]

sns.clustermap(df.T, cmap='coolwarm', row_cluster=False, col_cluster=False, z_score=1, vmin=-1.5, vmax=1.5, figsize=(40, 8))  # standard_scale=1, z_score=1
plt.savefig('macro_motifs_enrich_luz_heatmap.pdf')
plt.close()
