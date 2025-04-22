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





with open('motif_enrichment.pk', 'rb') as file:
    motif_info = pickle.load(file)
