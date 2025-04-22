## workdir: /fs/home/jiluzhang/Nature_methods/Figure_2/pan_cancer/cisformer/atac2rna
import pandas as pd
import numpy as np
from plotnine import *
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

plt.rcParams['pdf.fonttype'] = 42

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
