## workdir: /fs/home/jiluzhang/Nature_methods/attn_atac2rna

## immune snp enrichment
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from plotnine import *
import seaborn as sns

plt.rcParams['pdf.fonttype'] = 42

df = pd.read_table('GWAS_inside_immue_raw_data.tsv', index_col=0)
df = np.exp(df)-1
df = df[df.sum(axis=1)!=0]
df = df.loc[['White blood cell count', 'Red blood cell count',
             'Lymphocyte count', 'Monocyte count', 'Neutrophil count',
             'Basophil count', 'Eosinophil count'],
             ['T cell', 'Monocyte', 'Normal B cell', 'Tumor B cell']]
df = np.log2(df+1)

plt.figure(figsize=(8, 6))
sns.heatmap(df, cmap='bwr', vmin=0, vmax=4, center=1, xticklabels=True, yticklabels=True, annot=True, fmt='.2g')
plt.savefig('GWAS_inside_immue_heatmap.pdf')
plt.close()


## igv track (generate bedgraph)
import pandas as pd

df = pd.read_table('Tumor_B_CD19_100kbp_Chromatin_Accessibility_Stacked_Area_Plot_raw_data.tsv', index_col=0)

pd.DataFrame({'chrom':'chr16', 'start': df.index.values, 'end': df.index.values+1, 'val': df['T cell']}).to_csv('t_cell_atac.bedgraph', sep='\t', index=False, header=False)
pd.DataFrame({'chrom':'chr16', 'start': df.index.values, 'end': df.index.values+1, 'val': df['Monocyte']}).to_csv('mono_atac.bedgraph', sep='\t', index=False, header=False)
pd.DataFrame({'chrom':'chr16', 'start': df.index.values, 'end': df.index.values+1, 'val': df['Normal B cell']}).to_csv('normal_b_atac.bedgraph', sep='\t', index=False, header=False)
pd.DataFrame({'chrom':'chr16', 'start': df.index.values, 'end': df.index.values+1, 'val': df['Tumor B cell']}).to_csv('tumor_b_atac.bedgraph', sep='\t', index=False, header=False)

pd.DataFrame({'chrom':'chr16', 'start': df.index.values, 'end': df.index.values+1, 'val': df['T cell_Attention']}).to_csv('t_cell_attn.bedgraph', sep='\t', index=False, header=False)
pd.DataFrame({'chrom':'chr16', 'start': df.index.values, 'end': df.index.values+1, 'val': df['Monocyte_Attention']}).to_csv('mono_attn.bedgraph', sep='\t', index=False, header=False)
pd.DataFrame({'chrom':'chr16', 'start': df.index.values, 'end': df.index.values+1, 'val': df['Normal B cell_Attention']}).to_csv('normal_b_attn.bedgraph', sep='\t', index=False, header=False)
pd.DataFrame({'chrom':'chr16', 'start': df.index.values, 'end': df.index.values+1, 'val': df['Tumor B cell_Attention']}).to_csv('tumor_b_attn.bedgraph', sep='\t', index=False, header=False)
pd.DataFrame({'chrom':'chr16', 'start': df.index.values, 'end': df.index.values+1, 'val': df['total']}).to_csv('total_attn.bedgraph', sep='\t', index=False, header=False)

# tar -czvf atac_attn_cd19_bedgraph.tar *.bedgraph


import pickle

with open('Relation-of-Enhancer-Number-and-GEX-of-Monocyte-in-Tumor_B-dataset.pdf_raw_data.pkl', 'rb') as file:
    dat = pickle.load(file)
