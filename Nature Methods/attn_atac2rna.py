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


## marker gene heatmap
import pandas as pd
import matplotlib.pyplot as plt
from plotnine import *
import seaborn as sns

plt.rcParams['pdf.fonttype'] = 42

df = pd.read_table('marker_gene_pair_attention_raw_data.tsv', sep='\t', index_col=0).drop_duplicates()
df.index = df.index.map(lambda x:x.split(' & ')[1])
df.columns = ['tumor_b', 't_cell', 'normal_b', 'mono']
df = df.loc[:, ['t_cell', 'mono', 'normal_b', 'tumor_b']]

t_cell_marker_lst = ['PRKCH', 'FYN', 'BCL11B', 'INPP4B', 'CELF2', 'CD247', 'MBNL1', 'CBLB']  # THEMIS  LINC01934
monocyte_marker_lst = ['SLC8A1', 'RBM47', 'LRMDA', 'TYMP', 'DMXL2', 'DPYD', 'TNS3', 'MCTP1', 'PLXDC2', 'FMNL2']
normal_b_marker_lst = ['BANK1', 'RALGPS2', 'CCSER1', 'MEF2C', 'FCRL1', 'FCHSD2', 'MS4A1', 'AFF3', 'CDK14', 'RIPOR2']
tumor_b_marker_lst = ['TCF4', 'PAX5', 'ARHGAP24', 'AFF3', 'FCRL5', 'GPM6A', 'LINC01320', 'NIBAN3', 'FOXP1', 'RUBCNL']

df_t_cell = df.loc[t_cell_marker_lst]
df_t_cell = df_t_cell[df_t_cell.sum(axis=1)!=0]
df_t_cell = df_t_cell[df_t_cell.idxmax(axis=1)=='t_cell']  # 5

df_mono = df.loc[monocyte_marker_lst]
df_mono = df_mono[df_mono.sum(axis=1)!=0]
df_mono = df_mono[df_mono.idxmax(axis=1)=='mono']   # 10

df_normal_b = df.loc[normal_b_marker_lst]
df_normal_b = df_normal_b[df_normal_b.sum(axis=1)!=0]
df_normal_b = df_normal_b[df_normal_b.idxmax(axis=1)=='normal_b']  # 4

df_tumor_b = df.loc[tumor_b_marker_lst]
df_tumor_b = df_tumor_b[df_tumor_b.sum(axis=1)!=0]
df_tumor_b = df_tumor_b[df_tumor_b.idxmax(axis=1)=='tumor_b']  # 9

df = pd.concat([df_t_cell, df_mono, df_normal_b, df_tumor_b])  # 28
df = df[df.index!='ARHGAP24']


plt.figure(figsize=(6, 8))
#sns.clustermap(df, cmap='bwr', vmin=0, vmax=4, center=1, xticklabels=True, yticklabels=True, annot=True, fmt='.2g')

sns.clustermap(df, cmap='bwr', row_cluster=False, col_cluster=False, z_score=0, vmin=-1.5, vmax=1.5)   # vmin=0, vmax=0.01, yticklabels=False, figsize=[40, 5])

plt.savefig('marker_gene_pair_attention.pdf')
plt.close()





import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from plotnine import *

plt.rcParams['pdf.fonttype'] = 42

def set_bin(x):
    if x>10:
        return '11'
    else:
        return str(x)

# ct_lst = ['T-cell', 'Monocyte', 'Normal-B-cell', 'Tumor-B-cell']

## t cell
with open('Relation-of-Enhancer-Number-and-GEX-of-T-cell-in-Tumor_B-dataset.pdf_raw_data.pkl', 'rb') as file:
    df_t_cell = pickle.load(file)[9]
df_t_cell['num_enhancer'] = df_t_cell['num_enhancer'].apply(set_bin)
df_t_cell['num_enhancer'] = pd.Categorical(df_t_cell['num_enhancer'], categories=[str(i) for i in range(12)])

p = ggplot(df_t_cell, aes(x='num_enhancer', y='exp', fill='num_enhancer')) + geom_boxplot(width=0.5, show_legend=False, outlier_shape='') + xlab('') +\
                                                                             coord_cartesian(ylim=(0, 0.5)) +\
                                                                             scale_y_continuous(breaks=np.arange(0, 0.5+0.1, 0.1)) + theme_bw()
p.save(filename='gex_enhancer_num_t_cell.pdf', dpi=600, height=4, width=4)

## monocyte
with open('Relation-of-Enhancer-Number-and-GEX-of-Monocyte-in-Tumor_B-dataset.pdf_raw_data.pkl', 'rb') as file:
    df_mono = pickle.load(file)[9]
df_mono['num_enhancer'] = df_mono['num_enhancer'].apply(set_bin)
df_mono['num_enhancer'] = pd.Categorical(df_mono['num_enhancer'], categories=[str(i) for i in range(12)])

p = ggplot(df_mono, aes(x='num_enhancer', y='exp', fill='num_enhancer')) + geom_boxplot(width=0.5, show_legend=False, outlier_shape='') + xlab('') +\
                                                                           coord_cartesian(ylim=(0, 1.0)) +\
                                                                           scale_y_continuous(breaks=np.arange(0, 1.0+0.1, 0.2)) + theme_bw()
p.save(filename='gex_enhancer_num_mono.pdf', dpi=600, height=4, width=4)

## normal b
with open('Relation-of-Enhancer-Number-and-GEX-of-Normal-B-cell-in-Tumor_B-dataset.pdf_raw_data.pkl', 'rb') as file:
    df_normal_b = pickle.load(file)[9]
df_normal_b['num_enhancer'] = df_normal_b['num_enhancer'].apply(set_bin)
df_normal_b['num_enhancer'] = pd.Categorical(df_normal_b['num_enhancer'], categories=[str(i) for i in range(12)])

p = ggplot(df_normal_b, aes(x='num_enhancer', y='exp', fill='num_enhancer')) + geom_boxplot(width=0.5, show_legend=False, outlier_shape='') + xlab('') +\
                                                                               coord_cartesian(ylim=(0, 0.8)) +\
                                                                               scale_y_continuous(breaks=np.arange(0, 0.8+0.1, 0.2)) + theme_bw()
p.save(filename='gex_enhancer_num_normal_b.pdf', dpi=600, height=4, width=4)

## tumor b
with open('Relation-of-Enhancer-Number-and-GEX-of-Tumor-B-cell-in-Tumor_B-dataset.pdf_raw_data.pkl', 'rb') as file:
    df_tumor_b = pickle.load(file)[9]
df_tumor_b['num_enhancer'] = df_tumor_b['num_enhancer'].apply(set_bin)
df_tumor_b['num_enhancer'] = pd.Categorical(df_tumor_b['num_enhancer'], categories=[str(i) for i in range(12)])

p = ggplot(df_tumor_b, aes(x='num_enhancer', y='exp', fill='num_enhancer')) + geom_boxplot(width=0.5, show_legend=False, outlier_shape='') + xlab('') +\
                                                                              coord_cartesian(ylim=(0, 1.4)) +\
                                                                              scale_y_continuous(breaks=np.arange(0, 1.4+0.1, 0.2)) + theme_bw()
p.save(filename='gex_enhancer_num_tumor_b.pdf', dpi=600, height=4, width=4)

















