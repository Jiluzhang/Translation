## workdir: /fs/home/jiluzhang/Nature_methods/TF_binding_2

## scan motif with motifmatchr
# conda remove -n motifmatchr --all
# conda create -n motifmatchr
# conda activate motifmatchr
# conda install bioconda::bioconductor-dirichletmultinomial
# install.packages("BiocManager")
# BiocManager::install("motifmatchr")
# install.packages('https://cran.r-project.org/src/contrib/Archive/lattice/lattice_0.20-40.tar.gz')
# install.packages('https://cran.r-project.org/src/contrib/Archive/Matrix/Matrix_1.4-1.tar.gz')
# BiocManager::install("SummarizedExperiment")
# install.packages('https://cran.r-project.org/src/contrib/Archive/cpp11/cpp11_0.4.0.tar.gz')
# install.packages('RSQLite')
# conda install conda-forge::r-xml
# BiocManager::install('rtracklayer')
# install.packages('https://cran.r-project.org/src/contrib/Archive/tzdb/tzdb_0.1.0.tar.gz')
# install.packages('https://cran.r-project.org/src/contrib/Archive/vroom/vroom_1.0.2.tar.gz')
# install.packages('https://cran.r-project.org/src/contrib/Archive/readr/readr_1.3.1.tar.gz')
# install.packages('https://cran.r-project.org/src/contrib/Archive/gtable/gtable_0.3.0.tar.gz')
# install.packages('https://cran.r-project.org/src/contrib/Archive/MASS/MASS_7.3-51.5.tar.gz')
# BiocManager::install("motifmatchr")

# BiocManager::install("JASPAR2018")
# BiocManager::install("BSgenome.Hsapiens.UCSC.hg38")

## mannual: https://bioconductor.org/packages/release/bioc/vignettes/motifmatchr/inst/doc/motifmatchr.html
library(motifmatchr)
library(GenomicRanges)
library(JASPAR2018)

motifs <- TFBSTools::getMatrixSet(JASPAR2018, list('species'='Homo sapiens', 'collection'='CORE'))
if (!isTRUE(all.equal(TFBSTools::name(motifs), names(motifs))))
    names(motifs) <- paste(names(motifs), TFBSTools::name(motifs), sep="_")

peaks <- rtracklayer::import.bed('peaks.bed')
motif_ix <- matchMotifs(motif, peaks, genome="hg38", out='scores')
motif_mat <- motifCounts(motif_ix)

df <- data.frame(as.matrix(motif_mat))
write.table(df, 'peaks_motifs_count.txt', row.names=FALSE, col.names=TRUE, sep='\t')   # 204Mb


## Attention based chromatin accessibility regulator screening & ranking
import pandas as pd
import numpy as np
import random
from functools import partial
import sys
sys.path.append("M2Mmodel")
from utils import PairDataset
import scanpy as sc
import torch
from M2Mmodel.utils import *
from collections import Counter
import pickle as pkl
import yaml
from M2Mmodel.M2M import M2M_rna2atac

import h5py
from tqdm import tqdm

from multiprocessing import Pool
import pickle

# from config
with open("rna2atac_config_test.yaml", "r") as f:
    config = yaml.safe_load(f)

model = M2M_rna2atac(
            dim = config["model"].get("dim"),

            enc_num_gene_tokens = config["model"].get("total_gene") + 1, # +1 for <PAD>
            enc_num_value_tokens = config["model"].get('max_express') + 1, # +1 for <PAD>
            enc_depth = config["model"].get("enc_depth"),
            enc_heads = config["model"].get("enc_heads"),
            enc_ff_mult = config["model"].get("enc_ff_mult"),
            enc_dim_head = config["model"].get("enc_dim_head"),
            enc_emb_dropout = config["model"].get("enc_emb_dropout"),
            enc_ff_dropout = config["model"].get("enc_ff_dropout"),
            enc_attn_dropout = config["model"].get("enc_attn_dropout"),

            dec_depth = config["model"].get("dec_depth"),
            dec_heads = config["model"].get("dec_heads"),
            dec_ff_mult = config["model"].get("dec_ff_mult"),
            dec_dim_head = config["model"].get("dec_dim_head"),
            dec_emb_dropout = config["model"].get("dec_emb_dropout"),
            dec_ff_dropout = config["model"].get("dec_ff_dropout"),
            dec_attn_dropout = config["model"].get("dec_attn_dropout")
        )
model = model.half()
model.load_state_dict(torch.load('/fs/home/jiluzhang/Nature_methods/Figure_1/scenario_1/cisformer/save_mlt_40_large/2024-09-30_rna2atac_pbmc_34/pytorch_model.bin'))
device = torch.device('cuda:3')
model.to(device)
model.eval()

rna = sc.read_h5ad('rna_train.h5ad')  # 6974 × 16706
rna.obs['cell_anno'].value_counts()
# CD14 Mono           1988
# CD8 Naive            964
# CD4 TCM              793
# CD4 Naive            734
# CD4 intermediate     363
# CD8 TEM              357
# CD4 TEM              335
# B memory             317
# NK                   306
# CD16 Mono            266
# B naive              229
# cDC2                  88
# MAIT                  86
# Treg                  83
# pDC                   65

atac = sc.read_h5ad('atac_train.h5ad')   # 6974 × 236295

peaks = atac.var[['gene_ids']]
peaks['chr'] = peaks['gene_ids'].map(lambda x: x.split(':')[0])
peaks['start'] = peaks['gene_ids'].map(lambda x: x.split(':')[1].split('-')[0])
peaks['end'] = peaks['gene_ids'].map(lambda x: x.split(':')[1].split('-')[1])
peaks[['chr', 'start', 'end']].to_csv('peaks.bed', index=False, header=False, sep='\t')

################ CD14 Mono ################
random.seed(0)
idx = list(np.argwhere(rna.obs['cell_anno']=='CD14 Mono').flatten())
idx_20 = random.sample(list(idx), 20)
rna[idx_20].write('rna_cd14_mono_20.h5ad')
np.count_nonzero(rna[idx_20].X.toarray(), axis=1).max()  # 2726

atac[idx_20].write('atac_cd14_mono_20.h5ad')

# python data_preprocess.py -r rna_cd14_mono_20.h5ad -a atac_cd14_mono_20.h5ad -s pt_cd14_mono_20 --dt test -n cd14_mono_20 --config rna2atac_config_test_cd14_mono_20.yaml
# enc_max_len: 2726

cd14_mono_data = torch.load("./pt_cd14_mono_20/cd14_mono_20_0.pt")
dataset = PreDataset(cd14_mono_data)
dataloader_kwargs = {'batch_size': 1, 'shuffle': False}
loader = torch.utils.data.DataLoader(dataset, **dataloader_kwargs)

# time: 1 min
os.makedirs('./attn/cd14_mono')
i=0
for inputs in tqdm(loader, ncols=80, desc='output attention matrix'):
    rna_sequence, rna_value, atac_sequence, _, enc_pad_mask = [each.to(device) for each in inputs]
    attn = model.generate_attn_weight(rna_sequence, atac_sequence, rna_value, enc_mask=enc_pad_mask, which='decoder')
    attn = attn[0]  # attn = attn[0].to(torch.float16)
    with h5py.File('attn/cd14_mono/attn_cd14_mono_'+str(i)+'.h5', 'w') as f:
        f.create_dataset('attn', data=attn)
    torch.cuda.empty_cache()
    i += 1

rna_cd14_mono_20 = sc.read_h5ad('rna_cd14_mono_20.h5ad')
nonzero = np.count_nonzero(rna_cd14_mono_20.X.toarray(), axis=0)
gene_lst = rna_cd14_mono_20.var.index[nonzero>=3]  # 4722

atac_cd14_mono_20 = sc.read_h5ad('atac_cd14_mono_20.h5ad')

gene_attn = {}
for gene in gene_lst:
    gene_attn[gene] = np.zeros([atac_cd14_mono_20.shape[1], 1], dtype='float32')  # not float16

# 10s/cell
for i in range(20):
    rna_sequence = cd14_mono_data[0][i].flatten()
    with h5py.File('attn/cd14_mono/attn_cd14_mono_'+str(i)+'.h5', 'r') as f:
        attn = f['attn'][:]
        for gene in tqdm(gene_lst, ncols=80, desc='cell '+str(i)):
            idx = torch.argwhere((rna_sequence==(np.argwhere(rna_cd14_mono_20.var.index==gene))[0][0]+1).flatten())
            if len(idx)!=0:
                gene_attn[gene] += attn[:, [idx.item()]]

with open('./attn/cd14_mono/attn_20_cd14_mono_3cell.pkl', 'wb') as file:
    pickle.dump(gene_attn, file)

with open('./attn/cd14_mono/attn_20_cd14_mono_3cell.pkl', 'rb') as file:
    gene_attn = pickle.load(file)

for gene in gene_attn:
    gene_attn[gene] = gene_attn[gene].flatten()

gene_attn_df = pd.DataFrame(gene_attn)   # 236295 x 4722
gene_attn_df.index = atac_cd14_mono_20.var.index.values

gene_attn_df_avg = gene_attn_df.mean(axis=1)/20

motif_info_raw = pd.read_table('peaks_motifs_count.txt', header=0)
motif_info = motif_info_raw.copy()
motif_info[motif_info!=0] = 1
motif_info = motif_info.sum(axis=1)
motif_info.index = gene_attn_df.index.values

df = pd.DataFrame({'attn':gene_attn_df_avg, 'motif':motif_info})
df['motif_bin'] = '0'
df.loc[((df['motif']>0) & (df['motif']<=50)), 'motif_bin'] = '1'
df.loc[(df['motif']>50), 'motif_bin'] = '2'
# df['attn'] = np.log10(df['attn']/df['attn'].min())
# df['attn'] = (df['attn']-df['attn'].min())/(df['attn'].max()-df['attn'].min())
# df['attn'] = (df['attn']-df['attn'].mean())/df['attn'].std()
df[['attn', 'motif_bin']].to_csv('motif_attn_cd14_mono.csv')

df['attn'] = np.log10(df['attn']/df['attn'].min())
df['attn'] = (df['attn']-df['attn'].min())/(df['attn'].max()-df['attn'].min())
df[['attn', 'motif_bin']].to_csv('motif_attn_cd14_mono_norm.csv')


## plot box
from plotnine import *
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import stats

plt.rcParams['pdf.fonttype'] = 42

## no norm
df = pd.read_csv('motif_attn_cd14_mono.csv', index_col=0)
df['motif_bin'] = df['motif_bin'].apply(str)
df = df.replace([np.inf, -np.inf], np.nan).dropna()
df = df[df['attn']!=0]
df['attn'] = df['attn']*1000
df = df[df['motif_bin']!='1']  # 'mid' is not obvious

# p = ggplot(df, aes(x='motif_bin', y='attn', fill='motif_bin')) + geom_boxplot(width=0.5, show_legend=False, outlier_shape='') + xlab('') +\
#                                                                  scale_y_continuous(limits=[0.00, 0.10], breaks=np.arange(0.00, 0.10+0.01, 0.02)) + theme_bw()
p = ggplot(df, aes(x='motif_bin', y='attn', fill='motif_bin')) + geom_boxplot(width=0.5, show_legend=False, outlier_shape='') + xlab('') +\
                                                                 coord_cartesian(ylim=(0.17, 0.21)) + theme_bw()
p.save(filename='motif_attn_cd14_mono.pdf', dpi=600, height=4, width=2)

df[df['motif_bin']=='0']['attn'].median()   # 0.19093353999999998
df[df['motif_bin']=='2']['attn'].median()   # 0.19142382000000002

stats.ttest_ind(df[df['motif_bin']=='0']['attn'], df[df['motif_bin']=='2']['attn'])  # pvalue=7.804527617913404e-05

## with norm
df = pd.read_csv('motif_attn_cd14_mono_norm.csv', index_col=0)
df['motif_bin'] = df['motif_bin'].apply(str)
df = df.replace([np.inf, -np.inf], np.nan).dropna()
df = df[df['attn']!=0]
df = df[df['motif_bin']!='1']  # 'mid' is not obvious

p = ggplot(df, aes(x='motif_bin', y='attn', fill='motif_bin')) + geom_boxplot(width=0.5, show_legend=False, outlier_shape='') + xlab('') +\
                                                                 coord_cartesian(ylim=(0.948, 0.952)) +\
                                                                 scale_y_continuous(breaks=np.arange(0.948, 0.952+0.001, 0.001)) + theme_bw()
p.save(filename='motif_attn_cd14_mono_norm.pdf', dpi=600, height=4, width=2)

df[df['motif_bin']=='0']['attn'].median()   # 0.9502866
df[df['motif_bin']=='2']['attn'].median()   # 0.950325875

stats.ttest_ind(df[df['motif_bin']=='0']['attn'], df[df['motif_bin']=='2']['attn'])  # pvalue=0.0019024368003007642



################ CD8 Naive ################
random.seed(0)
idx = list(np.argwhere(rna.obs['cell_anno']=='CD8 Naive').flatten())
idx_20 = random.sample(list(idx), 20)
rna[idx_20].write('rna_cd8_naive_20.h5ad')
np.count_nonzero(rna[idx_20].X.toarray(), axis=1).max()  # 2008

atac[idx_20].write('atac_cd8_naive_20.h5ad')

# python data_preprocess.py -r rna_cd8_naive_20.h5ad -a atac_cd8_naive_20.h5ad -s pt_cd8_naive_20 --dt test -n cd8_naive_20 --config rna2atac_config_test_cd8_naive_20.yaml
# enc_max_len: 2008

cd8_naive_data = torch.load("./pt_cd8_naive_20/cd8_naive_20_0.pt")
dataset = PreDataset(cd8_naive_data)
dataloader_kwargs = {'batch_size': 1, 'shuffle': False}
loader = torch.utils.data.DataLoader(dataset, **dataloader_kwargs)

# time: 1 min
os.makedirs('./attn/cd8_naive')
i=0
for inputs in tqdm(loader, ncols=80, desc='output attention matrix'):
    rna_sequence, rna_value, atac_sequence, _, enc_pad_mask = [each.to(device) for each in inputs]
    attn = model.generate_attn_weight(rna_sequence, atac_sequence, rna_value, enc_mask=enc_pad_mask, which='decoder')
    attn = attn[0]  # attn = attn[0].to(torch.float16)
    with h5py.File('attn/cd8_naive/attn_cd8_naive_'+str(i)+'.h5', 'w') as f:
        f.create_dataset('attn', data=attn)
    torch.cuda.empty_cache()
    i += 1

rna_cd8_naive_20 = sc.read_h5ad('rna_cd8_naive_20.h5ad')
nonzero = np.count_nonzero(rna_cd8_naive_20.X.toarray(), axis=0)
gene_lst = rna_cd8_naive_20.var.index[nonzero>=3]  # 4086

atac_cd8_naive_20 = sc.read_h5ad('atac_cd8_naive_20.h5ad')

gene_attn = {}
for gene in gene_lst:
    gene_attn[gene] = np.zeros([atac_cd8_naive_20.shape[1], 1], dtype='float32')  # not float16

# 10s/cell
for i in range(20):
    rna_sequence = cd8_naive_data[0][i].flatten()
    with h5py.File('attn/cd8_naive/attn_cd8_naive_'+str(i)+'.h5', 'r') as f:
        attn = f['attn'][:]
        for gene in tqdm(gene_lst, ncols=80, desc='cell '+str(i)):
            idx = torch.argwhere((rna_sequence==(np.argwhere(rna_cd8_naive_20.var.index==gene))[0][0]+1).flatten())
            if len(idx)!=0:
                gene_attn[gene] += attn[:, [idx.item()]]

with open('./attn/cd8_naive/attn_20_cd8_naive_3cell.pkl', 'wb') as file:
    pickle.dump(gene_attn, file)

with open('./attn/cd8_naive/attn_20_cd8_naive_3cell.pkl', 'rb') as file:
    gene_attn = pickle.load(file)

for gene in gene_attn:
    gene_attn[gene] = gene_attn[gene].flatten()

gene_attn_df = pd.DataFrame(gene_attn)   # 236295 x 4086
gene_attn_df.index = atac_cd8_naive_20.var.index.values

gene_attn_df_avg = gene_attn_df.mean(axis=1)/20

motif_info_raw = pd.read_table('peaks_motifs_count.txt', header=0)
motif_info = motif_info_raw.copy()
motif_info[motif_info!=0] = 1
motif_info = motif_info.sum(axis=1)
motif_info.index = gene_attn_df.index.values

df = pd.DataFrame({'attn':gene_attn_df_avg, 'motif':motif_info})
df['motif_bin'] = '0'
df.loc[((df['motif']>0) & (df['motif']<=50)), 'motif_bin'] = '1'
df.loc[(df['motif']>50), 'motif_bin'] = '2'
# df['attn'] = np.log10(df['attn']/df['attn'].min())
# df['attn'] = (df['attn']-df['attn'].min())/(df['attn'].max()-df['attn'].min())
# df['attn'] = (df['attn']-df['attn'].mean())/df['attn'].std()
df[['attn', 'motif_bin']].to_csv('motif_attn_cd8_naive.csv')

df['attn'] = np.log10(df['attn']/df['attn'].min())
df['attn'] = (df['attn']-df['attn'].min())/(df['attn'].max()-df['attn'].min())
df[['attn', 'motif_bin']].to_csv('motif_attn_cd8_naive_norm.csv')

## plot box
from plotnine import *
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import stats

plt.rcParams['pdf.fonttype'] = 42

## no norm
df = pd.read_csv('motif_attn_cd8_naive.csv', index_col=0)
df['motif_bin'] = df['motif_bin'].apply(str)
df = df.replace([np.inf, -np.inf], np.nan).dropna()
df = df[df['attn']!=0]
df['attn'] = df['attn']*1000
df = df[df['motif_bin']!='1']  # 'mid' is not obvious

# p = ggplot(df, aes(x='motif_bin', y='attn', fill='motif_bin')) + geom_boxplot(width=0.5, show_legend=False, outlier_shape='') + xlab('') +\
#                                                                  scale_y_continuous(limits=[0.00, 0.10], breaks=np.arange(0.00, 0.10+0.01, 0.02)) + theme_bw()
p = ggplot(df, aes(x='motif_bin', y='attn', fill='motif_bin')) + geom_boxplot(width=0.5, show_legend=False, outlier_shape='') + xlab('') +\
                                                                 coord_cartesian(ylim=(0.16, 0.24)) + theme_bw()
p.save(filename='motif_attn_cd8_naive.pdf', dpi=600, height=4, width=2)

df[df['motif_bin']=='0']['attn'].median()   # 0.1947358
df[df['motif_bin']=='2']['attn'].median()   # 0.19708356999999999

stats.ttest_ind(df[df['motif_bin']=='0']['attn'], df[df['motif_bin']=='2']['attn'])  # pvalue=1.621873722532627e-06

## with norm
df = pd.read_csv('motif_attn_cd8_naive_norm.csv', index_col=0)
df['motif_bin'] = df['motif_bin'].apply(str)
df = df.replace([np.inf, -np.inf], np.nan).dropna()
df = df[df['attn']!=0]
df = df[df['motif_bin']!='1']  # 'mid' is not obvious

# p = ggplot(df, aes(x='motif_bin', y='attn', fill='motif_bin')) + geom_boxplot(width=0.5, show_legend=False, outlier_shape='') + xlab('') +\
#                                                                  scale_y_continuous(limits=[0.00, 0.10], breaks=np.arange(0.00, 0.10+0.01, 0.02)) + theme_bw()
p = ggplot(df, aes(x='motif_bin', y='attn', fill='motif_bin')) + geom_boxplot(width=0.5, show_legend=False, outlier_shape='') + xlab('') +\
                                                                 coord_cartesian(ylim=(0.951, 0.957)) +\
                                                                 scale_y_continuous(breaks=np.arange(0.951, 0.957+0.001, 0.001)) + theme_bw()
p.save(filename='motif_attn_cd8_naive_norm.pdf', dpi=600, height=4, width=2)

df[df['motif_bin']=='0']['attn'].median()   # 0.9537589
df[df['motif_bin']=='2']['attn'].median()   # 0.95392907

stats.ttest_ind(df[df['motif_bin']=='0']['attn'], df[df['motif_bin']=='2']['attn'])  # pvalue=0.0018312685962712527

################ CD4 TCM ################
random.seed(0)
idx = list(np.argwhere(rna.obs['cell_anno']=='CD4 TCM').flatten())
idx_20 = random.sample(list(idx), 20)
rna[idx_20].write('rna_cd4_tcm_20.h5ad')
np.count_nonzero(rna[idx_20].X.toarray(), axis=1).max()  # 2363

atac[idx_20].write('atac_cd4_tcm_20.h5ad')

# python data_preprocess.py -r rna_cd4_tcm_20.h5ad -a atac_cd4_tcm_20.h5ad -s pt_cd4_tcm_20 --dt test -n cd4_tcm_20 --config rna2atac_config_test_cd4_tcm_20.yaml
# enc_max_len: 2363

cd4_tcm_data = torch.load("./pt_cd4_tcm_20/cd4_tcm_20_0.pt")
dataset = PreDataset(cd4_tcm_data)
dataloader_kwargs = {'batch_size': 1, 'shuffle': False}
loader = torch.utils.data.DataLoader(dataset, **dataloader_kwargs)

# time: 1 min
os.makedirs('./attn/cd4_tcm')
i=0
for inputs in tqdm(loader, ncols=80, desc='output attention matrix'):
    rna_sequence, rna_value, atac_sequence, _, enc_pad_mask = [each.to(device) for each in inputs]
    attn = model.generate_attn_weight(rna_sequence, atac_sequence, rna_value, enc_mask=enc_pad_mask, which='decoder')
    attn = attn[0]  # attn = attn[0].to(torch.float16)
    with h5py.File('attn/cd4_tcm/attn_cd4_tcm_'+str(i)+'.h5', 'w') as f:
        f.create_dataset('attn', data=attn)
    torch.cuda.empty_cache()
    i += 1

rna_cd4_tcm_20 = sc.read_h5ad('rna_cd4_tcm_20.h5ad')
nonzero = np.count_nonzero(rna_cd4_tcm_20.X.toarray(), axis=0)
gene_lst = rna_cd4_tcm_20.var.index[nonzero>=3]  # 4287

atac_cd4_tcm_20 = sc.read_h5ad('atac_cd4_tcm_20.h5ad')

gene_attn = {}
for gene in gene_lst:
    gene_attn[gene] = np.zeros([atac_cd4_tcm_20.shape[1], 1], dtype='float32')  # not float16

# 10s/cell
for i in range(20):
    rna_sequence = cd4_tcm_data[0][i].flatten()
    with h5py.File('attn/cd4_tcm/attn_cd4_tcm_'+str(i)+'.h5', 'r') as f:
        attn = f['attn'][:]
        for gene in tqdm(gene_lst, ncols=80, desc='cell '+str(i)):
            idx = torch.argwhere((rna_sequence==(np.argwhere(rna_cd4_tcm_20.var.index==gene))[0][0]+1).flatten())
            if len(idx)!=0:
                gene_attn[gene] += attn[:, [idx.item()]]

with open('./attn/cd4_tcm/attn_20_cd4_tcm_3cell.pkl', 'wb') as file:
    pickle.dump(gene_attn, file)

with open('./attn/cd4_tcm/attn_20_cd4_tcm_3cell.pkl', 'rb') as file:
    gene_attn = pickle.load(file)

for gene in gene_attn:
    gene_attn[gene] = gene_attn[gene].flatten()

gene_attn_df = pd.DataFrame(gene_attn)   # 236295 x 4287
gene_attn_df.index = atac_cd4_tcm_20.var.index.values

gene_attn_df_avg = gene_attn_df.mean(axis=1)/20

motif_info_raw = pd.read_table('peaks_motifs_count.txt', header=0)
motif_info = motif_info_raw.copy()
motif_info[motif_info!=0] = 1
motif_info = motif_info.sum(axis=1)
motif_info.index = gene_attn_df.index.values

df = pd.DataFrame({'attn':gene_attn_df_avg, 'motif':motif_info})
df['motif_bin'] = '0'
df.loc[((df['motif']>0) & (df['motif']<=50)), 'motif_bin'] = '1'
df.loc[(df['motif']>50), 'motif_bin'] = '2'
# df['attn'] = np.log10(df['attn']/df['attn'].min())
# df['attn'] = (df['attn']-df['attn'].min())/(df['attn'].max()-df['attn'].min())
# df['attn'] = (df['attn']-df['attn'].mean())/df['attn'].std()
df[['attn', 'motif_bin']].to_csv('motif_attn_cd4_tcm.csv')

df['attn'] = np.log10(df['attn']/df['attn'].min())
df['attn'] = (df['attn']-df['attn'].min())/(df['attn'].max()-df['attn'].min())
df[['attn', 'motif_bin']].to_csv('motif_attn_cd4_tcm_norm.csv')

## plot box
from plotnine import *
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import stats

plt.rcParams['pdf.fonttype'] = 42

## no norm
df = pd.read_csv('motif_attn_cd4_tcm.csv', index_col=0)
df['motif_bin'] = df['motif_bin'].apply(str)
df = df.replace([np.inf, -np.inf], np.nan).dropna()
df = df[df['attn']!=0]
df['attn'] = df['attn']*1000
df = df[df['motif_bin']!='1']  # 'mid' is not obvious

# p = ggplot(df, aes(x='motif_bin', y='attn', fill='motif_bin')) + geom_boxplot(width=0.5, show_legend=False, outlier_shape='') + xlab('') +\
#                                                                  scale_y_continuous(limits=[0.00, 0.10], breaks=np.arange(0.00, 0.10+0.01, 0.02)) + theme_bw()
p = ggplot(df, aes(x='motif_bin', y='attn', fill='motif_bin')) + geom_boxplot(width=0.5, show_legend=False, outlier_shape='') + xlab('') +\
                                                                 coord_cartesian(ylim=(0.17, 0.23)) + theme_bw()
p.save(filename='motif_attn_cd4_tcm.pdf', dpi=600, height=4, width=2)

df[df['motif_bin']=='0']['attn'].median()   # 0.19977421
df[df['motif_bin']=='2']['attn'].median()   # 0.20124071

stats.ttest_ind(df[df['motif_bin']=='0']['attn'], df[df['motif_bin']=='2']['attn'])  # pvalue=8.089084209426905e-06

## with norm
df = pd.read_csv('motif_attn_cd4_tcm_norm.csv', index_col=0)
df['motif_bin'] = df['motif_bin'].apply(str)
df = df.replace([np.inf, -np.inf], np.nan).dropna()
df = df[df['attn']!=0]
df = df[df['motif_bin']!='1']  # 'mid' is not obvious

# p = ggplot(df, aes(x='motif_bin', y='attn', fill='motif_bin')) + geom_boxplot(width=0.5, show_legend=False, outlier_shape='') + xlab('') +\
#                                                                  scale_y_continuous(limits=[0.00, 0.10], breaks=np.arange(0.00, 0.10+0.01, 0.02)) + theme_bw()
p = ggplot(df, aes(x='motif_bin', y='attn', fill='motif_bin')) + geom_boxplot(width=0.5, show_legend=False, outlier_shape='') + xlab('') +\
                                                                 coord_cartesian(ylim=(0.947, 0.952)) +\
                                                                 scale_y_continuous(breaks=np.arange(0.947, 0.952+0.001, 0.001)) + theme_bw()
p.save(filename='motif_attn_cd4_tcm_norm.pdf', dpi=600, height=4, width=2)

df[df['motif_bin']=='0']['attn'].median()   # 0.94974244
df[df['motif_bin']=='2']['attn'].median()   # 0.94985387

stats.ttest_ind(df[df['motif_bin']=='0']['attn'], df[df['motif_bin']=='2']['attn'])  # pvalue=0.0012267649893935536

################ CD4 Naive ################
random.seed(0)
idx = list(np.argwhere(rna.obs['cell_anno']=='CD4 Naive').flatten())
idx_20 = random.sample(list(idx), 20)
rna[idx_20].write('rna_cd4_naive_20.h5ad')
np.count_nonzero(rna[idx_20].X.toarray(), axis=1).max()  # 2170

atac[idx_20].write('atac_cd4_naive_20.h5ad')

# python data_preprocess.py -r rna_cd4_naive_20.h5ad -a atac_cd4_naive_20.h5ad -s pt_cd4_naive_20 --dt test -n cd4_naive_20 --config rna2atac_config_test_cd4_naive_20.yaml
# enc_max_len: 2170

cd4_naive_data = torch.load("./pt_cd4_naive_20/cd4_naive_20_0.pt")
dataset = PreDataset(cd4_naive_data)
dataloader_kwargs = {'batch_size': 1, 'shuffle': False}
loader = torch.utils.data.DataLoader(dataset, **dataloader_kwargs)

# time: 1 min
os.makedirs('./attn/cd4_naive')
i=0
for inputs in tqdm(loader, ncols=80, desc='output attention matrix'):
    rna_sequence, rna_value, atac_sequence, _, enc_pad_mask = [each.to(device) for each in inputs]
    attn = model.generate_attn_weight(rna_sequence, atac_sequence, rna_value, enc_mask=enc_pad_mask, which='decoder')
    attn = attn[0]  # attn = attn[0].to(torch.float16)
    with h5py.File('attn/cd4_naive/attn_cd4_naive_'+str(i)+'.h5', 'w') as f:
        f.create_dataset('attn', data=attn)
    torch.cuda.empty_cache()
    i += 1

rna_cd4_naive_20 = sc.read_h5ad('rna_cd4_naive_20.h5ad')
nonzero = np.count_nonzero(rna_cd4_naive_20.X.toarray(), axis=0)
gene_lst = rna_cd4_naive_20.var.index[nonzero>=3]  # 3516

atac_cd4_naive_20 = sc.read_h5ad('atac_cd4_naive_20.h5ad')

gene_attn = {}
for gene in gene_lst:
    gene_attn[gene] = np.zeros([atac_cd4_naive_20.shape[1], 1], dtype='float32')  # not float16

# 10s/cell
for i in range(20):
    rna_sequence = cd4_naive_data[0][i].flatten()
    with h5py.File('attn/cd4_naive/attn_cd4_naive_'+str(i)+'.h5', 'r') as f:
        attn = f['attn'][:]
        for gene in tqdm(gene_lst, ncols=80, desc='cell '+str(i)):
            idx = torch.argwhere((rna_sequence==(np.argwhere(rna_cd4_naive_20.var.index==gene))[0][0]+1).flatten())
            if len(idx)!=0:
                gene_attn[gene] += attn[:, [idx.item()]]

with open('./attn/cd4_naive/attn_20_cd4_naive_3cell.pkl', 'wb') as file:
    pickle.dump(gene_attn, file)

with open('./attn/cd4_naive/attn_20_cd4_naive_3cell.pkl', 'rb') as file:
    gene_attn = pickle.load(file)

for gene in gene_attn:
    gene_attn[gene] = gene_attn[gene].flatten()

gene_attn_df = pd.DataFrame(gene_attn)   # 236295 x 3516
gene_attn_df.index = atac_cd4_naive_20.var.index.values

gene_attn_df_avg = gene_attn_df.mean(axis=1)/20

motif_info_raw = pd.read_table('peaks_motifs_count.txt', header=0)
motif_info = motif_info_raw.copy()
motif_info[motif_info!=0] = 1
motif_info = motif_info.sum(axis=1)
motif_info.index = gene_attn_df.index.values

df = pd.DataFrame({'attn':gene_attn_df_avg, 'motif':motif_info})
df['motif_bin'] = '0'
df.loc[((df['motif']>0) & (df['motif']<=50)), 'motif_bin'] = '1'
df.loc[(df['motif']>50), 'motif_bin'] = '2'
# df['attn'] = np.log10(df['attn']/df['attn'].min())
# df['attn'] = (df['attn']-df['attn'].min())/(df['attn'].max()-df['attn'].min())
# df['attn'] = (df['attn']-df['attn'].mean())/df['attn'].std()
df[['attn', 'motif_bin']].to_csv('motif_attn_cd4_naive.csv')

df['attn'] = np.log10(df['attn']/df['attn'].min())
df['attn'] = (df['attn']-df['attn'].min())/(df['attn'].max()-df['attn'].min())
df[['attn', 'motif_bin']].to_csv('motif_attn_cd4_naive_norm.csv')

## plot box
from plotnine import *
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import stats

plt.rcParams['pdf.fonttype'] = 42

## no norm
df = pd.read_csv('motif_attn_cd4_naive.csv', index_col=0)
df['motif_bin'] = df['motif_bin'].apply(str)
df = df.replace([np.inf, -np.inf], np.nan).dropna()
df = df[df['attn']!=0]
df['attn'] = df['attn']*1000
df = df[df['motif_bin']!='1']  # 'mid' is not obvious

# p = ggplot(df, aes(x='motif_bin', y='attn', fill='motif_bin')) + geom_boxplot(width=0.5, show_legend=False, outlier_shape='') + xlab('') +\
#                                                                  scale_y_continuous(limits=[0.00, 0.10], breaks=np.arange(0.00, 0.10+0.01, 0.02)) + theme_bw()
p = ggplot(df, aes(x='motif_bin', y='attn', fill='motif_bin')) + geom_boxplot(width=0.5, show_legend=False, outlier_shape='') + xlab('') +\
                                                                 coord_cartesian(ylim=(0.19, 0.26)) + theme_bw()
p.save(filename='motif_attn_cd4_naive.pdf', dpi=600, height=4, width=2)

df[df['motif_bin']=='0']['attn'].median()   # 0.22770822
df[df['motif_bin']=='2']['attn'].median()   # 0.22968281

stats.ttest_ind(df[df['motif_bin']=='0']['attn'], df[df['motif_bin']=='2']['attn'])  # pvalue=1.2792511946603712e-06

## with norm
df = pd.read_csv('motif_attn_cd4_naive_norm.csv', index_col=0)
df['motif_bin'] = df['motif_bin'].apply(str)
df = df.replace([np.inf, -np.inf], np.nan).dropna()
df = df[df['attn']!=0]
df = df[df['motif_bin']!='1']  # 'mid' is not obvious

# p = ggplot(df, aes(x='motif_bin', y='attn', fill='motif_bin')) + geom_boxplot(width=0.5, show_legend=False, outlier_shape='') + xlab('') +\
#                                                                  scale_y_continuous(limits=[0.00, 0.10], breaks=np.arange(0.00, 0.10+0.01, 0.02)) + theme_bw()
p = ggplot(df, aes(x='motif_bin', y='attn', fill='motif_bin')) + geom_boxplot(width=0.5, show_legend=False, outlier_shape='') + xlab('') +\
                                                                 coord_cartesian(ylim=(0.954, 0.959)) +\
                                                                 scale_y_continuous(breaks=np.arange(0.954, 0.959+0.001, 0.001)) + theme_bw()
p.save(filename='motif_attn_cd4_naive_norm.pdf', dpi=600, height=4, width=2)

df[df['motif_bin']=='0']['attn'].median()   # 0.956380225
df[df['motif_bin']=='2']['attn'].median()   # 0.956487975

stats.ttest_ind(df[df['motif_bin']=='0']['attn'], df[df['motif_bin']=='2']['attn'])  # pvalue=0.018569671054108276


################ CD4 intermediate ################
random.seed(0)
idx = list(np.argwhere(rna.obs['cell_anno']=='CD4 intermediate').flatten())
idx_20 = random.sample(list(idx), 20)
rna[idx_20].write('rna_cd4_inter_20.h5ad')
np.count_nonzero(rna[idx_20].X.toarray(), axis=1).max()  # 2082

atac[idx_20].write('atac_cd4_inter_20.h5ad')

# python data_preprocess.py -r rna_cd4_inter_20.h5ad -a atac_cd4_inter_20.h5ad -s pt_cd4_inter_20 --dt test -n cd4_inter_20 --config rna2atac_config_test_cd4_inter_20.yaml
# enc_max_len: 2082

cd4_inter_data = torch.load("./pt_cd4_inter_20/cd4_inter_20_0.pt")
dataset = PreDataset(cd4_inter_data)
dataloader_kwargs = {'batch_size': 1, 'shuffle': False}
loader = torch.utils.data.DataLoader(dataset, **dataloader_kwargs)

# time: 1 min
os.makedirs('./attn/cd4_inter')
i=0
for inputs in tqdm(loader, ncols=80, desc='output attention matrix'):
    rna_sequence, rna_value, atac_sequence, _, enc_pad_mask = [each.to(device) for each in inputs]
    attn = model.generate_attn_weight(rna_sequence, atac_sequence, rna_value, enc_mask=enc_pad_mask, which='decoder')
    attn = attn[0]  # attn = attn[0].to(torch.float16)
    with h5py.File('attn/cd4_inter/attn_cd4_inter_'+str(i)+'.h5', 'w') as f:
        f.create_dataset('attn', data=attn)
    torch.cuda.empty_cache()
    i += 1

rna_cd4_inter_20 = sc.read_h5ad('rna_cd4_inter_20.h5ad')
nonzero = np.count_nonzero(rna_cd4_inter_20.X.toarray(), axis=0)
gene_lst = rna_cd4_inter_20.var.index[nonzero>=3]  # 3893

atac_cd4_inter_20 = sc.read_h5ad('atac_cd4_inter_20.h5ad')

gene_attn = {}
for gene in gene_lst:
    gene_attn[gene] = np.zeros([atac_cd4_inter_20.shape[1], 1], dtype='float32')  # not float16

# 10s/cell
for i in range(20):
    rna_sequence = cd4_inter_data[0][i].flatten()
    with h5py.File('attn/cd4_inter/attn_cd4_inter_'+str(i)+'.h5', 'r') as f:
        attn = f['attn'][:]
        for gene in tqdm(gene_lst, ncols=80, desc='cell '+str(i)):
            idx = torch.argwhere((rna_sequence==(np.argwhere(rna_cd4_inter_20.var.index==gene))[0][0]+1).flatten())
            if len(idx)!=0:
                gene_attn[gene] += attn[:, [idx.item()]]

with open('./attn/cd4_inter/attn_20_cd4_inter_3cell.pkl', 'wb') as file:
    pickle.dump(gene_attn, file)

with open('./attn/cd4_inter/attn_20_cd4_inter_3cell.pkl', 'rb') as file:
    gene_attn = pickle.load(file)

for gene in gene_attn:
    gene_attn[gene] = gene_attn[gene].flatten()

gene_attn_df = pd.DataFrame(gene_attn)   # 236295 x 3893
gene_attn_df.index = atac_cd4_inter_20.var.index.values

gene_attn_df_avg = gene_attn_df.mean(axis=1)/20

motif_info_raw = pd.read_table('peaks_motifs_count.txt', header=0)
motif_info = motif_info_raw.copy()
motif_info[motif_info!=0] = 1
motif_info = motif_info.sum(axis=1)
motif_info.index = gene_attn_df.index.values

df = pd.DataFrame({'attn':gene_attn_df_avg, 'motif':motif_info})
df['motif_bin'] = '0'
df.loc[((df['motif']>0) & (df['motif']<=50)), 'motif_bin'] = '1'
df.loc[(df['motif']>50), 'motif_bin'] = '2'
# df['attn'] = np.log10(df['attn']/df['attn'].min())
# df['attn'] = (df['attn']-df['attn'].min())/(df['attn'].max()-df['attn'].min())
# df['attn'] = (df['attn']-df['attn'].mean())/df['attn'].std()
df[['attn', 'motif_bin']].to_csv('motif_attn_cd4_inter.csv')

df['attn'] = np.log10(df['attn']/df['attn'].min())
df['attn'] = (df['attn']-df['attn'].min())/(df['attn'].max()-df['attn'].min())
df[['attn', 'motif_bin']].to_csv('motif_attn_cd4_inter_norm.csv')

## plot box
from plotnine import *
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import stats

plt.rcParams['pdf.fonttype'] = 42

## no norm
df = pd.read_csv('motif_attn_cd4_inter.csv', index_col=0)
df['motif_bin'] = df['motif_bin'].apply(str)
df = df.replace([np.inf, -np.inf], np.nan).dropna()
df = df[df['attn']!=0]
df['attn'] = df['attn']*1000
df = df[df['motif_bin']!='1']  # 'mid' is not obvious

# p = ggplot(df, aes(x='motif_bin', y='attn', fill='motif_bin')) + geom_boxplot(width=0.5, show_legend=False, outlier_shape='') + xlab('') +\
#                                                                  scale_y_continuous(limits=[0.00, 0.10], breaks=np.arange(0.00, 0.10+0.01, 0.02)) + theme_bw()
p = ggplot(df, aes(x='motif_bin', y='attn', fill='motif_bin')) + geom_boxplot(width=0.5, show_legend=False, outlier_shape='') + xlab('') +\
                                                                 coord_cartesian(ylim=(0.18, 0.25)) + theme_bw()
p.save(filename='motif_attn_cd4_inter.pdf', dpi=600, height=4, width=2)

df[df['motif_bin']=='0']['attn'].median()   # 0.2109143
df[df['motif_bin']=='2']['attn'].median()   # 0.21416911

stats.ttest_ind(df[df['motif_bin']=='0']['attn'], df[df['motif_bin']=='2']['attn'])  # pvalue=4.5403433443482146e-10

## with norm
df = pd.read_csv('motif_attn_cd4_inter_norm.csv', index_col=0)
df['motif_bin'] = df['motif_bin'].apply(str)
df = df.replace([np.inf, -np.inf], np.nan).dropna()
df = df[df['attn']!=0]
df = df[df['motif_bin']!='1']  # 'mid' is not obvious

p = ggplot(df, aes(x='motif_bin', y='attn', fill='motif_bin')) + geom_boxplot(width=0.5, show_legend=False, outlier_shape='') + xlab('') +\
                                                                 coord_cartesian(ylim=(0.930, 0.935)) +\
                                                                 scale_y_continuous(breaks=np.arange(0.930, 0.935+0.001, 0.001)) + theme_bw()
p.save(filename='motif_attn_cd4_inter_norm.pdf', dpi=600, height=4, width=2)

df[df['motif_bin']=='0']['attn'].median()   # 0.9323842
df[df['motif_bin']=='2']['attn'].median()   # 0.93260145

stats.ttest_ind(df[df['motif_bin']=='0']['attn'], df[df['motif_bin']=='2']['attn'])  # pvalue=0.0003650971880714685


################ CD8 TEM ################
random.seed(0)
idx = list(np.argwhere(rna.obs['cell_anno']=='CD8 TEM').flatten())
idx_20 = random.sample(list(idx), 20)
rna[idx_20].write('rna_cd8_tem_20.h5ad')
np.count_nonzero(rna[idx_20].X.toarray(), axis=1).max()  # 2183

atac[idx_20].write('atac_cd8_tem_20.h5ad')

# python data_preprocess.py -r rna_cd8_tem_20.h5ad -a atac_cd8_tem_20.h5ad -s pt_cd8_tem_20 --dt test -n cd8_tem_20 --config rna2atac_config_test_cd8_tem_20.yaml
# enc_max_len: 2183

cd8_tem_data = torch.load("./pt_cd8_tem_20/cd8_tem_20_0.pt")
dataset = PreDataset(cd8_tem_data)
dataloader_kwargs = {'batch_size': 1, 'shuffle': False}
loader = torch.utils.data.DataLoader(dataset, **dataloader_kwargs)

# time: 1 min
os.makedirs('./attn/cd8_tem')
i=0
for inputs in tqdm(loader, ncols=80, desc='output attention matrix'):
    rna_sequence, rna_value, atac_sequence, _, enc_pad_mask = [each.to(device) for each in inputs]
    attn = model.generate_attn_weight(rna_sequence, atac_sequence, rna_value, enc_mask=enc_pad_mask, which='decoder')
    attn = attn[0]  # attn = attn[0].to(torch.float16)
    with h5py.File('attn/cd8_tem/attn_cd8_tem_'+str(i)+'.h5', 'w') as f:
        f.create_dataset('attn', data=attn)
    torch.cuda.empty_cache()
    i += 1

rna_cd8_tem_20 = sc.read_h5ad('rna_cd8_tem_20.h5ad')
nonzero = np.count_nonzero(rna_cd8_tem_20.X.toarray(), axis=0)
gene_lst = rna_cd8_tem_20.var.index[nonzero>=3]  # 4287

atac_cd8_tem_20 = sc.read_h5ad('atac_cd8_tem_20.h5ad')

gene_attn = {}
for gene in gene_lst:
    gene_attn[gene] = np.zeros([atac_cd8_tem_20.shape[1], 1], dtype='float32')  # not float16

# 10s/cell
for i in range(20):
    rna_sequence = cd8_tem_data[0][i].flatten()
    with h5py.File('attn/cd8_tem/attn_cd8_tem_'+str(i)+'.h5', 'r') as f:
        attn = f['attn'][:]
        for gene in tqdm(gene_lst, ncols=80, desc='cell '+str(i)):
            idx = torch.argwhere((rna_sequence==(np.argwhere(rna_cd8_tem_20.var.index==gene))[0][0]+1).flatten())
            if len(idx)!=0:
                gene_attn[gene] += attn[:, [idx.item()]]

with open('./attn/cd8_tem/attn_20_cd8_tem_3cell.pkl', 'wb') as file:
    pickle.dump(gene_attn, file)


################ CD4 TEM ################
random.seed(0)
idx = list(np.argwhere(rna.obs['cell_anno']=='CD4 TEM').flatten())
idx_20 = random.sample(list(idx), 20)
rna[idx_20].write('rna_cd4_tem_20.h5ad')
np.count_nonzero(rna[idx_20].X.toarray(), axis=1).max()  # 1933

atac[idx_20].write('atac_cd4_tem_20.h5ad')

# python data_preprocess.py -r rna_cd4_tem_20.h5ad -a atac_cd4_tem_20.h5ad -s pt_cd4_tem_20 --dt test -n cd4_tem_20 --config rna2atac_config_test_cd4_tem_20.yaml
# enc_max_len: 1933

cd4_tem_data = torch.load("./pt_cd4_tem_20/cd4_tem_20_0.pt")
dataset = PreDataset(cd4_tem_data)
dataloader_kwargs = {'batch_size': 1, 'shuffle': False}
loader = torch.utils.data.DataLoader(dataset, **dataloader_kwargs)

# time: 1 min
os.makedirs('./attn/cd4_tem')
i=0
for inputs in tqdm(loader, ncols=80, desc='output attention matrix'):
    rna_sequence, rna_value, atac_sequence, _, enc_pad_mask = [each.to(device) for each in inputs]
    attn = model.generate_attn_weight(rna_sequence, atac_sequence, rna_value, enc_mask=enc_pad_mask, which='decoder')
    attn = attn[0]  # attn = attn[0].to(torch.float16)
    with h5py.File('attn/cd4_tem/attn_cd4_tem_'+str(i)+'.h5', 'w') as f:
        f.create_dataset('attn', data=attn)
    torch.cuda.empty_cache()
    i += 1

rna_cd4_tem_20 = sc.read_h5ad('rna_cd4_tem_20.h5ad')
nonzero = np.count_nonzero(rna_cd4_tem_20.X.toarray(), axis=0)
gene_lst = rna_cd4_tem_20.var.index[nonzero>=3]  # 4287

atac_cd4_tem_20 = sc.read_h5ad('atac_cd4_tem_20.h5ad')

gene_attn = {}
for gene in gene_lst:
    gene_attn[gene] = np.zeros([atac_cd4_tem_20.shape[1], 1], dtype='float32')  # not float16

# 10s/cell
for i in range(20):
    rna_sequence = cd4_tem_data[0][i].flatten()
    with h5py.File('attn/cd4_tem/attn_cd4_tem_'+str(i)+'.h5', 'r') as f:
        attn = f['attn'][:]
        for gene in tqdm(gene_lst, ncols=80, desc='cell '+str(i)):
            idx = torch.argwhere((rna_sequence==(np.argwhere(rna_cd4_tem_20.var.index==gene))[0][0]+1).flatten())
            if len(idx)!=0:
                gene_attn[gene] += attn[:, [idx.item()]]

with open('./attn/cd4_tem/attn_20_cd4_tem_3cell.pkl', 'wb') as file:
    pickle.dump(gene_attn, file)

################ B memory ################
random.seed(0)
idx = list(np.argwhere(rna.obs['cell_anno']=='B memory').flatten())
idx_20 = random.sample(list(idx), 20)
rna[idx_20].write('rna_b_mem_20.h5ad')
np.count_nonzero(rna[idx_20].X.toarray(), axis=1).max()  # 3007

atac[idx_20].write('atac_b_mem_20.h5ad')

# python data_preprocess.py -r rna_b_mem_20.h5ad -a atac_b_mem_20.h5ad -s pt_b_mem_20 --dt test -n b_mem_20 --config rna2atac_config_test_b_mem_20.yaml
# enc_max_len: 3007

b_mem_data = torch.load("./pt_b_mem_20/b_mem_20_0.pt")
dataset = PreDataset(b_mem_data)
dataloader_kwargs = {'batch_size': 1, 'shuffle': False}
loader = torch.utils.data.DataLoader(dataset, **dataloader_kwargs)

# time: 1 min
os.makedirs('./attn/b_mem')
i=0
for inputs in tqdm(loader, ncols=80, desc='output attention matrix'):
    rna_sequence, rna_value, atac_sequence, _, enc_pad_mask = [each.to(device) for each in inputs]
    attn = model.generate_attn_weight(rna_sequence, atac_sequence, rna_value, enc_mask=enc_pad_mask, which='decoder')
    attn = attn[0]  # attn = attn[0].to(torch.float16)
    with h5py.File('attn/b_mem/attn_b_mem_'+str(i)+'.h5', 'w') as f:
        f.create_dataset('attn', data=attn)
    torch.cuda.empty_cache()
    i += 1

rna_b_mem_20 = sc.read_h5ad('rna_b_mem_20.h5ad')
nonzero = np.count_nonzero(rna_b_mem_20.X.toarray(), axis=0)
gene_lst = rna_b_mem_20.var.index[nonzero>=3]  # 4078

atac_b_mem_20 = sc.read_h5ad('atac_b_mem_20.h5ad')

gene_attn = {}
for gene in gene_lst:
    gene_attn[gene] = np.zeros([atac_b_mem_20.shape[1], 1], dtype='float32')  # not float16

# 10s/cell
for i in range(20):
    rna_sequence = b_mem_data[0][i].flatten()
    with h5py.File('attn/b_mem/attn_b_mem_'+str(i)+'.h5', 'r') as f:
        attn = f['attn'][:]
        for gene in tqdm(gene_lst, ncols=80, desc='cell '+str(i)):
            idx = torch.argwhere((rna_sequence==(np.argwhere(rna_b_mem_20.var.index==gene))[0][0]+1).flatten())
            if len(idx)!=0:
                gene_attn[gene] += attn[:, [idx.item()]]

with open('./attn/b_mem/attn_20_b_mem_3cell.pkl', 'wb') as file:
    pickle.dump(gene_attn, file)


################ NK ###############
random.seed(0)
idx = list(np.argwhere(rna.obs['cell_anno']=='NK').flatten())
idx_20 = random.sample(list(idx), 20)
rna[idx_20].write('rna_nk_20.h5ad')
np.count_nonzero(rna[idx_20].X.toarray(), axis=1).max()  # 2250

atac[idx_20].write('atac_nk_20.h5ad')

# python data_preprocess.py -r rna_nk_20.h5ad -a atac_nk_20.h5ad -s pt_nk_20 --dt test -n nk_20 --config rna2atac_config_test_nk_20.yaml
# enc_max_len: 2250

nk_data = torch.load("./pt_nk_20/nk_20_0.pt")
dataset = PreDataset(nk_data)
dataloader_kwargs = {'batch_size': 1, 'shuffle': False}
loader = torch.utils.data.DataLoader(dataset, **dataloader_kwargs)

# time: 1 min
os.makedirs('./attn/nk')
i=0
for inputs in tqdm(loader, ncols=80, desc='output attention matrix'):
    rna_sequence, rna_value, atac_sequence, _, enc_pad_mask = [each.to(device) for each in inputs]
    attn = model.generate_attn_weight(rna_sequence, atac_sequence, rna_value, enc_mask=enc_pad_mask, which='decoder')
    attn = attn[0]  # attn = attn[0].to(torch.float16)
    with h5py.File('attn/nk/attn_nk_'+str(i)+'.h5', 'w') as f:
        f.create_dataset('attn', data=attn)
    torch.cuda.empty_cache()
    i += 1

rna_nk_20 = sc.read_h5ad('rna_nk_20.h5ad')
nonzero = np.count_nonzero(rna_nk_20.X.toarray(), axis=0)
gene_lst = rna_nk_20.var.index[nonzero>=3]  # 4078

atac_nk_20 = sc.read_h5ad('atac_nk_20.h5ad')

gene_attn = {}
for gene in gene_lst:
    gene_attn[gene] = np.zeros([atac_nk_20.shape[1], 1], dtype='float32')  # not float16

# 10s/cell
for i in range(20):
    rna_sequence = nk_data[0][i].flatten()
    with h5py.File('attn/nk/attn_nk_'+str(i)+'.h5', 'r') as f:
        attn = f['attn'][:]
        for gene in tqdm(gene_lst, ncols=80, desc='cell '+str(i)):
            idx = torch.argwhere((rna_sequence==(np.argwhere(rna_nk_20.var.index==gene))[0][0]+1).flatten())
            if len(idx)!=0:
                gene_attn[gene] += attn[:, [idx.item()]]

with open('./attn/nk/attn_20_nk_3cell.pkl', 'wb') as file:
    pickle.dump(gene_attn, file)


################ CD16 Mono ###############
random.seed(0)
idx = list(np.argwhere(rna.obs['cell_anno']=='CD16 Mono').flatten())
idx_20 = random.sample(list(idx), 20)
rna[idx_20].write('rna_cd16_mono_20.h5ad')
np.count_nonzero(rna[idx_20].X.toarray(), axis=1).max()  # 3035

atac[idx_20].write('atac_cd16_mono_20.h5ad')

# python data_preprocess.py -r rna_cd16_mono_20.h5ad -a atac_cd16_mono_20.h5ad -s pt_cd16_mono_20 --dt test -n cd16_mono_20 --config rna2atac_config_test_cd16_mono_20.yaml
# enc_max_len: 3035

cd16_mono_data = torch.load("./pt_cd16_mono_20/cd16_mono_20_0.pt")
dataset = PreDataset(cd16_mono_data)
dataloader_kwargs = {'batch_size': 1, 'shuffle': False}
loader = torch.utils.data.DataLoader(dataset, **dataloader_kwargs)

# time: 1 min
os.makedirs('./attn/cd16_mono')
i=0
for inputs in tqdm(loader, ncols=80, desc='output attention matrix'):
    rna_sequence, rna_value, atac_sequence, _, enc_pad_mask = [each.to(device) for each in inputs]
    attn = model.generate_attn_weight(rna_sequence, atac_sequence, rna_value, enc_mask=enc_pad_mask, which='decoder')
    attn = attn[0]  # attn = attn[0].to(torch.float16)
    with h5py.File('attn/cd16_mono/attn_cd16_mono_'+str(i)+'.h5', 'w') as f:
        f.create_dataset('attn', data=attn)
    torch.cuda.empty_cache()
    i += 1

rna_cd16_mono_20 = sc.read_h5ad('rna_cd16_mono_20.h5ad')
nonzero = np.count_nonzero(rna_cd16_mono_20.X.toarray(), axis=0)
gene_lst = rna_cd16_mono_20.var.index[nonzero>=3]  # 4078

atac_cd16_mono_20 = sc.read_h5ad('atac_cd16_mono_20.h5ad')

gene_attn = {}
for gene in gene_lst:
    gene_attn[gene] = np.zeros([atac_cd16_mono_20.shape[1], 1], dtype='float32')  # not float16

# 10s/cell
for i in range(20):
    rna_sequence = cd16_mono_data[0][i].flatten()
    with h5py.File('attn/cd16_mono/attn_cd16_mono_'+str(i)+'.h5', 'r') as f:
        attn = f['attn'][:]
        for gene in tqdm(gene_lst, ncols=80, desc='cell '+str(i)):
            idx = torch.argwhere((rna_sequence==(np.argwhere(rna_cd16_mono_20.var.index==gene))[0][0]+1).flatten())
            if len(idx)!=0:
                gene_attn[gene] += attn[:, [idx.item()]]

with open('./attn/cd16_mono/attn_20_cd16_mono_3cell.pkl', 'wb') as file:
    pickle.dump(gene_attn, file)


################ B naive ###############
random.seed(0)
idx = list(np.argwhere(rna.obs['cell_anno']=='B naive').flatten())
idx_20 = random.sample(list(idx), 20)
rna[idx_20].write('rna_b_naive_20.h5ad')
np.count_nonzero(rna[idx_20].X.toarray(), axis=1).max()  # 1661

atac[idx_20].write('atac_b_naive_20.h5ad')

# python data_preprocess.py -r rna_b_naive_20.h5ad -a atac_b_naive_20.h5ad -s pt_b_naive_20 --dt test -n b_naive_20 --config rna2atac_config_test_b_naive_20.yaml
# enc_max_len: 1661

b_naive_data = torch.load("./pt_b_naive_20/b_naive_20_0.pt")
dataset = PreDataset(b_naive_data)
dataloader_kwargs = {'batch_size': 1, 'shuffle': False}
loader = torch.utils.data.DataLoader(dataset, **dataloader_kwargs)

# time: 1 min
os.makedirs('./attn/b_naive')
i=0
for inputs in tqdm(loader, ncols=80, desc='output attention matrix'):
    rna_sequence, rna_value, atac_sequence, _, enc_pad_mask = [each.to(device) for each in inputs]
    attn = model.generate_attn_weight(rna_sequence, atac_sequence, rna_value, enc_mask=enc_pad_mask, which='decoder')
    attn = attn[0]  # attn = attn[0].to(torch.float16)
    with h5py.File('attn/b_naive/attn_b_naive_'+str(i)+'.h5', 'w') as f:
        f.create_dataset('attn', data=attn)
    torch.cuda.empty_cache()
    i += 1

rna_b_naive_20 = sc.read_h5ad('rna_b_naive_20.h5ad')
nonzero = np.count_nonzero(rna_b_naive_20.X.toarray(), axis=0)
gene_lst = rna_b_naive_20.var.index[nonzero>=3]  # 4078

atac_b_naive_20 = sc.read_h5ad('atac_b_naive_20.h5ad')

gene_attn = {}
for gene in gene_lst:
    gene_attn[gene] = np.zeros([atac_b_naive_20.shape[1], 1], dtype='float32')  # not float16

# 10s/cell
for i in range(20):
    rna_sequence = b_naive_data[0][i].flatten()
    with h5py.File('attn/b_naive/attn_b_naive_'+str(i)+'.h5', 'r') as f:
        attn = f['attn'][:]
        for gene in tqdm(gene_lst, ncols=80, desc='cell '+str(i)):
            idx = torch.argwhere((rna_sequence==(np.argwhere(rna_b_naive_20.var.index==gene))[0][0]+1).flatten())
            if len(idx)!=0:
                gene_attn[gene] += attn[:, [idx.item()]]

with open('./attn/b_naive/attn_20_b_naive_3cell.pkl', 'wb') as file:
    pickle.dump(gene_attn, file)

################################################################ CD14 Mono CTCF ############################################################################
############ Peak vs. Attn ############
# wget -c https://www.encodeproject.org/files/ENCFF190KNC/@@download/ENCFF190KNC.bed.gz
# gunzip ENCFF190KNC.bed.gz
# mv ENCFF190KNC.bed cd14_mono_ctcf_peaks_raw.bed
# awk '{print $1 "\t" int(($2+$3)/2) "\t" int(($2+$3)/2+1)}' cd14_mono_ctcf_peaks_raw.bed | sort -k1,1 -k2,2n | uniq > cd14_mono_ctcf_peaks.bed  # 37191
# bedtools intersect -a ../peaks.bed -b cd14_mono_ctcf_peaks.bed -wa | sort -k1,1 -k2,2n | uniq | awk '{print $0 "\t" 1}' >> ccre_peaks_raw.bed
# bedtools intersect -a ../peaks.bed -b cd14_mono_ctcf_peaks.bed -wa -v | sort -k1,1 -k2,2n | uniq | awk '{print $0 "\t" 0}' >> ccre_peaks_raw.bed
# sort -k1,1 -k2,2n ccre_peaks_raw.bed > ccre_peaks.bed
# rm ccre_peaks_raw.bed

import pickle
import pandas as pd
import scanpy as sc

with open('./attn/cd14_mono/attn_20_cd14_mono_3cell.pkl', 'rb') as file:
    gene_attn = pickle.load(file)

for gene in gene_attn:
    gene_attn[gene] = gene_attn[gene].flatten()

gene_attn_ctcf = pd.DataFrame(gene_attn['CTCF'])
atac_cd14_mono_20 = sc.read_h5ad('atac_cd14_mono_20.h5ad')
gene_attn_ctcf.index = atac_cd14_mono_20.var.index.values
gene_attn_ctcf.columns = ['attn']
gene_attn_ctcf['attn'] = gene_attn_ctcf['attn']/20

ctcf_peaks = pd.read_table('tf_chipseq/cd14_mono_ctcf/ccre_peaks.bed', header=None)
ctcf_peaks.index = ctcf_peaks[0]+':'+ctcf_peaks[1].apply(str)+'-'+ctcf_peaks[2].apply(str)
ctcf_peaks.columns = ['chrom', 'start', 'end', 'peak_or_not']

df = gene_attn_ctcf.copy()
df['peak_or_not'] = ctcf_peaks.loc[gene_attn_ctcf.index, 'peak_or_not']
df.to_csv('./tf_chipseq/cd14_mono_ctcf/ctcf_attn_cd14_mono.csv')

## plot box
from plotnine import *
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.metrics import roc_curve, auc

plt.rcParams['pdf.fonttype'] = 42

df = pd.read_csv('./tf_chipseq/cd14_mono_ctcf/ctcf_attn_cd14_mono.csv', index_col=0)
df['peak_or_not'] = df['peak_or_not'].apply(str)
df = df.replace([np.inf, -np.inf], np.nan).dropna()
df['attn'] = np.log10(df['attn']/df['attn'].min())
df['attn'] = (df['attn']-df['attn'].min())/(df['attn'].max()-df['attn'].min())

p = ggplot(df, aes(x='peak_or_not', y='attn', fill='peak_or_not')) + geom_boxplot(width=0.5, show_legend=False, outlier_shape='') + xlab('') +\
                                                                     coord_cartesian(ylim=(0.3, 1.0)) +\
                                                                     scale_y_continuous(breaks=np.arange(0.3, 1.0+0.1, 0.1)) + theme_bw()
p.save(filename='ctcf_attn_cd14_mono.pdf', dpi=600, height=4, width=2)

df[df['peak_or_not']=='0']['attn'].median()   # 0.6310981248008007
df[df['peak_or_not']=='1']['attn'].median()   # 0.6500011675859599

stats.ttest_ind(df[df['peak_or_not']=='0']['attn'], df[df['peak_or_not']=='1']['attn'])  # pvalue=1.0818979614331524e-177

# fpr, tpr, _ = roc_curve(df['peak_or_not'].apply(int), df['attn'])
# auc(fpr, tpr)  # 0.5536149022035999

## calculate relative binding ratio (obs/exp)
df_random = df.copy()
df_random['attn'] = df_random['attn'].sample(frac=1, random_state=0).values

true_ratio = []
rand_ratio = []
for cnt in [5000, 10000, 15000, 20000, 25000, 30000]:
    true_ratio.append(round(df[df['attn']>=df['attn'].quantile(1-cnt/df['attn'].shape[0])]['peak_or_not'].value_counts(normalize=True)[1], 4))
    rand_ratio.append(round(df_random[df_random['attn']>=df_random['attn'].quantile(1-cnt/df_random['attn'].shape[0])]['peak_or_not'].value_counts(normalize=True)[1], 4))

# true_ratio  [0.1230, 0.1174, 0.1185, 0.1171, 0.1181, 0.1183]
# rand_ratio  [0.0994, 0.1004, 0.1009, 0.1028, 0.1032, 0.1030]
true_vs_rand = np.round(np.array(true_ratio)/np.array(rand_ratio), 4)
cnt_ratio = pd.DataFrame({'cnt':[5000, 10000, 15000, 20000, 25000, 30000],
                          'ratio':true_vs_rand})

p = ggplot(cnt_ratio, aes(x='cnt', y='ratio')) + geom_line(size=1) + geom_point(size=1) + xlab('') +\
                                                 scale_x_continuous(limits=[5000, 30000], breaks=np.arange(5000, 30000+1, 5000)) +\
                                                 scale_y_continuous(limits=[0.5, 1.5], breaks=np.arange(0.5, 1.5+0.1, 0.25)) + theme_bw()
p.save(filename='ctcf_cd14_mono_cnt_ratio.pdf', dpi=600, height=4, width=4)
##############################################


########## Signal vs. Attn #############
# install deeptools
# conda create -n deeptools python=3.8
# conda activate deeptools
# conda install -c conda-forge -c bioconda deeptools  # not work
# pip install deeptools  # works

# wget -c https://www.encodeproject.org/files/ENCFF496PSJ/@@download/ENCFF496PSJ.bigWig
# mv ENCFF496PSJ.bigWig cd14_mono_ctcf_signal.bigwig

# awk '{if($4==1) print($1 "\t" $2 "\t" $3)}' ccre_peaks.bed > ccre_ctcf.bed
# awk '{if($4==0) print($1 "\t" $2 "\t" $3)}' ccre_peaks.bed > ccre_not_ctcf.bed

import pandas as pd
import numpy as np

df = pd.read_csv('./tf_chipseq/cd14_mono_ctcf/ctcf_attn_cd14_mono.csv', index_col=0)
df['peak_or_not'] = df['peak_or_not'].apply(str)
df = df.replace([np.inf, -np.inf], np.nan).dropna()
df['attn'] = np.log10(df['attn']/df['attn'].min())
df['attn'] = (df['attn']-df['attn'].min())/(df['attn'].max()-df['attn'].min())

df_top = pd.DataFrame(df[df['attn']>=df['attn'].quantile(1-1000/df['attn'].shape[0])].index.values)
df_top['chrom'] = df_top[0].apply(lambda x: x.split(':')[0])
df_top['start'] = df_top[0].apply(lambda x: x.split(':')[1].split('-')[0])
df_top['end'] = df_top[0].apply(lambda x: x.split(':')[1].split('-')[1])
df_top.iloc[:, 1:].to_csv('./tf_chipseq/cd14_mono_ctcf/attn_top.bed', index=False, header=False, sep='\t')

df_btm = pd.DataFrame(df[df['attn']<=df['attn'].quantile(1000/df['attn'].shape[0])].index.values)
df_btm['chrom'] = df_btm[0].apply(lambda x: x.split(':')[0])
df_btm['start'] = df_btm[0].apply(lambda x: x.split(':')[1].split('-')[0])
df_btm['end'] = df_btm[0].apply(lambda x: x.split(':')[1].split('-')[1])
df_btm.iloc[:, 1:].to_csv('./tf_chipseq/cd14_mono_ctcf/attn_btm.bed', index=False, header=False, sep='\t')

computeMatrix reference-point --referencePoint center -p 20 -S cd14_mono_ctcf_signal.bigwig \
                              -R attn_top.bed attn_btm.bed -o cd14_mono_attn_top_btm_signal.gz -a 5000 -b 5000 -bs 100
plotProfile -m cd14_mono_attn_top_btm_signal.gz --yMin 0 --yMax 10 -out cd14_mono_attn_top_btm_signal.pdf
################################################################################################################################################


################################################################ CD14 Mono IRF1 ############################################################################
############ Peak vs. Attn ############
# wget -c https://remap.univ-amu.fr/storage/remap2022/hg38/MACS2/DATASET/GSE100381.IRF1.monocyte_notreatment.bed.gz
# gunzip GSE100381.IRF1.monocyte_notreatment.bed.gz
# mv GSE100381.IRF1.monocyte_notreatment.bed cd14_mono_irf1_peaks_raw.bed
# awk '{print $1 "\t" int(($2+$3)/2) "\t" int(($2+$3)/2+1)}' cd14_mono_irf1_peaks_raw.bed | sort -k1,1 -k2,2n | uniq > cd14_mono_irf1_peaks.bed  # 15741
# bedtools intersect -a ../../peaks.bed -b cd14_mono_irf1_peaks.bed -wa | sort -k1,1 -k2,2n | uniq | awk '{print $0 "\t" 1}' >> ccre_peaks_raw.bed
# bedtools intersect -a ../../peaks.bed -b cd14_mono_irf1_peaks.bed -wa -v | sort -k1,1 -k2,2n | uniq | awk '{print $0 "\t" 0}' >> ccre_peaks_raw.bed
# sort -k1,1 -k2,2n ccre_peaks_raw.bed > ccre_peaks.bed
# rm ccre_peaks_raw.bed

import pickle
import pandas as pd
import scanpy as sc

with open('./attn/cd14_mono/attn_20_cd14_mono_3cell.pkl', 'rb') as file:
    gene_attn = pickle.load(file)

for gene in gene_attn:
    gene_attn[gene] = gene_attn[gene].flatten()

gene_attn_irf1 = pd.DataFrame(gene_attn['IRF1'])
atac_cd14_mono_20 = sc.read_h5ad('atac_cd14_mono_20.h5ad')
gene_attn_irf1.index = atac_cd14_mono_20.var.index.values
gene_attn_irf1.columns = ['attn']
gene_attn_irf1['attn'] = gene_attn_irf1['attn']/20

irf1_peaks = pd.read_table('tf_chipseq/cd14_mono_irf1/ccre_peaks.bed', header=None)
irf1_peaks.index = irf1_peaks[0]+':'+irf1_peaks[1].apply(str)+'-'+irf1_peaks[2].apply(str)
irf1_peaks.columns = ['chrom', 'start', 'end', 'peak_or_not']

df = gene_attn_irf1.copy()
df['peak_or_not'] = irf1_peaks.loc[gene_attn_irf1.index, 'peak_or_not']
df.to_csv('./tf_chipseq/cd14_mono_irf1/irf1_attn_cd14_mono.csv')

## plot box
from plotnine import *
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.metrics import roc_curve, auc

plt.rcParams['pdf.fonttype'] = 42

df = pd.read_csv('./tf_chipseq/cd14_mono_irf1/irf1_attn_cd14_mono.csv', index_col=0)
df['peak_or_not'] = df['peak_or_not'].apply(str)
df = df.replace([np.inf, -np.inf], np.nan).dropna()
df['attn'] = np.log10(df['attn']/df['attn'].min())
df['attn'] = (df['attn']-df['attn'].min())/(df['attn'].max()-df['attn'].min())

p = ggplot(df, aes(x='peak_or_not', y='attn', fill='peak_or_not')) + geom_boxplot(width=0.5, show_legend=False, outlier_shape='') + xlab('') +\
                                                                     coord_cartesian(ylim=(0.2, 0.9)) +\
                                                                     scale_y_continuous(breaks=np.arange(0.2, 0.9+0.1, 0.1)) + theme_bw()
p.save(filename='irf1_attn_cd14_mono.pdf', dpi=600, height=4, width=2)

df[df['peak_or_not']=='0']['attn'].median()   # 0.5438227360560042
df[df['peak_or_not']=='1']['attn'].median()   # 0.5331902388707281

stats.ttest_ind(df[df['peak_or_not']=='0']['attn'], df[df['peak_or_not']=='1']['attn'])  # pvalue=2.003971466762959e-13

# fpr, tpr, _ = roc_curve(df['peak_or_not'].apply(int), df['attn'])
# auc(fpr, tpr)  # 0.4736166987783853

## calculate relative binding ratio (obs/exp)
df_random = df.copy()
df_random['attn'] = df_random['attn'].sample(frac=1, random_state=0).values

true_ratio = []
rand_ratio = []
for cnt in [5000, 10000, 15000, 20000, 25000, 30000]:
    true_ratio.append(round(df[df['attn']>=df['attn'].quantile(1-cnt/df['attn'].shape[0])]['peak_or_not'].value_counts(normalize=True)[1], 4))
    rand_ratio.append(round(df_random[df_random['attn']>=df_random['attn'].quantile(1-cnt/df_random['attn'].shape[0])]['peak_or_not'].value_counts(normalize=True)[1], 4))

# true_ratio  [0.0206, 0.0203, 0.0224, 0.0236, 0.024, 0.0241]
# rand_ratio  [0.025, 0.0252, 0.0269, 0.0272, 0.0268, 0.0269]
true_vs_rand = np.round(np.array(true_ratio)/np.array(rand_ratio), 4)
cnt_ratio = pd.DataFrame({'cnt':[5000, 10000, 15000, 20000, 25000, 30000],
                          'ratio':true_vs_rand})

p = ggplot(cnt_ratio, aes(x='cnt', y='ratio')) + geom_line(size=1) + geom_point(size=1) + xlab('') +\
                                                 scale_x_continuous(limits=[5000, 30000], breaks=np.arange(5000, 30000+1, 5000)) +\
                                                 scale_y_continuous(limits=[0.5, 1.5], breaks=np.arange(0.5, 1.5+0.1, 0.25)) + theme_bw()
p.save(filename='irf1_cd14_mono_cnt_ratio.pdf', dpi=600, height=4, width=4)
################################################################################################################################################


################################################################ CD14 Mono STAT1 ############################################################################
############ Peak vs. Attn ############
# wget -c https://remap.univ-amu.fr/storage/remap2022/hg38/MACS2/DATASET/GSE43036.STAT1.CD14.bed.gz
# gunzip GSE43036.STAT1.CD14.bed.gz
# mv GSE43036.STAT1.CD14.bed cd14_mono_stat1_peaks_raw.bed
# awk '{print $1 "\t" int(($2+$3)/2) "\t" int(($2+$3)/2+1)}' cd14_mono_stat1_peaks_raw.bed | sort -k1,1 -k2,2n | uniq > cd14_mono_stat1_peaks.bed  # 10636
# bedtools intersect -a ../../peaks.bed -b cd14_mono_stat1_peaks.bed -wa | sort -k1,1 -k2,2n | uniq | awk '{print $0 "\t" 1}' >> ccre_peaks_raw.bed
# bedtools intersect -a ../../peaks.bed -b cd14_mono_stat1_peaks.bed -wa -v | sort -k1,1 -k2,2n | uniq | awk '{print $0 "\t" 0}' >> ccre_peaks_raw.bed
# sort -k1,1 -k2,2n ccre_peaks_raw.bed > ccre_peaks.bed
# rm ccre_peaks_raw.bed

import pickle
import pandas as pd
import scanpy as sc

with open('./attn/cd14_mono/attn_20_cd14_mono_3cell.pkl', 'rb') as file:
    gene_attn = pickle.load(file)

for gene in gene_attn:
    gene_attn[gene] = gene_attn[gene].flatten()

gene_attn_stat1 = pd.DataFrame(gene_attn['STAT1'])
atac_cd14_mono_20 = sc.read_h5ad('atac_cd14_mono_20.h5ad')
gene_attn_stat1.index = atac_cd14_mono_20.var.index.values
gene_attn_stat1.columns = ['attn']
gene_attn_stat1['attn'] = gene_attn_stat1['attn']/20

stat1_peaks = pd.read_table('tf_chipseq/cd14_mono_stat1/ccre_peaks.bed', header=None)
stat1_peaks.index = stat1_peaks[0]+':'+stat1_peaks[1].apply(str)+'-'+stat1_peaks[2].apply(str)
stat1_peaks.columns = ['chrom', 'start', 'end', 'peak_or_not']

df = gene_attn_stat1.copy()
df['peak_or_not'] = stat1_peaks.loc[gene_attn_stat1.index, 'peak_or_not']
df.to_csv('./tf_chipseq/cd14_mono_stat1/stat1_attn_cd14_mono.csv')

## plot box
from plotnine import *
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.metrics import roc_curve, auc

plt.rcParams['pdf.fonttype'] = 42

df = pd.read_csv('./tf_chipseq/cd14_mono_stat1/stat1_attn_cd14_mono.csv', index_col=0)
df['peak_or_not'] = df['peak_or_not'].apply(str)
df = df.replace([np.inf, -np.inf], np.nan).dropna()
df['attn'] = np.log10(df['attn']/df['attn'].min())
df['attn'] = (df['attn']-df['attn'].min())/(df['attn'].max()-df['attn'].min())

p = ggplot(df, aes(x='peak_or_not', y='attn', fill='peak_or_not')) + geom_boxplot(width=0.5, show_legend=False, outlier_shape='') + xlab('') +\
                                                                     coord_cartesian(ylim=(0.25, 0.95)) +\
                                                                     scale_y_continuous(breaks=np.arange(0.25, 0.95+0.1, 0.1)) + theme_bw()
p.save(filename='stat1_attn_cd14_mono.pdf', dpi=600, height=4, width=2)

df[df['peak_or_not']=='0']['attn'].median()   # 0.5901030305514414
df[df['peak_or_not']=='1']['attn'].median()   # 0.5985104680202513

stats.ttest_ind(df[df['peak_or_not']=='0']['attn'], df[df['peak_or_not']=='1']['attn'])  # pvalue=1.3305348309480372e-09

# fpr, tpr, _ = roc_curve(df['peak_or_not'].apply(int), df['attn'])
# auc(fpr, tpr)  # 0.5223489090716011

## calculate relative binding ratio (obs/exp)
df_random = df.copy()
df_random['attn'] = df_random['attn'].sample(frac=1, random_state=0).values

true_ratio = []
rand_ratio = []
for cnt in [5000, 10000, 15000, 20000, 25000, 30000]:
    true_ratio.append(round(df[df['attn']>=df['attn'].quantile(1-cnt/df['attn'].shape[0])]['peak_or_not'].value_counts(normalize=True)[1], 4))
    rand_ratio.append(round(df_random[df_random['attn']>=df_random['attn'].quantile(1-cnt/df_random['attn'].shape[0])]['peak_or_not'].value_counts(normalize=True)[1], 4))

# true_ratio  [0.0508, 0.0474, 0.0432, 0.0415, 0.0404, 0.0389]
# rand_ratio  [0.0246, 0.0244, 0.0254, 0.0265, 0.0269, 0.0267]
true_vs_rand = np.round(np.array(true_ratio)/np.array(rand_ratio), 4)
cnt_ratio = pd.DataFrame({'cnt':[5000, 10000, 15000, 20000, 25000, 30000],
                          'ratio':true_vs_rand})

p = ggplot(cnt_ratio, aes(x='cnt', y='ratio')) + geom_line(size=1) + geom_point(size=1) + xlab('') +\
                                                 scale_x_continuous(limits=[5000, 30000], breaks=np.arange(5000, 30000+1, 5000)) +\
                                                 scale_y_continuous(limits=[0.5, 2.5], breaks=np.arange(0.5, 2.5+0.1, 0.5)) + theme_bw()
p.save(filename='stat1_cd14_mono_cnt_ratio.pdf', dpi=600, height=4, width=4)
################################################################################################################################################


################################################################ CD14 Mono CEBPB ############################################################################
############ Peak vs. Attn ############
# wget -c https://remap.univ-amu.fr/storage/remap2022/hg38/MACS2/DATASET/GSE98367.CEBPB.monocyte.bed.gz
# gunzip GSE98367.CEBPB.monocyte.bed.gz
# mv GSE98367.CEBPB.monocyte.bed cd14_mono_cebpb_peaks_raw.bed
# awk '{print $1 "\t" int(($2+$3)/2) "\t" int(($2+$3)/2+1)}' cd14_mono_cebpb_peaks_raw.bed | sort -k1,1 -k2,2n | uniq > cd14_mono_cebpb_peaks.bed  # 26244
# bedtools intersect -a ../../peaks.bed -b cd14_mono_cebpb_peaks.bed -wa | sort -k1,1 -k2,2n | uniq | awk '{print $0 "\t" 1}' >> ccre_peaks_raw.bed
# bedtools intersect -a ../../peaks.bed -b cd14_mono_cebpb_peaks.bed -wa -v | sort -k1,1 -k2,2n | uniq | awk '{print $0 "\t" 0}' >> ccre_peaks_raw.bed
# sort -k1,1 -k2,2n ccre_peaks_raw.bed > ccre_peaks.bed
# rm ccre_peaks_raw.bed

import pickle
import pandas as pd
import scanpy as sc

with open('./attn/cd14_mono/attn_20_cd14_mono_3cell.pkl', 'rb') as file:
    gene_attn = pickle.load(file)

for gene in gene_attn:
    gene_attn[gene] = gene_attn[gene].flatten()

gene_attn_cebpb = pd.DataFrame(gene_attn['CEBPB'])
atac_cd14_mono_20 = sc.read_h5ad('atac_cd14_mono_20.h5ad')
gene_attn_cebpb.index = atac_cd14_mono_20.var.index.values
gene_attn_cebpb.columns = ['attn']
gene_attn_cebpb['attn'] = gene_attn_cebpb['attn']/20

cebpb_peaks = pd.read_table('tf_chipseq/cd14_mono_cebpb/ccre_peaks.bed', header=None)
cebpb_peaks.index = cebpb_peaks[0]+':'+cebpb_peaks[1].apply(str)+'-'+cebpb_peaks[2].apply(str)
cebpb_peaks.columns = ['chrom', 'start', 'end', 'peak_or_not']

df = gene_attn_cebpb.copy()
df['peak_or_not'] = cebpb_peaks.loc[gene_attn_cebpb.index, 'peak_or_not']
df.to_csv('./tf_chipseq/cd14_mono_cebpb/cebpb_attn_cd14_mono.csv')

## plot box
from plotnine import *
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.metrics import roc_curve, auc

plt.rcParams['pdf.fonttype'] = 42

df = pd.read_csv('./tf_chipseq/cd14_mono_cebpb/cebpb_attn_cd14_mono.csv', index_col=0)
df['peak_or_not'] = df['peak_or_not'].apply(str)
df = df.replace([np.inf, -np.inf], np.nan).dropna()
df['attn'] = np.log10(df['attn']/df['attn'].min())
df['attn'] = (df['attn']-df['attn'].min())/(df['attn'].max()-df['attn'].min())

p = ggplot(df, aes(x='peak_or_not', y='attn', fill='peak_or_not')) + geom_boxplot(width=0.5, show_legend=False, outlier_shape='') + xlab('') +\
                                                                     coord_cartesian(ylim=(0.20, 1.00)) +\
                                                                     scale_y_continuous(breaks=np.arange(0.20, 1.00+0.2, 0.2)) + theme_bw()
p.save(filename='cebpb_attn_cd14_mono.pdf', dpi=600, height=4, width=2)

df[df['peak_or_not']=='0']['attn'].median()   # 0.6782954577592024
df[df['peak_or_not']=='1']['attn'].median()   # 0.6744988676522976

stats.ttest_ind(df[df['peak_or_not']=='0']['attn'], df[df['peak_or_not']=='1']['attn'])  # pvalue=8.176733384818817e-06

# fpr, tpr, _ = roc_curve(df['peak_or_not'].apply(int), df['attn'])
# auc(fpr, tpr)  # 0.4902619670262879

## calculate relative binding ratio (obs/exp)
df_random = df.copy()
df_random['attn'] = df_random['attn'].sample(frac=1, random_state=0).values

true_ratio = []
rand_ratio = []
for cnt in [5000, 10000, 15000, 20000, 25000, 30000]:
    true_ratio.append(round(df[df['attn']>=df['attn'].quantile(1-cnt/df['attn'].shape[0])]['peak_or_not'].value_counts(normalize=True)[1], 4))
    rand_ratio.append(round(df_random[df_random['attn']>=df_random['attn'].quantile(1-cnt/df_random['attn'].shape[0])]['peak_or_not'].value_counts(normalize=True)[1], 4))

# true_ratio  [0.0694, 0.0807, 0.0811, 0.08, 0.0778, 0.076]
# rand_ratio  [0.0638, 0.0624, 0.0615, 0.061, 0.0606, 0.0608]
true_vs_rand = np.round(np.array(true_ratio)/np.array(rand_ratio), 4)
cnt_ratio = pd.DataFrame({'cnt':[5000, 10000, 15000, 20000, 25000, 30000],
                          'ratio':true_vs_rand})

p = ggplot(cnt_ratio, aes(x='cnt', y='ratio')) + geom_line(size=1) + geom_point(size=1) + xlab('') +\
                                                 scale_x_continuous(limits=[5000, 30000], breaks=np.arange(5000, 30000+1, 5000)) +\
                                                 scale_y_continuous(limits=[0.5, 1.5], breaks=np.arange(0.5, 1.5+0.1, 0.25)) + theme_bw()
p.save(filename='cebpb_cd14_mono_cnt_ratio.pdf', dpi=600, height=4, width=4)
################################################################################################################################################


################################################################ CD14 Mono SPI1 ############################################################################
############ Peak vs. Attn ############
# wget -c https://remap.univ-amu.fr/storage/remap2022/hg38/MACS2/DATASET/GSE31621.SPI1.monocyte.bed.gz
# gunzip GSE31621.SPI1.monocyte.bed.gz
# mv GSE31621.SPI1.monocyte.bed cd14_mono_spi1_peaks_raw.bed
# awk '{print $1 "\t" int(($2+$3)/2) "\t" int(($2+$3)/2+1)}' cd14_mono_spi1_peaks_raw.bed | sort -k1,1 -k2,2n | uniq > cd14_mono_spi1_peaks.bed  # 53628
# bedtools intersect -a ../../peaks.bed -b cd14_mono_spi1_peaks.bed -wa | sort -k1,1 -k2,2n | uniq | awk '{print $0 "\t" 1}' >> ccre_peaks_raw.bed
# bedtools intersect -a ../../peaks.bed -b cd14_mono_spi1_peaks.bed -wa -v | sort -k1,1 -k2,2n | uniq | awk '{print $0 "\t" 0}' >> ccre_peaks_raw.bed
# sort -k1,1 -k2,2n ccre_peaks_raw.bed > ccre_peaks.bed
# rm ccre_peaks_raw.bed

import pickle
import pandas as pd
import scanpy as sc

with open('./attn/cd14_mono/attn_20_cd14_mono_3cell.pkl', 'rb') as file:
    gene_attn = pickle.load(file)

for gene in gene_attn:
    gene_attn[gene] = gene_attn[gene].flatten()

gene_attn_spi1 = pd.DataFrame(gene_attn['SPI1'])
atac_cd14_mono_20 = sc.read_h5ad('atac_cd14_mono_20.h5ad')
gene_attn_spi1.index = atac_cd14_mono_20.var.index.values
gene_attn_spi1.columns = ['attn']
gene_attn_spi1['attn'] = gene_attn_spi1['attn']/20

spi1_peaks = pd.read_table('tf_chipseq/cd14_mono_spi1/ccre_peaks.bed', header=None)
spi1_peaks.index = spi1_peaks[0]+':'+spi1_peaks[1].apply(str)+'-'+spi1_peaks[2].apply(str)
spi1_peaks.columns = ['chrom', 'start', 'end', 'peak_or_not']

df = gene_attn_spi1.copy()
df['peak_or_not'] = spi1_peaks.loc[gene_attn_spi1.index, 'peak_or_not']
df.to_csv('./tf_chipseq/cd14_mono_spi1/spi1_attn_cd14_mono.csv')

## plot box
from plotnine import *
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.metrics import roc_curve, auc

plt.rcParams['pdf.fonttype'] = 42

df = pd.read_csv('./tf_chipseq/cd14_mono_spi1/spi1_attn_cd14_mono.csv', index_col=0)
df['peak_or_not'] = df['peak_or_not'].apply(str)
df = df.replace([np.inf, -np.inf], np.nan).dropna()
df['attn'] = np.log10(df['attn']/df['attn'].min())
df['attn'] = (df['attn']-df['attn'].min())/(df['attn'].max()-df['attn'].min())

p = ggplot(df, aes(x='peak_or_not', y='attn', fill='peak_or_not')) + geom_boxplot(width=0.5, show_legend=False, outlier_shape='') + xlab('') +\
                                                                     coord_cartesian(ylim=(0.30, 1.00)) +\
                                                                     scale_y_continuous(breaks=np.arange(0.30, 1.00+0.1, 0.1)) + theme_bw()
p.save(filename='spi1_attn_cd14_mono.pdf', dpi=600, height=4, width=2)

df[df['peak_or_not']=='0']['attn'].median()   # 0.6913961599352173
df[df['peak_or_not']=='1']['attn'].median()   # 0.6687961538524598

stats.ttest_ind(df[df['peak_or_not']=='0']['attn'], df[df['peak_or_not']=='1']['attn'])  # pvalue=0.0

# fpr, tpr, _ = roc_curve(df['peak_or_not'].apply(int), df['attn'])
# auc(fpr, tpr)  # 0.4344259561190602

## calculate relative binding ratio (obs/exp)
df_random = df.copy()
df_random['attn'] = df_random['attn'].sample(frac=1, random_state=0).values

true_ratio = []
rand_ratio = []
for cnt in [5000, 10000, 15000, 20000, 25000, 30000]:
    true_ratio.append(round(df[df['attn']>=df['attn'].quantile(1-cnt/df['attn'].shape[0])]['peak_or_not'].value_counts(normalize=True)[1], 4))
    rand_ratio.append(round(df_random[df_random['attn']>=df_random['attn'].quantile(1-cnt/df_random['attn'].shape[0])]['peak_or_not'].value_counts(normalize=True)[1], 4))

# true_ratio  [0.0908, 0.0926, 0.0936, 0.0953, 0.0954, 0.0974]
# rand_ratio  [0.1284, 0.1321, 0.132, 0.1334, 0.1338, 0.1352]
true_vs_rand = np.round(np.array(true_ratio)/np.array(rand_ratio), 4)
cnt_ratio = pd.DataFrame({'cnt':[5000, 10000, 15000, 20000, 25000, 30000],
                          'ratio':true_vs_rand})

p = ggplot(cnt_ratio, aes(x='cnt', y='ratio')) + geom_line(size=1) + geom_point(size=1) + xlab('') +\
                                                 scale_x_continuous(limits=[5000, 30000], breaks=np.arange(5000, 30000+1, 5000)) +\
                                                 scale_y_continuous(limits=[0.5, 1.5], breaks=np.arange(0.5, 1.5+0.1, 0.25)) + theme_bw()
p.save(filename='spi1_cd14_mono_cnt_ratio.pdf', dpi=600, height=4, width=4)
################################################################################################################################################


################################################################ CD14 Mono STAT3 ############################################################################
############ Peak vs. Attn ############
# wget -c https://remap.univ-amu.fr/storage/remap2022/hg38/MACS2/DATASET/GSE120943.STAT3.monocyte_resting.bed.gz
# gunzip GSE120943.STAT3.monocyte_resting.bed.gz
# mv GSE120943.STAT3.monocyte_resting.bed cd14_mono_stat3_peaks_raw.bed
# awk '{print $1 "\t" int(($2+$3)/2) "\t" int(($2+$3)/2+1)}' cd14_mono_stat3_peaks_raw.bed | sort -k1,1 -k2,2n | uniq > cd14_mono_stat3_peaks.bed  # 20232
# bedtools intersect -a ../../peaks.bed -b cd14_mono_stat3_peaks.bed -wa | sort -k1,1 -k2,2n | uniq | awk '{print $0 "\t" 1}' >> ccre_peaks_raw.bed
# bedtools intersect -a ../../peaks.bed -b cd14_mono_stat3_peaks.bed -wa -v | sort -k1,1 -k2,2n | uniq | awk '{print $0 "\t" 0}' >> ccre_peaks_raw.bed
# sort -k1,1 -k2,2n ccre_peaks_raw.bed > ccre_peaks.bed
# rm ccre_peaks_raw.bed

import pickle
import pandas as pd
import scanpy as sc

with open('./attn/cd14_mono/attn_20_cd14_mono_3cell.pkl', 'rb') as file:
    gene_attn = pickle.load(file)

for gene in gene_attn:
    gene_attn[gene] = gene_attn[gene].flatten()

gene_attn_stat3 = pd.DataFrame(gene_attn['STAT3'])
atac_cd14_mono_20 = sc.read_h5ad('atac_cd14_mono_20.h5ad')
gene_attn_stat3.index = atac_cd14_mono_20.var.index.values
gene_attn_stat3.columns = ['attn']
gene_attn_stat3['attn'] = gene_attn_stat3['attn']/20

stat3_peaks = pd.read_table('tf_chipseq/cd14_mono_stat3/ccre_peaks.bed', header=None)
stat3_peaks.index = stat3_peaks[0]+':'+stat3_peaks[1].apply(str)+'-'+stat3_peaks[2].apply(str)
stat3_peaks.columns = ['chrom', 'start', 'end', 'peak_or_not']

df = gene_attn_stat3.copy()
df['peak_or_not'] = stat3_peaks.loc[gene_attn_stat3.index, 'peak_or_not']
df.to_csv('./tf_chipseq/cd14_mono_stat3/stat3_attn_cd14_mono.csv')


## plot box
from plotnine import *
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.metrics import roc_curve, auc

plt.rcParams['pdf.fonttype'] = 42

df = pd.read_csv('./tf_chipseq/cd14_mono_stat3/stat3_attn_cd14_mono.csv', index_col=0)
df['peak_or_not'] = df['peak_or_not'].apply(str)
df = df.replace([np.inf, -np.inf], np.nan).dropna()
df['attn'] = np.log10(df['attn']/df['attn'].min())
df['attn'] = (df['attn']-df['attn'].min())/(df['attn'].max()-df['attn'].min())

p = ggplot(df, aes(x='peak_or_not', y='attn', fill='peak_or_not')) + geom_boxplot(width=0.5, show_legend=False, outlier_shape='') + xlab('') +\
                                                                     coord_cartesian(ylim=(0.20, 1.00)) +\
                                                                     scale_y_continuous(breaks=np.arange(0.20, 1.00+0.1, 0.2)) + theme_bw()
p.save(filename='stat3_attn_cd14_mono.pdf', dpi=600, height=4, width=2)

df[df['peak_or_not']=='0']['attn'].median()   # 0.738167578603145
df[df['peak_or_not']=='1']['attn'].median()   # 0.66908304196512

stats.ttest_ind(df[df['peak_or_not']=='0']['attn'], df[df['peak_or_not']=='1']['attn'])  # pvalue=0.0

# fpr, tpr, _ = roc_curve(df['peak_or_not'].apply(int), df['attn'])
# auc(fpr, tpr)  # 0.38628834180140725

## calculate relative binding ratio (obs/exp)
df_random = df.copy()
df_random['attn'] = df_random['attn'].sample(frac=1, random_state=0).values

true_ratio = []
rand_ratio = []
for cnt in [5000, 10000, 15000, 20000, 25000, 30000]:
    true_ratio.append(round(df[df['attn']>=df['attn'].quantile(1-cnt/df['attn'].shape[0])]['peak_or_not'].value_counts(normalize=True)[1], 4))
    rand_ratio.append(round(df_random[df_random['attn']>=df_random['attn'].quantile(1-cnt/df_random['attn'].shape[0])]['peak_or_not'].value_counts(normalize=True)[1], 4))

# true_ratio  [0.0238, 0.0261, 0.0275, 0.0277, 0.0284, 0.0289]
# rand_ratio  [0.0588, 0.0562, 0.0565, 0.055, 0.0558, 0.0561]
true_vs_rand = np.round(np.array(true_ratio)/np.array(rand_ratio), 4)
cnt_ratio = pd.DataFrame({'cnt':[5000, 10000, 15000, 20000, 25000, 30000],
                          'ratio':true_vs_rand})

p = ggplot(cnt_ratio, aes(x='cnt', y='ratio')) + geom_line(size=1) + geom_point(size=1) + xlab('') +\
                                                 scale_x_continuous(limits=[5000, 30000], breaks=np.arange(5000, 30000+1, 5000)) +\
                                                 scale_y_continuous(limits=[0.25, 1.25], breaks=np.arange(0.25, 1.25+0.1, 0.25)) + theme_bw()
p.save(filename='stat3_cd14_mono_cnt_ratio.pdf', dpi=600, height=4, width=4)
################################################################################################################################################


################################################################ CD14 Mono SREBP2 ############################################################################
############ Peak vs. Attn ############
# wget -c https://remap.univ-amu.fr/storage/remap2022/hg38/MACS2/DATASET/GSE129202.SREBP2.monocyte.bed.gz
# gunzip GSE129202.SREBP2.monocyte.bed.gz
# mv GSE129202.SREBP2.monocyte.bed cd14_mono_srebp2_peaks_raw.bed
# awk '{print $1 "\t" int(($2+$3)/2) "\t" int(($2+$3)/2+1)}' cd14_mono_srebp2_peaks_raw.bed | sort -k1,1 -k2,2n | uniq > cd14_mono_srebp2_peaks.bed  # 44464
# bedtools intersect -a ../../peaks.bed -b cd14_mono_srebp2_peaks.bed -wa | sort -k1,1 -k2,2n | uniq | awk '{print $0 "\t" 1}' >> ccre_peaks_raw.bed
# bedtools intersect -a ../../peaks.bed -b cd14_mono_srebp2_peaks.bed -wa -v | sort -k1,1 -k2,2n | uniq | awk '{print $0 "\t" 0}' >> ccre_peaks_raw.bed
# sort -k1,1 -k2,2n ccre_peaks_raw.bed > ccre_peaks.bed
# rm ccre_peaks_raw.bed

import pickle
import pandas as pd
import scanpy as sc

with open('./attn/cd14_mono/attn_20_cd14_mono_3cell.pkl', 'rb') as file:
    gene_attn = pickle.load(file)

for gene in gene_attn:
    gene_attn[gene] = gene_attn[gene].flatten()

gene_attn_srebp2 = pd.DataFrame(gene_attn['SREBP2'])   # error: no 'SREBP2'
# atac_cd14_mono_20 = sc.read_h5ad('atac_cd14_mono_20.h5ad')
# gene_attn_srebp2.index = atac_cd14_mono_20.var.index.values
# gene_attn_srebp2.columns = ['attn']
# gene_attn_srebp2['attn'] = gene_attn_srebp2['attn']/20

# srebp2_peaks = pd.read_table('tf_chipseq/cd14_mono_srebp2/ccre_peaks.bed', header=None)
# srebp2_peaks.index = srebp2_peaks[0]+':'+srebp2_peaks[1].apply(str)+'-'+srebp2_peaks[2].apply(str)
# srebp2_peaks.columns = ['chrom', 'start', 'end', 'peak_or_not']

# df = gene_attn_srebp2.copy()
# df['peak_or_not'] = srebp2_peaks.loc[gene_attn_srebp2.index, 'peak_or_not']
# df.to_csv('./tf_chipseq/cd14_mono_srebp2/srebp2_attn_cd14_mono.csv')
################################################################################################################################################


# CD14 Mono           1988
# CD8 Naive            964
# CD4 TCM              793
# CD4 Naive            734
# CD4 intermediate     363
# CD8 TEM              357
# CD4 TEM              335
# B memory             317
# NK                   306
# CD16 Mono            266
# B naive              229
# cDC2                  88
# MAIT                  86
# Treg                  83
# pDC                   65

K562    ~ NK
GM12878 ~ B cell

## K562 & GM12878 TF ChIP-seq datasets (overlap)  【110 TFs】
############ Peak vs. Attn ############
# gunzip *.bed.gz
# awk '{print $1 "\t" int(($2+$3)/2) "\t" int(($2+$3)/2+1)}' K562_CTCF.bed | sort -k1,1 -k2,2n | uniq > K562_CTCF_peaks.bed  # 37191
# bedtools intersect -a /fs/home/jiluzhang/Nature_methods/TF_binding/peaks.bed -b K562_CTCF_peaks.bed -wa | sort -k1,1 -k2,2n | uniq | awk '{print $0 "\t" 1}' >> ccre_peaks_raw.bed
# bedtools intersect -a /fs/home/jiluzhang/Nature_methods/TF_binding/peaks.bed -b K562_CTCF_peaks.bed -wa -v | sort -k1,1 -k2,2n | uniq | awk '{print $0 "\t" 0}' >> ccre_peaks_raw.bed
# sort -k1,1 -k2,2n ccre_peaks_raw.bed > K562_CTCF_ccre_peaks.bed
# rm ccre_peaks_raw.bed


# for file in `ls *.bed`;do
#     cl_tf=${file//.bed/}
#     awk '{print $1 "\t" int(($2+$3)/2) "\t" int(($2+$3)/2+1)}' $cl_tf.bed | sort -k1,1 -k2,2n | uniq > $cl_tf\_peaks.bed
#     bedtools intersect -a /fs/home/jiluzhang/Nature_methods/TF_binding/peaks.bed -b $cl_tf\_peaks.bed -wa | sort -k1,1 -k2,2n | uniq | awk '{print $0 "\t" 1}' >> ccre_peaks_raw.bed
#     bedtools intersect -a /fs/home/jiluzhang/Nature_methods/TF_binding/peaks.bed -b $cl_tf\_peaks.bed -wa -v | sort -k1,1 -k2,2n | uniq | awk '{print $0 "\t" 0}' >> ccre_peaks_raw.bed
#     sort -k1,1 -k2,2n ccre_peaks_raw.bed > $cl_tf\_ccre_peaks.bed
#     rm $cl_tf\_peaks.bed ccre_peaks_raw.bed
#     echo $cl_tf done
# done


# for file in `ls | grep bed | grep -v ccre`;do wc -l $file >> stat.txt; done
# awk '{if($1>10000) print$2}' stat.txt | sed 's/GM12878_//g' | sed 's/K562_//g' | sed 's/.bed//g' | sort | uniq -c | sed 's/   //g' | sed 's/ /\t/g' | awk '{if($1==2) print $2}' > TFs_10000.txt  # 44
# awk '{if($1>5000) print$2}' stat.txt | sed 's/GM12878_//g' | sed 's/K562_//g' | sed 's/.bed//g' | sort | uniq -c | sed 's/   //g' | sed 's/ /\t/g' | awk '{if($1==2) print $2}' > TFs_5000.txt    # 72

import pickle
import pandas as pd
import scanpy as sc
import numpy as np
from tqdm import tqdm

TFs = pd.read_table('/fs/home/jiluzhang/Nature_methods/TF_binding/tf_chipseq/k562_gm12878/TFs_10000.txt', header=None)
ct_lst = ['cd14_mono', 'cd8_naive', 'cd4_tcm', 'cd4_naive', 'cd4_inter',
          'cd8_tem', 'cd4_tem', 'b_mem', 'nk', 'cd16_mono', 'b_naive']
atac_ct_20 = sc.read_h5ad('atac_cd14_mono_20.h5ad')

def cal_mat(cl='K562'):
    mat = pd.DataFrame(np.zeros([TFs.shape[0], len(ct_lst)]))
    mat.index = TFs[0].values
    mat.columns = ct_lst
    
    for ct in ct_lst:
        with open('/fs/home/jiluzhang/Nature_methods/TF_binding/attn/'+ct+'/attn_20_'+ct+'_3cell.pkl', 'rb') as file:
            gene_attn = pickle.load(file)
        
        for gene in gene_attn:
            gene_attn[gene] = gene_attn[gene].flatten()
        
        for tf in tqdm(TFs[0], desc=cl+' '+ct+' cal ratio', ncols=80):
            if tf in gene_attn.keys():
                gene_attn_tf = pd.DataFrame(gene_attn[tf])
                gene_attn_tf.index = atac_ct_20.var.index.values
                gene_attn_tf.columns = ['attn']
                gene_attn_tf['attn'] = gene_attn_tf['attn']/20
                
                tf_peaks = pd.read_table('/fs/home/jiluzhang/Nature_methods/TF_binding/tf_chipseq/k562_gm12878/'+cl+'_'+tf+'_ccre_peaks.bed', header=None)
                tf_peaks.index = tf_peaks[0]+':'+tf_peaks[1].apply(str)+'-'+tf_peaks[2].apply(str)
                tf_peaks.columns = ['chrom', 'start', 'end', 'peak_or_not']
    
                df = gene_attn_tf.copy()
                df['peak_or_not'] = tf_peaks.loc[gene_attn_tf.index, 'peak_or_not']
                df['peak_or_not'] = df['peak_or_not'].apply(str)
                df = df.replace([np.inf, -np.inf], np.nan).dropna()
                # df['attn'] = np.log10(df['attn']/df['attn'].min())
                # df['attn'] = (df['attn']-df['attn'].min())/(df['attn'].max()-df['attn'].min())
                
                df_random = df.copy()
                df_random['attn'] = df_random['attn'].sample(frac=1, random_state=0).values
                
                true_ratio = round(df[df['attn']>=df['attn'].quantile(1-10000/df['attn'].shape[0])]['peak_or_not'].value_counts(normalize=True).iloc[1], 4)  # select top 10,000
                rand_ratio = round(df_random[df_random['attn']>=df_random['attn'].quantile(1-10000/df_random['attn'].shape[0])]['peak_or_not'].value_counts(normalize=True).iloc[1], 4)
                
                mat.loc[tf, ct] = np.round(true_ratio/rand_ratio, 4)
    return mat

k562_mat = cal_mat(cl='K562')
k562_mat.to_csv('k562_tf_10000_ratio.txt', sep='\t')

gm12878_mat = cal_mat(cl='GM12878')
gm12878_mat.to_csv('gm12878_tf_10000_ratio.txt', sep='\t')

##############################################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from plotnine import *

k562_df = pd.read_table('k562_tf_10000_ratio.txt', index_col=0)

## heatmap
# plt.figure(figsize=(6, 6))
# #sns.heatmap(k562_df, cmap='bwr', vmin=0, vmax=2, xticklabels=False, yticklabels=False)  # cbar=False, vmin=0, vmax=10, linewidths=0.1, linecolor='grey'
# sns.clustermap(k562_df, cmap='bwr', vmin=0, vmax=2, xticklabels=True, yticklabels=True)  # cbar=False, vmin=0, vmax=10, linewidths=0.1, linecolor='grey'
# plt.savefig('k562_tf_10000_ratio.pdf')
# plt.close()

## boxplot
k562_df = pd.melt(k562_df)
k562_df.columns = ['ct', 'val']
k562_df = k562_df[k562_df['val']!=0]

plt.rcParams['pdf.fonttype'] = 42
p = ggplot(k562_df, aes(x='ct', y='val', fill='ct')) + geom_boxplot(width=0.5, outlier_shape='', show_legend=False) +\
                                                       scale_y_continuous(limits=[0, 2], breaks=np.arange(0, 2+0.1, 0.5)) + theme_bw()
p.save(filename='k562_tf_10000_ratio.pdf', dpi=600, height=4, width=10)


gm12878_df = pd.read_table('gm12878_tf_10000_ratio.txt', index_col=0)

## boxplot
gm12878_df = pd.melt(gm12878_df)
gm12878_df.columns = ['ct', 'val']
gm12878_df = gm12878_df[gm12878_df['val']!=0]

plt.rcParams['pdf.fonttype'] = 42
p = ggplot(gm12878_df, aes(x='ct', y='val', fill='ct')) + geom_boxplot(width=0.5, outlier_shape='', show_legend=False) +\
                                                          scale_y_continuous(limits=[0, 2], breaks=np.arange(0, 2+0.1, 0.5)) + theme_bw()
p.save(filename='gm12878_tf_10000_ratio.pdf', dpi=600, height=4, width=10)
