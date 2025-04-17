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
device = torch.device('cuda:7')
model.to(device)
model.eval()

# motif_info_raw = pd.read_table('peaks_motifs_count.txt', header=0)
# motif_info = motif_info_raw.copy()
# motif_info[motif_info!=0] = 1
# motif_info = motif_info.sum(axis=1)

# tf_lst = []
# tf_idx = []
# for tf in motif_info_raw.columns:
#     tf_out = tf.split('_')[1]
#     if '.' not in tf_out:
#         tf_lst.append(tf_out)
#         tf_idx.append(True)
#     else:
#         tf_idx.append(False)
# maybe count is too small

cr_lst = ['SMARCA4', 'SMARCA2', 'ARID1A', 'ARID1B', 'SMARCB1',
          'CHD1', 'CHD2', 'CHD3', 'CHD4', 'CHD5', 'CHD6', 'CHD7', 'CHD8', 'CHD9',
          'BRD2', 'BRD3', 'BRD4', 'BRDT',
          'SMARCA5', 'SMARCA1', 'ACTL6A', 'ACTL6B',
          'SSRP1', 'SUPT16H',
          'EP400',
          'SMARCD1', 'SMARCD2', 'SMARCD3']   # 28
tf_raw = pd.read_table('attn/TF_jaspar.txt', header=None)
tf_lst = list(tf_raw[0].values)   # 735
cr_tf_lst = cr_lst + tf_lst   # 763

################ CD14 Mono ################
random.seed(827)
cd14_mono_idx = list(np.argwhere(rna.obs['cell_anno']=='CD14 Mono').flatten())
cd14_mono_idx_100 = random.sample(list(cd14_mono_idx), 100)
rna[cd14_mono_idx_100].write('rna_cd14_mono_100.h5ad')
np.count_nonzero(rna[cd14_mono_idx_100].X.toarray(), axis=1).max()  # 3336

atac[cd14_mono_idx_100].write('atac_cd14_mono_100.h5ad')

# python data_preprocess.py -r rna_cd14_mono_100.h5ad -a atac_cd14_mono_100.h5ad -s pt_cd14_mono_100 --dt test -n cd14_mono_100 --config rna2atac_config_test_cd14_mono.yaml
# enc_max_len: 3336

cd14_mono_data = torch.load("./pt_cd14_mono_100/cd14_mono_100_0.pt")
dataset = PreDataset(cd14_mono_data)
dataloader_kwargs = {'batch_size': 1, 'shuffle': False}
loader = torch.utils.data.DataLoader(dataset, **dataloader_kwargs)

rna_cd14_mono = sc.read_h5ad('rna_cd14_mono_100.h5ad')
atac_cd14_mono = sc.read_h5ad('atac_cd14_mono_100.h5ad')

out = rna_cd14_mono.copy()
out.X = np.zeros(out.shape)
i = 0
for inputs in tqdm(loader, ncols=80, desc='output attention matrix'):
    rna_sequence, rna_value, atac_sequence, _, enc_pad_mask = [each.to(device) for each in inputs]
    attn_tmp = model.generate_attn_weight(rna_sequence, atac_sequence, rna_value, enc_mask=enc_pad_mask, which='decoder')[0]
    out.X[i, (rna_sequence[rna_sequence!=0]-1).cpu()] = np.nansum(attn_tmp, axis=0)
    torch.cuda.empty_cache()
    i += 1

out.write('attn_cd14_mono_100_gene.h5ad')

attn = sc.read_h5ad('attn_cd14_mono_100_gene.h5ad')

df = pd.DataFrame({'attn':attn.X.sum(axis=0)})
df.index = out.var.index.values
df = df[df['attn']!=0]  # remove genes with no attention score
df['attn'] = np.log10(df['attn'])
df['attn'] = (df['attn']-df['attn'].min())/(df['attn'].max()-df['attn'].min())

gene_idx = []
for g in df.index:
    if g in cr_lst:
        gene_idx.append('CR')
    elif g in tf_lst:
        gene_idx.append('TF')
    else:
        gene_idx.append('Others')

df['gene_idx'] = gene_idx
df.to_csv('attn/gene_attn_cd14_mono_norm.txt', sep='\t')

## for k562_gm12878
TFs_chipseq = pd.read_table('/fs/home/jiluzhang/Nature_methods/TF_binding/tf_chipseq/k562_gm12878/TFs_10000.txt', header=None)
tfs_lst = np.intersect1d(TFs_chipseq[0].values, rna_cd14_mono.var.index)
out_2 = pd.DataFrame(np.zeros([atac_cd14_mono.n_vars, len(tfs_lst)]))
out_2.columns = [np.argwhere(rna_cd14_mono.var.index==tf).item() for tf in tfs_lst]
i = 0
for inputs in tqdm(loader, ncols=80, desc='output attention matrix'):
    rna_sequence, rna_value, atac_sequence, _, enc_pad_mask = [each.to(device) for each in inputs]
    attn_m = pd.DataFrame(model.generate_attn_weight(rna_sequence, atac_sequence, rna_value, enc_mask=enc_pad_mask, which='decoder')[0])
    attn_m.columns = (rna_sequence[rna_sequence!=0]-1).cpu().numpy()
    ovp_idx = np.intersect1d(out_2.columns, attn_m.columns)
    out_2.loc[:, ovp_idx] += attn_m.loc[:, ovp_idx]
    torch.cuda.empty_cache()
    i += 1

out_2.index = atac_cd14_mono.var.index.values
out_2.columns = tfs_lst
out_2.to_csv('attn_cd14_mono_100_peak_gene.txt', sep='\t')

## plot box
import pandas as pd
from plotnine import *
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

plt.rcParams['pdf.fonttype'] = 42

df = pd.read_table('gene_attn_cd14_mono_norm.txt', index_col=0)
df['gene_idx'] = pd.Categorical(df['gene_idx'], categories=['CR', 'TF', 'Others'])

p = ggplot(df, aes(x='gene_idx', y='attn', fill='gene_idx')) + geom_boxplot(width=0.5, show_legend=False, outlier_shape='') + xlab('') +\
                                                               coord_cartesian(ylim=(0.2, 1.0)) +\
                                                               scale_y_continuous(breaks=np.arange(0.2, 1+0.1, 0.2)) + theme_bw()
p.save(filename='gene_attn_cd14_mono_norm.pdf', dpi=600, height=4, width=4)

df[df['gene_idx']=='CR']['attn'].median()          # 0.6454177518930972
df[df['gene_idx']=='TF']['attn'].median()          # 0.6313849450007517
df[df['gene_idx']=='Others']['attn'].median()      # 0.6254420337306308

stats.ttest_ind(df[df['gene_idx']=='CR']['attn'], df[df['gene_idx']=='TF']['attn'])      # pvalue=0.23953398259655
stats.ttest_ind(df[df['gene_idx']=='CR']['attn'], df[df['gene_idx']=='Others']['attn'])  # pvalue=0.1271388349028105
stats.ttest_ind(df[df['gene_idx']=='TF']['attn'], df[df['gene_idx']=='Others']['attn'])  # pvalue=0.3990297335773566


################ CD8 Naive ################
random.seed(827)
cd8_naive_idx = list(np.argwhere(rna.obs['cell_anno']=='CD8 Naive').flatten())
cd8_naive_idx_100 = random.sample(list(cd8_naive_idx), 100)
rna[cd8_naive_idx_100].write('rna_cd8_naive_100.h5ad')
np.count_nonzero(rna[cd8_naive_idx_100].X.toarray(), axis=1).max()  # 2056

atac[cd8_naive_idx_100].write('atac_cd8_naive_100.h5ad')

# python data_preprocess.py -r rna_cd8_naive_100.h5ad -a atac_cd8_naive_100.h5ad -s pt_cd8_naive_100 --dt test -n cd8_naive_100 --config rna2atac_config_test_cd8_naive.yaml
# enc_max_len: 2056

cd8_naive_data = torch.load("./pt_cd8_naive_100/cd8_naive_100_0.pt")
dataset = PreDataset(cd8_naive_data)
dataloader_kwargs = {'batch_size': 1, 'shuffle': False}
loader = torch.utils.data.DataLoader(dataset, **dataloader_kwargs)

rna_cd8_naive = sc.read_h5ad('rna_cd8_naive_100.h5ad')
atac_cd8_naive = sc.read_h5ad('atac_cd8_naive_100.h5ad')

out = rna_cd8_naive.copy()
out.X = np.zeros(out.shape)
i = 0
for inputs in tqdm(loader, ncols=80, desc='output attention matrix'):
    rna_sequence, rna_value, atac_sequence, _, enc_pad_mask = [each.to(device) for each in inputs]
    attn_tmp = model.generate_attn_weight(rna_sequence, atac_sequence, rna_value, enc_mask=enc_pad_mask, which='decoder')[0]
    out.X[i, (rna_sequence[rna_sequence!=0]-1).cpu()] = np.nansum(attn_tmp, axis=0)
    torch.cuda.empty_cache()
    i += 1

out.write('attn_cd8_naive_100_gene.h5ad')

attn = sc.read_h5ad('attn_cd8_naive_100_gene.h5ad')

df = pd.DataFrame({'attn':attn.X.sum(axis=0)})
df.index = out.var.index.values
df = df[df['attn']!=0]
df['attn'] = np.log10(df['attn'])
df['attn'] = (df['attn']-df['attn'].min())/(df['attn'].max()-df['attn'].min())

gene_idx = []
for g in df.index:
    if g in cr_lst:
        gene_idx.append('CR')
    elif g in tf_lst:
        gene_idx.append('TF')
    else:
        gene_idx.append('Others')

df['gene_idx'] = gene_idx
df.to_csv('attn/gene_attn_cd8_naive_norm.txt', sep='\t')

## for k562_gm12878
TFs_chipseq = pd.read_table('/fs/home/jiluzhang/Nature_methods/TF_binding/tf_chipseq/k562_gm12878/TFs_10000.txt', header=None)
tfs_lst = np.intersect1d(TFs_chipseq[0].values, rna_cd8_naive.var.index)
out_2 = pd.DataFrame(np.zeros([atac_cd8_naive.n_vars, len(tfs_lst)]))
out_2.columns = [np.argwhere(rna_cd8_naive.var.index==tf).item() for tf in tfs_lst]
i = 0
for inputs in tqdm(loader, ncols=80, desc='output attention matrix'):
    rna_sequence, rna_value, atac_sequence, _, enc_pad_mask = [each.to(device) for each in inputs]
    attn_m = pd.DataFrame(model.generate_attn_weight(rna_sequence, atac_sequence, rna_value, enc_mask=enc_pad_mask, which='decoder')[0])
    attn_m.columns = (rna_sequence[rna_sequence!=0]-1).cpu().numpy()
    ovp_idx = np.intersect1d(out_2.columns, attn_m.columns)
    out_2.loc[:, ovp_idx] += attn_m.loc[:, ovp_idx]
    torch.cuda.empty_cache()
    i += 1

out_2.index = atac_cd8_naive.var.index.values
out_2.columns = tfs_lst
out_2.to_csv('attn_cd8_naive_100_peak_gene.txt', sep='\t')

## plot box
df = pd.read_table('gene_attn_cd8_naive_norm.txt', index_col=0)
df['gene_idx'] = pd.Categorical(df['gene_idx'], categories=['CR', 'TF', 'Others'])

p = ggplot(df, aes(x='gene_idx', y='attn', fill='gene_idx')) + geom_boxplot(width=0.5, show_legend=False, outlier_shape='') + xlab('') +\
                                                               coord_cartesian(ylim=(0.2, 1.0)) +\
                                                               scale_y_continuous(breaks=np.arange(0.2, 1.0+0.1, 0.2)) + theme_bw()
p.save(filename='gene_attn_cd8_naive_norm.pdf', dpi=600, height=4, width=4)

df[df['gene_idx']=='CR']['attn'].median()          # 0.6035652929381616
df[df['gene_idx']=='TF']['attn'].median()          # 0.5815663267884156
df[df['gene_idx']=='Others']['attn'].median()      # 0.5735152981753229

stats.ttest_ind(df[df['gene_idx']=='CR']['attn'], df[df['gene_idx']=='TF']['attn'])      # pvalue=0.12725422209252021
stats.ttest_ind(df[df['gene_idx']=='CR']['attn'], df[df['gene_idx']=='Others']['attn'])  # pvalue=0.044194324992587616
stats.ttest_ind(df[df['gene_idx']=='TF']['attn'], df[df['gene_idx']=='Others']['attn'])  # pvalue=0.23630539262304362

################ CD4 TCM ################
random.seed(827)
cd4_tcm_idx = list(np.argwhere(rna.obs['cell_anno']=='CD4 TCM').flatten())
cd4_tcm_idx_100 = random.sample(list(cd4_tcm_idx), 100)
rna[cd4_tcm_idx_100].write('rna_cd4_tcm_100.h5ad')
np.count_nonzero(rna[cd4_tcm_idx_100].X.toarray(), axis=1).max()  # 3178

atac[cd4_tcm_idx_100].write('atac_cd4_tcm_100.h5ad')

# python data_preprocess.py -r rna_cd4_tcm_100.h5ad -a atac_cd4_tcm_100.h5ad -s pt_cd4_tcm_100 --dt test -n cd4_tcm_100 --config rna2atac_config_test_cd4_tcm.yaml
# enc_max_len: 3178

cd4_tcm_data = torch.load("./pt_cd4_tcm_100/cd4_tcm_100_0.pt")
dataset = PreDataset(cd4_tcm_data)
dataloader_kwargs = {'batch_size': 1, 'shuffle': False}
loader = torch.utils.data.DataLoader(dataset, **dataloader_kwargs)

rna_cd4_tcm = sc.read_h5ad('rna_cd4_tcm_100.h5ad')
atac_cd4_tcm = sc.read_h5ad('atac_cd4_tcm_100.h5ad')

out = rna_cd4_tcm.copy()
out.X = np.zeros(out.shape)
i = 0
for inputs in tqdm(loader, ncols=80, desc='output attention matrix'):
    rna_sequence, rna_value, atac_sequence, _, enc_pad_mask = [each.to(device) for each in inputs]
    attn_tmp = model.generate_attn_weight(rna_sequence, atac_sequence, rna_value, enc_mask=enc_pad_mask, which='decoder')[0]
    out.X[i, (rna_sequence[rna_sequence!=0]-1).cpu()] = np.nansum(attn_tmp, axis=0)
    torch.cuda.empty_cache()
    i += 1

out.write('attn_cd4_tcm_100_gene.h5ad')

attn = sc.read_h5ad('attn_cd4_tcm_100_gene.h5ad')

df = pd.DataFrame({'attn':attn.X.sum(axis=0)})
df.index = out.var.index.values
df = df[df['attn']!=0]
df['attn'] = np.log10(df['attn'])
df['attn'] = (df['attn']-df['attn'].min())/(df['attn'].max()-df['attn'].min())

gene_idx = []
for g in df.index:
    if g in cr_lst:
        gene_idx.append('CR')
    elif g in tf_lst:
        gene_idx.append('TF')
    else:
        gene_idx.append('Others')

df['gene_idx'] = gene_idx
df.to_csv('attn/gene_attn_cd4_tcm_norm.txt', sep='\t')

## for k562_gm12878
TFs_chipseq = pd.read_table('/fs/home/jiluzhang/Nature_methods/TF_binding/tf_chipseq/k562_gm12878/TFs_10000.txt', header=None)
tfs_lst = np.intersect1d(TFs_chipseq[0].values, rna_cd4_tcm.var.index)
out_2 = pd.DataFrame(np.zeros([atac_cd4_tcm.n_vars, len(tfs_lst)]))
out_2.columns = [np.argwhere(rna_cd4_tcm.var.index==tf).item() for tf in tfs_lst]
i = 0
for inputs in tqdm(loader, ncols=80, desc='output attention matrix'):
    rna_sequence, rna_value, atac_sequence, _, enc_pad_mask = [each.to(device) for each in inputs]
    attn_m = pd.DataFrame(model.generate_attn_weight(rna_sequence, atac_sequence, rna_value, enc_mask=enc_pad_mask, which='decoder')[0])
    attn_m.columns = (rna_sequence[rna_sequence!=0]-1).cpu().numpy()
    ovp_idx = np.intersect1d(out_2.columns, attn_m.columns)
    out_2.loc[:, ovp_idx] += attn_m.loc[:, ovp_idx]
    torch.cuda.empty_cache()
    i += 1

out_2.index = atac_cd4_tcm.var.index.values
out_2.columns = tfs_lst
out_2.to_csv('attn_cd4_tcm_100_peak_gene.txt', sep='\t')

## plot box
df = pd.read_table('gene_attn_cd4_tcm_norm.txt', index_col=0)
df['gene_idx'] = pd.Categorical(df['gene_idx'], categories=['CR', 'TF', 'Others'])

p = ggplot(df, aes(x='gene_idx', y='attn', fill='gene_idx')) + geom_boxplot(width=0.5, show_legend=False, outlier_shape='') + xlab('') +\
                                                               coord_cartesian(ylim=(0.2, 1.0)) +\
                                                               scale_y_continuous(breaks=np.arange(0.2, 1.0+0.1, 0.2)) + theme_bw()
p.save(filename='gene_attn_cd4_tcm_norm.pdf', dpi=600, height=4, width=4)

df[df['gene_idx']=='CR']['attn'].median()          # 0.5771065563622687
df[df['gene_idx']=='TF']['attn'].median()          # 0.5556245895484668
df[df['gene_idx']=='Others']['attn'].median()      # 0.549114708611519

stats.ttest_ind(df[df['gene_idx']=='CR']['attn'], df[df['gene_idx']=='TF']['attn'])      # pvalue=0.1560173475231402
stats.ttest_ind(df[df['gene_idx']=='CR']['attn'], df[df['gene_idx']=='Others']['attn'])  # pvalue=0.051429821750443194
stats.ttest_ind(df[df['gene_idx']=='TF']['attn'], df[df['gene_idx']=='Others']['attn'])  # pvalue=0.14362160547127634



################ CD4 Naive ################
random.seed(827)
cd4_naive_idx = list(np.argwhere(rna.obs['cell_anno']=='CD4 Naive').flatten())
cd4_naive_idx_100 = random.sample(list(cd4_naive_idx), 100)
rna[cd4_naive_idx_100].write('rna_cd4_naive_100.h5ad')
np.count_nonzero(rna[cd4_naive_idx_100].X.toarray(), axis=1).max()  # 2307

atac[cd4_naive_idx_100].write('atac_cd4_naive_100.h5ad')

# python data_preprocess.py -r rna_cd4_naive_100.h5ad -a atac_cd4_naive_100.h5ad -s pt_cd4_naive_100 --dt test -n cd4_naive_100 --config rna2atac_config_test_cd4_naive.yaml
# enc_max_len: 2307

cd4_naive_data = torch.load("./pt_cd4_naive_100/cd4_naive_100_0.pt")
dataset = PreDataset(cd4_naive_data)
dataloader_kwargs = {'batch_size': 1, 'shuffle': False}
loader = torch.utils.data.DataLoader(dataset, **dataloader_kwargs)

rna_cd4_naive = sc.read_h5ad('rna_cd4_naive_100.h5ad')
atac_cd4_naive = sc.read_h5ad('atac_cd4_naive_100.h5ad')

out = rna_cd4_naive.copy()
out.X = np.zeros(out.shape)
i = 0
for inputs in tqdm(loader, ncols=80, desc='output attention matrix'):
    rna_sequence, rna_value, atac_sequence, _, enc_pad_mask = [each.to(device) for each in inputs]
    attn_tmp = model.generate_attn_weight(rna_sequence, atac_sequence, rna_value, enc_mask=enc_pad_mask, which='decoder')[0]
    out.X[i, (rna_sequence[rna_sequence!=0]-1).cpu()] = np.nansum(attn_tmp, axis=0)
    torch.cuda.empty_cache()
    i += 1

out.write('attn_cd4_naive_100_gene.h5ad')

attn = sc.read_h5ad('attn_cd4_naive_100_gene.h5ad')

df = pd.DataFrame({'attn':attn.X.sum(axis=0)})
df.index = out.var.index.values
df = df[df['attn']!=0]
df['attn'] = np.log10(df['attn'])
df['attn'] = (df['attn']-df['attn'].min())/(df['attn'].max()-df['attn'].min())

gene_idx = []
for g in df.index:
    if g in cr_lst:
        gene_idx.append('CR')
    elif g in tf_lst:
        gene_idx.append('TF')
    else:
        gene_idx.append('Others')

df['gene_idx'] = gene_idx
df.to_csv('attn/gene_attn_cd4_naive_norm.txt', sep='\t')

## for k562_gm12878
TFs_chipseq = pd.read_table('/fs/home/jiluzhang/Nature_methods/TF_binding/tf_chipseq/k562_gm12878/TFs_10000.txt', header=None)
tfs_lst = np.intersect1d(TFs_chipseq[0].values, rna_cd4_naive.var.index)
out_2 = pd.DataFrame(np.zeros([atac_cd4_naive.n_vars, len(tfs_lst)]))
out_2.columns = [np.argwhere(rna_cd4_naive.var.index==tf).item() for tf in tfs_lst]
i = 0
for inputs in tqdm(loader, ncols=80, desc='output attention matrix'):
    rna_sequence, rna_value, atac_sequence, _, enc_pad_mask = [each.to(device) for each in inputs]
    attn_m = pd.DataFrame(model.generate_attn_weight(rna_sequence, atac_sequence, rna_value, enc_mask=enc_pad_mask, which='decoder')[0])
    attn_m.columns = (rna_sequence[rna_sequence!=0]-1).cpu().numpy()
    ovp_idx = np.intersect1d(out_2.columns, attn_m.columns)
    out_2.loc[:, ovp_idx] += attn_m.loc[:, ovp_idx]
    torch.cuda.empty_cache()
    i += 1

out_2.index = atac_cd4_naive.var.index.values
out_2.columns = tfs_lst
out_2.to_csv('attn_cd4_naive_100_peak_gene.txt', sep='\t')

## plot box
df = pd.read_table('gene_attn_cd4_naive_norm.txt', index_col=0)
df['gene_idx'] = pd.Categorical(df['gene_idx'], categories=['CR', 'TF', 'Others'])

p = ggplot(df, aes(x='gene_idx', y='attn', fill='gene_idx')) + geom_boxplot(width=0.5, show_legend=False, outlier_shape='') + xlab('') +\
                                                               coord_cartesian(ylim=(0.2, 1.0)) +\
                                                               scale_y_continuous(breaks=np.arange(0.2, 1.0+0.1, 0.2)) + theme_bw()
p.save(filename='gene_attn_cd4_naive_norm.pdf', dpi=600, height=4, width=4)

df[df['gene_idx']=='CR']['attn'].median()          # 0.6413345888992622
df[df['gene_idx']=='TF']['attn'].median()          # 0.6191496034721382
df[df['gene_idx']=='Others']['attn'].median()      # 0.6080437485070576

stats.ttest_ind(df[df['gene_idx']=='CR']['attn'], df[df['gene_idx']=='TF']['attn'])      # pvalue=0.14778059262283616
stats.ttest_ind(df[df['gene_idx']=='CR']['attn'], df[df['gene_idx']=='Others']['attn'])  # pvalue=0.058211211936213605
stats.ttest_ind(df[df['gene_idx']=='TF']['attn'], df[df['gene_idx']=='Others']['attn'])  # pvalue=0.27588136254382695

################ CD4 Intermediate ################
random.seed(827)
cd4_inter_idx = list(np.argwhere(rna.obs['cell_anno']=='CD4 intermediate').flatten())
cd4_inter_idx_100 = random.sample(list(cd4_inter_idx), 100)
rna[cd4_inter_idx_100].write('rna_cd4_inter_100.h5ad')
np.count_nonzero(rna[cd4_inter_idx_100].X.toarray(), axis=1).max()  # 2851

atac[cd4_inter_idx_100].write('atac_cd4_inter_100.h5ad')

# python data_preprocess.py -r rna_cd4_inter_100.h5ad -a atac_cd4_inter_100.h5ad -s pt_cd4_inter_100 --dt test -n cd4_inter_100 --config rna2atac_config_test_cd4_inter.yaml
# enc_max_len: 2851

cd4_inter_data = torch.load("./pt_cd4_inter_100/cd4_inter_100_0.pt")
dataset = PreDataset(cd4_inter_data)
dataloader_kwargs = {'batch_size': 1, 'shuffle': False}
loader = torch.utils.data.DataLoader(dataset, **dataloader_kwargs)

rna_cd4_inter = sc.read_h5ad('rna_cd4_inter_100.h5ad')
atac_cd4_inter = sc.read_h5ad('atac_cd4_inter_100.h5ad')

out = rna_cd4_inter.copy()
out.X = np.zeros(out.shape)
i = 0
for inputs in tqdm(loader, ncols=80, desc='output attention matrix'):
    rna_sequence, rna_value, atac_sequence, _, enc_pad_mask = [each.to(device) for each in inputs]
    attn_tmp = model.generate_attn_weight(rna_sequence, atac_sequence, rna_value, enc_mask=enc_pad_mask, which='decoder')[0]
    out.X[i, (rna_sequence[rna_sequence!=0]-1).cpu()] = np.nansum(attn_tmp, axis=0)
    torch.cuda.empty_cache()
    i += 1

out.write('attn_cd4_inter_100_gene.h5ad')

attn = sc.read_h5ad('attn_cd4_inter_100_gene.h5ad')

df = pd.DataFrame({'attn':attn.X.sum(axis=0)})
df.index = out.var.index.values
df = df[df['attn']!=0]
df['attn'] = np.log10(df['attn'])
df['attn'] = (df['attn']-df['attn'].min())/(df['attn'].max()-df['attn'].min())

gene_idx = []
for g in df.index:
    if g in cr_lst:
        gene_idx.append('CR')
    elif g in tf_lst:
        gene_idx.append('TF')
    else:
        gene_idx.append('Others')

df['gene_idx'] = gene_idx
df.to_csv('attn/gene_attn_cd4_inter_norm.txt', sep='\t')

## for k562_gm12878
TFs_chipseq = pd.read_table('/fs/home/jiluzhang/Nature_methods/TF_binding/tf_chipseq/k562_gm12878/TFs_10000.txt', header=None)
tfs_lst = np.intersect1d(TFs_chipseq[0].values, rna_cd4_inter.var.index)
out_2 = pd.DataFrame(np.zeros([atac_cd4_inter.n_vars, len(tfs_lst)]))
out_2.columns = [np.argwhere(rna_cd4_inter.var.index==tf).item() for tf in tfs_lst]
i = 0
for inputs in tqdm(loader, ncols=80, desc='output attention matrix'):
    rna_sequence, rna_value, atac_sequence, _, enc_pad_mask = [each.to(device) for each in inputs]
    attn_m = pd.DataFrame(model.generate_attn_weight(rna_sequence, atac_sequence, rna_value, enc_mask=enc_pad_mask, which='decoder')[0])
    attn_m.columns = (rna_sequence[rna_sequence!=0]-1).cpu().numpy()
    ovp_idx = np.intersect1d(out_2.columns, attn_m.columns)
    out_2.loc[:, ovp_idx] += attn_m.loc[:, ovp_idx]
    torch.cuda.empty_cache()
    i += 1

out_2.index = atac_cd4_inter.var.index.values
out_2.columns = tfs_lst
out_2.to_csv('attn_cd4_inter_100_peak_gene.txt', sep='\t')

## plot box
df = pd.read_table('gene_attn_cd4_inter_norm.txt', index_col=0)
df['gene_idx'] = pd.Categorical(df['gene_idx'], categories=['CR', 'TF', 'Others'])

p = ggplot(df, aes(x='gene_idx', y='attn', fill='gene_idx')) + geom_boxplot(width=0.5, show_legend=False, outlier_shape='') + xlab('') +\
                                                               coord_cartesian(ylim=(0.2, 1.0)) +\
                                                               scale_y_continuous(breaks=np.arange(0.2, 1.0+0.1, 0.2)) + theme_bw()
p.save(filename='gene_attn_cd4_inter_norm.pdf', dpi=600, height=4, width=4)

df[df['gene_idx']=='CR']['attn'].median()          # 0.6212076023905401
df[df['gene_idx']=='TF']['attn'].median()          # 0.5988560647495268
df[df['gene_idx']=='Others']['attn'].median()      # 0.5932509238313652

stats.ttest_ind(df[df['gene_idx']=='CR']['attn'], df[df['gene_idx']=='TF']['attn'])      # pvalue=0.1395801947674662
stats.ttest_ind(df[df['gene_idx']=='CR']['attn'], df[df['gene_idx']=='Others']['attn'])  # pvalue=0.04684837217466316
stats.ttest_ind(df[df['gene_idx']=='TF']['attn'], df[df['gene_idx']=='Others']['attn'])  # pvalue=0.15416150338435375

################ CD8 TEM ################
random.seed(827)
cd8_tem_idx = list(np.argwhere(rna.obs['cell_anno']=='CD8 TEM').flatten())
cd8_tem_idx_100 = random.sample(list(cd8_tem_idx), 100)
rna[cd8_tem_idx_100].write('rna_cd8_tem_100.h5ad')
np.count_nonzero(rna[cd8_tem_idx_100].X.toarray(), axis=1).max()  # 3450

atac[cd8_tem_idx_100].write('atac_cd8_tem_100.h5ad')

# python data_preprocess.py -r rna_cd8_tem_100.h5ad -a atac_cd8_tem_100.h5ad -s pt_cd8_tem_100 --dt test -n cd8_tem_100 --config rna2atac_config_test_cd8_tem.yaml
# enc_max_len: 3450

cd8_tem_data = torch.load("./pt_cd8_tem_100/cd8_tem_100_0.pt")
dataset = PreDataset(cd8_tem_data)
dataloader_kwargs = {'batch_size': 1, 'shuffle': False}
loader = torch.utils.data.DataLoader(dataset, **dataloader_kwargs)

rna_cd8_tem = sc.read_h5ad('rna_cd8_tem_100.h5ad')
atac_cd8_tem = sc.read_h5ad('atac_cd8_tem_100.h5ad')

out = rna_cd8_tem.copy()
out.X = np.zeros(out.shape)
i = 0
for inputs in tqdm(loader, ncols=80, desc='output attention matrix'):
    rna_sequence, rna_value, atac_sequence, _, enc_pad_mask = [each.to(device) for each in inputs]
    attn_tmp = model.generate_attn_weight(rna_sequence, atac_sequence, rna_value, enc_mask=enc_pad_mask, which='decoder')[0]
    out.X[i, (rna_sequence[rna_sequence!=0]-1).cpu()] = np.nansum(attn_tmp, axis=0)
    torch.cuda.empty_cache()
    i += 1

out.write('attn_cd8_tem_100_gene.h5ad')

attn = sc.read_h5ad('attn_cd8_tem_100_gene.h5ad')

df = pd.DataFrame({'attn':attn.X.sum(axis=0)})
df.index = out.var.index.values
df = df[df['attn']!=0]
df['attn'] = np.log10(df['attn'])
df['attn'] = (df['attn']-df['attn'].min())/(df['attn'].max()-df['attn'].min())

gene_idx = []
for g in df.index:
    if g in cr_lst:
        gene_idx.append('CR')
    elif g in tf_lst:
        gene_idx.append('TF')
    else:
        gene_idx.append('Others')

df['gene_idx'] = gene_idx
df.to_csv('attn/gene_attn_cd8_tem_norm.txt', sep='\t')

## for k562_gm12878
TFs_chipseq = pd.read_table('/fs/home/jiluzhang/Nature_methods/TF_binding/tf_chipseq/k562_gm12878/TFs_10000.txt', header=None)
tfs_lst = np.intersect1d(TFs_chipseq[0].values, rna_cd8_tem.var.index)
out_2 = pd.DataFrame(np.zeros([atac_cd8_tem.n_vars, len(tfs_lst)]))
out_2.columns = [np.argwhere(rna_cd8_tem.var.index==tf).item() for tf in tfs_lst]
i = 0
for inputs in tqdm(loader, ncols=80, desc='output attention matrix'):
    rna_sequence, rna_value, atac_sequence, _, enc_pad_mask = [each.to(device) for each in inputs]
    attn_m = pd.DataFrame(model.generate_attn_weight(rna_sequence, atac_sequence, rna_value, enc_mask=enc_pad_mask, which='decoder')[0])
    attn_m.columns = (rna_sequence[rna_sequence!=0]-1).cpu().numpy()
    ovp_idx = np.intersect1d(out_2.columns, attn_m.columns)
    out_2.loc[:, ovp_idx] += attn_m.loc[:, ovp_idx]
    torch.cuda.empty_cache()
    i += 1

out_2.index = atac_cd8_tem.var.index.values
out_2.columns = tfs_lst
out_2.to_csv('attn_cd8_tem_100_peak_gene.txt', sep='\t')

## plot box
df = pd.read_table('gene_attn_cd8_tem_norm.txt', index_col=0)
df['gene_idx'] = pd.Categorical(df['gene_idx'], categories=['CR', 'TF', 'Others'])

p = ggplot(df, aes(x='gene_idx', y='attn', fill='gene_idx')) + geom_boxplot(width=0.5, show_legend=False, outlier_shape='') + xlab('') +\
                                                               coord_cartesian(ylim=(0.2, 1.0)) +\
                                                               scale_y_continuous(breaks=np.arange(0.2, 1.0+0.1, 0.2)) + theme_bw()
p.save(filename='gene_attn_cd8_tem_norm.pdf', dpi=600, height=4, width=4)

df[df['gene_idx']=='CR']['attn'].median()          # 0.582101111708013
df[df['gene_idx']=='TF']['attn'].median()          # 0.5738423336434949
df[df['gene_idx']=='Others']['attn'].median()      # 0.5625384788134904

stats.ttest_ind(df[df['gene_idx']=='CR']['attn'], df[df['gene_idx']=='TF']['attn'])      # pvalue=0.3485796082352455
stats.ttest_ind(df[df['gene_idx']=='CR']['attn'], df[df['gene_idx']=='Others']['attn'])  # pvalue=0.10478131921037626
stats.ttest_ind(df[df['gene_idx']=='TF']['attn'], df[df['gene_idx']=='Others']['attn'])  # pvalue=0.025339326545570997

################ CD4 TEM ################
random.seed(827)
cd4_tem_idx = list(np.argwhere(rna.obs['cell_anno']=='CD4 TEM').flatten())
cd4_tem_idx_100 = random.sample(list(cd4_tem_idx), 100)
rna[cd4_tem_idx_100].write('rna_cd4_tem_100.h5ad')
np.count_nonzero(rna[cd4_tem_idx_100].X.toarray(), axis=1).max()  # 2971

atac[cd4_tem_idx_100].write('atac_cd4_tem_100.h5ad')

# python data_preprocess.py -r rna_cd4_tem_100.h5ad -a atac_cd4_tem_100.h5ad -s pt_cd4_tem_100 --dt test -n cd4_tem_100 --config rna2atac_config_test_cd4_tem.yaml
# enc_max_len: 2971

cd4_tem_data = torch.load("./pt_cd4_tem_100/cd4_tem_100_0.pt")
dataset = PreDataset(cd4_tem_data)
dataloader_kwargs = {'batch_size': 1, 'shuffle': False}
loader = torch.utils.data.DataLoader(dataset, **dataloader_kwargs)

rna_cd4_tem = sc.read_h5ad('rna_cd4_tem_100.h5ad')
atac_cd4_tem = sc.read_h5ad('atac_cd4_tem_100.h5ad')

out = rna_cd4_tem.copy()
out.X = np.zeros(out.shape)
i = 0
for inputs in tqdm(loader, ncols=80, desc='output attention matrix'):
    rna_sequence, rna_value, atac_sequence, _, enc_pad_mask = [each.to(device) for each in inputs]
    attn_tmp = model.generate_attn_weight(rna_sequence, atac_sequence, rna_value, enc_mask=enc_pad_mask, which='decoder')[0]
    out.X[i, (rna_sequence[rna_sequence!=0]-1).cpu()] = np.nansum(attn_tmp, axis=0)
    torch.cuda.empty_cache()
    i += 1

out.write('attn_cd4_tem_100_gene.h5ad')

attn = sc.read_h5ad('attn_cd4_tem_100_gene.h5ad')

df = pd.DataFrame({'attn':attn.X.sum(axis=0)})
df.index = out.var.index.values
df = df[df['attn']!=0]
df['attn'] = np.log10(df['attn'])
df['attn'] = (df['attn']-df['attn'].min())/(df['attn'].max()-df['attn'].min())

gene_idx = []
for g in df.index:
    if g in cr_lst:
        gene_idx.append('CR')
    elif g in tf_lst:
        gene_idx.append('TF')
    else:
        gene_idx.append('Others')

df['gene_idx'] = gene_idx
df.to_csv('attn/gene_attn_cd4_tem_norm.txt', sep='\t')

## for k562_gm12878
TFs_chipseq = pd.read_table('/fs/home/jiluzhang/Nature_methods/TF_binding/tf_chipseq/k562_gm12878/TFs_10000.txt', header=None)
tfs_lst = np.intersect1d(TFs_chipseq[0].values, rna_cd4_tem.var.index)
out_2 = pd.DataFrame(np.zeros([atac_cd4_tem.n_vars, len(tfs_lst)]))
out_2.columns = [np.argwhere(rna_cd4_tem.var.index==tf).item() for tf in tfs_lst]
i = 0
for inputs in tqdm(loader, ncols=80, desc='output attention matrix'):
    rna_sequence, rna_value, atac_sequence, _, enc_pad_mask = [each.to(device) for each in inputs]
    attn_m = pd.DataFrame(model.generate_attn_weight(rna_sequence, atac_sequence, rna_value, enc_mask=enc_pad_mask, which='decoder')[0])
    attn_m.columns = (rna_sequence[rna_sequence!=0]-1).cpu().numpy()
    ovp_idx = np.intersect1d(out_2.columns, attn_m.columns)
    out_2.loc[:, ovp_idx] += attn_m.loc[:, ovp_idx]
    torch.cuda.empty_cache()
    i += 1

out_2.index = atac_cd4_tem.var.index.values
out_2.columns = tfs_lst
out_2.to_csv('attn_cd4_tem_100_peak_gene.txt', sep='\t')

## plot box
df = pd.read_table('gene_attn_cd4_tem_norm.txt', index_col=0)
df['gene_idx'] = pd.Categorical(df['gene_idx'], categories=['CR', 'TF', 'Others'])

p = ggplot(df, aes(x='gene_idx', y='attn', fill='gene_idx')) + geom_boxplot(width=0.5, show_legend=False, outlier_shape='') + xlab('') +\
                                                               coord_cartesian(ylim=(0.2, 1.0)) +\
                                                               scale_y_continuous(breaks=np.arange(0.2, 1.0+0.1, 0.2)) + theme_bw()
p.save(filename='gene_attn_cd4_tem_norm.pdf', dpi=600, height=4, width=4)

df[df['gene_idx']=='CR']['attn'].median()          # 0.5895524490111493
df[df['gene_idx']=='TF']['attn'].median()          # 0.558375661095446
df[df['gene_idx']=='Others']['attn'].median()      # 0.5489987669938118

stats.ttest_ind(df[df['gene_idx']=='CR']['attn'], df[df['gene_idx']=='TF']['attn'])      # pvalue=0.17387120706218734
stats.ttest_ind(df[df['gene_idx']=='CR']['attn'], df[df['gene_idx']=='Others']['attn'])  # pvalue=0.040502892901245674
stats.ttest_ind(df[df['gene_idx']=='TF']['attn'], df[df['gene_idx']=='Others']['attn'])  # pvalue=0.04864860945738346

################ B memory ################
random.seed(827)
b_mem_idx = list(np.argwhere(rna.obs['cell_anno']=='B memory').flatten())
b_mem_idx_100 = random.sample(list(b_mem_idx), 100)
rna[b_mem_idx_100].write('rna_b_mem_100.h5ad')
np.count_nonzero(rna[b_mem_idx_100].X.toarray(), axis=1).max()  # 3323

atac[b_mem_idx_100].write('atac_b_mem_100.h5ad')

# python data_preprocess.py -r rna_b_mem_100.h5ad -a atac_b_mem_100.h5ad -s pt_b_mem_100 --dt test -n b_mem_100 --config rna2atac_config_test_b_mem.yaml
# enc_max_len: 3323

b_mem_data = torch.load("./pt_b_mem_100/b_mem_100_0.pt")
dataset = PreDataset(b_mem_data)
dataloader_kwargs = {'batch_size': 1, 'shuffle': False}
loader = torch.utils.data.DataLoader(dataset, **dataloader_kwargs)

rna_b_mem = sc.read_h5ad('rna_b_mem_100.h5ad')
atac_b_mem = sc.read_h5ad('atac_b_mem_100.h5ad')

out = rna_b_mem.copy()
out.X = np.zeros(out.shape)
i = 0
for inputs in tqdm(loader, ncols=80, desc='output attention matrix'):
    rna_sequence, rna_value, atac_sequence, _, enc_pad_mask = [each.to(device) for each in inputs]
    attn_tmp = model.generate_attn_weight(rna_sequence, atac_sequence, rna_value, enc_mask=enc_pad_mask, which='decoder')[0]
    out.X[i, (rna_sequence[rna_sequence!=0]-1).cpu()] = np.nansum(attn_tmp, axis=0)
    torch.cuda.empty_cache()
    i += 1

out.write('attn_b_mem_100_gene.h5ad')

attn = sc.read_h5ad('attn_b_mem_100_gene.h5ad')

df = pd.DataFrame({'attn':attn.X.sum(axis=0)})
df.index = out.var.index.values
df = df[df['attn']!=0]
df['attn'] = np.log10(df['attn'])
df['attn'] = (df['attn']-df['attn'].min())/(df['attn'].max()-df['attn'].min())

gene_idx = []
for g in df.index:
    if g in cr_lst:
        gene_idx.append('CR')
    elif g in tf_lst:
        gene_idx.append('TF')
    else:
        gene_idx.append('Others')

df['gene_idx'] = gene_idx
df.to_csv('attn/gene_attn_b_mem_norm.txt', sep='\t')

## for k562_gm12878
TFs_chipseq = pd.read_table('/fs/home/jiluzhang/Nature_methods/TF_binding/tf_chipseq/k562_gm12878/TFs_10000.txt', header=None)
tfs_lst = np.intersect1d(TFs_chipseq[0].values, rna_b_mem.var.index)
out_2 = pd.DataFrame(np.zeros([atac_b_mem.n_vars, len(tfs_lst)]))
out_2.columns = [np.argwhere(rna_b_mem.var.index==tf).item() for tf in tfs_lst]
i = 0
for inputs in tqdm(loader, ncols=80, desc='output attention matrix'):
    rna_sequence, rna_value, atac_sequence, _, enc_pad_mask = [each.to(device) for each in inputs]
    attn_m = pd.DataFrame(model.generate_attn_weight(rna_sequence, atac_sequence, rna_value, enc_mask=enc_pad_mask, which='decoder')[0])
    attn_m.columns = (rna_sequence[rna_sequence!=0]-1).cpu().numpy()
    ovp_idx = np.intersect1d(out_2.columns, attn_m.columns)
    out_2.loc[:, ovp_idx] += attn_m.loc[:, ovp_idx]
    torch.cuda.empty_cache()
    i += 1

out_2.index = atac_b_mem.var.index.values
out_2.columns = tfs_lst
out_2.to_csv('attn_b_mem_100_peak_gene.txt', sep='\t')

## plot box
df = pd.read_table('gene_attn_b_mem_norm.txt', index_col=0)
df['gene_idx'] = pd.Categorical(df['gene_idx'], categories=['CR', 'TF', 'Others'])

p = ggplot(df, aes(x='gene_idx', y='attn', fill='gene_idx')) + geom_boxplot(width=0.5, show_legend=False, outlier_shape='') + xlab('') +\
                                                               coord_cartesian(ylim=(0.2, 1.0)) +\
                                                               scale_y_continuous(breaks=np.arange(0.2, 1.0+0.1, 0.2)) + theme_bw()
p.save(filename='gene_attn_b_mem_norm.pdf', dpi=600, height=4, width=4)

df[df['gene_idx']=='CR']['attn'].median()          # 0.5719922304371735
df[df['gene_idx']=='TF']['attn'].median()          # 0.5499967421955819
df[df['gene_idx']=='Others']['attn'].median()      # 0.5506269910101489

stats.ttest_ind(df[df['gene_idx']=='CR']['attn'], df[df['gene_idx']=='TF']['attn'])      # pvalue=0.20649172559177867
stats.ttest_ind(df[df['gene_idx']=='CR']['attn'], df[df['gene_idx']=='Others']['attn'])  # pvalue=0.07829921220679327
stats.ttest_ind(df[df['gene_idx']=='TF']['attn'], df[df['gene_idx']=='Others']['attn'])  # pvalue=0.13760342082995577

################ NK ################
random.seed(827)
nk_idx = list(np.argwhere(rna.obs['cell_anno']=='NK').flatten())
nk_idx_100 = random.sample(list(nk_idx), 100)
rna[nk_idx_100].write('rna_nk_100.h5ad')
np.count_nonzero(rna[nk_idx_100].X.toarray(), axis=1).max()  # 2890

atac[nk_idx_100].write('atac_nk_100.h5ad')

# python data_preprocess.py -r rna_nk_100.h5ad -a atac_nk_100.h5ad -s pt_nk_100 --dt test -n nk_100 --config rna2atac_config_test_nk.yaml
# enc_max_len: 2890

nk_data = torch.load("./pt_nk_100/nk_100_0.pt")
dataset = PreDataset(nk_data)
dataloader_kwargs = {'batch_size': 1, 'shuffle': False}
loader = torch.utils.data.DataLoader(dataset, **dataloader_kwargs)

rna_nk = sc.read_h5ad('rna_nk_100.h5ad')
atac_nk = sc.read_h5ad('atac_nk_100.h5ad')

out = rna_nk.copy()
out.X = np.zeros(out.shape)
i = 0
for inputs in tqdm(loader, ncols=80, desc='output attention matrix'):
    rna_sequence, rna_value, atac_sequence, _, enc_pad_mask = [each.to(device) for each in inputs]
    attn_tmp = model.generate_attn_weight(rna_sequence, atac_sequence, rna_value, enc_mask=enc_pad_mask, which='decoder')[0]
    out.X[i, (rna_sequence[rna_sequence!=0]-1).cpu()] = np.nansum(attn_tmp, axis=0)
    torch.cuda.empty_cache()
    i += 1

out.write('attn_nk_100_gene.h5ad')

attn = sc.read_h5ad('attn_nk_100_gene.h5ad')

df = pd.DataFrame({'attn':attn.X.sum(axis=0)})
df.index = out.var.index.values
df = df[df['attn']!=0]
df['attn'] = np.log10(df['attn'])
df['attn'] = (df['attn']-df['attn'].min())/(df['attn'].max()-df['attn'].min())

gene_idx = []
for g in df.index:
    if g in cr_lst:
        gene_idx.append('CR')
    elif g in tf_lst:
        gene_idx.append('TF')
    else:
        gene_idx.append('Others')

df['gene_idx'] = gene_idx
df.to_csv('attn/gene_attn_nk_norm.txt', sep='\t')

## for k562_gm12878
TFs_chipseq = pd.read_table('/fs/home/jiluzhang/Nature_methods/TF_binding/tf_chipseq/k562_gm12878/TFs_10000.txt', header=None)
tfs_lst = np.intersect1d(TFs_chipseq[0].values, rna_nk.var.index)
out_2 = pd.DataFrame(np.zeros([atac_nk.n_vars, len(tfs_lst)]))
out_2.columns = [np.argwhere(rna_nk.var.index==tf).item() for tf in tfs_lst]
i = 0
for inputs in tqdm(loader, ncols=80, desc='output attention matrix'):
    rna_sequence, rna_value, atac_sequence, _, enc_pad_mask = [each.to(device) for each in inputs]
    attn_m = pd.DataFrame(model.generate_attn_weight(rna_sequence, atac_sequence, rna_value, enc_mask=enc_pad_mask, which='decoder')[0])
    attn_m.columns = (rna_sequence[rna_sequence!=0]-1).cpu().numpy()
    ovp_idx = np.intersect1d(out_2.columns, attn_m.columns)
    out_2.loc[:, ovp_idx] += attn_m.loc[:, ovp_idx]
    torch.cuda.empty_cache()
    i += 1

out_2.index = atac_nk.var.index.values
out_2.columns = tfs_lst
out_2.to_csv('attn_nk_100_peak_gene.txt', sep='\t')

## plot box
df = pd.read_table('gene_attn_nk_norm.txt', index_col=0)
df['gene_idx'] = pd.Categorical(df['gene_idx'], categories=['CR', 'TF', 'Others'])

p = ggplot(df, aes(x='gene_idx', y='attn', fill='gene_idx')) + geom_boxplot(width=0.5, show_legend=False, outlier_shape='') + xlab('') +\
                                                               coord_cartesian(ylim=(0.2, 1.0)) +\
                                                               scale_y_continuous(breaks=np.arange(0.2, 1.0+0.1, 0.2)) + theme_bw()
p.save(filename='gene_attn_nk_norm.pdf', dpi=600, height=4, width=4)

df[df['gene_idx']=='CR']['attn'].median()          # 0.5613611918213242
df[df['gene_idx']=='TF']['attn'].median()          # 0.5559758379524299
df[df['gene_idx']=='Others']['attn'].median()      # 0.5580211336995884

stats.ttest_ind(df[df['gene_idx']=='CR']['attn'], df[df['gene_idx']=='TF']['attn'])      # pvalue=0.2182299997505756
stats.ttest_ind(df[df['gene_idx']=='CR']['attn'], df[df['gene_idx']=='Others']['attn'])  # pvalue=0.10835547351173073
stats.ttest_ind(df[df['gene_idx']=='TF']['attn'], df[df['gene_idx']=='Others']['attn'])  # pvalue=0.33427124214615944

################ CD16 Mono ################
random.seed(827)
cd16_mono_idx = list(np.argwhere(rna.obs['cell_anno']=='CD16 Mono').flatten())
cd16_mono_idx_100 = random.sample(list(cd16_mono_idx), 100)
rna[cd16_mono_idx_100].write('rna_cd16_mono_100.h5ad')
np.count_nonzero(rna[cd16_mono_idx_100].X.toarray(), axis=1).max()  # 3438

atac[cd16_mono_idx_100].write('atac_cd16_mono_100.h5ad')

# python data_preprocess.py -r rna_cd16_mono_100.h5ad -a atac_cd16_mono_100.h5ad -s pt_cd16_mono_100 --dt test -n cd16_mono_100 --config rna2atac_config_test_cd16_mono.yaml
# enc_max_len: 3438

cd16_mono_data = torch.load("./pt_cd16_mono_100/cd16_mono_100_0.pt")
dataset = PreDataset(cd16_mono_data)
dataloader_kwargs = {'batch_size': 1, 'shuffle': False}
loader = torch.utils.data.DataLoader(dataset, **dataloader_kwargs)

rna_cd16_mono = sc.read_h5ad('rna_cd16_mono_100.h5ad')
atac_cd16_mono = sc.read_h5ad('atac_cd16_mono_100.h5ad')

out = rna_cd16_mono.copy()
out.X = np.zeros(out.shape)
i = 0
for inputs in tqdm(loader, ncols=80, desc='output attention matrix'):
    rna_sequence, rna_value, atac_sequence, _, enc_pad_mask = [each.to(device) for each in inputs]
    attn_tmp = model.generate_attn_weight(rna_sequence, atac_sequence, rna_value, enc_mask=enc_pad_mask, which='decoder')[0]
    out.X[i, (rna_sequence[rna_sequence!=0]-1).cpu()] = np.nansum(attn_tmp, axis=0)
    torch.cuda.empty_cache()
    i += 1

out.write('attn_cd16_mono_100_gene.h5ad')

attn = sc.read_h5ad('attn_cd16_mono_100_gene.h5ad')

df = pd.DataFrame({'attn':attn.X.sum(axis=0)})
df.index = out.var.index.values
df = df[df['attn']!=0]
df['attn'] = np.log10(df['attn'])
df['attn'] = (df['attn']-df['attn'].min())/(df['attn'].max()-df['attn'].min())

gene_idx = []
for g in df.index:
    if g in cr_lst:
        gene_idx.append('CR')
    elif g in tf_lst:
        gene_idx.append('TF')
    else:
        gene_idx.append('Others')

df['gene_idx'] = gene_idx
df.to_csv('attn/gene_attn_cd16_mono_norm.txt', sep='\t')

## for k562_gm12878
TFs_chipseq = pd.read_table('/fs/home/jiluzhang/Nature_methods/TF_binding/tf_chipseq/k562_gm12878/TFs_10000.txt', header=None)
tfs_lst = np.intersect1d(TFs_chipseq[0].values, rna_cd16_mono.var.index)
out_2 = pd.DataFrame(np.zeros([atac_cd16_mono.n_vars, len(tfs_lst)]))
out_2.columns = [np.argwhere(rna_cd16_mono.var.index==tf).item() for tf in tfs_lst]
i = 0
for inputs in tqdm(loader, ncols=80, desc='output attention matrix'):
    rna_sequence, rna_value, atac_sequence, _, enc_pad_mask = [each.to(device) for each in inputs]
    attn_m = pd.DataFrame(model.generate_attn_weight(rna_sequence, atac_sequence, rna_value, enc_mask=enc_pad_mask, which='decoder')[0])
    attn_m.columns = (rna_sequence[rna_sequence!=0]-1).cpu().numpy()
    ovp_idx = np.intersect1d(out_2.columns, attn_m.columns)
    out_2.loc[:, ovp_idx] += attn_m.loc[:, ovp_idx]
    torch.cuda.empty_cache()
    i += 1

out_2.index = atac_cd16_mono.var.index.values
out_2.columns = tfs_lst
out_2.to_csv('attn_cd16_mono_100_peak_gene.txt', sep='\t')

## plot box
df = pd.read_table('gene_attn_cd16_mono_norm.txt', index_col=0)
df['gene_idx'] = pd.Categorical(df['gene_idx'], categories=['CR', 'TF', 'Others'])

p = ggplot(df, aes(x='gene_idx', y='attn', fill='gene_idx')) + geom_boxplot(width=0.5, show_legend=False, outlier_shape='') + xlab('') +\
                                                               coord_cartesian(ylim=(0.2, 1.0)) +\
                                                               scale_y_continuous(breaks=np.arange(0.2, 1.0+0.1, 0.2)) + theme_bw()
p.save(filename='gene_attn_cd16_mono_norm.pdf', dpi=600, height=4, width=4)

df[df['gene_idx']=='CR']['attn'].median()          # 0.6355727089398019
df[df['gene_idx']=='TF']['attn'].median()          # 0.6206220154682527
df[df['gene_idx']=='Others']['attn'].median()      # 0.6121352267448128

stats.ttest_ind(df[df['gene_idx']=='CR']['attn'], df[df['gene_idx']=='TF']['attn'])      # pvalue=0.2069185377881961
stats.ttest_ind(df[df['gene_idx']=='CR']['attn'], df[df['gene_idx']=='Others']['attn'])  # pvalue=0.1215945143690751
stats.ttest_ind(df[df['gene_idx']=='TF']['attn'], df[df['gene_idx']=='Others']['attn'])  # pvalue=0.502982714879039

################ B naive ################
random.seed(827)
b_naive_idx = list(np.argwhere(rna.obs['cell_anno']=='B naive').flatten())
b_naive_idx_100 = random.sample(list(b_naive_idx), 100)
rna[b_naive_idx_100].write('rna_b_naive_100.h5ad')
np.count_nonzero(rna[b_naive_idx_100].X.toarray(), axis=1).max()  # 3069

atac[b_naive_idx_100].write('atac_b_naive_100.h5ad')

# python data_preprocess.py -r rna_b_naive_100.h5ad -a atac_b_naive_100.h5ad -s pt_b_naive_100 --dt test -n b_naive_100 --config rna2atac_config_test_b_naive.yaml
# enc_max_len: 3069

b_naive_data = torch.load("./pt_b_naive_100/b_naive_100_0.pt")
dataset = PreDataset(b_naive_data)
dataloader_kwargs = {'batch_size': 1, 'shuffle': False}
loader = torch.utils.data.DataLoader(dataset, **dataloader_kwargs)

rna_b_naive = sc.read_h5ad('rna_b_naive_100.h5ad')
atac_b_naive = sc.read_h5ad('atac_b_naive_100.h5ad')

out = rna_b_naive.copy()
out.X = np.zeros(out.shape)
i = 0
for inputs in tqdm(loader, ncols=80, desc='output attention matrix'):
    rna_sequence, rna_value, atac_sequence, _, enc_pad_mask = [each.to(device) for each in inputs]
    attn_tmp = model.generate_attn_weight(rna_sequence, atac_sequence, rna_value, enc_mask=enc_pad_mask, which='decoder')[0]
    out.X[i, (rna_sequence[rna_sequence!=0]-1).cpu()] = np.nansum(attn_tmp, axis=0)
    torch.cuda.empty_cache()
    i += 1

out.write('attn_b_naive_100_gene.h5ad')

attn = sc.read_h5ad('attn_b_naive_100_gene.h5ad')

df = pd.DataFrame({'attn':attn.X.sum(axis=0)})
df.index = out.var.index.values
df = df[df['attn']!=0]
df['attn'] = np.log10(df['attn'])
df['attn'] = (df['attn']-df['attn'].min())/(df['attn'].max()-df['attn'].min())

gene_idx = []
for g in df.index:
    if g in cr_lst:
        gene_idx.append('CR')
    elif g in tf_lst:
        gene_idx.append('TF')
    else:
        gene_idx.append('Others')

df['gene_idx'] = gene_idx
df.to_csv('attn/gene_attn_b_naive_norm.txt', sep='\t')

## for k562_gm12878
TFs_chipseq = pd.read_table('/fs/home/jiluzhang/Nature_methods/TF_binding/tf_chipseq/k562_gm12878/TFs_10000.txt', header=None)
tfs_lst = np.intersect1d(TFs_chipseq[0].values, rna_b_naive.var.index)
out_2 = pd.DataFrame(np.zeros([atac_b_naive.n_vars, len(tfs_lst)]))
out_2.columns = [np.argwhere(rna_b_naive.var.index==tf).item() for tf in tfs_lst]
i = 0
for inputs in tqdm(loader, ncols=80, desc='output attention matrix'):
    rna_sequence, rna_value, atac_sequence, _, enc_pad_mask = [each.to(device) for each in inputs]
    attn_m = pd.DataFrame(model.generate_attn_weight(rna_sequence, atac_sequence, rna_value, enc_mask=enc_pad_mask, which='decoder')[0])
    attn_m.columns = (rna_sequence[rna_sequence!=0]-1).cpu().numpy()
    ovp_idx = np.intersect1d(out_2.columns, attn_m.columns)
    out_2.loc[:, ovp_idx] += attn_m.loc[:, ovp_idx]
    torch.cuda.empty_cache()
    i += 1

out_2.index = atac_b_naive.var.index.values
out_2.columns = tfs_lst
out_2.to_csv('attn_b_naive_100_peak_gene.txt', sep='\t')

## plot box
df = pd.read_table('gene_attn_b_naive_norm.txt', index_col=0)
df['gene_idx'] = pd.Categorical(df['gene_idx'], categories=['CR', 'TF', 'Others'])

p = ggplot(df, aes(x='gene_idx', y='attn', fill='gene_idx')) + geom_boxplot(width=0.5, show_legend=False, outlier_shape='') + xlab('') +\
                                                               coord_cartesian(ylim=(0.2, 1.0)) +\
                                                               scale_y_continuous(breaks=np.arange(0.2, 1.0+0.1, 0.2)) + theme_bw()
p.save(filename='gene_attn_b_naive_norm.pdf', dpi=600, height=4, width=4)

df[df['gene_idx']=='CR']['attn'].median()          # 0.5875201298734866
df[df['gene_idx']=='TF']['attn'].median()          # 0.5574929181110417
df[df['gene_idx']=='Others']['attn'].median()      # 0.5462597326619831

stats.ttest_ind(df[df['gene_idx']=='CR']['attn'], df[df['gene_idx']=='TF']['attn'])      # pvalue=0.13154291484881145
stats.ttest_ind(df[df['gene_idx']=='CR']['attn'], df[df['gene_idx']=='Others']['attn'])  # pvalue=0.04040017738345766
stats.ttest_ind(df[df['gene_idx']=='TF']['attn'], df[df['gene_idx']=='Others']['attn'])  # pvalue=0.19584252449117587




# ################ CD4 Intermediate ################
# random.seed(1)
# cd4_inter_idx = list(np.argwhere(rna.obs['cell_anno']=='CD4 intermediate').flatten())
# cd4_inter_idx_100 = random.sample(list(cd4_inter_idx), 100)
# rna[cd4_inter_idx_100].write('rna_cd4_inter_100.h5ad')
# np.count_nonzero(rna[cd4_inter_idx_100].X.toarray(), axis=1).max()  # 2617

# atac[cd4_inter_idx_100].write('atac_cd4_inter_100.h5ad')

# # python data_preprocess.py -r rna_cd4_inter_100.h5ad -a atac_cd4_inter_100.h5ad -s pt_cd4_inter_100 --dt test -n cd4_inter_100 --config rna2atac_config_test_cd4_inter.yaml
# # enc_max_len: 2617

# cd4_inter_data = torch.load("./pt_cd4_inter_100/cd4_inter_100_0.pt")
# dataset = PreDataset(cd4_inter_data)
# dataloader_kwargs = {'batch_size': 1, 'shuffle': False}
# loader = torch.utils.data.DataLoader(dataset, **dataloader_kwargs)

# rna_cd4_inter = sc.read_h5ad('rna_cd4_inter_100.h5ad')
# atac_cd4_inter = sc.read_h5ad('atac_cd4_inter_100.h5ad')

# tf_idx = np.argwhere(rna_cd4_inter.var.index.isin(tf_lst)).flatten()

# out = atac_cd4_inter.copy()
# out.X = np.zeros(out.shape)
# i = 0
# for inputs in tqdm(loader, ncols=80, desc='output attention matrix'):
#     rna_sequence, rna_value, atac_sequence, _, enc_pad_mask = [each.to(device) for each in inputs]
#     attn_tmp = model.generate_attn_weight(rna_sequence, atac_sequence, rna_value, enc_mask=enc_pad_mask, which='decoder')[0]
#     out.X[i, :] = attn_tmp[:, [a.item() in tf_idx for a in (rna_sequence[rna_sequence!=0]-1)]].sum(axis=1) 
#     torch.cuda.empty_cache()
#     i += 1

# out.write('attn_cd4_inter_100_peak.h5ad')

# attn = sc.read_h5ad('attn_cd4_inter_100_peak.h5ad')

# motif_info.index = attn.var.index.values

# df = pd.DataFrame({'attn':attn.X.sum(axis=0), 'motif':motif_info}).dropna()
# df['attn'] = (df['attn']-df['attn'].min())/(df['attn'].max()-df['attn'].min())
# df.to_csv('attn/motif_attn_cd4_inter_norm.txt', sep='\t')

# ## plot box
# df = pd.read_table('motif_attn_cd4_inter_norm.txt', index_col=0)
# df['motif_bin'] = 'with'
# df.loc[df['motif']==0, 'motif_bin'] = 'without'
# df['motif_bin'] = pd.Categorical(df['motif_bin'], categories=['without', 'with'])

# p = ggplot(df, aes(x='motif_bin', y='attn', fill='motif_bin')) + geom_boxplot(width=0.5, show_legend=False, outlier_shape='') + xlab('') +\
#                                                                  coord_cartesian(ylim=(0, 0.4)) +\
#                                                                  scale_y_continuous(breaks=np.arange(0, 0.4+0.1, 0.1)) + theme_bw()
# p.save(filename='motif_attn_cd4_inter_norm.pdf', dpi=600, height=4, width=4)

# df[df['motif_bin']=='with']['attn'].median()      # 0.0789243868698908
# df[df['motif_bin']=='without']['attn'].median()   # 0.06244332828186935

# stats.ttest_ind(df[df['motif_bin']=='with']['attn'], df[df['motif_bin']=='without']['attn'])  # pvalue=0.0006961820189437559


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

import pandas as pd
import scanpy as sc
import numpy as np
from tqdm import tqdm

TFs = pd.read_table('/fs/home/jiluzhang/Nature_methods/TF_binding/tf_chipseq/k562_gm12878/TFs_10000.txt', header=None)
ct_lst = ['cd14_mono', 'cd8_naive', 'cd4_tcm', 'cd4_naive', 'cd4_inter',
          'cd8_tem', 'cd4_tem', 'b_mem', 'nk', 'cd16_mono', 'b_naive']
atac_ct_100 = sc.read_h5ad('../atac_cd14_mono_100.h5ad')

def cal_mat(cl='K562'):
    mat = pd.DataFrame(np.zeros([TFs.shape[0], len(ct_lst)]))
    mat.index = TFs[0].values
    mat.columns = ct_lst
    
    for ct in ct_lst:
        gene_attn = pd.read_table('../attn_'+ct+'_100_peak_gene.txt', index_col=0)
        
        for tf in tqdm(TFs[0], desc=cl+' '+ct+' cal ratio', ncols=80):
            if tf in gene_attn.columns:
                gene_attn_tf = pd.DataFrame(gene_attn[tf])
                # gene_attn_tf.index = atac_ct_100.var.index.values
                gene_attn_tf.columns = ['attn']
                # gene_attn_tf['attn'] = gene_attn_tf['attn']/20
                
                tf_peaks = pd.read_table('/fs/home/jiluzhang/Nature_methods/TF_binding/tf_chipseq/k562_gm12878/'+cl+'_'+tf+'_ccre_peaks.bed', header=None)
                tf_peaks.index = tf_peaks[0]+':'+tf_peaks[1].apply(str)+'-'+tf_peaks[2].apply(str)
                tf_peaks.columns = ['chrom', 'start', 'end', 'peak_or_not']
    
                df = gene_attn_tf.copy()
                df['peak_or_not'] = tf_peaks.loc[gene_attn_tf.index, 'peak_or_not']
                df['peak_or_not'] = df['peak_or_not'].apply(str)
                # df = df.replace([np.inf, -np.inf], np.nan).dropna()
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
import seaborn as sns

plt.rcParams['pdf.fonttype'] = 42

## gm12878
# gm12878_df = pd.read_table('gm12878_tf_10000_ratio.txt', index_col=0)
# gm12878_df = pd.melt(gm12878_df)
# gm12878_df.columns = ['ct', 'val']
# gm12878_df = gm12878_df[gm12878_df['val']!=0]
# p = ggplot(gm12878_df, aes(x='ct', y='val', fill='ct')) + geom_boxplot(width=0.5, outlier_shape='', show_legend=False) +\
#                                                           coord_cartesian(ylim=(0, 2)) + scale_y_continuous(breaks=np.arange(0, 2+0.1, 0.5)) + theme_bw()
# p.save(filename='gm12878_tf_10000_ratio.pdf', dpi=600, height=4, width=10)

gm12878_df = pd.read_table('gm12878_tf_10000_ratio.txt', index_col=0)
gm12878_df = gm12878_df.loc[:, ['cd14_mono', 'cd16_mono',
                                'cd4_naive', 'cd4_inter', 'cd4_tem', 'cd4_tcm',
                                'cd8_naive', 'cd8_tem',
                                'b_naive', 'b_mem',
                                'nk']]
gm12878_df['avg'] = gm12878_df.mean(axis=1)
gm12878_df = gm12878_df[gm12878_df['avg']!=0]
gm12878_df.sort_values('avg', ascending=False, inplace=True)

plt.figure(figsize=(6, 10))
sns.heatmap(gm12878_df.iloc[:, :-1], cmap='bwr', vmin=0, vmax=2, center=1, xticklabels=True, yticklabels=True)  # cbar=False, vmin=0, vmax=10, linewidths=0.1, linecolor='grey'
plt.savefig('gm12878_df_10000_ratio_heatmap.pdf')
plt.close()


## k562
# k562_df = pd.read_table('k562_tf_10000_ratio.txt', index_col=0)
# k562_df = pd.melt(k562_df)
# k562_df.columns = ['ct', 'val']
# k562_df = k562_df[k562_df['val']!=0]
# p = ggplot(k562_df, aes(x='ct', y='val', fill='ct')) + geom_boxplot(width=0.5, outlier_shape='', show_legend=False) +\
#                                                        coord_cartesian(ylim=(0, 2)) + scale_y_continuous(breaks=np.arange(0, 2+0.1, 0.5)) + theme_bw()
# p.save(filename='k562_tf_10000_ratio.pdf', dpi=600, height=4, width=10)

k562_df = pd.read_table('k562_tf_10000_ratio.txt', index_col=0)
k562_df = k562_df.loc[:, ['cd14_mono', 'cd16_mono',
                          'cd4_naive', 'cd4_inter', 'cd4_tem', 'cd4_tcm',
                          'cd8_naive', 'cd8_tem',
                          'b_naive', 'b_mem',
                          'nk']]
k562_df['avg'] = k562_df.mean(axis=1)
k562_df = k562_df[k562_df['avg']!=0]
k562_df = k562_df.loc[gm12878_df.index, :]

plt.figure(figsize=(6, 10))
sns.heatmap(k562_df.iloc[:, :-1], cmap='bwr', vmin=0, vmax=2, center=1, xticklabels=True, yticklabels=True)  # cbar=False, vmin=0, vmax=10, linewidths=0.1, linecolor='grey'
plt.savefig('k562_df_10000_ratio_heatmap.pdf')
plt.close()


########## ETS1 B memory GM12878 #############
# install deeptools
# conda create -n deeptools python=3.8
# conda activate deeptools
# conda install -c conda-forge -c bioconda deeptools  # not work
# pip install deeptools  # works

## ets1 gm12878
# wget -c https://www.encodeproject.org/files/ENCFF686FAA/@@download/ENCFF686FAA.bigWig
# mv ENCFF686FAA.bigWig gm12878_ets1_signal.bigwig

## ets1 k562
# wget -c https://www.encodeproject.org/files/ENCFF151QKU/@@download/ENCFF151QKU.bigWig
# mv ENCFF151QKU.bigWig k562_ets1_signal.bigwig

## ctcf gm12878
# wget -c https://www.encodeproject.org/files/ENCFF637RGD/@@download/ENCFF637RGD.bigWig
# mv ENCFF637RGD.bigWig gm12878_ctcf_signal.bigwig

## ctcf k562
# wget -c https://www.encodeproject.org/files/ENCFF168IFW/@@download/ENCFF168IFW.bigWig
# mv ENCFF168IFW.bigWig k562_ctcf_signal.bigwig

import pandas as pd
import numpy as np

## b memory ets1
df = pd.read_table('../attn_b_mem_100_peak_gene.txt', index_col=0)['ETS1'].dropna()

df_top = pd.DataFrame(df.sort_values(ascending=False)[:10000].index)
df_top['chrom'] = df_top[0].apply(lambda x: x.split(':')[0])
df_top['start'] = df_top[0].apply(lambda x: x.split(':')[1].split('-')[0])
df_top['end'] = df_top[0].apply(lambda x: x.split(':')[1].split('-')[1])
df_top.iloc[:, 1:].to_csv('attn_top_10000_b_mem_ets1.bed', index=False, header=False, sep='\t')

df_btm = pd.DataFrame(df.sort_values(ascending=True)[:10000].index)
df_btm['chrom'] = df_btm[0].apply(lambda x: x.split(':')[0])
df_btm['start'] = df_btm[0].apply(lambda x: x.split(':')[1].split('-')[0])
df_btm['end'] = df_btm[0].apply(lambda x: x.split(':')[1].split('-')[1])
df_btm.iloc[:, 1:].to_csv('attn_btm_10000_b_mem_ets1.bed', index=False, header=False, sep='\t')

## b memory ctcf
df = pd.read_table('../attn_b_mem_100_peak_gene.txt', index_col=0)['CTCF'].dropna()

df_top = pd.DataFrame(df.sort_values(ascending=False)[:10000].index)
df_top['chrom'] = df_top[0].apply(lambda x: x.split(':')[0])
df_top['start'] = df_top[0].apply(lambda x: x.split(':')[1].split('-')[0])
df_top['end'] = df_top[0].apply(lambda x: x.split(':')[1].split('-')[1])
df_top.iloc[:, 1:].to_csv('attn_top_10000_b_mem_ctcf.bed', index=False, header=False, sep='\t')

df_btm = pd.DataFrame(df.sort_values(ascending=True)[:10000].index)
df_btm['chrom'] = df_btm[0].apply(lambda x: x.split(':')[0])
df_btm['start'] = df_btm[0].apply(lambda x: x.split(':')[1].split('-')[0])
df_btm['end'] = df_btm[0].apply(lambda x: x.split(':')[1].split('-')[1])
df_btm.iloc[:, 1:].to_csv('attn_btm_10000_b_mem_ctcf.bed', index=False, header=False, sep='\t')

## nk ets1
df = pd.read_table('../attn_nk_100_peak_gene.txt', index_col=0)['ETS1'].dropna()

df_top = pd.DataFrame(df.sort_values(ascending=False)[:10000].index)
df_top['chrom'] = df_top[0].apply(lambda x: x.split(':')[0])
df_top['start'] = df_top[0].apply(lambda x: x.split(':')[1].split('-')[0])
df_top['end'] = df_top[0].apply(lambda x: x.split(':')[1].split('-')[1])
df_top.iloc[:, 1:].to_csv('attn_top_10000_nk_ets1.bed', index=False, header=False, sep='\t')

df_btm = pd.DataFrame(df.sort_values(ascending=True)[:10000].index)
df_btm['chrom'] = df_btm[0].apply(lambda x: x.split(':')[0])
df_btm['start'] = df_btm[0].apply(lambda x: x.split(':')[1].split('-')[0])
df_btm['end'] = df_btm[0].apply(lambda x: x.split(':')[1].split('-')[1])
df_btm.iloc[:, 1:].to_csv('attn_btm_10000_nk_ets1.bed', index=False, header=False, sep='\t')

## nk ctcf
df = pd.read_table('../attn_nk_100_peak_gene.txt', index_col=0)['CTCF'].dropna()

df_top = pd.DataFrame(df.sort_values(ascending=False)[:10000].index)
df_top['chrom'] = df_top[0].apply(lambda x: x.split(':')[0])
df_top['start'] = df_top[0].apply(lambda x: x.split(':')[1].split('-')[0])
df_top['end'] = df_top[0].apply(lambda x: x.split(':')[1].split('-')[1])
df_top.iloc[:, 1:].to_csv('attn_top_10000_nk_ctcf.bed', index=False, header=False, sep='\t')

df_btm = pd.DataFrame(df.sort_values(ascending=True)[:10000].index)
df_btm['chrom'] = df_btm[0].apply(lambda x: x.split(':')[0])
df_btm['start'] = df_btm[0].apply(lambda x: x.split(':')[1].split('-')[0])
df_btm['end'] = df_btm[0].apply(lambda x: x.split(':')[1].split('-')[1])
df_btm.iloc[:, 1:].to_csv('attn_btm_10000_nk_ctcf.bed', index=False, header=False, sep='\t')

computeMatrix reference-point --referencePoint center -p 20 -S gm12878_ets1_signal.bigwig \
                              -R attn_top_10000_b_mem_ets1.bed attn_btm_10000_b_mem_ets1.bed \
                              -o gm12878_b_mem_attn_top_btm_ets1_signal.gz -a 5000 -b 5000 -bs 100
plotProfile -m gm12878_b_mem_attn_top_btm_ets1_signal.gz --yMin 0 --yMax 4 -out gm12878_b_mem_attn_top_btm_ets1_signal.pdf

computeMatrix reference-point --referencePoint center -p 20 -S gm12878_ctcf_signal.bigwig \
                              -R attn_top_10000_b_mem_ctcf.bed attn_btm_10000_b_mem_ctcf.bed \
                              -o gm12878_b_mem_attn_top_btm_ctcf_signal.gz -a 5000 -b 5000 -bs 100
plotProfile -m gm12878_b_mem_attn_top_btm_ctcf_signal.gz --yMin 0 --yMax 15 -out gm12878_b_mem_attn_top_btm_ctcf_signal.pdf

computeMatrix reference-point --referencePoint center -p 20 -S k562_ets1_signal.bigwig \
                              -R attn_top_10000_nk_ets1.bed attn_btm_10000_nk_ets1.bed \
                              -o k562_nk_attn_top_btm_ets1_signal.gz -a 5000 -b 5000 -bs 100
plotProfile -m k562_nk_attn_top_btm_ets1_signal.gz --yMin 0 --yMax 3 -out k562_nk_attn_top_btm_ets1_signal.pdf

computeMatrix reference-point --referencePoint center -p 20 -S k562_ctcf_signal.bigwig \
                              -R attn_top_10000_nk_ctcf.bed attn_btm_10000_nk_ctcf.bed \
                              -o k562_nk_attn_top_btm_ctcf_signal.gz -a 5000 -b 5000 -bs 100
plotProfile -m k562_nk_attn_top_btm_ctcf_signal.gz --yMin 0 --yMax 12 -out k562_nk_attn_top_btm_ctcf_signal.pdf


## Remap2022: https://remap.univ-amu.fr/
## B cell
wget -c https://remap.univ-amu.fr/storage/remap2022/hg38/MACS2/DATASET/ENCSR000AUV.CTCF.B-cell.bed.gz -O CTCF.bed.gz
wget -c https://remap.univ-amu.fr/storage/remap2022/hg38/MACS2/DATASET/GSE43350.BCL6.B-cell_GERMINAL_CENTER.bed.gz -O BCL6.bed.gz
wget -c https://remap.univ-amu.fr/storage/remap2022/hg38/MACS2/DATASET/GSE43350.BCOR.B-cell_GERMINAL_CENTER.bed.gz -O BCOR.bed.gz
wget -c https://remap.univ-amu.fr/storage/remap2022/hg38/MACS2/DATASET/GSE43350.NCOR2.B-cell_GERMINAL_CENTER.bed.gz -O NCOR2.bed.gz
wget -c https://remap.univ-amu.fr/storage/remap2022/hg38/MACS2/DATASET/GSE102460.BACH2.B-cell_IL2.bed.gz -O BACH2.bed.gz
wget -c https://remap.univ-amu.fr/storage/remap2022/hg38/MACS2/DATASET/GSE142493.IRF4.B-cell.bed.gz -O IRF4.bed.gz
wget -c https://remap.univ-amu.fr/storage/remap2022/hg38/MACS2/DATASET/GSE114803.FOXP1.B-cell_IgD-pos.bed.gz -O FOXP1.bed.gz
wget -c https://remap.univ-amu.fr/storage/remap2022/hg38/MACS2/DATASET/GSE123398.STAT3.B-cell.bed.gz -O STAT3.bed.gz























