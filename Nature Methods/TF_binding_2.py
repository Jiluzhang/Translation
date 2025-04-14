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

################ CD14 Mono ################
random.seed(0)
cd14_mono_idx = list(np.argwhere(rna.obs['cell_anno']=='CD14 Mono').flatten())
cd14_mono_idx_100 = random.sample(list(cd14_mono_idx), 100)
rna[cd14_mono_idx_100].write('rna_cd14_mono_100.h5ad')
np.count_nonzero(rna[cd14_mono_idx_100].X.toarray(), axis=1).max()  # 3174

atac[cd14_mono_idx_100].write('atac_cd14_mono_100.h5ad')

# python data_preprocess.py -r rna_cd14_mono_100.h5ad -a atac_cd14_mono_100.h5ad -s pt_cd14_mono_100 --dt test -n cd14_mono_100 --config rna2atac_config_test_cd14_mono.yaml
# enc_max_len: 3174

with open("rna2atac_config_test_cd14_mono.yaml", "r") as f:
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

cd14_mono_data = torch.load("./pt_cd14_mono_100/cd14_mono_100_0.pt")
dataset = PreDataset(cd14_mono_data)
dataloader_kwargs = {'batch_size': 1, 'shuffle': False}
loader = torch.utils.data.DataLoader(dataset, **dataloader_kwargs)

rna_cd14_mono = sc.read_h5ad('rna_cd14_mono_100.h5ad')
atac_cd14_mono = sc.read_h5ad('atac_cd14_mono_100.h5ad')

cr = ['SMARCA4', 'SMARCA2', 'ARID1A', 'ARID1B', 'SMARCB1',
      'CHD1', 'CHD2', 'CHD3', 'CHD4', 'CHD5', 'CHD6', 'CHD7', 'CHD8', 'CHD9',
      'BRD2', 'BRD3', 'BRD4', 'BRDT',
      'SMARCA5', 'SMARCA1', 'ACTL6A', 'ACTL6B',
      'SSRP1', 'SUPT16H',
      'EP400',
      'SMARCD1', 'SMARCD2', 'SMARCD3']   # 28
tf_lst = pd.read_table('attn/TF_jaspar.txt', header=None)
tf = list(tf_lst[0].values)   # 735
cr_tf = cr + tf   # 763
cr_tf_idx = np.argwhere(rna_cd14_mono.var.index.isin(cr_tf)).flatten()

out = atac_cd14_mono.copy()
out.X = np.zeros(out.shape)
i = 0
for inputs in tqdm(loader, ncols=80, desc='output attention matrix'):
    rna_sequence, rna_value, atac_sequence, _, enc_pad_mask = [each.to(device) for each in inputs]
    attn_tmp = model.generate_attn_weight(rna_sequence, atac_sequence, rna_value, enc_mask=enc_pad_mask, which='decoder')[0]
    out.X[i, :] = attn_tmp[:, [a.item() in cr_tf_idx for a in (rna_sequence[rna_sequence!=0]-1)]].sum(axis=1)  # only select CR & TF
    torch.cuda.empty_cache()
    i += 1

out.write('attn_cd14_mono_100_peak.h5ad')

attn = sc.read_h5ad('attn_cd14_mono_100_peak.h5ad')

motif_info_raw = pd.read_table('peaks_motifs_count.txt', header=0)
motif_info = motif_info_raw.copy()
motif_info[motif_info!=0] = 1
motif_info = motif_info.sum(axis=1)
motif_info.index = attn.var.index.values

df = pd.DataFrame({'attn':attn.X.sum(axis=0), 'motif':motif_info}).dropna()   # 229330
df['attn'] = (df['attn']-df['attn'].min())/(df['attn'].max()-df['attn'].min())
df.to_csv('attn/motif_attn_cd14_mono_norm.txt', sep='\t')

## plot box
from plotnine import *
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import stats

plt.rcParams['pdf.fonttype'] = 42

df = pd.read_table('motif_attn_cd14_mono_norm.txt', index_col=0)
df['motif_bin'] = 'with'
df.loc[df['motif']==0, 'motif_bin'] = 'without'
df['motif_bin'] = pd.Categorical(df['motif_bin'], categories=['without', 'with'])

p = ggplot(df, aes(x='motif_bin', y='attn', fill='motif_bin')) + geom_boxplot(width=0.5, show_legend=False, outlier_shape='') + xlab('') +\
                                                                 coord_cartesian(ylim=(0, 0.4)) +\
                                                                 scale_y_continuous(breaks=np.arange(0, 0.4+0.1, 0.1)) + theme_bw()
p.save(filename='motif_attn_cd14_mono_norm.pdf', dpi=600, height=4, width=4)

df[df['motif_bin']=='with']['attn'].median()      # 0.069279139521442
df[df['motif_bin']=='without']['attn'].median()   # 0.0669267925585442

stats.ttest_ind(df[df['motif_bin']=='with']['attn'], df[df['motif_bin']=='without']['attn'])  # pvalue=0.17099318898249696


################ CD8 Naive ################
random.seed(0)
cd8_naive_idx = list(np.argwhere(rna.obs['cell_anno']=='CD8 Naive').flatten())
cd8_naive_idx_100 = random.sample(list(cd8_naive_idx), 100)
rna[cd8_naive_idx_100].write('rna_cd8_naive_100.h5ad')
np.count_nonzero(rna[cd8_naive_idx_100].X.toarray(), axis=1).max()  # 2121

atac[cd8_naive_idx_100].write('atac_cd8_naive_100.h5ad')

# python data_preprocess.py -r rna_cd8_naive_100.h5ad -a atac_cd8_naive_100.h5ad -s pt_cd8_naive_100 --dt test -n cd8_naive_100 --config rna2atac_config_test_cd8_naive.yaml
# enc_max_len: 2121

with open("rna2atac_config_test_cd8_naive.yaml", "r") as f:
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

cd8_naive_data = torch.load("./pt_cd8_naive_100/cd8_naive_100_0.pt")
dataset = PreDataset(cd8_naive_data)
dataloader_kwargs = {'batch_size': 1, 'shuffle': False}
loader = torch.utils.data.DataLoader(dataset, **dataloader_kwargs)

rna_cd8_naive = sc.read_h5ad('rna_cd8_naive_100.h5ad')
atac_cd8_naive = sc.read_h5ad('atac_cd8_naive_100.h5ad')

cr = ['SMARCA4', 'SMARCA2', 'ARID1A', 'ARID1B', 'SMARCB1',
      'CHD1', 'CHD2', 'CHD3', 'CHD4', 'CHD5', 'CHD6', 'CHD7', 'CHD8', 'CHD9',
      'BRD2', 'BRD3', 'BRD4', 'BRDT',
      'SMARCA5', 'SMARCA1', 'ACTL6A', 'ACTL6B',
      'SSRP1', 'SUPT16H',
      'EP400',
      'SMARCD1', 'SMARCD2', 'SMARCD3']   # 28
tf_lst = pd.read_table('attn/TF_jaspar.txt', header=None)
tf = list(tf_lst[0].values)   # 735
cr_tf = cr + tf   # 763
cr_tf_idx = np.argwhere(rna_cd8_naive.var.index.isin(cr_tf)).flatten()

out = atac_cd8_naive.copy()
out.X = np.zeros(out.shape)
i = 0
for inputs in tqdm(loader, ncols=80, desc='output attention matrix'):
    rna_sequence, rna_value, atac_sequence, _, enc_pad_mask = [each.to(device) for each in inputs]
    attn_tmp = model.generate_attn_weight(rna_sequence, atac_sequence, rna_value, enc_mask=enc_pad_mask, which='decoder')[0]
    out.X[i, :] = attn_tmp[:, [a.item() in cr_tf_idx for a in (rna_sequence[rna_sequence!=0]-1)]].sum(axis=1)  # only select CR & TF
    torch.cuda.empty_cache()
    i += 1

out.write('attn_cd8_naive_100_peak.h5ad')

attn = sc.read_h5ad('attn_cd8_naive_100_peak.h5ad')

motif_info_raw = pd.read_table('peaks_motifs_count.txt', header=0)
motif_info = motif_info_raw.copy()
motif_info[motif_info!=0] = 1
motif_info = motif_info.sum(axis=1)
motif_info.index = attn.var.index.values

df = pd.DataFrame({'attn':attn.X.sum(axis=0), 'motif':motif_info}).dropna()   # 229525
df['attn'] = (df['attn']-df['attn'].min())/(df['attn'].max()-df['attn'].min())
df.to_csv('attn/motif_attn_cd8_naive_norm.txt', sep='\t')

## plot box
from plotnine import *
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import stats

plt.rcParams['pdf.fonttype'] = 42

df = pd.read_table('motif_attn_cd8_naive_norm.txt', index_col=0)
df['motif_bin'] = 'with'
df.loc[df['motif']==0, 'motif_bin'] = 'without'
df['motif_bin'] = pd.Categorical(df['motif_bin'], categories=['without', 'with'])

p = ggplot(df, aes(x='motif_bin', y='attn', fill='motif_bin')) + geom_boxplot(width=0.5, show_legend=False, outlier_shape='') + xlab('') +\
                                                                 coord_cartesian(ylim=(0, 0.4)) +\
                                                                 scale_y_continuous(breaks=np.arange(0, 0.4+0.1, 0.1)) + theme_bw()
p.save(filename='motif_attn_cd8_naive_norm.pdf', dpi=600, height=4, width=4)

df[df['motif_bin']=='with']['attn'].median()      # 0.0695966908002019
df[df['motif_bin']=='without']['attn'].median()   # 0.0556702148976712

stats.ttest_ind(df[df['motif_bin']=='with']['attn'], df[df['motif_bin']=='without']['attn'])  # pvalue=0.0003975540601187627


################ CD4 TCM ################
random.seed(0)
cd4_tcm_idx = list(np.argwhere(rna.obs['cell_anno']=='CD4 TCM').flatten())
cd4_tcm_idx_100 = random.sample(list(cd4_tcm_idx), 100)
rna[cd4_tcm_idx_100].write('rna_cd4_tcm_100.h5ad')
np.count_nonzero(rna[cd4_tcm_idx_100].X.toarray(), axis=1).max()  # 3529

atac[cd4_tcm_idx_100].write('atac_cd4_tcm_100.h5ad')

# python data_preprocess.py -r rna_cd4_tcm_100.h5ad -a atac_cd4_tcm_100.h5ad -s pt_cd4_tcm_100 --dt test -n cd4_tcm_100 --config rna2atac_config_test_cd4_tcm.yaml
# enc_max_len: 3529

with open("rna2atac_config_test_cd4_tcm.yaml", "r") as f:
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

cd4_tcm_data = torch.load("./pt_cd4_tcm_100/cd4_tcm_100_0.pt")
dataset = PreDataset(cd4_tcm_data)
dataloader_kwargs = {'batch_size': 1, 'shuffle': False}
loader = torch.utils.data.DataLoader(dataset, **dataloader_kwargs)

rna_cd4_tcm = sc.read_h5ad('rna_cd4_tcm_100.h5ad')
atac_cd4_tcm = sc.read_h5ad('atac_cd4_tcm_100.h5ad')

cr = ['SMARCA4', 'SMARCA2', 'ARID1A', 'ARID1B', 'SMARCB1',
      'CHD1', 'CHD2', 'CHD3', 'CHD4', 'CHD5', 'CHD6', 'CHD7', 'CHD8', 'CHD9',
      'BRD2', 'BRD3', 'BRD4', 'BRDT',
      'SMARCA5', 'SMARCA1', 'ACTL6A', 'ACTL6B',
      'SSRP1', 'SUPT16H',
      'EP400',
      'SMARCD1', 'SMARCD2', 'SMARCD3']   # 28
tf_lst = pd.read_table('attn/TF_jaspar.txt', header=None)
tf = list(tf_lst[0].values)   # 735
cr_tf = cr + tf   # 763
cr_tf_idx = np.argwhere(rna_cd4_tcm.var.index.isin(cr_tf)).flatten()

out = atac_cd4_tcm.copy()
out.X = np.zeros(out.shape)
i = 0
for inputs in tqdm(loader, ncols=80, desc='output attention matrix'):
    rna_sequence, rna_value, atac_sequence, _, enc_pad_mask = [each.to(device) for each in inputs]
    attn_tmp = model.generate_attn_weight(rna_sequence, atac_sequence, rna_value, enc_mask=enc_pad_mask, which='decoder')[0]
    out.X[i, :] = attn_tmp[:, [a.item() in cr_tf_idx for a in (rna_sequence[rna_sequence!=0]-1)]].sum(axis=1)  # only select CR & TF
    torch.cuda.empty_cache()
    i += 1

out.write('attn_cd4_tcm_100_peak.h5ad')

attn = sc.read_h5ad('attn_cd4_tcm_100_peak.h5ad')

motif_info_raw = pd.read_table('peaks_motifs_count.txt', header=0)
motif_info = motif_info_raw.copy()
motif_info[motif_info!=0] = 1
motif_info = motif_info.sum(axis=1)
motif_info.index = attn.var.index.values

df = pd.DataFrame({'attn':attn.X.sum(axis=0), 'motif':motif_info}).dropna()   # 229525
df['attn'] = (df['attn']-df['attn'].min())/(df['attn'].max()-df['attn'].min())
df.to_csv('attn/motif_attn_cd4_tcm_norm.txt', sep='\t')

## plot box
from plotnine import *
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import stats

plt.rcParams['pdf.fonttype'] = 42

df = pd.read_table('motif_attn_cd4_tcm_norm.txt', index_col=0)
df['motif_bin'] = 'with'
df.loc[df['motif']==0, 'motif_bin'] = 'without'
df['motif_bin'] = pd.Categorical(df['motif_bin'], categories=['without', 'with'])

p = ggplot(df, aes(x='motif_bin', y='attn', fill='motif_bin')) + geom_boxplot(width=0.5, show_legend=False, outlier_shape='') + xlab('') +\
                                                                 coord_cartesian(ylim=(0, 0.5)) +\
                                                                 scale_y_continuous(breaks=np.arange(0, 0.5+0.1, 0.1)) + theme_bw()
p.save(filename='motif_attn_cd4_tcm_norm.pdf', dpi=600, height=4, width=4)

df[df['motif_bin']=='with']['attn'].median()      # 0.11073750773503599
df[df['motif_bin']=='without']['attn'].median()   # 0.081111019237932

stats.ttest_ind(df[df['motif_bin']=='with']['attn'], df[df['motif_bin']=='without']['attn'])  # pvalue=0.0006936879074059761



