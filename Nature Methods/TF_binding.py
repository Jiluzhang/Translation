## workdir: /fs/home/jiluzhang/Nature_methods/TF_binding

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
device = torch.device('cuda:6')
model.to(device)
model.eval()

################################ CD14 Mono ################################
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

random.seed(0)
idx = list(np.argwhere(rna.obs['cell_anno']=='CD14 Mono').flatten())
idx_20 = random.sample(list(idx), 20)
rna[idx_20].write('rna_cd14_mono_20.h5ad')
np.count_nonzero(rna[idx_20].X.toarray(), axis=1).max()  # 2726

atac = sc.read_h5ad('atac_train.h5ad')   # 6974 × 236295
atac[idx_20].write('atac_cd14_mono_20.h5ad')

## output peaks with bed format
peaks = atac.var[['gene_ids']]
peaks['chr'] = peaks['gene_ids'].map(lambda x: x.split(':')[0])
peaks['start'] = peaks['gene_ids'].map(lambda x: x.split(':')[1].split('-')[0])
peaks['end'] = peaks['gene_ids'].map(lambda x: x.split(':')[1].split('-')[1])
peaks[['chr', 'start', 'end']].to_csv('peaks.bed', index=False, header=False, sep='\t')

# python data_preprocess.py -r rna_cd14_mono_20.h5ad -a atac_cd14_mono_20.h5ad -s pt_cd14_mono_20 --dt test -n cd14_mono_20 --config rna2atac_config_test_cd14_mono_20.yaml
# enc_max_len: 2726

cd14_mono_data = torch.load("./pt_cd14_mono_20/cd14_mono_20_0.pt")
dataset = PreDataset(cd14_mono_data)
dataloader_kwargs = {'batch_size': 1, 'shuffle': False}
loader = torch.utils.data.DataLoader(dataset, **dataloader_kwargs)

# time: 1 min
i=0
for inputs in tqdm(loader, ncols=80, desc='output attention matrix'):
    rna_sequence, rna_value, atac_sequence, _, enc_pad_mask = [each.to(device) for each in inputs]
    attn = model.generate_attn_weight(rna_sequence, atac_sequence, rna_value, enc_mask=enc_pad_mask, which='decoder')
    attn = attn[0]  # attn = attn[0].to(torch.float16)
    with h5py.File('attn/attn_cd14_mono_'+str(i)+'.h5', 'w') as f:
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
    with h5py.File('attn/attn_cd14_mono_'+str(i)+'.h5', 'r') as f:
        attn = f['attn'][:]
        for gene in tqdm(gene_lst, ncols=80, desc='cell '+str(i)):
            idx = torch.argwhere((rna_sequence==(np.argwhere(rna_cd14_mono_20.var.index==gene))[0][0]+1).flatten())
            if len(idx)!=0:
                gene_attn[gene] += attn[:, [idx.item()]]

with open('./attn/attn_20_cd14_mono_3cell.pkl', 'wb') as file:
    pickle.dump(gene_attn, file)

# for gene in gene_attn:
#     gene_attn[gene] = gene_attn[gene].flatten()

# gene_attn_df = pd.DataFrame(gene_attn)
# gene_attn_df.index = atac_cd14_mono_20.var.index.values
# gene_attn_df.to_csv('./attn/attn_no_norm_cnt_20_cd14_mono_3cell.txt', sep='\t', header=True, index=True)

# gene_attn_sum = [gene_attn[gene].sum() for gene in gene_lst]
# df = pd.DataFrame({'gene':gene_lst, 'cnt':gene_attn_sum})
# df.to_csv('./attn/attn_no_norm_cnt_20_cd14_mono_3cell.txt', sep='\t', header=None, index=None)


with open('./attn/attn_20_cd14_mono_3cell.pkl', 'rb') as file:
    gene_attn = pickle.load(file)

for gene in gene_attn:
    gene_attn[gene] = gene_attn[gene].flatten()

gene_attn_df = pd.DataFrame(gene_attn)   # 236295 x 4722
gene_attn_df.index = atac_cd14_mono_20.var.index.values

motif_info_raw = pd.read_table('peaks_motifs_count.txt', header=0)

motif_tf_lst = []
motif_tf_idx = []
p = 0
for i in motif_info_raw.columns:
    tf = i.split('_')[1]
    if '.' not in tf:
        motif_tf_lst.append(tf)
        motif_tf_idx.append(p)
    p += 1

motif_tf = []
motif_0_attn = []
motif_1_attn = []
for i in tqdm(range(len(motif_tf_lst)), ncols=80):
    tf = motif_tf_lst[i]
    if tf in gene_attn_df.columns:
        gene_attn_df_tf = gene_attn_df[tf]
        #gene_attn_df_tf = np.log10(gene_attn_df_tf/gene_attn_df_tf.min())
        #gene_attn_df_tf = (gene_attn_df_tf-gene_attn_df_tf.min())/(gene_attn_df_tf.max()-gene_attn_df_tf.min())
        
        motif_info = pd.DataFrame({'motif_or_not':motif_info_raw.iloc[:, motif_tf_idx[i]]})
        motif_info[motif_info['motif_or_not']>0] = 1
        motif_info.index = gene_attn_df.index.values

        motif_attn = pd.concat([gene_attn_df_tf, motif_info], axis=1)
        motif_attn.columns = ['attn', 'motif']

        motif_tf.append(tf)
        motif_0_attn.append(motif_attn[motif_attn['motif']==0]['attn'].mean())
        motif_1_attn.append(motif_attn[motif_attn['motif']==1]['attn'].mean())


motif_info.loc[gene_attn_df_tf.sort_values(ascending=False)[:10000].index].value_counts(normalize=True)

################################################## HERE ##################################################
################################################## HERE ##################################################
################################################## HERE ##################################################
################################################## HERE ##################################################
################################################## HERE ##################################################

df_0 = pd.DataFrame({'attn':motif_0_attn, 'motif':0})
df_1 = pd.DataFrame({'attn':motif_1_attn, 'motif':1})
df = pd.concat([df_0, df_1], axis=0)
df.to_csv('motif_attn.csv')

# ## one tf
# ccnl2 = pd.concat([gene_attn_df['CCNL2'], motif_info['motif_or_not']], axis=1)
# ccnl2.columns = ['attn', 'motif']
# ccnl2.to_csv('ccnl2_attn_motif_box.csv')

from plotnine import *
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import stats

plt.rcParams['pdf.fonttype'] = 42

df = pd.read_csv('motif_attn.csv', index_col=0)
df['motif'] = df['motif'].apply(str)  
df = df.replace([np.inf, -np.inf], np.nan).dropna()
df = df[df['attn']!=0]

p = ggplot(df, aes(x='motif', y='attn', fill='motif')) + geom_boxplot(width=0.5, show_legend=False, outlier_shape='') + xlab('') +\
                                                         scale_y_continuous(limits=[0.8, 1.0], breaks=np.arange(0.8, 1.0+0.1, 0.04)) + theme_bw()
p.save(filename='motif_attn.pdf', dpi=600, height=4, width=4)

stats.ttest_ind(df[df['motif']=='0']['attn'], df[df['motif']=='1']['attn'])  # pvalue=0.0020325117440889865
stats.ttest_rel(df[df['motif']=='0']['attn'], df[df['motif']=='1']['attn'])  # pvalue=0.0020325117440889865



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









