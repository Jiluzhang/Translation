## workdir: /fs/home/jiluzhang/Nature_methods/Figure_2/pan_cancer/cisformer/Macrophage

import scanpy as sc
import harmonypy as hm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

plt.rcParams['pdf.fonttype'] = 42

rna = sc.read_h5ad('../rna.h5ad')   # 144409 × 19160
macro = rna[rna.obs['cell_anno']=='Macrophages'].copy()  # 52030 × 19160

sc.pp.normalize_total(macro, target_sum=1e4)
sc.pp.log1p(macro)
sc.pp.highly_variable_genes(macro, batch_key='cancer_type')
macro.raw = macro
macro = macro[:, macro.var.highly_variable]   # 52030 × 2108
sc.tl.pca(macro)
ho = hm.run_harmony(macro.obsm['X_pca'], macro.obs, 'cancer_type')  # Converged after 2 iterations

macro.obsm['X_pca_harmony'] = ho.Z_corr.T
sc.pp.neighbors(macro, use_rep='X_pca_harmony')
sc.tl.umap(macro)

macro.write('res_macro.h5ad')

sc.tl.leiden(macro, resolution=0.75)
macro.obs['leiden'].value_counts()
# 0     7599
# 1     6902
# 2     6569
# 3     5310
# 4     4431
# 5     4326
# 6     4234
# 7     4089
# 8     2508
# 9     2316
# 10    1796
# 11     651
# 12     577
# 13     472
# 14     250

# Mono_FCN1 -> Macro_IL32  -> Macro_C1QC
sc.pl.matrixplot(macro, ['FCN1', 'VCAN',
                         'IL32',
                         'APOE', 'C1QC', 'SPP1', 'CDC20', 'FCGR3A'], groupby='leiden', cmap='coolwarm', standard_scale='var', save='leiden_heatmap_macro.pdf')
# 7: Mono_FCN1
# 5: Macro_IL32
# 4: Macro_C1QC

sc.pl.umap(macro, color='leiden', legend_fontsize='5', legend_loc='right margin', size=5,
           title='', frameon=True, save='_leiden_macro.pdf')

macro.write('res_macro.h5ad')

## macro_fcn1_il32_c1qc
macro_fcn1_il32_c1qc = macro[macro.obs['leiden'].isin(['4', '5', '7'])].copy()  # 12846 × 2108

macro_fcn1_il32_c1qc.obs['cell_anno'] = macro_fcn1_il32_c1qc.obs['leiden']
macro_fcn1_il32_c1qc.obs['cell_anno'].replace({'4':'Macro_C1QC', '5':'Macro_IL32', '7':'Mono_FCN1'}, inplace=True)

sc.pl.umap(macro_fcn1_il32_c1qc, color='cell_anno', legend_fontsize='5', legend_loc='right margin', size=5,
           title='', frameon=True, save='_cell_anno_macro_fcn1_il32_c1qc.pdf')

macro_fcn1_il32_c1qc.obs['cell_anno'] = pd.Categorical(macro_fcn1_il32_c1qc.obs['cell_anno'], categories=['Mono_FCN1', 'Macro_IL32', 'Macro_C1QC'])

sc.pl.matrixplot(macro_fcn1_il32_c1qc, ['FCN1', 'IL32', 'C1QC'],
                 groupby='cell_anno', cmap='coolwarm', standard_scale='var',
                 save='cell_anno_heatmap_macro_fcn1_il32_c1qc.pdf')

sc.tl.paga(macro_fcn1_il32_c1qc, groups='cell_anno')
sc.pl.paga(macro_fcn1_il32_c1qc, layout='fa', color=['cell_anno'], save='_macro_fcn1_il32_c1qc.pdf')

sc.tl.draw_graph(macro_fcn1_il32_c1qc, init_pos='paga')
sc.pl.draw_graph(macro_fcn1_il32_c1qc, color=['cell_anno'], legend_loc='on data', save='_macro_fcn1_il32_c1qc_cell_anno.pdf')

macro_fcn1_il32_c1qc.uns["iroot"] = np.flatnonzero(macro_fcn1_il32_c1qc.obs["cell_anno"]=='Mono_FCN1')[0]
sc.tl.dpt(macro_fcn1_il32_c1qc)
sc.pl.draw_graph(macro_fcn1_il32_c1qc, color=['cell_anno', 'dpt_pseudotime'], vmax=0.5, legend_loc='on data', save='_macro_fcn1_il32_c1qc_pseudo.pdf')

sc.pl.umap(macro_fcn1_il32_c1qc, color='dpt_pseudotime', legend_fontsize='5', legend_loc='right margin', size=5, vmax=0.5, cmap='coolwarm',
           title='', frameon=True, save='_cell_anno_macro_fcn1_il32_c1qc_pseudo.pdf')

macro_fcn1_il32_c1qc.write('res_macro_fcn1_il32_c1qc.h5ad')








## Attention score comparison during CD8+ T cell exhaustion process
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

#################################### naive T ####################################
rna_naive_effect_ex = sc.read_h5ad('res_cd8_t_naive_effect_ex.h5ad')
random.seed(0)
naive_idx = list(np.argwhere(rna_naive_effect_ex.obs['cell_anno']=='naive').flatten())
naive_idx_20 = random.sample(list(naive_idx), 20)
out = sc.AnnData(rna_naive_effect_ex[naive_idx_20].raw.X, obs=rna_naive_effect_ex[naive_idx_20].obs, var=rna_naive_effect_ex[naive_idx_20].raw.var)
np.count_nonzero(out.X.toarray(), axis=1).max()  # 1868
out.write('rna_naive_20.h5ad')

atac = sc.read_h5ad('../atac.h5ad')
atac_naive = atac[rna_naive_effect_ex.obs.index[naive_idx_20]].copy()
atac_naive.write('atac_naive_20.h5ad')

# python data_preprocess.py -r rna_naive_20.h5ad -a atac_naive_20.h5ad -s naive_pt_20 --dt test -n naive --config rna2atac_config_test_naive.yaml   # enc_max_len: 1868 

with open("rna2atac_config_test_naive.yaml", "r") as f:
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
model.load_state_dict(torch.load('../save/2024-10-25_rna2atac_pan_cancer_5/pytorch_model.bin'))
device = torch.device('cuda:7')
model.to(device)
model.eval()

naive_data = torch.load("./naive_pt_20/naive_0.pt")
dataset = PreDataset(naive_data)
dataloader_kwargs = {'batch_size': 1, 'shuffle': False}
loader = torch.utils.data.DataLoader(dataset, **dataloader_kwargs)

# time: 1.5 min
i=0
for inputs in tqdm(loader, ncols=80, desc='output attention matrix'):
    rna_sequence, rna_value, atac_sequence, _, enc_pad_mask = [each.to(device) for each in inputs]
    attn = model.generate_attn_weight(rna_sequence, atac_sequence, rna_value, enc_mask=enc_pad_mask, which='decoder')
    attn = attn[0]  # attn = attn[0].to(torch.float16)
    with h5py.File('attn/attn_naive_'+str(i)+'.h5', 'w') as f:
        f.create_dataset('attn', data=attn)
    torch.cuda.empty_cache()
    i += 1

rna_naive_20 = sc.read_h5ad('rna_naive_20.h5ad')
nonzero = np.count_nonzero(rna_naive_20.X.toarray(), axis=0)
gene_lst = rna_naive_20.var.index[nonzero>=3]  # 2693

atac_naive_20 = sc.read_h5ad('atac_naive_20.h5ad')

gene_attn = {}
for gene in gene_lst:
    gene_attn[gene] = np.zeros([atac_naive_20.shape[1], 1], dtype='float32')  # not float16

for i in range(20):
    rna_sequence = naive_data[0][i].flatten()
    with h5py.File('attn/attn_naive_'+str(i)+'.h5', 'r') as f:
        attn = f['attn'][:]
        for gene in tqdm(gene_lst, ncols=80, desc='cell '+str(i)):
            idx = torch.argwhere((rna_sequence==(np.argwhere(rna_naive_20.var.index==gene))[0][0]+1).flatten())
            if len(idx)!=0:
                gene_attn[gene] += attn[:, [idx.item()]]

with open('./attn/attn_20_naive_3cell.pkl', 'wb') as file:
    pickle.dump(gene_attn, file)

gene_attn_sum = [gene_attn[gene].sum() for gene in gene_lst]
df = pd.DataFrame({'gene':gene_lst, 'attn':gene_attn_sum})
df.to_csv('./attn/attn_no_norm_cnt_20_naive_3cell.txt', sep='\t', header=None, index=None)

df['attn'].sum()   # 5300202.5


#################################### effect T ####################################
rna_naive_effect_ex = sc.read_h5ad('res_cd8_t_naive_effect_ex.h5ad')
random.seed(0)
effect_idx = list(np.argwhere(rna_naive_effect_ex.obs['cell_anno']=='effect').flatten())
effect_idx_20 = random.sample(list(effect_idx), 20)
out = sc.AnnData(rna_naive_effect_ex[effect_idx_20].raw.X, obs=rna_naive_effect_ex[effect_idx_20].obs, var=rna_naive_effect_ex[effect_idx_20].raw.var)
np.count_nonzero(out.X.toarray(), axis=1).max()  # 1345
out.write('rna_effect_20.h5ad')

atac = sc.read_h5ad('../atac.h5ad')
atac_effect = atac[rna_naive_effect_ex.obs.index[effect_idx_20]].copy()
atac_effect.write('atac_effect_20.h5ad')

# python data_preprocess.py -r rna_effect_20.h5ad -a atac_effect_20.h5ad -s effect_pt_20 --dt test -n effect --config rna2atac_config_test_effect.yaml   # enc_max_len: 1345 

with open("rna2atac_config_test_effect.yaml", "r") as f:
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
model.load_state_dict(torch.load('../save/2024-10-25_rna2atac_pan_cancer_5/pytorch_model.bin'))
device = torch.device('cuda:7')
model.to(device)
model.eval()

effect_data = torch.load("./effect_pt_20/effect_0.pt")
dataset = PreDataset(effect_data)
dataloader_kwargs = {'batch_size': 1, 'shuffle': False}
loader = torch.utils.data.DataLoader(dataset, **dataloader_kwargs)

# time: 1.5 min
i=0
for inputs in tqdm(loader, ncols=80, desc='output attention matrix'):
    rna_sequence, rna_value, atac_sequence, _, enc_pad_mask = [each.to(device) for each in inputs]
    attn = model.generate_attn_weight(rna_sequence, atac_sequence, rna_value, enc_mask=enc_pad_mask, which='decoder')
    attn = attn[0]  # attn = attn[0].to(torch.float16)
    with h5py.File('attn/attn_effect_'+str(i)+'.h5', 'w') as f:
        f.create_dataset('attn', data=attn)
    torch.cuda.empty_cache()
    i += 1

rna_effect_20 = sc.read_h5ad('rna_effect_20.h5ad')
nonzero = np.count_nonzero(rna_effect_20.X.toarray(), axis=0)
gene_lst = rna_effect_20.var.index[nonzero>=3]  # 2368

atac_effect_20 = sc.read_h5ad('atac_effect_20.h5ad')

gene_attn = {}
for gene in gene_lst:
    gene_attn[gene] = np.zeros([atac_effect_20.shape[1], 1], dtype='float32')  # not float16

for i in range(20):
    rna_sequence = effect_data[0][i].flatten()
    with h5py.File('attn/attn_effect_'+str(i)+'.h5', 'r') as f:
        attn = f['attn'][:]
        for gene in tqdm(gene_lst, ncols=80, desc='cell '+str(i)):
            idx = torch.argwhere((rna_sequence==(np.argwhere(rna_effect_20.var.index==gene))[0][0]+1).flatten())
            if len(idx)!=0:
                gene_attn[gene] += attn[:, [idx.item()]]

with open('./attn/attn_20_effect_3cell.pkl', 'wb') as file:
    pickle.dump(gene_attn, file)

gene_attn_sum = [gene_attn[gene].sum() for gene in gene_lst]
df = pd.DataFrame({'gene':gene_lst, 'attn':gene_attn_sum})
df.to_csv('./attn/attn_no_norm_cnt_20_effect_3cell.txt', sep='\t', header=None, index=None)

df['attn'].sum()   # 5207205.0


#################################### ex T ####################################
rna_naive_effect_ex = sc.read_h5ad('res_cd8_t_naive_effect_ex.h5ad')
random.seed(0)
ex_idx = list(np.argwhere(rna_naive_effect_ex.obs['cell_anno']=='ex').flatten())
ex_idx_20 = random.sample(list(ex_idx), 20)
out = sc.AnnData(rna_naive_effect_ex[ex_idx_20].raw.X, obs=rna_naive_effect_ex[ex_idx_20].obs, var=rna_naive_effect_ex[ex_idx_20].raw.var)
np.count_nonzero(out.X.toarray(), axis=1).max()  # 2152
out.write('rna_ex_20.h5ad')

atac = sc.read_h5ad('../atac.h5ad')
atac_ex = atac[rna_naive_effect_ex.obs.index[ex_idx_20]].copy()
atac_ex.write('atac_ex_20.h5ad')

# python data_preprocess.py -r rna_ex_20.h5ad -a atac_ex_20.h5ad -s ex_pt_20 --dt test -n ex --config rna2atac_config_test_ex.yaml   # enc_max_len: 2152 

with open("rna2atac_config_test_ex.yaml", "r") as f:
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
model.load_state_dict(torch.load('../save/2024-10-25_rna2atac_pan_cancer_5/pytorch_model.bin'))
device = torch.device('cuda:7')
model.to(device)
model.eval()

ex_data = torch.load("./ex_pt_20/ex_0.pt")
dataset = PreDataset(ex_data)
dataloader_kwargs = {'batch_size': 1, 'shuffle': False}
loader = torch.utils.data.DataLoader(dataset, **dataloader_kwargs)

# time: 1.5 min
i=0
for inputs in tqdm(loader, ncols=80, desc='output attention matrix'):
    rna_sequence, rna_value, atac_sequence, _, enc_pad_mask = [each.to(device) for each in inputs]
    attn = model.generate_attn_weight(rna_sequence, atac_sequence, rna_value, enc_mask=enc_pad_mask, which='decoder')
    attn = attn[0]  # attn = attn[0].to(torch.float16)
    with h5py.File('attn/attn_ex_'+str(i)+'.h5', 'w') as f:
        f.create_dataset('attn', data=attn)
    torch.cuda.empty_cache()
    i += 1

rna_ex_20 = sc.read_h5ad('rna_ex_20.h5ad')
nonzero = np.count_nonzero(rna_ex_20.X.toarray(), axis=0)
gene_lst = rna_ex_20.var.index[nonzero>=3]  # 2772

atac_ex_20 = sc.read_h5ad('atac_ex_20.h5ad')

gene_attn = {}
for gene in gene_lst:
    gene_attn[gene] = np.zeros([atac_ex_20.shape[1], 1], dtype='float32')  # not float16

for i in range(20):
    rna_sequence = ex_data[0][i].flatten()
    with h5py.File('attn/attn_ex_'+str(i)+'.h5', 'r') as f:
        attn = f['attn'][:]
        for gene in tqdm(gene_lst, ncols=80, desc='cell '+str(i)):
            idx = torch.argwhere((rna_sequence==(np.argwhere(rna_ex_20.var.index==gene))[0][0]+1).flatten())
            if len(idx)!=0:
                gene_attn[gene] += attn[:, [idx.item()]]

with open('./attn/attn_20_ex_3cell.pkl', 'wb') as file:
    pickle.dump(gene_attn, file)

gene_attn_sum = [gene_attn[gene].sum() for gene in gene_lst]
df = pd.DataFrame({'gene':gene_lst, 'attn':gene_attn_sum})
df.to_csv('./attn/attn_no_norm_cnt_20_ex_3cell.txt', sep='\t', header=None, index=None)

df['attn'].sum()   # 5167448.0



###### rank genes based on attention score
import pandas as pd
import numpy as np
from plotnine import *
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['pdf.fonttype'] = 42

cr = ['SMARCA4', 'SMARCA2', 'ARID1A', 'ARID1B', 'SMARCB1',
      'CHD1', 'CHD2', 'CHD3', 'CHD4', 'CHD5', 'CHD6', 'CHD7', 'CHD8', 'CHD9',
      'BRD2', 'BRD3', 'BRD4', 'BRDT',
      'SMARCA5', 'SMARCA1', 'ACTL6A', 'ACTL6B',
      'SSRP1', 'SUPT16H',
      'EP400',
      'SMARCD1', 'SMARCD2', 'SMARCD3']   # 28
tf_lst = pd.read_table('TF_jaspar.txt', header=None)
tf = list(tf_lst[0].values)   # 735
cr_tf = cr + tf   # 763

df_naive = pd.read_csv('attn_no_norm_cnt_20_naive_3cell.txt', header=None, sep='\t')
df_naive.columns = ['gene', 'attn']
df_naive['attn_norm'] = df_naive['attn']/df_naive['attn'].max()

df_effect = pd.read_csv('attn_no_norm_cnt_20_effect_3cell.txt', header=None, sep='\t')
df_effect.columns = ['gene', 'attn']
df_effect['attn_norm'] = df_effect['attn']/df_effect['attn'].max()

df_ex = pd.read_csv('attn_no_norm_cnt_20_ex_3cell.txt', header=None, sep='\t')
df_ex.columns = ['gene', 'attn']
df_ex['attn_norm'] = df_ex['attn']/df_ex['attn'].max()

sum(df_naive['attn']/df_naive['attn'].max())
# 13.304571276765705

sum(df_effect['attn']/df_effect['attn'].max())
# 14.894872854946211

sum(df_ex['attn']/df_ex['attn'].max())
# 9.518939784170293

df_naive[df_naive['gene'].isin(cr_tf)]['attn_norm'].sum()
# 0.6764622481005079

df_effect[df_effect['gene'].isin(cr_tf)]['attn_norm'].sum()
# 1.3884113822466762

df_ex[df_ex['gene'].isin(cr_tf)]['attn_norm'].sum()
# 0.935870134228533

dat = pd.DataFrame({'idx':['naive', 'effect', 'ex'],
                    'val':[0.676, 1.388, 0.936]})
dat['idx'] = pd.Categorical(dat['idx'], categories=['naive', 'effect', 'ex'])

p = ggplot(dat, aes(x='idx', y='val', fill='idx')) + geom_col(position='dodge') +\
                                                     scale_y_continuous(limits=[0, 1.5], breaks=np.arange(0, 1.5+0.5, 0.5)) + theme_bw()  # limits min must set to 0
p.save(filename='cd8_t_naive_effect_ex_reg_score.pdf', dpi=600, height=4, width=4)

## heatmap of factors (may be retrain the model)
df_naive_effect = pd.merge(df_naive[['gene', 'attn_norm']], df_effect[['gene', 'attn_norm']], how='outer', on='gene')
df_naive_effect_ex = pd.merge(df_naive_effect, df_ex[['gene', 'attn_norm']], how='outer')
df_naive_effect_ex.fillna(0, inplace=True)
df_naive_effect_ex.rename(columns={'attn_norm_x':'attn_norm_naive', 'attn_norm_y':'attn_norm_effect', 'attn_norm':'attn_norm_ex'}, inplace=True)

df_naive_effect_ex_tf = df_naive_effect_ex[df_naive_effect_ex['gene'].isin(cr_tf)]

naive_sp = df_naive_effect_ex_tf[(df_naive_effect_ex_tf['attn_norm_naive']>df_naive_effect_ex_tf['attn_norm_effect']) & 
                                 (df_naive_effect_ex_tf['attn_norm_naive']>df_naive_effect_ex_tf['attn_norm_ex'])]
effect_sp = df_naive_effect_ex_tf[(df_naive_effect_ex_tf['attn_norm_effect']>df_naive_effect_ex_tf['attn_norm_naive']) & 
                                  (df_naive_effect_ex_tf['attn_norm_effect']>df_naive_effect_ex_tf['attn_norm_ex'])]
ex_sp = df_naive_effect_ex_tf[(df_naive_effect_ex_tf['attn_norm_ex']>df_naive_effect_ex_tf['attn_norm_naive']) & 
                              (df_naive_effect_ex_tf['attn_norm_ex']>df_naive_effect_ex_tf['attn_norm_effect'])]
sp = pd.concat([naive_sp, effect_sp, ex_sp])
sp.index = sp['gene'].values

sns.clustermap(sp[['attn_norm_naive', 'attn_norm_effect', 'attn_norm_ex']],
               cmap='coolwarm', row_cluster=False, col_cluster=False, z_score=0, vmin=-1.2, vmax=1.2, figsize=(5, 40)) 
plt.savefig('cd8_t_naive_effect_ex_specific_factor_heatmap.pdf')
plt.close()
