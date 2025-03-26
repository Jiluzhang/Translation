## tractory analysis: https://scanpy.readthedocs.io/en/latest/tutorials/trajectories/paga-paul15.html
## workdir: /fs/home/jiluzhang/Nature_methods/Figure_2/pan_cancer/cisformer/T_cell

import scanpy as sc
import harmonypy as hm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

plt.rcParams['pdf.fonttype'] = 42

rna = sc.read_h5ad('../rna.h5ad')   # 144409 × 19160
tcell = rna[rna.obs['cell_anno']=='T-cells'].copy()  # 36898 × 19160

sc.pp.normalize_total(tcell, target_sum=1e4)
sc.pp.log1p(tcell)
# fibro.obs['batch'] = [x.split('_')[0] for x in fibro.obs.index]  # exist samples with only one fibroblast
sc.pp.highly_variable_genes(tcell, batch_key='cancer_type')
tcell.raw = tcell
tcell = tcell[:, tcell.var.highly_variable]   # 36898 × 1868
sc.tl.pca(tcell)
ho = hm.run_harmony(tcell.obsm['X_pca'], tcell.obs, 'cancer_type')  # Converged after 2 iterations

tcell.obsm['X_pca_harmony'] = ho.Z_corr.T
sc.pp.neighbors(tcell, use_rep='X_pca_harmony')
sc.tl.umap(tcell)

sc.tl.leiden(tcell, resolution=0.75)
tcell.obs['leiden'].value_counts()
# 0    10811
# 1     6304
# 2     5198
# 3     4721
# 4     2767
# 5     2570
# 6     1588
# 7     1587
# 8     1352

sc.pl.matrixplot(tcell, ['CD4', 'CD8A', 'CD8B'], groupby='leiden', cmap='coolwarm', standard_scale='var', save='leiden_heatmap_tcell.pdf')
# cluster 1 & 3 highly express CD8A & CD8B

sc.pl.umap(tcell, color='leiden', legend_fontsize='5', legend_loc='right margin', size=5,
           title='', frameon=True, save='_leiden_tcell.pdf')

tcell.write('res_tcell.h5ad')

## cd8 t
cd8_t = tcell[tcell.obs['leiden'].isin(['1', '3'])].copy()  # 11025 × 1868

sc.tl.leiden(cd8_t, resolution=0.90)
cd8_t.obs['leiden'].value_counts()
# 0    2937
# 1    2697
# 2    2010
# 3    1393
# 4     960
# 5     529
# 6     499

# naive T  &  effector T  &  memory T  &  exhausted T
sc.pl.matrixplot(cd8_t, ['CCR7', 'LEF1', 'SELL', 'CD3E', 'CD3D',
                         'GNLY', 'GZMB', 'NKG7', 'CD3D', 'PRF1',
                         'CCR7', 'CD3G', 'IL7R', 'SELL', 'CD3D',
                         'PDCD1', 'CTLA4', 'HAVCR2', 'CD101', 'CD38'], groupby='leiden', cmap='coolwarm', standard_scale='var', dendrogram=True, save='leiden_heatmap_cd8_t.pdf')

sc.pl.umap(cd8_t, color='leiden', legend_fontsize='5', legend_loc='right margin', size=5,
           title='', frameon=True, save='_leiden_cd8_t.pdf')

# marker genes from NBT
# sc.pl.matrixplot(cd8_t, ['CCR7', 'TCF7',
#                          'EOMES', 'IFNG',
#                          'PDCD1', 'CTLA4', 'HAVCR2', 'CD101', 'CD38',
#                          'CD69', 'CD7', 'CRTAM', 'CD3G', 'IL7R', 'SELL'], groupby='leiden', cmap='coolwarm', standard_scale='var', save='cd8_t_leiden_heatmap.pdf')

cd8_t.obs['cell_anno'] = cd8_t.obs['leiden']
cd8_t.obs['cell_anno'].replace({'0':'naive', '1':'ex', '2':'memory', '3':'ex', '4':'effect', '5':'effect', '6':'naive'}, inplace=True)  # 3 (effect -> ex)

sc.pl.umap(cd8_t, color='cell_anno', legend_fontsize='5', legend_loc='right margin', size=5,
           title='', frameon=True, save='_cell_anno_cd8_t.pdf')

cd8_t.write('res_cd8_t.h5ad')

## cd8_t_naive_effect_ex
cd8_t_naive_effect_ex = cd8_t[cd8_t.obs['cell_anno'].isin(['naive', 'effect', 'ex'])].copy()  # 9015 × 1868

cd8_t_naive_effect_ex.uns['cell_anno_colors'] = ['#ff7f0e', '#1f77b4', '#d62728']
sc.pl.umap(cd8_t_naive_effect_ex, color='cell_anno', legend_fontsize='5', legend_loc='right margin', size=5,
           title='', frameon=True, save='_cell_anno_cd8_t_naive_effect_ex.pdf')

# sc.pl.dotplot(cd8_t_naive_effect_ex, ['CCR7', 'LEF1', 'SELL', 'CD3E', 'CD3D',
#                                       'EOMES', 'IFNG', 'GNLY', 'GZMB', 'NKG7', 'CD3D', 'PRF1',
#                                       'PDCD1', 'CTLA4', 'HAVCR2', 'CD101'], 
#               groupby='cell_anno', standard_scale='var',
#               save='cell_anno_cd8_t_naive_effect_ex.pdf')

cd8_t_naive_effect_ex.obs['cell_anno'] = pd.Categorical(cd8_t_naive_effect_ex.obs['cell_anno'], categories=['naive', 'effect', 'ex'])

sc.pl.matrixplot(cd8_t_naive_effect_ex, ['CCR7', 'LEF1', 'SELL',
                                         'NKG7', 'PRF1',
                                         'PDCD1', 'CTLA4', 'HAVCR2'],
                 groupby='cell_anno', cmap='coolwarm', standard_scale='var',
                 save='cell_anno_heatmap_cd8_t_naive_effect_ex.pdf')

sc.tl.paga(cd8_t_naive_effect_ex, groups='cell_anno')
sc.pl.paga(cd8_t_naive_effect_ex, layout='fa', color=['cell_anno'], save='_cd8_t_naive_effect_ex.pdf')

sc.tl.draw_graph(cd8_t_naive_effect_ex, init_pos='paga')
sc.pl.draw_graph(cd8_t_naive_effect_ex, color=['cell_anno'], legend_loc='on data', save='_cd8_t_naive_effect_ex_cell_anno.pdf')

cd8_t_naive_effect_ex.uns["iroot"] = np.flatnonzero(cd8_t_naive_effect_ex.obs["cell_anno"]=='naive')[0]
sc.tl.dpt(cd8_t_naive_effect_ex)
sc.pl.draw_graph(cd8_t_naive_effect_ex, color=['cell_anno', 'dpt_pseudotime'], vmax=0.5, legend_loc='on data', save='_cd8_t_naive_effect_ex_pseudo.pdf')

sc.pl.umap(cd8_t_naive_effect_ex, color='dpt_pseudotime', legend_fontsize='5', legend_loc='right margin', size=5, vmax=0.15, cmap='coolwarm',
           title='', frameon=True, save='_cell_anno_cd8_t_naive_effect_ex_pseudo.pdf')

cd8_t_naive_effect_ex.write('res_cd8_t_naive_effect_ex.h5ad')


######################################################################## naive ########################################################################
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

rna_cd8_t_naive_effect_ex = sc.read_h5ad('res_cd8_t_naive_effect_ex.h5ad')
random.seed(0)
naive_idx = list(np.argwhere(rna_cd8_t_naive_effect_ex.obs['cell_anno']=='naive').flatten())
naive_idx_100 = random.sample(list(naive_idx), 100)
out = sc.AnnData(rna_cd8_t_naive_effect_ex[naive_idx_100].raw.X, obs=rna_cd8_t_naive_effect_ex[naive_idx_100].obs, var=rna_cd8_t_naive_effect_ex[naive_idx_100].raw.var)
np.count_nonzero(out.X.toarray(), axis=1).max()  # 1892
out.write('rna_naive_100.h5ad')

atac = sc.read_h5ad('../atac.h5ad')
atac_naive = atac[rna_cd8_t_naive_effect_ex.obs.index[naive_idx_100]].copy()
atac_naive.write('atac_naive_100.h5ad')

# python data_preprocess.py -r rna_naive_100.h5ad -a atac_naive_100.h5ad -s naive_pt_100 --dt test -n naive --config rna2atac_config_test_naive.yaml   # enc_max_len: 1892 

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

naive_data = torch.load("./naive_pt_100/naive_0.pt")
dataset = PreDataset(naive_data)
dataloader_kwargs = {'batch_size': 1, 'shuffle': False}
loader = torch.utils.data.DataLoader(dataset, **dataloader_kwargs)

rna_naive = sc.read_h5ad('rna_naive_100.h5ad')
atac_naive = sc.read_h5ad('atac_naive_100.h5ad')
out = rna_naive.copy()
out.X = np.zeros(out.shape)
i = 0
for inputs in tqdm(loader, ncols=80, desc='output attention matrix'):
    rna_sequence, rna_value, atac_sequence, _, enc_pad_mask = [each.to(device) for each in inputs]
    out.X[i, (rna_sequence[rna_sequence!=0]-1).cpu()] = model.generate_attn_weight(rna_sequence, atac_sequence, rna_value, enc_mask=enc_pad_mask, which='decoder')[0].sum(axis=0)
    torch.cuda.empty_cache()
    i += 1

out.write('attn_naive_100.h5ad')

attn = sc.read_h5ad('attn_naive_100.h5ad')
(attn.X / attn.X.max(axis=1)[:, np.newaxis]).sum(axis=1).mean()  # 6.505314659025544
sum(attn.X.sum(axis=0) / (attn.X.sum(axis=0).max()))             # 13.433583207272893

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

(attn.X.sum(axis=0) / (attn.X.sum(axis=0).max()))[attn.var.index.isin(cr_tf)].sum()   # 0.7030022788395824


######################################################################## effect ########################################################################
rna_cd8_t_naive_effect_ex = sc.read_h5ad('res_cd8_t_naive_effect_ex.h5ad')
random.seed(0)
effect_idx = list(np.argwhere(rna_cd8_t_naive_effect_ex.obs['cell_anno']=='effect').flatten())
effect_idx_100 = random.sample(list(effect_idx), 100)
out = sc.AnnData(rna_cd8_t_naive_effect_ex[effect_idx_100].raw.X, obs=rna_cd8_t_naive_effect_ex[effect_idx_100].obs, var=rna_cd8_t_naive_effect_ex[effect_idx_100].raw.var)
np.count_nonzero(out.X.toarray(), axis=1).max()  # 2576
out.write('rna_effect_100.h5ad')

atac = sc.read_h5ad('../atac.h5ad')
atac_effect = atac[rna_cd8_t_naive_effect_ex.obs.index[effect_idx_100]].copy()
atac_effect.write('atac_effect_100.h5ad')

# python data_preprocess.py -r rna_effect_100.h5ad -a atac_effect_100.h5ad -s effect_pt_100 --dt test -n effect --config rna2atac_config_test_effect.yaml   # enc_max_len: 2576 

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

effect_data = torch.load("./effect_pt_100/effect_0.pt")
dataset = PreDataset(effect_data)
dataloader_kwargs = {'batch_size': 1, 'shuffle': False}
loader = torch.utils.data.DataLoader(dataset, **dataloader_kwargs)

rna_effect = sc.read_h5ad('rna_effect_100.h5ad')
atac_effect = sc.read_h5ad('atac_effect_100.h5ad')
out = rna_effect.copy()
out.X = np.zeros(out.shape)
i = 0
for inputs in tqdm(loader, ncols=80, desc='output attention matrix'):
    rna_sequence, rna_value, atac_sequence, _, enc_pad_mask = [each.to(device) for each in inputs]
    out.X[i, (rna_sequence[rna_sequence!=0]-1).cpu()] = model.generate_attn_weight(rna_sequence, atac_sequence, rna_value, enc_mask=enc_pad_mask, which='decoder')[0].sum(axis=0)
    torch.cuda.empty_cache()
    i += 1

out.write('attn_effect_100.h5ad')

attn = sc.read_h5ad('attn_effect_100.h5ad')
(attn.X / attn.X.max(axis=1)[:, np.newaxis]).sum(axis=1).mean()  # 6.5810430940410365
sum(attn.X.sum(axis=0) / (attn.X.sum(axis=0).max()))             # 13.764085492765144

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

(attn.X.sum(axis=0) / (attn.X.sum(axis=0).max()))[attn.var.index.isin(cr_tf)].sum()   # 0.8288548825845072


######################################################################## ex ########################################################################
rna_cd8_t_naive_effect_ex = sc.read_h5ad('res_cd8_t_naive_effect_ex.h5ad')
random.seed(0)
ex_idx = list(np.argwhere(rna_cd8_t_naive_effect_ex.obs['cell_anno']=='ex').flatten())
ex_idx_100 = random.sample(list(ex_idx), 100)
out = sc.AnnData(rna_cd8_t_naive_effect_ex[ex_idx_100].raw.X, obs=rna_cd8_t_naive_effect_ex[ex_idx_100].obs, var=rna_cd8_t_naive_effect_ex[ex_idx_100].raw.var)
np.count_nonzero(out.X.toarray(), axis=1).max()  # 2333
out.write('rna_ex_100.h5ad')

atac = sc.read_h5ad('../atac.h5ad')
atac_ex = atac[rna_cd8_t_naive_effect_ex.obs.index[ex_idx_100]].copy()
atac_ex.write('atac_ex_100.h5ad')

# python data_preprocess.py -r rna_ex_100.h5ad -a atac_ex_100.h5ad -s ex_pt_100 --dt test -n ex --config rna2atac_config_test_ex.yaml   # enc_max_len: 2333 

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

ex_data = torch.load("./ex_pt_100/ex_0.pt")
dataset = PreDataset(ex_data)
dataloader_kwargs = {'batch_size': 1, 'shuffle': False}
loader = torch.utils.data.DataLoader(dataset, **dataloader_kwargs)

rna_ex = sc.read_h5ad('rna_ex_100.h5ad')
atac_ex = sc.read_h5ad('atac_ex_100.h5ad')
out = rna_ex.copy()
out.X = np.zeros(out.shape)
i = 0
for inputs in tqdm(loader, ncols=80, desc='output attention matrix'):
    rna_sequence, rna_value, atac_sequence, _, enc_pad_mask = [each.to(device) for each in inputs]
    out.X[i, (rna_sequence[rna_sequence!=0]-1).cpu()] = model.generate_attn_weight(rna_sequence, atac_sequence, rna_value, enc_mask=enc_pad_mask, which='decoder')[0].sum(axis=0)
    torch.cuda.empty_cache()
    i += 1

out.write('attn_ex_100.h5ad')

attn = sc.read_h5ad('attn_ex_100.h5ad')
(attn.X / attn.X.max(axis=1)[:, np.newaxis]).sum(axis=1).mean()  # 6.621620476128524
sum(attn.X.sum(axis=0) / (attn.X.sum(axis=0).max()))             # 14.534750947380784

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

(attn.X.sum(axis=0) / (attn.X.sum(axis=0).max()))[attn.var.index.isin(cr_tf)].sum()   # 1.0526708883364444


###### calculate TF_CR regulatory strength
import scanpy as sc
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

attn_naive = sc.read_h5ad('../attn_naive_100.h5ad')
(attn_naive.X.sum(axis=0) / (attn_naive.X.sum(axis=0).max()))[attn_naive.var.index.isin(cr_tf)].sum()   # 0.7030022788395824
(attn_naive.X.sum(axis=0) / (attn_naive.X.sum(axis=0).max()))[attn_naive.var.index.isin(tf)].sum()      # 0.6617182942278171

attn_effect = sc.read_h5ad('../attn_effect_100.h5ad')
(attn_effect.X.sum(axis=0) / (attn_effect.X.sum(axis=0).max()))[attn_effect.var.index.isin(cr_tf)].sum()   # 0.8288548825845072
(attn_effect.X.sum(axis=0) / (attn_effect.X.sum(axis=0).max()))[attn_effect.var.index.isin(tf)].sum()      # 0.782808529601877

attn_ex = sc.read_h5ad('../attn_ex_100.h5ad')
(attn_ex.X.sum(axis=0) / (attn_ex.X.sum(axis=0).max()))[attn_ex.var.index.isin(cr_tf)].sum()   # 1.0526708883364444
(attn_ex.X.sum(axis=0) / (attn_ex.X.sum(axis=0).max()))[attn_ex.var.index.isin(tf)].sum()      # 1.0351343935422546

dat = pd.DataFrame({'idx':['naive', 'effect', 'ex'],
                    'val':[0.703, 0.829, 1.053]})
dat['idx'] = pd.Categorical(dat['idx'], categories=['naive', 'effect', 'ex'])

p = ggplot(dat, aes(x='idx', y='val', fill='idx')) + geom_col(position='dodge') +\
                                                     scale_y_continuous(limits=[0, 1.2], breaks=np.arange(0, 1.2+0.2, 0.2)) + theme_bw()  # limits min must set to 0
p.save(filename='cd8_t_naive_effect_ex_reg_score.pdf', dpi=600, height=4, width=4)

## heatmap of factors (may be retrain the model)
df_naive_effect_ex = pd.DataFrame({'gene':attn_naive.var.index.values,
                                   'attn_norm_naive':(attn_naive.X.sum(axis=0) / (attn_naive.X.sum(axis=0).max())),
                                   'attn_norm_effect':(attn_effect.X.sum(axis=0) / (attn_effect.X.sum(axis=0).max())),
                                   'attn_norm_ex':(attn_ex.X.sum(axis=0) / (attn_ex.X.sum(axis=0).max()))})
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
               cmap='coolwarm', row_cluster=False, col_cluster=False, z_score=0, vmin=-1.2, vmax=1.2, figsize=(5, 150)) 
plt.savefig('cd8_t_naive_effect_ex_specific_factor_heatmap.pdf')
plt.close()


naive_rank = pd.DataFrame({'gene':df_naive_effect_ex_tf.sort_values('attn_norm_naive', ascending=False)['gene'].values, 'naive_rank':range(601)})
effect_rank = pd.DataFrame({'gene':df_naive_effect_ex_tf.sort_values('attn_norm_effect', ascending=False)['gene'].values, 'effect_rank':range(601)})
ex_rank = pd.DataFrame({'gene':df_naive_effect_ex_tf.sort_values('attn_norm_ex', ascending=False)['gene'].values, 'ex_rank':range(601)})

naive_effect_rank = pd.merge(naive_rank, effect_rank)
naive_effect_ex_rank = pd.merge(naive_effect_rank, ex_rank)



naive_rank = pd.DataFrame({'gene':df_naive_effect_ex.sort_values('attn_norm_naive', ascending=False)['gene'].values, 'naive_rank':range(19160)})
effect_rank = pd.DataFrame({'gene':df_naive_effect_ex.sort_values('attn_norm_effect', ascending=False)['gene'].values, 'effect_rank':range(19160)})
ex_rank = pd.DataFrame({'gene':df_naive_effect_ex.sort_values('attn_norm_ex', ascending=False)['gene'].values, 'ex_rank':range(19160)})

naive_effect_rank = pd.merge(naive_rank, effect_rank)
naive_effect_ex_rank = pd.merge(naive_effect_rank, ex_rank)


# ## Attention score comparison during CD8+ T cell exhaustion process
# import pandas as pd
# import numpy as np
# import random
# from functools import partial
# import sys
# sys.path.append("M2Mmodel")
# from utils import PairDataset
# import scanpy as sc
# import torch
# from M2Mmodel.utils import *
# from collections import Counter
# import pickle as pkl
# import yaml
# from M2Mmodel.M2M import M2M_rna2atac

# import h5py
# from tqdm import tqdm

# from multiprocessing import Pool
# import pickle

# #################################### naive T ####################################
# rna_naive_effect_ex = sc.read_h5ad('res_cd8_t_naive_effect_ex.h5ad')
# random.seed(0)
# naive_idx = list(np.argwhere(rna_naive_effect_ex.obs['cell_anno']=='naive').flatten())
# naive_idx_20 = random.sample(list(naive_idx), 20)
# out = sc.AnnData(rna_naive_effect_ex[naive_idx_20].raw.X, obs=rna_naive_effect_ex[naive_idx_20].obs, var=rna_naive_effect_ex[naive_idx_20].raw.var)
# np.count_nonzero(out.X.toarray(), axis=1).max()  # 1868
# out.write('rna_naive_20.h5ad')

# atac = sc.read_h5ad('../atac.h5ad')
# atac_naive = atac[rna_naive_effect_ex.obs.index[naive_idx_20]].copy()
# atac_naive.write('atac_naive_20.h5ad')

# # python data_preprocess.py -r rna_naive_20.h5ad -a atac_naive_20.h5ad -s naive_pt_20 --dt test -n naive --config rna2atac_config_test_naive.yaml   # enc_max_len: 1868 

# with open("rna2atac_config_test_naive.yaml", "r") as f:
#     config = yaml.safe_load(f)

# model = M2M_rna2atac(
#             dim = config["model"].get("dim"),

#             enc_num_gene_tokens = config["model"].get("total_gene") + 1, # +1 for <PAD>
#             enc_num_value_tokens = config["model"].get('max_express') + 1, # +1 for <PAD>
#             enc_depth = config["model"].get("enc_depth"),
#             enc_heads = config["model"].get("enc_heads"),
#             enc_ff_mult = config["model"].get("enc_ff_mult"),
#             enc_dim_head = config["model"].get("enc_dim_head"),
#             enc_emb_dropout = config["model"].get("enc_emb_dropout"),
#             enc_ff_dropout = config["model"].get("enc_ff_dropout"),
#             enc_attn_dropout = config["model"].get("enc_attn_dropout"),

#             dec_depth = config["model"].get("dec_depth"),
#             dec_heads = config["model"].get("dec_heads"),
#             dec_ff_mult = config["model"].get("dec_ff_mult"),
#             dec_dim_head = config["model"].get("dec_dim_head"),
#             dec_emb_dropout = config["model"].get("dec_emb_dropout"),
#             dec_ff_dropout = config["model"].get("dec_ff_dropout"),
#             dec_attn_dropout = config["model"].get("dec_attn_dropout")
#         )
# model = model.half()
# model.load_state_dict(torch.load('../save/2024-10-25_rna2atac_pan_cancer_5/pytorch_model.bin'))
# device = torch.device('cuda:7')
# model.to(device)
# model.eval()

# naive_data = torch.load("./naive_pt_20/naive_0.pt")
# dataset = PreDataset(naive_data)
# dataloader_kwargs = {'batch_size': 1, 'shuffle': False}
# loader = torch.utils.data.DataLoader(dataset, **dataloader_kwargs)

# # time: 1.5 min
# i=0
# for inputs in tqdm(loader, ncols=80, desc='output attention matrix'):
#     rna_sequence, rna_value, atac_sequence, _, enc_pad_mask = [each.to(device) for each in inputs]
#     attn = model.generate_attn_weight(rna_sequence, atac_sequence, rna_value, enc_mask=enc_pad_mask, which='decoder')
#     attn = attn[0]  # attn = attn[0].to(torch.float16)
#     with h5py.File('attn/attn_naive_'+str(i)+'.h5', 'w') as f:
#         f.create_dataset('attn', data=attn)
#     torch.cuda.empty_cache()
#     i += 1

# rna_naive_20 = sc.read_h5ad('rna_naive_20.h5ad')
# nonzero = np.count_nonzero(rna_naive_20.X.toarray(), axis=0)
# gene_lst = rna_naive_20.var.index[nonzero>=3]  # 2693

# atac_naive_20 = sc.read_h5ad('atac_naive_20.h5ad')

# gene_attn = {}
# for gene in gene_lst:
#     gene_attn[gene] = np.zeros([atac_naive_20.shape[1], 1], dtype='float32')  # not float16

# for i in range(20):
#     rna_sequence = naive_data[0][i].flatten()
#     with h5py.File('attn/attn_naive_'+str(i)+'.h5', 'r') as f:
#         attn = f['attn'][:]
#         for gene in tqdm(gene_lst, ncols=80, desc='cell '+str(i)):
#             idx = torch.argwhere((rna_sequence==(np.argwhere(rna_naive_20.var.index==gene))[0][0]+1).flatten())
#             if len(idx)!=0:
#                 gene_attn[gene] += attn[:, [idx.item()]]

# with open('./attn/attn_20_naive_3cell.pkl', 'wb') as file:
#     pickle.dump(gene_attn, file)

# gene_attn_sum = [gene_attn[gene].sum() for gene in gene_lst]
# df = pd.DataFrame({'gene':gene_lst, 'attn':gene_attn_sum})
# df.to_csv('./attn/attn_no_norm_cnt_20_naive_3cell.txt', sep='\t', header=None, index=None)

# df['attn'].sum()   # 5300202.5


# #################################### effect T ####################################
# rna_naive_effect_ex = sc.read_h5ad('res_cd8_t_naive_effect_ex.h5ad')
# random.seed(0)
# effect_idx = list(np.argwhere(rna_naive_effect_ex.obs['cell_anno']=='effect').flatten())
# effect_idx_20 = random.sample(list(effect_idx), 20)
# out = sc.AnnData(rna_naive_effect_ex[effect_idx_20].raw.X, obs=rna_naive_effect_ex[effect_idx_20].obs, var=rna_naive_effect_ex[effect_idx_20].raw.var)
# np.count_nonzero(out.X.toarray(), axis=1).max()  # 1345
# out.write('rna_effect_20.h5ad')

# atac = sc.read_h5ad('../atac.h5ad')
# atac_effect = atac[rna_naive_effect_ex.obs.index[effect_idx_20]].copy()
# atac_effect.write('atac_effect_20.h5ad')

# # python data_preprocess.py -r rna_effect_20.h5ad -a atac_effect_20.h5ad -s effect_pt_20 --dt test -n effect --config rna2atac_config_test_effect.yaml   # enc_max_len: 1345 

# with open("rna2atac_config_test_effect.yaml", "r") as f:
#     config = yaml.safe_load(f)

# model = M2M_rna2atac(
#             dim = config["model"].get("dim"),

#             enc_num_gene_tokens = config["model"].get("total_gene") + 1, # +1 for <PAD>
#             enc_num_value_tokens = config["model"].get('max_express') + 1, # +1 for <PAD>
#             enc_depth = config["model"].get("enc_depth"),
#             enc_heads = config["model"].get("enc_heads"),
#             enc_ff_mult = config["model"].get("enc_ff_mult"),
#             enc_dim_head = config["model"].get("enc_dim_head"),
#             enc_emb_dropout = config["model"].get("enc_emb_dropout"),
#             enc_ff_dropout = config["model"].get("enc_ff_dropout"),
#             enc_attn_dropout = config["model"].get("enc_attn_dropout"),

#             dec_depth = config["model"].get("dec_depth"),
#             dec_heads = config["model"].get("dec_heads"),
#             dec_ff_mult = config["model"].get("dec_ff_mult"),
#             dec_dim_head = config["model"].get("dec_dim_head"),
#             dec_emb_dropout = config["model"].get("dec_emb_dropout"),
#             dec_ff_dropout = config["model"].get("dec_ff_dropout"),
#             dec_attn_dropout = config["model"].get("dec_attn_dropout")
#         )
# model = model.half()
# model.load_state_dict(torch.load('../save/2024-10-25_rna2atac_pan_cancer_5/pytorch_model.bin'))
# device = torch.device('cuda:7')
# model.to(device)
# model.eval()

# effect_data = torch.load("./effect_pt_20/effect_0.pt")
# dataset = PreDataset(effect_data)
# dataloader_kwargs = {'batch_size': 1, 'shuffle': False}
# loader = torch.utils.data.DataLoader(dataset, **dataloader_kwargs)

# # time: 1.5 min
# i=0
# for inputs in tqdm(loader, ncols=80, desc='output attention matrix'):
#     rna_sequence, rna_value, atac_sequence, _, enc_pad_mask = [each.to(device) for each in inputs]
#     attn = model.generate_attn_weight(rna_sequence, atac_sequence, rna_value, enc_mask=enc_pad_mask, which='decoder')
#     attn = attn[0]  # attn = attn[0].to(torch.float16)
#     with h5py.File('attn/attn_effect_'+str(i)+'.h5', 'w') as f:
#         f.create_dataset('attn', data=attn)
#     torch.cuda.empty_cache()
#     i += 1

# rna_effect_20 = sc.read_h5ad('rna_effect_20.h5ad')
# nonzero = np.count_nonzero(rna_effect_20.X.toarray(), axis=0)
# gene_lst = rna_effect_20.var.index[nonzero>=3]  # 2368

# atac_effect_20 = sc.read_h5ad('atac_effect_20.h5ad')

# gene_attn = {}
# for gene in gene_lst:
#     gene_attn[gene] = np.zeros([atac_effect_20.shape[1], 1], dtype='float32')  # not float16

# for i in range(20):
#     rna_sequence = effect_data[0][i].flatten()
#     with h5py.File('attn/attn_effect_'+str(i)+'.h5', 'r') as f:
#         attn = f['attn'][:]
#         for gene in tqdm(gene_lst, ncols=80, desc='cell '+str(i)):
#             idx = torch.argwhere((rna_sequence==(np.argwhere(rna_effect_20.var.index==gene))[0][0]+1).flatten())
#             if len(idx)!=0:
#                 gene_attn[gene] += attn[:, [idx.item()]]

# with open('./attn/attn_20_effect_3cell.pkl', 'wb') as file:
#     pickle.dump(gene_attn, file)

# gene_attn_sum = [gene_attn[gene].sum() for gene in gene_lst]
# df = pd.DataFrame({'gene':gene_lst, 'attn':gene_attn_sum})
# df.to_csv('./attn/attn_no_norm_cnt_20_effect_3cell.txt', sep='\t', header=None, index=None)

# df['attn'].sum()   # 5207205.0


# #################################### ex T ####################################
# rna_naive_effect_ex = sc.read_h5ad('res_cd8_t_naive_effect_ex.h5ad')
# random.seed(0)
# ex_idx = list(np.argwhere(rna_naive_effect_ex.obs['cell_anno']=='ex').flatten())
# ex_idx_20 = random.sample(list(ex_idx), 20)
# out = sc.AnnData(rna_naive_effect_ex[ex_idx_20].raw.X, obs=rna_naive_effect_ex[ex_idx_20].obs, var=rna_naive_effect_ex[ex_idx_20].raw.var)
# np.count_nonzero(out.X.toarray(), axis=1).max()  # 2152
# out.write('rna_ex_20.h5ad')

# atac = sc.read_h5ad('../atac.h5ad')
# atac_ex = atac[rna_naive_effect_ex.obs.index[ex_idx_20]].copy()
# atac_ex.write('atac_ex_20.h5ad')

# # python data_preprocess.py -r rna_ex_20.h5ad -a atac_ex_20.h5ad -s ex_pt_20 --dt test -n ex --config rna2atac_config_test_ex.yaml   # enc_max_len: 2152 

# with open("rna2atac_config_test_ex.yaml", "r") as f:
#     config = yaml.safe_load(f)

# model = M2M_rna2atac(
#             dim = config["model"].get("dim"),

#             enc_num_gene_tokens = config["model"].get("total_gene") + 1, # +1 for <PAD>
#             enc_num_value_tokens = config["model"].get('max_express') + 1, # +1 for <PAD>
#             enc_depth = config["model"].get("enc_depth"),
#             enc_heads = config["model"].get("enc_heads"),
#             enc_ff_mult = config["model"].get("enc_ff_mult"),
#             enc_dim_head = config["model"].get("enc_dim_head"),
#             enc_emb_dropout = config["model"].get("enc_emb_dropout"),
#             enc_ff_dropout = config["model"].get("enc_ff_dropout"),
#             enc_attn_dropout = config["model"].get("enc_attn_dropout"),

#             dec_depth = config["model"].get("dec_depth"),
#             dec_heads = config["model"].get("dec_heads"),
#             dec_ff_mult = config["model"].get("dec_ff_mult"),
#             dec_dim_head = config["model"].get("dec_dim_head"),
#             dec_emb_dropout = config["model"].get("dec_emb_dropout"),
#             dec_ff_dropout = config["model"].get("dec_ff_dropout"),
#             dec_attn_dropout = config["model"].get("dec_attn_dropout")
#         )
# model = model.half()
# model.load_state_dict(torch.load('../save/2024-10-25_rna2atac_pan_cancer_5/pytorch_model.bin'))
# device = torch.device('cuda:7')
# model.to(device)
# model.eval()

# ex_data = torch.load("./ex_pt_20/ex_0.pt")
# dataset = PreDataset(ex_data)
# dataloader_kwargs = {'batch_size': 1, 'shuffle': False}
# loader = torch.utils.data.DataLoader(dataset, **dataloader_kwargs)

# # time: 1.5 min
# i=0
# for inputs in tqdm(loader, ncols=80, desc='output attention matrix'):
#     rna_sequence, rna_value, atac_sequence, _, enc_pad_mask = [each.to(device) for each in inputs]
#     attn = model.generate_attn_weight(rna_sequence, atac_sequence, rna_value, enc_mask=enc_pad_mask, which='decoder')
#     attn = attn[0]  # attn = attn[0].to(torch.float16)
#     with h5py.File('attn/attn_ex_'+str(i)+'.h5', 'w') as f:
#         f.create_dataset('attn', data=attn)
#     torch.cuda.empty_cache()
#     i += 1

# rna_ex_20 = sc.read_h5ad('rna_ex_20.h5ad')
# nonzero = np.count_nonzero(rna_ex_20.X.toarray(), axis=0)
# gene_lst = rna_ex_20.var.index[nonzero>=3]  # 2772

# atac_ex_20 = sc.read_h5ad('atac_ex_20.h5ad')

# gene_attn = {}
# for gene in gene_lst:
#     gene_attn[gene] = np.zeros([atac_ex_20.shape[1], 1], dtype='float32')  # not float16

# for i in range(20):
#     rna_sequence = ex_data[0][i].flatten()
#     with h5py.File('attn/attn_ex_'+str(i)+'.h5', 'r') as f:
#         attn = f['attn'][:]
#         for gene in tqdm(gene_lst, ncols=80, desc='cell '+str(i)):
#             idx = torch.argwhere((rna_sequence==(np.argwhere(rna_ex_20.var.index==gene))[0][0]+1).flatten())
#             if len(idx)!=0:
#                 gene_attn[gene] += attn[:, [idx.item()]]

# with open('./attn/attn_20_ex_3cell.pkl', 'wb') as file:
#     pickle.dump(gene_attn, file)

# gene_attn_sum = [gene_attn[gene].sum() for gene in gene_lst]
# df = pd.DataFrame({'gene':gene_lst, 'attn':gene_attn_sum})
# df.to_csv('./attn/attn_no_norm_cnt_20_ex_3cell.txt', sep='\t', header=None, index=None)

# df['attn'].sum()   # 5167448.0



# ###### rank genes based on attention score
# import pandas as pd
# import numpy as np
# from plotnine import *
# import matplotlib.pyplot as plt
# import seaborn as sns

# plt.rcParams['pdf.fonttype'] = 42

# cr = ['SMARCA4', 'SMARCA2', 'ARID1A', 'ARID1B', 'SMARCB1',
#       'CHD1', 'CHD2', 'CHD3', 'CHD4', 'CHD5', 'CHD6', 'CHD7', 'CHD8', 'CHD9',
#       'BRD2', 'BRD3', 'BRD4', 'BRDT',
#       'SMARCA5', 'SMARCA1', 'ACTL6A', 'ACTL6B',
#       'SSRP1', 'SUPT16H',
#       'EP400',
#       'SMARCD1', 'SMARCD2', 'SMARCD3']   # 28
# tf_lst = pd.read_table('TF_jaspar.txt', header=None)
# tf = list(tf_lst[0].values)   # 735
# cr_tf = cr + tf   # 763

# df_naive = pd.read_csv('attn_no_norm_cnt_20_naive_3cell.txt', header=None, sep='\t')
# df_naive.columns = ['gene', 'attn']
# df_naive['attn_norm'] = df_naive['attn']/df_naive['attn'].max()

# df_effect = pd.read_csv('attn_no_norm_cnt_20_effect_3cell.txt', header=None, sep='\t')
# df_effect.columns = ['gene', 'attn']
# df_effect['attn_norm'] = df_effect['attn']/df_effect['attn'].max()

# df_ex = pd.read_csv('attn_no_norm_cnt_20_ex_3cell.txt', header=None, sep='\t')
# df_ex.columns = ['gene', 'attn']
# df_ex['attn_norm'] = df_ex['attn']/df_ex['attn'].max()

# sum(df_naive['attn']/df_naive['attn'].max())
# # 13.304571276765705

# sum(df_effect['attn']/df_effect['attn'].max())
# # 14.894872854946211

# sum(df_ex['attn']/df_ex['attn'].max())
# # 9.518939784170293

# df_naive[df_naive['gene'].isin(cr_tf)]['attn_norm'].sum()
# # 0.6764622481005079

# df_effect[df_effect['gene'].isin(cr_tf)]['attn_norm'].sum()
# # 1.3884113822466762

# df_ex[df_ex['gene'].isin(cr_tf)]['attn_norm'].sum()
# # 0.935870134228533

# dat = pd.DataFrame({'idx':['naive', 'effect', 'ex'],
#                     'val':[0.676, 1.388, 0.936]})
# dat['idx'] = pd.Categorical(dat['idx'], categories=['naive', 'effect', 'ex'])

# p = ggplot(dat, aes(x='idx', y='val', fill='idx')) + geom_col(position='dodge') +\
#                                                      scale_y_continuous(limits=[0, 1.5], breaks=np.arange(0, 1.5+0.5, 0.5)) + theme_bw()  # limits min must set to 0
# p.save(filename='cd8_t_naive_effect_ex_reg_score.pdf', dpi=600, height=4, width=4)

# ## heatmap of factors (may be retrain the model)
# df_naive_effect = pd.merge(df_naive[['gene', 'attn_norm']], df_effect[['gene', 'attn_norm']], how='outer', on='gene')
# df_naive_effect_ex = pd.merge(df_naive_effect, df_ex[['gene', 'attn_norm']], how='outer')
# df_naive_effect_ex.fillna(0, inplace=True)
# df_naive_effect_ex.rename(columns={'attn_norm_x':'attn_norm_naive', 'attn_norm_y':'attn_norm_effect', 'attn_norm':'attn_norm_ex'}, inplace=True)

# df_naive_effect_ex_tf = df_naive_effect_ex[df_naive_effect_ex['gene'].isin(cr_tf)]

# naive_sp = df_naive_effect_ex_tf[(df_naive_effect_ex_tf['attn_norm_naive']>df_naive_effect_ex_tf['attn_norm_effect']) & 
#                                  (df_naive_effect_ex_tf['attn_norm_naive']>df_naive_effect_ex_tf['attn_norm_ex'])]
# effect_sp = df_naive_effect_ex_tf[(df_naive_effect_ex_tf['attn_norm_effect']>df_naive_effect_ex_tf['attn_norm_naive']) & 
#                                   (df_naive_effect_ex_tf['attn_norm_effect']>df_naive_effect_ex_tf['attn_norm_ex'])]
# ex_sp = df_naive_effect_ex_tf[(df_naive_effect_ex_tf['attn_norm_ex']>df_naive_effect_ex_tf['attn_norm_naive']) & 
#                               (df_naive_effect_ex_tf['attn_norm_ex']>df_naive_effect_ex_tf['attn_norm_effect'])]
# sp = pd.concat([naive_sp, effect_sp, ex_sp])
# sp.index = sp['gene'].values

# sns.clustermap(sp[['attn_norm_naive', 'attn_norm_effect', 'attn_norm_ex']],
#                cmap='coolwarm', row_cluster=False, col_cluster=False, z_score=0, vmin=-1.2, vmax=1.2, figsize=(5, 40)) 
# plt.savefig('cd8_t_naive_effect_ex_specific_factor_heatmap.pdf')
# plt.close()
