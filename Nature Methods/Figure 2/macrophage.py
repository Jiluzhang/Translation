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

sc.tl.leiden(macro, resolution=0.95)
macro.obs['leiden'].value_counts()
# 0     7312
# 1     6651
# 2     6227
# 3     5120
# 4     4459
# 5     4273
# 6     4271
# 7     4135
# 8     2724
# 9     2582
# 10    1970
# 11     667
# 12     472
# 13     324
# 14     301
# 15     278
# 16     264

# Mono_FCN1 -> Macro_IL32  -> Macro_C1QC
sc.pl.matrixplot(macro, ['FCN1', 'VCAN',
                         'IL32',
                         'APOE', 'C1QC', 'SPP1', 'CDC20', 'FCGR3A', 'C1QA', 'C1QB'],
                 dendrogram=True, groupby='leiden', cmap='coolwarm', standard_scale='var', save='leiden_heatmap_macro.pdf')
# 7: Mono_FCN1
# 6: Macro_IL32
# 13: Macro_C1QC

sc.pl.umap(macro, color='leiden', legend_fontsize='5', legend_loc='right margin', size=5,
           title='', frameon=True, save='_leiden_macro.pdf')

macro.write('res_macro.h5ad')

## macro_fcn1_il32_c1qc
macro_fcn1_il32_c1qc = macro[macro.obs['leiden'].isin(['13', '6', '7'])].copy()  # 8730 × 2108

macro_fcn1_il32_c1qc.obs['cell_anno'] = macro_fcn1_il32_c1qc.obs['leiden']
macro_fcn1_il32_c1qc.obs['cell_anno'].replace({'13':'Macro_C1QC', '6':'Macro_IL32', '7':'Mono_FCN1'}, inplace=True)

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



######################################################################## Mono_FCN1 ########################################################################
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

rna_macro_fcn1_il32_c1qc = sc.read_h5ad('res_macro_fcn1_il32_c1qc.h5ad')
random.seed(0)
fcn1_idx = list(np.argwhere(rna_macro_fcn1_il32_c1qc.obs['cell_anno']=='Mono_FCN1').flatten())
fcn1_idx_100 = random.sample(list(fcn1_idx), 100)
out = sc.AnnData(rna_macro_fcn1_il32_c1qc[fcn1_idx_100].raw.X, obs=rna_macro_fcn1_il32_c1qc[fcn1_idx_100].obs, var=rna_macro_fcn1_il32_c1qc[fcn1_idx_100].raw.var)
np.count_nonzero(out.X.toarray(), axis=1).max()  # 3414
out.write('rna_fcn1_100.h5ad')

atac = sc.read_h5ad('../atac.h5ad')
atac_fcn1 = atac[rna_macro_fcn1_il32_c1qc.obs.index[fcn1_idx_100]].copy()
atac_fcn1.write('atac_fcn1_100.h5ad')

# python data_preprocess.py -r rna_fcn1_100.h5ad -a atac_fcn1_100.h5ad -s fcn1_pt_100 --dt test -n fcn1 --config rna2atac_config_test_fcn1.yaml   # enc_max_len: 4051 

with open("rna2atac_config_test_fcn1.yaml", "r") as f:
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

fcn1_data = torch.load("./fcn1_pt_100/fcn1_0.pt")
dataset = PreDataset(fcn1_data)
dataloader_kwargs = {'batch_size': 1, 'shuffle': False}
loader = torch.utils.data.DataLoader(dataset, **dataloader_kwargs)

rna_fcn1 = sc.read_h5ad('rna_fcn1_100.h5ad')
atac_fcn1 = sc.read_h5ad('atac_fcn1_100.h5ad')
out = rna_fcn1.copy()
out.X = np.zeros(out.shape)
i = 0
for inputs in tqdm(loader, ncols=80, desc='output attention matrix'):
    rna_sequence, rna_value, atac_sequence, _, enc_pad_mask = [each.to(device) for each in inputs]
    out.X[i, (rna_sequence[rna_sequence!=0]-1).cpu()] = model.generate_attn_weight(rna_sequence, atac_sequence, rna_value, enc_mask=enc_pad_mask, which='decoder')[0].sum(axis=0)
    torch.cuda.empty_cache()
    i += 1

out.write('attn_fcn1_100.h5ad')

attn = sc.read_h5ad('attn_fcn1_100.h5ad')
(attn.X / attn.X.max(axis=1)[:, np.newaxis]).sum(axis=1).mean()  # 6.3104026102418
sum(attn.X.sum(axis=0) / (attn.X.sum(axis=0).max()))             # 18.398807684808443

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

(attn.X.sum(axis=0) / (attn.X.sum(axis=0).max()))[attn.var.index.isin(cr_tf)].sum()              # 0.6027194894236233


######################################################################## Macro_IL32 ########################################################################
rna_macro_fcn1_il32_c1qc = sc.read_h5ad('res_macro_fcn1_il32_c1qc.h5ad')
random.seed(0)
il32_idx = list(np.argwhere(rna_macro_fcn1_il32_c1qc.obs['cell_anno']=='Macro_IL32').flatten())
il32_idx_100 = random.sample(list(il32_idx), 100)
out = sc.AnnData(rna_macro_fcn1_il32_c1qc[il32_idx_100].raw.X, obs=rna_macro_fcn1_il32_c1qc[il32_idx_100].obs, var=rna_macro_fcn1_il32_c1qc[il32_idx_100].raw.var)
np.count_nonzero(out.X.toarray(), axis=1).max()  # 4670
out.write('rna_il32_100.h5ad')

atac = sc.read_h5ad('../atac.h5ad')
atac_il32 = atac[rna_macro_fcn1_il32_c1qc.obs.index[il32_idx_100]].copy()
atac_il32.write('atac_il32_100.h5ad')

# python data_preprocess.py -r rna_il32_100.h5ad -a atac_il32_100.h5ad -s il32_pt_100 --dt test -n il32 --config rna2atac_config_test_il32.yaml   # enc_max_len: 4670 

with open("rna2atac_config_test_il32.yaml", "r") as f:
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

il32_data = torch.load("./il32_pt_100/il32_0.pt")
dataset = PreDataset(il32_data)
dataloader_kwargs = {'batch_size': 1, 'shuffle': False}
loader = torch.utils.data.DataLoader(dataset, **dataloader_kwargs)

rna_il32 = sc.read_h5ad('rna_il32_100.h5ad')
atac_il32 = sc.read_h5ad('atac_il32_100.h5ad')
out = rna_il32.copy()
out.X = np.zeros(out.shape)
i = 0
for inputs in tqdm(loader, ncols=80, desc='output attention matrix'):
    rna_sequence, rna_value, atac_sequence, _, enc_pad_mask = [each.to(device) for each in inputs]
    out.X[i, (rna_sequence[rna_sequence!=0]-1).cpu()] = model.generate_attn_weight(rna_sequence, atac_sequence, rna_value, enc_mask=enc_pad_mask, which='decoder')[0].sum(axis=0)
    torch.cuda.empty_cache()
    i += 1

out.write('attn_il32_100.h5ad')

attn = sc.read_h5ad('attn_il32_100.h5ad')
(attn.X / attn.X.max(axis=1)[:, np.newaxis]).sum(axis=1).mean()  # 6.63963537231081
sum(attn.X.sum(axis=0) / (attn.X.sum(axis=0).max()))             # 20.680930198354

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

(attn.X.sum(axis=0) / (attn.X.sum(axis=0).max()))[attn.var.index.isin(cr_tf)].sum()              # 0.7982198998103225


######################################################################## Macro_C1QC ########################################################################
rna_macro_fcn1_il32_c1qc = sc.read_h5ad('res_macro_fcn1_il32_c1qc.h5ad')
random.seed(0)
c1qc_idx = list(np.argwhere(rna_macro_fcn1_il32_c1qc.obs['cell_anno']=='Macro_C1QC').flatten())
c1qc_idx_100 = random.sample(list(c1qc_idx), 100)
out = sc.AnnData(rna_macro_fcn1_il32_c1qc[c1qc_idx_100].raw.X, obs=rna_macro_fcn1_il32_c1qc[c1qc_idx_100].obs, var=rna_macro_fcn1_il32_c1qc[c1qc_idx_100].raw.var)
np.count_nonzero(out.X.toarray(), axis=1).max()  # 4765
out.write('rna_c1qc_100.h5ad')

atac = sc.read_h5ad('../atac.h5ad')
atac_c1qc = atac[rna_macro_fcn1_il32_c1qc.obs.index[c1qc_idx_100]].copy()
atac_c1qc.write('atac_c1qc_100.h5ad')

# python data_preprocess.py -r rna_c1qc_100.h5ad -a atac_c1qc_100.h5ad -s c1qc_pt_100 --dt test -n c1qc --config rna2atac_config_test_c1qc.yaml   # enc_max_len: 4765 

with open("rna2atac_config_test_c1qc.yaml", "r") as f:
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

c1qc_data = torch.load("./c1qc_pt_100/c1qc_0.pt")
dataset = PreDataset(c1qc_data)
dataloader_kwargs = {'batch_size': 1, 'shuffle': False}
loader = torch.utils.data.DataLoader(dataset, **dataloader_kwargs)

rna_c1qc = sc.read_h5ad('rna_c1qc_100.h5ad')
atac_c1qc = sc.read_h5ad('atac_c1qc_100.h5ad')
out = rna_c1qc.copy()
out.X = np.zeros(out.shape)
i = 0
for inputs in tqdm(loader, ncols=80, desc='output attention matrix'):
    rna_sequence, rna_value, atac_sequence, _, enc_pad_mask = [each.to(device) for each in inputs]
    out.X[i, (rna_sequence[rna_sequence!=0]-1).cpu()] = model.generate_attn_weight(rna_sequence, atac_sequence, rna_value, enc_mask=enc_pad_mask, which='decoder')[0].sum(axis=0)
    torch.cuda.empty_cache()
    i += 1

out.write('attn_c1qc_100.h5ad')

attn = sc.read_h5ad('attn_c1qc_100.h5ad')
(attn.X / attn.X.max(axis=1)[:, np.newaxis]).sum(axis=1).mean()  # 6.291713555490276
sum(attn.X.sum(axis=0) / (attn.X.sum(axis=0).max()))             # 10.961686286506977

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

(attn.X.sum(axis=0) / (attn.X.sum(axis=0).max()))[attn.var.index.isin(cr_tf)].sum()   # 0.09610137466087999


# ## Attention score comparison during macrophage differentiation process
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

# #################################### Mono_FCN1 ####################################
# rna_macro_fcn1_il32_c1qc = sc.read_h5ad('res_macro_fcn1_il32_c1qc.h5ad')
# random.seed(0)
# mono_fcn1_idx = list(np.argwhere(rna_macro_fcn1_il32_c1qc.obs['cell_anno']=='Mono_FCN1').flatten())
# mono_fcn1_idx_20 = random.sample(list(mono_fcn1_idx), 20)
# out = sc.AnnData(rna_macro_fcn1_il32_c1qc[mono_fcn1_idx_20].raw.X, obs=rna_macro_fcn1_il32_c1qc[mono_fcn1_idx_20].obs, var=rna_macro_fcn1_il32_c1qc[mono_fcn1_idx_20].raw.var)
# np.count_nonzero(out.X.toarray(), axis=1).max()  # 3414
# out.write('rna_mono_fcn1_20.h5ad')

# atac = sc.read_h5ad('../atac.h5ad')
# atac_mono_fcn1 = atac[rna_macro_fcn1_il32_c1qc.obs.index[mono_fcn1_idx_20]].copy()
# atac_mono_fcn1.write('atac_mono_fcn1_20.h5ad')

# # python data_preprocess.py -r rna_mono_fcn1_20.h5ad -a atac_mono_fcn1_20.h5ad -s mono_fcn1_pt_20 --dt test -n mono_fcn1 --config rna2atac_config_test_mono_fcn1.yaml   # enc_max_len: 3414 

# with open("rna2atac_config_test_mono_fcn1.yaml", "r") as f:
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
# device = torch.device('cuda:2')
# model.to(device)
# model.eval()

# mono_fcn1_data = torch.load("./mono_fcn1_pt_20/mono_fcn1_0.pt")
# dataset = PreDataset(mono_fcn1_data)
# dataloader_kwargs = {'batch_size': 1, 'shuffle': False}
# loader = torch.utils.data.DataLoader(dataset, **dataloader_kwargs)

# # time: 1.5 min
# i=0
# for inputs in tqdm(loader, ncols=80, desc='output attention matrix'):
#     rna_sequence, rna_value, atac_sequence, _, enc_pad_mask = [each.to(device) for each in inputs]
#     attn = model.generate_attn_weight(rna_sequence, atac_sequence, rna_value, enc_mask=enc_pad_mask, which='decoder')
#     attn = attn[0]  # attn = attn[0].to(torch.float16)
#     with h5py.File('attn/attn_mono_fcn1_'+str(i)+'.h5', 'w') as f:
#         f.create_dataset('attn', data=attn)
#     torch.cuda.empty_cache()
#     i += 1

# rna_mono_fcn1_20 = sc.read_h5ad('rna_mono_fcn1_20.h5ad')
# nonzero = np.count_nonzero(rna_mono_fcn1_20.X.toarray(), axis=0)
# gene_lst = rna_mono_fcn1_20.var.index[nonzero>=3]  # 4476

# atac_mono_fcn1_20 = sc.read_h5ad('atac_mono_fcn1_20.h5ad')

# gene_attn = {}
# for gene in gene_lst:
#     gene_attn[gene] = np.zeros([atac_mono_fcn1_20.shape[1], 1], dtype='float32')  # not float16

# for i in range(20):
#     rna_sequence = mono_fcn1_data[0][i].flatten()
#     with h5py.File('attn/attn_mono_fcn1_'+str(i)+'.h5', 'r') as f:
#         attn = f['attn'][:]
#         for gene in tqdm(gene_lst, ncols=80, desc='cell '+str(i)):
#             idx = torch.argwhere((rna_sequence==(np.argwhere(rna_mono_fcn1_20.var.index==gene))[0][0]+1).flatten())
#             if len(idx)!=0:
#                 gene_attn[gene] += attn[:, [idx.item()]]

# with open('./attn/attn_20_mono_fcn1_3cell.pkl', 'wb') as file:
#     pickle.dump(gene_attn, file)

# gene_attn_sum = [gene_attn[gene].sum() for gene in gene_lst]
# df = pd.DataFrame({'gene':gene_lst, 'attn':gene_attn_sum})
# df.to_csv('./attn/attn_no_norm_cnt_20_mono_fcn1_3cell.txt', sep='\t', header=None, index=None)

# df['attn'].sum()   # 5749281.0


# #################################### Macro_IL32 ####################################
# rna_macro_fcn1_il32_c1qc = sc.read_h5ad('res_macro_fcn1_il32_c1qc.h5ad')
# random.seed(0)
# macro_il32_idx = list(np.argwhere(rna_macro_fcn1_il32_c1qc.obs['cell_anno']=='Macro_IL32').flatten())
# macro_il32_idx_20 = random.sample(list(macro_il32_idx), 20)
# out = sc.AnnData(rna_macro_fcn1_il32_c1qc[macro_il32_idx_20].raw.X, obs=rna_macro_fcn1_il32_c1qc[macro_il32_idx_20].obs, var=rna_macro_fcn1_il32_c1qc[macro_il32_idx_20].raw.var)
# np.count_nonzero(out.X.toarray(), axis=1).max()  # 2515
# out.write('rna_macro_il32_20.h5ad')

# atac = sc.read_h5ad('../atac.h5ad')
# atac_macro_il32 = atac[rna_macro_fcn1_il32_c1qc.obs.index[macro_il32_idx_20]].copy()
# atac_macro_il32.write('atac_macro_il32_20.h5ad')

# # python data_preprocess.py -r rna_macro_il32_20.h5ad -a atac_macro_il32_20.h5ad -s macro_il32_pt_20 --dt test -n macro_il32 --config rna2atac_config_test_macro_il32.yaml   # enc_max_len: 2515 

# with open("rna2atac_config_test_macro_il32.yaml", "r") as f:
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
# device = torch.device('cuda:2')
# model.to(device)
# model.eval()

# macro_il32_data = torch.load("./macro_il32_pt_20/macro_il32_0.pt")
# dataset = PreDataset(macro_il32_data)
# dataloader_kwargs = {'batch_size': 1, 'shuffle': False}
# loader = torch.utils.data.DataLoader(dataset, **dataloader_kwargs)

# # time: 1.5 min
# i=0
# for inputs in tqdm(loader, ncols=80, desc='output attention matrix'):
#     rna_sequence, rna_value, atac_sequence, _, enc_pad_mask = [each.to(device) for each in inputs]
#     attn = model.generate_attn_weight(rna_sequence, atac_sequence, rna_value, enc_mask=enc_pad_mask, which='decoder')
#     attn = attn[0]  # attn = attn[0].to(torch.float16)
#     with h5py.File('attn/attn_macro_il32_'+str(i)+'.h5', 'w') as f:
#         f.create_dataset('attn', data=attn)
#     torch.cuda.empty_cache()
#     i += 1

# rna_macro_il32_20 = sc.read_h5ad('rna_macro_il32_20.h5ad')
# nonzero = np.count_nonzero(rna_macro_il32_20.X.toarray(), axis=0)
# gene_lst = rna_macro_il32_20.var.index[nonzero>=3]  # 3743

# atac_macro_il32_20 = sc.read_h5ad('atac_macro_il32_20.h5ad')

# gene_attn = {}
# for gene in gene_lst:
#     gene_attn[gene] = np.zeros([atac_macro_il32_20.shape[1], 1], dtype='float32')  # not float16

# for i in range(20):
#     rna_sequence = macro_il32_data[0][i].flatten()
#     with h5py.File('attn/attn_macro_il32_'+str(i)+'.h5', 'r') as f:
#         attn = f['attn'][:]
#         for gene in tqdm(gene_lst, ncols=80, desc='cell '+str(i)):
#             idx = torch.argwhere((rna_sequence==(np.argwhere(rna_macro_il32_20.var.index==gene))[0][0]+1).flatten())
#             if len(idx)!=0:
#                 gene_attn[gene] += attn[:, [idx.item()]]

# with open('./attn/attn_20_macro_il32_3cell.pkl', 'wb') as file:
#     pickle.dump(gene_attn, file)

# gene_attn_sum = [gene_attn[gene].sum() for gene in gene_lst]
# df = pd.DataFrame({'gene':gene_lst, 'attn':gene_attn_sum})
# df.to_csv('./attn/attn_no_norm_cnt_20_macro_il32_3cell.txt', sep='\t', header=None, index=None)

# df['attn'].sum()   # 5723912.0


# #################################### Macro_C1QC ####################################
# rna_macro_fcn1_il32_c1qc = sc.read_h5ad('res_macro_fcn1_il32_c1qc.h5ad')
# random.seed(0)
# macro_c1qc_idx = list(np.argwhere(rna_macro_fcn1_il32_c1qc.obs['cell_anno']=='Macro_C1QC').flatten())
# macro_c1qc_idx_20 = random.sample(list(macro_c1qc_idx), 20)
# out = sc.AnnData(rna_macro_fcn1_il32_c1qc[macro_c1qc_idx_20].raw.X, obs=rna_macro_fcn1_il32_c1qc[macro_c1qc_idx_20].obs, var=rna_macro_fcn1_il32_c1qc[macro_c1qc_idx_20].raw.var)
# np.count_nonzero(out.X.toarray(), axis=1).max()  # 3439
# out.write('rna_macro_c1qc_20.h5ad')

# atac = sc.read_h5ad('../atac.h5ad')
# atac_macro_c1qc = atac[rna_macro_fcn1_il32_c1qc.obs.index[macro_c1qc_idx_20]].copy()
# atac_macro_c1qc.write('atac_macro_c1qc_20.h5ad')

# # python data_preprocess.py -r rna_macro_c1qc_20.h5ad -a atac_macro_c1qc_20.h5ad -s macro_c1qc_pt_20 --dt test -n macro_c1qc --config rna2atac_config_test_macro_c1qc.yaml   # enc_max_len: 3439 

# with open("rna2atac_config_test_macro_c1qc.yaml", "r") as f:
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
# device = torch.device('cuda:2')
# model.to(device)
# model.eval()

# macro_c1qc_data = torch.load("./macro_c1qc_pt_20/macro_c1qc_0.pt")
# dataset = PreDataset(macro_c1qc_data)
# dataloader_kwargs = {'batch_size': 1, 'shuffle': False}
# loader = torch.utils.data.DataLoader(dataset, **dataloader_kwargs)

# # time: 1.5 min
# i=0
# for inputs in tqdm(loader, ncols=80, desc='output attention matrix'):
#     rna_sequence, rna_value, atac_sequence, _, enc_pad_mask = [each.to(device) for each in inputs]
#     attn = model.generate_attn_weight(rna_sequence, atac_sequence, rna_value, enc_mask=enc_pad_mask, which='decoder')
#     attn = attn[0]  # attn = attn[0].to(torch.float16)
#     with h5py.File('attn/attn_macro_c1qc_'+str(i)+'.h5', 'w') as f:
#         f.create_dataset('attn', data=attn)
#     torch.cuda.empty_cache()
#     i += 1

# rna_macro_c1qc_20 = sc.read_h5ad('rna_macro_c1qc_20.h5ad')
# nonzero = np.count_nonzero(rna_macro_c1qc_20.X.toarray(), axis=0)
# gene_lst = rna_macro_c1qc_20.var.index[nonzero>=3]  # 5316

# atac_macro_c1qc_20 = sc.read_h5ad('atac_macro_c1qc_20.h5ad')

# gene_attn = {}
# for gene in gene_lst:
#     gene_attn[gene] = np.zeros([atac_macro_c1qc_20.shape[1], 1], dtype='float32')  # not float16

# for i in range(20):
#     rna_sequence = macro_c1qc_data[0][i].flatten()
#     with h5py.File('attn/attn_macro_c1qc_'+str(i)+'.h5', 'r') as f:
#         attn = f['attn'][:]
#         for gene in tqdm(gene_lst, ncols=80, desc='cell '+str(i)):
#             idx = torch.argwhere((rna_sequence==(np.argwhere(rna_macro_c1qc_20.var.index==gene))[0][0]+1).flatten())
#             if len(idx)!=0:
#                 gene_attn[gene] += attn[:, [idx.item()]]

# with open('./attn/attn_20_macro_c1qc_3cell.pkl', 'wb') as file:
#     pickle.dump(gene_attn, file)

# gene_attn_sum = [gene_attn[gene].sum() for gene in gene_lst]
# df = pd.DataFrame({'gene':gene_lst, 'attn':gene_attn_sum})
# df.to_csv('./attn/attn_no_norm_cnt_20_macro_c1qc_3cell.txt', sep='\t', header=None, index=None)

# df['attn'].sum()   # 6138694.0



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

# df_mono_fcn1 = pd.read_csv('attn_no_norm_cnt_20_mono_fcn1_3cell.txt', header=None, sep='\t')
# df_mono_fcn1.columns = ['gene', 'attn']
# df_mono_fcn1['attn_norm'] = df_mono_fcn1['attn']/df_mono_fcn1['attn'].max()

# df_macro_il32 = pd.read_csv('attn_no_norm_cnt_20_macro_il32_3cell.txt', header=None, sep='\t')
# df_macro_il32.columns = ['gene', 'attn']
# df_macro_il32['attn_norm'] = df_macro_il32['attn']/df_macro_il32['attn'].max()

# df_macro_c1qc = pd.read_csv('attn_no_norm_cnt_20_macro_c1qc_3cell.txt', header=None, sep='\t')
# df_macro_c1qc.columns = ['gene', 'attn']
# df_macro_c1qc['attn_norm'] = df_macro_c1qc['attn']/df_macro_c1qc['attn'].max()

# sum(df_mono_fcn1['attn']/df_mono_fcn1['attn'].max())
# # 10.954096368292666

# sum(df_macro_il32['attn']/df_macro_il32['attn'].max())
# # 19.590898300820562

# sum(df_macro_c1qc['attn']/df_macro_c1qc['attn'].max())
# # 9.15821913900683

# df_mono_fcn1[df_mono_fcn1['gene'].isin(cr_tf)]['attn_norm'].sum()
# # 0.411096969936237

# df_macro_il32[df_macro_il32['gene'].isin(cr_tf)]['attn_norm'].sum()
# # 1.2680660991411905

# df_macro_c1qc[df_macro_c1qc['gene'].isin(cr_tf)]['attn_norm'].sum()
# # 0.06947326793785928





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
