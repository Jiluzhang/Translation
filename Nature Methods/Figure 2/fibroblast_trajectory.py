## workdir: /fs/home/jiluzhang/Nature_methods/Figure_2/pan_cancer/cisformer/Fibroblast

import scanpy as sc
import harmonypy as hm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

plt.rcParams['pdf.fonttype'] = 42

rna = sc.read_h5ad('../rna.h5ad')   # 144409 × 19160
fibro = rna[rna.obs['cell_anno']=='Fibroblasts'].copy()  # 28023 × 19160

sc.pp.normalize_total(fibro, target_sum=1e4)
sc.pp.log1p(fibro)
sc.pp.highly_variable_genes(fibro, batch_key='cancer_type')
fibro.raw = fibro
fibro = fibro[:, fibro.var.highly_variable]   # 28023 × 2178
sc.tl.pca(fibro)
ho = hm.run_harmony(fibro.obsm['X_pca'], fibro.obs, 'cancer_type')  # Converged after 2 iterations

fibro.obsm['X_pca_harmony'] = ho.Z_corr.T
sc.pp.neighbors(fibro, use_rep='X_pca_harmony')
sc.tl.umap(fibro)

sc.tl.leiden(fibro, resolution=0.75)
fibro.obs['leiden'].value_counts()
# 0     4676
# 1     3795
# 2     3097
# 3     2539
# 4     2510
# 5     1592
# 6     1522
# 7     1493
# 8     1203
# 9     1101
# 10    1019
# 11     744
# 12     625
# 13     507
# 14     461
# 15     407
# 16     339
# 17     210
# 18     183

sc.pl.matrixplot(fibro, ['SAA1', 'CDC20',
                         'CCL5', 'PTPRC', 'CCL4',
                         'MYH11', 'ACTA2', 'CCL19', 'RGS5'],
                 dendrogram=True, groupby='leiden', cmap='coolwarm', standard_scale='var', save='leiden_heatmap_tcell.pdf')
# 11,15: qFibro
# 3: apFibro
# 8,12,14: MyoFibro

# only select one cluster
# 15: qFibro
# 3: apFibro
# 14: MyoFibro

sc.pl.umap(fibro, color='leiden', legend_fontsize='5', legend_loc='right margin', size=5,
           title='', frameon=True, save='_leiden_fibro.pdf')

fibro.write('res_fibro.h5ad')


## qFibro -> apFibro -> MyoFibro
# qFibro_apFibro_MyoFibro = fibro[fibro.obs['leiden'].isin(['3', '8', '11', '12', '14', '15'])].copy() 
qFibro_apFibro_MyoFibro = fibro[fibro.obs['leiden'].isin(['3', '14', '15'])].copy()  # 3407 × 2178

qFibro_apFibro_MyoFibro.obs['cell_anno'] = qFibro_apFibro_MyoFibro.obs['leiden']
# qFibro_apFibro_MyoFibro.obs['cell_anno'].replace({'3':'apFibro', '8':'MyoFibro', '11':'qFibro',
#                                                   '12':'MyoFibro', '14':'MyoFibro', '15':'qFibro'}, inplace=True)
qFibro_apFibro_MyoFibro.obs['cell_anno'].replace({'3':'apFibro', '14':'MyoFibro', '15':'qFibro'}, inplace=True)

sc.pl.umap(qFibro_apFibro_MyoFibro, color='cell_anno', legend_fontsize='5', legend_loc='right margin', size=5,
           title='', frameon=True, save='_cell_anno_qFibro_apFibro_MyoFibro.pdf')

sc.pl.umap(qFibro_apFibro_MyoFibro, color='leiden', legend_fontsize='5', legend_loc='right margin', size=5,
           title='', frameon=True, save='_leiden_qFibro_apFibro_MyoFibro.pdf')

qFibro_apFibro_MyoFibro.obs['cell_anno'] = pd.Categorical(qFibro_apFibro_MyoFibro.obs['cell_anno'], categories=['qFibro', 'apFibro', 'MyoFibro'])

sc.pl.matrixplot(qFibro_apFibro_MyoFibro, ['SAA1', 'CDC20',
                                           'CCL5', 'PTPRC', 'CCL4',
                                           'MYH11', 'ACTA2', 'CCL19', 'RGS5'],
                 groupby='cell_anno', cmap='coolwarm', standard_scale='var',
                 save='cell_anno_heatmap_qFibro_apFibro_MyoFibro.pdf')

sc.tl.paga(qFibro_apFibro_MyoFibro, groups='cell_anno')
sc.pl.paga(qFibro_apFibro_MyoFibro, layout='fa', color=['cell_anno'], save='_qFibro_apFibro_MyoFibro.pdf')

sc.tl.draw_graph(qFibro_apFibro_MyoFibro, init_pos='paga')
sc.pl.draw_graph(qFibro_apFibro_MyoFibro, color=['cell_anno'], legend_loc='on data', save='_qFibro_apFibro_MyoFibro_cell_anno.pdf')

qFibro_apFibro_MyoFibro.uns["iroot"] = np.flatnonzero(qFibro_apFibro_MyoFibro.obs["cell_anno"]=='qFibro')[0]
sc.tl.dpt(qFibro_apFibro_MyoFibro)
sc.pl.draw_graph(qFibro_apFibro_MyoFibro, color=['cell_anno', 'dpt_pseudotime'], vmax=0.5, legend_loc='on data', save='_qFibro_apFibro_MyoFibro_pseudo.pdf')

sc.pl.umap(qFibro_apFibro_MyoFibro, color='dpt_pseudotime', legend_fontsize='5', legend_loc='right margin', size=5, vmax=0.7, cmap='coolwarm',
           title='', frameon=True, save='_cell_anno_qFibro_apFibro_MyoFibro_pseudo.pdf')

qFibro_apFibro_MyoFibro.write('res_qFibro_apFibro_MyoFibro.h5ad')


######################################################################## qFibro ########################################################################
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

rna_qFibro_apFibro_MyoFibro = sc.read_h5ad('res_qFibro_apFibro_MyoFibro.h5ad')
random.seed(0)
qFibro_idx = list(np.argwhere(rna_qFibro_apFibro_MyoFibro.obs['cell_anno']=='qFibro').flatten())
qFibro_idx_100 = random.sample(list(qFibro_idx), 100)
out = sc.AnnData(rna_qFibro_apFibro_MyoFibro[qFibro_idx_100].raw.X, obs=rna_qFibro_apFibro_MyoFibro[qFibro_idx_100].obs, var=rna_qFibro_apFibro_MyoFibro[qFibro_idx_100].raw.var)
np.count_nonzero(out.X.toarray(), axis=1).max()  # 4051
out.write('rna_qFibro_100.h5ad')

atac = sc.read_h5ad('../atac.h5ad')
atac_qFibro = atac[rna_qFibro_apFibro_MyoFibro.obs.index[qFibro_idx_100]].copy()
atac_qFibro.write('atac_qFibro_100.h5ad')

# python data_preprocess.py -r rna_qFibro_100.h5ad -a atac_qFibro_100.h5ad -s qFibro_pt_100 --dt test -n qFibro --config rna2atac_config_test_qFibro.yaml   # enc_max_len: 4051 

with open("rna2atac_config_test_qFibro.yaml", "r") as f:
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
device = torch.device('cuda:6')
model.to(device)
model.eval()

qFibro_data = torch.load("./qFibro_pt_100/qFibro_0.pt")
dataset = PreDataset(qFibro_data)
dataloader_kwargs = {'batch_size': 1, 'shuffle': False}
loader = torch.utils.data.DataLoader(dataset, **dataloader_kwargs)

rna_qFibro = sc.read_h5ad('rna_qFibro_100.h5ad')
atac_qFibro = sc.read_h5ad('atac_qFibro_100.h5ad')
out = rna_qFibro.copy()
out.X = np.zeros(out.shape)
i = 0
for inputs in tqdm(loader, ncols=80, desc='output attention matrix'):
    rna_sequence, rna_value, atac_sequence, _, enc_pad_mask = [each.to(device) for each in inputs]
    out.X[i, (rna_sequence[rna_sequence!=0]-1).cpu()] = model.generate_attn_weight(rna_sequence, atac_sequence, rna_value, enc_mask=enc_pad_mask, which='decoder')[0].sum(axis=0)
    torch.cuda.empty_cache()
    i += 1

out.write('attn_qFibro_100.h5ad')

attn = sc.read_h5ad('attn_qFibro_100.h5ad')
(attn.X / attn.X.max(axis=1)[:, np.newaxis]).sum(axis=1).mean()  # 6.255087049780761   (seed=1  6.164570695891723)
sum(attn.X.sum(axis=0) / (attn.X.sum(axis=0).max()))             # 17.889998986005185  (seed=1  15.962734157227999)

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

(attn.X.sum(axis=0) / (attn.X.sum(axis=0).max()))[attn.var.index.isin(cr_tf)].sum()   # 0.5951161888828844  (seed=1  0.6013686011716683)


######################################################################## apFibro ########################################################################
rna_qFibro_apFibro_MyoFibro = sc.read_h5ad('res_qFibro_apFibro_MyoFibro.h5ad')
random.seed(0)
apFibro_idx = list(np.argwhere(rna_qFibro_apFibro_MyoFibro.obs['cell_anno']=='apFibro').flatten())
apFibro_idx_100 = random.sample(list(apFibro_idx), 100)
out = sc.AnnData(rna_qFibro_apFibro_MyoFibro[apFibro_idx_100].raw.X, obs=rna_qFibro_apFibro_MyoFibro[apFibro_idx_100].obs, var=rna_qFibro_apFibro_MyoFibro[apFibro_idx_100].raw.var)
np.count_nonzero(out.X.toarray(), axis=1).max()  # 6791
out.write('rna_apFibro_100.h5ad')

atac = sc.read_h5ad('../atac.h5ad')
atac_apFibro = atac[rna_qFibro_apFibro_MyoFibro.obs.index[apFibro_idx_100]].copy()
atac_apFibro.write('atac_apFibro_100.h5ad')

# python data_preprocess.py -r rna_apFibro_100.h5ad -a atac_apFibro_100.h5ad -s apFibro_pt_100 --dt test -n apFibro --config rna2atac_config_test_apFibro.yaml   # enc_max_len: 6791 

with open("rna2atac_config_test_apFibro.yaml", "r") as f:
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

apFibro_data = torch.load("./apFibro_pt_100/apFibro_0.pt")
dataset = PreDataset(apFibro_data)
dataloader_kwargs = {'batch_size': 1, 'shuffle': False}
loader = torch.utils.data.DataLoader(dataset, **dataloader_kwargs)

rna_apFibro = sc.read_h5ad('rna_apFibro_100.h5ad')
atac_apFibro = sc.read_h5ad('atac_apFibro_100.h5ad')
out = rna_apFibro.copy()
out.X = np.zeros(out.shape)
i = 0
for inputs in tqdm(loader, ncols=80, desc='output attention matrix'):
    rna_sequence, rna_value, atac_sequence, _, enc_pad_mask = [each.to(device) for each in inputs]
    out.X[i, (rna_sequence[rna_sequence!=0]-1).cpu()] = model.generate_attn_weight(rna_sequence, atac_sequence, rna_value, enc_mask=enc_pad_mask, which='decoder')[0].sum(axis=0)
    torch.cuda.empty_cache()
    i += 1

out.write('attn_apFibro_100.h5ad')

attn = sc.read_h5ad('attn_apFibro_100.h5ad')
(attn.X / attn.X.max(axis=1)[:, np.newaxis]).sum(axis=1).mean()  # 6.426582606976216  (seed=1  6.239077361769187)
sum(attn.X.sum(axis=0) / (attn.X.sum(axis=0).max()))             # 17.01195932997782  (seed=1  16.070358739278365)

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

(attn.X.sum(axis=0) / (attn.X.sum(axis=0).max()))[attn.var.index.isin(cr_tf)].sum()   # 0.6601534981235282  (seed=1  0.6560938370026412)


######################################################################## MyoFibro ########################################################################
rna_qFibro_apFibro_MyoFibro = sc.read_h5ad('res_qFibro_apFibro_MyoFibro.h5ad')
random.seed(0)
MyoFibro_idx = list(np.argwhere(rna_qFibro_apFibro_MyoFibro.obs['cell_anno']=='MyoFibro').flatten())
MyoFibro_idx_100 = random.sample(list(MyoFibro_idx), 100)
out = sc.AnnData(rna_qFibro_apFibro_MyoFibro[MyoFibro_idx_100].raw.X, obs=rna_qFibro_apFibro_MyoFibro[MyoFibro_idx_100].obs, var=rna_qFibro_apFibro_MyoFibro[MyoFibro_idx_100].raw.var)
np.count_nonzero(out.X.toarray(), axis=1).max()  # 3519
out.write('rna_MyoFibro_100.h5ad')

atac = sc.read_h5ad('../atac.h5ad')
atac_MyoFibro = atac[rna_qFibro_apFibro_MyoFibro.obs.index[MyoFibro_idx_100]].copy()
atac_MyoFibro.write('atac_MyoFibro_100.h5ad')

# python data_preprocess.py -r rna_MyoFibro_100.h5ad -a atac_MyoFibro_100.h5ad -s MyoFibro_pt_100 --dt test -n MyoFibro --config rna2atac_config_test_MyoFibro.yaml   # enc_max_len: 3519 

with open("rna2atac_config_test_MyoFibro.yaml", "r") as f:
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
device = torch.device('cuda:2')
model.to(device)
model.eval()

MyoFibro_data = torch.load("./MyoFibro_pt_100/MyoFibro_0.pt")
dataset = PreDataset(MyoFibro_data)
dataloader_kwargs = {'batch_size': 1, 'shuffle': False}
loader = torch.utils.data.DataLoader(dataset, **dataloader_kwargs)

rna_MyoFibro = sc.read_h5ad('rna_MyoFibro_100.h5ad')
atac_MyoFibro = sc.read_h5ad('atac_MyoFibro_100.h5ad')
out = rna_MyoFibro.copy()
out.X = np.zeros(out.shape)
i = 0
for inputs in tqdm(loader, ncols=80, desc='output attention matrix'):
    rna_sequence, rna_value, atac_sequence, _, enc_pad_mask = [each.to(device) for each in inputs]
    out.X[i, (rna_sequence[rna_sequence!=0]-1).cpu()] = model.generate_attn_weight(rna_sequence, atac_sequence, rna_value, enc_mask=enc_pad_mask, which='decoder')[0].sum(axis=0)
    torch.cuda.empty_cache()
    i += 1

out.write('attn_MyoFibro_100.h5ad')

attn = sc.read_h5ad('attn_MyoFibro_100.h5ad')
(attn.X / attn.X.max(axis=1)[:, np.newaxis]).sum(axis=1).mean()  # 6.019685058037266 (seed=1  6.062857704446831)
sum(attn.X.sum(axis=0) / (attn.X.sum(axis=0).max()))             # 18.41228501488108 (seed=1  20.270413808054894)


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

(attn.X.sum(axis=0) / (attn.X.sum(axis=0).max()))[attn.var.index.isin(cr_tf)].sum()   # 0.7327993408691938  (seed=1  0.9140492756775177)












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

# #################################### qFibro ####################################
# rna_qFibro_apFibro_MyoFibro = sc.read_h5ad('res_qFibro_apFibro_MyoFibro.h5ad')
# random.seed(0)
# qFibro_idx = list(np.argwhere(rna_qFibro_apFibro_MyoFibro.obs['cell_anno']=='qFibro').flatten())  # 407
# qFibro_idx_20 = random.sample(list(qFibro_idx), 40)
# out = sc.AnnData(rna_qFibro_apFibro_MyoFibro[qFibro_idx_20].raw.X, obs=rna_qFibro_apFibro_MyoFibro[qFibro_idx_20].obs, var=rna_qFibro_apFibro_MyoFibro[qFibro_idx_20].raw.var)
# np.count_nonzero(out.X.toarray(), axis=1).max()  # 4051
# out.write('rna_qFibro_20.h5ad')

# atac = sc.read_h5ad('../atac.h5ad')
# atac_qFibro = atac[rna_qFibro_apFibro_MyoFibro.obs.index[qFibro_idx_20]].copy()
# atac_qFibro.write('atac_qFibro_20.h5ad')

# # python data_preprocess.py -r rna_qFibro_20.h5ad -a atac_qFibro_20.h5ad -s qFibro_pt_20 --dt test -n qFibro --config rna2atac_config_test_qFibro.yaml   # enc_max_len: 1939 

# with open("rna2atac_config_test_qFibro.yaml", "r") as f:
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

# qFibro_data = torch.load("./qFibro_pt_20/qFibro_0.pt")
# dataset = PreDataset(qFibro_data)
# dataloader_kwargs = {'batch_size': 1, 'shuffle': False}
# loader = torch.utils.data.DataLoader(dataset, **dataloader_kwargs)

# # time: 1.5 min
# i=0
# for inputs in tqdm(loader, ncols=80, desc='output attention matrix'):
#     rna_sequence, rna_value, atac_sequence, _, enc_pad_mask = [each.to(device) for each in inputs]
#     attn = model.generate_attn_weight(rna_sequence, atac_sequence, rna_value, enc_mask=enc_pad_mask, which='decoder')
#     attn = attn[0]  # attn = attn[0].to(torch.float16)
#     with h5py.File('attn/attn_qFibro_'+str(i)+'.h5', 'w') as f:
#         f.create_dataset('attn', data=attn)
#     torch.cuda.empty_cache()
#     i += 1

# rna_qFibro_20 = sc.read_h5ad('rna_qFibro_20.h5ad')
# nonzero = np.count_nonzero(rna_qFibro_20.X.toarray(), axis=0)
# gene_lst = rna_qFibro_20.var.index[nonzero>=3]  # 3512

# atac_qFibro_20 = sc.read_h5ad('atac_qFibro_20.h5ad')

# gene_attn = {}
# for gene in gene_lst:
#     gene_attn[gene] = np.zeros([atac_qFibro_20.shape[1], 1], dtype='float32')  # not float16

# for i in range(20):
#     rna_sequence = qFibro_data[0][i].flatten()
#     with h5py.File('attn/attn_qFibro_'+str(i)+'.h5', 'r') as f:
#         attn = f['attn'][:]
#         for gene in tqdm(gene_lst, ncols=80, desc='cell '+str(i)):
#             idx = torch.argwhere((rna_sequence==(np.argwhere(rna_qFibro_20.var.index==gene))[0][0]+1).flatten())
#             if len(idx)!=0:
#                 gene_attn[gene] += attn[:, [idx.item()]]

# with open('./attn/attn_20_qFibro_3cell.pkl', 'wb') as file:
#     pickle.dump(gene_attn, file)

# gene_attn_sum = [gene_attn[gene].sum() for gene in gene_lst]
# df = pd.DataFrame({'gene':gene_lst, 'attn':gene_attn_sum})
# df.to_csv('./attn/attn_no_norm_cnt_20_qFibro_3cell.txt', sep='\t', header=None, index=None)

# df['attn'].sum()   # 5185111.0


# #################################### apFibro ####################################
# rna_qFibro_apFibro_MyoFibro = sc.read_h5ad('res_qFibro_apFibro_MyoFibro.h5ad')
# random.seed(0)
# apFibro_idx = list(np.argwhere(rna_qFibro_apFibro_MyoFibro.obs['cell_anno']=='apFibro').flatten())
# apFibro_idx_20 = random.sample(list(apFibro_idx), 20)
# out = sc.AnnData(rna_qFibro_apFibro_MyoFibro[apFibro_idx_20].raw.X, obs=rna_qFibro_apFibro_MyoFibro[apFibro_idx_20].obs, var=rna_qFibro_apFibro_MyoFibro[apFibro_idx_20].raw.var)
# np.count_nonzero(out.X.toarray(), axis=1).max()  # 6791
# out.write('rna_apFibro_20.h5ad')

# atac = sc.read_h5ad('../atac.h5ad')
# atac_apFibro = atac[rna_qFibro_apFibro_MyoFibro.obs.index[apFibro_idx_20]].copy()
# atac_apFibro.write('atac_apFibro_20.h5ad')

# # python data_preprocess.py -r rna_apFibro_20.h5ad -a atac_apFibro_20.h5ad -s apFibro_pt_20 --dt test -n apFibro --config rna2atac_config_test_apFibro.yaml   # enc_max_len: 6791 

# with open("rna2atac_config_test_apFibro.yaml", "r") as f:
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

# apFibro_data = torch.load("./apFibro_pt_20/apFibro_0.pt")
# dataset = PreDataset(apFibro_data)
# dataloader_kwargs = {'batch_size': 1, 'shuffle': False}
# loader = torch.utils.data.DataLoader(dataset, **dataloader_kwargs)

# # time: 1.5 min
# i=0
# for inputs in tqdm(loader, ncols=80, desc='output attention matrix'):
#     rna_sequence, rna_value, atac_sequence, _, enc_pad_mask = [each.to(device) for each in inputs]
#     attn = model.generate_attn_weight(rna_sequence, atac_sequence, rna_value, enc_mask=enc_pad_mask, which='decoder')
#     attn = attn[0]  # attn = attn[0].to(torch.float16)
#     with h5py.File('attn/attn_apFibro_'+str(i)+'.h5', 'w') as f:
#         f.create_dataset('attn', data=attn)
#     torch.cuda.empty_cache()
#     i += 1

# rna_apFibro_20 = sc.read_h5ad('rna_apFibro_20.h5ad')
# nonzero = np.count_nonzero(rna_apFibro_20.X.toarray(), axis=0)
# gene_lst = rna_apFibro_20.var.index[nonzero>=3]  # 6388

# atac_apFibro_20 = sc.read_h5ad('atac_apFibro_20.h5ad')

# gene_attn = {}
# for gene in gene_lst:
#     gene_attn[gene] = np.zeros([atac_apFibro_20.shape[1], 1], dtype='float32')  # not float16

# for i in range(20):
#     rna_sequence = apFibro_data[0][i].flatten()
#     with h5py.File('attn/attn_apFibro_'+str(i)+'.h5', 'r') as f:
#         attn = f['attn'][:]
#         for gene in tqdm(gene_lst, ncols=80, desc='cell '+str(i)):
#             idx = torch.argwhere((rna_sequence==(np.argwhere(rna_apFibro_20.var.index==gene))[0][0]+1).flatten())
#             if len(idx)!=0:
#                 gene_attn[gene] += attn[:, [idx.item()]]

# with open('./attn/attn_20_apFibro_3cell.pkl', 'wb') as file:
#     pickle.dump(gene_attn, file)

# gene_attn_sum = [gene_attn[gene].sum() for gene in gene_lst]
# df = pd.DataFrame({'gene':gene_lst, 'attn':gene_attn_sum})
# df.to_csv('./attn/attn_no_norm_cnt_20_apFibro_3cell.txt', sep='\t', header=None, index=None)

# df['attn'].sum()   # 6102081.0


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

# df_qFibro = pd.read_csv('attn_no_norm_cnt_20_qFibro_3cell.txt', header=None, sep='\t')
# df_qFibro.columns = ['gene', 'attn']
# df_qFibro['attn_norm'] = df_qFibro['attn']/df_qFibro['attn'].max()

# df_apFibro = pd.read_csv('attn_no_norm_cnt_20_apFibro_3cell.txt', header=None, sep='\t')
# df_apFibro.columns = ['gene', 'attn']
# df_apFibro['attn_norm'] = df_apFibro['attn']/df_apFibro['attn'].max()

# df_MyoFibro = pd.read_csv('attn_no_norm_cnt_20_MyoFibro_3cell.txt', header=None, sep='\t')
# df_MyoFibro.columns = ['gene', 'attn']
# df_MyoFibro['attn_norm'] = df_MyoFibro['attn']/df_MyoFibro['attn'].max()

# sum(df_qFibro['attn']/df_qFibro['attn'].max())
# # 11.71064976529011

# sum(df_apFibro['attn']/df_apFibro['attn'].max())
# # 14.095044444787515

# sum(df_MyoFibro['attn']/df_MyoFibro['attn'].max())
# # 13.721087342739994

# df_qFibro[df_qFibro['gene'].isin(cr_tf)]['attn_norm'].sum()
# # 0.5025441479088468

# df_apFibro[df_apFibro['gene'].isin(cr_tf)]['attn_norm'].sum()
# # 0.5334179425086834

# df_MyoFibro[df_MyoFibro['gene'].isin(cr_tf)]['attn_norm'].sum()
# # 0.3812379842675654
