## workdir: /fs/home/jiluzhang/Nature_methods/Figure_2/pan_cancer/cisformer/CTHRC1

import scanpy as sc
import harmonypy as hm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

plt.rcParams['pdf.fonttype'] = 42

rna = sc.read_h5ad('../rna.h5ad')                         # 144409 × 19160
fibro = rna[rna.obs['cell_anno']=='Fibroblasts'].copy()   # 28023 × 19160

sc.pp.normalize_total(fibro, target_sum=1e4)
sc.pp.log1p(fibro)
# fibro.obs['batch'] = [x.split('_')[0] for x in fibro.obs.index]  # exist samples with only one fibroblast
sc.pp.highly_variable_genes(fibro, batch_key='cancer_type')
fibro.raw = fibro
fibro = fibro[:, fibro.var.highly_variable]
sc.tl.pca(fibro)
ho = hm.run_harmony(fibro.obsm['X_pca'], fibro.obs, 'cancer_type')  # Converged after 2 iterations

fibro.obsm['X_pca_harmony'] = ho.Z_corr.T
sc.pp.neighbors(fibro, use_rep='X_pca_harmony')
sc.tl.umap(fibro)
sc.pl.umap(fibro, color='cancer_type', legend_fontsize='5', legend_loc='right margin', size=2, 
           title='', frameon=True, save='_batch_harmony_cancer_type.pdf')

sc.tl.leiden(fibro, resolution=0.5)
fibro.obs['leiden'].value_counts()
# 0     5308  (CTHRC1+)
# 1     4645
# 2     4167
# 3     4113
# 4     2550
# 5     1850
# 6     1433
# 7     1001
# 8      691
# 9      490
# 10     448
# 11     385
# 12     339
# 13     296
# 14     171
# 15     136

sc.tl.rank_genes_groups(fibro, 'leiden')
pd.DataFrame(fibro.uns['rank_genes_groups']['names']).head(50)

anno_lst = []
for i in fibro.obs['leiden']:
    if i=='0':
        anno_lst.append('CTHRC1+')
    else:
        anno_lst.append('CTHRC1-')

fibro.obs['CTHRC1_pos_neg'] = anno_lst

sc.pl.umap(fibro, color='leiden', legend_fontsize='5', legend_loc='right margin', size=2,
           title='', frameon=True, save='_leiden.pdf')

fibro.uns['CTHRC1_pos_neg_colors'] = ['red', 'grey']
sc.pl.umap(fibro, color='CTHRC1_pos_neg', legend_fontsize='5', legend_loc='right margin', size=2,
           title='', frameon=True, save='_cthrc1_pos_neg.pdf')

# sc.pl.umap(fibro, color=['CTHRC1'],
#            cmap=LinearSegmentedColormap.from_list('lightgray_red', ['lightgray', 'red']),
#            legend_loc="on data", frameon=True, ncols=3, vmin=0, vmax=5, save='_cthrc1.pdf')

ax= sc.pl.violin(fibro, keys='CTHRC1', groupby='CTHRC1_pos_neg', rotation=90, stripplot=False,
                 order=['CTHRC1-', 'CTHRC1+'], show=False)
ax.set_ylim(0-0.2, 4+0.4)
ax.figure.savefig('./figures/violin_cthrc1.pdf')

fibro.write('res_fibro.h5ad')


#### cell type-specific TFs identification
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

## cthrc1 pos
rna_fibro = sc.read_h5ad('res_fibro.h5ad')
random.seed(0)
pos_idx = list(np.argwhere(rna_fibro.obs['CTHRC1_pos_neg']=='CTHRC1+').flatten())
pos_idx_20 = random.sample(list(pos_idx), 20)
out = sc.AnnData(rna_fibro[pos_idx_20].raw.X, obs=rna_fibro[pos_idx_20].obs, var=rna_fibro[pos_idx_20].raw.var)
np.count_nonzero(out.X.toarray(), axis=1).max()  # 4508
out.write('rna_cthrc1_pos_20.h5ad')

atac = sc.read_h5ad('../atac.h5ad')
atac_fibro = atac[atac.obs['cell_anno']=='Fibroblasts'].copy()
atac_fibro[pos_idx_20].write('atac_cthrc1_pos_20.h5ad')

# python data_preprocess.py -r rna_cthrc1_pos_20.h5ad -a atac_cthrc1_pos_20.h5ad -s cthrc1_pos_pt_20 --dt test -n cthrc1_pos --config rna2atac_config_test.yaml   # enc_max_len: 4508 

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
model.load_state_dict(torch.load('../save/2024-10-25_rna2atac_pan_cancer_5/pytorch_model.bin'))
device = torch.device('cuda:7')
model.to(device)
model.eval()

cthrc1_pos_data = torch.load("./cthrc1_pos_pt_20/cthrc1_pos_0.pt")
dataset = PreDataset(cthrc1_pos_data)
dataloader_kwargs = {'batch_size': 1, 'shuffle': False}
loader = torch.utils.data.DataLoader(dataset, **dataloader_kwargs)

# time: 1.5 min
i=0
for inputs in tqdm(loader, ncols=80, desc='output attention matrix'):
    rna_sequence, rna_value, atac_sequence, _, enc_pad_mask = [each.to(device) for each in inputs]
    attn = model.generate_attn_weight(rna_sequence, atac_sequence, rna_value, enc_mask=enc_pad_mask, which='decoder')
    attn = attn[0]  # attn = attn[0].to(torch.float16)
    with h5py.File('attn/attn_cthrc1_pos_'+str(i)+'.h5', 'w') as f:
        f.create_dataset('attn', data=attn)
    torch.cuda.empty_cache()
    i += 1

rna_cthrc1_pos_20 = sc.read_h5ad('rna_cthrc1_pos_20.h5ad')
nonzero = np.count_nonzero(rna_cthrc1_pos_20.X.toarray(), axis=0)
gene_lst = rna_cthrc1_pos_20.var.index[nonzero>=3]  # 5829

atac_cthrc1_pos_20 = sc.read_h5ad('atac_cthrc1_pos_20.h5ad')

gene_attn = {}
for gene in gene_lst:
    gene_attn[gene] = np.zeros([atac_cthrc1_pos_20.shape[1], 1], dtype='float32')  # not float16

for i in range(20):
    rna_sequence = cthrc1_pos_data[0][i].flatten()
    with h5py.File('attn/attn_cthrc1_pos_'+str(i)+'.h5', 'r') as f:
        attn = f['attn'][:]
        for gene in tqdm(gene_lst, ncols=80, desc='cell '+str(i)):
            idx = torch.argwhere((rna_sequence==(np.argwhere(rna_cthrc1_pos_20.var.index==gene))[0][0]+1).flatten())
            if len(idx)!=0:
                gene_attn[gene] += attn[:, [idx.item()]]

with open('./attn/attn_20_cthrc1_3cell.pkl', 'wb') as file:
    pickle.dump(gene_attn, file)


gene_attn_sum = [gene_attn[gene].sum() for gene in gene_lst]
df = pd.DataFrame({'gene':gene_lst, 'cnt':gene_attn_sum})
df.to_csv('./attn/attn_no_norm_cnt_20_bcell_3cell.txt', sep='\t', header=None, index=None)











