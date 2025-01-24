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

atac = sc.read_h5ad('atac_train.h5ad')
atac[idx_20].write('atac_cd14_mono_20.h5ad')

# python data_preprocess.py -r rna_cd14_mono_20.h5ad -a atac_cd14_mono_20.h5ad -s pt_cd14_mono_20 --dt test -n cd14_mono_20 --config rna2atac_config_test.yaml

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

for gene in gene_attn:
    gene_attn[gene] = gene_attn[gene].flatten()

gene_attn_df = pd.DataFrame(gene_attn)
gene_attn_df.index = atac_cd14_mono_20.var.index.values
gene_attn_df.to_csv('./attn/attn_no_norm_cnt_20_cd14_mono_3cell.txt', sep='\t', header=True, index=True)

# gene_attn_sum = [gene_attn[gene].sum() for gene in gene_lst]
# df = pd.DataFrame({'gene':gene_lst, 'cnt':gene_attn_sum})
# df.to_csv('./attn/attn_no_norm_cnt_20_cd14_mono_3cell.txt', sep='\t', header=None, index=None)
