# conda install bioconda::bedtools
# /fse/home/dongxin/Projects/0Finished/SCRIPT/motif/human/human_motif_bed/bed
# cp /mnt/Saturn/home/jiluzhang/scM2M_no_dec_attn/pan_cancer/all_data/data_with_annotation/h5ad/scM2M/fibro_tf_peak_0.0001_no_norm_cnt_3.txt .
# cp /mnt/Saturn/home/jiluzhang/scM2M_no_dec_attn/pan_cancer/all_data/data_with_annotation/h5ad/scM2M/tumor_tf_peak_0.0001_no_norm_cnt_3.txt .

## make tf_lst
for filename in `ls /fse/home/dongxin/Projects/0Finished/SCRIPT/motif/human/human_motif_bed/bed`;do
    echo "${filename%.*}" >> motif_tf.txt
    done

awk '{if($2!=0) print$0}' tumor_tf_peak_0.0001_no_norm_cnt_3.txt > tumor_tf.txt

import pandas as pd
motif_tf = pd.read_table('motif_tf.txt', header=None)
motif_tf.columns = ['tf']
tumor_tf = pd.read_table('tumor_tf.txt', header=None)
tumor_tf.columns = ['tf', 'cnt']
tumor_motif_tf = pd.merge(tumor_tf, motif_tf)
tumor_motif_tf.to_csv('tumor_motif_tf.txt', header=None, index=None, sep='\t')


import pandas as pd
import numpy as np
import random
from functools import partial
import sys
sys.path.append("M2Mmodel")
from utils import PairDataset
import scanpy as sc
import torch
from utils import *
from collections import Counter
import pickle as pkl
import yaml
from M2M import M2M_rna2atac

import h5py
from tqdm import tqdm

from multiprocessing import Pool

data = torch.load("/fs/home/jiluzhang/scM2M_no_dec_attn/pan_cancer/all_data/data_with_annotation/h5ad/scM2M/preprocessed_data_test/preprocessed_data_0.pt")
# from config
with open("rna2atac_config_train.yaml", "r") as f:
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

model.load_state_dict(torch.load('./save/2024-07-11_rna2atac_train_1/pytorch_model.bin'))

rna  = sc.read_h5ad('VF026V1-S1_rna.h5ad')
atac = sc.read_h5ad('VF026V1-S1_atac.h5ad')

## tumor cells
tumor_idx = np.argwhere(rna.obs.cell_anno=='Tumor')[:20].flatten()
tumor_data = []
for i in range(len(data)):
    tumor_data.append(data[i][tumor_idx])

dataset = PreDataset(tumor_data)
dataloader_kwargs = {'batch_size': 1, 'shuffle': False}
loader = torch.utils.data.DataLoader(dataset, **dataloader_kwargs)
device = torch.device('cuda:0')  # device = select_least_used_gpu()
model.to(device)

tf_lst = pd.read_table('/mnt/Venus/home/jiluzhang/TF_motifs/tumor_motif_tf.txt', header=None)[0].values

tf_attn = {}
for tf in tf_lst:
    tf_attn[tf] = np.zeros([atac.shape[1], 1], dtype='float16') #set()

for i in range(3):#for i in range(20):
    rna_sequence = tumor_data[0][i].flatten()
    with h5py.File('attn_tumor_'+str(i)+'.h5', 'r') as f:
        attn = f['attn'][:]
        for tf in tqdm(tf_lst, ncols=80, desc='cell '+str(i)):
            idx = torch.argwhere(rna_sequence==(np.argwhere(rna.var.index==tf))[0][0]+1)
            if len(idx)!=0:
                tf_attn[tf] += attn[:, [idx.item()]]
            

def write_bed(tf):
    df = pd.DataFrame(atac.var.index[tf_attn[tf].flatten()/3>0.0001])
    df['chrom'] = df[0].apply(lambda x: x.split(':')[0])
    df['start'] = df[0].apply(lambda x: x.split(':')[1].split('-')[0])
    df['end'] = df[0].apply(lambda x: x.split(':')[1].split('-')[1])
    df[['chrom', 'start', 'end']].to_csv(tf+'_tumor.bed', sep='\t', index=None, header=None)

with Pool(5) as p:
    p.map(write_bed, tf_lst)





