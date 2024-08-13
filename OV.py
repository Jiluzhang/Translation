## Using OV dataset for training & prediction
# VF026V1-S1
# Tumor          424
# Fibroblasts     41
# Macrophages     14
# B-cells          5
# Endothelial      4
# T-cells          4
# Plasma           2

python split_train_val.py --RNA rna.h5ad --ATAC atac.h5ad --train_pct 0.9
python data_preprocess.py -r rna_train.h5ad -a atac_train.h5ad -s preprocessed_data_train --dt train --config rna2atac_config_train.yaml
python data_preprocess.py -r rna_val.h5ad -a atac_val.h5ad -s preprocessed_data_val --dt val --config rna2atac_config_val.yaml
nohup accelerate launch --config_file accelerator_config_train.yaml --main_process_port 29823 rna2atac_train.py --config_file rna2atac_config_train.yaml \
                        --train_data_dir ./preprocessed_data_train --val_data_dir ./preprocessed_data_val -n rna2atac_train > 20240813.log &  

python data_preprocess.py -r rna.h5ad -a atac.h5ad -s preprocessed_data_test --dt test --config rna2atac_config_test.yaml  # ~10 min
accelerate launch --config_file accelerator_config_test.yaml --main_process_port 29822 rna2atac_test.py \
                  -d ./preprocessed_data_test \
                  -l save/2024-08-13_rna2atac_train_55/pytorch_model.bin --config_file rna2atac_config_test.yaml
python npy2h5ad.py
python csr2array.py --pred rna2atac_scm2m_raw.h5ad --true atac.h5ad
python cal_auroc_auprc.py --pred rna2atac_scm2m.h5ad --true atac.h5ad
python cal_cluster_plot.py --pred rna2atac_scm2m.h5ad --true atac.h5ad



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

model.load_state_dict(torch.load('./save/2024-08-13_rna2atac_train_55/pytorch_model.bin'))

rna  = sc.read_h5ad('rna.h5ad')
atac = sc.read_h5ad('atac.h5ad')

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


###### without min-max normalization
i = 0
for inputs in loader:
    rna_sequence, rna_value, atac_sequence, _, enc_pad_mask = [each.to(device) for each in inputs]
    attn = model.generate_attn_weight(rna_sequence, atac_sequence, rna_value, enc_mask=enc_pad_mask, which='decoder')
    # attn = (attn[0]-attn[0].min())/(attn[0].max()-attn[0].min())
    # attn[attn<(1e-5)] = 0
    attn = attn[0].to(torch.float16)
    with h5py.File('attn_tumor_'+str(i)+'.h5', 'w') as f:
        f.create_dataset('attn', data=attn)
    
    i += 1
    torch.cuda.empty_cache()
    print(str(i), 'cell done')
    
    if i==3:
        break

rna_tumor_20 = rna[tumor_idx].copy()
nonzero = np.count_nonzero(rna_tumor_20.X.toarray(), axis=0)
tf_lst = rna.var.index[nonzero>=5]

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
            
def cnt(tf):
    return sum((tf_attn[tf].flatten())>((0.005)*3))

with Pool(40) as p:
    tf_peak_cnt = p.map(cnt, tf_lst)

df = pd.DataFrame({'tf':tf_lst, 'cnt':tf_peak_cnt})
# df.sort_values('cnt', ascending=False)[:100]['tf'].to_csv('tmp.txt',  index=False, header=False)
df.to_csv('tumor_tf_peak_0.005_no_norm_cnt_3.txt', sep='\t', header=None, index=None)


tf_attn_sum = [tf_attn[tf].flatten().sum() for tf in tf_lst]
df = pd.DataFrame({'tf':tf_lst, 'cnt':tf_attn_sum})
df.to_csv('tumor_tf_peak_0.005_no_norm_cnt_3_sum.txt', sep='\t', header=None, index=None)


## factor enrichment
import scanpy as sc
import pandas as pd
import numpy as np
from plotnine import *
from scipy import stats

rna = sc.read_h5ad('rna.h5ad')

# 0: chromatin remodeler
# 1: chromatin remodeling
# 2: transcription factor
# 3: ovarian cancer specific transcription factor


# cr_lst = pd.read_table('cr.txt', header=None)
# cr_lst = pd.read_table('tf.txt', header=None)
# cr_lst = pd.read_table('ov_tf.txt', header=None)
# cr = cr_lst[0].values


cr = ['SMARCA4', 'SMARCA2', 'ARID1A', 'ARID1B', 'SMARCB1',
      'CHD1', 'CHD2', 'CHD3', 'CHD4', 'CHD5', 'CHD6', 'CHD7', 'CHD8', 'CHD9',
      'BRD2', 'BRD3', 'BRD4', 'BRDT',
      'SMARCA5', 'SMARCA1', 'ACTL6A', 'ACTL6B',
      'SSRP1', 'SUPT16H',
      'EP400',
      'SMARCD1', 'SMARCD2', 'SMARCD3']
cr = ['MECOM', 'PAX8', 'WT1', 'SOX17']

## cutoff: 0.005
df = pd.read_csv('tumor_tf_peak_0.005_no_norm_cnt_3.txt', header=None, sep='\t')
df.columns = ['tf', 'cnt']
df = df.sort_values('cnt', ascending=False)
df.index = range(df.shape[0])
df = df[df['cnt']!=0]

p_value = stats.ttest_ind(df[~df['tf'].isin(cr)]['cnt'], df[df['tf'].isin(cr)]['cnt'])[1]

df_cr = pd.DataFrame({'idx': 'CR', 'cnt': df[df['tf'].isin(cr)]['cnt'].values})
df_non_cr = pd.DataFrame({'idx': 'Non_CR', 'cnt': df[~df['tf'].isin(cr)]['cnt'].values})
df_cr_non_cr = pd.concat([df_cr, df_non_cr])
df_cr_non_cr['Norm_cnt'] =  df_cr_non_cr['cnt']/1033239

p = ggplot(df_cr_non_cr, aes(x='idx', y='Norm_cnt', fill='idx')) + geom_boxplot(width=0.5, show_legend=False, outlier_shape='') + xlab('') +\
                                                              scale_y_continuous(limits=[0, 0.001], breaks=np.arange(0, 0.001+0.0001, 0.0002)) + theme_bw() +\
                                                              annotate("text", x=1.5, y=0.0008, label=f"P = {p_value:.3f}", ha='center')
p.save(filename='CR_Non_CR_0.005_box_tf_tumor_sum.pdf', dpi=100, height=4, width=4)


## cutoff: 0.001
df = pd.read_csv('tumor_tf_peak_0.001_no_norm_cnt_3.txt', header=None, sep='\t')
df.columns = ['tf', 'cnt']
df = df.sort_values('cnt', ascending=False)
df.index = range(df.shape[0])
df = df[df['cnt']!=0]

p_value = stats.ttest_ind(df[~df['tf'].isin(cr)]['cnt'], df[df['tf'].isin(cr)]['cnt'])[1]

df_cr = pd.DataFrame({'idx': 'CR', 'cnt': df[df['tf'].isin(cr)]['cnt'].values})
df_non_cr = pd.DataFrame({'idx': 'Non_CR', 'cnt': df[~df['tf'].isin(cr)]['cnt'].values})
df_cr_non_cr = pd.concat([df_cr, df_non_cr])
df_cr_non_cr['Norm_cnt'] =  df_cr_non_cr['cnt']/1033239

p = ggplot(df_cr_non_cr, aes(x='idx', y='Norm_cnt', fill='idx')) + geom_boxplot(width=0.5, show_legend=False, outlier_shape='') + xlab('') +\
                                                              scale_y_continuous(limits=[0, 0.1], breaks=np.arange(0, 0.1+0.01, 0.02)) + theme_bw() +\
                                                              annotate("text", x=1.5, y=0.08, label=f"P = {p_value:.3f}", ha='center')
p.save(filename='CR_Non_CR_0.001_box_tf_tumor.pdf', dpi=100, height=4, width=4)


## cutoff: 0.0001
df = pd.read_csv('tumor_tf_peak_0.0001_no_norm_cnt_3.txt', header=None, sep='\t')
df.columns = ['tf', 'cnt']
df = df.sort_values('cnt', ascending=False)
df.index = range(df.shape[0])
df = df[df['cnt']!=0]

p_value = stats.ttest_ind(df[~df['tf'].isin(cr)]['cnt'], df[df['tf'].isin(cr)]['cnt'])[1]

df_cr = pd.DataFrame({'idx': 'CR', 'cnt': df[df['tf'].isin(cr)]['cnt'].values})
df_non_cr = pd.DataFrame({'idx': 'Non_CR', 'cnt': df[~df['tf'].isin(cr)]['cnt'].values})
df_cr_non_cr = pd.concat([df_cr, df_non_cr])
df_cr_non_cr['Norm_cnt'] =  df_cr_non_cr['cnt']/1033239

p = ggplot(df_cr_non_cr, aes(x='idx', y='Norm_cnt', fill='idx')) + geom_boxplot(width=0.5, show_legend=False, outlier_shape='') + xlab('') +\
                                                              scale_y_continuous(limits=[0, 0.5], breaks=np.arange(0, 0.5+0.01, 0.1)) + theme_bw() +\
                                                              annotate("text", x=1.5, y=0.4, label=f"P = {p_value:.3f}", ha='center')
p.save(filename='CR_Non_CR_0.0001_box_tf_tumor.pdf', dpi=100, height=4, width=4)






















