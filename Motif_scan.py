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
    df[['chrom', 'start', 'end']].to_csv('TF_bed/'+tf+'_tumor.bed', sep='\t', index=None, header=None)

with Pool(5) as p:
    p.map(write_bed, tf_lst)


for tf in `cut -f 1 tumor_motif_tf.txt`;do
    true_cnt=$(bedtools intersect -a /mnt/Saturn/home/jiluzhang/scM2M_no_dec_attn/pan_cancer/all_data/data_with_annotation/h5ad/scM2M/TF_bed/$tf\_tumor.bed \
                                  -b /fse/home/dongxin/Projects/0Finished/SCRIPT/motif/human/human_motif_bed/bed/$tf.bed -wa | uniq | wc -l)
    rand_cnt=$(shuf /mnt/Saturn/home/jiluzhang/scM2M_no_dec_attn/pan_cancer/all_data/data_with_annotation/h5ad/scM2M/PAX8/human_cCREs.bed | \
               head -n `cat /mnt/Saturn/home/jiluzhang/scM2M_no_dec_attn/pan_cancer/all_data/data_with_annotation/h5ad/scM2M/TF_bed/$tf\_tumor.bed | wc -l` |\
               bedtools intersect -a stdin -b /fse/home/dongxin/Projects/0Finished/SCRIPT/motif/human/human_motif_bed/bed/$tf.bed -wa | uniq | wc -l)
    echo -e $tf"\t"$true_cnt"\t"$rand_cnt >> tumor_motif_tf_res.txt
    echo $tf done
done


## calculate & plot observed vesus random
from scipy import stats
from plotnine import *
import pandas as pd
import numpy as np

df = pd.read_table('tumor_motif_tf_res.txt', header=None)
df[3] = df[1]/df[2]
df.columns = ['tf', 'true_cnt', 'rand_cnt', 'obs_exp']
df.sort_values('obs_exp', inplace=True)
df.index = range(df.shape[0])
stats.ttest_1samp(df['obs_exp'], 1)[1]  # 0.5538423816763332

p = ggplot(df, aes(x=df.index.values, y='obs_exp')) + geom_point(size=0.05) + scale_y_continuous(limits=[0.0, 2.0], breaks=np.arange(0.0, 2.0+0.01, 0.5)) + \
                                                      geom_hline(yintercept=1, color="red", linetype="dashed") + xlab('TFs') + ylab('Observed / Random') + theme_bw()
p.save(filename='tumor_motif_tf_res_scatter_1.pdf', dpi=300, height=4, width=4)
sum(df['obs_exp']>1)   # 96
sum(df['obs_exp']<=1)  # 122



## ChIP-seq data
grep -i "ov" human_factor_full_QC.txt | grep -v LoVo | grep -v ENCSR | grep -v OVOL2 | grep -v overexpressing | sed '1d'
# 39012	Homo sapiens	GSM1038268	CHRM2	SK-OV-3	Epithelium	Ovary	38.0	0.8069	0.992	58.0	0.008641	0.563218390805
# 52249	Homo sapiens	GSM1415630	AR	None	Granulosa cell	Ovary	39.0	0.8019	0.937	57.0	0.00806475	0.6729999999999999
# 52250	Homo sapiens	GSM1415631	AR	None	Granulosa cell	Ovary	38.0	0.7888	0.9620000000000001	51.0	0.002093	0.541666666667
# 52251	Homo sapiens	GSM1415632	AR	None	Granulosa cell	Ovary	38.0	0.8092	0.968	48.0	0.002207	0.5209580838319999
# 55778	Homo sapiens	GSM1817662	CTCF	Ovcar8	None	None	37.0	0.8198	0.988	17476.0	0.33333605144699996	0.9668
# 55779	Homo sapiens	GSM1817663	CTCFL	Ovcar8	None	None	38.0	0.5304	0.992	17562.0	0.14246525	0.9678
# 68002	Homo sapiens	GSM1825677	GRHL2	OVCA429	None	None	39.0	0.7279	0.983	12721.0	0.11023725	0.9754
# 68003	Homo sapiens	GSM1825676	GRHL2	NIH:OVCAR-3	None	None	38.0	0.7254	0.98	2772.0	0.02880175	0.927
# 70775	Homo sapiens	GSM2267289	CTCF	SK-OV-3	SKOV3	None	40.0	0.8032	0.991	83.0	0.00413275	0.35222672064800004
# 73240	Homo sapiens	GSM2054576	BRD4	OVCAR3	None	None	39.0	0.9001	0.9570000000000001	89.0	0.00268425	0.554054054054
# 73241	Homo sapiens	GSM2054575	BRD4	OVCAR3	None	None	39.0	0.899	0.9740000000000001	70.0	0.00445	0.816433566434
# 76876	Homo sapiens	GSM2044817	TERT	A2780	Epithelium	Ovary	37.0	0.8673	0.98	126.0	0.00341275	0.541114058355
# 88621	Homo sapiens	GSM1377693	YAP1	OVCAR5	None	ovarian	38.0	0.8968	0.9009999999999999	11.0	0.0196335	0.9684




