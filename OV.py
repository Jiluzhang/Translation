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
import pickle

rna  = sc.read_h5ad('rna.h5ad')
atac = sc.read_h5ad('atac.h5ad')

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

################################ tumor cells ################################
tumor_idx = np.argwhere(rna.obs.cell_anno=='Tumor')[:20].flatten()
tumor_data = []
for i in range(len(data)):
    tumor_data.append(data[i][tumor_idx])

dataset = PreDataset(tumor_data)
dataloader_kwargs = {'batch_size': 1, 'shuffle': False}
loader = torch.utils.data.DataLoader(dataset, **dataloader_kwargs)
device = torch.device('cuda:0')  # device = select_least_used_gpu()
model.to(device)

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
    
    if i==20:
        break

rna_tumor_20 = rna[tumor_idx].copy()
nonzero = np.count_nonzero(rna_tumor_20.X.toarray(), axis=0)
gene_lst = rna.var.index[nonzero>0]  # 10956

gene_attn = {}
for gene in tqdm(gene_lst, ncols=80):
    gene_attn[gene] = np.zeros([atac.shape[1], 1], dtype='float16')

# ~ 40 min
for i in range(20):
    rna_sequence = tumor_data[0][i].flatten()
    with h5py.File('attn_tumor_'+str(i)+'.h5', 'r') as f:
        attn = f['attn'][:]
        for gene in tqdm(gene_lst, ncols=80, desc='cell '+str(i)):
            idx = torch.argwhere((rna_sequence==(np.argwhere(rna.var.index==gene))[0][0]+1).flatten())
            if len(idx)!=0:
                gene_attn[gene] += attn[:, [idx.item()]]

with open('tumor_attn_20_cell.pkl', 'wb') as file:
    pickle.dump(gene_attn, file)

gene_attn_sum = [gene_attn[gene].sum() for gene in gene_lst]
df = pd.DataFrame({'gene':gene_lst, 'cnt':gene_attn_sum})
df.to_csv('tumor_attn_no_norm_cnt_20.txt', sep='\t', header=None, index=None)

# def cnt(tf):
#     return sum((tf_attn[tf].flatten())>((0.000)*3))

# with Pool(40) as p:
#     tf_peak_cnt = p.map(cnt, tf_lst)

# df = pd.DataFrame({'tf':tf_lst, 'cnt':tf_peak_cnt})
# # df.sort_values('cnt', ascending=False)[:100]['tf'].to_csv('tmp.txt',  index=False, header=False)
# df.to_csv('tumor_attn_no_norm_cnt_3.txt', sep='\t', header=None, index=None)


## factor enrichment
import scanpy as sc
import pandas as pd
import numpy as np
from plotnine import *
from scipy import stats

rna = sc.read_h5ad('rna.h5ad')

# wget -c https://jaspar.elixir.no/download/data/2024/CORE/JASPAR2024_CORE_vertebrates_non-redundant_pfms_jaspar.txt
# grep ">" JASPAR2024_CORE_vertebrates_non-redundant_pfms_jaspar.txt | grep -v "::" | grep -v "-" | awk '{print toupper($2)}' | sort | uniq > TF_jaspar.txt

cr = ['SMARCA4', 'SMARCA2', 'ARID1A', 'ARID1B', 'SMARCB1',
      'CHD1', 'CHD2', 'CHD3', 'CHD4', 'CHD5', 'CHD6', 'CHD7', 'CHD8', 'CHD9',
      'BRD2', 'BRD3', 'BRD4', 'BRDT',
      'SMARCA5', 'SMARCA1', 'ACTL6A', 'ACTL6B',
      'SSRP1', 'SUPT16H',
      'EP400',
      'SMARCD1', 'SMARCD2', 'SMARCD3']

tf_lst = pd.read_table('TF_jaspar.txt', header=None)
tf = list(tf_lst[0].values)

df = pd.read_csv('tumor_attn_no_norm_cnt_20.txt', header=None, sep='\t')
df.columns = ['gene', 'attn']
df = df.sort_values('attn', ascending=False)
df.index = range(df.shape[0])

df_cr = pd.DataFrame({'idx': 'CR', 'attn': df[df['gene'].isin(cr)]['attn'].values})               # 24
df_tf = pd.DataFrame({'idx': 'TF', 'attn': df[df['gene'].isin(tf)]['attn'].values})               # 352
df_others = pd.DataFrame({'idx': 'Others', 'attn': df[~df['gene'].isin(cr+tf)]['attn'].values})   # 10580
df_cr_tf_others = pd.concat([df_cr, df_tf, df_others])
df_cr_tf_others['idx'] = pd.Categorical(df_cr_tf_others['idx'], categories=['CR', 'TF', 'Others'])
df_cr_tf_others['Avg_attn'] =  df_cr_tf_others['attn']/(df_cr_tf_others.shape[0]*20)

p = ggplot(df_cr_tf_others, aes(x='idx', y='Avg_attn', fill='idx')) + geom_boxplot(width=0.5, show_legend=False, outlier_shape='') + xlab('') +\
                                                                      scale_y_continuous(limits=[0, 0.03], breaks=np.arange(0, 0.03+0.001, 0.005)) + theme_bw()
                                                                      # + annotate("text", x=1.5, y=0.035, label=f"P = {p_value:.2e}", ha='center')
p.save(filename='cr_tf_others_box_cnt_20_tumor.pdf', dpi=300, height=4, width=4)
stats.ttest_ind(df_cr['attn'], df_tf['attn'])[1]      # 0.006237903399623997
stats.ttest_ind(df_cr['attn'], df_others['attn'])[1]  # 3.583299943765924e-05
stats.ttest_ind(df_tf['attn'], df_others['attn'])[1]  # 0.0025025401898605736


# all_genes = pd.DataFrame({'tf': rna.var.index.values})
# tmp = pd.merge(all_genes, df, how='left')
# tmp.fillna(0, inplace=True)
# df = tmp.copy()

# sum(df['cnt']>1000)                       # 120 (120/863=0.13904982618771727)
# sum((df['cnt']>100) & (df['cnt']<=1000))  # 257 (257/863=0.29779837775202783)
# sum(df['cnt']<=100)                       # 486 (486/863=0.5631517960602549)

# sum(df_cr['cnt']>1000)                          # 12 (12/105=0.11428571428571428)
# sum((df_cr['cnt']>100) & (df_cr['cnt']<=1000))  # 32 (32/105=0.3047619047619048)
# sum(df_cr['cnt']<=100)                          # 61 (61/105=0.580952380952381)

# sum(df_non_cr['cnt']>1000)                              # 108 (108/758=0.1424802110817942)
# sum((df_non_cr['cnt']>100) & (df_non_cr['cnt']<=1000))  # 225 (225/758=0.29683377308707126)
# sum(df_non_cr['cnt']<=100)                              # 425 (425/758=0.5606860158311345)

# from scipy.stats import chi2_contingency
# chi2_contingency(np.array([[108, 225, 425], [12, 32, 61]]))


######################## Fibroblasts ###########################
data = torch.load("/fs/home/jiluzhang/scM2M_no_dec_attn/pan_cancer/all_data/data_with_annotation/h5ad/scM2M/preprocessed_data_test/preprocessed_data_0.pt")
for i in range(1, 5):
    data_tmp = torch.load('/fs/home/jiluzhang/scM2M_no_dec_attn/pan_cancer/all_data/data_with_annotation/h5ad/scM2M/preprocessed_data_test/preprocessed_data_'+str(i)+'.pt')
    data[0] = torch.cat([data[0], data_tmp[0]], axis=0)
    data[1] = torch.cat([data[1], data_tmp[1]], axis=0)
    data[2] = torch.cat([data[2], data_tmp[2]], axis=0)
    data[3] = torch.cat([data[3], data_tmp[3]], axis=0)
    data[4] = torch.cat([data[4], data_tmp[4]], axis=0)
    
fibro_idx = np.argwhere(rna.obs.cell_anno=='Fibroblasts')[:20].flatten()
fibro_data = []
for i in range(len(data)):
    fibro_data.append(data[i][fibro_idx])

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

dataset = PreDataset(fibro_data)
dataloader_kwargs = {'batch_size': 1, 'shuffle': False}
loader = torch.utils.data.DataLoader(dataset, **dataloader_kwargs)
device = torch.device('cuda:0') 
model.to(device)

i = 0
for inputs in loader:
    rna_sequence, rna_value, atac_sequence, _, enc_pad_mask = [each.to(device) for each in inputs]
    attn = model.generate_attn_weight(rna_sequence, atac_sequence, rna_value, enc_mask=enc_pad_mask, which='decoder')
    attn = attn[0].to(torch.float16)
    with h5py.File('attn_fibro_'+str(i)+'.h5', 'w') as f:
        f.create_dataset('attn', data=attn)
    
    i += 1
    torch.cuda.empty_cache()
    print(str(i), 'cell done')
    
    if i==20:
        break

rna_fibro_20 = rna[fibro_idx].copy()
nonzero = np.count_nonzero(rna_fibro_20.X.toarray(), axis=0)
gene_lst = rna.var.index[nonzero>0]  # 11222

gene_attn = {}
for gene in tqdm(gene_lst, ncols=80):
    gene_attn[gene] = np.zeros([atac.shape[1], 1], dtype='float16')

# ~ 40 min
for i in range(20):
    rna_sequence = fibro_data[0][i].flatten()
    with h5py.File('attn_fibro_'+str(i)+'.h5', 'r') as f:
        attn = f['attn'][:]
        for gene in tqdm(gene_lst, ncols=80, desc='cell '+str(i)):
            idx = torch.argwhere((rna_sequence==(np.argwhere(rna.var.index==gene))[0][0]+1).flatten())
            if len(idx)!=0:
                gene_attn[gene] += attn[:, [idx.item()]]

with open('fibro_attn_20_cell.pkl', 'wb') as file:
    pickle.dump(gene_attn, file)

gene_attn_sum = [gene_attn[gene].sum() for gene in gene_lst]
df = pd.DataFrame({'gene':gene_lst, 'cnt':gene_attn_sum})
df.to_csv('fibro_attn_no_norm_cnt_20.txt', sep='\t', header=None, index=None)


## factor enrichment
import scanpy as sc
import pandas as pd
import numpy as np
from plotnine import *
from scipy import stats

rna = sc.read_h5ad('rna.h5ad')

cr = ['SMARCA4', 'SMARCA2', 'ARID1A', 'ARID1B', 'SMARCB1',
      'CHD1', 'CHD2', 'CHD3', 'CHD4', 'CHD5', 'CHD6', 'CHD7', 'CHD8', 'CHD9',
      'BRD2', 'BRD3', 'BRD4', 'BRDT',
      'SMARCA5', 'SMARCA1', 'ACTL6A', 'ACTL6B',
      'SSRP1', 'SUPT16H',
      'EP400',
      'SMARCD1', 'SMARCD2', 'SMARCD3']

tf_lst = pd.read_table('TF_jaspar.txt', header=None)
tf = list(tf_lst[0].values)

df = pd.read_csv('fibro_attn_no_norm_cnt_20.txt', header=None, sep='\t')
df.columns = ['gene', 'attn']
df = df.sort_values('attn', ascending=False)
df.index = range(df.shape[0])

df_cr = pd.DataFrame({'idx': 'CR', 'attn': df[df['gene'].isin(cr)]['attn'].values})               # 23
df_tf = pd.DataFrame({'idx': 'TF', 'attn': df[df['gene'].isin(tf)]['attn'].values})               # 351
df_others = pd.DataFrame({'idx': 'Others', 'attn': df[~df['gene'].isin(cr+tf)]['attn'].values})   # 10848
df_cr_tf_others = pd.concat([df_cr, df_tf, df_others])
df_cr_tf_others['idx'] = pd.Categorical(df_cr_tf_others['idx'], categories=['CR', 'TF', 'Others'])
df_cr_tf_others['Avg_attn'] =  df_cr_tf_others['attn']/(df_cr_tf_others.shape[0]*20)

p = ggplot(df_cr_tf_others, aes(x='idx', y='Avg_attn', fill='idx')) + geom_boxplot(width=0.5, show_legend=False, outlier_shape='') + xlab('') +\
                                                                      scale_y_continuous(limits=[0, 0.03], breaks=np.arange(0, 0.03+0.001, 0.005)) + theme_bw()
p.save(filename='cr_tf_others_box_cnt_20_fibro.pdf', dpi=300, height=4, width=4)
stats.ttest_ind(df_cr['attn'], df_tf['attn'])[1]      # 0.07812001669105603
stats.ttest_ind(df_cr['attn'], df_others['attn'])[1]  # 0.0003680603810052755
stats.ttest_ind(df_tf['attn'], df_others['attn'])[1]  # 1.122492152760193e-05


########## motif checking ############
## tumor cells
import pickle
import pandas as pd
from multiprocessing import Pool
import scanpy as sc

with open('tumor_attn_20_cell.pkl', 'rb') as file:
    tf_attn = pickle.load(file)

tf_raw = pd.read_table('TF_jaspar.txt', header=None)
tf_raw_lst = list(tf_raw[0].values)

rna  = sc.read_h5ad('rna.h5ad')
atac = sc.read_h5ad('atac.h5ad')

tf_lst = [tf for tf in tf_raw_lst if tf in tf_attn.keys()]

# def write_bed(tf):
#     df = pd.DataFrame(atac.var.index[(tf_attn[tf].flatten()/20/len(tf_attn))>(1e-7)])
#     if df.shape[0]>10000:
#         df['chrom'] = df[0].apply(lambda x: x.split(':')[0])
#         df['start'] = df[0].apply(lambda x: x.split(':')[1].split('-')[0])
#         df['end'] = df[0].apply(lambda x: x.split(':')[1].split('-')[1])
#         df[['chrom', 'start', 'end']].to_csv('TF_bed/'+tf+'_tumor.bed', sep='\t', index=None, header=None)

# top 10000
def write_bed(tf):
    df = pd.DataFrame(atac.var.index[np.argsort(tf_attn[tf].flatten())[-10000:][::-1]])
    df['chrom'] = df[0].apply(lambda x: x.split(':')[0])
    df['start'] = df[0].apply(lambda x: x.split(':')[1].split('-')[0])
    df['end'] = df[0].apply(lambda x: x.split(':')[1].split('-')[1])
    df[['chrom', 'start', 'end']].to_csv('TF_bed/'+tf+'_tumor_tmp.bed', sep='\t', index=None, header=None)

with Pool(5) as p:
    p.map(write_bed, tf_lst)



for filename in `ls /mnt/Saturn/home/jiluzhang/scM2M_no_dec_attn/pan_cancer/all_data/data_with_annotation/h5ad/scM2M/OV/TF_bed`;do
    tf=${filename%_*}
    true_cnt=$(sort -k1,1 -k2,2n /mnt/Saturn/home/jiluzhang/scM2M_no_dec_attn/pan_cancer/all_data/data_with_annotation/h5ad/scM2M/OV/TF_bed/$tf\_tumor.bed | \
               bedtools intersect -a stdin -b /fse/home/dongxin/Projects/0Finished/SCRIPT/motif/human/human_motif_bed/bed/$tf.bed -wa | uniq | wc -l)
    #tf_cnt=$(cat /mnt/Saturn/home/jiluzhang/scM2M_no_dec_attn/pan_cancer/all_data/data_with_annotation/h5ad/scM2M/OV/TF_bed/$tf\_tumor.bed | wc -l)
    exp_cnt=$(bedtools intersect -a /mnt/Saturn/home/jiluzhang/scM2M_no_dec_attn/pan_cancer/all_data/data_with_annotation/h5ad/scM2M/PAX8/human_cCREs.bed \
                                  -b /fse/home/dongxin/Projects/0Finished/SCRIPT/motif/human/human_motif_bed/bed/$tf.bed -wa | uniq | wc -l)
    echo -e $tf"\t"$true_cnt"\t"10000"\t"$exp_cnt"\t"1033239 >> tumor_motif_tf_res.txt
    echo $tf done
done



from scipy import stats
from plotnine import *
import pandas as pd
import numpy as np
#from scipy.stats import chi2_contingency


df = pd.read_table('/mnt/Venus/home/jiluzhang/TF_motifs/v2/tumor_motif_tf_res.txt', header=None)
df.columns = ['tf', 'true_cnt', 'true_total', 'exp_cnt', 'exp_total']
df = df[df['exp_cnt']!=0]
df['obs_exp'] = (df['true_cnt']/df['true_total']) / (df['exp_cnt']/df['exp_total'])
#df['p_value'] = [chi2_contingency(np.array([[df.iloc[i]['true_cnt'], df.iloc[i]['true_total']], [df.iloc[i]['exp_cnt'], df.iloc[i]['exp_total']]]))[1] for i in range(df.shape[0])]
df.sort_values('obs_exp', inplace=True)
df.index = range(df.shape[0])
df['label'] = df['obs_exp'].apply(lambda x: '1' if x>1 else '0')
# stats.ttest_1samp(df['obs_exp'], 1)[1]  
p = ggplot(df, aes(x=df.index.values, y='obs_exp', color='label')) + geom_point(size=0.05) + scale_color_manual(values=['blue', 'red']) + \
                                                                     scale_y_continuous(limits=[0, 2], breaks=np.arange(0, 2+0.01, 0.5)) + \
                                                                     geom_hline(yintercept=1, color="orange", linetype="dashed") + xlab('TFs') + ylab('Observed / Expected') + theme_bw()
p.save(filename='tumor_motif_tf_res_scatter.pdf', dpi=300, height=4, width=4)
print(sum(df['obs_exp']>1))   # 160
print(sum(df['obs_exp']<=1))  # 179



## ChIP-seq data validation
## GRHL2
awk '{print $1 "\t" $2-500 "\t" $2+500}' /mnt/Saturn/home/jiluzhang/scM2M_no_dec_attn/pan_cancer/all_data/data_with_annotation/h5ad/scM2M/OV/TF_bed/GRHL2_tumor.bed |\
bedtools intersect -a stdin -b GRHL2_peaks.bed -wa | sort | uniq | wc -l   # 648 (648/10000*100=6.48)

awk '{print $1 "\t" $2-500 "\t" $2+500}' /mnt/Saturn/home/jiluzhang/scM2M_no_dec_attn/pan_cancer/all_data/data_with_annotation/h5ad/scM2M/PAX8/human_cCREs.bed |\
bedtools intersect -a stdin -b GRHL2_peaks.bed -wa | sort | uniq | wc -l   # 66717 (66717/1033239*100=6.457073339275811)

for i in `seq 1000`;do
    shuf /mnt/Saturn/home/jiluzhang/scM2M_no_dec_attn/pan_cancer/all_data/data_with_annotation/h5ad/scM2M/PAX8/human_cCREs.bed | \
               head -n `cat /mnt/Saturn/home/jiluzhang/scM2M_no_dec_attn/pan_cancer/all_data/data_with_annotation/h5ad/scM2M/OV/TF_bed/GRHL2_tumor.bed | wc -l` |\
               awk '{print $1 "\t" $2-500 "\t" $2+500}' |\
               bedtools intersect -a stdin -b GRHL2_peaks.bed -wa | sort | uniq | wc -l >> GRHL2_random.txt
    echo $i           
done

wc -l /mnt/Saturn/home/jiluzhang/scM2M_no_dec_attn/pan_cancer/all_data/data_with_annotation/h5ad/scM2M/OV/TF_bed/GRHL2_tumor.bed  # 10000
awk '{sum+=$1} END {print sum/1000}' GRHL2_random.txt  # 645.634
awk '{if($1>648) print$0}' GRHL2_random.txt | wc -l   # 448
wc -l GRHL2_peaks.bed  # 34487


## PAX8
awk '{print $1 "\t" $2-500 "\t" $2+500}' /mnt/Saturn/home/jiluzhang/scM2M_no_dec_attn/pan_cancer/all_data/data_with_annotation/h5ad/scM2M/OV/TF_bed/PAX8_tumor.bed |\
bedtools intersect -a stdin -b /mnt/Saturn/home/jiluzhang/scM2M_no_dec_attn/pan_cancer/all_data/data_with_annotation/h5ad/scM2M/PAX8/pax8_hg38.bed -wa | sort | uniq | wc -l   
# 546 (546/10000*100=5.46)

awk '{print $1 "\t" $2-500 "\t" $2+500}' /mnt/Saturn/home/jiluzhang/scM2M_no_dec_attn/pan_cancer/all_data/data_with_annotation/h5ad/scM2M/PAX8/human_cCREs.bed |\
bedtools intersect -a stdin -b /mnt/Saturn/home/jiluzhang/scM2M_no_dec_attn/pan_cancer/all_data/data_with_annotation/h5ad/scM2M/PAX8/pax8_hg38.bed -wa | sort | uniq | wc -l   
# 65091 (65091/1033239*100=6.2997041342806455)

for i in `seq 1000`;do
    shuf /mnt/Saturn/home/jiluzhang/scM2M_no_dec_attn/pan_cancer/all_data/data_with_annotation/h5ad/scM2M/PAX8/human_cCREs.bed | \
               head -n `cat /mnt/Saturn/home/jiluzhang/scM2M_no_dec_attn/pan_cancer/all_data/data_with_annotation/h5ad/scM2M/OV/TF_bed/PAX8_tumor.bed | wc -l` |\
               awk '{print $1 "\t" $2-500 "\t" $2+500}' |\
               bedtools intersect -a stdin -b /mnt/Saturn/home/jiluzhang/scM2M_no_dec_attn/pan_cancer/all_data/data_with_annotation/h5ad/scM2M/PAX8/pax8_hg38.bed -wa | sort | uniq | wc -l \
               >> PAX8_random.txt
    echo $i           
done

wc -l /mnt/Saturn/home/jiluzhang/scM2M_no_dec_attn/pan_cancer/all_data/data_with_annotation/h5ad/scM2M/OV/TF_bed/PAX8_tumor.bed  # 10000
awk '{sum+=$1} END {print sum/1000}' PAX8_random.txt  # 630.836
awk '{if($1>546) print$0}' PAX8_random.txt | wc -l   # 1000
wc -l /mnt/Saturn/home/jiluzhang/scM2M_no_dec_attn/pan_cancer/all_data/data_with_annotation/h5ad/scM2M/PAX8/pax8_hg38.bed  # 32700


## MECOM
awk '{print $1 "\t" $2-500 "\t" $2+500}' /mnt/Saturn/home/jiluzhang/scM2M_no_dec_attn/pan_cancer/all_data/data_with_annotation/h5ad/scM2M/OV/TF_bed/PAX8_tumor.bed |\
bedtools intersect -a stdin -b /mnt/Saturn/home/jiluzhang/scM2M_no_dec_attn/pan_cancer/all_data/data_with_annotation/h5ad/scM2M/PAX8/pax8_hg38.bed -wa | sort | uniq | wc -l   
# 546 (546/10000*100=5.46)

awk '{print $1 "\t" $2-500 "\t" $2+500}' /mnt/Saturn/home/jiluzhang/scM2M_no_dec_attn/pan_cancer/all_data/data_with_annotation/h5ad/scM2M/PAX8/human_cCREs.bed |\
bedtools intersect -a stdin -b /mnt/Saturn/home/jiluzhang/scM2M_no_dec_attn/pan_cancer/all_data/data_with_annotation/h5ad/scM2M/PAX8/pax8_hg38.bed -wa | sort | uniq | wc -l   
# 65091 (65091/1033239*100=6.2997041342806455)

for i in `seq 1000`;do
    shuf /mnt/Saturn/home/jiluzhang/scM2M_no_dec_attn/pan_cancer/all_data/data_with_annotation/h5ad/scM2M/PAX8/human_cCREs.bed | \
               head -n `cat /mnt/Saturn/home/jiluzhang/scM2M_no_dec_attn/pan_cancer/all_data/data_with_annotation/h5ad/scM2M/OV/TF_bed/PAX8_tumor.bed | wc -l` |\
               awk '{print $1 "\t" $2-500 "\t" $2+500}' |\
               bedtools intersect -a stdin -b /mnt/Saturn/home/jiluzhang/scM2M_no_dec_attn/pan_cancer/all_data/data_with_annotation/h5ad/scM2M/PAX8/pax8_hg38.bed -wa | sort | uniq | wc -l \
               >> PAX8_random.txt
    echo $i           
done

wc -l /mnt/Saturn/home/jiluzhang/scM2M_no_dec_attn/pan_cancer/all_data/data_with_annotation/h5ad/scM2M/OV/TF_bed/PAX8_tumor.bed  # 10000
awk '{sum+=$1} END {print sum/1000}' PAX8_random.txt  # 630.836
awk '{if($1>546) print$0}' PAX8_random.txt | wc -l   # 1000
wc -l /mnt/Saturn/home/jiluzhang/scM2M_no_dec_attn/pan_cancer/all_data/data_with_annotation/h5ad/scM2M/PAX8/pax8_hg38.bed  # 32700




