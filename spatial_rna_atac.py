######## spatial rna & atac data (from Fan lab) ########

# RNA-Seq HumanBrain_50um (GSM6206885)
wget -c https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM6206nnn/GSM6206885/suppl/GSM6206885_HumanBrain_50um_matrix.tsv.gz
wget -c https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM6206nnn/GSM6206885/suppl/GSM6206885_HumanBrain_50um_spatial.tar.gz

# ATAC-Seq HumanBrain_50um (GSM6206884)
wget -c https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM6206nnn/GSM6206884/suppl/GSM6206884_HumanBrain_50um_fragments.tsv.gz
wget -c https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM6206nnn/GSM6206884/suppl/GSM6206884_HumanBrain_50um_spatial.tar.gz


## data preprocessing
import scanpy as sc
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
import pybedtools
from tqdm import tqdm

########## rna ##########
rna_raw = pd.read_csv('GSM6206885_HumanBrain_50um_matrix.tsv.gz', sep='\t')
rna_raw = rna_raw.T  # cell * gene

obs = pd.DataFrame(index=rna_raw.index)
var = pd.DataFrame({'gene_ids': rna_raw.columns, 'feature_types': 'Gene Expression'})
var.index = var['gene_ids'].values
rna = sc.AnnData(rna_raw, obs=obs, var=var)
rna.X = csr_matrix(rna.X)
rna.write('spatial_rna_raw.h5ad')

genes = pd.read_table('human_genes.txt', names=['gene_id', 'gene_name'])

rna_genes = rna.var.copy()
rna_exp = pd.concat([rna_genes, pd.DataFrame(rna.X.T.todense(), index=rna_genes.index)], axis=1)
rna_exp.rename(columns={'gene_ids': 'gene_name'}, inplace=True)

X_new = pd.merge(genes, rna_exp, how='left', on='gene_name').iloc[:, 3:].T
X_new = np.array(X_new, dtype='float32')
X_new[np.isnan(X_new)] = 0

rna_new_var = pd.DataFrame({'gene_ids': genes['gene_id'], 'feature_types': 'Gene Expression'})
rna_new = sc.AnnData(X_new, obs=rna.obs, var=rna_new_var)  
rna_new.var.index = genes['gene_name'].values  # set index for var
rna_new.X = csr_matrix(rna_new.X)

rna_new.write('spatial_rna.h5ad')


########## atac ##########
atac_raw = pd.read_csv('GSM6206884_HumanBrain_50um_fragments.tsv.gz', comment='#', header=None, sep='\t')  # 44,940,844
atac_raw.columns = ['chr', 'start', 'end', 'barcode', 'count']
atac_raw = atac_raw[atac_raw['chr'].isin(['chr'+str(i) for i in range(1, 23)])]  # chr1-chr22  43,668,969
atac_raw['peak_id'] = atac_raw['chr']+':'+atac_raw['start'].astype(str)+'-'+atac_raw['end'].astype(str)

cCREs = pd.read_table('human_cCREs.bed', names=['chr', 'start', 'end'])
cCREs['idx'] = range(cCREs.shape[0])
cCREs_bed = pybedtools.BedTool.from_dataframe(cCREs)

m = np.zeros([2500, cCREs_bed.to_dataframe().shape[0]], dtype='float32')
## ~1.5h
for i in tqdm(range(2500)):
  atac_tmp = atac_raw[atac_raw['barcode']==rna_new.obs.index[i]+'-1']
  
  peaks = atac_tmp[['chr', 'start', 'end']]
  peaks['idx'] = range(peaks.shape[0])
  
  peaks_bed = pybedtools.BedTool.from_dataframe(peaks)
  idx_map = peaks_bed.intersect(cCREs_bed, wa=True, wb=True).to_dataframe().iloc[:, [3, 7]]
  idx_map.columns = ['peaks_idx', 'cCREs_idx']
  
  m[i][idx_map['cCREs_idx'].values] = 1

atac_new_var = pd.DataFrame({'gene_ids': cCREs['chr'] + ':' + cCREs['start'].map(str) + '-' + cCREs['end'].map(str), 'feature_types': 'Peaks'})
atac_new = sc.AnnData(m, obs=rna_new.obs, var=atac_new_var)
atac_new.var.index = atac_new.var['gene_ids'].values  # set index
atac_new.X = csr_matrix(m)

atac_new.write('spatial_atac.h5ad')





## spatial_rna_atac_eval.py
import sys
import os
sys.path.append("M2Mmodel")
import torch
from M2M import M2M_rna2atac
from utils import *
import scanpy as sc
import matplotlib.pyplot as plt
from torchmetrics import AUROC
from torcheval.metrics.functional import binary_auprc

test_atac_file = '/fs/home/jiluzhang/2023_nature_RF/human_brain/spatial_atac.h5ad'
test_rna_file = "/fs/home/jiluzhang/2023_nature_RF/human_brain/spatial_rna.h5ad"

enc_max_len = 38244
dec_max_len = 1033239

batch_size = 1
rna_max_value = 255
attn_window_size = 2*2048

device = torch.device("cuda:0")

test_atac = sc.read_h5ad(test_atac_file)
test_atac.X = test_atac.X.toarray()
test_rna = sc.read_h5ad(test_rna_file)
test_rna.X = test_rna.X.toarray()
test_rna.X = np.round(test_rna.X/(test_rna.X.sum(axis=1, keepdims=True))*10000)
test_rna.X[test_rna.X>255] = 255

# parameters for dataloader construction
test_kwargs = {'batch_size': batch_size, 'shuffle': True}
cuda_kwargs = {
    # 'pin_memory': True,  # 将加载的数据张量复制到 CUDA 设备的固定内存中，提高效率但会消耗更多内存
    "num_workers": 4,
    'prefetch_factor': 2
}  
test_kwargs.update(cuda_kwargs)

test_dataset = FullLenPairDataset(test_rna.X, test_atac.X, rna_max_value)
test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)

model = M2M_rna2atac(
            dim = 128, #Luz 16,
            enc_num_tokens = rna_max_value + 1,
            enc_seq_len = enc_max_len,
            enc_depth = 2,
            enc_heads = 4,
            enc_attn_window_size = attn_window_size,
            enc_dilation_growth_rate = 2,
            enc_ff_mult = 4,

            latent_len = 256, #Luz 256, 
            num_middle_MLP = 1, #Luz 2, 
            latent_dropout = 0.1, #Luz 0.1,

            dec_seq_len = dec_max_len,
            dec_depth = 2,
            dec_heads = 8,
            dec_attn_window_size = attn_window_size,
            dec_dilation_growth_rate = 5,
            dec_ff_mult = 4
)

model.load_state_dict(torch.load('/fs/home/jiluzhang/scM2M_v2/models_32124_lr_0.00001_epoch_3/32124_cells_10000_epoch_1/pytorch_model.bin'))
model.to(device)

auroc_lst = []
auprc_lst = []
auroc = AUROC(task='binary')

model.eval()
with torch.no_grad():
  data_iter = enumerate(test_loader)
  for i, (src, tgt) in data_iter:
    src = src.long().to(device)
    tgt = tgt.float().to(device)
    logist = model(src)
    
    logist = logist
    tgt = tgt
    auroc_lst.append(auroc(logist, tgt).item())
    auprc_lst.append(binary_auprc(torch.sigmoid(logist), tgt, num_tasks = logist.shape[0]).item())
    print('cell', i, 'done')

out = pd.DataFrame({'auroc': auroc_lst, 'auprc': auprc_lst})
out.to_csv('spatial_rna_atac_auroc_auprc.txt', index=False, sep='\t')



## 2124_rna_atac_eval.py
import sys
import os
sys.path.append("M2Mmodel")
import torch
from M2M import M2M_rna2atac
from utils import *
import scanpy as sc
import matplotlib.pyplot as plt
from torchmetrics import AUROC
from torcheval.metrics.functional import binary_auprc

test_atac_file = 'atac_32124_shuf_4.h5ad'
test_rna_file = "rna_32124_shuf_4.h5ad"

enc_max_len = 38244
dec_max_len = 1033239

batch_size = 1
rna_max_value = 255
attn_window_size = 2*2048

device = torch.device("cuda:0")

test_atac = sc.read_h5ad(test_atac_file)
test_atac.X = test_atac.X.toarray()
test_rna = sc.read_h5ad(test_rna_file)
test_rna.X = test_rna.X.toarray()
test_rna.X = np.round(test_rna.X/(test_rna.X.sum(axis=1, keepdims=True))*10000)
test_rna.X[test_rna.X>255] = 255

# parameters for dataloader construction
test_kwargs = {'batch_size': batch_size, 'shuffle': True}
cuda_kwargs = {
    # 'pin_memory': True,  # 将加载的数据张量复制到 CUDA 设备的固定内存中，提高效率但会消耗更多内存
    "num_workers": 4,
    'prefetch_factor': 2
}  
test_kwargs.update(cuda_kwargs)

test_dataset = FullLenPairDataset(test_rna.X, test_atac.X, rna_max_value)
test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)

model = M2M_rna2atac(
            dim = 128, #Luz 16,
            enc_num_tokens = rna_max_value + 1,
            enc_seq_len = enc_max_len,
            enc_depth = 2,
            enc_heads = 4,
            enc_attn_window_size = attn_window_size,
            enc_dilation_growth_rate = 2,
            enc_ff_mult = 4,

            latent_len = 256, #Luz 256, 
            num_middle_MLP = 1, #Luz 2, 
            latent_dropout = 0.1, #Luz 0.1,

            dec_seq_len = dec_max_len,
            dec_depth = 2,
            dec_heads = 8,
            dec_attn_window_size = attn_window_size,
            dec_dilation_growth_rate = 5,
            dec_ff_mult = 4
)

model.load_state_dict(torch.load('/fs/home/jiluzhang/scM2M_v2/models_32124_lr_0.00001_epoch_3/32124_cells_10000_epoch_1/pytorch_model.bin'))
model.load_state_dict(torch.load('/fs/home/jiluzhang/scM2M_v2/models/test_loss_epoch_5/pytorch_model.bin'))
model.to(device)

auroc_lst = []
auprc_lst = []
auroc = AUROC(task='binary')


model.eval()
with torch.no_grad():
  #data_iter = enumerate(test_loader)
  i, (src, tgt) = next(data_iter)
  src = src.long().to(device)
  tgt = tgt.float().to(device)
  logist = model(src)

sum(src[0].to('cpu').numpy()!=0)
sum(tgt[0].to('cpu').numpy()!=0)
sum(torch.sigmoid(logist).to('cpu').numpy()[0]>0.9)
torch.sigmoid(logist)


model.eval()
with torch.no_grad():
  data_iter = enumerate(test_loader)
  for i, (src, tgt) in data_iter:
    src = src.long().to(device)
    tgt = tgt.float().to(device)
    logist = model(src)
    
    logist = logist
    tgt = tgt
    auroc_lst.append(auroc(logist, tgt).item())
    auprc_lst.append(binary_auprc(torch.sigmoid(logist), tgt, num_tasks = logist.shape[0]).item())
    print('cell', i, 'done')

out = pd.DataFrame({'auroc': auroc_lst, 'auprc': auprc_lst})
out.to_csv('2124_rna_atac_auroc_auprc.txt', index=False, sep='\t')


