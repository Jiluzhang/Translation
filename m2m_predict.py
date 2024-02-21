import sys
import tqdm

import torch
from torch import nn
from torch.utils.data import Dataset
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F

sys.path.append("M2Mmodel")

from M2M import M2M
# from sklearn.metrics import accuracy_score, precision_score, recall_score
from torchmetrics import AUROC; from torcheval.metrics.functional import binary_auprc
import datetime
import time
from torch.utils.tensorboard import SummaryWriter

# DDP 
# import New_Accelerate as Accelerator
from accelerate import Accelerator
from utils import *
from torch.cuda.amp import autocast as autocast


torch.autograd.set_detect_anomaly = True

import scanpy as sc


def main():
    test_atac_file = "atac_test_dm_100.h5ad"
    test_rna_file = "rna_test_dm_100.h5ad"

    enc_max_len = 38244
    dec_max_len = 1036913

    batch_size = 6
    rna_max_value = 255
    attn_window_size = 2*2048
    SEED = 2023
    lr = 1e-3
    gamma_step = 1  # Learning rate step (default: 2000 (not used))
    gamma = 1
    epoch = 5
    val_every = 1

    # init
    setup_seed(SEED)

    test_atac = read_data_from_luzhang(test_atac_file)
    test_rna = read_data_from_luzhang(test_rna_file)
    test_dataset = FullLenPairDataset(test_rna.matrix, test_atac.matrix, rna_max_value)
    test_loader = torch.utils.data.DataLoader(test_dataset)

    model = M2M(
        dim = 64,
        enc_num_tokens = rna_max_value + 1,
        enc_seq_len = enc_max_len,
        enc_depth = 2,
        enc_heads = 1,
        enc_attn_window_size = attn_window_size,
        enc_dilation_growth_rate = 2,
        enc_ff_mult = 4,
        
        latent_len = 256,
        num_middle_MLP = 2,
        latent_dropout = 0.1,

        dec_seq_len = dec_max_len,
        dec_depth = 2,
        dec_heads = 1,
        dec_attn_window_size = attn_window_size,
        dec_dilation_growth_rate = 5,
        dec_ff_mult = 4
    )
    model.load_state_dict(torch.load('03/save/2023-12-11_5tissue_testing/pytorch_model.bin'))

    model.eval()
    atac_ref = sc.read_h5ad('atac_test_dm_100.h5ad')
    atac_res = sc.AnnData(np.zeros((atac_ref.n_obs, atac_ref.n_vars)), obs=atac_ref.obs, var=atac_ref.var)
    y_hat_out = torch.Tensor().cuda()
    for i, (src, tgt) in enumerate(test_loader):
        src = src.long()
        y_hat = model(src)
        y_hat_out = torch.cat((y_hat_out, y_hat))
        atac_res.X = torch.sigmoid(y_hat_out).to("cpu").numpy()
    atac_res.write('atac_predicted.h5ad')

if __name__ == "__main__":
    main()





import sys
import os
sys.path.append("M2Mmodel")
import torch
from M2M import M2M_rna2atac
from utils import *
import scanpy as sc
import matplotlib.pyplot as plt

train_atac_file = "datasets/atac_3_pseudo_1000.h5ad"
train_rna_file = "datasets/rna_3_pseudo.h5ad"
#saved_model_path = "save_raw/2024-01-11_None_epoch_1/pytorch_model.bin"

enc_max_len = 38244
dec_max_len = 1033239  #1036913

batch_size = 1
rna_max_value = 255
attn_window_size = 2*2048
SEED = 2023

device = torch.device("cuda:0")

train_atac = sc.read_h5ad(train_atac_file)
train_rna = sc.read_h5ad(train_rna_file)

# parameters for dataloader construction
train_kwargs = {'batch_size': batch_size, 'shuffle': True}
cuda_kwargs = {
    # 'pin_memory': True,  # 将加载的数据张量复制到 CUDA 设备的固定内存中，提高效率但会消耗更多内存
    "num_workers": 4,
    'prefetch_factor': 2
}  
train_kwargs.update(cuda_kwargs)

# 将数据加载到dataloader，后面这里需要加代码来根据输入数据类型调整使用策略
# rna: value, id, pad_mask.
# target_atac: 没有任何修饰
# sepcial tokens: {"<PAD>": 0, "<MASK>": 1, "<unk>": 2}
train_dataset = FullLenPairDataset(train_rna.X, train_atac.X, rna_max_value)
train_loader = torch.utils.data.DataLoader(train_dataset, **train_kwargs)


model_1 = M2M_rna2atac(
    dim = 256,
    enc_num_tokens = rna_max_value + 1,
    enc_seq_len = enc_max_len,
    enc_depth = 2,
    enc_heads = 8,
    enc_attn_window_size = attn_window_size,
    enc_dilation_growth_rate = 2,
    enc_ff_mult = 4,

    dec_seq_len = dec_max_len,
    dec_depth = 2,
    dec_heads = 8,
    dec_attn_window_size = attn_window_size,
    dec_dilation_growth_rate = 5,
    dec_ff_mult = 4
)

model_2 = M2M_rna2atac(
    dim = 16,
    enc_num_tokens = rna_max_value + 1,
    enc_seq_len = enc_max_len,
    enc_depth = 1,
    enc_heads = 1,
    enc_attn_window_size = attn_window_size,
    enc_dilation_growth_rate = 2,
    enc_ff_mult = 4,
    
    dec_seq_len = dec_max_len,
    dec_depth = 1,
    dec_heads = 1,
    dec_attn_window_size = attn_window_size,
    dec_dilation_growth_rate = 5,
    dec_ff_mult = 4
)

#model.load_state_dict(torch.load('/data/home/zouqihang/desktop/project/M2M/version3.2/save/2023-12-25_5tissue_testing_new_attention_weight/pytorch_model.bin'))
#model.load_state_dict(torch.load('/fs/home/jiluzhang/scM2M_v2/save_raw/2024-01-11_None_epoch_1/pytorch_model.bin'))
model_1.load_state_dict(torch.load('/fs/home/jiluzhang/scM2M_v2/save/2024-01-17_None_epoch_-1/pytorch_model.bin'))
model_2.load_state_dict(torch.load('/fs/home/jiluzhang/scM2M_v2/save/2024-01-17_None_epoch_0/pytorch_model.bin'))

model_1.load_state_dict(torch.load('/fs/home/jiluzhang/scM2M_nonmlp/save/2024-01-23_None/pytorch_model_1.bin'))
model_2.load_state_dict(torch.load('/fs/home/jiluzhang/scM2M_nonmlp/save/2024-01-23_None/pytorch_model.bin'))

model.to(device)

model.state_dict()['enc.longnet.net.layers.0.0.fn.to_v.weight']
model.state_dict()['trans.model.0.0.weight']

for i in model_1.state_dict():
    print(i, model_1.state_dict()[i].shape)
    #print(i)

for i in model_1.state_dict():
    if model_1.state_dict()[i].equal(model_2.state_dict()[i]):
        print(i)

param_sum = 0
for name, param in model_1.named_parameters():
    if param.requires_grad:
        param_sum += np.prod(list(param.shape))
    
for name, param in model_1.named_parameters():
    if name=='dec.longnet.net.layers.0.0.fn.to_v.weight':
        print(name, param)


############### pseudo cell ###############
import numpy as np
import scanpy as sc

rna = sc.read_h5ad('rna_10.h5ad')
rna.X = np.random.randint(0,255, (10, 38244))
rna.write('rna_10_pseudo.h5ad')

atac = sc.read_h5ad('atac_10.h5ad')
atac.X[atac.X>=100] = 1
atac.X[atac.X!=1] = 0
atac.write('atac_10_pseudo.h5ad')


rna = sc.read_h5ad('rna_10.h5ad')
rna.X = np.random.randint(0,255, (10, 38244))
rna.write('rna_10_pseudo_random.h5ad')

atac = sc.read_h5ad('atac_10.h5ad')
atac.X[0] = [1]*1033239
atac.X[1] = [0]*1033239
atac.X[2] = [1]*1033239
atac.X[3] = [0]*1033239
atac.X[4] = [1]*1033239
atac.write('atac_10_pseudo_random.h5ad')
############################################


device = 'cuda:1'
src, tgt = next(iter(train_loader))
src = src.long().to(device)
tgt = tgt.to(device)
model.to(device)

model.eval()
with torch.no_grad():
    enc_attn, dec_attn, logist = model(src, attn_matrix =True)

gene_name = train_rna.var.index
peak_name = train_atac.var.index

x = gene_name[enc_attn["k_indices"]]
y = gene_name[enc_attn["q_indices"]]
#x = peak_name[dec_attn['self_attn']["k_indices"]]
#y = peak_name[dec_attn['self_attn']["q_indices"]]

# 调整图像尺寸
fig = plt.figure(figsize=(20, 20))
plt.imshow(enc_attn["attn"][0].to("cpu"), cmap='Reds', interpolation='nearest')
#plt.imshow(dec_attn["self_attn"]['attn'][0].to("cpu"), cmap='Reds', interpolation='nearest')

# 添加刻度值
plt.xticks(ticks = list(range(len(x))), labels = x, rotation=70, ha='right')
plt.yticks(ticks = list(range(len(y))), labels = y)

# 添加颜色条
plt.colorbar()  

#plt.savefig('gene_attention_tmp_1.png')
plt.savefig('peak_attention_10_cells_3.png')
plt.close()

src = torch.randint(0,255,(1,38244)).to(device)


#CE347E1-S1K1: 4126 cells
#CE349E1-S1: 5323 cells

import scanpy as sc
import anndata as ad

## concat rna
s1_rna = sc.read_h5ad('CE347E1-S1K1_rna.h5ad')
s1_rna.obs.index = s1_rna.obs.index + '-s1'
s2_rna = sc.read_h5ad('CE349E1-S1_rna.h5ad')
s2_rna.obs.index = s2_rna.obs.index + '-s2'
rna = ad.concat([s2_rna, s1_rna], axis=0)
rna.var = s1_rna.var
rna.X = rna.X.toarray()
rna.write('rna_10k.h5ad')  # 9449 × 38244

## concat atac
s1_atac = sc.read_h5ad('CE347E1-S1K1_atac.h5ad')
s1_atac.obs.index = s1_atac.obs.index + '-s1'
s2_atac = sc.read_h5ad('CE349E1-S1_atac.h5ad')
s2_atac.obs.index = s2_atac.obs.index + '-s2'
atac = ad.concat([s2_atac, s1_atac], axis=0)
atac.var = s1_atac.var
atac.X = atac.X.toarray()
atac.write('atac_10k.h5ad')






import sys
import os
sys.path.append("M2Mmodel")
import torch
from M2M import M2M_rna2atac
from utils import *
import scanpy as sc
import matplotlib.pyplot as plt

train_atac_file = "atac_10.h5ad"
train_rna_file = "rna_10.h5ad"

enc_max_len = 38244
dec_max_len = 1033239

batch_size = 1
rna_max_value = 255
attn_window_size = 2*2048
SEED = 2023

device = torch.device("cuda:0")

train_atac = sc.read_h5ad(train_atac_file)
train_rna = sc.read_h5ad(train_rna_file)
train_rna.X = np.round(train_rna.X/(train_rna.X.sum(axis=1, keepdims=True))*10000)
train_rna.X[train_rna.X>255] = 255

# parameters for dataloader construction
train_kwargs = {'batch_size': batch_size, 'shuffle': True}
cuda_kwargs = {
    # 'pin_memory': True,  # 将加载的数据张量复制到 CUDA 设备的固定内存中，提高效率但会消耗更多内存
    "num_workers": 4,
    'prefetch_factor': 2
}  
train_kwargs.update(cuda_kwargs)

train_dataset = FullLenPairDataset(train_rna.X, train_atac.X, rna_max_value)
train_loader = torch.utils.data.DataLoader(train_dataset, **train_kwargs)

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

#model.load_state_dict(torch.load('/fs/home/jiluzhang/scM2M_v2/save/10_cells/2024-02-01_None_epoch_9/pytorch_model.bin'))
model.load_state_dict(torch.load('/fs/home/jiluzhang/scM2M_v2/save/100_cells/2024-02-01_None_epoch_4/pytorch_model.bin'))
#model.load_state_dict(torch.load('/fs/home/jiluzhang/scM2M_v2/save/1k_cells/2024-02-01_None_epoch_1/pytorch_model.bin'))

#src, tgt = next(iter(train_loader))
rna = sc.read_h5ad('rna_10.h5ad')
src = torch.from_numpy(rna.X[[1]]).long().to(device)
#tgt = tgt.to(device)
model.to(device)

model.eval()
with torch.no_grad():
    enc_attn, dec_attn, logist = model(src, attn_matrix =True)

gene_name = train_rna.var.index
peak_name = train_atac.var.index

x = gene_name[enc_attn["k_indices"]]
y = gene_name[enc_attn["q_indices"]]

# 调整图像尺寸
fig = plt.figure(figsize=(20, 20))
plt.imshow(enc_attn["attn"][0].to("cpu"), cmap='Reds', interpolation='nearest')

# 添加刻度值
plt.xticks(ticks = list(range(len(x))), labels = x, rotation=70, ha='right')
plt.yticks(ticks = list(range(len(y))), labels = y)

# 添加颜色条
plt.colorbar()  

plt.savefig('gene_attention_100_cells_2.png')
plt.close()



x = peak_name[dec_attn['self_attn']["k_indices"]]
y = peak_name[dec_attn['self_attn']["q_indices"]]

# 调整图像尺寸
fig = plt.figure(figsize=(20, 20))
plt.imshow(dec_attn["self_attn"]['attn'][0].to("cpu"), cmap='Reds', interpolation='nearest')

# 添加刻度值
plt.xticks(ticks = list(range(len(x))), labels = x, rotation=70, ha='right')
plt.yticks(ticks = list(range(len(y))), labels = y)

# 添加颜色条
plt.colorbar()  

plt.savefig('peak_attention_100_cells_2.png')
plt.close()



x = gene_name[dec_attn['cross_attn']["k_indices"]]
y = peak_name[dec_attn['cross_attn']["q_indices"]]

# 调整图像尺寸
fig = plt.figure(figsize=(20, 20))
plt.imshow(dec_attn['cross_attn']["attn"][0].to("cpu"), cmap='Reds', interpolation='nearest')

# 添加刻度值
plt.xticks(ticks = list(range(len(x))), labels = x, rotation=70, ha='right')
plt.yticks(ticks = list(range(len(y))), labels = y)

# 添加颜色条
plt.colorbar()  

plt.savefig('gene_peak_attention_100_cells_2.png')
plt.close()




for i in range(enc_attn["attn"][0].shape[0]):
  enc_attn['attn'][0][i][i] = 0

for i in range(dec_attn["self_attn"]['attn'][0].shape[0]):
  dec_attn["self_attn"]['attn'][0][i][i] = 0
  
for i in range(train_rna.X.shape[1]):
  if sum(train_rna.X[:, i])>10:
    print(i)


## normalization
rna.X = rna.X/(rna.X.sum(axis=1, keepdims=True))*10000




#################### Evaluation ###################
import sys
import os
sys.path.append("M2Mmodel")
import torch
from M2M import M2M_rna2atac
from utils import *
import scanpy as sc
import matplotlib.pyplot as plt
from torchmetrics import AUROC; from torcheval.metrics.functional import binary_auprc



train_atac_file = "PM1380P1-T1_atac.h5ad"
train_rna_file = "PM1380P1-T1_rna.h5ad"

enc_max_len = 38244
dec_max_len = 1033239

batch_size = 1
rna_max_value = 255
attn_window_size = 2*2048
SEED = 2023

device = torch.device("cuda:0")

train_atac = sc.read_h5ad(train_atac_file)
train_atac.X = train_atac.X.toarray()
train_rna = sc.read_h5ad(train_rna_file)
train_rna.X = train_rna.X.toarray()
train_rna.X = np.round(train_rna.X/(train_rna.X.sum(axis=1, keepdims=True))*10000)
train_rna.X[train_rna.X>255] = 255

# parameters for dataloader construction
train_kwargs = {'batch_size': batch_size, 'shuffle': True}
cuda_kwargs = {
    # 'pin_memory': True,  # 将加载的数据张量复制到 CUDA 设备的固定内存中，提高效率但会消耗更多内存
    "num_workers": 4,
    'prefetch_factor': 2
}  
train_kwargs.update(cuda_kwargs)

train_dataset = FullLenPairDataset(train_rna.X, train_atac.X, rna_max_value)
train_loader = torch.utils.data.DataLoader(train_dataset, **train_kwargs)

auroc = AUROC(task='binary')

model_1 = M2M_rna2atac(
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

model_2 = M2M_rna2atac(
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

model_3 = M2M_rna2atac(
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


src, tgt = next(iter(train_loader))
src = src.long().to(device)
tgt = tgt.float().to(device)

model_1.load_state_dict(torch.load('/fs/home/jiluzhang/scM2M_v2/models_epoch_1/4/pytorch_model.bin'))
model_2.load_state_dict(torch.load('/fs/home/jiluzhang/scM2M_v2/models_epoch_2/4/pytorch_model.bin'))
model_3.load_state_dict(torch.load('/fs/home/jiluzhang/scM2M_v2/models/4/pytorch_model.bin'))

model_1.to(device)
model_2.to(device)
model_3.to(device)
model_1.eval()
model_2.eval()
model_3.eval()

with torch.no_grad():
  logist_1 = model_1(src)
  logist_2 = model_2(src)
  logist_3 = model_3(src)
  print("Model 1 ", "AUROC:", auroc(logist_1, tgt).item(), "AUPRC:", binary_auprc(torch.sigmoid(logist_1), tgt, num_tasks = logist_1.shape[0]).mean().item())
  print("Model 2 ", "AUROC:", auroc(logist_2, tgt).item(), "AUPRC:", binary_auprc(torch.sigmoid(logist_2), tgt, num_tasks = logist_2.shape[0]).mean().item())
  print("Model 3 ", "AUROC:", auroc(logist_3, tgt).item(), "AUPRC:", binary_auprc(torch.sigmoid(logist_3), tgt, num_tasks = logist_3.shape[0]).mean().item())
  

model.to(device)

for i in range(1, 5):
  model.load_state_dict(torch.load('/fs/home/jiluzhang/scM2M_v2/models/'+str(i)+'/pytorch_model.bin'))
  #model.eval()
  with torch.no_grad():
    logist = model(src)
  print(i, "AUROC:", auroc(logist, tgt).item(), "AUPRC:", binary_auprc(torch.sigmoid(logist), tgt, num_tasks = logist.shape[0]).mean().item())




## combine samples
import pandas as pd
import scanpy as sc
import anndata as ad
import numpy as np

samples = pd.read_table('normal_samples_id.txt', header=None)
rna_0 = sc.read_h5ad('/fs/home/jiluzhang/2023_nature_LD/normal_h5ad/'+samples[0][0]+'_rna.h5ad')
atac_0 = sc.read_h5ad('/fs/home/jiluzhang/2023_nature_LD/normal_h5ad/'+samples[0][0]+'_atac.h5ad')
rna = rna_0.copy()
atac = atac_0.copy()

for i in range(1, samples.shape[0]):
    rna_tmp = sc.read_h5ad('/fs/home/jiluzhang/2023_nature_LD/normal_h5ad/'+samples[0][i]+'_rna.h5ad')
    atac_tmp = sc.read_h5ad('/fs/home/jiluzhang/2023_nature_LD/normal_h5ad/'+samples[0][i]+'_atac.h5ad')
    rna = ad.concat([rna, rna_tmp])
    atac = ad.concat([atac, atac_tmp])
    print(samples[0][i], 'done')

rna.var = rna_0.var
atac.var = atac_0.var

# CE336E1-S1	CEAD    CE336E1-S1     828
# CE348E1-S1K1	CESC    CE348E1-S1K1   3702
# CM354C2-T1Y2	CRC     CM354C2-T1     8475
# CPT2373DU-S1	UCEC    CPT2373DU-S1   2012
# HT181P1-T1A3	PDAC    HT181P1-T1A3   2969
# HT263B1-S1H1	BRCA    HT263B1-S1H1   4021
# ML123M1-Ty1	SKCM    ML123M1-Ty1    2442
# P5504-N1	    HNSCC   P5504-N1       2336
# PM581P1-T1N1	PDAC    PM581P1-T1     2567
# VF034V1-T1Y1	OV      VF034V1-T1     2772
# Total count: 32,124

samples = pd.read_table('normal_samples_id.txt', header=None)
rna_0 = sc.read_h5ad('/fs/home/jiluzhang/2023_nature_LD/normal_h5ad/CE336E1-S1_rna.h5ad')
rna_0.obs.index = 'CE336E1-S1-'+rna_0.obs.index  # for duplicated cell barcode
atac_0 = sc.read_h5ad('/fs/home/jiluzhang/2023_nature_LD/normal_h5ad/CE336E1-S1_atac.h5ad')
atac_0.obs.index = 'CE336E1-S1-'+atac_0.obs.index
rna = rna_0.copy()
atac = atac_0.copy()

for sample in ['CE348E1-S1K1', 'CM354C2-T1', 'CPT2373DU-S1', 'HT181P1-T1A3', 'HT263B1-S1H1',
               'ML123M1-Ty1', 'P5504-N1', 'PM581P1-T1', 'VF034V1-T1']:
    rna_tmp = sc.read_h5ad('/fs/home/jiluzhang/2023_nature_LD/normal_h5ad/'+sample+'_rna.h5ad')
    rna_tmp.obs.index = sample+'-'+rna_tmp.obs.index
    atac_tmp = sc.read_h5ad('/fs/home/jiluzhang/2023_nature_LD/normal_h5ad/'+sample+'_atac.h5ad')
    atac_tmp.obs.index = sample+'-'+atac_tmp.obs.index
    rna = ad.concat([rna, rna_tmp])
    atac = ad.concat([atac, atac_tmp])
    print(sample, 'done')

rna.var = rna_0.var
atac.var = atac_0.var
rna.write('rna_32124.h5ad')
atac.write('atac_32124.h5ad')

np.random.seed(0)
shuf_idx = np.arange(rna.shape[0])
np.random.shuffle(shuf_idx)
rna[shuf_idx, :].write('rna_32124_shuf.h5ad')
atac[shuf_idx, :].write('atac_32124_shuf.h5ad')

# rna_32124_shuf_10000.h5ad   # head 10000
# atac_32124_shuf_10000.ha5d  # head 10000

accelerate launch --config_file default_config.yaml rna2atac_pre-train_v2.py --atac atac_32124_shuf.h5ad --rna rna_32124_shuf.h5ad --save models --name 32124_cells 


# rna_32124_shuf_10000_2.h5ad   # 10000-20000
# atac_32124_shuf_10000_2.ha5d  # 10000-20000

python rna2atac_pre-train_v2.py --atac atac_32124_shuf_10000_2.h5ad --rna rna_32124_shuf_10000_2.h5ad --save models --name 32124_cells \
                                --load /fs/home/jiluzhang/scM2M_v2/models_32124_lr_0.00001_epoch_1/32124_cells_10000_epoch_1/pytorch_model.bin


