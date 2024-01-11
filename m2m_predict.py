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

train_atac_file = "atac_test_pbmc_100.h5ad"
train_rna_file = "rna_test_pbmc_100.h5ad"
saved_model_path = "save_att/pytorch_model.bin"

enc_max_len = 38244
dec_max_len = 1036913

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


model = M2M_rna2atac(
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

model.load_state_dict(torch.load('/data/home/zouqihang/desktop/project/M2M/version3.2/save/2023-12-25_5tissue_testing_new_attention_weight/pytorch_model.bin'))
model.to(device)


src, tgt = next(iter(train_loader))
src = src.long().to(device)
tgt = tgt.to(device)

model.eval()
with torch.no_grad():
    enc_attn, dec_attn, logist = model(src, attn_matrix =True)

gene_name = train_rna.var.index
peak_name = train_atac.var['gene_ids']

x = gene_name[enc_attn["k_indices"]]
y = gene_name[enc_attn["q_indices"]]
#x = peak_name[dec_attn['self_attn']["k_indices"][0].to("cpu")]
#y = peak_name[dec_attn['self_attn']["q_indices"][0].to("cpu")]

# 调整图像尺寸
fig = plt.figure(figsize=(20, 20))
plt.imshow(enc_attn["attn"][0].to("cpu"), cmap='viridis', interpolation='nearest')

# 添加刻度值
plt.xticks(ticks = list(range(len(x))), labels = x, rotation=70, ha='right')
plt.yticks(ticks = list(range(len(y))), labels = y)

# 添加颜色条
plt.colorbar()  

plt.savefig('encoder_self_local_attn_2.png')





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





