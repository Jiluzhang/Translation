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
