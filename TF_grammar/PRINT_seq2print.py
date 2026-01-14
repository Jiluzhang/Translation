########## train seq2print model ##########
## workdir: /fs/home/jiluzhang/TF_grammar/scPrinter/20260114/seq2print
## datasets.py (mask some code)
## env: scPrinter
# cp /fs/home/jiluzhang/TF_grammar/scPrinter/scPrinter-main/scprinter/seq/scripts/seq2print_lora_train.py .
# peaks.bed 
# chr21	13979412	13980412
# chr21	13979754	13980754
# chr21	14026837	14027837
# chr21	14115899	14116899
# chr21	14119816	14120816
# chr21	14121452	14122452
# chr21	14215610	14216610
# chr21	14240957	14241957
# chr21	14253357	14254357
# chr21	14259364	14260364
# cp ../test/PBMC_bulkATAC_Bcell_0_fold0.JSON config.JSON
# cp ../test/PBMC_bulkATAC_scprinter_supp_raw/Bcell_0.bw .

# files in cache dir for import scprinter: CisBP_Human_FigR  CisBP_Human_FigR_meme  CisBP_Mouse_FigR  CisBP_Mouse_FigR_meme TFBS_0_conv_v2.pt  TFBS_1_conv_v2.pt
# files for seq2print model training: hg38_bias_v2.h5 gencode_v41_GRCh38.fa.gz

CUDA_VISIBLE_DEVICES=7 python seq2print_lora_train.py --config ./config.JSON --temp_dir ./temp  --model_dir ./model \


########## train TFBS model ##########
## workdir: /fs/home/jiluzhang/TF_grammar/scPrinter/20260114/tfbs
# cp /fs/home/jiluzhang/TF_grammar/scPrinter/test/luz/Unibind/data/TFBSPrediction/HepG2_pred_data.tsv .
# cp /fs/home/jiluzhang/TF_grammar/scPrinter/test/luz/Unibind/data/HepG2/test_peaks.bed .
#### TFBS_finetune.ipynb
## conda activate scPrinter
# reload imported modules
%load_ext autoreload
%autoreload 2

import scprinter as scp
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
import time
import pandas as pd
import numpy as np
import os
import time
import re
import pyBigWig
import numpy as np
import pandas as pd
from concurrent.futures import *
from tqdm.auto import *
import pickle

## split range to chrom & start & end
def read_TF_loci(path):
    TF_prediction = pd.read_csv(path, sep='\t')
    if 'range' in TF_prediction.columns:
        ranges = pd.DataFrame([re.split(':|-', xx) for xx in TF_prediction['range']])
        ranges = ranges[[0, 1, 2, 3]]
        v = np.array(ranges[3])
        v[v==''] = '-'
        ranges[3] = v
        ranges.columns = ['chrom', 'start', 'end', 'strand']
        ranges['start'] = ranges['start'].astype('int')
        ranges['end'] = ranges['end'].astype('int')
        TF_prediction = pd.concat([TF_prediction, ranges], axis=1)
    else:
        TF_prediction = pd.read_csv(path, sep='\t', header=None)
        TF_prediction.columns = ['chrom','start','end', 'bound', 'TF']
        TF_prediction['strand'] = '+'
    TF_prediction['summit'] = (TF_prediction['start']-1 + TF_prediction['end']) // 2
    return TF_prediction

def read_peaks(path):
    df = pd.read_csv(path, sep='\t', header=None)
    df.columns = ['chrom', 'start', 'end']
    return df

def get_normalization_factor(model, feat, peak, low=0.05, median = 0.5, high=0.95, sample_num=30000):
    vs = []
    with pyBigWig.open(f"{model}/{feat}", 'r') as f:
        chroms = f.chroms().keys()
        print (chroms)
        peaks = peak[peak['chrom'].isin(chroms)].copy()
        peaks['summit'] = (peaks['start'] + peaks['end']) // 2
        peaks['start'] = peaks['summit'] - 400
        peaks['end'] = peaks['summit'] + 400
        if sample_num > 0:
            peaks = peaks.sample(sample_num, replace=True).copy()
        for chrom, start, end in zip(peaks['chrom'], peaks['start'], peaks['end']):
            v = f.values(chrom, int(start), int(end), numpy=True)
            vs.append(v)
            
    vs = np.concatenate(vs)
    vs = np.nan_to_num(vs)
    return np.quantile(vs, low), np.quantile(vs, median), np.quantile(vs, high)

# wget -c https://raw.githubusercontent.com/ruochiz/FigRmotifs/main/human_pfms_v4.txt
motifs = scp.motifs.Motifs("./human_pfms_v4.txt", scp.genome.hg38.fetch_fa(), scp.genome.hg38.bg)
motif2matrix = {motif.name.split("_")[2]: np.array([motif.counts['A'], motif.counts['C'], motif.counts['G'], motif.counts['T']]) for motif in motifs.all_motifs}

## normalization to keep colsum of 1
for m in motif2matrix:
    mm = motif2matrix[m]
    mm = mm / np.sum(mm, axis=0, keepdims=True)
    motif2matrix[m] = mm

tsv_paths = {'HepG2': "./HepG2_pred_data.tsv"}
peaks_path = {'HepG2': "./test_peaks.bed"}

tsvs = {cell:read_TF_loci(tsv_paths[cell]) for cell in tsv_paths}
peaks = {cell:read_peaks(peaks_path[cell]) for cell in peaks_path}
model_name = {'HepG2': ["./"]}


feats = ['attr.count.shap_hypo_0_.0.85.bigwig', 'attr.just_sum.shap_hypo_0-30_.0.85.bigwig']  # bigwig generated after model training finishied

pool= ProcessPoolExecutor(max_workers=3)
norm_factor = {}
p_list = []
for cell in model_name:
    norm_factor[cell] = {}
    for feat in feats:
        norm_factor[cell][feat] = []
        for model in model_name[cell]:
            p = pool.submit(get_normalization_factor, model, feat, peaks[cell], sample_num=-1)
            norm_factor[cell][feat].append(p)
            p_list.append(p)

for cell in model_name:
    for feat in feats:
        for i, model in enumerate(model_name[cell]):
            norm_factor[cell][feat][i] = norm_factor[cell][feat][i].result()

for cell in model_name:
    for feat in feats:
        for i, model in enumerate(model_name[cell]):
            lo, mid, hi = norm_factor[cell][feat][i]
            print(cell, feat, i, "%.2f" % mid, "%.2f"%(hi-lo))

from pyfaidx import Fasta
from scipy.stats import *

def cosine(x,y):
    return (x*y).sum() / (np.linalg.norm(x) * np.linalg.norm(y))

def get_feats(model, feat, tsv,
              flank,
              low=None, median=None, high=None, chroms = None,
             verbose=False):
    if low is None:
        low = 0
        median = 0
        high = 1
    fasta = Fasta(scp.genome.hg38.fetch_fa())
    signals = []
    similarity = []
    strengths = []
    with pyBigWig.open(f"{model}/{feat}", 'r') as f:
        if chroms is None:
            chroms = f.chroms().keys()
        tsv = tsv[tsv['chrom'].isin(chroms)].copy()
        chrom, summit, strand = np.array(tsv['chrom']), np.array(tsv['summit']), np.array(tsv['strand'])
        starts, ends, strands = np.array(tsv['start']), np.array(tsv['end']), np.array(tsv['strand'])
        tfs = np.array(tsv['TF'])
        labels = np.array(tsv['bound'])

        for c,s,sd,tf, start,end in zip(tqdm(chrom, disable=not verbose), summit, strands, tfs, starts, ends):
            
            try:
                v = f.values(c, s-flank, s+flank+1, numpy=True)
            except:
                print (c, s-flank, s+flank+1)
                v = np.zeros((int(2*flank+1)))
            if sd != '+':
                v = v[::-1]

            # seq = hg38.fetch_onehot_seq(c, start-1, end).numpy()
            seq = fasta[c][start-1:end].seq
            seq = scp.utils.DNA_one_hot(seq).numpy()
            seq_trend = f.values(c, start-1, end, numpy=True)
            if sd == '-':
                seq = seq[::-1][:, ::-1]
                seq_trend = seq_trend[::-1]
            motif_trend = (motif2matrix[tf] * seq).sum(axis=0)
            motif_match = cosine(motif2matrix[tf].reshape((-1)), seq.reshape((-1)))
            
            signals.append(v)
            similarity.append([cosine(seq_trend, motif_trend), 
                               # motif_match,
                               ((seq_trend - median) / (high - low)).mean()])
    
    signals = np.array(signals)
    similarity = np.array(similarity)
    signals = np.nan_to_num(signals)
    similarity =  np.nan_to_num(similarity)

    signals = (signals - median) / (high - low)
    return signals, similarity, tsv


feats_all = {}
labels_all = {}
pool= ProcessPoolExecutor(max_workers=3)
p_list = []
for cell in model_name:
    for feat in feats:
        for i, model in enumerate(model_name[cell]):
            lo, mid, hi = None, None, None
            p_list.append(pool.submit(get_feats, model, feat, tsvs[cell], 100, lo, mid, hi))

ct = 0
for cell in model_name:
    feats_cell = []
    sims_cell = []
    for feat in feats:
        feats_v = 0
        sims_v = 0
        for i, model in enumerate(model_name[cell]):
            feats_, sims_, labels = p_list[ct].result()
            ct += 1
            feats_v += feats_
            sims_v += sims_
        feats_v = feats_v / len(model_name[cell])
        sims_v = sims_v / len(model_name[cell])
        feats_cell.append(feats_v)
        sims_cell.append(sims_v)
    feats_cell = np.stack(feats_cell, axis=1)
    sims_cell = np.stack(sims_cell, axis=1)
    feats_all[cell] = [feats_cell, sims_cell]
    labels_all[cell] = labels

pool.shutdown(wait=True)

from sklearn.model_selection import train_test_split
hepg2_train, hepg2_val = train_test_split(np.arange(len(labels_all['HepG2'])), test_size=0.1, random_state=42)
gm_train, gm_val = train_test_split(np.arange(len(labels_all['HepG2'])), test_size=0.1, random_state=0)

train_cells_hepg2 = ['HepG2']
train_cells_gm12878 = ['HepG2']

def feats_for_cell(feats_all, labels_all, cell, window=201, n_feats=np.arange(8)):
    seq_attr = feats_all[cell][0]
    pad = (seq_attr.shape[-1] - window) // 2
    if pad > 0:
        seq_attr = seq_attr[..., pad:-pad]
    # seq_attr = seq_attr[:, n_feats].reshape((len(seq_attr), -1))
    seq_motif_sim = feats_all[cell][1][:, n_feats]
    motifs = np.asarray(labels_all[cell]['motifMatchScore'])[:, None]
    motifs = np.stack([motifs] * len(n_feats), axis=1)
    # feats = seq_motif_sim.reshape((len(seq_attr), -1))
    feats = np.concatenate([motifs, seq_motif_sim, seq_attr], axis=-1)
    print (motifs.shape, seq_motif_sim.shape, seq_attr.shape)
    return feats

train_feats = np.concatenate([feats_for_cell(feats_all, labels_all, cell,n_feats=np.arange(2))[hepg2_train] for cell in train_cells_hepg2] + 
                             [feats_for_cell(feats_all, labels_all, cell,n_feats=np.arange(2))[gm_train] for cell in train_cells_gm12878], axis=0)
valid_feats = np.concatenate([feats_for_cell(feats_all, labels_all, cell,n_feats=np.arange(2))[hepg2_val] for cell in train_cells_hepg2] + 
                             [feats_for_cell(feats_all, labels_all, cell,n_feats=np.arange(2))[gm_val] for cell in train_cells_gm12878], axis=0)

train_labels = np.concatenate([np.asarray(labels_all[cell]['bound'])[hepg2_train] for cell in train_cells_hepg2] + 
                              [np.asarray(labels_all[cell]['bound'])[gm_train] for cell in train_cells_gm12878], axis=0)
valid_labels = np.concatenate([np.asarray(labels_all[cell]['bound'])[hepg2_val] for cell in train_cells_hepg2] + 
                              [np.asarray(labels_all[cell]['bound'])[gm_val] for cell in train_cells_gm12878], axis=0)

import pickle
pickle.dump([train_feats, valid_feats, train_labels, valid_labels], open("finetune_TFBS.pkl", "wb"))

train_feats = np.concatenate([feats_for_cell(feats_all, labels_all, cell,n_feats=np.arange(2))[gm_train] for cell in train_cells_gm12878], axis=0)
valid_feats = np.concatenate([feats_for_cell(feats_all, labels_all, cell,n_feats=np.arange(2))[gm_val] for cell in train_cells_gm12878], axis=0)
pickle.dump([train_feats, valid_feats, train_labels, valid_labels], open("finetune_TFBS_GM.pkl", "wb"))

train_labels = np.concatenate(
[np.asarray(labels_all[cell]['bound'])[gm_train] for cell in train_cells_gm12878], axis=0)
valid_labels = np.concatenate([np.asarray(labels_all[cell]['bound'])[gm_val] for cell in train_cells_gm12878], axis=0)

import argparse
import pickle

import numpy as np
import torch.nn.functional as F
from sklearn.metrics import average_precision_score


# Validation step with AUPR reporting
def validate_model(model, val_loader, criterion, device="cpu"):
    model.eval()  # Set the model to evaluation mode
    val_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():  # No need to compute gradient when evaluating
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)[..., 0, :]
            val_loss += criterion(outputs, labels.float()).item()

            # # Store predictions and labels
            all_preds.extend(torch.sigmoid(outputs).cpu().numpy())  # Assuming binary classification
            all_labels.extend(labels.cpu().numpy())

    val_loss /= len(val_loader.dataset)
    val_loss *= 100
    val_aupr = average_precision_score(all_labels, all_preds)  # Compute AUPR
    return val_loss, val_aupr


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from ema_pytorch import EMA

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=False, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = nn.functional.cross_entropy(inputs, targets, reduction="none")
        else:
            BCE_loss = nn.functional.binary_cross_entropy_with_logits(
                inputs, targets, reduction="none"
            )
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss


# 1. Define the MLP Model
class MLP(nn.Module):
    def __init__(self, input_size, input_channel, mean, scale):
        super(MLP, self).__init__()
        self.mean = nn.Parameter(torch.from_numpy(mean).float())
        self.scale = nn.Parameter(torch.from_numpy(scale).float())
        self.fc1 = nn.Linear((input_size) * input_channel, 256)  # First hidden layer
        self.activation1 = nn.GELU()
        self.dropout1 = nn.Dropout(0.25)
        self.fc2 = nn.Linear(256, 128)  # Second hidden layer
        self.activation2 = nn.GELU()
        self.dropout2 = nn.Dropout(0.25)
        self.fc3 = nn.Linear(128, 64)  # Output layer, assuming 10 classes
        self.activation3 = nn.GELU()
        self.dropout3 = nn.Dropout(0.25)
        self.fc4 = nn.Linear(64, 1)  # Output layer, assuming 10 classes

        # self.fc5 = nn.Linear(3, 1)

    def forward(self, x):
        x = (x - self.mean[-x.shape[-1] :]) / self.scale[-x.shape[-1] :]
        xx = self.dropout1(self.activation1(self.fc1(x[..., :].reshape(x.shape[0], -1))))
        xx = self.dropout2(self.activation2(self.fc2(xx)))
        xx = self.dropout3(self.activation3(self.fc3(xx)))
        xx = self.fc4(xx)  # No activation, will use nn.CrossEntropyLoss
        # xx = xx + self.fc5(x[..., :3].mean(dim=1))
        return xx


class TFConv(nn.Module):
    def __init__(self, input_size, input_channel, mean, scale):
        super(TFConv, self).__init__()
        self.mean = nn.Parameter(torch.from_numpy(mean).float())
        self.scale = nn.Parameter(torch.from_numpy(scale).float())
        self.mean.requires_grad = False
        self.scale.requires_grad = False
        self.input_size = input_size
        self.input_channel = input_channel

        self.conv1 = nn.Conv1d(input_channel, 256, input_size)
        self.activation1 = nn.GELU()
        self.dropout1 = nn.Dropout(0.25)
        self.conv2 = nn.Conv1d(256, 128, 1)
        self.activation2 = nn.GELU()
        self.dropout2 = nn.Dropout(0.25)
        self.conv3 = nn.Conv1d(128, 64, 1)
        self.activation3 = nn.GELU()
        self.dropout3 = nn.Dropout(0.25)
        self.conv4 = nn.Conv1d(64, 1, 1)

    def forward(self, x):

        if x.shape[-1] != self.mean.shape[-1]:
            x = (x - self.mean[3:]) / self.scale[3:]
            x = torch.cat(
                [
                    torch.zeros([x.shape[0], x.shape[1], 3], device=x.device, dtype=x.dtype),
                    x,
                ],
                dim=2,
            )
        else:
            x = (x - self.mean) / self.scale
        # x = (x - self.mean) / self.scale
        xx = self.dropout1(self.activation1(self.conv1(x)))
        xx = self.dropout2(self.activation2(self.conv2(xx)))
        xx = self.dropout3(self.activation3(self.conv3(xx)))
        xx = self.conv4(xx)
        return xx


class TFConv_translate(nn.Module):
    def __init__(self, tfconv):
        super(TFConv_translate, self).__init__()
        # self = deepcopy(tfconv)
        print(
            tfconv.conv1.weight.shape,
            tfconv.conv1.bias.shape,
            tfconv.mean.shape,
            tfconv.scale.shape,
        )
        new_weight_0 = 1 / tfconv.scale[None] * tfconv.conv1.weight[:, 0, :]
        new_weight_1 = (
            tfconv.conv1.bias - (tfconv.mean / tfconv.scale) @ tfconv.conv1.weight[:, 0, :].T
        )
        print(new_weight_0.shape, new_weight_1.shape)

        self.conv1 = nn.Conv1d(tfconv.input_channel, 256, tfconv.input_size)
        self.activation1 = nn.GELU()
        self.dropout1 = nn.Dropout(0.25)
        self.conv2 = nn.Conv1d(256, 128, 1)
        self.activation2 = nn.GELU()
        self.dropout2 = nn.Dropout(0.25)
        self.conv3 = nn.Conv1d(128, 64, 1)
        self.activation3 = nn.GELU()
        self.dropout3 = nn.Dropout(0.25)
        self.conv4 = nn.Conv1d(64, 1, 1)

        self.conv1.weight.data = new_weight_0[:, None, :]
        self.conv1.bias.data = new_weight_1
        self.with_motif = True
        self.conv2.weight.data = tfconv.conv2.weight.data
        self.conv2.bias.data = tfconv.conv2.bias.data
        self.conv3.weight.data = tfconv.conv3.weight.data
        self.conv3.bias.data = tfconv.conv3.bias.data
        self.conv4.weight.data = tfconv.conv4.weight.data
        self.conv4.bias.data = tfconv.conv4.bias.data

    def with_motif(self, flag):
        self.with_motif = flag

    def forward(self, x):
        # if not self.with_motif:
        #     x = F.conv1d(x, self.conv1.weight[:, :, 3:], self.conv1.bias)
        # else:
        x = F.conv1d(x, self.conv1.weight, self.conv1.bias)

        xx = self.dropout1(self.activation1(x))
        xx = self.dropout2(self.activation2(self.conv2(xx)))
        xx = self.dropout3(self.activation3(self.conv3(xx)))
        xx = self.conv4(xx)
        return xx

torch.set_num_threads(3)

from sklearn.preprocessing import *
from scipy.stats import pearsonr

for feats in [[0], [1]]:
    feats = np.array(feats)
    
    train_feats, valid_feats, train_labels, valid_labels = pickle.load(open("finetune_TFBS.pkl", "rb"))
    train_feats = train_feats[:, feats, 3:]
    train_shape = train_feats.shape
    train_feats = train_feats.reshape((len(train_feats), len(feats), -1))
    valid_feats = valid_feats[:, feats, 3:]
    valid_shape = valid_feats.shape
    valid_feats = valid_feats.reshape((len(valid_feats), len(feats), -1))
    
    scaler = StandardScaler().fit(train_feats[:, 0, :])
    mean, std = scaler.mean_, scaler.scale_
    
    X_train = torch.as_tensor(train_feats).float()  # [:, 0, :]
    y_train = torch.as_tensor(train_labels[:, None]).long()
    X_val = torch.as_tensor(valid_feats).float()  # [:, 0, :]
    y_val = torch.as_tensor(valid_labels[:, None]).long()
    
    print(X_train.shape, y_train.shape, X_val.shape, y_val.shape)
    
    class_counts = y_train.sum()
    class_counts = torch.as_tensor([(class_counts) / len(y_train)])
    class_weights = 1.0 / class_counts.float()
    # Create DataLoader instances
    
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    
    batch_size = 512  # You can adjust the batch size
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # Initialize the MLP
    model = TFConv(X_train.shape[-1], X_train.shape[-2], mean, std).cuda()
    ema = EMA(model, beta=0.9999,  # exponential moving average factor
              update_after_step=100,  # only after this number of .update() calls will it start updating
              update_every=1).cuda()
    # 3. Training Loop
    criterion = nn.BCEWithLogitsLoss(weight=class_weights.cuda())
    
    optimizer = optim.Adam(model.parameters(), lr=5e-4)  # Learning rate can be adjusted
    
    num_epochs = 1000  # Number of epochs can be adjusted
    val_freq = 1000
    best_val_loss = 0
    no_improv_thres = 10
    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        ct = 0
        for inputs, labels in train_loader:
            inputs = inputs.cuda()
            labels = labels.cuda()
            optimizer.zero_grad()  # Zero the parameter gradients
            outputs = model(inputs)[..., 0, :]  # Forward pass
            loss = criterion(outputs, labels.float())  # Compute the loss
            loss.backward()  # Backward pass
            optimizer.step()  # Optimize
            ema.update()
            ct += 1
            if ct >= val_freq:
                ct = 0
                break
    
        # 4. Validation Step
        m = ema.ema_model
        val_loss, val_aupr = validate_model(m, val_loader, criterion, "cuda")
        print(f"Epoch: {epoch + 1}, Validation Loss: {val_loss:.4f}, Validation AUPR: {val_aupr:.4f}")
        if val_aupr > best_val_loss:
            best_val_loss = val_aupr
            no_improv = 0
        else:
            no_improv += 1
    
        if no_improv >= no_improv_thres:
            break
        model.train()
        ema.train()
    m2 = TFConv_translate(m)
    print(m2)
    m = m.eval()
    m2 = m2.eval()
    a, b = m(inputs), m2(inputs)
    print(a.shape, b.shape, a.reshape((-1)), b.reshape((-1)), torch.allclose(a, b, atol=1e-3, rtol=1e-3))
    print(pearsonr(a.reshape((-1)).cpu().detach().numpy(), b.reshape((-1)).cpu().detach().numpy()))
    m2 = torch.jit.script(m2)
    # Save to file
    torch.jit.save(m2, f"TFBS_{feats[0]}_conv_v2.pt")



















                                                      --data_dir . --project 20160114
