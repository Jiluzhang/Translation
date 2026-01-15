################################################################## train seq2print model #######################################################################################
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
                                                      --data_dir . --project 20160114
####################################################################################################################################################################################

################################################################################ train TFBS model ################################################################################
## workdir: /fs/home/jiluzhang/TF_grammar/scPrinter/20260114/tfbs
# cp /fs/home/jiluzhang/TF_grammar/scPrinter/test/luz/Unibind/data/TFBSPrediction/HepG2_pred_data.tsv .
# cp /fs/home/jiluzhang/TF_grammar/scPrinter/test/luz/Unibind/data/HepG2/test_peaks.bed .
# cp /fs/home/jiluzhang/TF_grammar/scPrinter/20260114/seq2print/model/Bcell_0_fold0-hmxm767i.pt_deepshap_sample30000/*bigwig .
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
from pyfaidx import Fasta
from scipy.stats import *
from sklearn.model_selection import train_test_split
import pickle
import argparse
import torch.nn.functional as F
from sklearn.metrics import average_precision_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from ema_pytorch import EMA
from sklearn.preprocessing import *
from scipy.stats import pearsonr

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

def cosine(x,y):
    return (x*y).sum() / (np.linalg.norm(x) * np.linalg.norm(y))

def get_feats(model, feat, tsv, flank, low=None, median=None, high=None, 
              chroms=None, verbose=False):
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
            
            seq = fasta[c][start-1:end].seq
            seq = scp.utils.DNA_one_hot(seq).numpy()
            seq_trend = f.values(c, start-1, end, numpy=True)
            if sd == '-':
                seq = seq[::-1][:, ::-1]
                seq_trend = seq_trend[::-1]
            motif_trend = (motif2matrix[tf] * seq).sum(axis=0)
            motif_match = cosine(motif2matrix[tf].reshape((-1)), seq.reshape((-1)))
            
            signals.append(v)
            similarity.append([cosine(seq_trend, motif_trend), # motif_match,
                               ((seq_trend - median) / (high - low)).mean()])
    
    signals = np.array(signals)
    similarity = np.array(similarity)
    signals = np.nan_to_num(signals)
    similarity =  np.nan_to_num(similarity)
    
    signals = (signals - median) / (high - low)
    return signals, similarity, tsv

def feats_for_cell(feats_all, labels_all, cell, window=201, n_feats=np.arange(8)):
    seq_attr = feats_all[cell][0]
    pad = (seq_attr.shape[-1] - window) // 2
    if pad > 0:
        seq_attr = seq_attr[..., pad:-pad]
    seq_motif_sim = feats_all[cell][1][:, n_feats]
    motifs = np.asarray(labels_all[cell]['motifMatchScore'])[:, None]
    motifs = np.stack([motifs] * len(n_feats), axis=1)
    feats = np.concatenate([motifs, seq_motif_sim, seq_attr], axis=-1)
    print (motifs.shape, seq_motif_sim.shape, seq_attr.shape)
    return feats

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
        x = F.conv1d(x, self.conv1.weight, self.conv1.bias)
        xx = self.dropout1(self.activation1(x))
        xx = self.dropout2(self.activation2(self.conv2(xx)))
        xx = self.dropout3(self.activation3(self.conv3(xx)))
        xx = self.conv4(xx)
        return xx

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

feats = ['attr.count.shap_hypo_0_.0.85.bigwig', 'attr.just_sum.shap_hypo_0-30_.0.85.bigwig']  # bigwig generated after seq2print model training finishied

pool= ProcessPoolExecutor(max_workers=2)
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
            lo, mid, hi = norm_factor[cell][feat][i]
            print(cell, feat, i, "%.2f" % mid, "%.2f"%(hi-lo))

feats_all = {}
labels_all = {}
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

hepg2_train, hepg2_val = train_test_split(np.arange(len(labels_all['HepG2'])), test_size=0.1, random_state=42)
gm_train, gm_val = train_test_split(np.arange(len(labels_all['HepG2'])), test_size=0.1, random_state=0)

train_cells_hepg2 = ['HepG2']
train_cells_gm12878 = ['HepG2']
train_feats = np.concatenate([feats_for_cell(feats_all, labels_all, cell, n_feats=np.arange(2))[hepg2_train] for cell in train_cells_hepg2] + 
                             [feats_for_cell(feats_all, labels_all, cell, n_feats=np.arange(2))[gm_train] for cell in train_cells_gm12878], axis=0)
valid_feats = np.concatenate([feats_for_cell(feats_all, labels_all, cell, n_feats=np.arange(2))[hepg2_val] for cell in train_cells_hepg2] + 
                             [feats_for_cell(feats_all, labels_all, cell, n_feats=np.arange(2))[gm_val] for cell in train_cells_gm12878], axis=0)
train_labels = np.concatenate([np.asarray(labels_all[cell]['bound'])[hepg2_train] for cell in train_cells_hepg2] + 
                              [np.asarray(labels_all[cell]['bound'])[gm_train] for cell in train_cells_gm12878], axis=0)
valid_labels = np.concatenate([np.asarray(labels_all[cell]['bound'])[hepg2_val] for cell in train_cells_hepg2] + 
                              [np.asarray(labels_all[cell]['bound'])[gm_val] for cell in train_cells_gm12878], axis=0)
pickle.dump([train_feats, valid_feats, train_labels, valid_labels], open("finetune_TFBS.pkl", "wb"))

torch.set_num_threads(3)

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
    
    # Training Loop
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
    
        # Validation Step
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
##########################################################################################################################################################################



################################################################################ TF training data ################################################################################
## Rscript peak_bed2rds.R
## workdir: /fs/home/jiluzhang/TF_grammar/scPrinter/20260114/tfdata/data/HepG2
## Download GSM6672041_HepG2_peaks.bed from GSE216464 (SHARE-Seq of HepG2 cells) (extended to 1kb)
## generate "regionsRanges.rds"
## shuf GSM6672041_HepG2_peaks.bed | head -n 1000 > test_peaks.bed
suppressPackageStartupMessages(library(GenomicRanges))
regionBed <- read.table("test_peaks.bed")
regions <- GRanges(paste0(regionBed$V1, ":", regionBed$V2, "-", regionBed$V3))
regions <- resize(regions, 1000, fix="center")
saveRDS(regions, "regionRanges.rds")


## Rscript disp_rds.R
## h5 to rds for dispersion model
## workdir: /fs/home/jiluzhang/TF_grammar/scPrinter/20260114/tfdata/dispersionModel
# cp /fs/home/jiluzhang/TF_grammar/scPrinter/test/luz/data/BAC/dispModelData/dispModelData2bp.txt .
# cp /fs/home/jiluzhang/TF_grammar/scPrinter/test/luz/data/shared/dispModel/dispersionModel2bp.h5 .
library(reticulate)
use_condaenv('PRINT')
Sys.setenv(CUDA_VISIBLE_DEVICES='7')
keras <- import("keras")
for(footprintRadius in 2:2){
  dispModelData <- read.table(paste0("dispModelData", footprintRadius, "bp.txt"), sep="\t")
  dispersionModel <- keras$models$load_model(paste0("dispersionModel", footprintRadius, "bp.h5"))
  modelWeights <- dispersionModel$get_weights()
  modelFeatures <- dispModelData[, c("leftFlankBias", "rightFlankBias", "centerBias", "leftTotalInsertion", "rightTotalInsertion")]
  modelTargets <- dispModelData[, c("leftRatioMean", "leftRatioSD", "rightRatioMean", "rightRatioSD")]
  dispersionModel <- list(modelWeights=modelWeights,
                          featureMean=colMeans(modelFeatures), featureSD=apply(modelFeatures, 2, sd),
                          targetMean=colMeans(modelTargets), targetSD=apply(modelTargets, 2, sd))
  saveRDS(dispersionModel, paste0("dispersionModel", footprintRadius, "bp.rds"))
}


#### https://github.com/buenrostrolab/PRINT/blob/main/analyses/TFBSPrediction/
#### TFBSTrainingData.R (for generate TFBSDataUnibind.h5 file)
## conda activate PRINT
## workdir: /fs/home/jiluzhang/TF_grammar/scPrinter/20260114/tfdata
# cp /fs/home/jiluzhang/TF_grammar/PRINT/code/predictBias.py ./code
# zcat /fs/home/jiluzhang/TF_grammar/scPrinter/test/luz/data/BAC/rawData/test.fragments.tsv.gz | head -n 10000 > test.fragments.tsv && gzip test.fragments.tsv
# cp /fs/home/jiluzhang/TF_grammar/scPrinter/test/luz/Unibind/data/shared/Tn5_NN_model.h5 ./data/shared

source('/fs/home/jiluzhang/TF_grammar/PRINT/code/utils.R')
source("/fs/home/jiluzhang/TF_grammar/PRINT/code/getFootprints.R")
source("/fs/home/jiluzhang/TF_grammar/PRINT/code/getCounts.R")
source("/fs/home/jiluzhang/TF_grammar/PRINT/code/getBias.R")
source("/fs/home/jiluzhang/TF_grammar/PRINT/code/getAggregateFootprint.R")
source("/fs/home/jiluzhang/TF_grammar/PRINT/code/getTFBS.R")
use_condaenv('PRINT')

projectName <- "HepG2"    # One of "K562", "GM12878", "HepG2"
ChIPDataset <- "Unibind"  # One of "ENCODE", "Unibind"
project <- footprintingProject(projectName=projectName, refGenome="hg38")
projectMainDir <- "./"
projectDataDir <- paste0(projectMainDir, "data/", projectName, "/")
dataDir(project) <- projectDataDir
mainDir(project) <- projectMainDir

tmpDir <- dataDir(project)
chunkSize <- regionChunkSize(project)

regionRanges(project) <- readRDS(paste0(projectDataDir, "regionRanges.rds"))

dispModel(project, 2) <- readRDS(paste0(projectMainDir, "dispersionModel/dispersionModel", '2', "bp.rds"))
dispersionModel <- project@dispModel[[2]]

barcodeGroups <- data.frame(barcode=paste("rep", 1:5, sep=""), group=1:5)
groups(project) <- as.character(mixedsort(unique(barcodeGroups$group)))
groupCellType(project) <- c('1', '2')  # maybe 'HepG2'
cellTypeLabels <- groupCellType(project)

pathToFrags <- paste0(projectDataDir, "test.fragments.tsv.gz")
projectCountTensor <- countTensor(getCountTensor(project, pathToFrags, barcodeGroups, returnCombined=T,
                                                 chunkSize=chunkSize, nCores=32))  # region  position  group  count

project <- getRegionBias(project, nCores=16)  # Tn5_NN_model.h5 in shared dir
saveRDS(regionBias(project), paste0(projectDataDir, "predBias.rds"))

################################################################################
################################################################################
################################ HERE ##########################################
################################################################################
################################################################################
################################################################################


footprintResults <- get_footprints(projectCountTensor=projectCountTensor, dispersionModel=dispersionModel,
                                   tmpDir=tmpDir, mode='2', footprintRadius=2, flankRadius=2,
                                   cellTypeLabels=cellTypeLabels, chunkSize=chunkSize,
                                   returnCellTypeScores=FALSE, nCores=8)

# Load footprints
footprintRadii <- c(2) #footprintRadii <- c(10, 20, 30, 50, 80, 100)
multiScaleFootprints <- list()
for(footprintRadius in footprintRadii){
  
  print(paste0("Loading data for footprint radius = ", footprintRadius))  
  chunkDir <- paste0("./data/", projectName, "/chunkedFootprintResults/", footprintRadius, "/")
  chunkFiles <- gtools::mixedsort(list.files(chunkDir))
  scaleFootprints <- pbmcapply::pbmclapply(
    chunkFiles,
    function(chunkFile){
      chunkData <- readRDS(paste0(chunkDir, chunkFile))
      as.data.frame(t(sapply(
        chunkData,
        function(regionData){
          regionData$aggregateScores
        }
      )))
    },
    mc.cores = 16
  )
  scaleFootprints <- data.table::rbindlist(scaleFootprints)
  multiScaleFootprints[[as.character(footprintRadius)]] <- as.matrix(scaleFootprints)
}

# Load PWM data
cisBPMotifs <- readRDS(paste0(projectMainDir, "/data/shared/cisBP_human_pwms_2021.rds"))  # wget -c https://zenodo.org/records/15224770/files/cisBP_human_pwms_2021.rds?download=1

#### generate TF ChIP ranges ####
# see 'getUnibindData.R'
# If running in Rstudio, set the working directory to current path

library(GenomicRanges)

dataset <- "HepG2"

# Get dataset metadata
ChIPDir <- "/fs/home/jiluzhang/TF_grammar/scPrinter/test/luz/Unibind/damo_hg38_all_TFBS/"
ChIPSubDirs <- list.files(ChIPDir)
ChIPMeta <- as.data.frame(t(sapply(
  ChIPSubDirs, 
  function(ChIPSubDir){
    strsplit(ChIPSubDir, "\\.")[[1]]
  }
)))
rownames(ChIPMeta) <- NULL
colnames(ChIPMeta) <- c("ID", "dataset", "TF")
ChIPMeta$Dir <- ChIPSubDirs

# Select cell types
# datasetName <- list("K562" = "K562_myelogenous_leukemia",
#                     "HepG2" = "HepG2_hepatoblastoma",
#                     "GM12878" = "GM12878_female_B-cells_lymphoblastoid_cell_line",
#                     "A549" = "A549_lung_carcinoma")
datasetName <- list("HepG2"="HepG2_hepatoblastoma")
ChIPMeta <- ChIPMeta[ChIPMeta$dataset == datasetName[[dataset]], ]
ChIPTFs <- ChIPMeta$TF

# If we have multiple files for the same TF, decide whether to keep the intersection or union of sites
mode <- "intersect"

# Extract TFBS ChIP ranges
unibindTFBS <- pbmcapply::pbmclapply(
  sort(unique(ChIPTFs)),
  function(TF){
    
    # Retrieve the list of ChIP Ranges
    ChIPRangeList <- lapply(
      which(ChIPTFs %in% TF),
      function(entry){
        ChIPSubDir <- ChIPMeta$Dir[entry]
        ChIPFiles <- list.files(paste0(ChIPDir, ChIPSubDir))
        ChIPRanges <- lapply(
          ChIPFiles,
          function(ChIPFile){
            ChIPBed <- read.table(paste0(ChIPDir, ChIPSubDir, "/", ChIPFile))
            ChIPRange <- GRanges(seqnames = ChIPBed$V1, ranges = IRanges(start = ChIPBed$V2, end = ChIPBed$V3))
            ChIPRange$score <- ChIPBed$V5
            ChIPRange
          }
        )
        if(mode == "intersect"){
          ChIPRanges <- Reduce(subsetByOverlaps, ChIPRanges)
        }else if (mode == "union"){
          ChIPRanges <- mergeRegions(Reduce(c, ChIPRanges))
        }
        
        ChIPRanges
      }
    )
    
    # For TFs with more than one ChIP files, take the intersection or union of regions in all files
    if(mode == "intersect"){
      ChIPRanges <- Reduce(subsetByOverlaps, ChIPRangeList)
    }else if (mode == "union"){
      ChIPRanges <- mergeRegions(Reduce(c, ChIPRangeList))
    }
    ChIPRanges
  },
  mc.cores = 3
)
names(unibindTFBS) <- sort(unique(ChIPTFs))

# Save results to file
# if(mode == "intersect"){
#   saveRDS(unibindTFBS, paste0("../../../data/shared/unibind/", dataset, "ChIPRanges.rds"))
# }else if (mode == "union"){
#   saveRDS(unibindTFBS, paste0("../../../data/shared/unibind/", dataset, "UnionChIPRanges.rds"))
# }
saveRDS(unibindTFBS, paste0("./", dataset, "ChIPRanges.rds"))
##########


# Load TF ChIP ranges
TFChIPRanges <- readRDS(paste0("./", projectName, "ChIPRanges.rds"))

# Only keep TFs with both motif and ChIP data
keptTFs <- intersect(names(TFChIPRanges), names(cisBPMotifs))
cisBPMotifs <- cisBPMotifs[keptTFs]
TFChIPRanges <- TFChIPRanges[keptTFs]

# Find motif matches across all regions
path <- paste0(dataDir(project), "motifPositionsList.rds")
nCores <- 3
combineTFs <- FALSE
if(file.exists(path)){
  motifMatches <- readRDS(path)
}else{
  regions <- regionRanges(project)
  # Find motif matches across all regions
  print("Getting TF motif matches within CREs")
  motifMatches <- pbmcapply::pbmclapply(
    names(cisBPMotifs),
    function(TF){
      TFMotifPositions <- motifmatchr::matchMotifs(pwms=cisBPMotifs[[TF]], subject=regions, 
                                                   genome=refGenome(project), out="positions")[[1]]
      if(length(TFMotifPositions) > 0){
        TFMotifPositions$TF <- TF
        TFMotifPositions$score <- rank(TFMotifPositions$score) / length(TFMotifPositions$score)
        TFMotifPositions <- mergeRegions(TFMotifPositions)
        TFMotifPositions
      }
    },
    mc.cores = nCores
  )
  names(motifMatches) <- names(cisBPMotifs)
  if(combineTFs){motifMatches <- Reduce(c, motifMatches)}
  saveRDS(motifMatches, path)
}

# Get ATAC tracks for each region
project <- getATACTracks(project)
regionATAC <- ATACTracks(project)

##########################################
# Get multi-scale footprints around TFBS #
##########################################
TFBSData <- getTFBSTrainingData(multiScaleFootprints, motifMatches, 
                                TFChIPRanges, regions, percentBoundThreshold=0.1)

#####################
# Save data to file #
#####################
# Write TFBS training data to a file
h5_path <- paste0("./data/", projectName, "/TFBSData", ChIPDataset, ".h5")
if(file.exists(h5_path)){
  system(paste0("rm ./data/", projectName, "/TFBSData", ChIPDataset, ".h5"))
}
h5file <- hdf5r::H5File$new(h5_path, mode="w")
h5file[["motifFootprints"]] <- TFBSData[["motifFootprints"]]
h5file[["metadata"]] <- TFBSData[["metadata"]]
h5file$close_all()

#### generate pred_data.tsv (from footprint_to_TF.ipynb)
import h5py
import numpy as np
import pandas as pd

external_dataset = "HepG2"
external_hf = h5py.File("./data/" + external_dataset + "/TFBSDataUnibind.h5", 'r')

external_metadata = external_hf['metadata']
external_metadata = np.array(external_metadata)

external_TF_bound = np.array([i[0] for i in external_metadata])
external_motif_scores = np.array([i[1] for i in external_metadata])
external_TF_labels = np.array([i[2].decode('ascii') for i in external_metadata])
external_kept_TFs = np.unique(external_TF_labels)

external_metadata = pd.DataFrame(external_metadata)
external_metadata["range"] = [i.decode('ascii') for i in external_metadata["range"]]
external_metadata["TF"] = [i.decode('ascii') for i in external_metadata["TF"]]
# external_metadata["predScore"] = external_pred
external_metadata.to_csv("./data/TFBSPrediction/"+external_dataset+"_pred_data.tsv", sep="\t")  # mkdir ./data/TFBSPrediction
########################################################################################################################################################################
