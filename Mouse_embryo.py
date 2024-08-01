## Decoding gene regulation in the mouse embryo using single-cell multi-omics
## https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE205117
## https://github.com/rargelaguet/mouse_organogenesis_10x_multiome_publication
## workdir: /fs/home/jiluzhang/scM2M_no_dec_attn/mouse_embryo
# nohup wget -c https://ftp.ncbi.nlm.nih.gov/geo/series/GSE205nnn/GSE205117/suppl/GSE205117_RAW.tar > 20240715_download.log &

## CRISPR_T_WT vs. CRISPR_T_KO (E8.5)
## genes are same for wt & ko
## merge wt & ko peaks
zcat GSM6205422_E8.5_CRISPR_T_WT_GEX/features.tsv.gz | grep "Peaks" | cut -f 4-6 > wt_peaks.bed  # 241050
zcat GSM6205421_E8.5_CRISPR_T_KO_GEX/features.tsv.gz | grep "Peaks" | cut -f 4-6 > ko_peaks.bed  # 233258
cat wt_peaks.bed ko_peaks.bed | sort -k1,1 -k2,2n > wt_ko_peaks_sorted.bed
bedtools merge -i wt_ko_peaks_sorted.bed > wt_ko_peaks_raw.bed  # 276986
grep -v -E "GL|JH|chrX|chrY" wt_ko_peaks_raw.bed | sort -k 1.4,1 -n -s -k2,2n > wt_ko_peaks.bed  # 271529
rm wt_ko_peaks_sorted.bed wt_ko_peaks_raw.bed

## process rna & atac
import scanpy as sc
import pandas as pd
import pybedtools
import numpy as np
from tqdm import tqdm
from scipy.sparse import csr_matrix

## wt
wt = sc.read_10x_mtx('GSM6205422_E8.5_CRISPR_T_WT_GEX', gex_only=False)  # 7604 × 273335

## rna
rna_info = pd.read_table('GSM6205422_E8.5_CRISPR_T_WT_GEX/features.tsv.gz', header=None)
rna_info = rna_info[rna_info[2]=="Gene Expression"]
rna_info = rna_info[rna_info[3].isin(['chr'+str(i) for i in range(1, 20)])]

wt_rna = wt[:, wt.var['feature_types']=='Gene Expression'].copy()        # 7604 × 32285
wt_rna_out = wt_rna[:, wt_rna.var['gene_ids'].isin(rna_info[0])].copy()  # 7604 × 30364
wt_rna_out.write('rna_wt.h5ad')

## atac
wt_atac = wt[:, wt.var['feature_types']=='Peaks'].copy()  # 7604 × 241050
wt_atac.X[wt_atac.X>0] = 1

peaks = pd.DataFrame({'id': wt_atac.var_names})
peaks['chr'] = peaks['id'].map(lambda x: x.split(':')[0])
peaks['start'] = peaks['id'].map(lambda x: x.split(':')[1].split('-')[0])
peaks['end'] = peaks['id'].map(lambda x: x.split(':')[1].split('-')[1])
peaks.drop(columns='id', inplace=True)
peaks['idx'] = range(peaks.shape[0])

cCREs = pd.read_table('wt_ko_peaks.bed', names=['chr', 'start', 'end'])
cCREs['idx'] = range(cCREs.shape[0])
cCREs_bed = pybedtools.BedTool.from_dataframe(cCREs)
peaks_bed = pybedtools.BedTool.from_dataframe(peaks)
idx_map = peaks_bed.intersect(cCREs_bed, wa=True, wb=True).to_dataframe().iloc[:, [3, 7]]
idx_map.columns = ['peaks_idx', 'cCREs_idx']

m = np.zeros([wt_atac.n_obs, cCREs_bed.to_dataframe().shape[0]], dtype='float32')
for i in tqdm(range(wt_atac.X.shape[0]), ncols=80, desc='Aligning ATAC peaks'):
    m[i][idx_map[idx_map['peaks_idx'].isin(peaks.iloc[np.nonzero(wt_atac.X[i])[1]]['idx'])]['cCREs_idx']] = 1

wt_atac_new = sc.AnnData(m, obs=wt_atac.obs, var=pd.DataFrame({'gene_ids': cCREs['chr']+':'+cCREs['start'].map(str)+'-'+cCREs['end'].map(str), 'feature_types': 'Peaks'}))
wt_atac_new.var.index = wt_atac_new.var['gene_ids'].values
wt_atac_new.X = csr_matrix(wt_atac_new.X)
wt_atac_new.write('atac_wt.h5ad')


## ko
ko = sc.read_10x_mtx('GSM6205421_E8.5_CRISPR_T_KO_GEX', gex_only=False)  # 7359 × 265543

## rna
rna_info = pd.read_table('GSM6205421_E8.5_CRISPR_T_KO_GEX/features.tsv.gz', header=None)
rna_info = rna_info[rna_info[2]=="Gene Expression"]
rna_info = rna_info[rna_info[3].isin(['chr'+str(i) for i in range(1, 20)])]

ko_rna = ko[:, ko.var['feature_types']=='Gene Expression'].copy()        # 7359 × 32285
ko_rna_out = ko_rna[:, ko_rna.var['gene_ids'].isin(rna_info[0])].copy()  # 7359 × 30364
ko_rna_out.write('rna_ko.h5ad')

## atac
ko_atac = ko[:, ko.var['feature_types']=='Peaks'].copy()  # 7359 × 233258
ko_atac.X[ko_atac.X>0] = 1

peaks = pd.DataFrame({'id': ko_atac.var_names})
peaks['chr'] = peaks['id'].map(lambda x: x.split(':')[0])
peaks['start'] = peaks['id'].map(lambda x: x.split(':')[1].split('-')[0])
peaks['end'] = peaks['id'].map(lambda x: x.split(':')[1].split('-')[1])
peaks.drop(columns='id', inplace=True)
peaks['idx'] = range(peaks.shape[0])

cCREs = pd.read_table('wt_ko_peaks.bed', names=['chr', 'start', 'end'])
cCREs['idx'] = range(cCREs.shape[0])
cCREs_bed = pybedtools.BedTool.from_dataframe(cCREs)
peaks_bed = pybedtools.BedTool.from_dataframe(peaks)
idx_map = peaks_bed.intersect(cCREs_bed, wa=True, wb=True).to_dataframe().iloc[:, [3, 7]]
idx_map.columns = ['peaks_idx', 'cCREs_idx']

m = np.zeros([ko_atac.n_obs, cCREs_bed.to_dataframe().shape[0]], dtype='float32')
for i in tqdm(range(ko_atac.X.shape[0]), ncols=80, desc='Aligning ATAC peaks'):
    m[i][idx_map[idx_map['peaks_idx'].isin(peaks.iloc[np.nonzero(ko_atac.X[i])[1]]['idx'])]['cCREs_idx']] = 1

ko_atac_new = sc.AnnData(m, obs=ko_atac.obs, var=pd.DataFrame({'gene_ids': cCREs['chr']+':'+cCREs['start'].map(str)+'-'+cCREs['end'].map(str), 'feature_types': 'Peaks'}))
ko_atac_new.var.index = ko_atac_new.var['gene_ids'].values
ko_atac_new.X = csr_matrix(ko_atac_new.X)
ko_atac_new.write('atac_ko.h5ad')  # 7359 × 271529


## add cell annotation
import scanpy as sc
import pandas as pd

meta_info = pd.read_table('sample_metadata.txt.gz')

wt_info = meta_info[(meta_info['sample']=='E8.5_CRISPR_T_WT')][['barcode', 'celltype']]  # 7891
wt_info.index = wt_info['barcode'].values
del wt_info['barcode']

wt_rna = sc.read_h5ad('rna_wt.h5ad')  # 7604
wt_rna.obs['cell_anno'] = wt_info.loc[wt_rna.obs.index.values]['celltype'].values
wt_rna_out =  wt_rna[wt_rna.obs['cell_anno'].isin(['NMP', 'Spinal_cord', 'Somitic_mesoderm'])].copy()
# Spinal_cord         526
# Somitic_mesoderm    362
# NMP                 326
wt_rna_out.write('rna_wt_3_types.h5ad')  # 1214 × 30364

wt_atac = sc.read_h5ad('atac_wt.h5ad')  # 7604
wt_atac_out = wt_atac[wt_rna_out.obs.index, :].copy()
wt_atac_out.obs['cell_anno'] = wt_rna_out.obs['cell_anno']
wt_atac_out.write('atac_wt_3_types.h5ad')  # 1214 × 271529


ko_info = meta_info[(meta_info['sample']=='E8.5_CRISPR_T_KO')][['barcode', 'celltype']]  # 7741
ko_info.index = ko_info['barcode'].values
del ko_info['barcode']

ko_rna = sc.read_h5ad('rna_ko.h5ad')  # 7359
ko_rna.obs['cell_anno'] = ko_info.loc[ko_rna.obs.index.values]['celltype'].values
ko_rna_out =  ko_rna[ko_rna.obs['cell_anno'].isin(['NMP', 'Spinal_cord', 'Somitic_mesoderm'])].copy()
# NMP                 745
# Spinal_cord         436
# Somitic_mesoderm      5
ko_rna_out.write('rna_ko_3_types.h5ad')  # 1186 × 30364

ko_atac = sc.read_h5ad('atac_ko.h5ad')  # 7359
ko_atac_out = ko_atac[ko_rna_out.obs.index, :].copy()
ko_atac_out.obs['cell_anno'] = ko_rna_out.obs['cell_anno']
ko_atac_out.write('atac_ko_3_types.h5ad')  # 1186 × 271529



## preprocess & train model
python split_train_val.py --RNA rna_wt_3_types.h5ad --ATAC atac_wt_3_types.h5ad --train_pct 0.9
python data_preprocess.py -r rna_wt_3_types_train.h5ad -a atac_wt_3_types_train.h5ad -s preprocessed_data_train --dt train --config rna2atac_config_train.yaml   # list_digits: 6 -> 7
python data_preprocess.py -r rna_wt_3_types_val.h5ad -a atac_wt_3_types_val.h5ad -s preprocessed_data_val --dt val --config rna2atac_config_val.yaml
nohup accelerate launch --config_file accelerator_config_train.yaml --main_process_port 29823 rna2atac_train.py --config_file rna2atac_config_train.yaml \
                        --train_data_dir ./preprocessed_data_train --val_data_dir ./preprocessed_data_val -n rna2atac_train > 20240726.log &  

python data_preprocess.py -r rna_wt_3_types.h5ad -a atac_wt_3_types.h5ad -s preprocessed_data_test --dt test --config rna2atac_config_test.yaml
accelerate launch --config_file accelerator_config_test.yaml --main_process_port 29822 rna2atac_test.py \
                  -d ./preprocessed_data_test \
                  -l save_mlt_40/2024-07-26_rna2atac_train_300/pytorch_model.bin --config_file rna2atac_config_test.yaml
python npy2h5ad.py
python csr2array.py --pred rna2atac_scm2m_raw.h5ad --true atac_wt_3_types.h5ad
python cal_auroc_auprc.py --pred rna2atac_scm2m.h5ad --true atac_wt_3_types.h5ad
python cal_cluster_plot.py --pred rna2atac_scm2m.h5ad --true atac_wt_3_types.h5ad

## evaluation
## ground truth
import snapatac2 as snap
import scanpy as sc
import numpy as np

true = snap.read('atac_wt_3_types.h5ad', backed=None)
snap.pp.select_features(true)
snap.tl.spectral(true)
snap.tl.umap(true)
sc.pl.umap(true, color='cell_anno', legend_fontsize='7', legend_loc='right margin', size=10,
           title='', frameon=True, save='_atac_3_types_true.pdf')
true[:550, :].write('rna2atac_true.h5ad')

rna = snap.read('rna_wt_3_types.h5ad', backed=None)
np.count_nonzero(rna.X.toarray(), axis=1).max()  # 9516



# ## output nmp rna & atac
# import scanpy as sc

# wt_rna = sc.read_h5ad('rna_wt_3_types.h5ad')
# wt_rna_nmp = wt_rna[wt_rna.obs['cell_anno']=='NMP']  # 326
# wt_rna_nmp.write('rna_wt_nmp.h5ad')
# wt_atac = sc.read_h5ad('atac_wt_3_types.h5ad')
# wt_atac[wt_rna_nmp.obs.index].copy().write('atac_wt_nmp.h5ad')
# np.count_nonzero(wt_rna_nmp.X.toarray(), axis=1).max()  # 8533

# ## preprocess nmp
# python data_preprocess.py -r rna_wt_nmp.h5ad -a atac_wt_nmp.h5ad -s ./preprocessed_data_test_nmp --dt test --config rna2atac_config_test.yaml

# ## ko genes
# import scanpy as sc
# import torch
# import numpy as np

# wt_rna_nmp = sc.read_h5ad('rna_wt_nmp.h5ad')

# gene_idx = np.argwhere(np.count_nonzero(wt_rna_nmp.X.toarray(), axis=0)>(326/2)).flatten()  # 4203

# # rna_idx & rna_value & atac_idx & atac_val & rna_mask
# dat_0 = torch.load('preprocessed_data_test_nmp/preprocessed_data_0.pt')
# dat_0_ko = dat_0[]






mkdir gene_files/preprocessed_data_test_gene_KO
## in silico ko
# python out_ko_h5ad.py
import scanpy as sc
from multiprocessing import Pool

wt_rna = sc.read_h5ad('rna_wt_3_types.h5ad')
wt_rna_nmp = wt_rna[wt_rna.obs['cell_anno']=='NMP']  # 326
wt_atac = sc.read_h5ad('atac_wt_3_types.h5ad')
#sum(np.count_nonzero(wt_rna_nmp.X.toarray(), axis=0)>(326/2)) # 4203

def out_ko_rna_atac(i):
    gene = wt_rna.var.index[i]
    ko_rna = wt_rna_nmp[wt_rna_nmp.X[:, i].toarray().flatten()!=0].copy()
    if ko_rna.shape[0]>(326/2):
        ko_rna.X[:, i] = 0
        ko_rna.write('gene_files/rna_'+gene+'_ko_nmp.h5ad')
        wt_atac[ko_rna.obs.index].write('gene_files/atac_'+gene+'_ko_nmp.h5ad')

with Pool(40) as p:
    p.map(out_ko_rna_atac, list(range(wt_rna.n_vars)))


## get genes
import scanpy as sc
import pandas as pd
import numpy as np

wt_rna = sc.read_h5ad('rna_wt_3_types.h5ad')
wt_rna_nmp = wt_rna[wt_rna.obs['cell_anno']=='NMP'] 
df = pd.DataFrame(wt_rna_nmp.var.index[np.count_nonzero(wt_rna_nmp.X.toarray(), axis=0)>(326/2)])
df.to_csv('genes.txt', index=None, header=None)


## split genes list files
split -l 200 genes.txt -d -a 2 genes_p
ls | grep genes_p | xargs -i{} mv {} {}.txt  # genes_p00.txt -> genes_p21.txt

## preprocess genes ko rna & atac
# ./prc_ko.sh
for i in `printf "%02d\n" $(seq 00 21)`;do
{
    for gene in `cat genes_p$i.txt`;do
        python data_preprocess.py -r gene_files/h5ad/rna_$gene\_ko_nmp.h5ad -a gene_files/h5ad/atac_$gene\_ko_nmp.h5ad -s gene_files/preprocessed_data_test_ko/$gene --dt test --config rna2atac_config_test.yaml
    done
}&
done
wait


accelerate launch --config_file accelerator_config_test.yaml --main_process_port 29822 rna2atac_test.py \
                  -d gene_files/preprocessed_data_test_$gene\_KO \
                  -l save_mlt_40/2024-07-26_rna2atac_train_300/pytorch_model.bin --config_file rna2atac_config_test.yaml
mv predict.npy gene_files/predict_$gene.npy
echo $gene done



## KO genes directly before inputting to model
python data_preprocess.py -r rna_wt_nmp.h5ad -a atac_wt_nmp.h5ad -s ./preprocessed_data_test_nmp --dt test --config rna2atac_config_test.yaml  # 8534 (8533+1)

nohup accelerate launch --config_file accelerator_config_test_1.yaml --main_process_port 29822 rna2atac_test_ko_1.py \
                        -d preprocessed_data_test_nmp \
                        -l save_mlt_40/2024-07-26_rna2atac_train_300/pytorch_model.bin --config_file rna2atac_config_test.yaml > predict_ko_1.log &
# 2713258

nohup accelerate launch --config_file accelerator_config_test_2.yaml --main_process_port 29823 rna2atac_test_ko_2.py \
                        -d preprocessed_data_test_nmp \
                        -l save_mlt_40/2024-07-26_rna2atac_train_300/pytorch_model.bin --config_file rna2atac_config_test.yaml > predict_ko_2.log &
# 2713293


python data_preprocess.py -r rna_wt_nmp.h5ad -a atac_wt_nmp.h5ad -s ./preprocessed_data_test_nmp_9516 --dt test --config rna2atac_config_test.yaml  # 9516

nohup accelerate launch --config_file accelerator_config_test_1.yaml --main_process_port 29822 rna2atac_test_ko_1.py \
                        -d preprocessed_data_test_nmp_9516 \
                        -l save_mlt_40/2024-07-26_rna2atac_train_300/pytorch_model.bin --config_file rna2atac_config_test.yaml > predict_ko_1.log &
# 3875769

nohup accelerate launch --config_file accelerator_config_test_2.yaml --main_process_port 29823 rna2atac_test_ko_2.py \
                        -d preprocessed_data_test_nmp_9516 \
                        -l save_mlt_40/2024-07-26_rna2atac_train_300/pytorch_model.bin --config_file rna2atac_config_test.yaml > predict_ko_2.log &
# 3877373

import sys
import tqdm
import argparse
import os
import yaml
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F

sys.path.append("M2Mmodel")
from M2M import M2M_rna2atac
import datetime
import time
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import precision_score, roc_auc_score
from utils import *

torch.autograd.set_detect_anomaly=True
torch.multiprocessing.set_sharing_strategy('file_system')

import scanpy as sc

def main():
    parser = argparse.ArgumentParser(description='evaluating script of M2M') #Luz parser = argparse.ArgumentParser(description='pretrain script of M2M')
    
    parser.add_argument('-d', '--data_dir', type=str, default=None, help="dir of preprocessed paired data")
    parser.add_argument('-s', '--save', type=str, default="save", help="where tht model's state_dict should be saved at")
    parser.add_argument('-n', '--name', type=str, help="the name of this project")
    parser.add_argument('-c', '--config_file', type=str, default="rna2atac_config.yaml", help="config file for model parameters")
    parser.add_argument('-l', '--load', type=str, default=None, help="if set, load previous model, path to previous model state_dict")
    
    args = parser.parse_args()
    data_dir = args.data_dir
    save_path = args.save
    save_name = args.name
    saved_model_path = args.load
    config_file = args.config_file
    
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    
    batch_size = config["training"].get('batch_size')
    SEED = config["training"].get('SEED')
    lr = float(config["training"].get('lr'))
    gamma_step = config["training"].get('gamma_step')
    gamma = config["training"].get('gamma')
    num_workers = config["training"].get('num_workers')
    epoch = config["training"].get('epoch')
    log_every = config["training"].get('log_every')

    setup_seed(SEED)
    accelerator = New_Accelerator(step_scheduler_with_optimizer = False)
    
    if os.path.isdir(data_dir):
        files = os.listdir(data_dir)
    else:
        data_dir, files = os.path.split(data_dir)
        files = [files]
    
    dataloader_kwargs = {'batch_size': batch_size, 'shuffle': False}
    cuda_kwargs = {
        "num_workers": num_workers,
        'prefetch_factor': 2
    }  
    dataloader_kwargs.update(cuda_kwargs)
    
    with accelerator.main_process_first():
        
        model = M2M_rna2atac(
            dim = config["model"].get("dim"),
            enc_num_gene_tokens = config["model"].get("total_gene") + 1,
            enc_num_value_tokens = config["model"].get('max_express') + 1, # +special tokens
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
        
        if saved_model_path:
            model.load_state_dict(torch.load(saved_model_path), strict=False)
            accelerator.print("previous model loaded!")
        else:
            accelerator.print("warning: you haven't load previous model!")
        
    device = accelerator.device
    model = accelerator.prepare(model)
    model.eval()

    wt_rna = sc.read_h5ad('rna_wt_3_types.h5ad')
    wt_rna_nmp = wt_rna[wt_rna.obs['cell_anno']=='NMP']
    gene_idx_lst = np.argwhere(np.count_nonzero(wt_rna_nmp.X.toarray(), axis=0)>(326/2)).flatten()  # 4203
    
    for gene_idx in gene_idx_lst:
        gene_name = wt_rna_nmp.var.index[gene_idx]
        y_hat_out = torch.Tensor()
        with torch.no_grad():
            files_iter = tqdm.tqdm(files, desc='Predicting KO '+gene_name, ncols=80, total=len(files)) if accelerator.is_main_process else files
            for file in files_iter:
                test_data = torch.load(os.path.join(data_dir, file))
                idx = torch.where(test_data[0]==(gene_idx+1))  # notice carefully
                if len(idx[0])!=0:
                    test_data[0][idx] = 0  # rna idx ko
                    test_data[1][idx] = 0  # rna val ko
                    test_data[4][idx] = 0  # rna mask
                    
                    test_data[0] = test_data[0][idx[0]]
                    test_data[1] = test_data[1][idx[0]]
                    test_data[2] = test_data[2][idx[0]]
                    test_data[3] = test_data[3][idx[0]]
                    test_data[4] = test_data[4][idx[0]]
                    
                    test_dataset = PreDataset(test_data)
                    test_loader = torch.utils.data.DataLoader(test_dataset, **dataloader_kwargs)
                    test_loader = accelerator.prepare(test_loader)
                    data_iter = enumerate(test_loader)
                    
                    for _, (rna_sequence, rna_value, atac_sequence, tgt, enc_pad_mask) in data_iter:
                        # rna_value: b, n
                        # rna_sequence: b, n, 6
                        # rna_pad_mask: b, n
                        # atac_sequence: b, n, 6
                        # tgt: b, n (A 01 sequence)
                        
                        logist = model(rna_sequence, atac_sequence, rna_value, enc_mask=enc_pad_mask) # b, n
                        logist_hat = torch.sigmoid(logist)
                        y_hat_out = torch.cat((y_hat_out, accelerator.gather_for_metrics(logist_hat).to('cpu')), dim=0)
                        accelerator.wait_for_everyone()
                    accelerator.free_dataloader()
        
        np.save('gene_files/npy/'+gene_name+'_predict.npy', y_hat_out.numpy())

if __name__ == "__main__":
    main()


accelerate launch --config_file accelerator_config_test.yaml --main_process_port 29824 rna2atac_test_ko.py \
                        -d preprocessed_data_test_nmp \
                        -l save_mlt_40/2024-07-26_rna2atac_train_300/pytorch_model.bin --config_file rna2atac_config_test.yaml

accelerate launch --config_file accelerator_config_test.yaml --main_process_port 29824 rna2atac_test_ko.py \
                        -d preprocessed_data_test_nmp_9516 \
                        -l save_mlt_40/2024-07-26_rna2atac_train_300/pytorch_model.bin --config_file rna2atac_config_test.yaml

import scanpy as sc
wt_atac = sc.read_h5ad('atac_wt_3_types.h5ad')
wt_atac[wt_atac.obs['cell_anno']=='NMP'].copy().write('atac_wt_nmp.h5ad')
wt_atac[wt_atac.obs['cell_anno']=='Somitic_mesoderm'].copy().write('atac_wt_som.h5ad')


## python cal_delt_r.py -g gene_lst.txt 
import argparse
import scanpy as sc
import numpy as np
import snapatac2 as snap
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import pandas as pd
from multiprocessing import Pool
    
parser = argparse.ArgumentParser(description='In silico KO infers cell differentiation')
parser.add_argument('-g', '--genes', type=str, help='gene files')

args = parser.parse_args()
gene_file = args.genes
gene_lst = []
with open(gene_file, 'r') as f:
    for line in f:
        gene_lst.append(line.strip('\n'))

wt_nmp_true = sc.read_h5ad('atac_wt_nmp.h5ad')
wt_som_true = sc.read_h5ad('atac_wt_som.h5ad')
wt_rna = sc.read_h5ad('rna_wt_3_types.h5ad')
wt_atac_pred = sc.read_h5ad('mlt_40_predict/rna2atac_scm2m_binary.h5ad')
delta_r = []

for gene in tqdm(gene_lst, ncols=80, desc='In silico KO'):
    gene_idx = np.argwhere(wt_rna.var.index==gene).item()
    wt_nmp_pred = wt_atac_pred[(wt_rna.obs['cell_anno']=='NMP') & (wt_rna.X[:, gene_idx].toarray().flatten()!=0)].copy()
    
    ko_nmp = np.load('gene_files/npy/'+gene+'_predict.npy')
    ko_nmp[ko_nmp>0.5] = 1
    ko_nmp[ko_nmp<=0.5] = 0
    
    dat = sc.AnnData(X=np.vstack([wt_som_true.X.toarray(), wt_nmp_pred.X.toarray(), ko_nmp]),
                     obs={'cell_anno': ['wt_som']*wt_som_true.shape[0]+['wt_nmp']*wt_nmp_pred.shape[0]+['ko_nmp']*ko_nmp.shape[0]}, 
                     var=wt_som_true.var)
    dat.X = csr_matrix(dat.X)
    snap.pp.select_features(dat, verbose=False)
    snap.tl.spectral(dat)
    
    wt_r = cosine_similarity(dat[dat.obs['cell_anno']=='wt_som'].obsm['X_spectral'], dat[dat.obs['cell_anno']=='wt_nmp'].obsm['X_spectral'])
    ko_r = cosine_similarity(dat[dat.obs['cell_anno']=='wt_som'].obsm['X_spectral'], dat[dat.obs['cell_anno']=='ko_nmp'].obsm['X_spectral'])
    pd.DataFrame({'gene':[gene], 'delta_r':[(ko_r-wt_r).mean()]}).to_csv('gene_files/delta_r/'+gene+'_delta_r_out.txt', index=None, header=None, sep='\t')


sed -n '1,1000p' gene_lst.txt > gene_lst_p1.txt
sed -n '1001,2000p' gene_lst.txt > gene_lst_p2.txt
sed -n '2001,3000p' gene_lst.txt > gene_lst_p3.txt
sed -n '3001,4000p' gene_lst.txt > gene_lst_p4.txt
sed -n '4001,4203p' gene_lst.txt > gene_lst_p5.txt

nohup python cal_delt_r.py -g gene_lst_p1.txt > gene_lst_p1.log &   # 1003529
nohup python cal_delt_r.py -g gene_lst_p2.txt > gene_lst_p2.log &   # 1003530
nohup python cal_delt_r.py -g gene_lst_p3.txt > gene_lst_p3.log &   # 1003531
nohup python cal_delt_r.py -g gene_lst_p4.txt > gene_lst_p4.log &   # 1003532
nohup python cal_delt_r.py -g gene_lst_p5.txt > gene_lst_p5.log &   # 1003533

# with Pool(1) as p:
#     p.map(cal_delta_r, gene_lst[:1])




# from sklearn.manifold import SpectralEmbedding
# embedding = SpectralEmbedding(n_components=20)
# spec_model = embedding.fit(wt_som_true.X.toarray())
# som = spec_model.fit_transform(wt_som_true.X.toarray())
# wt = spec_model.fit_transform(wt_nmp_pred.X.toarray())
# ko = spec_model.fit_transform(ko_nmp)
# wt_r = cosine_similarity(wt, som)
# ko_r = cosine_similarity(ko, som)
# (ko_r-wt_r).mean()

# wt_r = cosine_similarity(wt_nmp_pred.X.toarray(), wt_som_true.X.toarray())
# ko_r = cosine_similarity(ko_nmp, wt_som_true.X.toarray())
# (ko_r-wt_r).mean()


# from scipy import stats
# stats.ttest_rel(ko_r.flatten(), wt_r.flatten())

# enc_length: 8534
# T:       -1.54024272004445e-05
# Topors:  5.243919720902774e-06
# Cdx2:    -3.87178734486313e-05
# Ctcf:    0.0001503606044919368
# Got2:    2.7477864537743706e-05
# Tcea1:   -1.4352654955767388e-05


## KO & OE
## stablize cell identity
################################### KO data is necessary? ##################################



wt_nmp_true = sc.read_h5ad('atac_wt_nmp.h5ad')
wt_som_true = sc.read_h5ad('atac_wt_som.h5ad')
wt_rna = sc.read_h5ad('rna_wt_3_types.h5ad')
wt_atac_pred = sc.read_h5ad('mlt_40_predict/rna2atac_scm2m_binary.h5ad')

gene = 'Cdx2'
gene_idx = np.argwhere(wt_rna.var.index==gene).item()
wt_nmp_pred = wt_atac_pred[(wt_rna.obs['cell_anno']=='NMP') & (wt_rna.X[:, gene_idx].toarray().flatten()!=0)].copy()

ko_nmp = np.load('gene_files/npy/'+gene+'_predict.npy')
ko_nmp[ko_nmp>0.5] = 1
ko_nmp[ko_nmp<=0.5] = 0

dat = sc.AnnData(X=np.vstack([wt_som_true.X.toarray(), wt_nmp_pred.X.toarray(), ko_nmp]),
                 obs={'cell_anno': ['wt_som']*wt_som_true.shape[0]+['wt_nmp']*wt_nmp_pred.shape[0]+['ko_nmp']*ko_nmp.shape[0]}, 
                 var=wt_som_true.var)
dat.X = csr_matrix(dat.X)
snap.pp.select_features(dat, verbose=False)
snap.tl.spectral(dat)
snap.tl.umap(dat)
sc.pl.umap(dat, color='cell_anno', legend_fontsize='7', legend_loc='right margin', size=10,
           title='', frameon=True, save='_tmp_new.pdf')

import matplotlib.pyplot as plt
start_indices = list(range(wt_som_true.shape[0], wt_som_true.shape[0]+wt_nmp_pred.shape[0]))
end_indices = list(range(wt_som_true.shape[0]+wt_nmp_pred.shape[0], wt_som_true.shape[0]+wt_nmp_pred.shape[0]+ko_nmp.shape[0]))
for start_idx, end_idx in zip(start_indices, end_indices):
    arrow_start = dat.obsm['X_umap'][start_idx]
    arrow_end = dat.obsm['X_umap'][end_idx]
    arrow_vector = arrow_end - arrow_start
    plt.arrow(arrow_start[0], arrow_start[1],
              arrow_vector[0], arrow_vector[1],
              head_width=0.1, head_length=0.1, fc='k', ec='k')
plt.savefig('umap_tmp_new.pdf')
plt.close()














