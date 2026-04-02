#### workdir: /fs/home/jiluzhang/TF_grammar/Chromnitron
#### github: https://github.com/tanjimin/Chromnitron/tree/main/chromnitron

# wget -c https://codeload.github.com/tanjimin/Chromnitron/zip/refs/heads/main
# conda create -n chromnitron python==3.9
# conda activate chromnitron
# pip install -r ./Chromnitron-main/chromnitron/requirements_cuda12.txt

# pip install -i https://pypi.tuna.tsinghua.edu.cn/simple pandas==2.3.1 zarr==2.18.0 numcodecs==0.12.1 tqdm==2.2.3 \
#                                                         pyyaml==5.4.1 pyBigWig==0.3.24 torch==2.5.0 torchvision==0.20.0 \
#                                                         loralib   # from requirements_cuda12.txt
# pip install ipython

# wget -O input_resources.tar \
# https://www.dropbox.com/scl/fi/pdtb53vrioxufkyddc5je/input_resources.tar?rlkey=1re0r0k6cisazqyyspcnrhdhg&st=ykb14uen&dl=1

# wget -O model_weights_subset.tar \
# https://www.dropbox.com/scl/fi/dbi8gc2ej2zi4vktambnn/model_weights_subset.tar?rlkey=gd69we2pq6awjhkfh944iusdz&st=w0i2zgop&dl=1

# tar -xf model_weights_subset.tar
# tar -xf input_resources.tar

# Edit configuration and prepare for inference

## add class ConvTranspose1d to /fs/home/jiluzhang/softwares/miniconda3/envs/chromnitron/lib/python3.9/site-packages/loralib/layers.py

python ./Chromnitron-main/chromnitron/inference.py ./Chromnitron-main/chromnitron/examples/local_config.yaml

# "use_finetune: disable" (auto -> disable) in local_config.yaml





## modify inference.py -> training.py
import sys
import os
import argparse
import yaml
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

sys.path.append('/fs/home/jiluzhang/TF_grammar/Chromnitron/Chromnitron-main/chromnitron')

from chromnitron_model.load_model import load_chromnitron
from chromnitron_model.chromnitron_models import Chromnitron

# Use TF32
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def load_inputs(config):
    loci_info, chrs = read_region_bed(os.path.join(config['inference_config']['input']['root'], config['inference_config']['input']['locus_list_path']))
    celltype_list = read_list(os.path.join(config['inference_config']['input']['root'], config['inference_config']['input']['celltype_list_path']))
    cap_list = read_list(os.path.join(config['inference_config']['input']['root'], config['inference_config']['input']['cap_list_path']))
    return loci_info, chrs, celltype_list, cap_list

def read_region_bed(region_path):
    loci_info = []
    chrs = set()
    with open(region_path, 'r') as file:
        for line in file:
            chrom, start, end = line.strip().split('\t')[:3]
            # Remove any non autosomes
            chr_num = chrom.split('chr')[1]
            if chr_num.isdigit():
                loci_info.append([chrom, int(start), int(end)])
                chrs.add(chrom)
    chrs = list(chrs)
    return loci_info, chrs

def read_list(list_path):
    item_list = []
    with open(list_path, 'r') as file:
        for line in file:
            item_list.append(line.strip())
    return item_list

def get_chr_sizes(config, chrs):
    chr_sizes = {}
    input_dict = config['input_resource']
    chr_path = os.path.join(input_dict['root'], input_dict['sequence'], f'{config["inference_config"]["input"]["assembly"]}.chrom.sizes')
    with open(chr_path, 'r') as file:
        for line in file:
            chrom, size = line.strip().split('\t')
            chr_sizes[chrom] = int(size)
    # Only includes chromosomes in chrs for easy processing
    chr_sizes = {chr: chr_sizes[chr] for chr in chrs}
    return chr_sizes

#Luz def load_data(config, celltype, loci_info, cap, chr_sizes):
def load_data(config, celltype, loci_info, cap, chr_sizes, binding_label_path):
    input_dict = config['input_resource']
    input_seq_path = os.path.join(input_dict['root'], input_dict['sequence'], f'{config["inference_config"]["input"]["assembly"]}.zarr')
    input_features_path = os.path.join(input_dict['root'], input_dict['atac'], f'{celltype}.zarr')
    assembly = config['inference_config']['input']['assembly']
    esm_feature_path = os.path.join(input_dict['root'], input_dict['cap'], f'{cap}.npz')

    if config['inference_config']['input']['excluded_region_path'] == 'auto':
        excluded_region_path = f"{config['input_resource']['root']}/{config['input_resource']['sequence']}/{assembly}-blacklist.v2.bed"
    else:
        excluded_region_path = config['inference_config']['input']['excluded_region_path']
    if not os.path.exists(excluded_region_path):
        print(f'WARNING: {excluded_region_path} does not exist, using all regions')

    from chromnitron_data.chromnitron_dataset import InferenceDataset
    data = InferenceDataset(loci_info, input_seq_path, input_features_path, esm_feature_path, binding_label_path, assembly, chr_sizes, metadata_key = celltype, excluded_region_path = excluded_region_path)
    #Luz data = InferenceDataset(loci_info, input_seq_path, input_features_path, esm_feature_path, assembly, chr_sizes, metadata_key = celltype, excluded_region_path = excluded_region_path)

    batch_size = config['inference_config']['inference']['batch_size']
    num_workers = config['inference_config']['inference']['num_workers']
    batch_size = min(batch_size, len(data) // 2 + 1) # Ensure at least 2 batches
    dataloader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return dataloader


# def run_inference(config, model, dataloader, celltype, cap, use_tqdm=True):
#     pred_cache = []
#     label_cache_dict = {'chr':[], 'start':[], 'end':[], 'region_id':[],
#                         'celltype':celltype, 'cap':cap}

#     # Use TF32
#     torch.backends.cuda.matmul.allow_tf32 = True
#     torch.backends.cudnn.allow_tf32 = True

#     # Run inference
#     with torch.no_grad():
#         if use_tqdm:
#             from tqdm import tqdm
#             dataloader = tqdm(dataloader)
#         for batch in dataloader:
#             device = 'cuda' if torch.cuda.is_available() else 'cpu'
#             seq, input_features, esm_embeddings, loc_info = batch
#             seq = seq.to(device)
#             input_features = input_features.to(device)
#             esm_embeddings = esm_embeddings.to(device)

#             esm_embeddings = esm_embeddings.float().transpose(-1, -2)

#             batch_size, mini_bs, seq_len, seq_dim = seq.shape
#             seq = seq.view(batch_size*mini_bs, seq_len, seq_dim)
#             input_features = input_features.view(batch_size*mini_bs, -1)
#             seq = seq.transpose(1, 2).float()
#             input_features = input_features.unsqueeze(2).transpose(1, 2).float()

#             inputs = (seq, input_features)

#             preds, confidence = model(inputs, esm_embeddings)
#             preds = preds.detach().cpu().numpy()[:, 0, :]
#             pred_cache.append(preds)
#             label_cache_dict['start'].extend(loc_info[0].tolist())
#             label_cache_dict['end'].extend(loc_info[1].tolist())
#             label_cache_dict['chr'].extend(loc_info[2])
#             label_cache_dict['region_id'].extend(loc_info[3])

#     pred_cache = np.concatenate(pred_cache, axis=0)
#     # Exponential transform
#     pred_cache = np.exp(pred_cache) - 1
#     label_df = pd.DataFrame(label_cache_dict)
#     return pred_cache, label_df


# def run_training(config, model, dataloader, celltype, cap, use_tqdm=True):
def run_training(config, model, train_dataloader, val_dataloader=None, num_epochs=10, lr=1e-4):
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    train_loss_history = []
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        train_loader = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch_idx, batch in enumerate(train_loader):
            #Luz seq, input_features, esm_embeddings, loc_info = batch  # shape: seq[8, 1, 8192, 5], input_features[8, 1, 8192], esm_embeddings[8, 1, 4096, 2560], loc_info[5]
            seq, input_features, esm_embeddings, loc_info, binding_label = batch
            seq = seq.to(device)
            input_features = input_features.to(device)
            esm_embeddings = esm_embeddings.to(device)
            esm_embeddings = esm_embeddings.float().transpose(-1, -2)
            
            batch_size, mini_bs, seq_len, seq_dim = seq.shape
            seq = seq.view(batch_size*mini_bs, seq_len, seq_dim)
            input_features = input_features.view(batch_size*mini_bs, -1)
            seq = seq.transpose(1, 2).float()
            input_features = input_features.unsqueeze(2).transpose(1, 2).float()
            
            inputs = (seq, input_features)

            optimizer.zero_grad()
            preds = model(inputs, esm_embeddings)  # preds, confidence = model(inputs, esm_embeddings)  # shape: preds[8, 1, 8192], confidence[8, 1, 8192]
            
            #Luz y_true = torch.randn(preds.shape).to(device)  # 请替换为真实标签!!!!!!!!!!
            y_true = binding_label.to(device)
            y_true = y_true.view(batch_size*mini_bs, -1)
            total_loss = criterion(preds, y_true)
            
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += total_loss.item()
            train_loader.set_postfix_str(f"loss: {total_loss.item():.4f}")  # show loss in the tqdm
        
        scheduler.step()
        
        avg_loss = epoch_loss / len(train_dataloader)
        train_loss_history.append(avg_loss)
        print(f"Epoch {epoch+1} average training loss: {avg_loss:.4f}")
        
        if val_dataloader is not None:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for val_batch in val_dataloader:
                    seq, input_features, esm_embeddings, _, binding_label = val_batch
                    #Luz seq, input_features, esm_embeddings, _ = val_batch
                    seq = seq.to(device)
                    input_features = input_features.to(device)
                    esm_embeddings = esm_embeddings.float().transpose(-1, -2).to(device)
                    
                    batch_size, mini_bs, seq_len, seq_dim = seq.shape
                    seq = seq.view(batch_size*mini_bs, seq_len, seq_dim).transpose(1,2).float()
                    input_features = input_features.view(batch_size*mini_bs, -1)
                    input_features = input_features.unsqueeze(2).transpose(1,2).float()
                    
                    inputs = (seq, input_features)
                    preds = model(inputs, esm_embeddings) #preds, confidence = model(inputs, esm_embeddings)
                    
                    y_true = binding_label.to(device) #Luz y_true = torch.randn(preds.shape).to(device)
                    y_true = y_true.view(batch_size*mini_bs, -1)
                    val_loss += criterion(preds, y_true).item()
            
            print(f"Validating loss: {val_loss/len(val_dataloader):.4f}\n")
    
    print("Training done!!!")
    # return model, train_loss_history
    torch.save(model, 'chromnitron.pth')


config_path = './Chromnitron-main/chromnitron/examples/local_config.yaml'
config = load_yaml(config_path)
loci_info, chrs, celltype_list, cap_list = load_inputs(config)   # Load inputs
binding_label_path = config['input_resource']['binding']

## Training  (similar to inference step)
# if config['training_config']['training']['enable']:

## init_model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Chromnitron()
model = model.to(device)

## load input
celltype = 'HepG2'
cap_1, cap_2 = 'CTCF', 'JUND'
chr_sizes = get_chr_sizes(config, chrs)
train_dataloader = load_data(config, celltype, loci_info, cap_1, chr_sizes, binding_label_path)
val_dataloader = load_data(config, celltype, loci_info, cap_2, chr_sizes, binding_label_path)
#Luz train_dataloader = load_data(config, celltype, loci_info, cap_1, chr_sizes)
#Luz val_dataloader = load_data(config, celltype, loci_info, cap_2, chr_sizes)

# pred_cache, label_df = run_inference(config, model, dataloader, celltype, cap)
run_training(config, model, train_dataloader, val_dataloader, num_epochs=10, lr=1e-5)



def export_to_zarr(zarr_name, chr_sizes, data_dict, chunk_size=1000000):
    import zarr
    import numcodecs
    root = zarr.group(store=zarr_name, overwrite=True) # init zarr group
    zarr_compressor = numcodecs.Blosc(cname='zstd', clevel=3, shuffle=numcodecs.Blosc.SHUFFLE) # Setup compressor
    for chr_name in chr_sizes.keys():
        print('Processing and saving', chr_name)
        chr_arr = data_dict[chr_name]
        root.create_dataset(f'chrs/{chr_name}', data=chr_arr, chunks=chunk_size, compressor=zarr_compressor)


## bigwig to dict
import pyBigWig
import pandas as pd

data_dict = {}
chrom_sizes = pd.read_table('/fs/home/jiluzhang/TF_grammar/Chromnitron/input_resources/DNA_sequence/hg38.chrom.sizes', header=None)
with pyBigWig.open('/fs/home/jiluzhang/TF_grammar/PRINT/cfoot_atac_cfoot/pipeline/human_nakedDNA.bw', 'r') as f:
    for chrom in ['chr'+str(i) for i in range(1, 23)]+['chrX']:
        data_dict[chrom] = f.values(chrom, 0, chrom_sizes[chrom_sizes[0]==chrom][1].item(), numpy=True)
        print(chrom)

export_to_zarr('Luz', chrom_sizes.set_index(0)[1].to_dict(), data_dict, chunk_size=1000000)





























