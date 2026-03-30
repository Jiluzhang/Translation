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
from chromnitron_model.load_model import load_chromnitron
from chromnitron_model.chromnitron_models import Chromnitron

def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def load_inputs(config):
    loci_info, chrs = read_region_bed(os.path.join(config['inference_config']['input']['root'], config['inference_config']['input']['locus_list_path']))
    celltype_list = read_list(os.path.join(config['inference_config']['input']['root'], config['inference_config']['input']['celltype_list_path']))
    cap_list = read_list(os.path.join(config['inference_config']['input']['root'], config['inference_config']['input']['cap_list_path']))
    return loci_info, chrs, celltype_list, cap_list

def load_data(config, celltype, loci_info, cap, chr_sizes):
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
    data = InferenceDataset(loci_info, input_seq_path, input_features_path, esm_feature_path, assembly, chr_sizes, metadata_key = celltype, excluded_region_path = excluded_region_path)

    batch_size = config['inference_config']['inference']['batch_size']
    num_workers = config['inference_config']['inference']['num_workers']
    batch_size = min(batch_size, len(data) // 2 + 1) # Ensure at least 2 batches
    dataloader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return dataloader


config_path = sys.argv[1]
config = load_yaml(config_path)
loci_info, chrs, celltype_list, cap_list = load_inputs(config) # Load inputs

## Training  (similar to inference step)
if config['training_config']['training']['enable']:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Chromnitron()
    model = model.to(device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model.train()

    ## load input
    chr_sizes = get_chr_sizes(config, chrs)
    dataloader = load_data(config, celltype, loci_info, cap, chr_sizes)
    
    pred_cache, label_df = run_inference(config, model, dataloader, celltype, cap)
    save_prediction(pred_cache, label_df, config, celltype, cap)
    

    
    





























