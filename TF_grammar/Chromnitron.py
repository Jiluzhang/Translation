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

def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def load_inputs(config):
    loci_info, chrs = read_region_bed(os.path.join(config['inference_config']['input']['root'], config['inference_config']['input']['locus_list_path']))
    celltype_list = read_list(os.path.join(config['inference_config']['input']['root'], config['inference_config']['input']['celltype_list_path']))
    cap_list = read_list(os.path.join(config['inference_config']['input']['root'], config['inference_config']['input']['cap_list_path']))
    return loci_info, chrs, celltype_list, cap_list

def init_model(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    from chromnitron_model.chromnitron_models import Chromnitron
    model = Chromnitron()
    model = model.to(device)
    return model

config_path = sys.argv[1]
config = load_yaml(config_path)
loci_info, chrs, celltype_list, cap_list = load_inputs(config) # Load inputs

## Training
if config['training_config']['training']['enable']:
    model = load_chromnitron(config, cap)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    
    





























