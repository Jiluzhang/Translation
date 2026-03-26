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

python ./Chromnitron-main/chromnitron/inference.py ./Chromnitron-main/chromnitron/examples/local_config.yaml  # bug: Missing key(s) in state_dict

