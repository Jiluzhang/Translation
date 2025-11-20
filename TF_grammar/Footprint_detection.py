## workdir: /fs/home/jiluzhang/TF_grammar/scPrinter
## https://github.com/buenrostrolab/scPrinter
## https://github.com/buenrostrolab/PRINT/tree/main
## https://zenodo.org/records/15399859

# conda create --name scPrinter python=3.9
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
# conda install conda-forge::anndata
# pip install -i https://pypi.tuna.tsinghua.edu.cn/simple pynndescent
# conda install tqdm
# conda install -c bioconda moods pybedtools pyranges 
# pip install -i https://pypi.tuna.tsinghua.edu.cn/simple snapatac2 bioframe seaborn adjustText pyBigWig gffutils Bio statsmodels tangermeme dna_features_viewer scanpy wandb shap
# pip install -i https://pypi.tuna.tsinghua.edu.cn/simple numpy==1.25.0
# export PYTHONPATH="$PYTHONPATH:/fs/home/jiluzhang/TF_grammar/scPrinter/scPrinter-main"  -->> ~.bashrc


# https://ruochiz.com/scprinter_doc/tutorials/PBMC_bulkATAC_tutorial.html

import scprinter as scp
# Downloading several files to '/data/home/jiluzhang/.cache/scprinter'.

# https://zenodo.org/records/14866808/files/PBMC_bulk_ATAC_tutorial_Bcell_0_frags.tsv.gz?download=1
# https://zenodo.org/records/14866808/files/PBMC_bulk_ATAC_tutorial_Bcell_1_frags.tsv.gz?download=1
