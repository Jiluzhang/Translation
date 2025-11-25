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
# pip install ema_pytorch
# export PYTHONPATH="$PYTHONPATH:/fs/home/jiluzhang/TF_grammar/scPrinter/scPrinter-main"  -->> ~.bashrc


# https://ruochiz.com/scprinter_doc/tutorials/PBMC_bulkATAC_tutorial.html

import scprinter as scp
# Downloading several files to '/data/home/jiluzhang/.cache/scprinter'.

# https://zenodo.org/records/14866808/files/PBMC_bulk_ATAC_tutorial_Bcell_0_frags.tsv.gz?download=1
# https://zenodo.org/records/14866808/files/PBMC_bulk_ATAC_tutorial_Bcell_1_frags.tsv.gz?download=1
# https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_41/GRCh38.primary_assembly.genome.fa.gz
# https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_41/gencode.v41.basic.annotation.gff3.gz

# https://zenodo.org/records/15399859/files/sacCer3_bias_v2.h5?download=1

# datasets.py (line 89: "hg38_bias_v2.h5": "md5:c56f03c32bc793bf2a5cf23d4f66f20e", -> "hg38_bias_v2.h5": "md5:a5e8c0162be0152c75e0af20125b9e42",)
# https://github.com/Boyle-Lab/Blacklist/raw/master/lists/hg38-blacklist.v2.bed.gz

zcat PBMC_bulk_ATAC_tutorial_Bcell_0_frags.tsv.gz | awk '{if($1=="chr21" && $2>10000000 && $2<20000000) print $0}' > PBMC_bulk_ATAC_tutorial_Bcell_0_frags_chr21.tsv
gzip PBMC_bulk_ATAC_tutorial_Bcell_0_frags_chr21.tsv
zcat PBMC_bulk_ATAC_tutorial_Bcell_1_frags.tsv.gz | awk '{if($1=="chr21" && $2>10000000 && $2<20000000) print $0}' > PBMC_bulk_ATAC_tutorial_Bcell_1_frags_chr21.tsv
gzip PBMC_bulk_ATAC_tutorial_Bcell_1_frags_chr21.tsv

import os
import pandas as pd
import scprinter as scp

main_dir = '/fs/home/jiluzhang/TF_grammar/scPrinter/test'
work_dir = '/fs/home/jiluzhang/TF_grammar/scPrinter/test/seq2print'
frag_files = ['PBMC_bulk_ATAC_tutorial_Bcell_0_frags_chr21.tsv.gz',
              'PBMC_bulk_ATAC_tutorial_Bcell_1_frags_chr21.tsv.gz']
samples = ['Bcell_0', 'Bcell_1']

## preprocessing
printer = scp.pp.import_fragments(path_to_frags=frag_files, barcodes=[None]*len(frag_files),
                                  savename=os.path.join(work_dir, 'PBMC_bulkATAC_scprinter.h5ad'),
                                  genome=scp.genome.hg38, min_num_fragments=1000, min_tsse=7,
                                  sorted_by_barcode=False, low_memory=False, n_jobs=10)

## Call peaks, this set of peaks are recommended to train seq2PRINT model
scp.pp.call_peaks(printer=printer, frag_file=frag_files, cell_grouping=[None],
                  group_names=['all'], preset='seq2PRINT', overwrite=False)
cleaned_peaks = pd.DataFrame(printer.uns["peak_calling"]['all_cleaned'][:])
cleaned_peaks.to_csv('seq2print_cleaned_narrowPeak.bed', sep='\t', header=False, index=False)

## Call peaks using chromvar preset, this set of peak are recommended to be use as cell x peak for scATAC-seq data, or analysis
scp.pp.call_peaks(printer=printer, frag_file=frag_files, cell_grouping=[None],
                  group_names=['chromvar_all'], preset='chromvar', overwrite=False)
cleaned_peaks = pd.DataFrame(printer.uns["peak_calling"]['chromvar_all_cleaned'][:])
cleaned_peaks.to_csv('regions.bed', sep='\t', header=False, index=False)

## Training separate seq2PRINT models for each sample
model_configs = []
for sample in samples:
    fold=0
    model_config = scp.tl.seq_model_config(printer, region_path='seq2print_cleaned_narrowPeak.bed',
                                          cell_grouping=[sample], group_names=sample,
                                          genome=printer.genome, fold=fold, overwrite_bigwig=False,
                                          model_name='PBMC_bulkATAC_'+sample,
                                          additional_config={"notes":"v3", "tags":["PBMC_bulkATAC", sample, f"fold{fold}"]},
                                          path_swap=(work_dir, ''), config_save_path=f'{work_dir}/configs/PBMC_bulkATAC_{sample}_fold{fold}.JSON')
    model_configs.append(model_config)

for sample in samples:
    scp.tl.launch_seq2print(model_config_path=f'{work_dir}/configs/PBMC_bulkATAC_{sample}_fold{fold}.JSON',
                            temp_dir=f'{work_dir}/temp', model_dir=f'{work_dir}/model', data_dir=work_dir,
                            gpus=0, wandb_project='scPrinter_seq_PBMC_bulkATAC',
                            verbose=False, launch=False)

CUDA_VISIBLE_DEVICES=0 python /fs/home/jiluzhang/TF_grammar/scPrinter/scPrinter-main/scprinter/seq/scripts/seq2print_lora_train.py \
                                       --config /fs/home/jiluzhang/TF_grammar/scPrinter/test/seq2print/configs/PBMC_bulkATAC_Bcell_0_fold0.JSON \
                                       --temp_dir /fs/home/jiluzhang/TF_grammar/scPrinter/test/seq2print/temp \
                                       --model_dir /fs/home/jiluzhang/TF_grammar/scPrinter/test/seq2print/model \
                                       --data_dir /fs/home/jiluzhang/TF_grammar/scPrinter/test/seq2print \
                                       --project scPrinter_seq_PBMC_bulkATAC --enable_wandb


[urlOpen] Couldn't open /fs/home/jiluzhang/TF_grammar/scPrinter/test/seq2print/PBMC_bulkATAC_scprinter_supp/Bcell_0.bw for reading
[urlOpen] Couldn't open /fs/home/jiluzhang/TF_grammar/scPrinter/test/seq2print/PBMC_bulkATAC_scprinter_supp/Bcell_0.bw for reading


# CUDA_VISIBLE_DEVICES=0 seq2print_train --config /fs/home/jiluzhang/TF_grammar/scPrinter/test/seq2print/configs/PBMC_bulkATAC_Bcell_0_fold0.JSON \
#                                        --temp_dir /fs/home/jiluzhang/TF_grammar/scPrinter/test/seq2print/temp \
#                                        --model_dir /fs/home/jiluzhang/TF_grammar/scPrinter/test/seq2print/model \
#                                        --data_dir /fs/home/jiluzhang/TF_grammar/scPrinter/test/seq2print \
#                                        --project scPrinter_seq_PBMC_bulkATAC --enable_wandb

wandb login
#api key: 81a507e12b0e6ed78f49d25ac30c57b5bc3c26db

# /data/home/jiluzhang/miniconda3/envs/scPrinter/bin  cp macs3 macs2

