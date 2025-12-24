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

import pandas as pd
import scprinter as scp

main_dir = '/fs/home/jiluzhang/TF_grammar/scPrinter/test'
work_dir = '/fs/home/jiluzhang/TF_grammar/scPrinter/test'
frag_files = ['PBMC_bulk_ATAC_tutorial_Bcell_0_frags_chr21.tsv.gz',
              'PBMC_bulk_ATAC_tutorial_Bcell_1_frags_chr21.tsv.gz']
samples = ['Bcell_0', 'Bcell_1']

## preprocessing
printer = scp.pp.import_fragments(path_to_frags=frag_files, barcodes=[None]*len(frag_files),
                                  savename='PBMC_bulkATAC_scprinter.h5ad',
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
                                          genome=printer.genome, fold=fold, overwrite_bigwig=True,
                                          model_name='PBMC_bulkATAC_'+sample,
                                          additional_config={"notes":"v3", "tags":["PBMC_bulkATAC", sample, f"fold{fold}"]},
                                          path_swap=('.', ''), 
                                          config_save_path=f'./PBMC_bulkATAC_{sample}_fold{fold}.JSON')
    model_configs.append(model_config)

for sample in samples:
    scp.tl.launch_seq2print(model_config_path=f'PBMC_bulkATAC_{sample}_fold{fold}.JSON',
                            temp_dir='./temp', model_dir='./model', data_dir='.',
                            gpus=0, wandb_project='scPrinter_seq_PBMC_bulkATAC',
                            verbose=False, launch=False)


import h5py
import numpy as np
raw = h5py.File('hg38_bias_v2_raw.h5', 'r')
with h5py.File('hg38_bias_v2.h5', 'w') as f_out:
    f_out.create_dataset('chr21', data=np.concatenate([raw['chrI'][:]]*100))



CUDA_VISIBLE_DEVICES=3 python /fs/home/jiluzhang/TF_grammar/scPrinter/scPrinter-main/scprinter/seq/scripts/seq2print_lora_train.py \
                              --config /fs/home/jiluzhang/TF_grammar/scPrinter/test/PBMC_bulkATAC_Bcell_0_fold0.JSON \
                              --temp_dir /fs/home/jiluzhang/TF_grammar/scPrinter/test/temp \
                              --model_dir /fs/home/jiluzhang/TF_grammar/scPrinter/test/model \
                              --data_dir /fs/home/jiluzhang/TF_grammar/scPrinter/test \
                              --project scPrinter_seq_PBMC_bulkATAC_Bcell_0 --enable_wandb

## workdir: /fs/home/jiluzhang/TF_grammar/scPrinter/test/luz
CUDA_VISIBLE_DEVICES=3 python /fs/home/jiluzhang/TF_grammar/scPrinter/scPrinter-main/scprinter/seq/scripts/seq2print_lora_train.py \
                              --config ./PBMC_bulkATAC_Bcell_0_fold0.JSON \
                              --temp_dir ./temp \
                              --model_dir ./model \
                              --data_dir . \
                              --project scPrinter_seq_PBMC_bulkATAC_Bcell_0

# dispersion model: /data/home/jiluzhang/.cache/scprinter/dispersion_model_py_v2.h5
# "peaks": "seq2print_cleaned_narrowPeak.bed"
# "signals": "Bcell_0.bw"

## scripts to generate dispersion model
# https://github.com/buenrostrolab/PRINT/blob/cadfc55251fa57b006bd460d2d2e4d067e1c085b/analyses/BAC/getBackgroundDispersion.R
# https://github.com/buenrostrolab/PRINT/blob/cadfc55251fa57b006bd460d2d2e4d067e1c085b/analyses/BAC/dispersionModel.ipynb

conda install r-base

install.packages("BiocManager")
BiocManager::install("GenomicRanges")
BiocManager::install("dplyr")
BiocManager::install("lattice")
install.packages("https://cran.r-project.org/src/contrib/Archive/Matrix/Matrix_1.6-0.tar.gz")
BiocManager::install("reticulate")
BiocManager::install("SummarizedExperiment")
BiocManager::install("ggplot2")
BiocManager::install("hdf5r")
# wget -c https://bioconductor.org/packages/release/data/annotation/src/contrib/BSgenome.Hsapiens.UCSC.hg38_1.4.5.tar.gz
install.packages("./BSgenome.Hsapiens.UCSC.hg38_1.4.5.tar.gz")
BiocManager::install("pbmcapply")
BiocManager::install("gtools")
BiocManager::install("data.table")
install.packages('R.utils')
BiocManager::install("pbapply")


## workdir: /fs/home/jiluzhang/TF_grammar/scPrinter/test/luz
library(GenomicRanges)

source('/fs/home/jiluzhang/TF_grammar/PRINT/code/utils.R')
source('/fs/home/jiluzhang/TF_grammar/PRINT/code/getBias.R')
source('/fs/home/jiluzhang/TF_grammar/PRINT/code/getFootprints.R')
source('/fs/home/jiluzhang/TF_grammar/PRINT/code/getCounts.R')


use_condaenv('PRINT')  # not use python installed by uv

# shuf selectedBACs_raw.txt | head > selectedBACs.txt

####################################
# Get genomic ranges of BAC clones #
####################################
# Load list of BACs
selectedBACs <- read.table("selectedBACs.txt")
colnames(selectedBACs) <- c("ID", "chr", "start", "end")

# Get genomic ranges of BACs
BACRanges <- GRanges(seqnames=selectedBACs$chr,
                     ranges=IRanges(start=selectedBACs$start, end=selectedBACs$end))
names(BACRanges) <- selectedBACs$ID
saveRDS(BACRanges, "BACRanges.rds")

#######################
# Retrieve BAC ranges #
#######################
# Load genomic ranges of BACs
BACRanges <- readRDS("BACRanges.rds")

# Convert BAC genomic ranges into 1kb tiles
tileRanges <- Reduce("c", GenomicRanges::slidingWindows(BACRanges, width=1000, step=1000))
tileRanges <- tileRanges[width(tileRanges)==1000]
tileBACs <- names(tileRanges)
saveRDS(tileRanges, "tileRanges.rds")

###################################################
# Get BAC predicted bias and Tn5 insertion track #
##################################################
# Initialize a footprintingProject object
projectName <- "BAC"
project <- footprintingProject(projectName=projectName, refGenome="hg38")
projectMainDir <- "./"
projectDataDir <- paste0(projectMainDir, "data/", projectName, "/")
# mkdir data
dataDir(project) <- projectDataDir
mainDir(project) <- projectMainDir

# Set the regionRanges slot
regionRanges(project) <- tileRanges

# Use our neural network to predict Tn5 bias for all positions in all regions
# Remember to make a copy of the Tn5_NN_model.h5 file in projectDataDir!!
project <- getRegionBias(project, nCores=16)
# cp /fs/home/jiluzhang/TF_grammar/PRINT/code/predictBias.py /fs/home/jiluzhang/TF_grammar/scPrinter/test/luz/code
# cp /fs/home/jiluzhang/TF_grammar/PRINT/cfoot/obsBias_PRINT.tsv /fs/home/jiluzhang/TF_grammar/scPrinter/test/luz/data/BAC/obsBias.tsv
# cp /fs/home/jiluzhang/TF_grammar/PRINT/data/shared/Tn5_NN_model.h5 /fs/home/jiluzhang/TF_grammar/scPrinter/test/luz/data/shared
saveRDS(regionBias(project), paste0(projectDataDir, "predBias.rds"))

# Load barcodes for each replicate
barcodeGroups <- data.frame(barcode=paste("rep", 1:5, sep=""), group=1:5)
groups(project) <- mixedsort(unique(barcodeGroups$group))

# Get position-by-tile-by-replicate ATAC insertion count tensor
# We go through all down-sampling rates and get a count tensor for each of them
counts <- list()
pathToFrags <- paste0(projectDataDir, "rawData/all.fragments.tsv.gz")  
# cp /fs/home/jiluzhang/TF_grammar/scPrinter/test/PBMC_bulk_ATAC_tutorial_Bcell_0_frags.tsv.gz /fs/home/jiluzhang/TF_grammar/scPrinter/test/luz/data/BAC/rawData/all.fragments.tsv.gz
counts[["all"]] <- countTensor(getCountTensor(project, pathToFrags, barcodeGroups, returnCombined=T))





for(downSampleRate in c(0.5, 0.2, 0.1, 0.05, 0.02, 0.01)){
  print(paste0("Getting count tensor for down-sampling rate = ", downSampleRate))
  system("rm -r ../../data/BAC/chunkedCountTensor")
  pathToFrags <- paste0("../../data/BAC/downSampledFragments/fragmentsDownsample", downSampleRate, ".tsv.gz")
  counts[[as.character(downSampleRate)]] <- countTensor(getCountTensor(project, pathToFrags, barcodeGroups, returnCombined = T))
}
system("rm -r ../../data/BAC/chunkedCountTensor")
saveRDS(counts, "../../data/BAC/tileCounts.rds")



# CUDA_VISIBLE_DEVICES=0 seq2print_train --config /fs/home/jiluzhang/TF_grammar/scPrinter/test/seq2print/configs/PBMC_bulkATAC_Bcell_0_fold0.JSON \
#                                        --temp_dir /fs/home/jiluzhang/TF_grammar/scPrinter/test/seq2print/temp \
#                                        --model_dir /fs/home/jiluzhang/TF_grammar/scPrinter/test/seq2print/model \
#                                        --data_dir /fs/home/jiluzhang/TF_grammar/scPrinter/test/seq2print \
#                                        --project scPrinter_seq_PBMC_bulkATAC --enable_wandb

wandb login
#api key: 81a507e12b0e6ed78f49d25ac30c57b5bc3c26db

# /data/home/jiluzhang/miniconda3/envs/scPrinter/bin  cp macs3 macs2

