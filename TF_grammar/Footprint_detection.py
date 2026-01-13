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
import re
import json

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

regions_dict = {'chr11:47378230-47379030': 'SPI1', # Highly expressed in monocytes and B cells
                'chr5:140633035-140633835': 'CD14', # Monocyte marker
                'chr19:41876833-41877633': 'CD79a', # B cell marker
                'chr11:118342294-118343094': 'CD3D', # T cell marker
                'chr17:58281312-58282112': 'MPO', # Monocyte marker
                'chr16:31259594-31260394': 'ITGAM', # Expressed in myeloid cells (including monocyte)
                'chr3:39281281-39282081': 'CX3CR1', # A pretty important gene driving disease-related monocyte cell states
               }

# Save example regions to a bed file
regions_df = []
for region in regions_dict:
    regions_df.append(re.split("[:-]", region))
regions_df = pd.DataFrame(regions_df)
regions_df.to_csv(f'{work_dir}/regions_test.bed', sep='\t', header=False, index=False)

model_path_dict = {'B_cell_0':'/fs/home/jiluzhang/TF_grammar/scPrinter/test/model/PBMC_bulkATAC_Bcell_0_fold0-lemon-cosmos-16.pt',
                   'B_cell_1':}
adata_tfbs = {}
for sample_ind, sample in enumerate(samples):
    adata_tfbs[sample] = scp.tl.seq_tfbs_seq2print(seq_attr_count=None,
                          seq_attr_footprint=None,
                          genome=printer.genome,
                          region_path=f'{work_dir}/regions_test.bed',
                          gpus=[7], # change it to the available gpus
                          model_type='seq2print',
                          model_path=model_path_dict[sample], # For now we just run on one fold but you can provide a list of paths to all 5 folds
                          lora_config=json.load(open(f'{work_dir}/configs/PBMC_bulkATAC_{sample}_fold0.JSON', 'r')),
                          group_names=[sample],
                          verbose=False,
                          launch=True,
                          return_adata=True, # turn this as True
                          overwrite_seqattr=True,
                          post_normalize=False,
                          save_key=f'PBMC_bulkATAC_{sample}_roi', # and input a save_key
                          save_path=work_dir)

##########################################
##########################################
############### HERE #####################
##########################################
##########################################
##########################################


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
BiocManager::install("doParallel")
BiocManager::install("doSNOW")
BiocManager::install("cladoRcpp")
BiocManager::install("FNN")
BiocManager::install("keras")
install.packages('https://cran.r-project.org/src/contrib/Archive/MASS/MASS_7.3-58.tar.gz')
BiocManager::install("survcomp")  # speed up: options(repos = c(CRAN = "https://mirrors.tuna.tsinghua.edu.cn/CRAN/"))
BiocManager::install("caTools")
conda install gsl
BiocManager::install("DirichletMultinomial")
install.packages('https://cran.r-project.org/src/contrib/Archive/TFMPvalue/TFMPvalue_0.0.9.tar.gz')
BiocManager::install("TFBSTools")
BiocManager::install("motifmatchr")


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
pathToFrags <- paste0(projectDataDir, "rawData/test.fragments.tsv.gz")  # zcat all.fragments.tsv.gz | shuf | head -n 1000000 > test.fragments.tsv # pathToFrags <- paste0(projectDataDir, "rawData/all.fragments.tsv.gz")    
# wget -c https://ftp.ncbi.nlm.nih.gov/geo/series/GSE216nnn/GSE216403/suppl/GSE216403_BAC.fragments.tsv.gz  529Mb
# mv GSE216403_BAC.fragments.tsv.gz all.fragments.tsv.gz
counts[["all"]] <- countTensor(getCountTensor(project, pathToFrags, barcodeGroups, returnCombined=T, chunkSize=100, nCores=3))  # set chunkSize!!!!!!

# Down-sample fragments data (from getObservedBias.R)
frags <- data.table::fread(paste0(projectDataDir, "rawData/test.fragments.tsv.gz")) # frags <- data.table::fread(paste0(projectDataDir, "rawData/all.fragments.tsv.gz"))
nFrags <- dim(frags)[1]
if(!dir.exists("./data/BAC/downSampledFragments/")){
  system("mkdir ./data/BAC/downSampledFragments")
}
for(downSampleRate in c(0.5, 0.2, 0.1, 0.05, 0.02, 0.01)){
  print(paste0("Downsampling rate: ", downSampleRate))
  downSampleInd <- sample(1:nFrags, as.integer(nFrags*downSampleRate))
  downSampledFrags <- frags[downSampleInd, ]
  gz <- gzfile(paste0("./data/BAC/downSampledFragments/fragmentsDownsample", downSampleRate, ".tsv.gz"), "w")
  write.table(downSampledFrags, gz, quote=F, row.names=F, col.names=F, sep="\t")
  close(gz)
}

for(downSampleRate in c(0.5, 0.2, 0.1, 0.05, 0.02, 0.01)){
  print(paste0("Getting count tensor for down-sampling rate = ", downSampleRate))
  system("rm -r ./data/BAC/chunkedCountTensor")
  pathToFrags <- paste0("./data/BAC/downSampledFragments/fragmentsDownsample", downSampleRate, ".tsv.gz")
  counts[[as.character(downSampleRate)]] <- countTensor(getCountTensor(project, pathToFrags, barcodeGroups, returnCombined=T))
}
system("rm -r ./data/BAC/chunkedCountTensor")
saveRDS(counts, "./data/BAC/tileCounts.rds")

##################################
# Get background dispersion data #
##################################
if(!dir.exists("./data/BAC/dispModelData")){
  system("mkdir ./data/BAC/dispModelData")
}

for(footprintRadius in seq(2, 3, 1)){
# for(footprintRadius in seq(2, 100, 1)){
  
  ############################### Get naked DNA Tn5 observations ###############################
  print(paste0("Generating background dispersion data for footprintRadius = ", footprintRadius))
  flankRadius <- footprintRadius
  
  # Record background observations of Tn5 bias and insertion density in the BAC data
  bgObsData <- NULL
  for(downSampleRate in names(counts)){
    bgObsData <- rbind(bgObsData, data.table::rbindlist(
      pbmcapply::pbmclapply(
        1:length(counts[[downSampleRate]]),
        function(tileInd){
          
          # Get predicted bias
          predBias <- regionBias(project)[tileInd, ]
          
          # Get Tn5 insertion
          tileCountTensor <- counts[[downSampleRate]][[tileInd]]
          tileCountTensor <- tileCountTensor %>% group_by(position) %>% summarize(insertion = sum(count))

          Tn5Insertion <- rep(0, length(predBias))
          Tn5Insertion[tileCountTensor$position] <- tileCountTensor$insertion
          
          # Get sum of predicted bias in left flanking, center, and right flanking windows
          biasWindowSums <- footprintWindowSum(predBias, footprintRadius, flankRadius)
          
          # Get sum of insertion counts in left flanking, center, and right flanking windows
          insertionWindowSums <- footprintWindowSum(Tn5Insertion, footprintRadius, flankRadius)
          
          # Combine results into a data.frame
          tileObsData <- data.frame(leftFlankBias = biasWindowSums$leftFlank,
                                    leftFlankInsertion = round(insertionWindowSums$leftFlank),
                                    centerBias = biasWindowSums$center,
                                    centerInsertion = round(insertionWindowSums$center),
                                    rightFlankBias = biasWindowSums$rightFlank,
                                    rightFlankInsertion = round(insertionWindowSums$rightFlank))
          tileObsData$leftTotalInsertion <- tileObsData$leftFlankInsertion + tileObsData$centerInsertion
          tileObsData$rightTotalInsertion <- tileObsData$rightFlankInsertion + tileObsData$centerInsertion
          tileObsData$BACInd <- tileBACs[tileInd]
          
          tileObsData
        },
        mc.cores=2  #Luz
      )
    ))
  }
  
  # Filter out regions with zero reads
  bgObsData <- bgObsData[(bgObsData$leftTotalInsertion>=1) & (bgObsData$rightTotalInsertion>=1), ]
  
  # Get features of background observations for left-side testing
  bgLeftFeatures <- as.data.frame(bgObsData[, c("leftFlankBias", "centerBias", "leftTotalInsertion")])
  bgLeftFeatures <- data.frame(bgLeftFeatures)
  bgLeftFeatures$leftTotalInsertion <- log10(bgLeftFeatures$leftTotalInsertion)
  leftFeatureMean <- colMeans(bgLeftFeatures)
  leftFeatureSD <- apply(bgLeftFeatures, 2, sd)
  bgLeftFeatures <- sweep(bgLeftFeatures, 2, leftFeatureMean)
  bgLeftFeatures <- sweep(bgLeftFeatures, 2, leftFeatureSD, "/")
  
  # Get features of background observations for right-side testing
  bgRightFeatures <- as.data.frame(bgObsData[,c("rightFlankBias", "centerBias", "rightTotalInsertion")])
  bgRightFeatures <- data.frame(bgRightFeatures)
  bgRightFeatures$rightTotalInsertion <- log10(bgRightFeatures$rightTotalInsertion)
  rightFeatureMean <- colMeans(bgRightFeatures)
  rightFeatureSD <- apply(bgRightFeatures, 2, sd)
  bgRightFeatures <- sweep(bgRightFeatures, 2, rightFeatureMean)
  bgRightFeatures <- sweep(bgRightFeatures, 2, rightFeatureSD, "/")
  
  ############################### Match background KNN observations and calculate distribution of center / (center + flank) ratio ###############################
  # We sample 1e5 data points
  set.seed(123)
  sampleInds <- sample(1:dim(bgObsData)[1], 1e5)
  sampleObsData <- bgObsData[sampleInds, ]
  sampleLeftFeatures <- bgLeftFeatures[sampleInds, ]
  sampleRightFeatures <- bgRightFeatures[sampleInds, ]
  
  # Match KNN observations in (flank bias, center bias, count sum) 3-dimensional space 
  leftKNN <- FNN::get.knnx(bgLeftFeatures, sampleLeftFeatures, k=500)$nn.index
  leftRatioParams <- t(sapply(
    1:dim(leftKNN)[1],
    function(i){
      KNNObsData <- bgObsData[leftKNN[i,],]
      KNNRatio <- KNNObsData$centerInsertion / KNNObsData$leftTotalInsertion
      ratioMean <- mean(KNNRatio)
      ratioSD <- sd(KNNRatio)
      c(ratioMean, ratioSD)
    }
  ))
  leftRatioParams <- as.data.frame(leftRatioParams)
  colnames(leftRatioParams) <- c("leftRatioMean", "leftRatioSD")
  
  rightKNN <- FNN::get.knnx(bgRightFeatures, sampleRightFeatures, k = 500)$nn.index
  rightRatioParams <- t(sapply(
    1:dim(rightKNN)[1],
    function(i){
      KNNObsData <- bgObsData[rightKNN[i,],]
      KNNRatio <- KNNObsData$centerInsertion / KNNObsData$rightTotalInsertion
      ratioMean <- mean(KNNRatio)
      ratioSD <- sd(KNNRatio)
      c(ratioMean, ratioSD)
    }
  ))
  rightRatioParams <- as.data.frame(rightRatioParams)
  colnames(rightRatioParams) <- c("rightRatioMean", "rightRatioSD")
  
  # Combine background observations on both sides
  dispModelData <- cbind(sampleObsData, leftRatioParams, rightRatioParams)
  dispModelData$leftTotalInsertion <- log10(dispModelData$leftTotalInsertion)
  dispModelData$rightTotalInsertion <- log10(dispModelData$rightTotalInsertion)
  
  # Save background observations to file
  write.table(dispModelData, paste0("./data/BAC/dispModelData/dispModelData", footprintRadius, "bp.txt"), quote=F, sep="\t")
  # paste0("../../data/BAC/dispModelData/dispModelData", footprintRadius
}


## dispersionModel.ipynb
import os
import sys
import h5py
import pandas as pd
import numpy as np
import copy
import tqdm
import pickle
import matplotlib
import matplotlib.pyplot as plt
import multiprocessing as mp
import scipy.stats as ss
from datetime import datetime
from keras.models import load_model
from keras.models import Sequential
from keras.layers import *
from keras.models import Model
from keras import backend as K
import tensorflow as tf  #Luz

np.random.seed(42)
for footprint_radius in range(2, 101):
    
    print("Training model for footprint radius = " + str(footprint_radius))

    dispersion_data = pd.read_csv("./data/BAC/dispModelData/dispModelData" +\
                                  str(footprint_radius) + "bp.txt", sep = "\t")

    # Get model input data and prediction target
    data = copy.deepcopy(dispersion_data.loc[:,["leftFlankBias", "rightFlankBias", "centerBias", 
                                                "leftTotalInsertion", "rightTotalInsertion"]])
    target = copy.deepcopy(dispersion_data.loc[:,["leftRatioMean", "leftRatioSD", "rightRatioMean", 
                                 "rightRatioSD"]])
    data = data.values
    target = target.values
    
    # Remove infinite values
    filter = np.isfinite(np.sum(target, axis = 1))
    data = data[filter, :]
    target = target[filter, :]
    dispersion_data = dispersion_data.loc[filter, :]

    # Get BAC indices for each observation
    BAC_inds = dispersion_data.loc[:, "BACInd"].values
    BACs = np.unique(BAC_inds)

    # Randomly split BACs into training, validation and test
    inds = np.arange(len(BACs))
    np.random.shuffle(inds)
    n_inds = len(inds)
    training_BACs = BACs[inds[:int(n_inds * 0.5)]]               #Luz training_BACs = BACs[inds[:int(n_inds * 0.8)]]
    val_BACs = BACs[inds[int(n_inds * 0.5):int(n_inds * 0.8)]]   #Luz val_BACs = BACs[inds[int(n_inds * 0.8):int(n_inds * 0.9)]]
    test_BACs = BACs[inds[int(n_inds * 0.8):]]                   #Luz test_BACs = BACs[inds[int(n_inds * 0.9):]]

    # Split individual observations by BAC into training, validation and test
    training_inds = [i for i in range(len(BAC_inds)) if BAC_inds[i] in training_BACs]
    val_inds = [i for i in range(len(BAC_inds)) if BAC_inds[i] in val_BACs]
    test_inds = [i for i in range(len(BAC_inds)) if BAC_inds[i] in test_BACs]

    # Rescale the data and target values
    data_mean = np.mean(data, axis = 0)
    data_sd = np.std(data, axis = 0)
    data = (data - data_mean) / data_sd
    target_mean = np.mean(target, axis = 0)
    target_sd = np.std(target, axis = 0)
    target = (target - target_mean) / target_sd

    # Randomly shuffle training data
    training_inds = np.array(training_inds)
    np.random.shuffle(training_inds)

    # Split data into training, validation and test
    training_data = data[training_inds, :]
    val_data = data[val_inds, :]
    test_data = data[test_inds, :]

    # Split targets into training, validation and test
    training_target = target[training_inds, :]
    val_target = target[val_inds, :]
    test_target = target[test_inds, :]

    # Model Initialization
    print("Training Tn5 dispersion model")
    inputs = Input(shape = (np.shape(data)[1], ))  #Luz inputs = Input(shape = (np.shape(data)[1]))
    fc1 = Dense(32,activation = "relu")(inputs)
    out = Dense(np.shape(target)[1],activation = "linear")(fc1)
    model = Model(inputs=inputs,outputs=out)  
    model.summary()
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse'])

    # Model training
    mse = tf.keras.losses.MeanSquaredError()
    prev_loss = np.inf
    for n_epoch in range(500):

        # New training epoch
        model.fit(training_data, 
                  training_target, 
                  batch_size=32, epochs = 1, 
                  validation_data=(val_data, val_target))  

        # Get MSE loss on the valdation set after current epoch
        val_pred = model.predict(val_data)
        mse_loss = mse(val_target, val_pred).numpy()
        print("MSE on validation set" + str(mse_loss))

        # Get pred-target correlation on the validation set after current epoch
        pred_corrs = [ss.pearsonr(val_target[:, i], val_pred[:, i])[0]
                      for i in range(np.shape(target)[1])]
        print("Pred-target correlation " + " ".join([str(i) for i in pred_corrs]))

        # If loss on validation set stops decreasing quickly, stop training and adopt the previous saved version
        print(mse_loss, prev_loss)
        if mse_loss - prev_loss > -0.001 and n_epoch > 5:
            break
        else:
            prev_loss = mse_loss

    # Save model to file
    model.save("./data/shared/dispModel/dispersionModel" + str(footprint_radius) + "bp.h5")


# CUDA_VISIBLE_DEVICES=0 seq2print_train --config /fs/home/jiluzhang/TF_grammar/scPrinter/test/seq2print/configs/PBMC_bulkATAC_Bcell_0_fold0.JSON \
#                                        --temp_dir /fs/home/jiluzhang/TF_grammar/scPrinter/test/seq2print/temp \
#                                        --model_dir /fs/home/jiluzhang/TF_grammar/scPrinter/test/seq2print/model \
#                                        --data_dir /fs/home/jiluzhang/TF_grammar/scPrinter/test/seq2print \
#                                        --project scPrinter_seq_PBMC_bulkATAC --enable_wandb

wandb login
#api key: 81a507e12b0e6ed78f49d25ac30c57b5bc3c26db

# /data/home/jiluzhang/miniconda3/envs/scPrinter/bin  cp macs3 macs2



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

# wget -c https://raw.githubusercontent.com/ruochiz/FigRmotifs/main/human_pfms_v4.txt
# > ENSG00000267281_LINE960_AC0235093_I
# A [147 179 712 1 1 994 0 3 1 83 997 2 231 240]
# C [305 110 16 3 2 4 995 2 3 915 2 160 378 215]
# G [255 420 269 0 875 1 0 994 0 0 0 16 126 342]
# T [294 291 3 997 122 1 4 1 996 1 1 822 265 203]
motifs = scp.motifs.Motifs("./human_pfms_v4.txt", scp.genome.hg38.fetch_fa(), scp.genome.hg38.bg)
motif2matrix = {motif.name.split("_")[2]: np.array([motif.counts['A'], motif.counts['C'], motif.counts['G'], motif.counts['T']]) for motif in motifs.all_motifs}

## normalization to keep colsum of 1
for m in motif2matrix:
    mm = motif2matrix[m]
    mm = mm / np.sum(mm, axis=0, keepdims=True)
    motif2matrix[m] = mm

def read_TF_loci(path):
    TF_prediction = pd.read_csv(path, sep='\t')
    if 'range' in TF_prediction.columns:
        ranges = pd.DataFrame([re.split(':|-', xx) for xx in TF_prediction['range']])
        ranges = ranges[[0,1,2,3]]
        v = np.array(ranges[3])
        v[v == ''] = '-'
        ranges[3] = v
        ranges.columns = ['chrom','start','end', 'strand']
        ranges['start'] = ranges['start'].astype('int')
        ranges['end'] = ranges['end'].astype('int')
        TF_prediction = pd.concat([TF_prediction, ranges], axis=1)
    else:
        TF_prediction = pd.read_csv(path, sep='\t', header=None)
        TF_prediction.columns = ['chrom','start','end', 'bound', 'TF']
        TF_prediction['strand'] = '+'
    # TF_prediction = TF_prediction.sort_values(by=['bound'], ascending=False) # sort such that bound = 1, comes first, and will be kept.
    TF_prediction['summit'] = (TF_prediction['start']-1 + TF_prediction['end']) // 2
    # TF_prediction = TF_prediction.drop_duplicates(['chrom','start', 'end'])
    # TF_prediction = TF_prediction[TF_prediction['chrom'].isin(['chr1', 'chr3', 'chr6'])]
    return TF_prediction

def read_peaks(path):
    df = pd.read_csv(path, sep='\t', header=None)
    df.columns = ['chrom', 'start', 'end']
    #print (df)
    return df

def get_normalization_factor(model, feat, peak, low=0.05, 
                             median = 0.5,
                             high=0.95, sample_num=30000):
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


tsv_paths = {'HepG2': "./data/TFBSPrediction/HepG2_pred_data.tsv"}
peaks_path = {'HepG2': "./data/HepG2/test_peaks.bed"}

tsvs = {cell:read_TF_loci(tsv_paths[cell]) for cell in tsv_paths}
peaks = {cell:read_peaks(peaks_path[cell]) for cell in peaks_path}
model_name = {'HepG2': ["./"]}

# python ./code/evaluation_model.py --pt ../../model/PBMC_bulkATAC_Bcell_0_fold0-lemon-cosmos-16.pt --peaks ./data/HepG2/test_peaks.bed --method shap_hypo --wrapper just_sum \
#                                   --nth_output 0-30 --gpus 2 --genome hg38 --overwrite --decay 0.85   # may be pt file is not right

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


#### https://github.com/buenrostrolab/PRINT/blob/main/analyses/TFBSPrediction/
#### TFBSTrainingData.R (for generate TFBSDataUnibind.h5 file)
## conda activate PRINT
# cp /fs/home/jiluzhang/TF_grammar/PRINT/code/predictBias.py /fs/home/jiluzhang/TF_grammar/scPrinter/test/luz/Unibind/code

source('/fs/home/jiluzhang/TF_grammar/PRINT/code/utils.R')
source("/fs/home/jiluzhang/TF_grammar/PRINT/code/getFootprints.R")
source("/fs/home/jiluzhang/TF_grammar/PRINT/code/getCounts.R")
source("/fs/home/jiluzhang/TF_grammar/PRINT/code/getBias.R")
source("/fs/home/jiluzhang/TF_grammar/PRINT/code/getAggregateFootprint.R")
source("/fs/home/jiluzhang/TF_grammar/PRINT/code/getTFBS.R")
use_condaenv('PRINT')


source("../../code/getCounts.R")
source("../../code/getBias.R")
source("../../code/getFootprints.R")
source("../../code/visualization.R")
source("../../code/getTFBS.R")
source("../../code/getAggregateFootprint.R")
library(hdf5r)
library(motifmatchr)

###################
# Load input data #
###################

projectName <- "HepG2" # One of "K562", "GM12878", "HepG2"
ChIPDataset <- "Unibind"# One of "ENCODE", "Unibind"

# Load footprint SummarizedExperiment object
project <- footprintingProject(projectName=projectName, refGenome="hg38")
projectMainDir <- "./"
projectDataDir <- paste0(projectMainDir, "data/", projectName, "/")  # mkdir data
dataDir(project) <- projectDataDir
mainDir(project) <- projectMainDir

# Load region ranges

## Download GSM6672041_HepG2_peaks.bed from GSE216464
## generate "regionsRanges.rds"
# shuf GSM6672041_HepG2_peaks.bed | head -n 1000 > test_peaks.bed
# regionBed <- read.table(paste0(projectDataDir, "test_peaks.bed")) # regionBed <- read.table(paste0(projectDataDir, "GSM6672041_HepG2_peaks.bed"))
# regions <- GRanges(paste0(regionBed$V1, ":", regionBed$V2, "-", regionBed$V3))
# regions <- resize(regions, 1000, fix="center")
# saveRDS(regions, paste0(projectDataDir, "regionRanges.rds"))

## Read rds file
regions <- readRDS(paste0(projectDataDir, "regionRanges.rds"))

# Set the regionRanges slot
regionRanges(project) <- regions

## get footprints
tmpDir <- dataDir(project)
chunkSize <- regionChunkSize(project)

## h5 to rds for dispersion model
# Sys.setenv(CUDA_VISIBLE_DEVICES='7')
# keras <- import("keras")
# for(footprintRadius in 2:2){
#   # For each specific footprint radius, load dispersion model and training data
#   dispModelData <- read.table(paste0("/fs/home/jiluzhang/TF_grammar/scPrinter/test/luz/data/BAC/dispModelData/dispModelData", footprintRadius, "bp.txt"),sep = "\t")
#   dispersionModel <- keras$models$load_model(paste0("/fs/home/jiluzhang/TF_grammar/scPrinter/test/luz/data/shared/dispModel/dispersionModel", footprintRadius, "bp.h5"))
#   # dispersionModel <- keras::load_model_hdf5(paste0("../../data/shared/dispModel/dispersionModel", footprintRadius, "bp.h5"))
#   modelWeights <- dispersionModel$get_weights()  # modelWeights <- keras::get_weights(dispersionModel)
#   modelFeatures <- dispModelData[, c("leftFlankBias", "rightFlankBias", "centerBias", "leftTotalInsertion", "rightTotalInsertion")]
#   modelTargets <- dispModelData[, c("leftRatioMean", "leftRatioSD", "rightRatioMean", "rightRatioSD")]
  
#   # Organize everything we need for background dispersion modeling into a list
#   dispersionModel <- list(modelWeights=modelWeights, featureMean=colMeans(modelFeatures),
#                           featureSD=apply(modelFeatures, 2, sd), targetMean=colMeans(modelTargets),
#                           targetSD=apply(modelTargets, 2, sd))
  
#   saveRDS(dispersionModel, paste0("/fs/home/jiluzhang/TF_grammar/scPrinter/test/luz/data/shared/dispModel/dispersionModel", footprintRadius ,"bp.rds"))
# }

dispModel(project, 2) <- readRDS(paste0("/fs/home/jiluzhang/TF_grammar/scPrinter/test/luz/data/shared/dispModel/dispersionModel", '2' ,"bp.rds"))
dispersionModel <- project@dispModel[[2]]

# barcodeGroups <- data.frame(barcode=paste("rep", 1:5, sep=""), group=1:5)
# groups(project) <- mixedsort(unique(barcodeGroups$group))
# zcat /fs/home/jiluzhang/TF_grammar/scPrinter/test/luz/data/BAC/rawData/test.fragments.tsv.gz | head -n 10000 > test.fragments.tsv && gzip test.fragments.tsv
# pathToFrags <- paste0("./data/HepG2/test.fragments.tsv.gz")
# projectCountTensor <- countTensor(getCountTensor(project, pathToFrags, barcodeGroups, returnCombined=T, chunkSize=5000, nCores=32))  # generate chunk files

groups(project) <- as.character(groups(project))
groupCellType(project) <- c('1', '2')  # maybe 'HepG2'
cellTypeLabels <- groupCellType(project)
chunkSize = 5000

project <- getRegionBias(project, nCores=16)

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































