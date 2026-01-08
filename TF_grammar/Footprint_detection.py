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
BiocManager::install("doParallel")
BiocManager::install("doSNOW")
BiocManager::install("cladoRcpp")
BiocManager::install("FNN")
BiocManager::install("keras")



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

tsvs = {cell:read_TF_loci(tsv_paths[cell]) for cell in tsv_paths}
peaks = {cell:read_peaks(peaks_path[cell]) for cell in peaks_path}




#### https://github.com/buenrostrolab/PRINT/blob/main/analyses/TFBSPrediction/
#### TFBSTrainingData.R (for generate TFBSDataUnibind.h5 file)
## conda activate PRINT

source('/fs/home/jiluzhang/TF_grammar/PRINT/code/utils.R')
source("/fs/home/jiluzhang/TF_grammar/PRINT/code/getFootprints.R")
source("/fs/home/jiluzhang/TF_grammar/PRINT/code/getCounts.R")
use_condaenv('PRINT')


source("../../code/getCounts.R")
source("../../code/getBias.R")
source("../../code/getFootprints.R")
source("../../code/visualization.R")
source("../../code/getTFBS.R")
source("../../code/getAggregateFootprint.R")
library(hdf5r)

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
# regionBed <- read.table(paste0(projectDataDir, "GSM6672041_HepG2_peaks.bed"))
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

dispersionModel <- dispModel(project, mode='2')


barcodeGroups <- data.frame(barcode=paste("rep", 1:5, sep=""), group=1:5)
groups(project) <- mixedsort(unique(barcodeGroups$group))
pathToFrags <- paste0("/fs/home/jiluzhang/TF_grammar/scPrinter/test/luz/data/BAC/rawData/test.fragments.tsv.gz")
projectCountTensor <- countTensor(getCountTensor(project, pathToFrags, barcodeGroups, returnCombined=T, chunkSize=2000, nCores=8))

groups(project) <- as.character(groups(project))
groupCellType(project) <- c('1', '2')  # maybe 'HepG2'
cellTypeLabels <- groupCellType(project)

footprintResults <- get_footprints(projectCountTensor=projectCountTensor, dispersionModel=dispersionModel,
                                  tmpDir=tmpDir, mode='2', footprintRadius=2, flankRadius=2,
                                  cellTypeLabels=cellTypeLabels, chunkSize=chunkSize,
                                  returnCellTypeScores=FALSE, nCores=5)
#   task 2 failed - "ℹ In argument: `group %in% groupID`.
# Caused by error in `h()`:
# ! error in evaluating the argument 'x' in selecting a method for function '%in%': object 'group' not found"

############################################################
############################################################
####################### HERE ###############################
############################################################
############################################################
############################################################


# Load footprints
footprintRadii <- c(10, 20, 30, 50, 80, 100)
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
cisBPMotifs <- readRDS(paste0(projectMainDir, "/data/shared/cisBP_human_pwms_2021.rds"))

# Load TF ChIP ranges
if(ChIPDataset == "ENCODE"){
  TFChIPRanges <- readRDS(paste0("../../data/", projectName, "/ENCODEChIPRanges.rds"))
}else if(ChIPDataset == "Unibind"){
  TFChIPRanges <- readRDS(paste0("../../data/shared/unibind/", projectName, "ChIPRanges.rds"))
}

# Only keep TFs with both motif and ChIP data
keptTFs <- intersect(names(TFChIPRanges), names(cisBPMotifs))
cisBPMotifs <- cisBPMotifs[keptTFs]
TFChIPRanges <- TFChIPRanges[keptTFs]

# Find motif matches across all regions
path <- paste0(dataDir(project), "motifPositionsList.rds")
nCores <- 16
combineTFs <- F
if(file.exists(path)){
  motifMatches <- readRDS(path)
}else{
  regions <- regionRanges(project)
  # Find motif matches across all regions
  print("Getting TF motif matches within CREs")
  motifMatches <- pbmcapply::pbmclapply(
    names(cisBPMotifs),
    function(TF){
      TFMotifPositions <- motifmatchr::matchMotifs(pwms = cisBPMotifs[[TF]], 
                                                   subject = regions, 
                                                   genome = refGenome(project),
                                                   out = "positions")[[1]]
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

TFBSData <- getTFBSTrainingData(multiScaleFootprints,
                                motifMatches,
                                TFChIPRanges,
                                regions, 
                                percentBoundThreshold = 0.1)

#####################
# Save data to file #
#####################

# Write TFBS training data to a file
h5_path <- paste0("../../data/", projectName, "/TFBSData", ChIPDataset, ".h5")
if(file.exists(h5_path)){
  system(paste0("rm ../../data/", projectName, "/TFBSData", ChIPDataset, ".h5"))
}
h5file <- H5File$new(h5_path, mode="w")
h5file[["motifFootprints"]] <- TFBSData[["motifFootprints"]]
h5file[["metadata"]] <- TFBSData[["metadata"]]
h5file$close_all()

####################################
# Visualize footprints around TFBS #
####################################

h5_path <- paste0("../../data/", projectName, "/TFBSData", ChIPDataset, ".h5")
h5file <- H5File$new(h5_path, mode="r")
motifFootprints <- h5file[["motifFootprints"]]
metadata <- h5file[["metadata"]]
motifFootprints <- motifFootprints[1:motifFootprints$dims[1],]
metadata <- metadata[1:metadata$dims]
h5file$close_all()
footprintRadii <- c(10, 20, 30, 50, 80, 100)
contextRadius <- 100

library(ComplexHeatmap)
library(BuenColors)
library(circlize)
library(RColorBrewer)

# Split heatmap columns by kernel size
colGroups <- Reduce(c, lapply(
  footprintRadii,
  function(r){
    paste(rep(r, contextRadius * 2 + 1), "bp")
  }
))

# Label each kernel size
colNames <- Reduce(c, lapply(
  footprintRadii,
  function(r){
    c(rep("", contextRadius),
      paste(" ", r, "bp"),
      rep("", contextRadius))
  }
))

TF <- "NFE2L2"
TFFilter <- metadata[, 3] %in% TF
TFFingerprint <- motifFootprints[TFFilter, ]
sampleInd <- c(sample(which(metadata[TFFilter,1] == 1), 500, replace = T),
               sample(which(metadata[TFFilter,1] == 0), 500, replace = T))
TFBinding <- c("Unbound", "Bound")[(metadata[TFFilter,1] + 1)[sampleInd]]
rowOrder <- order(TFFingerprint[sampleInd, 502], decreasing = T)
colors <- colorRamp2(seq(0, quantile(TFFingerprint, 0.95), length.out=9),
                     colors = colorRampPalette(c(rep("white", 2),  
                                                 "#9ECAE1", "#08519C", "#08306B"))(9))

Heatmap(TFFingerprint[sampleInd[rowOrder],],
        col = colors,
        cluster_rows = F,
        show_row_dend = F,
        show_column_dend = F,
        column_split = factor(colGroups, levels = paste(footprintRadii, "bp")),
        row_split = factor(TFBinding[rowOrder], levels = c("Unbound", "Bound")),
        cluster_columns = F,
        cluster_column_slices = F,
        name = paste(TF, "\nfootprint\nscore"),
        border = TRUE,
        column_title = "Kernel sizes",
        column_title_side = "bottom",
        bottom_annotation = HeatmapAnnotation(
          text = anno_text(colNames, rot = 0, location = 0.5, just = "center")
        )
)

















