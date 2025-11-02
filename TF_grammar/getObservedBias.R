######################################
# Get ground truth observed Tn5 bias #
######################################

# Load reference genome
hg38 <- BSgenome.Hsapiens.UCSC.hg38::BSgenome.Hsapiens.UCSC.hg38

# Parameters for bias estimation
localRadius <- 50
contextRadius <- 50
contextLen <- 2 * contextRadius + 1
localAverageThreshold <- 20

biasDf <- NULL
for(BACInd in 1:length(BACRanges)){
  
  print(paste0("Processing BAC No.", BACInd))
  
  observedBias <- list()
  localAverage <- list()
  ATACTracks  <- list()
  for(downSampleRate in names(counts)){
    
    BACCounts <- counts[[downSampleRate]][[BACInd]]
    BACCounts <- BACCounts %>% group_by(position) %>% summarize(insertion = sum(count))
    ATACTrack <- rep(0, width(BACRanges)[BACInd])
    ATACTrack[BACCounts$position] <- BACCounts$insertion
    
    # Calculate local average Tn5 insertion
    localAverage[[downSampleRate]] <- pbmcapply::pbmcmapply(
      function(i){
        mean(ATACTrack[max(1, i - localRadius):min(length(ATACTrack), i + localRadius)])
      },
      1:length(ATACTrack),
      mc.cores = 12
    )
    
    # Calculate observed Tn5 bias = observed insertion / local coverage
    observedBias[[downSampleRate]] <- ATACTrack / localAverage[[downSampleRate]]
    
    ATACTracks[[downSampleRate]] <- ATACTrack
  }
  
  localAverage <- Reduce(cbind, localAverage)
  observedBias <- Reduce(cbind, observedBias)
  ATACTracks <- Reduce(cbind, ATACTracks)
  colnames(localAverage) <- paste0("localAverage_", names(counts))
  colnames(observedBias) <- paste0("obsBias_", names(counts)) 
  colnames(ATACTracks) <- paste0("ATAC_", names(counts)) 
  
  # Only select positions with enough numbers of observations
  selectedPositions <- localAverage[, "localAverage_all"] > localAverageThreshold
  
  # Get sequence of the peak (inclueding left and right padding rgions)
  seqContext <- Biostrings::getSeq(hg38, as.character(seqnames(BACRanges[BACInd])), 
                                   start = start(BACRanges[BACInd]) - contextRadius, 
                                   width = width(BACRanges[BACInd]) + contextLen - 1,
                                   as.character = T)
  
  # Get sequence context for every position
  positions <- 1:width(BACRanges[BACInd])
  context <- substring(seqContext, positions, positions + contextLen - 1)
  
  # Return results
  if(sum(selectedPositions) > 0){
    BACResults <- Reduce(cbind, list(data.frame(context = context[selectedPositions]), 
                                     as.data.frame(observedBias[selectedPositions,]),
                                     as.data.frame(localAverage[selectedPositions,]),
                                     as.data.frame(ATACTracks[selectedPositions,]),
                                     data.frame(BACInd = BACInd)))
    biasDf <- rbind(biasDf, BACResults) 
  }
  
}

write.table(biasDf, 
            paste0("../../data/BAC/obsBias.tsv"), 
            quote = F, row.names = F, sep = "\t")
