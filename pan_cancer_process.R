#remotes::install_github("pmbio/MuDataSeurat")
#library(MuDataSeurat)

# dat['attributes']['assays']['data'][0]['attributes']['cells']['attributes']['dimnames']['data'][0]['data'][:5]
# dat['attributes']['assays']['data'][0]['attributes']['features']['attributes']['dimnames']['data'][0]['data'][-5:]
# dat['attributes']['assays']['data'][0]['attributes']['layers']['data'][0]['attributes']  # genes
# dat['attributes']['assays']['data'][0]['attributes']['layers']['data'][1]['attributes']  # peaks

#saveRDS(raw, file='CE336E1-S1/CE336E1-S1.rds')


library(Seurat)
library(copykat)
library(Matrix)

## read sample id
ids <- scan('files.txt', what='c')

for (i in 1:length(ids)){

        ## create folds
        system(paste0('mkdir -p ', ids[i], '_out/copykat'))
        system(paste0('mkdir -p ', ids[i], '_out/normal'))
        system(paste0('mkdir -p ', ids[i], '_out/tumor'))
        
        dat <- Read10X(data.dir=ids[i])
        rna_atac <- CreateSeuratObject(counts=dat, project=ids[i], min.cells=0, min.features=0)
        
        ## filter based on rna
        gene_cnt <- colSums(rna_atac@assays$RNA$counts.Gene!=0)  # gene number stats
        rna_cells <- names(which(gene_cnt>200 & gene_cnt<10000))  # cutoff from Nature paper
        
        ## filter based on atac
        peak_cnt <- colSums(rna_atac@assays$RNA$counts.Peaks!=0)  # gene number stats
        atac_cells <- names(which(peak_cnt>500))
        
        ## filter based on rna & atac
        cells <- intersect(rna_cells, atac_cells)
        
        print(paste0(ids[i], ': ', length(cells), ' cells pass QC from ', length(gene_cnt), ' cells (', round(length(cells)/length(gene_cnt)*100, 3), '%)'))
        
        rna_mat <- as.matrix(rna_atac@assays$RNA$counts.Gene[, cells])
        
        copykat_out <- copykat(rawmat=rna_mat, id.type="S", ngene.chr=5, win.size=25, KS.cut=0.1, sam.name=paste0(ids[i], "_out/copykat/", ids[i]), distance="euclidean",
                               norm.cell.names="", output.seg="FLASE", plot.genes="TRUE", genome="hg20", n.cores=20)
        pred <- copykat_out$prediction
        
        ## output normal cells
        normal_cells <- pred[pred$copykat.pred=='diploid', 'cell.names']
        normal_out <- rbind(rna_atac@assays$RNA$counts.Gene[, normal_cells], rna_atac@assays$RNA$counts.Peaks[, normal_cells])
        writeMM(obj=normal_out, file=paste0(ids[i], '_out/normal/matrix.mtx'))
        system(paste0('gzip ', ids[i], '_out/normal/matrix.mtx'))
        write.table(normal_cells, paste0(ids[i], '_out/normal/barcodes.tsv'), row.names=FALSE, col.names=FALSE, quote=FALSE)
        system(paste0('gzip ', ids[i], '_out/normal/barcodes.tsv'))
        system(paste0('cp ', ids[i], '/features.tsv.gz ', ids[i], '_out/normal'))
        print(paste0(ids[i], ': ', length(normal_cells), ' cells from ', length(gene_cnt), ' cells (', round(length(normal_cells)/length(gene_cnt)*100, 3), '%)'))
        
        ## output tumor cells
        tumor_cells <- pred[pred$copykat.pred=='diploid', 'cell.names']
        tumor_out <- rbind(rna_atac@assays$RNA$counts.Gene[, tumor_cells], rna_atac@assays$RNA$counts.Peaks[, tumor_cells])
        writeMM(obj=tumor_out, file=paste0(ids[i], '_out/tumor/matrix.mtx'))
        system(paste0('gzip ', ids[i], '_out/tumor/matrix.mtx'))
        write.table(tumor_cells, paste0(ids[i], '_out/tumor/barcodes.tsv'), row.names=FALSE, col.names=FALSE, quote=FALSE)
        system(paste0('gzip ', ids[i], '_out/tumor/barcodes.tsv'))
        system(paste0('cp ', ids[i], '/features.tsv.gz ', ids[i], '_out/tumor'))
        print(paste0(ids[i], ': ', length(tumor_cells), ' cells from ', length(gene_cnt), ' cells (', round(length(tumor_cells)/length(gene_cnt)*100, 3), '%)'))
        print(paste0(ids[i], ' done'))
}


#ncol(raw@assays$RNA$counts.Gene)  # 625129 (cell number)
#nrow(raw@assays$RNA$counts.Gene)  # 36601  (gene number)

# ngene.chr: at least 5 genes in each chromosome
# win.size: at least 25 genes per segment (15-150)
# KS.cut: segmentation parameter (0-1, 0.05-0.15 usually)
# distance: parameter for clustering (correlational distances tend to favor noisy data, while euclidean distance tends to favor data with larger CN segments)

# [1] "running copykat v1.1.0"
# [1] "step1: read and filter data ..."
# [1] "36601 genes, 3085 cells in raw data"
# [1] "11213 genes past LOW.DR filtering"
# [1] "step 2: annotations gene coordinates ..."
# [1] "start annotation ..."
# [1] "step 3: smoothing data with dlm ..."
# [1] "step 4: measuring baselines ..."
# number of iterations= 446 
# number of iterations= 286 
# number of iterations= 249 
# number of iterations= 405 
# number of iterations= 756 
# number of iterations= 215 
# [1] "step 5: segmentation..."
# [1] "step 6: convert to genomic bins..."
# [1] "step 7: adjust baseline ..."
# [1] "step 8: final prediction ..."
# [1] "step 9: saving results..."
# [1] "step 10: ploting heatmap ..."
# Time difference of 7.710566 mins








