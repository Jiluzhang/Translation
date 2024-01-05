#remotes::install_github("pmbio/MuDataSeurat")

library(Seurat)
library(copykat)
library(MuDataSeurat)

dat <- Read10X(data.dir='CE336E1-S1')
raw <- CreateSeuratObject(counts=dat, project="CE336E1-S1", min.cells=0, min.features=0)

gene_cnt <- colSums(raw@assays$RNA$counts.Gene!=0)  # gene number stats
cells <- names(which(gene_cnt>200 & gene_cnt<10000))  # cutoff from Nature paper

print(paste0('CE336E1-S1: ', length(cells), ' cells pass QC from ', length(gene_cnt), ' cells (', round(length(cells)/length(gene_cnt)*100, 3), '%)'))
#[1] "CE336E1-S1: 3085 cells pass QC from 625129 cells (0.493%)"

rna_mat <- as.matrix(raw@assays$RNA$counts.Gene[, cells])  #raw@assays$RNA$counts.Peaks
#ncol(raw@assays$RNA$counts.Gene)  # 625129 (cell number)
#nrow(raw@assays$RNA$counts.Gene)  # 36601  (gene number)
res <- copykat(rawmat=rna_mat, id.type="S", ngene.chr=5, win.size=25, KS.cut=0.1, sam.name="CE336E1-S1/CE336E1-S1", distance="euclidean",
               norm.cell.names="", output.seg="FLASE", plot.genes="TRUE", genome="hg20", n.cores=20)

saveRDS(raw, file='CE336E1-S1/CE336E1-S1.rds')


dat['attributes']['assays']['data'][0]['attributes']['cells']['attributes']['dimnames']['data'][0]['data'][:5]
dat['attributes']['assays']['data'][0]['attributes']['features']['attributes']['dimnames']['data'][0]['data'][-5:]
dat['attributes']['assays']['data'][0]['attributes']['layers']['data'][0]['attributes']  # genes
dat['attributes']['assays']['data'][0]['attributes']['layers']['data'][1]['attributes']  # peaks

writeMM(obj=raw@assays$RNA$counts.Gene, file='matrix.mtx')


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








