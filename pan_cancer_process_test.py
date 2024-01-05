####################################### Pan-cancer multiomic data ##########################################
## https://data.humantumoratlas.org/explore?selectedFilters=%5B%7B%22group%22%3A%22AtlasName%22%2C%22value%22%3A%22HTAN+WUSTL%22%7D%5D&tab=file

#Atlas: HTAN WUSTL
#Level: Level 3
#Assay: scATAC-seq
#File: Formatgzip

# pip install synapseclient

# python download_data_from_HTAN.py

import synapseclient
import pandas as pd
import os

syn = synapseclient.Synapse()
syn.login('Luz','57888282')

ids = pd.read_csv('synapse_id.txt', header=None)
cnt = 0
for id in ids[0].values:
    meta = syn.get(entity=id, downloadFile=False)
    if meta.name.split('-')[-1] in ['barcodes.tsv.gz', 'features.tsv.gz', 'matrix.mtx.gz']:
        dir_name = '-'.join(meta.name.split('-')[:-1])
        if os.path.exists(dir_name):
            file = syn.get(entity=id, downloadLocation=dir_name)
        else:
            os.mkdir(dir_name)
            file = syn.get(entity=id, downloadLocation=dir_name)
        os.rename(dir_name+'/'+meta.name, dir_name+'/'+(meta.name.split('-')[-1]))
        cnt += 1
        print('file '+str(cnt)+' '+meta.name+' done')
    else:
        cnt += 1
        print('file '+str(cnt)+' '+meta.name+' skip')


## cell count stats for each sample
#for sample in `cat files.txt`; do
#    echo $sample `zcat $sample/barcodes.tsv.gz | wc -l` >> files_cell_cnt.txt
#done


## paper associated github: https://github.com/ding-lab/PanCan_snATAC_publication/tree/main


#conda create -n copykat
#conda install -c conda-forge r-devtools

#library(devtools)
#install_github("navinlabcode/copykat")
#install.packages('scales')
#install.packages('ggplot2')
#install.packages('plotly')
#install.packages('mixtools')
#install_github("navinlabcode/copykat")

#install.packages('Seurat')
#conda install r-polyclip
#install.packages('spatstat')
#install.packages('Seurat')


library(Seurat)
library(copykat)
dat <- Read10X(data.dir='CE336E1-S1')
raw <- CreateSeuratObject(counts=dat, project="CE336E1-S1", min.cells=0, min.features=0)

gene_cnt <- colSums(raw@assays$RNA$counts.Gene!=0)  # gene number stats

rna_mat <- as.matrix(raw@assays$RNA$counts.Gene[, names(which(gene_cnt>200 & gene_cnt<10000))])  #raw@assays$RNA$counts.Peaks
#ncol(raw@assays$RNA$counts.Gene)  # 625129 (cell number)
#nrow(raw@assays$RNA$counts.Gene)  # 36601  (gene number)
res <- copykat(rawmat=rna_mat, id.type="S", ngene.chr=5, win.size=25, KS.cut=0.1, sam.name="CE336E1-S1", distance="euclidean",
               norm.cell.names="", output.seg="FLASE", plot.genes="TRUE", genome="hg20", n.cores=10)
# ngene.chr: at least 5 genes in each chromosome
# win.size: at least 25 genes per segment (15-150)
# KS.cut: segmentation parameter (0-1, 0.05-0.15 usually)
# distance: parameter for clustering (correlational distances tend to favor noisy data, while euclidean distance tends to favor data with larger CN segments)

# 3000 cells ~ 10 mins !!!



library(Seurat)
library(copykat)

dat <- Read10X(data.dir='CE336E1-S1')
raw <- CreateSeuratObject(counts=dat, project="CE336E1-S1", min.cells=0, min.features=0)

gene_cnt <- colSums(raw@assays$RNA$counts.Gene!=0)  # gene number stats
cells <- names(which(gene_cnt>200 & gene_cnt<10000))

print(paste0('CE336E1-S1: ', length(cells), ' cells pass QC from ', length(gene_cnt), ' cells (', round(length(cells)/length(gene_cnt)*100, 3), '%)'))
#[1] "CE336E1-S1: 3085 cells pass QC from 625129 cells (0.493%)"

rna_mat <- as.matrix(raw@assays$RNA$counts.Gene[, cells])  #raw@assays$RNA$counts.Peaks
#ncol(raw@assays$RNA$counts.Gene)  # 625129 (cell number)
#nrow(raw@assays$RNA$counts.Gene)  # 36601  (gene number)
res <- copykat(rawmat=rna_mat, id.type="S", ngene.chr=5, win.size=25, KS.cut=0.1, sam.name="CE336E1-S1", distance="euclidean",
               norm.cell.names="", output.seg="FLASE", plot.genes="TRUE", genome="hg20", n.cores=10)
# ngene.chr: at least 5 genes in each chromosome
# win.size: at least 25 genes per segment (15-150)
# KS.cut: segmentation parameter (0-1, 0.05-0.15 usually)
# distance: parameter for clustering (correlational distances tend to favor noisy data, while euclidean distance tends to favor data with larger CN segments)

# 3000 cells ~ 10 mins !!!



# python filter_cells.py
import scanpy as sc
import pandas as pd
import h5py
import numpy as np

dat = sc.read_10x_mtx('/fs/home/jiluzhang/2023_nature_LD/CE336E1-S1', gex_only=False)
cell_anno = pd.read_table('/fs/home/jiluzhang/2023_nature_LD/CE336E1-S1_copykat_prediction.txt')
normal = dat[cell_anno[cell_anno['copykat.pred']=='diploid']['cell.names'], :]    # 1007 × 161569
tumor = dat[cell_anno[cell_anno['copykat.pred']=='aneuploid']['cell.names'], :]   # 1791 × 161569

## normal & tumor
set_1 = sc.AnnData.concatenate(normal[:500, :], tumor[:500, :])
out  = h5py.File('CEAD_normal_tumor_1000.h5', 'w')
g = out.create_group('matrix')
g.create_dataset('barcodes', data=set_1.obs.index.values)
g.create_dataset('data', data=set_1.X.data)
g_2 = g.create_group('features')
g_2.create_dataset('_all_tag_keys', data=np.array([b'genome', b'interval']))
g_2.create_dataset('feature_type', data=np.append([b'Gene Expression']*(sum(set_1.var['feature_types']=='Gene Expression')),
                                                  [b'Peaks']*(sum(set_1.var['feature_types']=='Peaks'))))
g_2.create_dataset('genome', data=np.array([b'GRCh38'] * (set_1.n_vars)))
g_2.create_dataset('id', data=set_1.var['gene_ids'].values)
g_2.create_dataset('interval', data=set_1.var['gene_ids'].values)
g_2.create_dataset('name', data=set_1.var.index.values)      
g.create_dataset('indices', data=set_1.X.indices)
g.create_dataset('indptr',  data=set_1.X.indptr)
l = list(set_1.X.shape)
l.reverse()
g.create_dataset('shape', data=l)
out.close()

## normal & normal
set_2 = normal[:1000, :]
out  = h5py.File('CEAD_normal_normal_1000.h5', 'w')
g = out.create_group('matrix')
g.create_dataset('barcodes', data=set_2.obs.index.values)
g.create_dataset('data', data=set_2.X.data)
g_2 = g.create_group('features')
g_2.create_dataset('_all_tag_keys', data=np.array([b'genome', b'interval']))
g_2.create_dataset('feature_type', data=np.append([b'Gene Expression']*(sum(set_2.var['feature_types']=='Gene Expression')),
                                                  [b'Peaks']*(sum(set_2.var['feature_types']=='Peaks'))))
g_2.create_dataset('genome', data=np.array([b'GRCh38'] * (set_2.n_vars)))
g_2.create_dataset('id', data=set_2.var['gene_ids'].values)
g_2.create_dataset('interval', data=set_2.var['gene_ids'].values)
g_2.create_dataset('name', data=set_2.var.index.values)      
g.create_dataset('indices', data=set_2.X.indices)
g.create_dataset('indptr',  data=set_2.X.indptr)
l = list(set_2.X.shape)
l.reverse()
g.create_dataset('shape', data=l)
out.close()


## python normal_tumor_h5_to_h5ad.py
import scanpy as sc
import pandas as pd
import numpy as np
import pybedtools
from tqdm import tqdm

dat = sc.read_10x_h5('CEAD_normal_tumor_1000.h5', gex_only=False)

########## rna ##########
rna = dat[:, dat.var['feature_types']=='Gene Expression'].copy()
rna.X = np.array(rna.X.todense()) 
genes = pd.read_table('human_genes.txt', names=['gene_id', 'gene_name'])

rna_genes = rna.var.copy()
rna_genes['gene_id'] = rna_genes['gene_ids'].map(lambda x: x.split('.')[0])  # delete ensembl id with version number
rna_exp = pd.concat([rna_genes, pd.DataFrame(rna.X.T, index=rna_genes.index)], axis=1)

## fit for the case of no ensembl id
if rna_exp['gene_id'][0][:4]=='ENSG':
    X_new = pd.merge(genes, rna_exp, how='left', on='gene_id').iloc[:, 5:].T
else:
    X_new = pd.merge(genes, rna_exp, how='left', left_on='gene_name', right_on='gene_id').iloc[:, 6:].T
X_new = np.array(X_new, dtype='float32')
X_new[np.isnan(X_new)] = 0

rna_new_var = pd.DataFrame({'gene_ids': genes['gene_id'], 'feature_types': 'Gene Expression', 'my_Id': range(genes.shape[0])})
rna_new = sc.AnnData(X_new, obs=rna.obs, var=rna_new_var)  

rna_new.var.index = genes['gene_name']  # set index for var

rna_new.write('CEAD_normal_tumor_rna_1000.h5ad')


########## atac ##########
atac = dat[:, dat.var['feature_types']=='Peaks'].copy()
atac.X = np.array(atac.X.todense())
atac.X[atac.X>1] = 1
atac.X[atac.X<1] = 0

peaks = pd.DataFrame({'id': atac.var_names})
peaks['chr'] = peaks['id'].map(lambda x: x.split(':')[0])
peaks['start'] = peaks['id'].map(lambda x: x.split(':')[1].split('-')[0])
peaks['end'] = peaks['id'].map(lambda x: x.split(':')[1].split('-')[1])
peaks.drop(columns='id', inplace=True)
peaks['idx'] = range(peaks.shape[0])

cCREs = pd.read_table('human_cCREs.bed', names=['chr', 'start', 'end'])
cCREs['idx'] = range(cCREs.shape[0])
cCREs_bed = pybedtools.BedTool.from_dataframe(cCREs)
peaks_bed = pybedtools.BedTool.from_dataframe(peaks)
idx_map = peaks_bed.intersect(cCREs_bed, wa=True, wb=True).to_dataframe().iloc[:, [3, 7]]
idx_map.columns = ['peaks_idx', 'cCREs_idx']

m = np.zeros([atac.n_obs, cCREs_bed.to_dataframe().shape[0]], dtype='float32')
for i in tqdm(range(atac.X.shape[0])):
    m[i][idx_map[idx_map['peaks_idx'].isin(peaks.iloc[np.nonzero(atac.X[i])]['idx'])]['cCREs_idx']] = 1

atac_new_var = pd.DataFrame({'gene_ids': cCREs['chr'] + ':' + cCREs['start'].map(str) + '-' + cCREs['end'].map(str), 'feature_types': 'Peaks', 'my_Id': range(cCREs.shape[0])})
atac_new = sc.AnnData(m, obs=atac.obs, var=atac_new_var)

atac_new.var.index = atac_new.var['gene_ids']  # set index

atac_new.write('CEAD_normal_tumor_atac_1000.h5ad')


## python normal_normal_h5_to_h5ad.py
import scanpy as sc
import pandas as pd
import numpy as np
import pybedtools
from tqdm import tqdm

dat = sc.read_10x_h5('CEAD_normal_normal_1000.h5', gex_only=False)

########## rna ##########
rna = dat[:, dat.var['feature_types']=='Gene Expression'].copy()
rna.X = np.array(rna.X.todense()) 
genes = pd.read_table('human_genes.txt', names=['gene_id', 'gene_name'])

rna_genes = rna.var.copy()
rna_genes['gene_id'] = rna_genes['gene_ids'].map(lambda x: x.split('.')[0])  # delete ensembl id with version number
rna_exp = pd.concat([rna_genes, pd.DataFrame(rna.X.T, index=rna_genes.index)], axis=1)

## fit for the case of no ensembl id
if rna_exp['gene_id'][0][:4]=='ENSG':
    X_new = pd.merge(genes, rna_exp, how='left', on='gene_id').iloc[:, 5:].T
else:
    X_new = pd.merge(genes, rna_exp, how='left', left_on='gene_name', right_on='gene_id').iloc[:, 6:].T
X_new = np.array(X_new, dtype='float32')
X_new[np.isnan(X_new)] = 0

rna_new_var = pd.DataFrame({'gene_ids': genes['gene_id'], 'feature_types': 'Gene Expression', 'my_Id': range(genes.shape[0])})
rna_new = sc.AnnData(X_new, obs=rna.obs, var=rna_new_var)  

rna_new.var.index = genes['gene_name']  # set index for var

rna_new.write('CEAD_normal_normal_rna_1000.h5ad')


########## atac ##########
atac = dat[:, dat.var['feature_types']=='Peaks'].copy()
atac.X = np.array(atac.X.todense())
atac.X[atac.X>1] = 1
atac.X[atac.X<1] = 0

peaks = pd.DataFrame({'id': atac.var_names})
peaks['chr'] = peaks['id'].map(lambda x: x.split(':')[0])
peaks['start'] = peaks['id'].map(lambda x: x.split(':')[1].split('-')[0])
peaks['end'] = peaks['id'].map(lambda x: x.split(':')[1].split('-')[1])
peaks.drop(columns='id', inplace=True)
peaks['idx'] = range(peaks.shape[0])

cCREs = pd.read_table('human_cCREs.bed', names=['chr', 'start', 'end'])
cCREs['idx'] = range(cCREs.shape[0])
cCREs_bed = pybedtools.BedTool.from_dataframe(cCREs)
peaks_bed = pybedtools.BedTool.from_dataframe(peaks)
idx_map = peaks_bed.intersect(cCREs_bed, wa=True, wb=True).to_dataframe().iloc[:, [3, 7]]
idx_map.columns = ['peaks_idx', 'cCREs_idx']

m = np.zeros([atac.n_obs, cCREs_bed.to_dataframe().shape[0]], dtype='float32')
for i in tqdm(range(atac.X.shape[0])):
    m[i][idx_map[idx_map['peaks_idx'].isin(peaks.iloc[np.nonzero(atac.X[i])]['idx'])]['cCREs_idx']] = 1

atac_new_var = pd.DataFrame({'gene_ids': cCREs['chr'] + ':' + cCREs['start'].map(str) + '-' + cCREs['end'].map(str), 'feature_types': 'Peaks', 'my_Id': range(cCREs.shape[0])})
atac_new = sc.AnnData(m, obs=atac.obs, var=atac_new_var)

atac_new.var.index = atac_new.var['gene_ids']  # set index

atac_new.write('CEAD_normal_normal_atac_1000.h5ad')







