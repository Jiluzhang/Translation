## mouse kidney
## https://cf.10xgenomics.com/samples/cell-arc/2.0.2/M_Kidney_Chromium_Nuc_Isolation_vs_SaltyEZ_vs_ComplexTissueDP/\
##                            M_Kidney_Chromium_Nuc_Isolation_vs_SaltyEZ_vs_ComplexTissueDP_filtered_feature_bc_matrix.h5
## kidney.h5

import scanpy as sc

dat = sc.read_10x_h5('kidney.h5', gex_only=False)                  # 14652 × 97637
dat.var_names_make_unique()

rna = dat[:, dat.var['feature_types']=='Gene Expression'].copy()   # 14652 × 32285
rna.write('rna.h5ad')

atac = dat[:, dat.var['feature_types']=='Peaks'].copy()            # 14652 × 65352
atac.X[atac.X!=0] = 1
atac.write('atac.h5ad')


python split_train_val.py --RNA rna.h5ad --ATAC atac.h5ad --train_pct 0.9
python data_preprocess.py -r rna_train.h5ad -a atac_train.h5ad -s preprocessed_data_train --dt train --config rna2atac_config_train.yaml
python data_preprocess.py -r rna_val.h5ad -a atac_val.h5ad -s preprocessed_data_val --dt val --config rna2atac_config_val.yaml
nohup accelerate launch --config_file accelerator_config_train.yaml --main_process_port 29823 rna2atac_train.py --config_file rna2atac_config_train.yaml \
                        --train_data_dir ./preprocessed_data_train --val_data_dir ./preprocessed_data_val -n rna2atac_train > 20240821.log &   
# accuracy improves until about 10 epoch
# 3199536


import scanpy as sc
import numpy as np
rna = sc.read_h5ad('rna_val.h5ad')
np.count_nonzero(rna.X.toarray(), axis=1).max()  # 13845

import snapatac2 as snap
import scanpy as sc
atac = sc.read_h5ad('atac_val.h5ad')
snap.pp.select_features(atac)
snap.tl.spectral(atac)
snap.tl.umap(atac)
snap.pp.knn(atac)
snap.tl.leiden(atac)
sc.pl.umap(atac, color='leiden', legend_fontsize='7', legend_loc='right margin', size=5,
           title='', frameon=True, save='_atac_val_true.pdf')
atac.write('atac_val_with_leiden.h5ad')


python data_preprocess.py -r rna_val.h5ad -a atac_val.h5ad -s preprocessed_data_test --dt test --config rna2atac_config_test.yaml
accelerate launch --config_file accelerator_config_test.yaml --main_process_port 29822 rna2atac_test.py \
                  -d ./preprocessed_data_test \
                  -l save/2024-08-21_rna2atac_train_56/pytorch_model.bin --config_file rna2atac_config_test.yaml
python npy2h5ad.py
python csr2array.py --pred rna2atac_scm2m_raw.h5ad --true atac_val_with_leiden.h5ad
python cal_auroc_auprc.py --pred rna2atac_scm2m.h5ad --true atac_val_with_leiden.h5ad
python cal_cluster_plot.py --pred rna2atac_scm2m.h5ad --true atac_val_with_leiden.h5ad


## Tabula Muris Senis: https://cellxgene.cziscience.com/collections/0b9d8a04-bb9d-44da-aa27-705bb65b54eb
# kidney dataset: wget -c https://datasets.cellxgene.cziscience.com/e896f2b3-e22b-42c0-81e8-9dd1b6e80d64.h5ad
# gene expression: https://cellxgene.cziscience.com/gene-expression
# -> kidney_aging.h5ad

import scanpy as sc
import pandas as pd
from scipy.sparse import csr_matrix
import numpy as np

rna = sc.read_h5ad('kidney_aging.h5ad')

## map rna
ref_rna = sc.read_h5ad('rna.h5ad')
genes = ref_rna.var[['gene_ids']]
genes['gene_name'] = genes.index.values
genes.reset_index(drop=True, inplace=True)

rna_exp = pd.concat([rna.var['feature_name'], pd.DataFrame(rna.raw.X.toarray().T, index=rna.var.index.values)], axis=1)
rna_exp['gene_ids'] = rna_exp.index.values
rna_exp.rename(columns={'feature_name': 'gene_name'}, inplace=True)
X_new = pd.merge(genes, rna_exp, how='left', on='gene_ids').iloc[:, 3:].T
X_new.fillna(value=0, inplace=True)

rna_new = sc.AnnData(X_new.values, obs=rna.obs, var=pd.DataFrame({'gene_ids': genes['gene_ids'], 'feature_types': 'Gene Expression'}))
rna_new.var.index = genes['gene_name'].values
rna_new.X = csr_matrix(rna_new.X)
rna_new.write('rna_unpaired.h5ad')

## pseudo atac
ref_atac = sc.read_h5ad('atac.h5ad')

atac_new = sc.AnnData(np.zeros([rna.n_obs, ref_atac.n_vars]),
                      obs=rna.obs, 
                      var=ref_atac.var)
atac_new.X[:, 0] = 1
atac_new.X = csr_matrix(atac_new.X)
atac_new.write('atac_unpaired.h5ad')


import scanpy as sc
import numpy as np
rna = sc.read_h5ad('rna_unpaired.h5ad')
np.count_nonzero(rna.X.toarray(), axis=1).max()  # 7617



python data_preprocess_unpaired.py -r rna_unpaired.h5ad -a atac_unpaired.h5ad -s preprocessed_data_unpaired --dt test --config rna2atac_config_unpaired.yaml
accelerate launch --config_file accelerator_config_test.yaml --main_process_port 29822 rna2atac_test.py \
                  -d ./preprocessed_data_unpaired \
                  -l save/2024-08-21_rna2atac_train_55/pytorch_model.bin --config_file rna2atac_config_unpaired.yaml   # ~ 0.5h

import scanpy as sc
import numpy as np
from scipy.sparse import csr_matrix
import snapatac2 as snap
import pandas as pd
from tqdm import tqdm

dat = np.load('predict_unpaired.npy')
dat[dat>0.5] = 1
dat[dat<=0.5] = 0
atac = sc.read_h5ad('atac_unpaired.h5ad')
atac.X = csr_matrix(dat)

snap.pp.select_features(atac)
snap.tl.spectral(atac, n_comps=100)  # snap.tl.spectral(atac, n_comps=100, weighted_by_sd=False)
snap.tl.umap(atac)

# sc.pl.umap(atac[atac.obs['cell_type'].isin(atac.obs.cell_type.value_counts()[:5].index), :], color='cell_type', legend_fontsize='7', legend_loc='right margin', size=10,
#            title='', frameon=True, save='_unpaired_tmp.pdf')

sc.pl.umap(atac[atac.obs['cell_type'].isin(['fenestrated cell', 'B cell']), :], color='cell_type', legend_fontsize='7', legend_loc='right margin', size=10,
            title='', frameon=True, save='_unpaired_luz.pdf')

sc.pl.umap(atac, color='cell_type', legend_fontsize='7', legend_loc='right margin', size=10,
           title='', frameon=True, save='_unpaired_tmp.pdf')

sc.pl.umap(atac, color='cell_type', legend_fontsize='7', legend_loc='right margin', size=10,
           title='', frameon=True, save='_unpaired.pdf')

sc.pl.umap(atac, color='cell_type', legend_fontsize='7', legend_loc='right margin', size=10,
           title='', frameon=True, save='_unpaired_n_comps_200.pdf')


def out_pred_bedgraph(cell_type='T cell'):
    atac_X = atac[atac.obs['cell_type']==cell_type, :].X.toarray().mean(axis=0)
    df = pd.DataFrame({'chr': atac.var['gene_ids'].map(lambda x: x.split(':')[0]).values,
                       'start': atac.var['gene_ids'].map(lambda x: x.split(':')[1].split('-')[0]).values,
                       'end': atac.var['gene_ids'].map(lambda x: x.split(':')[1].split('-')[1]).values,
                       'val': atac_X})
    df.to_csv(cell_type.replace(' ', '_')+'_atac_pred.bedgraph', index=False, header=False, sep='\t')

for cell_type in set(atac.obs.cell_type.values):
    out_pred_bedgraph(cell_type=cell_type)


def out_pred_norm_bedgraph(cell_type='T cell'):
    # atac_X_raw = atac[atac.obs['cell_type']==cell_type, :].X.toarray()
    # atac_X = (atac_X_raw/(atac_X_raw.sum(axis=1)[:, np.newaxis])).mean(axis=0)*10000
    atac_X_raw = atac[atac.obs['cell_type']==cell_type, :].X.toarray().sum(axis=0)
    atac_X = atac_X_raw/atac_X_raw.sum()*10000
    df = pd.DataFrame({'chr': atac.var['gene_ids'].map(lambda x: x.split(':')[0]).values,
                       'start': atac.var['gene_ids'].map(lambda x: x.split(':')[1].split('-')[0]).values,
                       'end': atac.var['gene_ids'].map(lambda x: x.split(':')[1].split('-')[1]).values,
                       'val': atac_X})
    df.to_csv(cell_type.replace(' ', '_')+'_atac_pred_norm.bedgraph', index=False, header=False, sep='\t')

for cell_type in tqdm(set(atac.obs.cell_type.values), ncols=80):
    out_pred_norm_bedgraph(cell_type=cell_type)


cell_type = 'kidney proximal convoluted tubule epithelial cell'
m_01 = atac[(atac.obs['cell_type']==cell_type) & (atac.obs['development_stage']=='4 weeks')].X.toarray()  # 938
m_01 = (m_01.sum(axis=0))/(m_01.shape[0])
m_03 = atac[(atac.obs['cell_type']==cell_type) & (atac.obs['development_stage']=='3 month-old stage')].X.toarray()  # 681
m_03 = (m_03.sum(axis=0))/(m_03.shape[0])
m_18 = atac[(atac.obs['cell_type']==cell_type) & (atac.obs['development_stage']=='18 month-old stage')].X.toarray() # 1117
m_18 = (m_18.sum(axis=0))/(m_18.shape[0])
m_20 = atac[(atac.obs['cell_type']==cell_type) & (atac.obs['development_stage']=='20 month-old stage and over')].X.toarray()  # 1724
m_20 = (m_20.sum(axis=0))/(m_20.shape[0])
m_all = np.vstack((m_01, m_03, m_18, m_20))


import statsmodels.api as sm
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests

np.argwhere((m_all[1, :]>m_all[0, :]) & (m_all[2, :]>m_all[1, :]) & (m_all[3, :]>m_all[2, :])).flatten()[:20]
# array([  7,  36,  65,  75,  94, 234, 238, 241, 304, 324, 356, 364, 400, 424, 428, 494, 517, 549, 623, 630])

np.argwhere((m_all[1, :]<m_all[0, :]) & (m_all[2, :]<m_all[1, :]) & (m_all[3, :]<m_all[2, :])).flatten()[:20]
# array([ 132,  142,  461,  548,  660,  695,  773,  817,  846,  911,  914, 915,  917,  958, 1059, 1144, 1807, 1841, 1857, 1917])


coef_lst = []
pval_lst = []
X = [1, 3, 18, 20]
X = sm.add_constant(X)
for i in tqdm(range(m_all.shape[1]), ncols=80):
    Y = m_all[:, i]
    model = sm.OLS(Y, X)
    results = model.fit()
    beta_hat = results.params[1]
    se_beta_hat = results.bse[1]
    wald_statistic = beta_hat / se_beta_hat
    p_value = 2 * (1 - norm.cdf(abs(wald_statistic)))
    coef_lst.append(beta_hat)
    pval_lst.append(p_value)

df = pd.DataFrame({'idx':atac.var.index.values, 'coef':coef_lst, 'pval':pval_lst})
df.dropna(inplace=True)
_, fdr, _, _ = multipletests(df['pval'].values, alpha=0.05, method='fdr_bh')
df['fdr'] = fdr

df_pos = df[df['fdr']<0.01]
out_1 = df_pos.sort_values('coef', ascending=False)[:20]
out_2 = df_pos.sort_values('coef', ascending=True)[:20]
pd.concat([out_2, out_1]).to_csv('kidney_proximal_convoluted_tubule_epithelial_cell_atac_test.txt', sep='\t', index=None)


## plot bar
import pandas as pd
from plotnine import *
import numpy as np

df = pd.read_table('kidney_proximal_convoluted_tubule_epithelial_cell_atac_test.txt')
df = df.sort_values('coef', ascending=True)
df['idx'] = pd.Categorical(df['idx'], categories=df['idx'])

p = ggplot(df, aes(x='idx', y='coef')) + geom_bar(stat='identity', fill='darkred', position=position_dodge()) + coord_flip() + \
                                                        xlab('Peaks') + ylab('Coef') + labs(title='Kidney proximal convoluted tubule epithelial cells')  + \
                                                        scale_y_continuous(limits=[-0.015, 0.015], breaks=np.arange(-0.015, 0.015+0.0001, 0.003)) + theme_bw() + theme(plot_title=element_text(hjust=0.5))
p.save(filename='kidney_proximal_convoluted_tubule_epithelial_cell_atac_test.pdf', dpi=600, height=8, width=15)




# kidney proximal convoluted tubule epithelial cell            4460
# B cell                                                       3124
# epithelial cell of proximal tubule                           3053
# kidney loop of Henle thick ascending limb epithelial cell    1554
# lymphocyte                                                   1536
# macrophage                                                   1407
# T cell                                                       1359
# fenestrated cell                                             1027
# kidney collecting duct principal cell                         789
# kidney distal convoluted tubule epithelial cell               744
# plasmatocyte                                                  496
# brush cell                                                    414
# kidney cortex artery cell                                     381
# plasma cell                                                   325
# mesangial cell                                                261
# kidney loop of Henle ascending limb epithelial cell           201
# kidney capillary endothelial cell                             161
# fibroblast                                                    161
# kidney proximal straight tubule epithelial cell                95
# natural killer cell                                            66
# kidney collecting duct epithelial cell                         25
# leukocyte                                                       5
# kidney cell                                                     3


# marker genes: https://drive.google.com/drive/u/0/folders/1JZgFDmdVlw-UN8Gb_lRxSAdCZfhlzdrH
# Single cell regulatory landscape of the mouse kidney highlights cellular differentiation programs and disease targets
# Differentail Expression: https://cellxgene.cziscience.com/differential-expression

## prob average seems almost same!!!
# dat = np.load('predict_unpaired.npy')
# atac = sc.read_h5ad('atac_unpaired.h5ad')
# atac.X = dat

# def out_pred_prob_bedgraph(cell_type='T cell'):
#     atac_X = atac[atac.obs['cell_type']==cell_type, :].X.mean(axis=0)
#     df = pd.DataFrame({'chr': atac.var['gene_ids'].map(lambda x: x.split(':')[0]).values,
#                        'start': atac.var['gene_ids'].map(lambda x: x.split(':')[1].split('-')[0]).values,
#                        'end': atac.var['gene_ids'].map(lambda x: x.split(':')[1].split('-')[1]).values,
#                        'val': atac_X})
#     df.to_csv(cell_type.replace(' ', '_')+'_atac_pred_prob.bedgraph', index=False, header=False, sep='\t')

# for cell_type in set(atac.obs.cell_type.values):
#     out_pred_prob_bedgraph(cell_type=cell_type)



###### EpiTrace - Estimating cell age from single-cell ATAC data
# https://epitrace.readthedocs.io/en/latest/
# https://epitrace.readthedocs.io/en/latest/MSscATAC.html
# EpiTrace application (Genome Biology): https://link.springer.com/article/10.1186/s13059-024-03265-z#article-info

################################ INSTALLATION ########################################
conda create -n epitrace conda-forge::r-base=4.1.3

install.packages('pak')
library(pak)
install.packages('https://cran.r-project.org/src/contrib/Archive/rjson/rjson_0.2.20.tar.gz')
install.packages('lattice')
install.packages('https://cran.r-project.org/src/contrib/Archive/Matrix/Matrix_1.5-0.tar.gz')
pak::pkg_install('caleblareau/easyLift')

conda install r-xml
Sys.setenv(XML_CONFIG="/data/home/jiluzhang/miniconda3/bin/xml2-config")
install.packages("XML")

conda create -n epitrace conda-forge::r-base=4.2.0
install.packages('pak')
library(pak)
install.packages('https://cran.r-project.org/src/contrib/Archive/rjson/rjson_0.2.20.tar.gz')
install.packages('lattice')
install.packages('https://cran.r-project.org/src/contrib/Archive/Matrix/Matrix_1.5-0.tar.gz')

pak::pkg_install('caleblareau/easyLift')
install.packages('https://cran.r-project.org/src/contrib/Archive/Matrix/Matrix_1.6-4.tar.gz')
install.packages('https://cran.r-project.org/src/contrib/Archive/MASS/MASS_7.3-55.tar.gz')
install.packages('testthat')

wget -c https://github.com/Kitware/CMake/releases/download/v3.30.3/cmake-3.30.3.tar.gz
tar -zvxf cmake-3.30.3.tar.gz
cd cmake-3.30.3
./bootstrap
./configure --prefix=/data/home/jiluzhang/biosoftware/cmake
make
make install
add env

install.packages('https://cran.r-project.org/src/contrib/Archive/nloptr/nloptr_2.0.0.tar.gz')
install.packages('https://www.bioconductor.org/packages/release/bioc/src/contrib/ggtree_3.12.0.tar.gz')

pak::pkg_install('MagpiePKU/EpiTrace')

install.packages('readr')
install.packages('openxlsx')

conda install conda-forge::r-devtools
ln -s libicuuc.so.70 libicuuc.so.58 (workdir:/data/home/jiluzhang/miniconda3/envs/epitrace/lib)
ln -s libicui18n.so.70 libicui18n.so.58
install.packages("https://cran.r-project.org/src/contrib/stringi_1.8.4.tar.gz")

install.packages("BiocManager")
devtools::install_github("GreenleafLab/ArchR", ref="master", repos = BiocManager::repositories())
install.packages('https://cran.r-project.org/src/contrib/Rcpp_1.0.13.tar.gz')

BiocManager::install("ChIPseeker")
###################################################################################################

wget -c https://github.com/MagpiePKU/EpiTrace/blob/master/data/mouse_clock_mm285_design_clock347_mm10.rds

###### transform h5ad to 10x
import numpy as np
import scanpy as sc
import scipy.io as sio
import pandas as pd
from scipy.sparse import csr_matrix

dat = np.load('../predict_unpaired.npy')
dat[dat>0.5] = 1
dat[dat<=0.5] = 0
atac = sc.read_h5ad('../atac_unpaired.h5ad')
atac.X = csr_matrix(dat)

## output peaks
peaks_df = pd.DataFrame({'chr': atac.var['gene_ids'].map(lambda x: x.split(':')[0]).values,
                         'start': atac.var['gene_ids'].map(lambda x: x.split(':')[1].split('-')[0]).values,
                         'end': atac.var['gene_ids'].map(lambda x: x.split(':')[1].split('-')[1]).values})
peaks_df.to_csv('scATAC_peaks.bed', sep='\t', header=False, index=False)

## output barcodes
pd.DataFrame(atac.obs.index).to_csv('scATAC_barcodes.tsv', sep='\t', header=False, index=False)

## output matrix
sio.mmwrite("scATAC_matrix.mtx", atac.X.T)  # peak*cell (not cell*peak)

## output meta
pd.DataFrame(atac.obs[['age', 'cell_type']]).to_csv('scATAC_meta.tsv', sep='\t', header=False, index=True)








library(readr)
library(EpiTrace)
library(dplyr)
library(GenomicRanges)
library(ggplot2)
library(Seurat)


## load the  “mouse clocks”
# https://github.com/MagpiePKU/EpiTrace/blob/master/data/mouse_clock_mm285_design_clock347_mm10.rds
mouse_clock_by_MM285 <- readRDS('mouse_clock_mm285_design_clock347_mm10.rds')

## prepare the scATAC data
peaks_df <- read_tsv('scATAC_peaks.bed',col_names = c('chr','start','end'))
cells <- read_tsv('scATAC_barcodes.tsv',col_names=c('cell'))
mm <- Matrix::readMM('scATAC_matrix.mtx')
peaks_df$peaks <- paste0(peaks_df$chr, '-', peaks_df$start, '-', peaks_df$end)
init_gr <- EpiTrace::Init_Peakset(peaks_df)
init_mm <- EpiTrace::Init_Matrix(peakname=peaks_df$peaks, cellname=cells$cell, matrix=mm)

## EpiTrace analysis
# trace(EpiTraceAge_Convergence, edit=TRUE)  ref_genome='hg19'???
epitrace_obj_age_conv_estimated_by_mouse_clock <- EpiTraceAge_Convergence(peakSet=init_gr, matrix=init_mm, ref_genome='mm10',
                                                                          clock_gr=mouse_clock_by_MM285, 
                                                                          iterative_time=5, min.cutoff=0, non_standard_clock=T,
                                                                          qualnum=10, ncore_lim=48, mean_error_limit=0.1)
## add meta info of cell types
meta <- as.data.frame(epitrace_obj_age_conv_estimated_by_mouse_clock@meta.data)
write.table(meta, 'epitrace_meta.txt', sep='\t', quote=FALSE)
cell_info <- read.table('scATAC_meta.tsv', sep='\t', col.names=c('cell', 'age', 'cell_type'))
meta_cell_info <- left_join(meta, cell_info) 
epitrace_obj_age_conv_estimated_by_mouse_clock <- AddMetaData(epitrace_obj_age_conv_estimated_by_mouse_clock, metadata=meta_cell_info[['cell_type']], col.name='cell_type')

epitrace_obj_age_conv_estimated_by_mouse_clock[['all']] <- Seurat::CreateAssayObject(counts=init_mm[,c(epitrace_obj_age_conv_estimated_by_mouse_clock$cell)], min.cells=0, min.features=0, check.matrix=F)
DefaultAssay(epitrace_obj_age_conv_estimated_by_mouse_clock) <- 'all'
saveRDS(epitrace_obj_age_conv_estimated_by_mouse_clock, file='epitrace_obj_age_conv_estimated_by_mouse_clock.rds')
epitrace_obj_age_conv_estimated_by_mouse_clock <- readRDS(file='epitrace_obj_age_conv_estimated_by_mouse_clock.rds')

## peak-age association (consider all cell types for comparison)
# asso_res_mouse_clock <- AssociationOfPeaksToAge(epitrace_obj_age_conv_estimated_by_mouse_clock,
#                                                 epitrace_age_name="EpiTraceAge_iterative", parallel=TRUE, peakSetName='all')
# asso_res_mouse_clock[order(asso_res_mouse_clock$correlation_of_EpiTraceAge, decreasing=TRUE), ][1:10, 1:2]
# asso_res_mouse_clock[order(asso_res_mouse_clock$scaled_correlation_of_EpiTraceAge, decreasing=TRUE), ][1:10, c(1,3)]

## plot top 20 up & down peaks
## kidney proximal convoluted tubule epithelial cell
asso_res_mouse_clock_01 <- AssociationOfPeaksToAge(subset(epitrace_obj_age_conv_estimated_by_mouse_clock, cell_type %in% c('kidney proximal convoluted tubule epithelial cell')),
                                                   epitrace_age_name="EpiTraceAge_iterative", parallel=T, peakSetName='all')
up <- asso_res_mouse_clock_01[order(asso_res_mouse_clock_01$scaled_correlation_of_EpiTraceAge, decreasing=TRUE), ][1:20, c(1,3)]
dw <- asso_res_mouse_clock_01[order(asso_res_mouse_clock_01$scaled_correlation_of_EpiTraceAge, decreasing=FALSE), ][1:20, c(1,3)]
dat <- rbind(up, dw)
rownames(dat) <- seq(1, nrow(dat))
colnames(dat) <- c('Peaks', 'Scaled_correlation')
dat <- dat[order(dat$Scaled_correlation, decreasing=FALSE), ]
dat$Peaks <- factor(dat$Peaks, levels=dat$Peaks)
p <- ggplot(dat, aes(x=Peaks, y=Scaled_correlation)) + geom_bar(stat='identity', fill='darkred', position=position_dodge()) + coord_flip() +
                                                       scale_y_continuous(limits=c(-2.5, 2.5), breaks=seq(-2.5, 2.5, 0.5)) + theme_bw() + 
                                                       ggtitle('Kidney proximal convoluted tubule epithelial cells') + theme(plot.title=element_text(hjust=0.5))
ggsave(p, filename='peaks_epitrace_corr_kidney_proximal_convoluted_tubule_epithelial_cell.pdf', dpi=300, height=10, width=6)

## B cell
asso_res_mouse_clock_01 <- AssociationOfPeaksToAge(subset(epitrace_obj_age_conv_estimated_by_mouse_clock, cell_type %in% c('B cell')),
                                                   epitrace_age_name="EpiTraceAge_iterative", parallel=T, peakSetName='all')
up <- asso_res_mouse_clock_01[order(asso_res_mouse_clock_01$scaled_correlation_of_EpiTraceAge, decreasing=TRUE), ][1:20, c(1,3)]
dw <- asso_res_mouse_clock_01[order(asso_res_mouse_clock_01$scaled_correlation_of_EpiTraceAge, decreasing=FALSE), ][1:20, c(1,3)]
dat <- rbind(up, dw)
rownames(dat) <- seq(1, nrow(dat))
colnames(dat) <- c('Peaks', 'Scaled_correlation')
dat <- dat[order(dat$Scaled_correlation, decreasing=FALSE), ]
dat$Peaks <- factor(dat$Peaks, levels=dat$Peaks)
p <- ggplot(dat, aes(x=Peaks, y=Scaled_correlation)) + geom_bar(stat='identity', fill='darkred', position=position_dodge()) + coord_flip() +
                                                       scale_y_continuous(limits=c(-2.5, 3.0), breaks=seq(-2.5, 3.0, 0.5)) + theme_bw() + 
                                                       ggtitle('B cells') + theme(plot.title=element_text(hjust=0.5))
ggsave(p, filename='peaks_epitrace_corr_B_cell.pdf', dpi=300, height=10, width=6)

## epithelial cell of proximal tubule
asso_res_mouse_clock_01 <- AssociationOfPeaksToAge(subset(epitrace_obj_age_conv_estimated_by_mouse_clock, cell_type %in% c('epithelial cell of proximal tubule')),
                                                   epitrace_age_name="EpiTraceAge_iterative", parallel=T, peakSetName='all')
up <- asso_res_mouse_clock_01[order(asso_res_mouse_clock_01$scaled_correlation_of_EpiTraceAge, decreasing=TRUE), ][1:20, c(1,3)]
dw <- asso_res_mouse_clock_01[order(asso_res_mouse_clock_01$scaled_correlation_of_EpiTraceAge, decreasing=FALSE), ][1:20, c(1,3)]
dat <- rbind(up, dw)
rownames(dat) <- seq(1, nrow(dat))
colnames(dat) <- c('Peaks', 'Scaled_correlation')
dat <- dat[order(dat$Scaled_correlation, decreasing=FALSE), ]
dat$Peaks <- factor(dat$Peaks, levels=dat$Peaks)
p <- ggplot(dat, aes(x=Peaks, y=Scaled_correlation)) + geom_bar(stat='identity', fill='darkred', position=position_dodge()) + coord_flip() +
                                                       scale_y_continuous(limits=c(-2.5, 2.5), breaks=seq(-2.5, 2.5, 0.5)) + theme_bw() + 
                                                       ggtitle('Epithelial cell of proximal tubule') + theme(plot.title=element_text(hjust=0.5))
ggsave(p, filename='peaks_epitrace_corr_epithelial cell of proximal tubule.pdf', dpi=300, height=10, width=6)

## T cell
asso_res_mouse_clock_01 <- AssociationOfPeaksToAge(subset(epitrace_obj_age_conv_estimated_by_mouse_clock, cell_type %in% c('T cell')),
                                                   epitrace_age_name="EpiTraceAge_iterative", parallel=T, peakSetName='all')
up <- asso_res_mouse_clock_01[order(asso_res_mouse_clock_01$scaled_correlation_of_EpiTraceAge, decreasing=TRUE), ][1:20, c(1,3)]
dw <- asso_res_mouse_clock_01[order(asso_res_mouse_clock_01$scaled_correlation_of_EpiTraceAge, decreasing=FALSE), ][1:20, c(1,3)]
dat <- rbind(up, dw)
rownames(dat) <- seq(1, nrow(dat))
colnames(dat) <- c('Peaks', 'Scaled_correlation')
dat <- dat[order(dat$Scaled_correlation, decreasing=FALSE), ]
dat$Peaks <- factor(dat$Peaks, levels=dat$Peaks)
p <- ggplot(dat, aes(x=Peaks, y=Scaled_correlation)) + geom_bar(stat='identity', fill='darkred', position=position_dodge()) + coord_flip() +
                                                       scale_y_continuous(limits=c(-2.5, 3.0), breaks=seq(-2.5, 3.0, 0.5)) + theme_bw() + 
                                                       ggtitle('T cells') + theme(plot.title=element_text(hjust=0.5))
ggsave(p, filename='peaks_epitrace_corr_T_cell.pdf', dpi=300, height=10, width=6)

## macrophage
asso_res_mouse_clock_01 <- AssociationOfPeaksToAge(subset(epitrace_obj_age_conv_estimated_by_mouse_clock, cell_type %in% c('macrophage')),
                                                   epitrace_age_name="EpiTraceAge_iterative", parallel=T, peakSetName='all')
up <- asso_res_mouse_clock_01[order(asso_res_mouse_clock_01$scaled_correlation_of_EpiTraceAge, decreasing=TRUE), ][1:20, c(1,3)]
dw <- asso_res_mouse_clock_01[order(asso_res_mouse_clock_01$scaled_correlation_of_EpiTraceAge, decreasing=FALSE), ][1:20, c(1,3)]
dat <- rbind(up, dw)
rownames(dat) <- seq(1, nrow(dat))
colnames(dat) <- c('Peaks', 'Scaled_correlation')
dat <- dat[order(dat$Scaled_correlation, decreasing=FALSE), ]
dat$Peaks <- factor(dat$Peaks, levels=dat$Peaks)
p <- ggplot(dat, aes(x=Peaks, y=Scaled_correlation)) + geom_bar(stat='identity', fill='darkred', position=position_dodge()) + coord_flip() +
                                                       scale_y_continuous(limits=c(-2.5, 3.0), breaks=seq(-2.5, 3.0, 0.5)) + theme_bw() + 
                                                       ggtitle('Macrophages') + theme(plot.title=element_text(hjust=0.5))
ggsave(p, filename='peaks_epitrace_corr_macrophage.pdf', dpi=300, height=10, width=6)


######### plot for each age
library(dplyr)
library(ggplot2)

meta <- read.table('epitrace_meta.txt')
cell_info <- read.table('scATAC_meta.tsv', sep='\t', col.names=c('cell', 'age', 'cell_type'))
meta_cell_info <- inner_join(meta, cell_info) 

sort(table(meta_cell_info['cell_type']), decreasing=TRUE)[1:10]
#         kidney proximal convoluted tubule epithelial cell 
#                                                      3179 
#                                                    B cell 
#                                                      2612 
#                        epithelial cell of proximal tubule 
#                                                      1691 
#                                                lymphocyte 
#                                                      1272 
#                                                    T cell 
#                                                      1168 
#                                                macrophage 
#                                                      1166 
# kidney loop of Henle thick ascending limb epithelial cell 
#                                                      1116 
#                     kidney collecting duct principal cell 
#                                                       627 
#                                          fenestrated cell 
#                                                       577 
#           kidney distal convoluted tubule epithelial cell 
#                                                       449

## kidney proximal convoluted tubule epithelial cell
dat <- meta_cell_info %>% filter(cell_type=='kidney proximal convoluted tubule epithelial cell') %>% 
       select(EpiTraceAge_iterative, age) %>% dplyr::rename(epitrace_age=EpiTraceAge_iterative)

# dat %>% filter(age=='1m') %>% summarize(mean_value=mean(epitrace_age))  # 0.5720584
# dat %>% filter(age=='3m') %>% summarize(mean_value=mean(epitrace_age))  # 0.5702354
# dat %>% filter(age=='18m') %>% summarize(mean_value=mean(epitrace_age))  # 0.6278309
# dat %>% filter(age=='21m') %>% summarize(mean_value=mean(epitrace_age))  # 0.6548316
# dat %>% filter(age=='30m') %>% summarize(mean_value=mean(epitrace_age))  # 0.5969956

# dat %>% filter(age=='1m') %>% summarize(mean_value=median(epitrace_age))  # 0.5757576
# dat %>% filter(age=='3m') %>% summarize(mean_value=median(epitrace_age))  # 0.5703463
# dat %>% filter(age=='18m') %>% summarize(mean_value=median(epitrace_age))  # 0.6645022
# dat %>% filter(age=='21m') %>% summarize(mean_value=median(epitrace_age))  # 0.6872294
# dat %>% filter(age=='30m') %>% summarize(mean_value=median(epitrace_age))  # 0.6093074

dat['age'] <- factor(dat[['age']], levels=c('1m', '3m', '18m', '21m', '30m'))
table(dat['age'])
#  1m  3m 18m 21m 30m 
# 543 488 864 683 601
x_labels = names(table(dat['age']))
x_count = as.vector(table(dat['age']))
p <- ggplot(dat, aes(x=age, y=epitrace_age, fill=age)) + geom_boxplot(width=0.5, show.legend=FALSE, outlier.alpha=0) +
                                                         scale_x_discrete(breaks=x_labels, labels=paste0(x_labels, '\n(N=', x_count, ')')) +
                                                         scale_y_continuous(limits=c(0, 1), breaks=seq(0, 1, 0.2)) + theme_bw() +
                                                         ggtitle('Kidney proximal convoluted tubule epithelial cells') + theme(plot.title=element_text(hjust = 0.5))
ggsave(p, filename='epitrace_age_kidney_proximal_convoluted_tubule_epithelial_cell.pdf', dpi=300, height=4, width=6)


## B cell
dat <- meta_cell_info %>% filter(cell_type=='B cell') %>% 
       select(EpiTraceAge_iterative, age) %>% dplyr::rename(epitrace_age=EpiTraceAge_iterative)

dat['age'] <- factor(dat[['age']], levels=c('1m', '3m', '18m', '21m', '24m', '30m'))
table(dat['age'])
# 1m   3m  18m  21m  24m  30m 
#  8    9   44   37 2447   67 
x_labels = names(table(dat['age']))
x_count = as.vector(table(dat['age']))
p <- ggplot(dat, aes(x=age, y=epitrace_age, fill=age)) + geom_boxplot(width=0.5, show.legend=FALSE, outlier.alpha=0) + 
                                                         scale_x_discrete(breaks=x_labels, labels=paste0(x_labels, '\n(N=', x_count, ')')) +
                                                         scale_y_continuous(limits=c(0, 1), breaks=seq(0, 1, 0.2)) + theme_bw() +
                                                         ggtitle('B cells') + theme(plot.title=element_text(hjust = 0.5))
ggsave(p, filename='epitrace_age_B_cell.pdf', dpi=300, height=4, width=6)


## epithelial cell of proximal tubule
dat <- meta_cell_info %>% filter(cell_type=='epithelial cell of proximal tubule') %>% 
       select(EpiTraceAge_iterative, age) %>% dplyr::rename(epitrace_age=EpiTraceAge_iterative)

dat['age'] <- factor(dat[['age']], levels=c('1m', '3m', '18m', '21m', '30m'))
table(dat['age'])
# 1m   3m  18m  21m  30m 
# 56  132  231   34 1238
x_labels = names(table(dat['age']))
x_count = as.vector(table(dat['age']))
p <- ggplot(dat, aes(x=age, y=epitrace_age, fill=age)) + geom_boxplot(width=0.5, show.legend=FALSE, outlier.alpha=0) + 
                                                         scale_x_discrete(breaks=x_labels, labels=paste0(x_labels, '\n(N=', x_count, ')')) +
                                                         scale_y_continuous(limits=c(0, 1), breaks=seq(0, 1, 0.2)) + theme_bw() +
                                                         ggtitle('Epithelial cells of proximal tubule') + theme(plot.title=element_text(hjust = 0.5))
ggsave(p, filename='epitrace_age_epithelial_cell_of_proximal_tubule.pdf', dpi=300, height=4, width=6)


##  T cell
dat <- meta_cell_info %>% filter(cell_type=='T cell') %>% 
       select(EpiTraceAge_iterative, age) %>% dplyr::rename(epitrace_age=EpiTraceAge_iterative)

dat['age'] <- factor(dat[['age']], levels=c('1m', '3m', '18m', '21m', '24m', '30m'))
table(dat['age'])
# 1m  3m 18m 21m 24m 30m 
# 10  17  54  41 868 178
x_labels = names(table(dat['age']))
x_count = as.vector(table(dat['age']))
p <- ggplot(dat, aes(x=age, y=epitrace_age, fill=age)) + geom_boxplot(width=0.5, show.legend=FALSE, outlier.alpha=0) + 
                                                         scale_x_discrete(breaks=x_labels, labels=paste0(x_labels, '\n(N=', x_count, ')')) +
                                                         scale_y_continuous(limits=c(0, 1), breaks=seq(0, 1, 0.2)) + theme_bw() +
                                                         ggtitle('T cells') + theme(plot.title=element_text(hjust = 0.5))
ggsave(p, filename='epitrace_age_T_cells.pdf', dpi=300, height=4, width=6)


##  macrophage
dat <- meta_cell_info %>% filter(cell_type=='macrophage') %>% 
       select(EpiTraceAge_iterative, age) %>% dplyr::rename(epitrace_age=EpiTraceAge_iterative)

dat['age'] <- factor(dat[['age']], levels=c('1m', '3m', '18m', '21m', '24m', '30m'))
table(dat['age'])
# 1m  3m 18m 21m 24m 30m 
# 52 114 225  88 244 443
x_labels = names(table(dat['age']))
x_count = as.vector(table(dat['age']))
p <- ggplot(dat, aes(x=age, y=epitrace_age, fill=age)) + geom_boxplot(width=0.5, show.legend=FALSE, outlier.alpha=0) + 
                                                         scale_x_discrete(breaks=x_labels, labels=paste0(x_labels, '\n(N=', x_count, ')')) +
                                                         scale_y_continuous(limits=c(0, 1), breaks=seq(0, 1, 0.2)) + theme_bw() +
                                                         ggtitle('Macrophages') + theme(plot.title=element_text(hjust = 0.5))
ggsave(p, filename='epitrace_age_macrophages.pdf', dpi=300, height=4, width=6)


## kidney loop of Henle thick ascending limb epithelial cell
dat <- meta_cell_info %>% filter(cell_type=='kidney loop of Henle thick ascending limb epithelial cell') %>% 
       select(EpiTraceAge_iterative, age) %>% dplyr::rename(epitrace_age=EpiTraceAge_iterative)

dat['age'] <- factor(dat[['age']], levels=c('1m', '3m', '18m', '21m', '30m'))
table(dat['age'])
#  1m  3m 18m 21m 30m 
# 329 218 175 160 234 
x_labels = names(table(dat['age']))
x_count = as.vector(table(dat['age']))
p <- ggplot(dat, aes(x=age, y=epitrace_age, fill=age)) + geom_boxplot(width=0.5, show.legend=FALSE, outlier.alpha=0) + 
                                                         scale_x_discrete(breaks=x_labels, labels=paste0(x_labels, '\n(N=', x_count, ')')) +
                                                         scale_y_continuous(limits=c(0, 1), breaks=seq(0, 1, 0.2)) + theme_bw() +
                                                         ggtitle('Kidney loop of Henle thick ascending limb epithelial cells') + theme(plot.title=element_text(hjust = 0.5))
ggsave(p, filename='epitrace_age_kidney_loop_of_Henle_thick_ascending_limb_epithelial_cell.pdf', dpi=300, height=4, width=6)


# import statsmodels.api as sm
# from scipy.stats import norm

# X = [1, 3, 18, 21, 30]
# X = sm.add_constant(X)
# Y = [0.5721, 0.5702, 0.6278, 0.6548, 0.5970]
# model = sm.OLS(Y, X)
# results = model.fit()
# beta_hat = results.params[1]
# se_beta_hat = results.bse[1]
# wald_statistic = beta_hat / se_beta_hat
# p_value = 2 * (1 - norm.cdf(abs(wald_statistic)))


# development_stage & age







































