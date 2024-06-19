############ BMMC dataset ############
# wget -c https://ftp.ncbi.nlm.nih.gov/geo/series/GSE194nnn/GSE194122/suppl/GSE194122_openproblems_neurips2021_multiome_BMMC_processed.h5ad.gz
import scanpy as sc
bmmc = sc.read_h5ad('GSE194122_openproblems_neurips2021_multiome_BMMC_processed.h5ad')
bmmc.obs.cell_type.value_counts()
# cell_type
# CD8+ T                 11589
# CD14+ Mono             10843
# NK                      6929
# CD4+ T activated        5526
# Naive CD20+ B           5052
# Erythroblast            4916
# CD4+ T naive            4398
# Transitional B          2810
# Proerythroblast         2300
# CD16+ Mono              1894
# B1 B                    1890
# Normoblast              1780
# Lymph prog              1779
# G/M prog                1203
# pDC                     1191
# HSC                     1072
# CD8+ T naive            1012
# MK/E prog                884
# cDC2                     859
# ILC                      835
# Plasma cell              379
# ID2-hi myeloid prog      108


# import scanpy as sc
# import pandas as pd
# import numpy as np
# import pybedtools
# from tqdm import tqdm
# from scipy.sparse import csr_matrix

# dat = sc.read_h5ad('GSE194122_openproblems_neurips2021_multiome_BMMC_processed.h5ad')  # 69249 × 129921

## RNA
# rna = dat[:, dat.var['feature_types']=='GEX'].copy()  # 69249 × 13431
# rna.X = rna.X.toarray()
# genes = pd.read_table('human_genes.txt', names=['gene_ids', 'gene_name'])

# rna_exp = pd.concat([rna.var, pd.DataFrame(rna.X.T, index=rna.var.index)], axis=1)
# rna_exp.rename(columns={'gene_id': 'gene_ids'}, inplace=True)
# X_new = pd.merge(genes, rna_exp, how='left', on='gene_ids').iloc[:, 3:].T
# X_new.fillna(value=0, inplace=True)

# rna_new = sc.AnnData(X_new.values, obs=rna.obs, var=pd.DataFrame({'gene_ids': genes['gene_ids'], 'feature_types': 'Gene Expression'}))
# rna_new.var.index = genes['gene_name'].values
# rna_new.X = csr_matrix(rna_new.X)
# rna_new.write('rna_aligned.h5ad')

## ATAC
# for p in range(7):
#   if p!=6:
#     atac = dat[(p*1000):(p+1)*1000, dat.var['feature_types']=='ATAC'].copy()
#   else:
#     atac = dat[(p*1000):, dat.var['feature_types']=='ATAC'].copy()
  
#   atac.X = atac.X.toarray()
#   atac.X[atac.X>0] = 1  # binarization
  
#   peaks = pd.DataFrame({'id': atac.var_names})
#   peaks['chr'] = peaks['id'].map(lambda x: x.split('-')[0])
#   peaks['start'] = peaks['id'].map(lambda x: x.split('-')[1])
#   peaks['end'] = peaks['id'].map(lambda x: x.split('-')[2])
#   peaks.drop(columns='id', inplace=True)
#   peaks['idx'] = range(peaks.shape[0])
  
#   cCREs = pd.read_table('human_cCREs.bed', names=['chr', 'start', 'end'])
#   cCREs['idx'] = range(cCREs.shape[0])
#   cCREs_bed = pybedtools.BedTool.from_dataframe(cCREs)
#   peaks_bed = pybedtools.BedTool.from_dataframe(peaks)
#   idx_map = peaks_bed.intersect(cCREs_bed, wa=True, wb=True).to_dataframe().iloc[:, [3, 7]]
#   idx_map.columns = ['peaks_idx', 'cCREs_idx']
  
#   m = np.zeros([atac.n_obs, cCREs_bed.to_dataframe().shape[0]], dtype='float32')  # 10,000 cells per iteration to save memory
#   for i in tqdm(range(atac.X.shape[0]), ncols=80, desc='Aligning ATAC peaks'):
#       m[i][idx_map[idx_map['peaks_idx'].isin(peaks.iloc[np.nonzero(atac.X[i])]['idx'])]['cCREs_idx']] = 1
  
#   atac_new = sc.AnnData(m, obs=atac.obs, var=pd.DataFrame({'gene_ids': cCREs['chr']+':'+cCREs['start'].map(str)+'-'+cCREs['end'].map(str), 'feature_types': 'Peaks'}))
#   atac_new.var.index = atac_new.var['gene_ids'].values
#   atac_new.X = csr_matrix(atac_new.X)
#   atac_new.write('atac_aligned_'+str(p)+'.h5ad')
#   print(p, 'done')


## filter genes & peaks & cells
# sc.pp.filter_genes(rna_new, min_cells=10)
# sc.pp.filter_cells(rna_new, min_genes=500)
# sc.pp.filter_cells(rna_new, max_genes=5000)

# sc.pp.filter_genes(atac_new, min_cells=10)
# sc.pp.filter_cells(atac_new, min_genes=1000)
# sc.pp.filter_cells(atac_new, max_genes=50000)

# idx = np.intersect1d(rna_new.obs.index, atac_new.obs.index)
# rna_new[idx, :].copy().write('rna.h5ad')
# atac_new[idx, :].copy().write('atac.h5ad')
# print(f'Select {rna_new.n_obs} / {rna.n_obs} cells')
# print(f'Select {rna_new.n_vars} / 38244 genes')
# print(f'Select {atac_new.n_vars} / 1033239 peaks')


## raw preprocess
import scanpy as sc
import numpy as np

dat = sc.read_h5ad('GSE194122_openproblems_neurips2021_multiome_BMMC_processed.h5ad')  # 69249 × 129921

rna = dat[:, dat.var['feature_types']=='GEX'].copy()    # 69249 × 13431
sc.pp.filter_genes(rna, min_cells=10)
sc.pp.filter_cells(rna, min_genes=500)  # 58458 × 13431

atac = dat[:, dat.var['feature_types']=='ATAC'].copy()  # 69249 × 116490
sc.pp.filter_genes(atac, min_cells=10)
sc.pp.filter_cells(atac, min_genes=1000)
sc.pp.filter_cells(atac, max_genes=50000)   # 61346 × 116490

idx = np.intersect1d(rna.obs.index, atac.obs.index)
rna[idx, :].copy().write('rna.h5ad')
atac[idx, :].copy().write('atac.h5ad')


python split_train_val_test.py --RNA rna.h5ad --ATAC atac.h5ad --train_pct 0.7 --valid_pct 0.1
python data_preprocess.py -r rna_train.h5ad -a atac_train.h5ad -s preprocessed_data_train --dt train --config rna2atac_config_train.yaml
python data_preprocess.py -r rna_val.h5ad -a atac_val.h5ad -s preprocessed_data_val --dt val --config rna2atac_config_val_eval.yaml
python data_preprocess.py -r rna_test.h5ad -a atac_test.h5ad -s preprocessed_data_test --dt test --config rna2atac_config_val_eval.yaml





