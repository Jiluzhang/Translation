############ BMMC dataset ############
# wget -c https://ftp.ncbi.nlm.nih.gov/geo/series/GSE194nnn/GSE194122/suppl/GSE194122_openproblems_neurips2021_multiome_BMMC_processed.h5ad.gz

######### atac peak index !!!!!!!!!!!!!!!!!!!!!!!!!!!

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

nohup accelerate launch --config_file accelerator_config.yaml --main_process_port 29822 rna2atac_train.py --config_file rna2atac_config_train.yaml \
                        --train_data_dir ./preprocessed_data_train --val_data_dir ./preprocessed_data_val -n rna2atac_train > 20240618.log &
# 1222703

accelerate launch --config_file accelerator_config.yaml --main_process_port 29823 rna2atac_evaluate.py \
                  -d ./preprocessed_data_test \
                  -l save/2024-06-19_rna2atac_train_40/pytorch_model.bin --config_file rna2atac_config_val_eval.yaml

import scanpy as sc
atac = sc.read_h5ad('atac_test.h5ad')
atac.obs['cell_anno'] = atac.obs.cell_type
atac.write('rna2atac_true.h5ad')

mv predict.npy test_predict.npy
python npy2h5ad.py
mv rna2atac_scm2m.h5ad benchmark
python cal_auroc_auprc.py --pred rna2atac_scm2m.h5ad --true rna2atac_true.h5ad
python cal_cluster_plot.py --pred rna2atac_scm2m.h5ad --true rna2atac_true.h5ad   # 20 min


#################### scButterfly-B ####################
python scbt_b.py  # 3845160
python cal_auroc_auprc.py --pred rna2atac_scbutterflyb.h5ad --true rna2atac_true.h5ad
python cal_cluster_plot.py --pred rna2atac_scbutterflyb.h5ad --true rna2atac_true_leiden.h5ad


########### BABLE ###########
## concat train & valid dataset
import scanpy as sc
import anndata as ad

rna_train = sc.read_h5ad('rna_train.h5ad')
rna_train_sim = sc.AnnData(rna_train.X, obs=rna_train.obs[['cell_type']], var=rna_train.var)
rna_val = sc.read_h5ad('rna_val.h5ad')
rna_val_sim = sc.AnnData(rna_val.X, obs=rna_val.obs[['cell_type']], var=rna_val.var)
rna_train_val = ad.concat([rna_train_sim, rna_val_sim])
rna_train_val.var = rna_train.var[['gene_id', 'feature_types']]
rna_train_val.write('rna_train_val.h5ad')

atac_train = sc.read_h5ad('atac_train.h5ad')
atac_train_sim = sc.AnnData(atac_train.X, obs=atac_train.obs[['cell_type']], var=atac_train.var)
atac_val = sc.read_h5ad('atac_val.h5ad')
atac_val_sim = sc.AnnData(atac_val.X, obs=atac_val.obs[['cell_type']], var=atac_val.var)
atac_train_val = ad.concat([atac_train_sim, atac_val_sim])
atac_train_val.var = atac_train.var[['gene_id', 'feature_types']]
atac_train_val.var.index = atac_train_val.var.index.map(lambda x: x.replace('-', ':', 1))
atac_train_val.write('atac_train_val.h5ad')

## process peak index for test file
import scanpy as sc
atac = sc.read_h5ad('atac_test_raw.h5ad')
atac.var.index = atac.var.index.map(lambda x: x.replace('-', ':', 1))
atac.write('atac_test.h5ad')


# python h5ad2h5.py -n train_val
# python h5ad2h5.py -n test
import h5py
import numpy as np
from scipy.sparse import csr_matrix, hstack
import argparse

parser = argparse.ArgumentParser(description='concat RNA and ATAC h5ad to h5 format')
parser.add_argument('-n', '--name', type=str, help='sample type or name')

args = parser.parse_args()
sn = args.name
rna  = h5py.File('rna_'+sn+'.h5ad', 'r')
atac = h5py.File('atac_'+sn+'.h5ad', 'r')
out  = h5py.File(sn+'.h5', 'w')

g = out.create_group('matrix')
g.create_dataset('barcodes', data=rna['obs']['_index'][:])
rna_atac_csr_mat = hstack((csr_matrix((rna['X']['data'], rna['X']['indices'],  rna['X']['indptr']), shape=[rna['obs']['_index'].shape[0], rna['var']['_index'].shape[0]]),
                           csr_matrix((atac['X']['data'], atac['X']['indices'],  atac['X']['indptr']), shape=[atac['obs']['_index'].shape[0], atac['var']['_index'].shape[0]])))
rna_atac_csr_mat = rna_atac_csr_mat.tocsr()
g.create_dataset('data', data=rna_atac_csr_mat.data)
g.create_dataset('indices', data=rna_atac_csr_mat.indices)
g.create_dataset('indptr',  data=rna_atac_csr_mat.indptr)
l = list(rna_atac_csr_mat.shape)
l.reverse()
g.create_dataset('shape', data=l)

g_2 = g.create_group('features')
g_2.create_dataset('_all_tag_keys', data=np.array([b'genome', b'interval']))
g_2.create_dataset('feature_type', data=np.append([b'Gene Expression']*(rna['var']['_index'].shape[0]), [b'Peaks']*(atac['var']['_index'].shape[0])))
g_2.create_dataset('genome', data=np.array([b'GRCh38'] * (rna['var']['_index'].shape[0]+atac['var']['_index'].shape[0])))
g_2.create_dataset('id', data=np.append(rna['var']['_index'][:], atac['var']['_index'][:]))
g_2.create_dataset('interval', data=np.append(rna['var']['_index'][:], atac['var']['_index'][:]))
g_2.create_dataset('name', data=np.append(rna['var']['_index'][:], atac['var']['_index'][:]))      

out.close()


python /fs/home/jiluzhang/BABEL/bin/train_model.py --data train_val.h5 --outdir babel_train_out --batchsize 512 --earlystop 25 --device 4 --nofilter  
# 423322
python /fs/home/jiluzhang/BABEL/bin/predict_model.py --checkpoint babel_train_out --data test.h5 --outdir babel_test_out --device 4 \
                                                     --nofilter --noplot --transonly


## python babel_revise.py
import scanpy as sc
babel = sc.read_h5ad('rna2atac_babel.h5ad')
true = sc.read_h5ad('rna2atac_true.h5ad')
out = babel[true.obs.index, :]
out.var = babel.var
out.X = out.X.toarray()
out.write('rna2atac_babeltrue.h5ad')



