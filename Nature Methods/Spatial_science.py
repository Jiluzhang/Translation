#### https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi
#### Single-cell genomics and regulatory networks for 388 human brains

## RT00372N
## wget -c https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM8155nnn/GSM8155483/suppl/GSM8155483_RT00372N_RNA_barcodes.txt.gz
## wget -c https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM8155nnn/GSM8155483/suppl/GSM8155483_RT00372N_RNA_matrix.mtx.gz
## wget -c https://ftp.ncbi.nlm.nih.gov/geo/series/GSE261nnn/GSE261983/suppl/GSE261983_RNA_gene_list.txt.gz
## wget -c https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM8155nnn/GSM8155503/suppl/GSM8155503_RT00372N_ATAC_barcodes.txt.gz
## wget -c https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM8155nnn/GSM8155503/suppl/GSM8155503_RT00372N_ATAC_matrix.mtx.gz
## wget -c https://ftp.ncbi.nlm.nih.gov/geo/series/GSE261nnn/GSE261983/suppl/GSE261983_ATAC_peakset.bed.gz

#zcat GSM8155503_RT00372N_ATAC_matrix.mtx.gz | head -n 2 >> ATAC_matrix.mtx
#zcat GSM8155503_RT00372N_ATAC_matrix.mtx.gz | tail -n +3 | awk '{print $2 "\t" $1 "\t" $3}' >> ATAC_matrix.mtx

import scanpy as sc
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
import pybedtools
from tqdm import tqdm

rna_raw = sc.read_mtx('GSM8155483_RT00372N_RNA_matrix.mtx.gz')  # 8369 × 35649
rna_raw_barcode = pd.read_table('GSM8155483_RT00372N_RNA_barcodes.txt.gz', header=None)
rna_raw.obs.index = rna_raw_barcode[0].values
rna_raw_feature = pd.read_table('GSE261983_RNA_gene_list.txt.gz', header=None)
rna_raw.var.index = rna_raw_feature[0].values
rna_raw.X = rna_raw.X.T

atac_raw = sc.read_mtx('ATAC_matrix.mtx')  # 8984 × 423443
atac_raw_barcode = pd.read_table('GSM8155503_RT00372N_ATAC_barcodes.txt.gz', header=None)
atac_raw.obs.index = atac_raw_barcode[0].values
atac_raw_feature = pd.read_table('GSE261983_ATAC_peakset.bed.gz', header=None)
atac_raw_feature['idx'] = atac_raw_feature[0]+':'+atac_raw_feature[1].astype('str')+'-'+atac_raw_feature[2].astype('str')
atac_raw.var.index = atac_raw_feature['idx'].values

idx = np.intersect1d(rna_raw.obs.index, atac_raw.obs.index)
rna_raw[idx, :].write('rna_raw.h5ad')
atac_raw[idx, :].write('atac_raw.h5ad')

## rna
rna = sc.read_h5ad('rna_raw.h5ad')  # 8369 × 35649
rna.X = rna.X.toarray()
genes = pd.read_table('human_genes.txt', names=['gene_ids', 'gene_name'])
rna_exp = pd.concat([rna.var, pd.DataFrame(rna.X.T, index=rna.var.index)], axis=1)
rna_exp['gene_name'] = rna_exp.index.values
X_new = pd.merge(genes, rna_exp, how='left', on='gene_name').iloc[:, 2:].T
X_new.fillna(value=0, inplace=True)
rna_new = sc.AnnData(X_new.values, obs=rna.obs, var=pd.DataFrame({'gene_ids': genes['gene_ids'], 'feature_types': 'Gene Expression'}))
rna_new.var.index = genes['gene_name'].values
rna_new.X = csr_matrix(rna_new.X)   # 8369 × 38244

## atac
atac = sc.read_h5ad('atac_raw.h5ad')  # 8369 × 423443
atac.X[atac.X>0] = 1  # binarization
atac.X = atac.X.toarray()
  
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
for i in tqdm(range(atac.X.shape[0]), ncols=80, desc='Aligning ATAC peaks'):
    m[i][idx_map[idx_map['peaks_idx'].isin(peaks.iloc[np.nonzero(atac.X[i])]['idx'])]['cCREs_idx']] = 1

atac_new = sc.AnnData(m, obs=atac.obs, var=pd.DataFrame({'gene_ids': cCREs['chr']+':'+cCREs['start'].map(str)+'-'+cCREs['end'].map(str), 'feature_types': 'Peaks'}))
atac_new.var.index = atac_new.var['gene_ids'].values
atac_new.X = csr_matrix(atac_new.X)  # 8369 × 1033239

## filtering
sc.pp.filter_genes(rna_new, min_cells=10)       # 8369 × 17532
sc.pp.filter_cells(rna_new, min_genes=200)      # 8369 × 17532
sc.pp.filter_cells(rna_new, max_genes=20000)    # 8369 × 17532
rna_new.obs['n_genes'].max()                    # 8121

sc.pp.filter_genes(atac_new, min_cells=10)     # 8369 × 404378
sc.pp.filter_cells(atac_new, min_genes=500)    # 8369 × 404378
sc.pp.filter_cells(atac_new, max_genes=50000)  # 8369 × 404378

rna_new.write('rna.h5ad')
atac_new.write('atac.h5ad')

## train model
python split_train_val.py --RNA rna.h5ad --ATAC atac.h5ad --train_pct 0.9
nohup python data_preprocess.py -r rna_train.h5ad -a atac_train.h5ad -s train_pt --dt train -n train --config rna2atac_config_train.yaml > data_preprocess.log &   # 1095915
python data_preprocess.py -r rna_val.h5ad -a atac_val.h5ad -s val_pt --dt val -n val --config rna2atac_config_val.yaml
nohup accelerate launch --config_file accelerator_config_train.yaml --main_process_port 29824 rna2atac_train.py --config_file rna2atac_config_train.yaml \
                        --train_data_dir train_pt --val_data_dir val_pt -s save -n rna2atac_brain > 20250116.log &   # 1163370


######################################################## filter & map rna & atac h5ad ########################################################
import scanpy as sc
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

rna = sc.read_h5ad('spatial_rna.h5ad')  #  2500 × 38244

sc.pp.filter_genes(rna, min_cells=10)       # 2500 × 13997
# sc.pp.filter_cells(rna, min_genes=200)      # 2196 × 13997  (do not filter cells)
# sc.pp.filter_cells(rna, max_genes=20000)    # 2196 × 13997
np.count_nonzero(rna.X.toarray(), axis=1).max()   # 6589

atac = sc.read_h5ad('spatial_atac.h5ad')  #  2500 × 1033239

sc.pp.filter_genes(atac, min_cells=10)       # 2500 × 657953
# sc.pp.filter_cells(atac, min_genes=500)      # 2429 × 657953
# sc.pp.filter_cells(atac, max_genes=50000)    # 2424 × 657953
np.count_nonzero(atac.X.toarray(), axis=1).max()   # 60239

# idx = np.intersect1d(rna.obs.index, atac.obs.index)
# del rna.var['n_cells']
# rna[idx, :].copy().write('spatial_rna_filtered.h5ad')     # 2500 × 13997
# del atac.var['n_cells']
# atac[idx, :].copy().write('spatial_atac_filtered.h5ad')   # 2500 × 657953

# rna = rna[idx, :].copy()
# atac = atac[idx, :].copy()

rna_ref = sc.read_h5ad('./rna.h5ad')   # 8369 × 17532
genes = rna_ref.var[['gene_ids']].copy()
genes['gene_name'] = genes.index.values
genes.reset_index(drop=True, inplace=True)
rna_exp = pd.concat([rna.var, pd.DataFrame(rna.X.toarray().T, index=rna.var.index)], axis=1)
X_new = pd.merge(genes, rna_exp, how='left', on='gene_ids').iloc[:, 4:].T
X_new.fillna(value=0, inplace=True)
rna_new = sc.AnnData(X_new.values, obs=rna.obs, var=pd.DataFrame({'gene_ids': genes['gene_ids'], 'feature_types': 'Gene Expression'})) 
# rna_new = sc.AnnData(X_new.values, obs=rna.obs[['cell_anno']], var=pd.DataFrame({'gene_ids': genes['gene_ids'], 'feature_types': 'Gene Expression'}))
rna_new.var.index = genes['gene_name'].values
rna_new.X = csr_matrix(rna_new.X)   # 2500 × 17532
rna_new.write('spatial_rna_test.h5ad')
np.count_nonzero(rna_new.X.toarray(), axis=1).max()  # 6378

## atac
atac_ref = sc.read_h5ad('./atac.h5ad')   # 8369 × 404378

peaks = atac_ref.var[['gene_ids']].copy()
peaks['gene_name'] = peaks.index.values
peaks.reset_index(drop=True, inplace=True)
atac_exp = pd.concat([atac.var, pd.DataFrame(atac.X.toarray().T, index=atac.var.index)], axis=1)
X_new = pd.merge(peaks, atac_exp, how='left', on='gene_ids').iloc[:, 4:].T
X_new.fillna(value=0, inplace=True)
atac_new = sc.AnnData(X_new.values, obs=atac.obs, var=pd.DataFrame({'gene_ids': peaks['gene_ids'], 'feature_types': 'Peaks'}))
# atac_new = sc.AnnData(X_new.values, obs=atac.obs[['cell_anno']], var=pd.DataFrame({'gene_ids': peaks['gene_ids'], 'feature_types': 'Peaks'}))
atac_new.var.index = peaks['gene_name'].values
atac_new.X = csr_matrix(atac_new.X)   # 2500 × 404378
atac_new.write('spatial_atac_test.h5ad')
########################################################################################################################################################################


#### clustering for true with rna
import scanpy as sc

rna = sc.read_h5ad('spatial_rna_test.h5ad')
sc.pp.normalize_total(rna, target_sum=1e4)
sc.pp.log1p(rna)
sc.pp.highly_variable_genes(rna)
rna = rna[:, rna.var.highly_variable]
sc.tl.pca(rna)
sc.pp.neighbors(rna)
sc.tl.umap(rna)
sc.tl.leiden(rna, resolution=0.9)
rna.obs['leiden'].value_counts()
# 0    688
# 1    424
# 2    400
# 3    301
# 4    299
# 5    174
# 6    152
# 7     62

rna.obs['cell_anno'] = rna.obs['leiden']
rna.write('rna_test.h5ad')

atac = sc.read_h5ad('spatial_atac_test.h5ad')
atac.obs['cell_anno'] = rna.obs['leiden']
atac.write('atac_test.h5ad')



python data_preprocess.py -r spatial_rna_test.h5ad -a spatial_atac_test.h5ad -s spatial_test_pt --dt test --config rna2atac_config_test.yaml
accelerate launch --config_file accelerator_config_test.yaml --main_process_port 29822 rna2atac_predict.py \
                  -d ./spatial_test_pt -l ./save/2025-01-17_rna2atac_brain_5/pytorch_model.bin --config_file rna2atac_config_test.yaml
python npy2h5ad.py
python cal_auroc_auprc.py --pred atac_cisformer.h5ad --true atac_test.h5ad
# Cell-wise AUROC: 0.6449
# Cell-wise AUPRC: 0.0229
# Peak-wise AUROC: 0.5162
# Peak-wise AUPRC: 0.0134
python plot_save_umap.py --pred atac_cisformer.h5ad --true atac_test.h5ad
python cal_cluster.py --file atac_cisformer_umap.h5ad
# AMI: 0.3397
# ARI: 0.2487
# HOM: 0.3527
# NMI: 0.3432


#### scbutterfly
nohup python scbt_b.py > scbt_b_20250117.log &   # 2391368

# cp predict/R2A.h5ad atac_scbt.h5ad
# python cal_auroc_auprc.py --pred atac_scbt.h5ad --true atac_test.h5ad
# python plot_save_umap.py --pred atac_scbt.h5ad --true atac_test.h5ad
# python cal_cluster.py --file atac_scbt_umap.h5ad

nohup python scbt_b_predict.py > scbt_b_predict.log &  # 2382519
# cp predict/R2A.h5ad atac_scbt.h5ad
# python cal_auroc_auprc.py --pred atac_scbt.h5ad --true atac_test.h5ad
# python plot_save_umap.py --pred atac_scbt.h5ad --true atac_test.h5ad
# python cal_cluster.py --file atac_scbt_umap.h5ad


#### BABEL
python h5ad2h5.py -n train_val
nohup python /fs/home/jiluzhang/BABEL/bin/train_model.py --data train_val.h5 --outdir train_out --batchsize 512 --earlystop 25 --device 2 --nofilter > train_20250117.log &  # 1234005

# python h5ad2h5.py -n test
# nohup python /fs/home/jiluzhang/BABEL/bin/predict_model.py --checkpoint train_out --data test.h5 --outdir test_out --device 1 --nofilter --noplot --transonly > predict_20241226.log &
# # 3582789
# python match_cell.py
# python cal_auroc_auprc.py --pred atac_babel.h5ad --true atac_test.h5ad
# python plot_save_umap.py --pred atac_babel.h5ad --true atac_test.h5ad
# python cal_cluster.py --file atac_babel_umap.h5ad

python h5ad2h5.py -n test
nohup python /fs/home/jiluzhang/BABEL/bin/predict_model.py --checkpoint ./train_out --data test.h5 --outdir test_out --device 3 --nofilter --noplot --transonly > predict_20250118.log &
# 2398520
python match_cell.py
python cal_auroc_auprc.py --pred atac_babel.h5ad --true atac_test.h5ad
# Cell-wise AUROC: 0.6839
# Cell-wise AUPRC: 0.0257
# Peak-wise AUROC: 0.6408
# Peak-wise AUPRC: 0.026
python plot_save_umap.py --pred atac_babel.h5ad --true atac_test.h5ad
python cal_cluster.py --file atac_babel_umap.h5ad
# AMI: 0.2982
# ARI: 0.1813
# HOM: 0.3041
# NMI: 0.3016


