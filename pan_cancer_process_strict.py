## pan_cancer_process_strict_filter.py

import pandas as pd
import scanpy as sc
import numpy as np

ids = pd.read_table('files_all.txt', header=None)

for i in range(len(ids)):
    sample_id = ids.iloc[i, 0]
    dat = sc.read_10x_mtx(sample_id, gex_only=False)
    
    rna = dat[:, dat.var['feature_types']=='Gene Expression'].copy()
    sc.pp.filter_cells(rna, min_genes=1000)
    sc.pp.filter_cells(rna, max_genes=10000)
    
    atac = dat[:, dat.var['feature_types']=='Peaks'].copy()
    sc.pp.filter_cells(atac, min_genes=2000)
    sc.pp.filter_cells(atac, max_genes=50000)
    
    idx = np.intersect1d(rna.obs.index, atac.obs.index)
    rna[idx, :].copy().write('strict_filtered_out/'+sample_id+'_rna.h5ad')
    atac[idx, :].copy().write('strict_filtered_out/'+sample_id+'_atac.h5ad')
    print(i, sample_id, 'done')


## out_strict_paired_h5ad.py
import scanpy as sc
import pandas as pd
import numpy as np
import pybedtools
from scipy.sparse import csr_matrix
from tqdm import tqdm

ids = pd.read_table('files_all.txt', header=None)

for i in range(len(ids)):
    sample_id = ids.iloc[i, 0]
    
    ########## rna ##########
    rna = sc.read_h5ad('strict_filtered_out/'+sample_id+'_rna.h5ad')
    rna.X = np.array(rna.X.todense()) 
    genes = pd.read_table('cCRE_gene/human_genes.txt', names=['gene_id', 'gene_name'])
    
    rna_genes = rna.var.copy()
    rna_genes.rename(columns={'gene_ids': 'gene_id'}, inplace=True)
    rna_exp = pd.concat([rna_genes, pd.DataFrame(rna.X.T, index=rna_genes.index)], axis=1)
    
    X_new = pd.merge(genes, rna_exp, how='left', on='gene_id').iloc[:, 3:].T
    X_new = np.array(X_new, dtype='float32')
    X_new[np.isnan(X_new)] = 0
    
    rna_new_var = pd.DataFrame({'gene_ids': genes['gene_id'], 'feature_types': 'Gene Expression'})
    rna_new = sc.AnnData(X_new, obs=rna.obs, var=rna_new_var)  
    rna_new.var.index = genes['gene_name'].values  # set index for var
    rna_new.X = csr_matrix(rna_new.X)
    
    rna_new.write('strict_filtered_out/'+sample_id+'_rna_aligned.h5ad')
    
    
    ########## atac ##########
    atac = sc.read_h5ad('strict_filtered_out/'+sample_id+'_atac.h5ad')
    atac.X = np.array(atac.X.todense())
    atac.X[atac.X>1] = 1
    atac.X[atac.X<1] = 0
    
    peaks = pd.DataFrame({'id': atac.var_names})
    peaks['chr'] = peaks['id'].map(lambda x: x.split(':')[0])
    peaks['start'] = peaks['id'].map(lambda x: x.split(':')[1].split('-')[0])
    peaks['end'] = peaks['id'].map(lambda x: x.split(':')[1].split('-')[1])
    peaks.drop(columns='id', inplace=True)
    peaks['idx'] = range(peaks.shape[0])
    
    cCREs = pd.read_table('cCRE_gene/human_cCREs.bed', names=['chr', 'start', 'end'])
    cCREs['idx'] = range(cCREs.shape[0])
    cCREs_bed = pybedtools.BedTool.from_dataframe(cCREs)
    peaks_bed = pybedtools.BedTool.from_dataframe(peaks)
    idx_map = peaks_bed.intersect(cCREs_bed, wa=True, wb=True).to_dataframe().iloc[:, [3, 7]]
    idx_map.columns = ['peaks_idx', 'cCREs_idx']
    
    m = np.zeros([atac.n_obs, cCREs_bed.to_dataframe().shape[0]], dtype='float32')
    for j in tqdm(range(atac.X.shape[0])):
        m[j][idx_map[idx_map['peaks_idx'].isin(peaks.iloc[np.nonzero(atac.X[j])]['idx'])]['cCREs_idx']] = 1
    
    atac_new_var = pd.DataFrame({'gene_ids': cCREs['chr'] + ':' + cCREs['start'].map(str) + '-' + cCREs['end'].map(str), 'feature_types': 'Peaks'})
    atac_new = sc.AnnData(m, obs=atac.obs, var=atac_new_var)
    atac_new.var.index = atac_new.var['gene_ids'].values  # set index
    atac_new.X = csr_matrix(atac_new.X)
    
    atac_new.write('strict_filtered_out/'+sample_id+'_atac_aligned.h5ad')

    print(i, sample_id, 'done')


## merge_filter_genes_peaks.py
import scanpy as sc
import pandas as pd
import numpy as np
import anndata as ad

ids = pd.read_table('../files_all.txt', header=None)

## count for genes & peaks (time-consuming)
for i in range(len(ids)):
    sample_id = ids.iloc[i, 0]
    if i==0:
        rna_init = sc.read_h5ad(sample_id+'_rna_aligned.h5ad')
        rna_cnt = np.count_nonzero(rna_init.X.toarray(), axis=0)
        atac_init = sc.read_h5ad(sample_id+'_atac_aligned.h5ad')
        atac_cnt = np.count_nonzero(atac_init.X.toarray(), axis=0)
    else:
        rna_tmp = sc.read_h5ad(sample_id+'_rna_aligned.h5ad')
        rna_cnt = rna_cnt + np.count_nonzero(rna_tmp.X.toarray(), axis=0)
        atac_tmp = sc.read_h5ad(sample_id+'_atac_aligned.h5ad')
        atac_cnt = atac_cnt + np.count_nonzero(atac_tmp.X.toarray(), axis=0)
    print(i, sample_id, 'done')

np.save('rna_cnt_stats.npy', rna_cnt)
np.save('atac_cnt_stats.npy', atac_cnt)

## count cells
for i in range(len(ids)):
    sample_id = ids.iloc[i, 0]
    if i==0:
        rna_init = sc.read_h5ad(sample_id+'_rna_aligned.h5ad')
        cell_cnt = rna_init.n_obs
    else:
        rna_tmp = sc.read_h5ad(sample_id+'_rna_aligned.h5ad')
        cell_cnt = cell_cnt + rna_tmp.n_obs
    print(i, sample_id, 'done')

# cell_cnt 481,024

gene_idx = rna_cnt>481024*0.001   # 20,539
peak_idx = atac_cnt>481024*0.001  # 542,369


for sample_id in ['CE336E1-S1', 'CE348E1-S1K1', 'CPT2373DU-S1', 'HT181P1-T1A3', 'HT263B1-S1H1',
                  'ML123M1-Ty1', 'P5504-N1', 'PM581P1-T1', 'VF034V1-T1', 'GBML018G1-M1']:
    if sample_id=='CE336E1-S1':
        rna = sc.read_h5ad(sample_id+'_rna_aligned.h5ad')[:, gene_idx]
        rna.obs.index = sample_id+'-'+rna.obs.index
        atac = sc.read_h5ad(sample_id+'_atac_aligned.h5ad')[:, peak_idx]
        atac.obs.index = sample_id+'-'+atac.obs.index
    else:
        rna_tmp = sc.read_h5ad(sample_id+'_rna_aligned.h5ad')[:, gene_idx]
        rna_tmp.obs.index = sample_id+'-'+rna_tmp.obs.index
        rna = ad.concat([rna, rna_tmp])
        atac_tmp = sc.read_h5ad(sample_id+'_atac_aligned.h5ad')[:, peak_idx]
        atac_tmp.obs.index = sample_id+'-'+atac_tmp.obs.index
        atac = ad.concat([atac, atac_tmp])
    print(sample_id, 'done')

rna.var = rna_tmp.var    # 27845 × 20539
atac.var = atac_tmp.var  # 27845 × 542369

np.random.seed(0)
shuf_idx = np.arange(rna.shape[0])
np.random.shuffle(shuf_idx)
rna[shuf_idx[:25000], :].write('pan_cancer_rna_train_25000.h5ad')
atac[shuf_idx[:25000], :].write('pan_cancer_atac_train_25000.h5ad')
rna[shuf_idx[25000:], :].write('pan_cancer_rna_test_2845.h5ad')
atac[shuf_idx[25000:], :].write('pan_cancer_atac_test_2845.h5ad')


rna_cnt = np.load('rna_cnt_stats.npy')
atac_cnt = np.load('atac_cnt_stats.npy')
gene_idx = rna_cnt>481024*0.001  
peak_idx = atac_cnt>481024*0.001
crc_rna = sc.read_h5ad('CM354C2-T1_rna_aligned.h5ad')[:, gene_idx]
crc_rna.obs.index = 'CM354C2-T1-'+crc_rna.obs.index
crc_rna.write('pan_cancer_rna_test_crc.h5ad')
crc_atac = sc.read_h5ad('CM354C2-T1_atac_aligned.h5ad')[:, peak_idx]
crc_atac.obs.index = 'CM354C2-T1-'+crc_atac.obs.index
crc_atac.write('pan_cancer_atac_test_crc.h5ad')


# scM2M_v2_pro
accelerate launch --main_process_port 29501 --config_file default_config.yaml rna2atac_pre-train.py \
                  --atac pan_cancer_atac_train_25000.h5ad --rna pan_cancer_rna_train_25000.h5ad \
                  --save models --name pan_cancer --enc_max_len 20539 --dec_max_len 542369 --batch_size 10 --lr 0.001


## output h5ad files (5 samples per group)
import scanpy as sc
import pandas as pd
import numpy as np
import anndata as ad

rna_cnt = np.load('rna_cnt_stats.npy')
atac_cnt = np.load('atac_cnt_stats.npy')

gene_idx = rna_cnt>481024*0.001   # 20,539
peak_idx = atac_cnt>481024*0.001  # 542,369

sample_shuf = pd.read_table('sample_id_shuf.txt', header=None)
for i in range(25):
    rna_0 = sc.read_h5ad(sample_shuf[0][i*5]+'_rna_aligned.h5ad')[:, gene_idx]
    rna_0.obs.index = sample_shuf[0][i*5]+'-'+rna_0.obs.index
    rna_1 = sc.read_h5ad(sample_shuf[0][i*5+1]+'_rna_aligned.h5ad')[:, gene_idx]
    rna_1.obs.index = sample_shuf[0][i*5+1]+'-'+rna_1.obs.index
    rna_2 = sc.read_h5ad(sample_shuf[0][i*5+2]+'_rna_aligned.h5ad')[:, gene_idx]
    rna_2.obs.index = sample_shuf[0][i*5+2]+'-'+rna_2.obs.index
    rna_3 = sc.read_h5ad(sample_shuf[0][i*5+3]+'_rna_aligned.h5ad')[:, gene_idx]
    rna_3.obs.index = sample_shuf[0][i*5+3]+'-'+rna_3.obs.index
    rna_4 = sc.read_h5ad(sample_shuf[0][i*5+4]+'_rna_aligned.h5ad')[:, gene_idx]
    rna_4.obs.index = sample_shuf[0][i*5+4]+'-'+rna_4.obs.index
    rna = ad.concat([rna_0, rna_1, rna_2, rna_3, rna_4])
    rna.var = rna_0.var 
    print(i, 'merge rna done')
    
    atac_0 = sc.read_h5ad(sample_shuf[0][i*5]+'_atac_aligned.h5ad')[:, peak_idx]
    atac_0.obs.index = sample_shuf[0][i*5]+'-'+atac_0.obs.index
    atac_1 = sc.read_h5ad(sample_shuf[0][i*5+1]+'_atac_aligned.h5ad')[:, peak_idx]
    atac_1.obs.index = sample_shuf[0][i*5+1]+'-'+atac_1.obs.index
    atac_2 = sc.read_h5ad(sample_shuf[0][i*5+2]+'_atac_aligned.h5ad')[:, peak_idx]
    atac_2.obs.index = sample_shuf[0][i*5+2]+'-'+atac_2.obs.index
    atac_3 = sc.read_h5ad(sample_shuf[0][i*5+3]+'_atac_aligned.h5ad')[:, peak_idx]
    atac_3.obs.index = sample_shuf[0][i*5+3]+'-'+atac_3.obs.index
    atac_4 = sc.read_h5ad(sample_shuf[0][i*5+4]+'_atac_aligned.h5ad')[:, peak_idx]
    atac_4.obs.index = sample_shuf[0][i*5+4]+'-'+atac_4.obs.index
    atac = ad.concat([atac_0, atac_1, atac_2, atac_3, atac_4])
    atac.var = atac_0.var
    print(i, 'merge atac done')
    
    np.random.seed(0)
    shuf_idx = np.arange(rna.shape[0])
    np.random.shuffle(shuf_idx)
    rna[shuf_idx, :].write('pan_cancer_rna_dataset_'+str(i)+'.h5ad')
    atac[shuf_idx, :].write('pan_cancer_atac_dataset_'+str(i)+'.h5ad')
    print(i, 'output h5ad done')


nohup accelerate launch --config_file default_config.yaml rna2atac_pre-train.py --SEED 0 --epoch 1 \
                        --rna pan_cancer_rna_dataset_0.h5ad --atac pan_cancer_atac_dataset_0.h5ad \               
                        --save models --name dataset_0_p1 --enc_max_len 20539 --dec_max_len 542369 --batch_size 10 --lr 0.001 > train_dataset_0_p1.h5ad &







##################################################################################################
## https://humantumoratlas.org/
library(Seurat)
dat <- readRDS('CE337E1-S1N1.rds')
dat@meta.data$cell_type
##################################################################################################

for i in range(len(ids)):
    sample_id = ids.iloc[i, 0]
    if i==0:
        rna = sc.read_h5ad(sample_id+'_rna_aligned.h5ad')
        rna.obs.index = sample_id+'-'+rna.obs.index
        atac = sc.read_h5ad(sample_id+'_atac_aligned.h5ad')
        atac.obs.index = sample_id+'-'+atac.obs.index
    else:
        rna_tmp = sc.read_h5ad(sample_id+'_rna_aligned.h5ad')
        rna_tmp.obs.index = sample_id+'-'+rna_tmp.obs.index
        rna = ad.concat([rna, rna_tmp])
        atac_tmp = sc.read_h5ad(sample_id+'_atac_aligned.h5ad')
        atac_tmp.obs.index = sample_id+'-'+atac_tmp.obs.index
        atac = ad.concat([atac, atac_tmp])
    print(i, sample_id, 'done')

rna.var = rna_tmp.var
atac.var = atac_tmp.var

rna.write('pan_cancer_rna_head_20_samples.h5ad')
atac.write('pan_cancer_atac_head_20_samples.h5ad')

sc.pp.filter_genes(rna, min_cells=1)
sc.pp.filter_genes(atac, min_cells=1)

np.random.seed(0)
shuf_idx = np.arange(rna.shape[0])
np.random.shuffle(shuf_idx)
rna[shuf_idx[:10000], :].write('pan_cancer_rna_train_shuf.h5ad')     # 69577 × 23502 (head 20 samples)
atac[shuf_idx[:10000], :].write('pan_cancer_atac_train_shuf.h5ad')   # 69577 × 539927
rna[shuf_idx[-10000:], :].write('pan_cancer_rna_test_shuf.h5ad')
atac[shuf_idx[-10000:], :].write('pan_cancer_atac_test_shuf.h5ad')



sc.pp.filter_genes(rna, min_cells=100)
sc.pp.filter_genes(atac, min_cells=100)

np.random.seed(0)
shuf_idx = np.arange(rna.shape[0])
np.random.shuffle(shuf_idx)
rna[shuf_idx[:10000], :].write('pan_cancer_rna_train_shuf_2.h5ad')     # 69577 × 19546 (head 20 samples)
atac[shuf_idx[:10000], :].write('pan_cancer_atac_train_shuf_2.h5ad')   # 69577 × 471457
rna[shuf_idx[-10000:], :].write('pan_cancer_rna_test_shuf_2.h5ad')
atac[shuf_idx[-10000:], :].write('pan_cancer_atac_test_shuf_2.h5ad')


sc.pp.filter_genes(rna, min_cells=1)
sc.pp.filter_genes(atac, min_cells=100)

np.random.seed(0)
shuf_idx = np.arange(rna.shape[0])
np.random.shuffle(shuf_idx)
rna[shuf_idx[:10000], :].write('pan_cancer_rna_train_shuf_3.h5ad')     # 69577 × 23502 (head 20 samples)
atac[shuf_idx[:10000], :].write('pan_cancer_atac_train_shuf_3.h5ad')   # 69577 × 471457
rna[shuf_idx[-10000:], :].write('pan_cancer_rna_test_shuf_3.h5ad')
atac[shuf_idx[-10000:], :].write('pan_cancer_atac_test_shuf_3.h5ad')



accelerate launch --config_file default_config.yaml rna2atac_pre-train_v2.py --atac atac_32124_shuf.h5ad --rna rna_32124_shuf.h5ad --save models --name 32124_cells 

python rna2atac_pre-train_v3.py --atac pan_cancer_atac_train_shuf.h5ad --rna pan_cancer_rna_train_shuf.h5ad --save models --name pan_cancer \
                                --enc_max_len 23502 --dec_max_len 539927 --batch_size 5 --lr 0.001

python rna2atac_pre-train_v3.py --atac pan_cancer_atac_train_shuf.h5ad --rna pan_cancer_rna_train_shuf.h5ad --save models --name pan_cancer \
                                --enc_max_len 23502 --dec_max_len 539927 --batch_size 5 --lr 0.0001 --load models/pan_cancer_epoch_1_iter_159/pytorch_model.bin

accelerate launch --main_process_port 29501 --config_file default_config.yaml rna2atac_pre-train_v3.py \
                  --atac pan_cancer_atac_train_shuf_2.h5ad --rna pan_cancer_rna_train_shuf_2.h5ad \
                  --save models --name pan_cancer --enc_max_len 19546 --dec_max_len 471457 --batch_size 10 --lr 0.001

nohup accelerate launch --main_process_port 29501 --config_file default_config.yaml rna2atac_pre-train_v3.py \
                        --atac pan_cancer_atac_train_shuf_3.h5ad --rna pan_cancer_rna_train_shuf_3.h5ad \
                        --save models --name pan_cancer --enc_max_len 23502 --dec_max_len 471457 --batch_size 10 --lr 0.001 > training_20240328_3.log &

#latent_len: 32
