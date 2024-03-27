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

sc.pp.filter_genes(rna, min_cells=1)
sc.pp.filter_genes(atac, min_cells=1)

np.random.seed(0)
shuf_idx = np.arange(rna.shape[0])
np.random.shuffle(shuf_idx)
rna[shuf_idx[:10000], :].write('pan_cancer_rna_train_shuf.h5ad')     # 69577 × 23502 (head 20 samples)
atac[shuf_idx[:10000], :].write('pan_cancer_atac_train_shuf.h5ad')   # 69577 × 539927
rna[shuf_idx[-10000:], :].write('pan_cancer_rna_test_shuf.h5ad')
atac[shuf_idx[-10000:], :].write('pan_cancer_atac_test_shuf.h5ad')



accelerate launch --config_file default_config.yaml rna2atac_pre-train_v2.py --atac atac_32124_shuf.h5ad --rna rna_32124_shuf.h5ad --save models --name 32124_cells 

python rna2atac_pre-train_v3.py --atac pan_cancer_atac_train_shuf.h5ad --rna pan_cancer_rna_train_shuf.h5ad --save models --name pan_cancer \
                                --enc_max_len 23502 --dec_max_len 539927 --batch_size 5 --lr 0.001

python rna2atac_pre-train_v3.py --atac pan_cancer_atac_train_shuf.h5ad --rna pan_cancer_rna_train_shuf.h5ad --save models --name pan_cancer \
                                --enc_max_len 23502 --dec_max_len 539927 --batch_size 5 --lr 0.0001 --load models/pan_cancer_epoch_1_iter_159/pytorch_model.bin

