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











