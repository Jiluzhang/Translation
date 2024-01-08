# mkdir normal_h5ad

# cCREs: wget -c https://downloads.wenglab.org/V3/GRCh38-cCREs.bed
awk '{if($1!="chrX" && $1!="chrY") print($1 "\t" $2 "\t" $3)}' GRCh38-cCREs.bed | sort -k1,1V -k2,2n | bedtools merge -i stdin > human_cCREs.bed  # 1,033,239

# human gtf file: wget -c https://ftp.ensembl.org/pub/release-110/gtf/homo_sapiens/Homo_sapiens.GRCh38.110.chr.gtf.gz
# gunzip Homo_sapiens.GRCh38.110.chr.gtf.gz
## cetered by gene_name
grep gene_name Homo_sapiens.GRCh38.110.chr.gtf | awk -F "\t" '{if($1!="X" && $1!="Y" && $1!="MT" && $3=="gene") print$9}' | awk '{print $2}' | sed 's/"//g' | sed 's/;//g' > gene_id.txt
grep gene_name Homo_sapiens.GRCh38.110.chr.gtf | awk -F "\t" '{if($1!="X" && $1!="Y" && $1!="MT" && $3=="gene") print$9}' | awk '{print $6}' | sed 's/"//g' | sed 's/;//g' > gene_name.txt
grep gene_name Homo_sapiens.GRCh38.110.chr.gtf | awk -F "\t" '{if($1!="X" && $1!="Y" && $1!="MT" && $3=="gene") print("chr" $1 "\t" $4 "\t" $5 "\t" $7)}' > gene_pos.txt
awk '{if($4=="+") print($1 "\t" $2); else print($1 "\t" $3)}' gene_pos.txt > gene_tss.txt
paste gene_id.txt gene_name.txt gene_tss.txt | sort -k3,3V -k4,4n | awk '{print $1 "\t" $2}' > human_genes_raw.txt
rm gene_id.txt gene_name.txt gene_pos.txt gene_tss.txt

## remove duplicates
import pandas as pd
dat = pd.read_table('human_genes_raw.txt', names=['gene_id', 'gene_name'])
gene_cnt = dat.groupby('gene_name').agg('count').reset_index()
gene_cnt.columns = ['gene_name', 'cnt']
out = pd.merge(dat, gene_cnt)
out = out[out['cnt']==1][['gene_id', 'gene_name']]
out.to_csv('human_genes.txt', sep='\t', index=False, header=False)  # 38,244

rm human_genes_raw.txt



## out_paired_h5ad.py
import scanpy as sc
import pandas as pd
import numpy as np
import pybedtools
from tqdm import tqdm
from scipy.sparse import csr_matrix
from datetime import datetime

ids = pd.read_table('files_all.txt', header=None)

for i in range(len(ids)):
    print(ids[0][i], 'start', datetime.now().replace(microsecond=0))
    
    dat = sc.read_10x_mtx(ids[0][i]+'_out/normal', gex_only=False)
    
    ########## rna ##########
    rna = dat[:, dat.var['feature_types']=='Gene Expression'].copy()
    rna.X = np.array(rna.X.todense()) 
    genes = pd.read_table('cCRE_gene/human_genes.txt', names=['gene_id', 'gene_name'])
    
    rna_genes = rna.var.copy()
    rna_genes['gene_id'] = rna_genes['gene_ids'].map(lambda x: x.split('.')[0])  # delete ensembl id with version number
    rna_exp = pd.concat([rna_genes, pd.DataFrame(rna.X.T, index=rna_genes.index)], axis=1)
    
    X_new = pd.merge(genes, rna_exp, how='left', on='gene_id').iloc[:, 4:].T
    X_new = np.array(X_new, dtype='float32')
    X_new[np.isnan(X_new)] = 0
    
    rna_new_var = pd.DataFrame({'gene_ids': genes['gene_id'], 'feature_types': 'Gene Expression'})
    rna_new = sc.AnnData(X_new, obs=rna.obs, var=rna_new_var)  
    rna_new.var.index = genes['gene_name'].values  # set index for var
    rna_new.X = csr_matrix(rna_new.X)
    
    rna_new.write('normal_h5ad/'+ids[0][i]+'_rna.h5ad')
    
    
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
    
    atac_new.write('normal_h5ad/'+ids[0][i]+'_atac.h5ad')

    print(ids[0][i], 'end', datetime.now().replace(microsecond=0))
