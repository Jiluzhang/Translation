import scanpy as sc
import pandas as pd
import numpy as np
import pybedtools
from tqdm import tqdm

dat = sc.read_10x_h5('/fs/home/jiluzhang/scMOG_raw/scMOG_modified/dataset/DM_rep4.h5', gex_only=False)

########## rna ##########
rna = dat[:, dat.var['feature_types']=='Gene Expression'].copy()
rna.X = np.array(rna.X.todense()) 
genes = pd.read_table('human_genes.txt', names=['gene_id', 'gene_name'])

rna_genes = rna.var.copy()
rna_genes['gene_id'] = rna_genes['gene_ids'].map(lambda x: x.split('.')[0])  # delete ensembl id with version number
rna_exp = pd.concat([rna_genes, pd.DataFrame(rna.X.T, index=rna_genes.index)], axis=1)

X_new = pd.merge(genes, rna_exp, how='left', on='gene_id').iloc[:, 5:].T  # X_new = pd.merge(genes, rna_exp, how='left', left_on='gene_name', right_on='gene_id').iloc[:, 6:].T
X_new = np.array(X_new, dtype='float32')
X_new[np.isnan(X_new)] = 0

rna_new_var = pd.DataFrame({'gene_ids': genes['gene_id'], 'feature_types': 'Gene Expression', 'my_Id': range(genes.shape[0])})
rna_new = sc.AnnData(X_new, obs=rna.obs, var=rna_new_var)  # 5517*38244

rna_new.var.index = genes['gene_name']  # set index for var

rna_new[:1000, :].write('rna_train_dm_1000.h5ad')
rna_new[1000:2000, :].write('rna_train_dm_1000_2000.h5ad')
rna_new[2000:3000, :].write('rna_train_dm_2000_3000.h5ad')
rna_new[3000:4000, :].write('rna_train_dm_3000_4000.h5ad')
rna_new[4000:5000, :].write('rna_train_dm_4000_5000.h5ad')
rna_new[5417:, :].write('rna_test_dm_100.h5ad')


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

atac_new[:1000, :].write('atac_train_dm_1000.h5ad')
atac_new[1000:2000, :].write('atac_train_dm_1000_2000.h5ad')
atac_new[2000:3000, :].write('atac_train_dm_2000_3000.h5ad')
atac_new[3000:4000, :].write('atac_train_dm_3000_4000.h5ad')
atac_new[4000:5000, :].write('atac_train_dm_4000_5000.h5ad')
atac_new[5417:, :].write('atac_test_dm_100.h5ad')


## concat dm & hsr cells
import scanpy as sc
rna_dm = sc.read_h5ad('rna_train_dm_1000.h5ad')
rna_dm.var.index = rna_dm.var['gene_ids']
rna_hsr = sc.read_h5ad('rna_train_hsr_1000.h5ad')
rna_hsr.var.index = rna_hsr.var['gene_ids']
sc.AnnData.concatenate(rna_dm[:500, :], rna_hsr[:500, :]).write('rna_train_dm_500_hsr_500_1.h5ad')
sc.AnnData.concatenate(rna_dm[500:1000, :], rna_hsr[500:1000, :]).write('rna_train_dm_500_hsr_500_2.h5ad')

atac_dm = sc.read_h5ad('atac_train_dm_1000.h5ad')
atac_dm.var.index = atac_dm.var['gene_ids']
atac_hsr = sc.read_h5ad('atac_train_hsr_1000.h5ad')
atac_hsr.var.index = atac_hsr.var['gene_ids']
sc.AnnData.concatenate(atac_dm[:500, :], atac_hsr[:500, :]).write('atac_train_dm_500_hsr_500_1.h5ad')
sc.AnnData.concatenate(atac_dm[500:1000, :], atac_hsr[500:1000, :]).write('atac_train_dm_500_hsr_500_2.h5ad')


## select cells with peaks less than 4,000
import scanpy as sc
import numpy as np
dat = sc.read_10x_h5('/fs/home/jiluzhang/scMOG_raw/scMOG_modified/dataset/DM_rep4.h5', gex_only=False)
atac = dat[:, dat.var['feature_types']=='Peaks'][:5000, :]  # avoid overlapping with testing set
cnt = np.count_nonzero(atac.X.toarray(), axis=1)
atac = atac[cnt<4000, :][:1000, ].copy()
##### preprocess #####


rna = dat[:, dat.var['feature_types']=='Gene Expression'][:5000, :]
rna = rna[cnt<4000, :][:1000, ].copy()
##### preprocess #####


## calculate mean peak number
import scanpy as sc
import numpy as np
np.mean(np.count_nonzero(sc.read_h5ad('atac_train_dm_1000.h5ad').X, axis=1))  # 31529.794
np.mean(np.count_nonzero(sc.read_h5ad('atac_train_dm_500_hsr_500_1.h5ad').X, axis=1))  # 25666.041
np.mean(np.count_nonzero(sc.read_h5ad('atac_train_dm_1000_less_peaks.h5ad').X, axis=1))  # 10360.896
np.mean(np.count_nonzero(sc.read_h5ad('atac_train_sci_car_1000.h5ad').X, axis=1))  # 249.813


######################## process snubar-coassay data ########################
# zcat raw/GSM5484482_SNuBarARC-HBCA.scATAC.filtered_peak_bc_matrix.peaks.bed.gz | awk '{print $1 ":" $2 "-" $3}' | awk '{print $0 "\t" $0 "\t" "Peaks"}' > features.tsv
# gzip features.tsv
# wget -c https://hgdownload.soe.ucsc.edu/admin/exe/linux.x86_64/liftOver
# wget -c https://hgdownload.soe.ucsc.edu/goldenPath/hg19/liftOver/hg19ToHg38.over.chain.gz

# zcat raw/GSM5484482_SNuBarARC-HBCA.scATAC.filtered_peak_bc_matrix.peaks.bed.gz | awk '{print $0 "\t" "peak_" NR}' > peaks_hg19.bed
# liftOver peaks_hg19.bed hg19ToHg38.over.chain.gz peaks_hg38.bed unmapped.bed

# wget -c https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM5484nnn/GSM5484484/suppl/GSM5484484_lib.coassay.std_cellname_dictionary.csv.gz

import scanpy as sc
import pandas as pd

rna = sc.read_10x_mtx('scRNA', gex_only=False)    # 27129 × 32738
atac = sc.read_10x_mtx('scATAC', gex_only=False)  # 24373 × 142391

## find correspondence for rna & atac
co_barcodes = pd.read_csv('GSM5484484_lib.coassay.std_cellname_dictionary.csv.gz')
co_barcodes['RNA_original_cell_name'] = co_barcodes['RNA_original_cell_name'] + '-1'
co_barcodes['ATAC_original_cell_name'] = co_barcodes['ATAC_original_cell_name'] + '-1'

rna_idx = pd.DataFrame({'RNA_original_cell_name': rna.obs.index})
atac_idx = pd.DataFrame({'ATAC_original_cell_name': atac.obs.index})
rna_barcodes = pd.merge(rna_idx, co_barcodes)
rna_atac_barcodes = pd.merge(atac_idx, rna_barcodes)

rna_co = rna[rna_atac_barcodes['RNA_original_cell_name']]
atac_co = atac[rna_atac_barcodes['ATAC_original_cell_name']]

## hg19 -> hg38 for peaks
peaks_hg19 = pd.read_table('scATAC/peaks_hg19.bed', names=['chr_hg19', 'start_hg19', 'end_hg19', 'idx'])
peaks_hg38 = pd.read_table('scATAC/peaks_hg38.bed', names=['chr_hg38', 'start_hg38', 'end_hg38', 'idx'])
peaks_hg19_hg38 = pd.merge(peaks_hg19, peaks_hg38)
peaks_hg19_hg38 = peaks_hg19_hg38[peaks_hg19_hg38['chr_hg38'].isin(['chr'+str(i) for i in range(1, 23)])]
peaks_idx_hg19 = peaks_hg19_hg38['chr_hg19']+':'+peaks_hg19_hg38['start_hg19'].astype(str)+'-'+peaks_hg19_hg38['end_hg19'].astype(str)
peaks_idx_hg38 = peaks_hg19_hg38['chr_hg38']+':'+peaks_hg19_hg38['start_hg38'].astype(str)+'-'+peaks_hg19_hg38['end_hg38'].astype(str)

atac_co = atac_co[:, peaks_idx_hg19]
atac_co.var.index = peaks_idx_hg38
atac_co.var['gene_ids'] = peaks_idx_hg38.values

rna_co.X = rna_co.X.toarray()
rna_co.write('breast_snubar_rna.h5ad')     # 22123 × 32738
atac_co.X = atac_co.X.toarray()
atac_co.write('breast_snubar_atac.h5ad')   # 22123 × 139068


################## process snare-seq data ##################
# pip install pyreadr
# import pyreadr
# dat = pyreadr.read_r('raw/GSE183273_Kidney_Healthy-Injury_Cell_Atlas_SNARE2-AC_Peak-Counts_03282022.RDS')
# not work!
# pyreadr.custom_errors.LibrdataError: The file contains an unrecognized object


# pip install rds2py
from rds2py import read_rds, as_sparse_matrix

atac = read_rds('raw/GSE183273_Kidney_Healthy-Injury_Cell_Atlas_SNARE2-AC_Peak-Counts_03282022.RDS')
# dict_keys(['data', 'package_name', 'class_name', 'attributes'])
# 'attributes'  dict_keys(['i', 'p', 'Dim', 'Dimnames', 'x', 'factors'])

atac['attributes']['Dimnames']['data'][0]['data'][:5]
#['chr1-180738-180958', 'chr1-181000-181565', 'chr1-191218-191601', 'chr1-267970-268170', 'chr1-629564-630410']
atac['attributes']['Dimnames']['data'][1]['data'][:5]
['KM14_AGATGTACGTACGCAACAGCGTTA', 'KM14_CAAGACTAGACTAGTACAGCGTTA', 'KM14_CAATGGAAGGAGAACACAGCGTTA', 'KM14_TGGAACAACACTTCGACACTTCGA', 'KM14_AATGTTGCCTGTAGCCCATACCAA']


rna = read_rds('raw/GSE183273_Kidney_Healthy-Injury_Cell_Atlas_SNARE2-RNA_Counts_03282022.RDS')



