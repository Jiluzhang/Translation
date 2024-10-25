#### pan-cancer dataset
## python download_rds.py
import synapseclient
import pandas as pd
import os

syn = synapseclient.Synapse()
syn.login('Luz','jiluzhang_57888282')

ids = pd.read_table('rds_synapse_id.txt', header=None)
for id in ids[1].values:
    meta = syn.get(entity=id, downloadFile=True, downloadLocation='.')
    print(id, 'done')

#### generate info files (barcode & cell_anno)
# processbar: https://blog.csdn.net/ZaoJewin/article/details/133894474
# library(Seurat)
# library(Signac)

samples <- read.table('rds_samples_id.txt')  # remove normal samples (CPT704DU-M1  GBML018G1-M1N2  GBML019G1-M1N1
                                             #                        HT291C1-M1A3 SN001H1-Ms1  SP369H1-Mc1  SP819H1-Mc1)
pb <- txtProgressBar(min=0, max=122, style=3)

for (i in 1:nrow(samples)){
    s_1 <- samples[i, 1]
    rds <- readRDS(paste0(s_1, '.rds'))
    out <- rds@meta.data[c('Original_barcode', 'cell_type')]
    out <- out[(out$cell_type %in% c('Endothelial', 'B-cells', 'Plasma',
                                     'Fibroblasts', 'T-cells', 'Macrophages')), ]

    s_2 <- samples[i, 2]
    dir.create(s_2)
    write.table(out[c('Original_barcode', 'cell_type')], paste0(s_2, '/', s_2, '_info.txt'), row.names=FALSE, col.names=FALSE, quote=FALSE, sep='\t')
    setTxtProgressBar(pb, i) 
}
close(pb)


########## plot stacked bars ##########
from plotnine import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

samples = pd.read_table('rds_samples_cancer_type.txt', header=None) 
stat_df = pd.DataFrame({'Cancer_type':['CEAD']*5+['CESC']*5+['CRC']*5+['UCEC']*5+['PDAC']*5+['BRCA']*5+['HNSCC']*5+['SKCM']*5+['OV']*5,
                        'Cell_type': ['Endothelial', 'B-cells', 'Fibroblasts', 'T-cells', 'Macrophages']*9,
                        'Cell_count':0})

for i in range(samples.shape[0]):
    s = samples[0][i]
    dat = pd.read_table(s+'/'+s+'_info.txt', header=None)
    cancer_type = samples[1][i]
    stat_dict = dict(dat[1].value_counts())
    for cell_type in stat_dict.keys():  # BRCA & BRCA_Basal  B-cells & Plasma
        if cancer_type=='BRCA_Basal':
            if cell_type=='Plasma':
                stat_df.loc[(stat_df['Cancer_type']=='BRCA')&(stat_df['Cell_type']=='B-cells'), 'Cell_count'] += stat_dict[cell_type]
            else:
                stat_df.loc[(stat_df['Cancer_type']=='BRCA')&(stat_df['Cell_type']==cell_type), 'Cell_count'] += stat_dict[cell_type]
        else:
            if cell_type=='Plasma':
                stat_df.loc[(stat_df['Cancer_type']==cancer_type)&(stat_df['Cell_type']=='B-cells'), 'Cell_count'] += stat_dict[cell_type]
            else:
                stat_df.loc[(stat_df['Cancer_type']==cancer_type)&(stat_df['Cell_type']==cell_type), 'Cell_count'] += stat_dict[cell_type]

cancer_type_cnt = stat_df.groupby('Cancer_type', as_index=False).agg('sum')[['Cancer_type', 'Cell_count']]
#   Cancer_type  Cell_count
# 0        BRCA       19729
# 1        CEAD         401
# 2        CESC        4660
# 3         CRC       28641
# 4       HNSCC       20124
# 5          OV        6478
# 6        PDAC       36575
# 7        SKCM       26561
# 8        UCEC        1240

plt.rcParams['pdf.fonttype'] = 42
p = ggplot(cancer_type_cnt, aes(x='Cancer_type', y='Cell_count', fill='Cancer_type')) + geom_bar(stat='identity', position='stack') + \
                                                                            xlab('Cancer type') + ylab('Cell count') + scale_fill_brewer(palette="Paired", type='qual', direction=-1) +\
                                                                            scale_y_continuous(limits=[0, 40000], breaks=np.arange(0, 40000+1, 10000)) + theme_bw()
p.save(filename='cancer_type_cell_count.pdf', dpi=600, height=4, width=5)


stat_df['Cell_ptg'] = stat_df['Cell_count'] / stat_df.groupby('Cancer_type')['Cell_count'].transform('sum')  # calculate ptg for each cancer type
cell_type_lst = ['Endothelial', 'B-cells', 'Fibroblasts', 'T-cells', 'Macrophages']
stat_df['Cell_type'] = pd.Categorical(stat_df['Cell_type'], categories=cell_type_lst)

# color map: https://www.jianshu.com/p/b88a77a609da
plt.rcParams['pdf.fonttype'] = 42
p = ggplot(stat_df, aes(x='Cancer_type', y='Cell_ptg', fill='Cell_type')) + geom_bar(stat='identity', position='stack') + coord_flip() + \
                                                                            xlab('Cancer type') + ylab('Proportion') + scale_fill_brewer(palette="Set2", type='qual', direction=-1) +\
                                                                            scale_y_continuous(limits=[0, 1.01], breaks=np.arange(0, 1.1, 0.2)) + theme_bw()
p.save(filename='cancer_type_cell_type_ptg.pdf', dpi=600, height=4, width=5)


## python filter_cells_with_cell_anno.py  # 30 min
# something wrong in rds files
# VF027V1-S1_1N1   VF027V1-S1
# VF027V1-S1Y1     VF027V1-S2
# ->
# VF027V1-S1_1N1   VF027V1-S2
# VF027V1-S1Y1     VF027V1-S1
# i=122 & i=123

import scanpy as sc
import pandas as pd
from tqdm import tqdm

samples = pd.read_table('rds_samples_id.txt', header=None)
for i in tqdm(range(samples.shape[0]), ncols=80):
    s_2= samples[1][i]
    info = pd.read_table(s_2+'/'+s_2+'_info.txt', header=None)
    info.columns = ['barcode', 'cell_anno']
    dat = sc.read_10x_mtx('/fs/home/jiluzhang/2023_nature_LD/'+s_2, gex_only=False)
    out = dat[info['barcode'], :].copy()
    out.obs['cell_anno'] = info['cell_anno'].values
    out.write(s_2+'/'+s_2+'_filtered.h5ad')


## python rna_atac_align.py
import scanpy as sc
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
import pybedtools
from tqdm import tqdm

samples = pd.read_table('rds_samples_id.txt', header=None)
for s in tqdm(samples[1], ncols=80):
    dat = sc.read_h5ad(s+'/'+s+'_filtered.h5ad')
    
    ########## rna ##########
    rna = dat[:, dat.var['feature_types']=='Gene Expression'].copy()
    rna.X = rna.X.toarray()
    genes = pd.read_table('human_genes.txt', names=['gene_id', 'gene_name'])
    
    rna_genes = rna.var.copy()
    rna_genes['gene_id'] = rna_genes['gene_ids'].map(lambda x: x.split('.')[0])  # delete ensembl id with version number
    rna_exp = pd.concat([rna_genes, pd.DataFrame(rna.X.T, index=rna_genes.index)], axis=1)
    
    X_new = pd.merge(genes, rna_exp, how='left', on='gene_id').iloc[:, 4:].T
    X_new.fillna(value=0, inplace=True)
    
    rna_new = sc.AnnData(X_new.values, obs=rna.obs, var=pd.DataFrame({'gene_ids': genes['gene_id'], 'feature_types': 'Gene Expression'}))
    rna_new.var.index = genes['gene_name'].values
    rna_new.X = csr_matrix(rna_new.X)
    rna_new.write(s+'/'+s+'_rna.h5ad')
    
    ########## atac ##########
    atac = dat[:, dat.var['feature_types']=='Peaks'].copy()
    atac.X = atac.X.toarray()
    atac.X[atac.X>0] = 1  # binarization
    
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
    atac_new.X = csr_matrix(atac_new.X)
    atac_out.write(s+'/'+s+'_atac.h5ad')

    # for chrom in tqdm(['chr'+str(i) for i in range(1, 23)], ncols=80, desc='Writing chrom atac h5ad'):
    #     atac_out = atac_new[:, atac_new.var.index.map(lambda x: x.split(':')[0]==chrom)].copy()
    #     atac_out.X = csr_matrix(atac_out.X)
    #     atac_out.write(s+'/'+s+'_atac_'+chrom+'.h5ad')


#### merge same h5ad of cancer type
import pandas as pd
import scanpy as sc
import anndata as ad

samples = pd.read_table('rds_samples_cancer_type.txt', header=None) 
cancer_type_lst = list(samples[1].drop_duplicates().values)
for cancer_type in cancer_type_lst:
    s_lst = list(samples[samples[1]==cancer_type][0])
    for i in range(len(s_lst)):
        s = s_lst[i]
        if i==0:
            rna_out = sc.read_h5ad(s+'/'+s+'_rna.h5ad')
            rna_out.obs.index = s+'_'+rna_out.obs.index
            atac_out = sc.read_h5ad(s+'/'+s+'_atac.h5ad')
            atac_out.obs.index = s+'_'+atac_out.obs.index
        else:
            rna = sc.read_h5ad(s+'/'+s+'_rna.h5ad')
            rna.obs.index = s+'_'+rna.obs.index
            rna_out = ad.concat([rna_out, rna])
            atac = sc.read_h5ad(s+'/'+s+'_atac.h5ad')
            atac.obs.index = s+'_'+atac.obs.index
            atac_out = ad.concat([atac_out, atac])
        
        print(cancer_type, s)

    rna_out.var = rna.var
    atac_out.var = atac.var
    rna_out.write(cancer_type+'_rna_raw.h5ad')
    atac_out.write(cancer_type+'_atac_raw.h5ad')


#### merge & filter raw h5ad
import scanpy as sc
import anndata as ad
import pandas as pd
from tqdm import tqdm

samples = pd.read_table('rds_samples_cancer_type.txt', header=None) 
cancer_type_lst = list(samples[1].drop_duplicates().values)

## rna
for cancer_type in tqdm(cancer_type_lst, ncols=80, desc='Merge rna h5ad'):
    if cancer_type==cancer_type_lst[0]:
        rna = sc.read_h5ad(cancer_type+'_rna_raw.h5ad')
        rna.obs['cancer_type'] = cancer_type
    else:
        rna_tmp = sc.read_h5ad(cancer_type+'_rna_raw.h5ad')
        rna_tmp.obs['cancer_type'] = cancer_type
        rna = ad.concat([rna, rna_tmp])

rna.var = rna_tmp.var                         # 144409 × 38244
rna.write('all_cancer_type_rna_raw.h5ad')

sc.pp.filter_genes(rna, min_cells=100)        # 144409 × 19160
sc.pp.filter_cells(rna, min_genes=200)        # 144409 × 19160
sc.pp.filter_cells(rna, max_genes=20000)      # 144409 × 19160
rna.write('rna.h5ad')

## atac
for cancer_type in tqdm(cancer_type_lst, ncols=80, desc='Merge atac h5ad'):
    if cancer_type==cancer_type_lst[0]:
        atac = sc.read_h5ad(cancer_type+'_atac_raw.h5ad')
    else:
        atac_tmp = sc.read_h5ad(cancer_type+'_atac_raw.h5ad')
        atac = ad.concat([atac, atac_tmp])

atac.var = atac_tmp.var                       # 144409 × 1033239
atac.obs['cancer_type'] = rna.obs['cancer_type']
atac.write('all_cancer_type_atac_raw.h5ad')

sc.pp.filter_genes(atac, min_cells=100)       # 144409 × 389267
sc.pp.filter_cells(atac, min_genes=500)       # 144409 × 389267
sc.pp.filter_cells(atac, max_genes=50000)     # 144409 × 389267
atac.write('atac.h5ad')


#### split dataset (ov for testing)
import scanpy as sc
import random

rna = sc.read_h5ad('rna.h5ad')
rna_test = rna[rna.obs['cancer_type']=='OV', :].copy()        # 6478 × 19160
rna_train_val = rna[rna.obs['cancer_type']!='OV', :].copy()   # 137931 × 19160

random.seed(0)
idx = list(range(rna_train_val.n_obs))
random.shuffle(idx)
train_idx = idx[:int(len(idx)*0.9)]
val_idx = idx[int(len(idx)*0.9):]

rna[train_idx, :].write('rna_train.h5ad')
rna[val_idx, :].write('rna_val.h5ad')
rna_test.write('rna_test.h5ad')

atac = sc.read_h5ad('atac.h5ad')
atac[train_idx, :].write('atac_train.h5ad')
atac[val_idx, :].write('atac_val.h5ad')
atac[rna_test.obs.index, :].write('atac_test.h5ad')

## max gene count
import scanpy as sc
sc.read_h5ad('rna_test.h5ad').obs.n_genes.max()  # 8779


#### cisformer
python data_preprocess.py -r rna_train.h5ad -a atac_train.h5ad -s train_pt --dt train -n train --config rna2atac_config_train.yaml   # ~2 h
python data_preprocess.py -r rna_val.h5ad -a atac_val.h5ad -s val_pt --dt val -n val --config rna2atac_config_val.yaml 
python data_preprocess.py -r rna_test.h5ad -a atac_test.h5ad -s test_pt --dt test -n test --config rna2atac_config_test.yaml 
nohup accelerate launch --config_file accelerator_config_train.yaml --main_process_port 29823 rna2atac_train.py --config_file rna2atac_config_train.yaml \
                        --train_data_dir train_pt --val_data_dir val_pt -s save -n rna2atac_pan_cancer > rna2atac_train_20241025.log &   # gamma_step:1  gamma:0.5
# 1366896
















