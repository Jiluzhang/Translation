## Previous pan-cancer model: /fs/home/jiluzhang/scM2M_no_dec_attn/pan_cancer/save/2024-06-25_rna2atac_train_3/pytorch_model.bin
## Current model with only decoder

## workdir: /fs/home/jiluzhang/scM2M_no_dec_attn/pan_cancer/all_data/data_with_annotation/h5ad/scM2M/TabulaTIME

## copy files
for sample in `ls /fs/home/jiluzhang/scM2M_no_dec_attn/pan_cancer/all_data/data_with_annotation/h5ad/scM2M/preprocessed_data_train`;do
    cp /fs/home/jiluzhang/scM2M_no_dec_attn/pan_cancer/all_data/data_with_annotation/h5ad/scM2M/preprocessed_data_train/$sample/*filtered.h5ad .
done

for sample in `ls /fs/home/jiluzhang/scM2M_no_dec_attn/pan_cancer/all_data/data_with_annotation/h5ad | grep VF`;do
    cp /fs/home/jiluzhang/scM2M_no_dec_attn/pan_cancer/all_data/data_with_annotation/h5ad/$sample/*filtered.h5ad .
done

cp /fs/home/jiluzhang/scM2M_no_dec_attn/pan_cancer/all_data/data_with_annotation/h5ad/scM2M/preprocessed_data_val/CPT704DU-M1/*filtered.h5ad .
cp /fs/home/jiluzhang/scM2M_no_dec_attn/pan_cancer/all_data/data_with_annotation/h5ad/rds_samples_id.txt .


## filter tumor cells & normal samples
import scanpy as sc
import pandas as pd
from tqdm import tqdm

samples = pd.read_table('../rds_samples_id.txt', header=None)
normal_samples = ['CPT704DU-M1', 'GBML018G1-M1N2', 'GBML019G1-M1N1', 'HT291C1-M1A3', 'SN001H1-Ms1', 'SP369H1-Mc1', 'CRC_SP819H1-Mc1']
no_tumor_sample = []
no_tumor_cnt = []

for i in tqdm(range(samples.shape[0]), ncols=80):
    if samples[0][i] not in normal_samples:
        s_2= samples[1][i]
        dat = sc.read_h5ad(s_2+'_filtered.h5ad')
        dat = dat[dat.obs['cell_anno']!='Tumor']
        dat.write('../without_tumor_data/'+s_2+'_no_tumor.h5ad')
        no_tumor_sample.append(s_2)
        no_tumor_cnt.append(dat.n_obs)

sum(no_tumor_cnt)  # 175,286
out = pd.DataFrame({'sample':no_tumor_sample, 'cnt':no_tumor_cnt})
out.to_csv('../pan_cancer_no_tumor_stat.txt', sep='\t', header=None, index=None)


## workdir: /fs/home/jiluzhang/scM2M_no_dec_attn/pan_cancer/all_data/data_with_annotation/h5ad/scM2M/TabulaTIME/without_tumor_data
# cp /fs/home/jiluzhang/scM2M_no_dec_attn/pan_cancer/all_data/data_with_annotation/h5ad/human_genes.txt .
# cp /fs/home/jiluzhang/scM2M_no_dec_attn/pan_cancer/all_data/data_with_annotation/h5ad/human_cCREs.bed .
# ls *_no_tumor.h5ad | sed 's/_.*//' > ../normal_samples.txt
## python rna_atac_align.py
import scanpy as sc
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
import pybedtools
from tqdm import tqdm

samples = pd.read_table('../normal_samples.txt', header=None)[0].values
for s in tqdm(samples[99:], ncols=80):
    dat = sc.read_h5ad(s+'_no_tumor.h5ad')
    
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
    rna_new.write(s+'_rna.h5ad')
    
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
    atac_new.write(s+'_atac.h5ad')


## workdir: /fs/home/jiluzhang/scM2M_no_dec_attn/pan_cancer/all_data/data_with_annotation/h5ad/scM2M/TabulaTIME
# mkdir preprocess_data
# utils.py
#     num_one = round(self.atac_len * self.dec_atac_one_rate)
# ->  num_one = round(self.atac_len * 0.5)
## ./data_preprocess.sh
for s in `cat normal_samples.txt`;do
    python data_preprocess.py -r ./without_tumor_data/$s\_rna.h5ad -a ./without_tumor_data/$s\_atac.h5ad \
                              -s ./preprocess_data -n $s --dt train --config rna2atac_config_train.yaml
    echo $s done
done


## train model
# rna2atac_train.py
# "from utils import *" -> "from M2Mmodel.utils import *"
# "from M2M import M2M_rna2atac"  ->  "from M2Mmodel.M2M import M2M_rna2atac"
nohup accelerate launch --config_file accelerator_config_train.yaml --main_process_port 29823 rna2atac_train.py --config_file rna2atac_config_train.yaml \
                        --train_data_dir ./preprocess_data_train --val_data_dir ./preprocess_data_val -n rna2atac_train > 20240924.log &


## Fibroblasts metacell
# conda activate copykat
library(Seurat)
library(Matrix)

dat <- readRDS('Merge_minicluster30_Fibroblast_Seurat_CCA_addMeta.rds')
meta <- readRDS('SeuratObj_meta_info.rds')

# output barcodes
write.table(data.frame(colnames(dat@assays$RNA@counts)), 'barcodes.tsv', quote=FALSE, col.names=FALSE, row.names=FALSE)

# output features
ft <- data.frame(v1=rownames(dat@assays$RNA@counts),
                 v2=rownames(dat@assays$RNA@counts),
                 v3='Gene Expression')
write.table(ft, 'features.tsv', sep='\t', quote=FALSE, col.names=FALSE, row.names=FALSE)

# output mtx
m_rounded <- round(dat@assays$RNA@counts)
m_tri <- summary(m_rounded)
m_tri_filtered <- m_tri[m_tri$x!=0, ]
m_out <- sparseMatrix(i=m_tri_filtered$i, j=m_tri_filtered$j, x=m_tri_filtered$x, dims=dim(m_rounded))
writeMM(m_out, file='matrix.mtx')

# output cell_anno
write.table(data.frame(v1=rownames(meta), v2=meta$curated_anno), 'cell_anno.txt', sep='\t', quote=FALSE, col.names=FALSE, row.names=FALSE)


## RNA alignment
import scanpy as sc
import pandas as pd
from scipy.sparse import csr_matrix

rna = sc.read_10x_mtx('.')
rna.X = rna.X.toarray()
genes = pd.read_table('../without_tumor_data/human_genes.txt', names=['gene_id', 'gene_name'])

rna_genes = rna.var.copy()
rna_exp = pd.concat([rna_genes, pd.DataFrame(rna.X.T, index=rna_genes.index)], axis=1)

X_new = pd.merge(genes, rna_exp, how='left', left_on='gene_name', right_on='gene_ids').iloc[:, 4:].T
X_new.fillna(value=0, inplace=True)

rna_new = sc.AnnData(X_new.values, obs=rna.obs, var=pd.DataFrame({'gene_ids': genes['gene_id'], 'feature_types': 'Gene Expression'}))
rna_new.var.index = genes['gene_name'].values
rna_new.X = csr_matrix(rna_new.X)
rna_new.write('fibroblast_rna.h5ad')


## pseudo ATAC
import scanpy as sc
import numpy as np
from scipy.sparse import csr_matrix

atac_ref = sc.read_h5ad('without_tumor_data/CE337E1-S1_atac.h5ad')
rna = sc.read_h5ad('fibroblast/fibroblast_rna.h5ad')
m = np.zeros([rna.n_obs, atac_ref.n_vars])
m[:,0]=1
atac = sc.AnnData(csr_matrix(m), obs=rna.obs, var=atac_ref.var)
atac.write('fibroblast/fibroblast_pseudo_atac.h5ad')


## prediction
## max enc length
import scanpy as sc
import numpy as np

rna = sc.read_h5ad('fibroblast/fibroblast_rna.h5ad')
np.count_nonzero(rna.X.toarray(), axis=1).max() # 4495

python data_preprocess.py -r fibroblast/fibroblast_rna.h5ad -a fibroblast/fibroblast_pseudo_atac.h5ad -s preprocess_data_test --dt test --config rna2atac_config_test.yaml  # ~1.5h

# rna2atac_test.py
# "from utils import *" -> "from M2Mmodel.utils import *"
# "from M2M import M2M_rna2atac"  ->  "from M2Mmodel.M2M import M2M_rna2atac"

for i in `seq 0 7`;do
    accelerate launch --config_file accelerator_config_test.yaml --main_process_port 29822 rna2atac_test.py -d ./preprocess_data_test/$i \
                      -l save/2024-09-24_rna2atac_train_3/pytorch_model.bin --config_file rna2atac_config_test.yaml
    mv predict.npy predict_$i.npy
done


## plot umap
import snapatac2 as snap
import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
import scanpy as sc
import pandas as pd
from tqdm import tqdm

pred = snap.read('fibroblast_pseudo_atac.h5ad', backed=None)
for i in tqdm(range(8), ncols=80):
    m = np.load('../predict_'+str(i)+'.npy')
    m[m>0.99]=1
    m[m<=0.99]=0
    pred.X[(1000*i):(1000*i+1000), :] = lil_matrix(m).tocsr()  # ~6 min

cell_anno = pd.read_table('cell_anno.txt', header=None)
pred.obs['cell_anno'] = cell_anno[1].values

snap.pp.select_features(pred)
snap.tl.spectral(pred)
snap.tl.umap(pred)
sc.pl.umap(pred, color='cell_anno', legend_fontsize='7', legend_loc='right margin', size=10,
           title='', frameon=True, save='_atac_predict_2.pdf')
