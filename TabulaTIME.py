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







