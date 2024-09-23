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











