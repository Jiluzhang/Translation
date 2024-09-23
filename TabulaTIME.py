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


## filter tumor cells
import scanpy as sc
import pandas as pd
from tqdm import tqdm

samples = pd.read_table('../rds_samples_id.txt', header=None)
no_tumor_cnt = []

for i in tqdm(range(samples.shape[0]), ncols=80):
    s_2= samples[1][i]
    dat = sc.read_h5ad(s_2+'_filtered.h5ad')
    dat = dat[dat.obs['cell_anno']!='Tumor']
    dat.write('../without_tumor_data/'+s_2+'_no_tumor.h5ad')
    no_tumor_cnt.append(dat.n_obs)

no_tumor_cnt
# [29,  544,   18, 1765,  731,   58,  883,   83,  248,  386,  121, 284,  214,   88, 1451, 6407, 2609,  733, 3902, 4518,  544, 1246,
#  1184,  538,  363, 1718,  373, 1714, 6240,  875, 1599, 2529, 2029, 66,  828, 4394, 4710,  458, 2470, 2862, 3517, 2217, 4686,  151,
#  2048, 3649,  703, 5571,  104,  240, 7471, 1056, 1069, 1553, 2250, 4968, 2636, 1036, 4431,  475,  365, 2770,  336,  253, 2734, 1542,
#  3074, 2121,  771,  137, 1248,  843,  197,  116,  382,  292,  948, 433,   59, 1502,  158, 1677, 2358,  206, 1149,  352, 1888, 2396,
#  318, 3141, 2892,  176, 4476,  370, 1115,  143, 2470,  561, 1327, 1016,  472, 3068,  207, 1480, 1661,  245, 3572,  970, 1087, 1423,
#  430,  845, 1038, 1551,   58,  394,  515,  565, 1096, 3687, 2602, 70,   69, 2379,  754, 2076,  205,  533,  392]

sum(no_tumor_cnt)  # 193,399


## remove normal samples & GBM (two GBM samples are both normal)
# CPT704DU-M1	    UCEC
# GBML018G1-M1N2	GBM
# GBML019G1-M1N1	GBM
# HT291C1-M1A3	    CRC
# SN001H1-Ms1	    SKCM
# SP369H1-Mc1	    CRC
# SP819H1-Mc1	    CRC

# rm CPT704DU-M1_no_tumor.h5ad GBML018G1-M1_no_tumor.h5ad GBML019G1-M1_no_tumor.h5ad
# rm HT291C1-M1A3_no_tumor.h5ad SN001H1-Ms1A3_no_tumor.h5ad SP369H1-Mc1_no_tumor.h5ad SP819H1-Mc1_no_tumor.h5ad







