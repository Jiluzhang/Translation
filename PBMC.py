## pbmc dataset

## alignment -> filtering -> clustering -> modeling -> evaluation

################ raw data process ################
# human_genes.txt  38244
# human_cCREs.bed  1033239
# python align_filter_rna_atac_10x.py -i pbmc_granulocyte_sorted_10k_filtered_feature_bc_matrix.h5
import argparse
import warnings
import scanpy as sc
import pandas as pd
import numpy as np
import pybedtools
from tqdm import tqdm
from scipy.sparse import csr_matrix

warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(description='Alignment for h5 file from 10x')
parser.add_argument('-i', '--input', type=str, help='h5 file')

args = parser.parse_args()
h5_file = args.input

dat = sc.read_10x_h5(h5_file, gex_only=False)
dat.var_names_make_unique()

## RNA
rna = dat[:, dat.var['feature_types']=='Gene Expression'].copy()
rna.X = rna.X.toarray()
genes = pd.read_table('human_genes.txt', names=['gene_ids', 'gene_name'])

rna_exp = pd.concat([rna.var, pd.DataFrame(rna.X.T, index=rna.var.index)], axis=1)
X_new = pd.merge(genes, rna_exp, how='left', on='gene_ids').iloc[:, 4:].T
X_new.fillna(value=0, inplace=True)

rna_new = sc.AnnData(X_new.values, obs=rna.obs, var=pd.DataFrame({'gene_ids': genes['gene_ids'], 'feature_types': 'Gene Expression'}))
rna_new.var.index = genes['gene_name'].values
rna_new.X = csr_matrix(rna_new.X)
rna_new.write('rna_aligned.h5ad')
print('######## RNA alignment done ########')

## ATAC
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
atac_new.write('atac_aligned.h5ad')
print('######## ATAC alignment done ########')

## filter genes & peaks & cells
sc.pp.filter_genes(rna_new, min_cells=10)
sc.pp.filter_cells(rna_new, min_genes=500)
sc.pp.filter_cells(rna_new, max_genes=5000)

sc.pp.filter_genes(atac_new, min_cells=10)
sc.pp.filter_cells(atac_new, min_genes=1000)
sc.pp.filter_cells(atac_new, max_genes=50000)

idx = np.intersect1d(rna_new.obs.index, atac_new.obs.index)
rna_new[idx, :].copy().write('rna.h5ad')
atac_new[idx, :].copy().write('atac.h5ad')
print(f'Select {rna_new.n_obs} / {rna.n_obs} cells')
print(f'Select {rna_new.n_vars} / 38244 genes')
print(f'Select {atac_new.n_vars} / 1033239 peaks')


######################## cell annotation ########################
# https://scanpy.readthedocs.io/en/stable/tutorials/basics/clustering-2017.html
import scanpy as sc
import pandas as pd

rna = sc.read_h5ad('rna.h5ad')
sc.pp.normalize_total(rna, target_sum=1e4)
sc.pp.log1p(rna)
sc.pp.highly_variable_genes(rna)
rna.raw = rna
rna = rna[:, rna.var.highly_variable]
sc.tl.pca(rna)
sc.pp.neighbors(rna)
sc.tl.umap(rna, min_dist=1.2)
sc.tl.leiden(rna, resolution=0.8)
sc.pl.umap(rna, color='leiden', legend_fontsize='5', legend_loc='on data',
           size=8, title='', frameon=True, save='_rna.pdf')

sc.tl.rank_genes_groups(rna, 'leiden', method='wilcoxon')
pd.DataFrame(rna.uns['rank_genes_groups']['names']).head(20)
#            0       1        2             3        4       5          6         7       8        9      10       11        12         13         14          15       16
# 0     INPP4B    FHIT     LEF1          VCAN   SLC8A1    CCL5      BANK1    TCF7L2    GNLY      LYZ  PLXDC2     NKG7      CST3      BANK1       TCF4     OSBPL10     SOX4
# 1        LTB    LEF1    PDE3B        PLXDC2     AOAH    NKG7    RALGPS2     MTSS1    PRF1     FCN1   DMXL2     CCL5      CD74      MS4A1       RHEX     RALGPS2   NKAIN2
# 2       IL32  BCL11B    NELL2          DPYD    NEAT1    GZMA       AFF3      LST1    NKG7     MNDA  SLC8A1     PRF1  HLA-DPB1    RALGPS2       ZFAT       CD79A    ZNRF1
# 3      ITGB1   CAMK4   THEMIS      ARHGAP26     TYMP    IL32      MS4A1    FCGR3A   CD247     CTSS   LRMDA     GZMA  HLA-DPA1       AFF3       IRF8       FCRL1     CDK6
# 4     CDC14A  MALAT1    BACH2         CSF3R     JAK2     A2M       CD74     COTL1   KLRD1    LRMDA   NAMPT     RORA  HLA-DRB1       PAX5     FCHSD2       TPD52   RNF220
# 5       ANK3   BACH2     CD8B         LRMDA    NAMPT   SYNE1       PAX5      PSAP    GZMA     VCAN  TBXAS1     GNLY   HLA-DRA      FCRL1       UGCG       MEF2C      ERG
# 6      SYNE2    TCF7   OXNAD1      MIR23AHG  TNFAIP2   SYNE2      FCRL1       LYN   MCTP2     DPYD   RBM47     CTSW  HLA-DQA1      CD79A      EPHB1        GNG7   ZNF521
# 7       IL7R    ANK3      TXK         NAMPT     DPYD   HLA-B      CD79A      AIF1   SPON2     CD36   MCTP1      A2M  HLA-DQB1      CD79B       PLD4        PAX5  ST8SIA6
# 8     TTC39C  INPP4B    CAMK4         RBM47   PLXDC2   PRKCH    HLA-DRA    IFITM3    CST7     JAK2   IRAK3    KLRB1     HDAC9       IGHM  LINC01374       CDK14  HNRNPA1
# 9     EEF1A1  RPS27A     CCR7         LRRK2    MCTP1    GNLY       EBF1     WARS1   SYNE1     TYMP   NEAT1    SYNE2  HLA-DRB5        BLK      PTPRS      SEL1L3     MSI2
# 10      RORA    CCR7   BCL11B         NEAT1  DENND1A    GZMH  LINC00926      CST3    CTSW   PLXDC2    FCN1    NCALD      ACTB  LINC00926     CCDC50       BANK1    AUTS2
# 11  ARHGAP15   PRKCA    APBA2          FOSB     PID1   HLA-C       IGHM    CDKN1C   KLRF1     TLR2   WARS1    ARL4C    SAMHD1       EBF1     BCL11A       MS4A1     MAP7
# 12     RPL41   MAML2    MAML2         IRAK3     FCN1   SAMD3    OSBPL10      CTSS  PYHIN1     ZEB2    GAB2   ZBTB16       LYZ     SEL1L3       CUX2      SETBP1    STMN1
# 13     RPL13   RPS25  SERINC5         ANXA1     CD83    PRF1        BLK  SERPINA1    PFN1     LYST    JAK2    KLRG1    CCSER1    OSBPL10     PLXNA4        IGKC    RPLP0
# 14      TPT1   RPS27   NDFIP1  ADAMTSL4-AS1     ZEB2  TGFBR3      CD79B     MS4A7  TGFBR3    NEAT1   LRRK2     TC2N   CCDC88A       CD22      RUNX2        TCF4    PPM1H
# 15      TRAC   RPL11     TCF7          FCN1    DMXL2   KLRG1   HLA-DRB1     IFI30    GZMB  CLEC12A    VCAN  SLC4A10      AFF3       CD74        APP      FCHSD2   SNHG29
# 16     RPS18   RPS12   ABLIM1          GAB2      FOS    CST7   HLA-DQA1    FCER1G   IL2RB   SLC8A1    TLR2    SKAP1       VIM     ADAM28  LINC00996        EBF1    MEIS1
# 17     RPS27  OXNAD1      ITK       DENND1A   PARP14    CTSW       CD22  TNFRSF1B    CCL5    RBM47    LYST     IL32       GRN     NIBAN3   RABGAP1L      ADAM28   CASC15
# 18     RPL34   RPS14    FOXP1        FNDC3B    SCLT1     PZP     SEL1L3      UTRN  ZBTB16   S100A9    CD36   TGFBR3     ALCAM      TPD52  LINC01478        AFF3    CALN1
# 19     MDFIC    RPL3     CD8A     RAB11FIP1      LYZ    PFN1       CD37    TYROBP   SAMD3   S100A8    TYMP      PZP     ANXA2       IGHD      ITM2C  PALM2AKAP2    SNHG7

# cluster_0:INPP4B,LTB,IL32,ITGB1,CDC14A,ANK3,SYNE2,IL7R,TTC39C,EEF1A1,RORA,ARHGAP15,RPL41,RPL13,TPT1,TRAC,RPS18,RPS27,RPL34,MDFIC
# cluster_1:FHIT,LEF1,BCL11B,CAMK4,MALAT1,BACH2,TCF7,ANK3,INPP4B,RPS27A,CCR7,PRKCA,MAML2,RPS25,RPS27,RPL11,RPS12,OXNAD1,RPS14,RPL3
# cluster_2:LEF1,PDE3B,NELL2,THEMIS,BACH2,CD8B,OXNAD1,TXK,CAMK4,CCR7,BCL11B,APBA2,MAML2,SERINC5,NDFIP1,TCF7,ABLIM1,ITK,FOXP1,CD8A
# cluster_3:VCAN,PLXDC2,DPYD,ARHGAP26,CSF3R,LRMDA,MIR23AHG,NAMPT,RBM47,LRRK2,NEAT1,FOSB,IRAK3,ANXA1,ADAMTSL4-AS1,FCN1,GAB2,DENND1A,FNDC3B,RAB11FIP1
# cluster_4:SLC8A1,AOAH,NEAT1,TYMP,JAK2,NAMPT,TNFAIP2,DPYD,PLXDC2,MCTP1,DENND1A,PID1,FCN1,CD83,ZEB2,DMXL2,FOS,PARP14,SCLT1,LYZ
# cluster_5:CCL5,NKG7,GZMA,IL32,A2M,SYNE1,SYNE2,HLA-B,PRKCH,GNLY,GZMH,HLA-C,SAMD3,PRF1,TGFBR3,KLRG1,CST7,CTSW,PZP,PFN1
# cluster_6:BANK1,RALGPS2,AFF3,MS4A1,CD74,PAX5,FCRL1,CD79A,HLA-DRA,EBF1,LINC00926,IGHM,OSBPL10,BLK,CD79B,HLA-DRB1,HLA-DQA1,CD22,SEL1L3,CD37
# cluster_7:TCF7L2,MTSS1,LST1,FCGR3A,COTL1,PSAP,LYN,AIF1,IFITM3,WARS1,CST3,CDKN1C,CTSS,SERPINA1,MS4A7,IFI30,FCER1G,TNFRSF1B,UTRN,TYROBP
# cluster_8:GNLY,PRF1,NKG7,CD247,KLRD1,GZMA,MCTP2,SPON2,CST7,SYNE1,CTSW,KLRF1,PYHIN1,PFN1,TGFBR3,GZMB,IL2RB,CCL5,ZBTB16,SAMD3
# cluster_9:LYZ,FCN1,MNDA,CTSS,LRMDA,VCAN,DPYD,CD36,JAK2,TYMP,PLXDC2,TLR2,ZEB2,LYST,NEAT1,CLEC12A,SLC8A1,RBM47,S100A9,S100A8
# cluster_10:PLXDC2,DMXL2,SLC8A1,LRMDA,NAMPT,TBXAS1,RBM47,MCTP1,IRAK3,NEAT1,FCN1,WARS1,GAB2,JAK2,LRRK2,VCAN,TLR2,LYST,CD36,TYMP
# cluster_11:NKG7,CCL5,PRF1,GZMA,RORA,GNLY,CTSW,A2M,KLRB1,SYNE2,NCALD,ARL4C,ZBTB16,KLRG1,TC2N,SLC4A10,SKAP1,IL32,TGFBR3,PZP
# cluster_12:CST3,CD74,HLA-DPB1,HLA-DPA1,HLA-DRB1,HLA-DRA,HLA-DQA1,HLA-DQB1,HDAC9,HLA-DRB5,ACTB,SAMHD1,LYZ,CCSER1,CCDC88A,AFF3,VIM,GRN,ALCAM,ANXA2
# cluster_13:BANK1,MS4A1,RALGPS2,AFF3,PAX5,FCRL1,CD79A,CD79B,IGHM,BLK,LINC00926,EBF1,SEL1L3,OSBPL10,CD22,CD74,ADAM28,NIBAN3,TPD52,IGHD
# cluster_14:TCF4,RHEX,ZFAT,IRF8,FCHSD2,UGCG,EPHB1,PLD4,LINC01374,PTPRS,CCDC50,BCL11A,CUX2,PLXNA4,RUNX2,APP,LINC00996,RABGAP1L,LINC01478,ITM2C
# cluster_15:OSBPL10,RALGPS2,CD79A,FCRL1,TPD52,MEF2C,GNG7,PAX5,CDK14,SEL1L3,BANK1,MS4A1,SETBP1,IGKC,TCF4,FCHSD2,EBF1,ADAM28,AFF3,PALM2AKAP2
# cluster_16:SOX4,NKAIN2,ZNRF1,CDK6,RNF220,ERG,ZNF521,ST8SIA6,HNRNPA1,MSI2,AUTS2,MAP7,STMN1,RPLP0,PPM1H,SNHG29,MEIS1,CASC15,CALN1,SNHG7

# cell type annotation tool: http://xteam.xbio.top/ACT
# cluster_0: Central memory CD4-positive, alpha-beta T cell
# cluster_1: Naive thymus-derived CD4-positive, alpha-beta T cell
# cluster_2: Naive thymus-derived CD8-positive, alpha-beta T cell
# cluster_3: Classical monocyte
# cluster_4: Classical monocyte
# cluster_5: CD8-positive, alpha-beta cytotoxic T cell
# cluster_6: Memory B cell
# cluster_7: Non-classical monocyte
# cluster_8: Natural killer cell
# cluster_9: Classical monocyte
# cluster_10: Classical monocyte
# cluster_11: CD8-positive, alpha-beta cytotoxic T cell
# cluster_12: Dendritic cell
# cluster_13: Memory B cell
# cluster_14: Plasmacytoid dendritic cell, human
# cluster_15: B cell
# cluster_16: Megakaryocyte-erythroid progenitor cell

# https://www.nature.com/articles/s41467-021-25771-5/figures/1
cell_anno_dict = {'0':'CD4 Memory',      '1':'CD4 Naive',      '2':'CD8 Naive', '3':'CD14 Mono', '4':'CD14 Mono',
                  '5':'CD8 Effector',    '6':'Naive B',        '7':'CD16 Mono', '8':'NK',        '9':'CD14 Mono',
                  '10':'CD14 Mono',      '11':'CD8 Effector',  '12':'DC',       '13':'Memory B', '14':'pDC',
                  '15':'Plasma',         '16':'Mega'}
rna.obs['cell_anno'] = rna.obs['leiden']
rna.obs['cell_anno'].replace(cell_anno_dict, inplace=True)

sc.pl.umap(rna, color='cell_anno', legend_fontsize='7', legend_loc='on data', size=5,
           title='', frameon=True, save='_cell_anno.pdf')

rna.write('rna_cell_anno.h5ad')

## ATAC clustering
# /fs/home/jiluzhang/softwares/miniconda3/envs/snapatac2/lib/python3.8/site-packages/snapatac2/tools/_embedding.py  min_dist=0.01
import snapatac2 as snap
import scanpy as sc

atac = snap.read('atac.h5ad', backed=None)
rna = snap.read('rna_cell_anno.h5ad', backed=None)
atac.obs['cell_anno'] = rna.obs['cell_anno']
snap.pp.select_features(atac, n_features=50000)
snap.tl.spectral(atac)
snap.tl.umap(atac)
sc.pl.umap(atac, color='cell_anno', legend_fontsize='7', legend_loc='on data', size=5,
           title='', frameon=True, save='_atac_50000.pdf')
atac.write('atac_cell_anno.h5ad')

# pip install episcanpy
# import episcanpy as epi
# import scanpy as sc
# atac = sc.read_h5ad('atac.h5ad')
# rna = sc.read_h5ad('rna_cell_anno.h5ad', backed=None)
# atac.obs['cell_anno'] = rna.obs['cell_anno']

########################################################
# cell: 11516 (train: 8061   valid: 1151  test:2304)
# rna: 17,295
# atac: 236,316
########################################################

# train: 5min   valid: 4min  multiple_rate: 20

python split_train_val_test.py --RNA rna.h5ad --ATAC atac.h5ad --train_pct 0.7 --valid_pct 0.1
python rna2atac_data_preprocess.py --config_file rna2atac_config.yaml --dataset_type train
python rna2atac_data_preprocess.py --config_file rna2atac_config_whole.yaml --dataset_type val
accelerate launch --config_file accelerator_config.yaml --main_process_port 29821 rna2atac_pretrain.py --config_file rna2atac_config.yaml \
                  --train_data_dir ./preprocessed_data_train --val_data_dir ./preprocessed_data_val -n rna2atac_train
accelerate launch --config_file accelerator_config_eval.yaml --main_process_port 29822 rna2atac_evaluate.py -d ./preprocessed_data_val \
                  -l save/pytorch_model_epoch_5.bin --config_file rna2atac_config_whole.yaml && mv predict.npy val_predict.npy
accelerate launch --config_file accelerator_config_eval.yaml --main_process_port 29822 rna2atac_evaluate.py -d ./preprocessed_data_test \
                  -l save/pytorch_model_epoch_1.bin --config_file rna2atac_config_whole.yaml && mv predict.npy test_predict.npy


import snapatac2 as snap
import numpy as np
from scipy.sparse import csr_matrix
import scanpy as sc

m_raw = np.load('val_predict_epoch_43.npy')
m = m_raw.copy()
# [sum(m_raw[i]>0.7) for i in range(5)]
m[m>0.7]=1
m[m<=0.7]=0

atac= snap.read('data/atac_cell_anno.h5ad', backed=None)
atac_val = snap.read('atac_val_0.h5ad', backed=None)
atac_true = atac[atac_val.obs.index, :]
atac_pred = atac_true.copy()
atac_pred.X = csr_matrix(m)

snap.pp.select_features(atac_pred)
snap.tl.spectral(atac_pred)
snap.tl.umap(atac_pred)
# snap.pl.umap(atac_pred, color='leiden', show=False, out_file='umap_val_predict.pdf', marker_size=2.0, height=500)
# sc.pl.umap(atac_pred, color='cell_anno', legend_fontsize='7', legend_loc='on data', size=5,
#            title='', frameon=True, save='_atac_val_predict.pdf')
sc.pl.umap(atac_pred, color='cell_anno', legend_fontsize='7', legend_loc='right margin', size=5,
           title='', frameon=True, save='_atac_val_predict_epoch_43.pdf')

sc.pl.umap(atac_pred[atac_pred.obs.cell_anno.isin(['CD14 Mono', 'CD16 Mono']), :], color='cell_anno', legend_fontsize='7', legend_loc='right margin', size=5,
           title='', frameon=True, save='_atac_val_predict_epoch_43_tmp.pdf', groups=['CD14 Mono', 'CD16 Mono'], 
           palette={'CD14 Mono': 'red', 'CD16 Mono': 'blue'})



############ scButterfly ############
## Github: https://github.com/BioX-NKU/scButterfly
## Tutorial: https://scbutterfly.readthedocs.io/en/latest/
conda create -n scButterfly python==3.9
conda activate scButterfly
pip install scButterfly -i https://pypi.tuna.tsinghua.edu.cn/simple
conda install pytorch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 pytorch-cuda=12.1 -c pytorch -c nvidia

from scButterfly.butterfly import Butterfly
from scButterfly.split_datasets import *
import scanpy as sc
import anndata as ad

butterfly = Butterfly()

RNA_data_train = sc.read_h5ad('rna_train_0.h5ad')
RNA_data_val   = sc.read_h5ad('rna_val_0.h5ad')
RNA_data = ad.concat([RNA_data_train, RNA_data_val])
RNA_data.var = RNA_data_train.var[['gene_ids', 'feature_types']]

ATAC_data_train = sc.read_h5ad('atac_train_0.h5ad')
ATAC_data_val   = sc.read_h5ad('atac_val_0.h5ad')
ATAC_data = ad.concat([ATAC_data_train, ATAC_data_val])
ATAC_data.var = ATAC_data_train.var[['gene_ids', 'feature_types']]
