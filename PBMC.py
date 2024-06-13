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
python rna2atac_data_preprocess.py --config_file rna2atac_config_whole.yaml --dataset_type test  # ~15 min
accelerate launch --config_file accelerator_config.yaml --main_process_port 29821 rna2atac_pretrain.py --config_file rna2atac_config.yaml \
                  --train_data_dir ./preprocessed_data_train --val_data_dir ./preprocessed_data_val -n rna2atac_train
accelerate launch --config_file accelerator_config_eval.yaml --main_process_port 29822 rna2atac_evaluate.py -d ./preprocessed_data_test \
                  -l save/rna2atac_train/pytorch_model.bin --config_file rna2atac_config_whole.yaml && mv predict.npy test_predict.npy   # 1 min per 400 cells

accelerate launch --config_file accelerator_config_eval.yaml --main_process_port 29822 rna2atac_evaluate.py -d ./preprocessed_data_test \
                  -l save/pytorch_model_epoch_12.bin --config_file rna2atac_config_whole.yaml
mv predict.npy test_predict.npy

# 3.5 min per 100 cells with 3 gpu and batch_size of 4

# depth: 6  heads: 6  parameters: 18,893,313


nohup accelerate launch --config_file accelerator_config.yaml --main_process_port 29821 rna2atac_pretrain.py --config_file rna2atac_config.yaml \
                  --train_data_dir ./preprocessed_data_train --val_data_dir ./preprocessed_data_val -n rna2atac_train > 20240607.log &



# 3423220
# train_batch_size: 16  val_batch_size=8
# validation dataset random 10,000 peaks
# 10,000 peaks per prediction for testing dataset
# watch -n 1 -d nvidia-smi


nohup accelerate launch --config_file accelerator_config.yaml --main_process_port 29821 rna2atac_pretrain.py --config_file rna2atac_config.yaml \
                        --train_data_dir ./preprocessed_data_train --val_data_dir ./preprocessed_data_val -n rna2atac_train > 20240606.log &
# 587379

accelerate launch --config_file accelerator_config_eval.yaml --main_process_port 29822 rna2atac_evaluate.py -d ./preprocessed_data_test \
                  -l save/rna2atac_train/pytorch_model.bin --config_file rna2atac_config_whole.yaml
mv predict.npy test_predict.npy

python npy2h5ad.py
mv rna2atac_scm2m.h5ad benchmark
python cal_auroc_auprc.py --pred rna2atac_scm2m.h5ad --true rna2atac_true.h5ad
python cal_cluster_plot.py --pred rna2atac_scm2m.h5ad --true rna2atac_true.h5ad


# modify rna2atac_config.yaml
python data_preprocess.py -r rna_train_0.h5ad -a atac_train_0.h5ad -s preprocessed_data_train
python data_preprocess.py -r rna_val_0.h5ad -a atac_val_0.h5ad -s preprocessed_data_val 
python data_preprocess.py -r rna_test_0.h5ad -a atac_test_0.h5ad -s preprocessed_data_test 

accelerate launch --config_file accelerator_config.yaml --main_process_port 29822 rna2atac_evaluate.py -d ./preprocessed_data_test \
                  -l save/2024-06-12_rna2atac_train_34/pytorch_model.bin --config_file rna2atac_config_whole.yaml


## npy -> h5ad
# python npy2h5ad.py
import scanpy as sc
import numpy as np

dat = np.load('test_predict.npy')
true = sc.read_h5ad('atac_test_0.h5ad')
pred = true.copy()
pred.X = dat
del pred.obs['n_genes']
del pred.var['n_cells']
pred.write('rna2atac_scm2m.h5ad')



import snapatac2 as snap
import numpy as np
from scipy.sparse import csr_matrix
import scanpy as sc
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score, normalized_mutual_info_score, homogeneity_score

m_raw = np.load('predict.npy')
m = m_raw.copy()
# [sum(m_raw[i]>0.7) for i in range(5)]
m[m>0.7]=1
m[m<=0.7]=0

atac= snap.read('data/atac_cell_anno.h5ad', backed=None)
atac_val = snap.read('atac_test_0.h5ad', backed=None)
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
           title='', frameon=True, save='_atac_val_predict.pdf')

# sc.pl.umap(atac_pred[atac_pred.obs.cell_anno.isin(['CD14 Mono', 'CD16 Mono']), :], color='cell_anno', legend_fontsize='7', legend_loc='right margin', size=5,
#            title='', frameon=True, save='_atac_val_predict_epoch_43_tmp.pdf', groups=['CD14 Mono', 'CD16 Mono'], 
#            palette={'CD14 Mono': 'red', 'CD16 Mono': 'blue'})

snap.pp.knn(atac_pred)
snap.tl.leiden(atac_pred)
AMI = adjusted_mutual_info_score(atac_pred.obs['cell_anno'], atac_pred.obs['leiden'])
ARI = adjusted_rand_score(atac_pred.obs['cell_anno'], atac_pred.obs['leiden'])
HOM = homogeneity_score(atac_pred.obs['cell_anno'], atac_pred.obs['leiden'])
NMI = normalized_mutual_info_score(atac_pred.obs['cell_anno'], atac_pred.obs['leiden'])
print(round(AMI, 4), round(ARI, 4), round(HOM, 4), round(NMI, 4))


## plot testing true
import snapatac2 as snap
import scanpy as sc

atac= snap.read('../data/atac_cell_anno.h5ad', backed=None)
atac_test = snap.read('rna2atac_true.h5ad', backed=None)
atac_true = atac[atac_test.obs.index, :]

snap.pp.select_features(atac_true)
snap.tl.spectral(atac_true)
snap.tl.umap(atac_true)
sc.pl.umap(atac_true, color='cell_anno', legend_fontsize='7', legend_loc='right margin', size=5,
           title='', frameon=True, save='_true.pdf')


##################################################################
auroc: 0.8385     auprc: 0.4052
depth_1_head_1_epoch_26: 0.0342 0.0336 0.0598 0.065

auroc: 0.8598     auprc: 0.4437
depth_2_head_2_epoch_22: 0.0331 0.0368 0.0595 0.0638
##################################################################


m_raw = np.load('test_predict.npy')
m = m_raw.copy()
# [sum(m_raw[i]>0.7) for i in range(5)]
m[m>0.7]=1
m[m<=0.7]=0

atac= snap.read('data/atac_cell_anno.h5ad', backed=None)
atac_test = snap.read('atac_test_0.h5ad', backed=None)
atac_true = atac[atac_test.obs.index, :]
atac_pred = atac_true.copy()
atac_pred.X = csr_matrix(m)





# earlystop based on AUPRC of validation dataset
# "max_epoch" set to 100
# 2207721


############ scButterfly ############
## Github: https://github.com/BioX-NKU/scButterfly
## Tutorial: https://scbutterfly.readthedocs.io/en/latest/
## https://scbutterfly.readthedocs.io/en/latest/Tutorial/RNA_ATAC_paired_prediction/RNA_ATAC_paired_scButterfly-C.html
conda create -n scButterfly python==3.9
conda activate scButterfly
pip install scButterfly -i https://pypi.tuna.tsinghua.edu.cn/simple
conda install pytorch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install jaxlib==0.4.27

# conda create -n scButterfly_clone --clone /fs/home/jiluzhang/softwares/miniconda3/envs/scButterfly

# train_model.py
# mark following code
'''
else:
    my_logger.info('calculate neighbors graph for following test ...')
    sc.pp.pca(A2R_predict)
    sc.pp.neighbors(A2R_predict)
    sc.pp.pca(R2A_predict)
    sc.pp.neighbors(R2A_predict)
'''

## python scbt_c.py
import os
from scButterfly.butterfly import Butterfly
from scButterfly.split_datasets import *
import scanpy as sc
import anndata as ad
import random

os.environ["CUDA_VISIBLE_DEVICES"] = "5"

RNA_data = sc.read_h5ad('rna.h5ad')
ATAC_data = sc.read_h5ad('atac.h5ad')

random.seed(0)
idx = list(range(RNA_data.n_obs))
random.shuffle(idx)
train_id = idx[:int(len(idx)*0.7)]
validation_id = idx[int(len(idx)*0.7):(int(len(idx)*0.7)+int(len(idx)*0.1))]
test_id = idx[(int(len(idx)*0.7)+int(len(idx)*0.1)):]

butterfly = Butterfly()
butterfly.load_data(RNA_data, ATAC_data, train_id, test_id, validation_id)
butterfly.data_preprocessing(normalize_total=False, log1p=False, use_hvg=False, n_top_genes=None, binary_data=False, 
                             filter_features=False, fpeaks=None, tfidf=False, normalize=False) 

butterfly.ATAC_data_p.var['chrom'] = butterfly.ATAC_data_p.var['gene_ids'].map(lambda x: x.split(':')[0])
chrom_list = []
last_one = ''
for i in range(len(butterfly.ATAC_data_p.var.chrom)):
    temp = butterfly.ATAC_data_p.var.chrom[i]
    if temp[0 : 3] == 'chr':
        if not temp == last_one:
            chrom_list.append(1)
            last_one = temp
        else:
            chrom_list[-1] += 1
    else:
        chrom_list[-1] += 1

butterfly.augmentation(aug_type='MultiVI_augmentation')  # butterfly.augmentation(aug_type=None)
butterfly.construct_model(chrom_list=chrom_list)
butterfly.train_model(batch_size=16)
A2R_predict, R2A_predict = butterfly.test_model(test_cluster=False, test_figure=False, output_data=True)
# A2R_predict, R2A_predict = butterfly.test_model(model_path='.', load_model=False, test_cluster=False, test_figure=False, output_data=False)


## python scbt_b.py
import os
from scButterfly.butterfly import Butterfly
from scButterfly.split_datasets import *
import scanpy as sc
import anndata as ad
import random

os.environ["CUDA_VISIBLE_DEVICES"] = "4"

RNA_data = sc.read_h5ad('rna.h5ad')
ATAC_data = sc.read_h5ad('atac.h5ad')

random.seed(0)
idx = list(range(RNA_data.n_obs))
random.shuffle(idx)
train_id = idx[:int(len(idx)*0.7)]
validation_id = idx[int(len(idx)*0.7):(int(len(idx)*0.7)+int(len(idx)*0.1))]
test_id = idx[(int(len(idx)*0.7)+int(len(idx)*0.1)):]

butterfly = Butterfly()
butterfly.load_data(RNA_data, ATAC_data, train_id, test_id, validation_id)
butterfly.data_preprocessing(normalize_total=False, log1p=False, use_hvg=False, n_top_genes=None, binary_data=False, 
                             filter_features=False, fpeaks=None, tfidf=False, normalize=False) 

butterfly.ATAC_data_p.var['chrom'] = butterfly.ATAC_data_p.var['gene_ids'].map(lambda x: x.split(':')[0])
chrom_list = []
last_one = ''
for i in range(len(butterfly.ATAC_data_p.var.chrom)):
    temp = butterfly.ATAC_data_p.var.chrom[i]
    if temp[0 : 3] == 'chr':
        if not temp == last_one:
            chrom_list.append(1)
            last_one = temp
        else:
            chrom_list[-1] += 1
    else:
        chrom_list[-1] += 1

butterfly.augmentation(aug_type=None)
butterfly.construct_model(chrom_list=chrom_list)
butterfly.train_model(batch_size=16)
A2R_predict, R2A_predict = butterfly.test_model(test_cluster=False, test_figure=False, output_data=True)


import snapatac2 as snap
import numpy as np
from scipy.sparse import csr_matrix
import scanpy as sc

atac_pred = snap.read('R2A.h5ad', backed=None)
m = atac_pred.X.toarray().copy()
# [sum(m[i]>0.7) for i in range(5)]  [8520, 2497, 1514, 12855, 1659]
m[m>0.7]=1
m[m<=0.7]=0
atac_pred.X = csr_matrix(m)

atac= snap.read('/fs/home/jiluzhang/scM2M_pbmc/data/atac_cell_anno.h5ad', backed=None)
atac_test = snap.read('/fs/home/jiluzhang/scM2M_pbmc/atac_test_0.h5ad', backed=None)
atac_true = atac[atac_test.obs.index, :]
atac_pred.obs['cell_anno'] = atac_true.obs['cell_anno']

snap.pp.select_features(atac_pred)
snap.tl.spectral(atac_pred)
snap.tl.umap(atac_pred)
sc.pl.umap(atac_pred, color='cell_anno', legend_fontsize='7', legend_loc='right margin', size=5,
           title='', frameon=True, save='_atac_test_predict.pdf')


########### BABLE ###########
## concat train & valid dataset
import scanpy as sc
import anndata as ad

rna_train = sc.read_h5ad('rna_train_0.h5ad')
rna_val = sc.read_h5ad('rna_val_0.h5ad')
rna_train_val = ad.concat([rna_train, rna_val])
rna_train_val.var = rna_train.var[['gene_ids', 'feature_types']]
rna_train_val.write('rna_train_val.h5ad')

atac_train = sc.read_h5ad('atac_train_0.h5ad')
atac_val = sc.read_h5ad('atac_val_0.h5ad')
atac_train_val = ad.concat([atac_train, atac_val])
atac_train_val.var = atac_train.var[['gene_ids', 'feature_types']]
atac_train_val.write('atac_train_val.h5ad')


# python h5ad2h5.py -n train_val
# python h5ad2h5.py -n test
import h5py
import numpy as np
from scipy.sparse import csr_matrix, hstack
import argparse

parser = argparse.ArgumentParser(description='concat RNA and ATAC h5ad to h5 format')
parser.add_argument('-n', '--name', type=str, help='sample type or name')

args = parser.parse_args()
sn = args.name
rna  = h5py.File('rna_'+sn+'.h5ad', 'r')
atac = h5py.File('atac_'+sn+'.h5ad', 'r')
out  = h5py.File(sn+'.h5', 'w')

g = out.create_group('matrix')
g.create_dataset('barcodes', data=rna['obs']['_index'][:])
rna_atac_csr_mat = hstack((csr_matrix((rna['X']['data'], rna['X']['indices'],  rna['X']['indptr']), shape=[rna['obs']['_index'].shape[0], rna['var']['_index'].shape[0]]),
                           csr_matrix((atac['X']['data'], atac['X']['indices'],  atac['X']['indptr']), shape=[atac['obs']['_index'].shape[0], atac['var']['_index'].shape[0]])))
rna_atac_csr_mat = rna_atac_csr_mat.tocsr()
g.create_dataset('data', data=rna_atac_csr_mat.data)
g.create_dataset('indices', data=rna_atac_csr_mat.indices)
g.create_dataset('indptr',  data=rna_atac_csr_mat.indptr)
l = list(rna_atac_csr_mat.shape)
l.reverse()
g.create_dataset('shape', data=l)

g_2 = g.create_group('features')
g_2.create_dataset('_all_tag_keys', data=np.array([b'genome', b'interval']))
g_2.create_dataset('feature_type', data=np.append([b'Gene Expression']*rna['var']['gene_ids'].shape[0], [b'Peaks']*atac['var']['gene_ids'].shape[0]))
g_2.create_dataset('genome', data=np.array([b'GRCh38'] * (rna['var']['gene_ids'].shape[0]+atac['var']['gene_ids'].shape[0])))
g_2.create_dataset('id', data=np.append(rna['var']['gene_ids'][:], atac['var']['gene_ids'][:]))
g_2.create_dataset('interval', data=np.append(rna['var']['gene_ids'][:], atac['var']['gene_ids'][:]))
g_2.create_dataset('name', data=np.append(rna['var']['_index'][:], atac['var']['_index'][:]))      

out.close()


python /fs/home/jiluzhang/BABEL/bin/train_model.py --data train_val.h5 --outdir babel_train_out --batchsize 512 --earlystop 25 --device 4 --nofilter
python /fs/home/jiluzhang/BABEL/bin/predict_model.py --checkpoint babel_train_out --data test.h5 --outdir babel_test_out --device 5 \
                                                     --nofilter --noplot --transonly
# 3298408

import snapatac2 as snap
import numpy as np
from scipy.sparse import csr_matrix
import scanpy as sc

atac_pred = snap.read('rna_atac_adata.h5ad', backed=None)
m = atac_pred.X.toarray().copy()
# [sum(m[i]>0.7) for i in range(5)]  [10206, 197, 7789, 6523, 523]
m[m>0.5]=1
m[m<=0.5]=0
atac_pred.X = csr_matrix(m)

atac= snap.read('/fs/home/jiluzhang/scM2M_pbmc/data/atac_cell_anno.h5ad', backed=None)
atac_test = snap.read('/fs/home/jiluzhang/scM2M_pbmc/atac_test_0.h5ad', backed=None)
atac_true = atac[atac_test.obs.index, :]
atac_pred.obs['cell_anno'] = atac_true.obs['cell_anno']

snap.pp.select_features(atac_pred)
snap.tl.spectral(atac_pred)
snap.tl.umap(atac_pred)
sc.pl.umap(atac_pred, color='cell_anno', legend_fontsize='7', legend_loc='right margin', size=5,
           title='', frameon=True, save='_atac_test_predict_2.pdf')



## generate true h5ad file
import scanpy as sc 

atac= sc.read_h5ad('/fs/home/jiluzhang/scM2M_pbmc/data/atac_cell_anno.h5ad')
atac_test = sc.read_h5ad('/fs/home/jiluzhang/scM2M_pbmc/atac_test_0.h5ad')
atac[atac_test.obs.index, :].write('rna2atac_true.h5ad')


#################### AUROC & AUPRC ####################
# python cal_auroc_auprc.py --pred rna2atac_scbutterflyc.h5ad --true rna2atac_true.h5ad
import argparse
import scanpy as sc
from sklearn.metrics import precision_recall_curve, roc_curve, auc
import numpy as np
import random
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Calculate AUROC')
parser.add_argument('--pred', type=str, help='prediction')
parser.add_argument('--true', type=str, help='ground truth')
args = parser.parse_args()
pred_file = args.pred
true_file = args.true

pred = sc.read_h5ad(pred_file).X
if type(pred) is not np.ndarray:
    pred = pred.toarray()
true = sc.read_h5ad(true_file).X.toarray()
print('Read h5ad files done')

## cell-wise
random.seed(0)
cell_idx = random.sample(list(range(pred.shape[0])), k=500)

cell_auroc = []
for i in tqdm(cell_idx, ncols=80, desc='Calculate cell-wise AUROC'):
    fpr, tpr, _ = roc_curve(true[i], pred[i])
    cell_auroc.append(auc(fpr, tpr))
print('Cell-wise AUROC:', round(np.mean(cell_auroc), 4))

cell_auprc = []
for i in tqdm(cell_idx, ncols=80, desc='Calculate cell-wise AUPRC'):
    precision, recall, thresholds = precision_recall_curve(true[i], pred[i])
    cell_auprc.append(auc(recall, precision))
print('Cell-wise AUPRC:', round(np.mean(cell_auprc), 4))

## peak-wise
peak_idx = random.sample(list(range(pred.shape[1])), k=10000)

peak_auroc = []
for i in tqdm(peak_idx, ncols=80, desc='Calculate peak-wise AUROC'):
    fpr, tpr, _ = roc_curve(true[:, i], pred[:, i])
    peak_auroc.append(auc(fpr, tpr))
print('Peak-wise AUROC:', round(np.mean(peak_auroc), 4))

peak_auprc = []
for i in tqdm(peak_idx, ncols=80, desc='Calculate peak-wise AUPRC'):
    precision, recall, thresholds = precision_recall_curve(true[:, i], pred[:, i])
    peak_auprc.append(auc(recall, precision))
print('Peak-wise AUPRC:', round(np.mean(peak_auprc), 4))


#################### ARI & AMI & NMI & HOM ####################
## python cal_cluster_plot.py --pred rna2atac_scbutterflyc.h5ad --true rna2atac_true.h5ad
import argparse
import snapatac2 as snap
import numpy as np
from scipy.sparse import csr_matrix
import scanpy as sc
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score, normalized_mutual_info_score, homogeneity_score

parser = argparse.ArgumentParser(description='Calculate clustering metrics and plot umap')
parser.add_argument('--pred', type=str, help='prediction')
parser.add_argument('--true', type=str, help='ground truth')
args = parser.parse_args()
pred_file = args.pred
true_file = args.true


pred_file = 'rna2atac_scm2m.h5ad'
true_file = 'rna2atac_true.h5ad'



pred = snap.read(pred_file, backed=None)
if type(pred.X) is not np.ndarray:
    m = pred.X.toarray().copy()
else:
    m = pred.X.copy()
m[m>0.5]=1
m[m<=0.5]=0
pred.X = csr_matrix(m)

true = snap.read(true_file, backed=None)
pred.obs['cell_anno'] = true.obs['cell_anno']

snap.pp.select_features(pred)
snap.tl.spectral(pred)
snap.tl.umap(pred)
sc.pl.umap(pred, color='cell_anno', legend_fontsize='7', legend_loc='right margin', size=5,
           title='', frameon=True, save='_'+pred_file.split('_')[1].split('.')[0]+'.pdf')
print('Plot umap done')

snap.pp.knn(pred)
snap.tl.leiden(pred)
AMI = adjusted_mutual_info_score(pred.obs['cell_anno'], pred.obs['leiden'])
ARI = adjusted_rand_score(pred.obs['cell_anno'], pred.obs['leiden'])
HOM = homogeneity_score(pred.obs['cell_anno'], pred.obs['leiden'])
NMI = normalized_mutual_info_score(pred.obs['cell_anno'], pred.obs['leiden'])
print('AMI:', round(AMI, 4))
print('ARI:', round(ARI, 4))
print('HOM:', round(HOM, 4))
print('NMI:', round(NMI, 4))





################# CRC #####################
import scanpy as sc
rna = sc.read_h5ad('/fs/home/jiluzhang/2023_nature_LD/normal_h5ad/CM354C2-T1_rna.h5ad')     # 8475 × 38244
sc.pp.filter_genes(rna, min_cells=10)            # 8475 × 18524
rna.write('crc_rna.h5ad')
atac = sc.read_h5ad('/fs/home/jiluzhang/2023_nature_LD/normal_h5ad/CM354C2-T1_atac.h5ad')     # 8475 × 1033239
sc.pp.filter_genes(atac, min_cells=10)              # 8475 × 185908
atac.write('crc_atac.h5ad')

## scM2M
python split_train_val_test.py --RNA rna.h5ad --ATAC atac.h5ad --train_pct 0.7 --valid_pct 0.1
python data_preprocess.py -r rna_train_0.h5ad -a atac_train_0.h5ad -s preprocessed_data_train --dt train --config rna2atac_config_train.yaml
python data_preprocess.py -r rna_val_0.h5ad -a atac_val_0.h5ad -s preprocessed_data_val --dt val --config rna2atac_config_val_eval.yaml
python data_preprocess.py -r rna_test_0.h5ad -a atac_test_0.h5ad -s preprocessed_data_test --dt test --config rna2atac_config_val_eval.yaml

nohup accelerate launch --config_file accelerator_config.yaml --main_process_port 29821 rna2atac_train.py --config_file rna2atac_config_train.yaml \
                        --train_data_dir ./preprocessed_data_train --val_data_dir ./preprocessed_data_val -n rna2atac_train > 20240612.log &

accelerate launch --config_file accelerator_config.yaml --main_process_port 29822 rna2atac_evaluate.py -d ./preprocessed_data_test \
                  -l save/2024-06-12_rna2atac_train_39/pytorch_model.bin --config_file rna2atac_config_val_eval.yaml

accelerate launch --config_file accelerator_config.yaml --main_process_port 29822 rna2atac_evaluate.py -d ./preprocessed_data_test \
                  -l save_depth_1_head_1/2024-06-13_rna2atac_train_19/pytorch_model.bin --config_file rna2atac_config_val_eval.yaml


mv predict.npy test_predict.npy

python npy2h5ad.py
mv rna2atac_scm2m.h5ad benchmark
python cal_auroc_auprc.py --pred rna2atac_scm2m.h5ad --true rna2atac_true.h5ad
python cal_cluster_plot.py --pred rna2atac_scm2m.h5ad --true rna2atac_true_leiden.h5ad



## scButterfly-B
python scbt_b.py
python cal_auroc_auprc.py --pred rna2atac_scbutterflyb.h5ad --true rna2atac_true.h5ad
python cal_cluster_plot.py --pred rna2atac_scbutterflyb.h5ad --true rna2atac_true_leiden.h5ad

## plot testing true (no cell annotation)
import snapatac2 as snap
import scanpy as sc

atac_true = snap.read('rna2atac_true.h5ad', backed=None)

snap.pp.select_features(atac_true)
snap.tl.spectral(atac_true)
snap.tl.umap(atac_true)
snap.pp.knn(atac_true)
snap.tl.leiden(atac_true)
sc.pl.umap(atac_true, color='leiden', legend_fontsize='7', legend_loc='right margin', size=5,
           title='', frameon=True, save='_true.pdf')
atac_true.write('rna2atac_true_leiden.h5ad')



# m = ((m.T > m.T.mean(axis=0)).T) & (m>m.mean(axis=0)).astype(int)


## select best threshold for binarization
import scanpy as sc 
from sklearn.metrics import f1_score
import numpy as np

pred_X = sc.read_h5ad('rna2atac_scbutterflyb.h5ad').X.toarray()
true_X = sc.read_h5ad('rna2atac_true.h5ad').X.toarray()
pred = pred_X[:2].flatten()
true = true_X[:2].flatten()

thresholds = np.arange(0.1, 1, 0.1)
f1_scores = [f1_score(true, (pred >= t).astype(int)) for t in thresholds]
best_f1_threshold_index = np.argmax(f1_scores)
best_f1_threshold = thresholds[best_f1_threshold_index]
best_f1_threshold





