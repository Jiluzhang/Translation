## pbmc dataset

## alignment -> filtering -> clustering -> modeling -> evaluation

## raw data process
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


## cell annotation
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
sc.tl.umap(rna, min_dist=1)
sc.tl.leiden(rna, resolution=0.5)
sc.pl.umap(rna, color='leiden', legend_fontsize='5', legend_loc='on data',
           title='', frameon=True, save='_rna.pdf')

# 0.75  0.5

sc.tl.rank_genes_groups(rna, 'leiden', method='wilcoxon')
pd.DataFrame(rna.uns['rank_genes_groups']['names']).head(20)
#            0         1       2        3       4          5         6       7       8         9         10         11          12       13
# 0       DPYD    INPP4B    FHIT     LEF1    CCL5      BANK1    TCF7L2    GNLY  PLXDC2      CST3      BANK1       TCF4     OSBPL10     SOX4
# 1     PLXDC2       LTB    LEF1    PDE3B    NKG7    RALGPS2     MTSS1    PRF1   DMXL2      CD74      MS4A1       RHEX     RALGPS2   NKAIN2
# 2      NEAT1      IL32  BCL11B    NELL2    GZMA       AFF3      PSAP    NKG7  SLC8A1  HLA-DPB1    RALGPS2       ZFAT       CD79A    ZNRF1
# 3      LRMDA     ITGB1   CAMK4   THEMIS     A2M      MS4A1     COTL1   CD247   LRMDA  HLA-DPA1       AFF3       IRF8       FCRL1     CDK6
# 4   ARHGAP26    CDC14A  MALAT1    BACH2    IL32       CD74      LST1   KLRD1   NAMPT  HLA-DRB1       PAX5     FCHSD2       TPD52   RNF220
# 5       FCN1      ANK3   BACH2     CD8B   SYNE2       PAX5      AIF1    GZMA  TBXAS1   HLA-DRA      FCRL1       UGCG       MEF2C      ERG
# 6       VCAN     SYNE2    TCF7   OXNAD1    GNLY      FCRL1    IFITM3   MCTP2   IRAK3  HLA-DQA1      CD79A      EPHB1        GNG7   ZNF521
# 7       JAK2      IL7R    ANK3      TXK    PRF1      CD79A     WARS1   SPON2   RBM47  HLA-DQB1      CD79B       PLD4        PAX5  ST8SIA6
# 8     SLC8A1    TTC39C  INPP4B    CAMK4   SYNE1    HLA-DRA      CST3    CST7   MCTP1     HDAC9       IGHM  LINC01374       CDK14  HNRNPA1
# 9    DENND1A    EEF1A1  RPS27A     CCR7   HLA-B       EBF1       LYN   SYNE1   NEAT1  HLA-DRB5        BLK      PTPRS      SEL1L3     MSI2
# 10     NAMPT      RORA    CCR7   BCL11B    CTSW  LINC00926    FCGR3A    CTSW    FCN1      ACTB  LINC00926     CCDC50       BANK1    AUTS2
# 11     RBM47  ARHGAP15   PRKCA    APBA2   KLRG1       IGHM  SERPINA1   KLRF1   WARS1       LYZ       EBF1     BCL11A       MS4A1     MAP7
# 12      ZEB2     RPL41   MAML2    MAML2   HLA-C    OSBPL10     IFI30  PYHIN1    GAB2    SAMHD1     SEL1L3       CUX2      SETBP1    STMN1
# 13      AOAH     RPL13   RPS25  SERINC5  TGFBR3        BLK      CTSS    PFN1    JAK2    CCSER1    OSBPL10     PLXNA4        IGKC    RPLP0
# 14     CSF3R      TPT1   RPS27   NDFIP1    CST7      CD79B    FCER1G  TGFBR3   LRRK2   CCDC88A       CD22      RUNX2        TCF4    PPM1H
# 15      TYMP     RPS18   RPL11     TCF7   PRKCH   HLA-DRB1  TNFRSF1B    GZMB    VCAN      AFF3       CD74        APP      FCHSD2   SNHG29
# 16     DMXL2      TRAC   RPS12   ABLIM1   SAMD3   HLA-DQA1      UTRN   IL2RB    CD36       VIM     ADAM28  LINC00996        EBF1    MEIS1
# 17     LRRK2     RPS27  OXNAD1      ITK     PZP       CD22     MS4A7  ZBTB16    TLR2       GRN     NIBAN3   RABGAP1L      ADAM28   CASC15
# 18      LYST     RPL34   RPS14    FOXP1    PFN1     SEL1L3    TYROBP    CCL5    LYST     ALCAM      TPD52  LINC01478        AFF3    CALN1
# 19       LYZ     MDFIC    RPL3     CD8A    HCST       CD37       HCK   SAMD3    TYMP     ANXA2       IGHD      ITM2C  PALM2AKAP2    SNHG7

# cluster_0:DPYD,PLXDC2,NEAT1,LRMDA,ARHGAP26,FCN1,VCAN,JAK2,SLC8A1,DENND1A,NAMPT,RBM47,ZEB2,AOAH,CSF3R,TYMP,DMXL2,LRRK2,LYST,LYZ
# cluster_1:INPP4B,LTB,IL32,ITGB1,CDC14A,ANK3,SYNE2,IL7R,TTC39C,EEF1A1,RORA,ARHGAP15,RPL41,RPL13,TPT1,RPS18,TRAC,RPS27,RPL34,MDFIC
# cluster_2:FHIT,LEF1,BCL11B,CAMK4,MALAT1,BACH2,TCF7,ANK3,INPP4B,RPS27A,CCR7,PRKCA,MAML2,RPS25,RPS27,RPL11,RPS12,OXNAD1,RPS14,RPL3
# cluster_3:LEF1,PDE3B,NELL2,THEMIS,BACH2,CD8B,OXNAD1,TXK,CAMK4,CCR7,BCL11B,APBA2,MAML2,SERINC5,NDFIP1,TCF7,ABLIM1,ITK,FOXP1,CD8A
# cluster_4:CCL5,NKG7,GZMA,A2M,IL32,SYNE2,GNLY,PRF1,SYNE1,HLA-B,CTSW,KLRG1,HLA-C,TGFBR3,CST7,PRKCH,SAMD3,PZP,PFN1,HCST
# cluster_5:BANK1,RALGPS2,AFF3,MS4A1,CD74,PAX5,FCRL1,CD79A,HLA-DRA,EBF1,LINC00926,IGHM,OSBPL10,BLK,CD79B,HLA-DRB1,HLA-DQA1,CD22,SEL1L3,CD37
# cluster_6:TCF7L2,MTSS1,PSAP,COTL1,LST1,AIF1,IFITM3,WARS1,CST3,LYN,FCGR3A,SERPINA1,IFI30,CTSS,FCER1G,TNFRSF1B,UTRN,MS4A7,TYROBP,HCK
# cluster_7:GNLY,PRF1,NKG7,CD247,KLRD1,GZMA,MCTP2,SPON2,CST7,SYNE1,CTSW,KLRF1,PYHIN1,PFN1,TGFBR3,GZMB,IL2RB,ZBTB16,CCL5,SAMD3
# cluster_8:PLXDC2,DMXL2,SLC8A1,LRMDA,NAMPT,TBXAS1,IRAK3,RBM47,MCTP1,NEAT1,FCN1,WARS1,GAB2,JAK2,LRRK2,VCAN,CD36,TLR2,LYST,TYMP
# cluster_9:CST3,CD74,HLA-DPB1,HLA-DPA1,HLA-DRB1,HLA-DRA,HLA-DQA1,HLA-DQB1,HDAC9,HLA-DRB5,ACTB,LYZ,SAMHD1,CCSER1,CCDC88A,AFF3,VIM,GRN,ALCAM,ANXA2
# cluster_10:BANK1,MS4A1,RALGPS2,AFF3,PAX5,FCRL1,CD79A,CD79B,IGHM,BLK,LINC00926,EBF1,SEL1L3,OSBPL10,CD22,CD74,ADAM28,NIBAN3,TPD52,IGHD
# cluster_11:TCF4,RHEX,ZFAT,IRF8,FCHSD2,UGCG,EPHB1,PLD4,LINC01374,PTPRS,CCDC50,BCL11A,CUX2,PLXNA4,RUNX2,APP,LINC00996,RABGAP1L,LINC01478,ITM2C
# cluster_12:OSBPL10,RALGPS2,CD79A,FCRL1,TPD52,MEF2C,GNG7,PAX5,CDK14,SEL1L3,BANK1,MS4A1,SETBP1,IGKC,TCF4,FCHSD2,EBF1,ADAM28,AFF3,PALM2AKAP2
# cluster_13:SOX4,NKAIN2,ZNRF1,CDK6,RNF220,ERG,ZNF521,ST8SIA6,HNRNPA1,MSI2,AUTS2,MAP7,STMN1,RPLP0,PPM1H,SNHG29,MEIS1,CASC15,CALN1,SNHG7

# cluster_0: Classical monocyte
# cluster_1: Central memory CD4-positive, alpha-beta T cell
# cluster_2: Naive thymus-derived CD4-positive, alpha-beta T cell
# cluster_3: Naive thymus-derived CD8-positive, alpha-beta T cell
# cluster_4: CD8-positive, alpha-beta cytotoxic T cell
# cluster_5: Memory B cell
# cluster_6: Non-classical monocyte
# cluster_7: Natural killer cell
# cluster_8: Classical monocyte
# cluster_9: Dendritic cell
# cluster_10: Memory B cell
# cluster_11: Plasmacytoid dendritic cell, human
# cluster_12: B cell
# cluster_13: Megakaryocyte-erythroid progenitor cell

## cell type annotation tool: http://xteam.xbio.top/ACT
## ref: https://www.nature.com/articles/s41467-023-36559-0/figures/5
cell_anno_dict = {'0':'T cell', '1':'Tumor B cell', '2':'Monocyte', '3':'Normal B cell', '4':'Other'}
adata.obs['cell_anno'] = adata.obs['leiden']
adata.obs['cell_anno'].replace(cell_anno_dict, inplace=True)

sc.pl.umap(adata, color='cell_anno', legend_fontsize='7', legend_loc='on data', size=3,
           title='', frameon=True, save='_cell_annotation.pdf')

adata.write('rna_tumor_B_res.h5ad')
