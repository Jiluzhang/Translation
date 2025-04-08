#### adult brain
#### celltype (not predicted.id)
# Oligodendrocyte
# VIP
# Microglia
# OPC
# Astrocyte
# SST
# IT
# PVALB

#### h5 to h5ad
import scanpy as sc
import pandas as pd
from scipy.sparse import csr_matrix
import pybedtools
import numpy as np
from tqdm import tqdm

dat = sc.read_10x_h5('human_brain_3k_filtered_feature_bc_matrix.h5', gex_only=False)  # 3233 × 170631
dat.var_names_make_unique()

## add cell annotation
meta_info = pd.read_table('meta_data.txt')
cell_barcode = pd.DataFrame({'barcode': dat.obs.index.values})
dat.obs['cell_anno'] = pd.merge(cell_barcode, meta_info, how='left')['celltype'].values
dat = dat[~dat.obs['cell_anno'].isna(), :].copy()   # 2855 × 170631

## rna
rna = dat[:, dat.var['feature_types']=='Gene Expression'].copy()  # 2855 × 36601
rna.X = rna.X.toarray()
genes = pd.read_table('human_genes.txt', names=['gene_ids', 'gene_name'])
rna_exp = pd.concat([rna.var, pd.DataFrame(rna.X.T, index=rna.var.index)], axis=1)
X_new = pd.merge(genes, rna_exp, how='left', on='gene_ids').iloc[:, 4:].T
X_new.fillna(value=0, inplace=True)
rna_new = sc.AnnData(X_new.values, obs=rna.obs, var=pd.DataFrame({'gene_ids': genes['gene_ids'], 'feature_types': 'Gene Expression'}))
rna_new.var.index = genes['gene_name'].values
rna_new.X = csr_matrix(rna_new.X)   # 2855 × 38244

sc.pp.filter_genes(rna_new, min_cells=10)       # 2855 × 17699
sc.pp.filter_cells(rna_new, min_genes=200)      # 2855 × 17699
sc.pp.filter_cells(rna_new, max_genes=20000)    # 2855 × 17699
rna_new.obs['n_genes'].max()                    # 14416

## atac
atac = dat[:, dat.var['feature_types']=='Peaks'].copy()  # 2855 × 134030
atac.X[atac.X>0] = 1  # binarization
atac.X = atac.X.toarray()
  
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

sc.pp.filter_genes(atac_new, min_cells=10)     # 2855 × 217893
sc.pp.filter_cells(atac_new, min_genes=500)    # 2855 × 217893
sc.pp.filter_cells(atac_new, max_genes=50000)  # 2597 × 217893

## output rna & atac
idx = np.intersect1d(rna_new.obs.index, atac_new.obs.index)
del rna_new.obs['n_genes']
del rna_new.var['n_cells']
rna_new[idx, :].copy().write('rna.h5ad')     # 2855 × 17699
del atac_new.obs['n_genes']
del atac_new.var['n_cells']
atac_new[idx, :].copy().write('atac.h5ad')   # 2597 × 217893


#### map to reference
import scanpy as sc
import pandas as pd
from scipy.sparse import csr_matrix
import numpy as np

## rna
rna = sc.read_h5ad('rna.h5ad')                                                             # 2597 × 17699
rna_ref = sc.read_h5ad('/fs/home/jiluzhang/Nature_methods/Figure_1/scenario_1/rna.h5ad')   # 9964 × 16706
genes = rna_ref.var[['gene_ids']].copy()
genes['gene_name'] = genes.index.values
genes.reset_index(drop=True, inplace=True)
rna.X = rna.X.toarray()
rna_exp = pd.concat([rna.var, pd.DataFrame(rna.X.T, index=rna.var.index)], axis=1)
X_new = pd.merge(genes, rna_exp, how='left', on='gene_ids').iloc[:, 3:].T
X_new.fillna(value=0, inplace=True)
rna_new = sc.AnnData(X_new.values, obs=rna.obs[['cell_anno']], var=pd.DataFrame({'gene_ids': genes['gene_ids'], 'feature_types': 'Gene Expression'}))
rna_new.var.index = genes['gene_name'].values
rna_new.X = csr_matrix(rna_new.X)   # 2597 × 16706
rna_new.write('rna_test.h5ad')
np.count_nonzero(rna_new.X.toarray(), axis=1).max()  # 11054

## atac
atac = sc.read_h5ad('atac.h5ad')                                                             # 2597 × 217893
atac_ref = sc.read_h5ad('/fs/home/jiluzhang/Nature_methods/Figure_1/scenario_1/atac.h5ad')   # 9964 × 236295

peaks = atac_ref.var[['gene_ids']].copy()
peaks['gene_name'] = peaks.index.values
peaks.reset_index(drop=True, inplace=True)
atac.X = atac.X.toarray()
atac_exp = pd.concat([atac.var, pd.DataFrame(atac.X.T, index=atac.var.index)], axis=1)
X_new = pd.merge(peaks, atac_exp, how='left', on='gene_ids').iloc[:, 3:].T
X_new.fillna(value=0, inplace=True)
atac_new = sc.AnnData(X_new.values, obs=atac.obs[['cell_anno']], var=pd.DataFrame({'gene_ids': peaks['gene_ids'], 'feature_types': 'Peaks'}))
atac_new.var.index = peaks['gene_name'].values
atac_new.X = csr_matrix(atac_new.X)   # 2597 × 236295
atac_new.write('atac_test.h5ad')


#### cisformer
python data_preprocess.py -r rna_test.h5ad -a atac_test.h5ad -s test_pt --dt test -n test --config rna2atac_config_test.yaml  # 7 min
accelerate launch --config_file accelerator_config_test.yaml --main_process_port 29822 rna2atac_predict.py -d test_pt \
                  -l /fs/home/jiluzhang/Nature_methods/Figure_1/scenario_1/cisformer/save_mlt_40_large/2024-09-30_rna2atac_pbmc_34/pytorch_model.bin \
                  --config_file rna2atac_config_test.yaml  # 8 min
python npy2h5ad.py
python cal_auroc_auprc.py --pred atac_cisformer.h5ad --true atac_test.h5ad
# Cell-wise AUROC: 0.8213
# Cell-wise AUPRC: 0.2928
# Peak-wise AUROC: 0.538
# Peak-wise AUPRC: 0.1458
python plot_save_umap.py --pred atac_cisformer.h5ad --true atac_test.h5ad  # 2 min
python cal_cluster.py --file atac_cisformer_umap.h5ad
# AMI: 0.6106
# ARI: 0.3328
# HOM: 0.759
# NMI: 0.613

#### scbutterfly
## python scbt_b_predict.py  # 10 min
import os
from scButterfly.butterfly import Butterfly
from scButterfly.split_datasets import *
import scanpy as sc
import anndata as ad

os.environ["CUDA_VISIBLE_DEVICES"] = "4"

RNA_data = sc.read_h5ad('rna_test.h5ad')
ATAC_data = sc.read_h5ad('atac_test.h5ad')

train_id = list(range(RNA_data.n_obs))[:10]       # pseudo
validation_id = list(range(RNA_data.n_obs))[:10]  # pseudo
test_id = list(range(RNA_data.n_obs))             # true

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
# butterfly.train_model(batch_size=16)
A2R_predict, R2A_predict = butterfly.test_model(load_model=True, model_path='/fs/home/jiluzhang/Nature_methods/Figure_1/scenario_1/scbt',
                                                test_cluster=False, test_figure=False, output_data=True)

cp predict/R2A.h5ad atac_scbt.h5ad
python cal_auroc_auprc.py --pred atac_scbt.h5ad --true atac_test.h5ad
# Cell-wise AUROC: 0.8182
# Cell-wise AUPRC: 0.269
# Peak-wise AUROC: 0.4496
# Peak-wise AUPRC: 0.1336

python plot_save_umap.py --pred atac_scbt.h5ad --true atac_test.h5ad  # 8 min
python cal_cluster.py --file atac_scbt_umap.h5ad
# AMI: 0.3163
# ARI: 0.128
# HOM: 0.4365
# NMI: 0.3213


#### BABEL
python h5ad2h5.py -n test
python /fs/home/jiluzhang/BABEL/bin/predict_model.py --checkpoint /fs/home/jiluzhang/Nature_methods/Figure_1/scenario_1/babel/train_out --data test.h5 --outdir test_out --device 5 --nofilter --noplot --transonly   # 9 min

python match_cell.py

python cal_auroc_auprc.py --pred atac_babel.h5ad --true atac_test.h5ad
# Cell-wise AUROC: 0.8270
# Cell-wise AUPRC: 0.2952
# Peak-wise AUROC: 0.5428
# Peak-wise AUPRC: 0.1486

python plot_save_umap.py --pred atac_babel.h5ad --true atac_test.h5ad  # 8 min
python cal_cluster.py --file atac_babel_umap.h5ad
# AMI: 0.1939
# ARI: 0.0679
# HOM: 0.2602
# NMI: 0.1991


#### plot metrics
## python plot_metrics.py (y range: 1->0.7)
#    ami_nmi_ari_hom.txt & auroc_auprc.txt
# -> ami_nmi_ari_hom.pdf & auroc_auprc.pdf

#### plot true atac umap
## python plot_true_umap.py



#### calculate umap for true rna & atac
import scanpy as sc
import matplotlib.pyplot as plt
import pandas as pd
import snapatac2 as snap
import numpy as np

#sc.settings.set_figure_params(dpi=600)
plt.rcParams['pdf.fonttype'] = 42

rna = sc.read_h5ad('rna_test.h5ad')  # 2597 × 16706
rna.obs['cell_anno'].value_counts()
# Oligodendrocyte    1595
# Astrocyte           402
# OPC                 174
# VIP                 115
# Microglia           109
# SST                 104
# IT                   83
# PVALB                15

sc.pp.normalize_total(rna, target_sum=1e4)
sc.pp.log1p(rna)
sc.pp.highly_variable_genes(rna)
rna.raw = rna
rna = rna[:, rna.var.highly_variable]  # 2597 × 2954
sc.tl.pca(rna)
sc.pp.neighbors(rna)
sc.tl.umap(rna)

# sc.tl.rank_genes_groups(rna, 'cell_anno')
# pd.DataFrame(rna.uns['rank_genes_groups']['names']).head(20)
#     Astrocyte       IT Microglia     OPC Oligodendrocyte     PVALB       SST      VIP
# 0     PITPNC1     SYT1     LRMDA  PCDH15            ST18     CNTN5  KIAA1217   SNHG14
# 1      ADGRV1    CSMD1    CHST11   DSCAM          CTNNA3    RBFOX1    DLGAP1    MYT1L
# 2       GPM6A   RBFOX1     FOXN3     TNR             MBP     DPP10    RBFOX1   CCSER1
# 3        GPC5    KALRN     ITPR2   OPCML           ENPP2   ZNF385D      SYT1     KAZN
# 4      SLC1A2    FGF14    SFMBT2  KCNIP4         SLC44A1      SYT1    SNHG14    FGF14
# 5        RORA     LDB2    FRMD4A   LUZP2         TMEM144     FGF12     GRIP1     MEG3
# 6        SOX5    AGBL4      CD83  LHFPL3          RNF220    DLGAP1     RIMS2   ADARB2
# 7        RYR3   DLGAP2   ST6GAL1   CSMD1              TF      NBEA      NRG3    RIMS2
# 8      SORBS1     NRG3  ARHGAP24  LRRC4C         PIP4K2A     MEF2C      PCLO     NRG3
# 9    OBI1-AS1    RIMS2    SRGAP2    SOX6          FRMD4B    DLGAP2    ATRNL1    PLCB1
# 10     ABLIM1    NRXN1     MEF2C   KCND2          PRUNE2     TENM4    CCSER1   DLGAP1
# 11       MSI2    CELF2     MEF2A   MDGA2         SLC24A2  TMEM132B     FGF14     DAB1
# 12       NEBL   DLGAP1     DOCK8   MMP16            DPYD      MEG3      XKR4    NRXN1
# 13  LINC00299    FGF12     CELF2    VCAN         PLEKHH1   CNTNAP5     NALF1    ROBO2
# 14     CTNNA2   SNHG14   LDLRAD4   SNTG1           KCNH8     CADPS     NXPH1    DSCAM
# 15       NRG3  STXBP5L      SRGN  KCNMB2            UGT8    THSD7A      KAZN  STXBP5L
# 16     NKAIN3  SLC4A10     RUNX1   XYLT1         SLCO1A2      RORA      SOX6   FRMD4A
# 17     SLC1A3  PHACTR1    INPP5D  SEMA5A     SLC7A14-AS1  TMEM132D     GRIA1  CACNA1B
# 18       DTNA    NELL2     DISC1   NRXN1           TTLL7     TAFA2     FGF12    FGF12
# 19      NRXN1   CCSER1     KCNQ3   FGF14         TMEM165     NALF1      OXR1     RGS7

rna.write('rna_test_umap.h5ad')

atac_true = sc.read_h5ad('atac_test.h5ad')  # 2597 × 236295
snap.pp.select_features(atac_true)
snap.tl.spectral(atac_true)
snap.tl.umap(atac_true)

atac_true.write('atac_test_umap.h5ad')

# atac_true.var['chrom'] = atac_true.var.index.map(lambda x: x.split(':')[0])
# atac_true.var['start'] = atac_true.var.index.map(lambda x: x.split(':')[1].split('-')[0]).astype('int')
# atac_true.var['end'] = atac_true.var.index.map(lambda x: x.split(':')[1].split('-')[1]).astype('int')
# gene_dict = {'ST18':['chr8', 52460959], 
#              'CTNNA3':['chr10', 67763637], 
#              'MBP':['chr18', 77133708],
#              'ENPP2':['chr8', 119673453],
#              'SLC44A1':['chr9', 105244622],
#              'TMEM144': ['chr4', 158201604],
#              'RNF220': ['chr1', 44404783],
#              'TF':['chr3', 133661998],
#              'PIP4K2A':['chr10', 22714578],
#              'FRMD4B':['chr3', 69542586]}
# for gene in gene_dict.keys():
#     atac_true.obs[gene] = atac_true.X[:, (atac_true.var['chrom']==gene_dict[gene][0]) & (abs((atac_true.var['start']+atac_true.var['end'])*0.5-gene_dict[gene][1])<10000)].toarray().sum(axis=1)
# sc.pl.umap(atac_true, color=gene_dict.keys(), legend_fontsize='5', legend_loc='right margin', size=5, cmap=LinearSegmentedColormap.from_list('lightgray_red', ['lightgray', 'red']),
#            title='', frameon=True, save='_st18_atac_true.pdf')


#### plot umap for marker peak & gene
import scanpy as sc
import matplotlib.pyplot as plt
import pandas as pd
import snapatac2 as snap
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

plt.rcParams['pdf.fonttype'] = 42

rna_true = sc.read_h5ad('rna_test_umap.h5ad')
sc.pl.umap(rna_true, color='cell_anno', legend_fontsize='5', legend_loc='right margin', size=5, 
           title='', frameon=True, save='_cell_anno_rna.pdf')

atac_true = sc.read_h5ad('atac_test_umap.h5ad')       # 2597 × 236295
atac_cifm = sc.read_h5ad('atac_cisformer_umap.h5ad')  # 2597 × 236295
atac_scbt = sc.read_h5ad('atac_scbt_umap.h5ad')       # 2597 × 236295
atac_babel = sc.read_h5ad('atac_babel_umap.h5ad')     # 2597 × 236295

atac_scbt.var.index = atac_true.var.index.values

true_marker_peaks = snap.tl.marker_regions(atac_true, groupby='cell_anno', pvalue=0.05)
cifm_marker_peaks = snap.tl.marker_regions(atac_cifm, groupby='cell_anno', pvalue=0.05)

## Oligodendrocyte
true_cifm_peaks = np.intersect1d(true_marker_peaks['Oligodendrocyte'], cifm_marker_peaks['Oligodendrocyte'])  # 115

sc.pl.umap(atac_true, color=true_cifm_peaks[:10], legend_fontsize='5', legend_loc='right margin', size=5, cmap=LinearSegmentedColormap.from_list('lightgray_red', ['lightgray', 'red']),
           title='', frameon=True, save='_marker_peaks_atac_true.pdf')
sc.pl.umap(atac_cifm, color=true_cifm_peaks[:10], legend_fontsize='5', legend_loc='right margin', size=5, cmap=LinearSegmentedColormap.from_list('lightgray_red', ['lightgray', 'red']),
           title='', frameon=True, save='_marker_peak_atac_cisformer.pdf')

# chr10:70672520-70672823   ADAMTS14
sc.pl.umap(atac_true, color='chr10:70672520-70672823', legend_fontsize='5', legend_loc='right margin', size=5, cmap=LinearSegmentedColormap.from_list('lightgray_red', ['lightgray', 'red']),
           title='', frameon=True, save='_chr10_70672520_70672823_atac_true.pdf')
sc.pl.umap(atac_cifm, color='chr10:70672520-70672823', legend_fontsize='5', legend_loc='right margin', size=5, cmap=LinearSegmentedColormap.from_list('lightgray_red', ['lightgray', 'red']),
           title='', frameon=True, save='_chr10_70672520_70672823_atac_cisformer.pdf')
sc.pl.umap(atac_scbt, color='chr10:70672520-70672823', legend_fontsize='5', legend_loc='right margin', size=5, cmap=LinearSegmentedColormap.from_list('lightgray_red', ['lightgray', 'red']),
           title='', frameon=True, save='_chr10_70672520_70672823_atac_scbt.pdf')
sc.pl.umap(atac_babel, color='chr10:70672520-70672823', legend_fontsize='5', legend_loc='right margin', size=5, cmap=LinearSegmentedColormap.from_list('lightgray_red', ['lightgray', 'red']),
           title='', frameon=True, save='_chr10_70672520_70672823_atac_babel.pdf')
sc.pl.umap(rna_true, color='ADAMTS14', legend_fontsize='5', legend_loc='right margin', size=5, cmap=LinearSegmentedColormap.from_list('lightgray_red', ['lightgray', 'red']),
           title='', frameon=True, vmax=2.5, save='_adamts14_rna.pdf')

## Astrocyte
true_cifm_peaks = np.intersect1d(true_marker_peaks['Astrocyte'], cifm_marker_peaks['Astrocyte'])  # 117

sc.pl.umap(atac_true, color=true_cifm_peaks[20:30], legend_fontsize='5', legend_loc='right margin', size=5, cmap=LinearSegmentedColormap.from_list('lightgray_blue', ['lightgray', 'blue']),
           title='', frameon=True, save='_marker_peaks_atac_true.pdf')
sc.pl.umap(atac_cifm, color=true_cifm_peaks[20:30], legend_fontsize='5', legend_loc='right margin', size=5, cmap=LinearSegmentedColormap.from_list('lightgray_blue', ['lightgray', 'blue']),
           title='', frameon=True, save='_marker_peak_atac_cisformer.pdf')

# chr17:57356784-57357132    MSI2
sc.pl.umap(atac_true, color='chr17:57356784-57357132', legend_fontsize='5', legend_loc='right margin', size=5, cmap=LinearSegmentedColormap.from_list('lightgray_blue', ['lightgray', 'blue']),
           title='', frameon=True, save='_chr17_57356784_57357132_atac_true.pdf')
sc.pl.umap(atac_cifm, color='chr17:57356784-57357132', legend_fontsize='5', legend_loc='right margin', size=5, cmap=LinearSegmentedColormap.from_list('lightgray_blue', ['lightgray', 'blue']),
           title='', frameon=True, save='_chr17_57356784_57357132_atac_cisformer.pdf')
sc.pl.umap(atac_scbt, color='chr17:57356784-57357132', legend_fontsize='5', legend_loc='right margin', size=5, cmap=LinearSegmentedColormap.from_list('lightgray_blue', ['lightgray', 'blue']),
           title='', frameon=True, save='_chr17_57356784_57357132_atac_scbt.pdf')
sc.pl.umap(atac_babel, color='chr17:57356784-57357132', legend_fontsize='5', legend_loc='right margin', size=5, cmap=LinearSegmentedColormap.from_list('lightgray_blue', ['lightgray', 'blue']),
           title='', frameon=True, save='_chr17_57356784_57357132_atac_babel.pdf')
sc.pl.umap(rna_true, color='MSI2', legend_fontsize='5', legend_loc='right margin', size=5, cmap=LinearSegmentedColormap.from_list('lightgray_blue', ['lightgray', 'blue']),
           title='', frameon=True, save='_msi2_rna.pdf')



## genomic tracks showing an example
import scanpy as sc
import pandas as pd

rna_true = sc.read_h5ad('rna_test.h5ad')      # 2597 × 16706
atac_true = sc.read_h5ad('atac_test.h5ad')    # 2597 × 236295

def out_norm_bedgraph(atac, cell_type='Oligodendrocyte', type='pred'):
    atac_X_raw = atac[atac.obs['cell_anno']==cell_type, :].X.toarray().sum(axis=0)
    atac_X = atac_X_raw/atac_X_raw.sum()*(1e6)
    df = pd.DataFrame({'chr': atac.var['gene_ids'].map(lambda x: x.split(':')[0]).values,
                       'start': atac.var['gene_ids'].map(lambda x: x.split(':')[1].split('-')[0]).values,
                       'end': atac.var['gene_ids'].map(lambda x: x.split(':')[1].split('-')[1]).values,
                       'val': atac_X})
    df.to_csv(cell_type.replace(' ', '_')+'_atac_'+type+'_norm.bedgraph', index=False, header=False, sep='\t')

ct_lst = set(atac_true.obs['cell_anno'])
for ct in ct_lst:
    out_norm_bedgraph(atac=atac_true, cell_type=ct, type='true')
    out_norm_bedgraph(atac=atac_true, cell_type=ct, type='true')
    out_norm_bedgraph(atac=atac_true, cell_type=ct, type='true')
    out_norm_bedgraph(atac=atac_true, cell_type=ct, type='true')


















