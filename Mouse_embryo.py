## Decoding gene regulation in the mouse embryo using single-cell multi-omics
## https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE205117
## https://github.com/rargelaguet/mouse_organogenesis_10x_multiome_publication
## workdir: /fs/home/jiluzhang/scM2M_no_dec_attn/mouse_embryo
# nohup wget -c https://ftp.ncbi.nlm.nih.gov/geo/series/GSE205nnn/GSE205117/suppl/GSE205117_RAW.tar > 20240715_download.log &

## CRISPR_T_WT vs. CRISPR_T_KO (E8.5)
## genes are same for wt & ko
## merge wt & ko peaks
zcat GSM6205422_E8.5_CRISPR_T_WT_GEX/features.tsv.gz | grep "Peaks" | cut -f 4-6 > wt_peaks.bed  # 241050
zcat GSM6205421_E8.5_CRISPR_T_KO_GEX/features.tsv.gz | grep "Peaks" | cut -f 4-6 > ko_peaks.bed  # 233258
cat wt_peaks.bed ko_peaks.bed | sort -k1,1 -k2,2n > wt_ko_peaks_sorted.bed
bedtools merge -i wt_ko_peaks_sorted.bed > wt_ko_peaks_raw.bed  # 276986
grep -v -E "GL|JH|chrX|chrY" wt_ko_peaks_raw.bed | sort -k 1.4,1 -n -s -k2,2n > wt_ko_peaks.bed  # 271529
rm wt_ko_peaks_sorted.bed wt_ko_peaks_raw.bed

## process rna & atac
import scanpy as sc
import pandas as pd
import pybedtools
import numpy as np
from tqdm import tqdm
from scipy.sparse import csr_matrix

## wt
wt = sc.read_10x_mtx('GSM6205422_E8.5_CRISPR_T_WT_GEX', gex_only=False)  # 7604 × 273335

## rna
rna_info = pd.read_table('GSM6205422_E8.5_CRISPR_T_WT_GEX/features.tsv.gz', header=None)
rna_info = rna_info[rna_info[2]=="Gene Expression"]
rna_info = rna_info[rna_info[3].isin(['chr'+str(i) for i in range(1, 20)])]

wt_rna = wt[:, wt.var['feature_types']=='Gene Expression'].copy()        # 7604 × 32285
wt_rna_out = wt_rna[:, wt_rna.var['gene_ids'].isin(rna_info[0])].copy()  # 7604 × 30364
wt_rna_out.write('rna_wt.h5ad')

## atac
wt_atac = wt[:, wt.var['feature_types']=='Peaks'].copy()  # 7604 × 241050
wt_atac.X[wt_atac.X>0] = 1

peaks = pd.DataFrame({'id': wt_atac.var_names})
peaks['chr'] = peaks['id'].map(lambda x: x.split(':')[0])
peaks['start'] = peaks['id'].map(lambda x: x.split(':')[1].split('-')[0])
peaks['end'] = peaks['id'].map(lambda x: x.split(':')[1].split('-')[1])
peaks.drop(columns='id', inplace=True)
peaks['idx'] = range(peaks.shape[0])

cCREs = pd.read_table('wt_ko_peaks.bed', names=['chr', 'start', 'end'])
cCREs['idx'] = range(cCREs.shape[0])
cCREs_bed = pybedtools.BedTool.from_dataframe(cCREs)
peaks_bed = pybedtools.BedTool.from_dataframe(peaks)
idx_map = peaks_bed.intersect(cCREs_bed, wa=True, wb=True).to_dataframe().iloc[:, [3, 7]]
idx_map.columns = ['peaks_idx', 'cCREs_idx']

m = np.zeros([wt_atac.n_obs, cCREs_bed.to_dataframe().shape[0]], dtype='float32')
for i in tqdm(range(wt_atac.X.shape[0]), ncols=80, desc='Aligning ATAC peaks'):
    m[i][idx_map[idx_map['peaks_idx'].isin(peaks.iloc[np.nonzero(wt_atac.X[i])[1]]['idx'])]['cCREs_idx']] = 1

wt_atac_new = sc.AnnData(m, obs=wt_atac.obs, var=pd.DataFrame({'gene_ids': cCREs['chr']+':'+cCREs['start'].map(str)+'-'+cCREs['end'].map(str), 'feature_types': 'Peaks'}))
wt_atac_new.var.index = wt_atac_new.var['gene_ids'].values
wt_atac_new.X = csr_matrix(wt_atac_new.X)
wt_atac_new.write('atac_wt.h5ad')


## ko
ko = sc.read_10x_mtx('GSM6205421_E8.5_CRISPR_T_KO_GEX', gex_only=False)  # 7359 × 265543

## rna
rna_info = pd.read_table('GSM6205421_E8.5_CRISPR_T_KO_GEX/features.tsv.gz', header=None)
rna_info = rna_info[rna_info[2]=="Gene Expression"]
rna_info = rna_info[rna_info[3].isin(['chr'+str(i) for i in range(1, 20)])]

ko_rna = ko[:, ko.var['feature_types']=='Gene Expression'].copy()        # 7359 × 32285
ko_rna_out = ko_rna[:, ko_rna.var['gene_ids'].isin(rna_info[0])].copy()  # 7359 × 30364
ko_rna_out.write('rna_ko.h5ad')

## atac
ko_atac = ko[:, ko.var['feature_types']=='Peaks'].copy()  # 7359 × 233258
ko_atac.X[ko_atac.X>0] = 1

peaks = pd.DataFrame({'id': ko_atac.var_names})
peaks['chr'] = peaks['id'].map(lambda x: x.split(':')[0])
peaks['start'] = peaks['id'].map(lambda x: x.split(':')[1].split('-')[0])
peaks['end'] = peaks['id'].map(lambda x: x.split(':')[1].split('-')[1])
peaks.drop(columns='id', inplace=True)
peaks['idx'] = range(peaks.shape[0])

cCREs = pd.read_table('wt_ko_peaks.bed', names=['chr', 'start', 'end'])
cCREs['idx'] = range(cCREs.shape[0])
cCREs_bed = pybedtools.BedTool.from_dataframe(cCREs)
peaks_bed = pybedtools.BedTool.from_dataframe(peaks)
idx_map = peaks_bed.intersect(cCREs_bed, wa=True, wb=True).to_dataframe().iloc[:, [3, 7]]
idx_map.columns = ['peaks_idx', 'cCREs_idx']

m = np.zeros([ko_atac.n_obs, cCREs_bed.to_dataframe().shape[0]], dtype='float32')
for i in tqdm(range(ko_atac.X.shape[0]), ncols=80, desc='Aligning ATAC peaks'):
    m[i][idx_map[idx_map['peaks_idx'].isin(peaks.iloc[np.nonzero(ko_atac.X[i])[1]]['idx'])]['cCREs_idx']] = 1

ko_atac_new = sc.AnnData(m, obs=ko_atac.obs, var=pd.DataFrame({'gene_ids': cCREs['chr']+':'+cCREs['start'].map(str)+'-'+cCREs['end'].map(str), 'feature_types': 'Peaks'}))
ko_atac_new.var.index = ko_atac_new.var['gene_ids'].values
ko_atac_new.X = csr_matrix(ko_atac_new.X)
ko_atac_new.write('atac_ko.h5ad')  # 7359 × 271529


## add cell annotation
import scanpy as sc
import pandas as pd

meta_info = pd.read_table('sample_metadata.txt.gz')

wt_info = meta_info[(meta_info['sample']=='E8.5_CRISPR_T_WT')][['barcode', 'celltype']]  # 7891
wt_info.index = wt_info['barcode'].values
del wt_info['barcode']

wt_rna = sc.read_h5ad('rna_wt.h5ad')  # 7604
wt_rna.obs['cell_anno'] = wt_info.loc[wt_rna.obs.index.values]['celltype'].values
wt_rna_out =  wt_rna[wt_rna.obs['cell_anno'].isin(['NMP', 'Spinal_cord', 'Somitic_mesoderm'])].copy()
# Spinal_cord         526
# Somitic_mesoderm    362
# NMP                 326
wt_rna_out.write('rna_wt_3_types.h5ad')  # 1214 × 30364

wt_atac = sc.read_h5ad('atac_wt.h5ad')  # 7604
wt_atac_out = wt_atac[wt_rna_out.obs.index, :].copy()
wt_atac_out.obs['cell_anno'] = wt_rna_out.obs['cell_anno']
wt_atac_out.write('atac_wt_3_types.h5ad')  # 1214 × 271529


ko_info = meta_info[(meta_info['sample']=='E8.5_CRISPR_T_KO')][['barcode', 'celltype']]  # 7741
ko_info.index = ko_info['barcode'].values
del ko_info['barcode']

ko_rna = sc.read_h5ad('rna_ko.h5ad')  # 7359
ko_rna.obs['cell_anno'] = ko_info.loc[ko_rna.obs.index.values]['celltype'].values
ko_rna_out =  ko_rna[ko_rna.obs['cell_anno'].isin(['NMP', 'Spinal_cord', 'Somitic_mesoderm'])].copy()
# NMP                 745
# Spinal_cord         436
# Somitic_mesoderm      5
ko_rna_out.write('rna_ko_3_types.h5ad')  # 1186 × 30364

ko_atac = sc.read_h5ad('atac_ko.h5ad')  # 7359
ko_atac_out = ko_atac[ko_rna_out.obs.index, :].copy()
ko_atac_out.obs['cell_anno'] = ko_rna_out.obs['cell_anno']
ko_atac_out.write('atac_ko_3_types.h5ad')  # 1186 × 271529



## preprocess & train model
python split_train_val.py --RNA rna_wt_3_types.h5ad --ATAC atac_wt_3_types.h5ad --train_pct 0.9
python data_preprocess.py -r rna_wt_3_types_train.h5ad -a atac_wt_3_types_train.h5ad -s preprocessed_data_train --dt train --config rna2atac_config_train.yaml   # list_digits: 6 -> 7
python data_preprocess.py -r rna_wt_3_types_val.h5ad -a atac_wt_3_types_val.h5ad -s preprocessed_data_val --dt val --config rna2atac_config_val.yaml
nohup accelerate launch --config_file accelerator_config_train.yaml --main_process_port 29823 rna2atac_train.py --config_file rna2atac_config_train.yaml \
                        --train_data_dir ./preprocessed_data_train --val_data_dir ./preprocessed_data_val -n rna2atac_train > 20240726.log &  

python data_preprocess.py -r rna_wt_3_types.h5ad -a atac_wt_3_types.h5ad -s preprocessed_data_test --dt test --config rna2atac_config_test.yaml
accelerate launch --config_file accelerator_config_test.yaml --main_process_port 29822 rna2atac_test.py \
                  -d ./preprocessed_data_test \
                  -l save_mlt_40/2024-07-26_rna2atac_train_300/pytorch_model.bin --config_file rna2atac_config_test.yaml
python npy2h5ad.py
python csr2array.py --pred rna2atac_scm2m_raw.h5ad --true atac_wt_3_types.h5ad
python cal_auroc_auprc.py --pred rna2atac_scm2m.h5ad --true atac_wt_3_types.h5ad
python cal_cluster_plot.py --pred rna2atac_scm2m.h5ad --true atac_wt_3_types.h5ad

## evaluation
## ground truth
import snapatac2 as snap
import scanpy as sc
import numpy as np

true = snap.read('atac_wt_3_types.h5ad', backed=None)
snap.pp.select_features(true)
snap.tl.spectral(true)
snap.tl.umap(true)
sc.pl.umap(true, color='cell_anno', legend_fontsize='7', legend_loc='right margin', size=10,
           title='', frameon=True, save='_atac_3_types_true.pdf')
true[:550, :].write('rna2atac_true.h5ad')

rna = snap.read('rna_wt_3_types.h5ad', backed=None)
np.count_nonzero(rna.X.toarray(), axis=1).max()  # 9516


## in silico ko
import scanpy as sc
import numpy as np

wt_rna = sc.read_h5ad('rna_wt_3_types.h5ad')
T_idx = np.argwhere(wt_rna.var.index=='Gapdh').item()
ko_rna = wt_rna[(wt_rna.obs['cell_anno']=='NMP') & (wt_rna.X[:, T_idx].toarray().flatten()!=0)].copy()
ko_rna.X[:, T_idx] = 0
ko_rna.write('rna_T_ko_nmp.h5ad')

wt_atac = sc.read_h5ad('atac_wt_3_types.h5ad')
wt_atac[ko_rna.obs.index].write('atac_T_ko_nmp.h5ad')

rm -r preprocessed_data_test_T_KO
python data_preprocess.py -r rna_T_ko_nmp.h5ad -a atac_T_ko_nmp.h5ad -s preprocessed_data_test_T_KO --dt test --config rna2atac_config_test.yaml
accelerate launch --config_file accelerator_config_test.yaml --main_process_port 29822 rna2atac_test.py \
                  -d ./preprocessed_data_test_T_KO \
                  -l save_mlt_40/2024-07-26_rna2atac_train_300/pytorch_model.bin --config_file rna2atac_config_test.yaml


import scanpy as sc
from scipy.stats import pearsonr
import numpy as np

wt_atac = sc.read_h5ad('atac_wt_3_types.h5ad')
wt_nmp = wt_atac[wt_atac.obs['cell_anno']=='NMP'].copy()
wt_som = wt_atac[wt_atac.obs['cell_anno']=='Somitic_mesoderm'].copy()
wt_nmp_dat = wt_nmp.X.toarray().sum(axis=0)
wt_som_dat = wt_som.X.toarray().sum(axis=0)
print(pearsonr(wt_nmp_dat, wt_som_dat)[0])  # 0.9272885433004112


wt_atac_pred = sc.read_h5ad('mlt_40_predict/rna2atac_scm2m_binary.h5ad')
wt_nmp_pred = wt_atac_pred[wt_atac_pred.obs['cell_anno']=='NMP'].copy()
wt_som_pred = wt_atac_pred[wt_atac_pred.obs['cell_anno']=='Somitic_mesoderm'].copy()
wt_nmp_dat_pred = wt_nmp_pred.X.toarray().sum(axis=0)
wt_som_dat_pred = wt_som_pred.X.toarray().sum(axis=0)
print(pearsonr(wt_nmp_dat_pred, wt_som_dat)[0])  # 0.553214528754214


ko_atac = np.load('predict.npy')
ko_atac[ko_atac>0.5] = 1
ko_atac[ko_atac<=0.5] = 0
ko_nmp_dat = ko_atac.sum(axis=0)
print(pearsonr(ko_nmp_dat, wt_som_dat)[0])  # 0.5482014271730806

# Sox2:    0.5535342709392107
# T:       0.5482014271730806
# Topors:  0.5504626938663995
# Foxc2:   0.549615402837736
# Pax6:    0.5488411971160208
# Meox1:   0.5514782279644563
# Gapdh:   0.5531658071787755


## KO & OE
## stablize cell identity
################################### KO data is necessary? ##################################


