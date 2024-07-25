## Decoding gene regulation in the mouse embryo using single-cell multi-omics
## https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE205117
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


