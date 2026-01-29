#################### calculate Voronoi area for each cell ####################
## workdir: /fs/home/jiluzhang/spatial_mechano/pipeline_3
import pandas as pd
import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d
import matplotlib.pyplot as plt
from tqdm import *
from scipy import stats
import scanpy as sc

df = pd.read_csv('R1_R77_4C4_cell_metadata.csv')
df = df[df['status']=='ok']  # 79088

vor = Voronoi(df[['global_x', 'global_y']])

## vertices should be ranked based on plot
def poly_area(vertices):
    x = vertices[:, 0]
    y = vertices[:, 1]
    return (0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))).item()

## calculate area
areas = []
region_indices = []
for idx, region in enumerate(vor.regions):
    # delete blank or no-border regions
    if not region or -1 in region:
        continue
    
    polygon_vertices = vor.vertices[region]
    area = poly_area(polygon_vertices)
    areas.append(area)
    region_indices.append(idx)

## assign area for each point
# 输入点 i → vor.point_region[i] → 区域索引 R → vor.regions[R] → 顶点坐标 → 计算面积 → 存储到 point_areas[i]
# ~ 3 min
point_areas = np.full(df.shape[0], np.nan)
for point_idx, region_idx in enumerate(tqdm(vor.point_region, ncols=80)):
    if region_idx in region_indices:
        area_idx = region_indices.index(region_idx)
        point_areas[point_idx] = areas[area_idx]

df['areas'] = point_areas

adata_raw = sc.read_h5ad('../cytospace_Luz/overall_merfish.h5ad')
adata = adata_raw[adata_raw.obs['sample_id']=='R77_4C4'].copy()  # 72962 × 238
adata.obs.index = [i.split('-')[0] for i in adata.obs.index]

adata.obs['areas'] = np.nan
df['cell_id'] = df['cell_id'].astype(str)
idx = np.intersect1d(df['cell_id'], adata.obs.index)
adata.obs.loc[idx, 'areas'] = df[df['cell_id'].isin(idx)]['areas'].values
adata.write('overall_merfish_areas.h5ad')


## single cell multiome (RNA+ATAC)
## Targeting immune–fibroblast cell communication in heart failure (Nature, 2024)
## GSE270788

## workdir: /fs/home/jiluzhang/spatial_mechano/pipeline_3/scMultiome

# MA5,Donor  √
# MA6,Donor  √
# MA7,ICM
# MA8,ICM
# MA9,AMI
# MA10,NICM
# MA11,AMI
# MA13,Donor  √
# MA14,NICM
# MA19,AMI
# MA20,AMI
# MA22,NICM
# MA23,Donor  √
# MA24,NICM
# MA25,NICM
# MA26,Donor  √
# MA27,ICM
# MA28,NICM
# MA29,Donor  √
# MA30,ICM
# MA31,AMI
# MA32,ICM
# MA33,Donor  √

# MA5:   wget -c ftp://ftp.ncbi.nlm.nih.gov/geo/samples/GSM8352nnn/GSM8352048/suppl/GSM8352048_MA5_filtered_feature_bc_matrix.h5
# MA6:   wget -c ftp://ftp.ncbi.nlm.nih.gov/geo/samples/GSM8352nnn/GSM8352049/suppl/GSM8352049_MA6_filtered_feature_bc_matrix.h5
# MA13:  wget -c https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM8352nnn/GSM8352055/suppl/GSM8352055_MA13_filtered_feature_bc_matrix.h5
# MA23:  wget -c https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM8352nnn/GSM8352060/suppl/GSM8352060_MA23_filtered_feature_bc_matrix.h5
# MA26:  wget -c https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM8352nnn/GSM8352063/suppl/GSM8352063_MA26_filtered_feature_bc_matrix.h5
# MA29:  wget -c https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM8352nnn/GSM8352066/suppl/GSM8352066_filtered_feature_bc_matrix.h5
# MA33:  wget -c https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM8352nnn/GSM8352070/suppl/GSM8352070_MA33_filtered_feature_bc_matrix.h5
# mv GSM8352048_MA5_filtered_feature_bc_matrix.h5 Donor_1.h5
# mv GSM8352049_MA6_filtered_feature_bc_matrix.h5 Donor_2.h5
# mv GSM8352055_MA13_filtered_feature_bc_matrix.h5 Donor_3.h5
# mv GSM8352060_MA23_filtered_feature_bc_matrix.h5 Donor_4.h5
# mv GSM8352063_MA26_filtered_feature_bc_matrix.h5 Donor_5.h5
# mv GSM8352066_filtered_feature_bc_matrix.h5 Donor_6.h5
# mv GSM8352070_MA33_filtered_feature_bc_matrix.h5 Donor_7.h5

#### split RNA & ATAC
import scanpy as sc
import pandas as pd
import anndata as ad
import pybedtools
import numpy as np
from tqdm import tqdm
from scipy.sparse import csr_matrix

## rna
d_1 = sc.read_10x_h5('Donor_1.h5', gex_only=False)  # 2197 × 125496
d_2 = sc.read_10x_h5('Donor_2.h5', gex_only=False)  # 2298 × 168850
d_3 = sc.read_10x_h5('Donor_3.h5', gex_only=False)  # 649 × 102376
d_4 = sc.read_10x_h5('Donor_4.h5', gex_only=False)  # 5226 × 162267
d_5 = sc.read_10x_h5('Donor_5.h5', gex_only=False)  # 1846 × 174078
d_6 = sc.read_10x_h5('Donor_6.h5', gex_only=False)  # 7692 × 134988
d_7 = sc.read_10x_h5('Donor_7.h5', gex_only=False)  # 5586 × 156529

d_1.var_names_make_unique()
d_2.var_names_make_unique()
d_3.var_names_make_unique()
d_4.var_names_make_unique()
d_5.var_names_make_unique()
d_6.var_names_make_unique()
d_7.var_names_make_unique()

# Donor_1  MA5   s1
# Donor_2  MA6   s2
# Donor_3  MA13  s8
# Donor_4  MA23  s13
# Donor_5  MA26  s16
# Donor_6  MA29  s19
# Donor_7  MA33  s23
d_1.obs_names = ['s1_'+i for i in d_1.obs_names]
d_2.obs_names = ['s2_'+i for i in d_2.obs_names]
d_3.obs_names = ['s8_'+i for i in d_3.obs_names]
d_4.obs_names = ['s13_'+i for i in d_4.obs_names]
d_5.obs_names = ['s16_'+i for i in d_5.obs_names]
d_6.obs_names = ['s19_'+i for i in d_6.obs_names]
d_7.obs_names = ['s23_'+i for i in d_7.obs_names]

rna = ad.concat([d_1[:, d_1.var['feature_types']=='Gene Expression'], 
                 d_2[:, d_2.var['feature_types']=='Gene Expression'],
                 d_3[:, d_3.var['feature_types']=='Gene Expression'],
                 d_4[:, d_4.var['feature_types']=='Gene Expression'],
                 d_5[:, d_5.var['feature_types']=='Gene Expression'],
                 d_6[:, d_6.var['feature_types']=='Gene Expression'],
                 d_7[:, d_7.var['feature_types']=='Gene Expression']])   # 25494 × 36601

rna.var['gene_ids'] = d_1[:, d_1.var['feature_types']=='Gene Expression'].var['gene_ids'].values
rna.write('Donors_rna.h5ad')


## atac
cCREs = pd.read_table('human_cCREs.bed', names=['chr', 'start', 'end'])
cCREs['idx'] = range(cCREs.shape[0])
cCREs_bed = pybedtools.BedTool.from_dataframe(cCREs)

def map_to_cre(d_1):
    atac = d_1[:, d_1.var['feature_types']=='Peaks'].copy()
    atac.X = atac.X.toarray()
    atac.X[atac.X>0] = 1  # binarization
    
    peaks = pd.DataFrame({'id': atac.var_names})
    peaks['chr'] = peaks['id'].map(lambda x: x.split(':')[0])
    peaks['start'] = peaks['id'].map(lambda x: x.split(':')[1].split('-')[0])
    peaks['end'] = peaks['id'].map(lambda x: x.split(':')[1].split('-')[1])
    peaks.drop(columns='id', inplace=True)
    peaks['idx'] = range(peaks.shape[0])
    peaks_bed = pybedtools.BedTool.from_dataframe(peaks)
    idx_map = peaks_bed.intersect(cCREs_bed, wa=True, wb=True).to_dataframe().iloc[:, [3, 7]]
    idx_map.columns = ['peaks_idx', 'cCREs_idx']
    
    m = np.zeros([atac.n_obs, cCREs_bed.to_dataframe().shape[0]], dtype='float32')
    for i in tqdm(range(atac.X.shape[0]), ncols=80, desc='Aligning ATAC peaks'):
        m[i][idx_map[idx_map['peaks_idx'].isin(peaks.iloc[np.nonzero(atac.X[i])]['idx'])]['cCREs_idx']] = 1
    
    atac_new = sc.AnnData(m, obs=atac.obs, var=pd.DataFrame({'gene_ids': cCREs['chr']+':'+cCREs['start'].map(str)+'-'+cCREs['end'].map(str), 'feature_types': 'Peaks'}))
    atac_new.var.index = atac_new.var['gene_ids'].values
    atac_new.X = csr_matrix(atac_new.X)  # speed up concatenating
    
    return atac_new

atac_d1 = map_to_cre(d_1)  # 2197 × 1033239
atac_d2 = map_to_cre(d_2)  # 2298 × 1033239
atac_d3 = map_to_cre(d_3)  # 649  × 1033239
atac_d4 = map_to_cre(d_4)  # 5226 × 1033239
atac_d5 = map_to_cre(d_5)  # 1846 × 1033239
atac_d6 = map_to_cre(d_6)  # 7692 × 1033239
atac_d7 = map_to_cre(d_7)  # 5586 × 1033239

atac = ad.concat([atac_d1, atac_d2, atac_d3, atac_d4, atac_d5, atac_d6, atac_d7])   # 25494 × 1033239   very fast for csr matrix
atac.var['gene_ids'] = atac_d1.var['gene_ids'].values

sc.pp.filter_genes(atac, min_cells=10)  # 25494 × 296743
atac.write('Donors_atac.h5ad')

## prepare files for cytospace
gex = pd.DataFrame(rna.X.toarray().T)
gex.index = rna.var.index.values
gex.columns = rna.obs.index.values
gex.index.name = 'GENES'

## meta
meta = pd.read_csv('GSE270788_metadata.csv.gz')
ma = meta[meta['sample'].isin(['MA5', 'MA6', 'MA13', 'MA23', 'MA26', 'MA29', 'MA33'])]  # 10727 × 25
ma = ma[['barcode', 'cell.type']]
ma.columns = ['Cell IDs', 'CellType']
ma.to_csv('scrna_ct.tsv', index=False, sep='\t')

gex.loc[:, ma['Cell IDs'].values].to_csv('scrna_gex.tsv', sep='\t')  # select cells with annotations
# 36601 genes & 10727 cells

# Adipocyte
# Cardiomyocyte
# Endocardium
# Endothelium
# Fibroblast
# Glia
# Lymphatic
# Mast
# Myeloid
# Pericyte
# TNKCells

#### unify cell annotation
import pandas as pd

## scrna cell type
scrna_ct = pd.read_table('scrna_ct.tsv')
scrna_ct['CellType'] = scrna_ct['CellType'].replace({'Adipocyte':'Epicardium', 'Cardiomyocyte':'Cardiomyocyte', 'Endocardium':'Endocardium',
                                                     'Endothelium':'Endothelium', 'Fibroblast':'Fibroblast', 'Glia':'Neuron',
                                                     'Lymphatic':'Immune', 'Mast':'Immune', 'Myeloid':'Immune',
                                                     'Pericyte':'Pericyte', 'TNKCells':'Immune'})
scrna_ct.to_csv('scrna_ct_aligned.tsv', index=False, sep='\t')

## st cell type
st_ct = pd.read_table('st_ct.tsv')
st_ct['CellType'] = st_ct['CellType'].replace({'BEC':'Endothelium', 'EPDC':'Epicardium', 'Epicardial':'Epicardium',
                                               'LEC':'Endothelium', 'Neuronal':'Neuron', 'Pericyte':'Pericyte',
                                               'VEC':'Endocardium', 'VIC':'Endocardium', 'VSMC':'Endothelium', 'WBC':'Immune',
                                               'aCM-LA':'Cardiomyocyte', 'aCM-RA':'Cardiomyocyte', 'aEndocardial':'Endocardium',
                                               'aFibro':'Fibroblast', 'adFibro':'Fibroblast', 'ncCM-AVC-like':'Cardiomyocyte',
                                               'ncCM-IFT-like':'Cardiomyocyte', 'vCM-His-Purkinje':'Cardiomyocyte',
                                               'vCM-LV-AV':'Cardiomyocyte', 'vCM-LV-Compact':'Cardiomyocyte',
                                               'vCM-LV-Trabecular':'Cardiomyocyte', 'vCM-Proliferating':'Cardiomyocyte',
                                               'vCM-RV-AV':'Cardiomyocyte', 'vCM-RV-Compact':'Cardiomyocyte',
                                               'vCM-RV-Trabecular':'Cardiomyocyte', 'vEndocardial':'Endocardium', 'vFibro':'Fibroblast',})
st_ct.to_csv('st_ct_aligned.tsv', index=False, sep='\t')


# scRNA cells: 10,727  &  st cells: 72,963
cytospace --single-cell \
          --scRNA-path ./scMultiome/scrna_gex.tsv \
          --cell-type-path ./scMultiome/scrna_ct_aligned.tsv \
          --st-path ./Merfish/st_gex.tsv \
          --coordinates-path ./Merfish/st_pos.tsv \
          --st-cell-type-path ./Merfish/st_ct_aligned.tsv \
          --output-folder cytospace_results_crc \
          --solver-method lap_CSPR \
          --number-of-selected-sub-spots 10000 \
          --number-of-processors 8   # ~20 min


#### test for all genes
import scanpy as sc
import pandas as pd
from tqdm import tqdm
import numpy as np

spatial_adata = sc.read_h5ad('./Merfish/overall_merfish_areas.h5ad')
spatial_adata.obs.index = ['cell_'+i for i in spatial_adata.obs.index]

scatac_adata = sc.read_h5ad('./scMultiome/Donors_atac.h5ad')
scatac_adata.obs['idx'] = range(scatac_adata.n_obs)  # for number index searching
m = scatac_adata.X.todense()

aligned = pd.read_csv('./cytospace_results_crc/assigned_locations.csv')
aligned.index = aligned['SpotID'].values


# for ct in spatial_adata.obs['populations'].drop_duplicates():

def test_cell(ct='VIC'):
    s_ratio_lst = []
    b_ratio_lst = []
    fc_lst = []
    ct_spatial_adata = spatial_adata[spatial_adata.obs.index[(spatial_adata.obs['populations']==ct) & (spatial_adata.obs['areas']<1000)]]
    s_x_idx = scatac_adata[aligned.loc[ct_spatial_adata.obs.index.values[ct_spatial_adata.obs['areas']<ct_spatial_adata.obs['areas'].quantile(0.2).item()]]['OriginalCID'].values].obs['idx'].values
    b_x_idx = scatac_adata[aligned.loc[ct_spatial_adata.obs.index.values[ct_spatial_adata.obs['areas']>ct_spatial_adata.obs['areas'].quantile(0.8).item()]]['OriginalCID'].values].obs['idx'].values

    for i in tqdm(range(scatac_adata.n_vars), ncols=80, desc=ct):
        s_peak_ratio = (m[s_x_idx, i].sum()/m[s_x_idx, i].shape[0]).item()
        b_peak_ratio = (m[b_x_idx, i].sum()/m[b_x_idx, i].shape[0]).item()
        s_ratio_lst.append(s_peak_ratio)
        b_ratio_lst.append(b_peak_ratio)
        fc_lst.append(np.log2((s_peak_ratio+0.00001)/(b_peak_ratio+0.00001)).item())
    
    res = pd.DataFrame({'ct':ct, 'peak':scatac_adata.var.index.values, 's_ratio':s_ratio_lst, 'b_ratio':b_ratio_lst, 'log2fc':fc_lst})
    return res

VIC = test_cell('VIC')

peaks = pd.DataFrame({'id': VIC[(VIC['log2fc']>0.3) & (VIC['s_ratio']>0.1)]['peak'].values})
peaks['chr'] = peaks['id'].map(lambda x: x.split(':')[0])
peaks['start'] = peaks['id'].map(lambda x: x.split(':')[1].split('-')[0])
peaks['end'] = peaks['id'].map(lambda x: x.split(':')[1].split('-')[1])
peaks.drop(columns='id', inplace=True)
peaks.to_csv('VIC_s_b_peaks.bed', sep='\t', index=False, header=False)

res.to_csv('log2fc_pvalue.txt', sep='\t', index=False)



























