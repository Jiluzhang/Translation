#################### calculate Voronoi area for each cell ####################
## workdir: /fs/home/jiluzhang/spatial_mechano/pipeline_2
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

adata = sc.read_10x_h5('Donor_1.h5', gex_only=False)
## atac
atac = adata[:, adata.var['feature_types']=='Peaks'].copy()             # 2197 × 88895
atac.write('atac.h5ad')

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

d_1.obs_names = ['d-1_'+i for i in d_1.obs_names]
d_2.obs_names = ['d-2_'+i for i in d_2.obs_names]
d_3.obs_names = ['d-3_'+i for i in d_3.obs_names]
d_4.obs_names = ['d-4_'+i for i in d_4.obs_names]
d_5.obs_names = ['d-5_'+i for i in d_5.obs_names]
d_6.obs_names = ['d-6_'+i for i in d_6.obs_names]
d_7.obs_names = ['d-7_'+i for i in d_7.obs_names]

adata = ad.concat([d_1[:, d_1.var['feature_types']=='Gene Expression'], 
                   d_2[:, d_2.var['feature_types']=='Gene Expression'],
                   d_3[:, d_3.var['feature_types']=='Gene Expression'],
                   d_4[:, d_4.var['feature_types']=='Gene Expression'],
                   d_5[:, d_5.var['feature_types']=='Gene Expression'],
                   d_6[:, d_6.var['feature_types']=='Gene Expression'],
                   d_7[:, d_7.var['feature_types']=='Gene Expression']])   # 25494 × 36601

adata.var['gene_ids'] = d_1[:, d_1.var['feature_types']=='Gene Expression'].var['gene_ids'].values
adata.write('Donors.h5')


rna = adata[:, adata.var['feature_types']=='Gene Expression'].copy()    # 2197 × 36601
gex = pd.DataFrame(rna.X.toarray().T)
gex.index = rna.var.index.values
gex.columns = ['s1_'+i for i in rna.obs.index.values]
gex.index.name = 'GENES'

## meta
meta = pd.read_csv('GSE270788_metadata.csv.gz')
ma5 = meta[meta['sample']=='MA5']
ma5 = ma5[['barcode', 'cell.type']]
ma5.columns = ['Cell IDs', 'CellType']
ma5.to_csv('scrna_ct.tsv', index=False, sep='\t')

gex.loc[:, ma5['barcode'].values].to_csv('scrna_gex.tsv', sep='\t')  # select cells with annotations

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

































