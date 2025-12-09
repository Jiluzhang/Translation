#### A computational pipeline for spatial mechano-transcriptomics (Nature Methods)
## Data: https://doi.org/10.5281/zenodo.13975707
## Code: https://doi.org/10.5281/zenodo.13975227
## Github: https://github.com/Computational-Morphogenomics-Group/TensionMap

# wget -c https://zenodo.org/records/13975228/files/Code_Hallou_He_et%20al_BioRxiv_Aug_2023.zip?download=1
# conda create --name TensionMap python=3.9
# conda install ipython scipy matplotlib scikit-image
# pip install -i https://pypi.tuna.tsinghua.edu.cn/simple numpy==1.23.0
# conda install pandas scikit-learn
# conda install conda-forge::nlopt
# pip install pandas==1.5.3

#### Run TensionMap to infer cell pressure, cell junction tensions and cellular stress tensor
# segment.py (line 262: 'for nv in obj.V_df.at[v, 'nverts']:'  ->  'for nv in np.atleast_1d(obj.V_df.at[v, 'nverts']):')
# VMSI.py (line 198: 'self.vertices['fourfold'] = [(np.shape(nverts)[0] != 3) for nverts in self.vertices['nverts']]'
#                 -> 'self.vertices['fourfold'] = [(np.shape(np.atleast_1d(nverts))[0] != 3) for nverts in self.vertices['nverts']]')
#         (line 297: 'if (np.shape(self.vertices['nverts'][v])[0] > 3  and not (0 in self.vertices['ncells'][v])):'
#                 -> 'if (np.shape(np.atleast_1d(self.vertices['nverts'][v]))[0] > 3  and not (0 in self.vertices['ncells'][v])):')
#         (line 1345: 'stress_ellipse = patches.Ellipse(centroid, eigval[0], eigval[1], theta, fill=False, color='red', lw=3)'
#                 ->  'stress_ellipse = patches.Ellipse(xy=centroid, width=eigval[0], height=eigval[1], angle=theta, fill=False, color='red', lw=3)')

import sys
import scipy.stats
# Change path to location of TensionMap directory
sys.path.append('/fs/home/jiluzhang/spatial_mechano/TensionMap-main')
from src.segment import Segmenter
from src.VMSI import *
import numpy as np
import matplotlib.pyplot as plt
import skimage
import pandas as pd

# img = skimage.io.imread('/fs/home/jiluzhang/spatial_mechano/test/test.tiff')  # 935*1098
# vmsi_model = run_VMSI(img)

img = skimage.io.imread('/fs/home/jiluzhang/spatial_mechano/TensionMap_test/reproduce_data/dataset3/segmentation_final.tif')  # 4096*4232

mask = img[1500:2000, 1500:2000]  # mask = img[1800:2700, 50:950]
plt.imshow(mask)
plt.savefig('raw.pdf')
plt.close()

## convert segmentation mask to boundary mask
mask = mask.astype(int)
mask = skimage.segmentation.find_boundaries(mask, mode='subpixel')
# subpixel: return a doubled image, with pixels *between* the original pixels marked as boundary where appropriate.
plt.imshow(mask)
plt.savefig('boundaries.pdf')
plt.close()

## process mask and detect holes in tissue
mask = 1-mask
mask = skimage.measure.label(mask)  # label connected regions
holes_mask = np.zeros_like(mask)
areas = pd.DataFrame(skimage.measure.regionprops_table(label_image=mask, properties=('label', 'area')))  # comput image properties
thresh = np.mean(areas['area'].tolist())*3  # holes are defined as tissue regions with area greater than 2x the mean cell area
for area in areas.iterrows():
    if area[1]['area'] > thresh:
        holes_mask[mask==area[1]['label']] = 1
# Holes denoted by value 1
# Segmented cells and boundaries denoted by value 0

plt.imshow(mask)
plt.savefig('mask.pdf')
plt.close()

plt.imshow(holes_mask)
plt.savefig('holes_mask.pdf')
plt.close()

## Perform force inference on mask
model = run_VMSI(mask, is_labelled=True, holes_mask=holes_mask, tile=False, verbose=False)

model.plot(['pressure'], mask, size=10, file='pressure.pdf')
model.plot(['tension'], line_thickness=15, size=10, file='tension.pdf')
model.plot(['stress', 'cap'], mask, size=10, file='stress.pdf')

results, neighbours = model.output_results(neighbours=True)
results.to_csv('results.txt', sep='\t')



#### CytoSPACE (Nature Biotechnology: High-resolution alignment of single-cell and spatial transcriptomes with CytoSPACE)
## github: https://github.com/digitalcytometry/cytospace

# wget -c https://codeload.github.com/digitalcytometry/cytospace/zip/refs/heads/main
# conda create --name cytospace python=3.9
# conda activate cytospace
# pip install .
# pip install -i https://pypi.tuna.tsinghua.edu.cn/simple scanpy
# conda install -c conda-forge datatable
# pip install -i https://pypi.tuna.tsinghua.edu.cn/simple ortools==9.3.10497
# pip install -i https://pypi.tuna.tsinghua.edu.cn/simple lapjv==1.3.14

cytospace --single-cell \
          --scRNA-path HumanColonCancerPatient2_scRNA_expressions_cytospace.tsv \
          --cell-type-path HumanColonCancerPatient2_scRNA_annotations_cytospace.tsv \
          --st-path HumanColonCancerPatient2_ST_expressions_cytospace.tsv \
          --coordinates-path HumanColonCancerPatient2_ST_coordinates_cytospace.tsv \
          --st-cell-type-path HumanColonCancerPatient2_ST_celltypes_cytospace.tsv \
          --output-folder cytospace_results_crc \
          --solver-method lap_CSPR \
          --number-of-selected-sub-spots 10000 \
          --number-of-processors 5

## st data source: https://datadryad.org/dataset/doi:10.5061/dryad.w0vt4b8vp#readme
#### prepare input data for cytospace
import pandas as pd
import scanpy as sc

## st gex data
gex = pd.read_csv('R1_R77_4C4_single_cell_raw_counts.csv')
gex = gex.T
gex.columns = ['cell_'+str(i) for i in gex.iloc[0, ]]
gex = gex.iloc[1:]
gex.index.name = 'GENES'

## st pos data
pos = pd.read_csv('R1_R77_4C4_cell_metadata.csv')
pos = pos[pos['status']=='ok']
pos['SpotID'] = ['cell_'+str(i) for i in pos['cell_id']]
pos.rename(columns={'global_x':'X', 'global_y':'Y'}, inplace=True)
pos = pos[['SpotID', 'X', 'Y']]
pos.index = pos['SpotID'].values

## st cell type data
adata = sc.read_h5ad('overall_merfish.h5ad')
r77_4c4_adata = adata[adata.obs['sample_id']=='R77_4C4'].copy()  # 72962 × 238
ct = pd.DataFrame({'SpotID':['cell_'+i.split('-')[0] for i in r77_4c4_adata.obs.index],
                   'CellType':r77_4c4_adata.obs['populations'].values})
ct.to_csv('st_ct.tsv', sep='\t', index=False)

# some cells are filtered 
gex.loc[:, ct['SpotID'].values].to_csv('st_gex.tsv', sep='\t')
pos.loc[ct['SpotID']].to_csv('st_pos.tsv', sep='\t', index=False)

# BEC: blood endothelial cell
# EPDC: epicardium-derived progenitor cell
# Epicardial: epicardial cell
# LEC: lymphatic endothelial cell
# Neuronal: neuronal cell
# Pericyte: pericyte
# VEC: valve endocardial cell
# VIC: valvular interstitial cell
# VSMC: vascular smooth muscle cells
# WBC: white blood cell
# aCM-LA: atrial cardiomyocyte (left atria)
# aCM-RA: atrial cardiomyocyte (right atria)
# aEndocardial: atria endocardial cell
# aFibro: atria firoblast
# adFibro: adventitial fibroblast
# ncCM-AVC-like: atrioventricular canal/node cardiomyocyte
# ncCM-IFT-like: inflow tract/pacemaker cardiomyocyte
# vCM-His-Purkinje: His-Purkinje ventricular cardiomyocyte 
# vCM-LV-AV: ventricular cardiomyocyte left ventricle atrioventricular
# vCM-LV-Compact: ventricular cardiomyocyte left ventricle compact
# vCM-LV-Trabecular: ventricular cardiomyocyte left ventricle trabecular
# vCM-Proliferating: ventricular cardiomyocyte proliferating
# vCM-RV-AV: ventricular cardiomyocyte right ventricle atrioventricular
# vCM-RV-Compact: ventricular cardiomyocyte right ventricle compact
# vCM-RV-Trabecular: ventricular cardiomyocyte right ventricle trabecular
# vEndocardial: ventricular endocardial cell
# vFibro: ventricular firoblast



#### Cellpose
## url: https://www.cellpose.org/
## github: https://github.com/MouseLand/cellpose
## tutorial: https://cellpose.readthedocs.io/en/latest/index.html

conda create --name cellpose python=3.10
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple cellpose

## mannually download model to /data/home/jiluzhang/.cellpose/models
wget -c hf-mirror.com/mouseland/cellpose-sam/resolve/main/cpsam  # not use huggingface.co



#### multiome dataset
## Targeting immune–fibroblast cell communication in heart failure (Nature, 2024)
## GSE270788

import scanpy as sc
import pandas as pd

adata = sc.read_10x_h5('GSM8352048_MA5_filtered_feature_bc_matrix.h5', gex_only=False)
## atac
atac = adata[:, adata.var['feature_types']=='Peaks'].copy()             # 2197 × 88895
atac.write('atac.h5ad')

## rna
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


# scRNA cells: 735  &  st cells: 72963
cytospace --single-cell \
          --scRNA-path scrna_gex.tsv \
          --cell-type-path scrna_ct_aligned.tsv \
          --st-path st_gex.tsv \
          --coordinates-path st_pos.tsv \
          --st-cell-type-path st_ct_aligned.tsv \
          --output-folder cytospace_results_crc \
          --solver-method lap_CSPR \
          --number-of-selected-sub-spots 10000 \
          --number-of-processors 5   # ~15 min


#################### infer mechano response elements ####################
#### workdir: /fs/home/jiluzhang/spatial_mechano/pipeline
head -n 1 R1_R77_4C4_assigned_barcodes.csv >> R1_R77_4C4_assigned_barcodes_fov_0_with_cell.csv
awk -F "," '{if($2==0 && $10!=0) print$0}' R1_R77_4C4_assigned_barcodes.csv >> R1_R77_4C4_assigned_barcodes_fov_0_with_cell.csv  # 50875
# fov: field of view
# maybe multiple fov

#### export tif image
import numpy as np
import pandas as pd
from PIL import Image
import math

df = pd.read_csv('R1_R77_4C4_assigned_barcodes_fov_0_with_cell.csv')

x_min, x_max = math.floor(df['global_x'].min()), math.ceil(df['global_x'].max())
y_min, y_max = math.floor(df['global_y'].min()), math.ceil(df['global_y'].max())
hist, x_edges, y_edges = np.histogram2d(df['global_x'], df['global_y'], bins=[500, 500],
                                        range=[[x_min, x_max], [y_min, y_max]])
# hist, x_edges, y_edges = np.histogram2d(df['global_x'], df['global_y'], bins=[x_max-x_min, y_max-y_min],
#                                         range=[[x_min, x_max], [y_min, y_max]])
img = Image.fromarray(hist)
img.save('R1_R77_4C4_assigned_barcodes_fov_0_with_cell.tif')

#### cell segmentationn
cellpose --image_path ./R1_R77_4C4_assigned_barcodes_fov_0_with_cell.tif --save_tif --no_npy --verbose  # ~30s

#### force inference
import sys
import scipy.stats
# Change path to location of TensionMap directory
sys.path.append('/fs/home/jiluzhang/spatial_mechano/TensionMap-main')
from src.segment import Segmenter
from src.VMSI import *
import numpy as np
import matplotlib.pyplot as plt
import skimage
import pandas as pd

img = skimage.io.imread('R1_R77_4C4_assigned_barcodes_fov_0_with_cell_cp_masks.tif')  # 500*500
img = skimage.segmentation.expand_labels(img, distance=500)

# # Check if we have disconnected regions that are too small
# labels = skimage.measure.label(img, connectivity=1)
# areas = pd.DataFrame(skimage.measure.regionprops_table(label_image=labels, properties=('label','area')))

# # Change '10' to some reasonable number of pixels below which we are certain isolated regions are artifacts
# min_size = 10
# if any(areas['area'] < min_size):
#     # Save image exterior
#     img_exterior = img==0
    
#     # Remove holes
#     # 1: set labels that correspond to isolated regions to 0
#     img[np.isin(labels, areas['label'].values[areas['area'] < min_size])] = 0
    
#     # 2: expand labels to fill holes
#     img = skimage.segmentation.expand_labels(img, distance=5)
    
#     # 3: set original image exterior back to 0
#     img[img_exterior] = 0
    
#     # 4: relabel image to make sure labels range from 1 to num_labels
#     img = skimage.measure.label(img)

## convert segmentation mask to boundary mask
mask = img.astype(int)
mask = skimage.segmentation.find_boundaries(mask, mode='subpixel')
# subpixel: return a doubled image, with pixels *between* the original pixels marked as boundary where appropriate.

## process mask and detect holes in tissue
mask = 1-mask
mask = skimage.measure.label(mask)  # label connected regions
holes_mask = np.zeros_like(mask)
areas = pd.DataFrame(skimage.measure.regionprops_table(label_image=mask, properties=('label', 'area')))  # comput image properties
thresh = np.mean(areas['area'].tolist())*3  # holes are defined as tissue regions with area greater than 2x the mean cell area
for area in areas.iterrows():
    if area[1]['area']>thresh:
        holes_mask[mask==area[1]['label']] = 1
# Holes denoted by value 1
# Segmented cells and boundaries denoted by value 0

## Perform force inference on mask
model = run_VMSI(mask, is_labelled=True, holes_mask=holes_mask, tile=False, verbose=False)

model.plot(['pressure'], mask, size=10, file='pressure.pdf')
model.plot(['tension'], line_thickness=15, size=10, file='tension.pdf')
model.plot(['stress', 'cap'], mask, size=10, file='stress.pdf')

results, neighbours = model.output_results(neighbours=True)
results.to_csv('results.txt', sep='\t')

plt.imshow(img)
plt.savefig('img.pdf')
plt.close()

plt.imshow(holes_mask)
plt.savefig('holes_mask.pdf')
plt.close()

