## A computational pipeline for spatial mechano-transcriptomics (Nature Methods)
## Data: https://doi.org/10.5281/zenodo.13975707
## Code: https://doi.org/10.5281/zenodo.13975227

# wget -c https://zenodo.org/records/13975228/files/Code_Hallou_He_et%20al_BioRxiv_Aug_2023.zip?download=1
# conda create --name TensionMap python=3.9
# conda install ipython scipy matplotlib scikit-image
# pip install -i https://pypi.tuna.tsinghua.edu.cn/simple numpy==1.23.0
# conda install pandas scikit-learn

## Run TensionMap to infer cell pressure, cell junction tensions and cellular stress tensor
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

mask_raw = skimage.io.imread('/fs/home/jiluzhang/spatial_mechano/test/test.png', as_gray=True)  # 4096*4232

mask = mask_raw[:500, :500]

## convert segmentation mask to boundary mask
mask = mask.astype(int)
mask = skimage.segmentation.find_boundaries(mask, mode='subpixel')

## process mask and detect holes in tissue
mask = 1-mask
mask = skimage.measure.label(mask)
holes_mask = np.zeros_like(mask)
areas = pd.DataFrame(skimage.measure.regionprops_table(label_image=mask, properties=('label', 'area')))
thresh = np.mean(areas['area'].tolist())*2
for area in areas.iterrows():
    if area[1]['area'] > thresh:
        holes_mask[mask==area[1]['label']] = 1

plt.imshow(mask)
plt.savefig('test.pdf')
plt.close()






