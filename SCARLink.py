## SCARlink
# https://github.com/snehamitra/SCARlink

#### Installation
conda create -n scarlink-env python=3.8.10
conda activate scarlink-env

conda install r-devtools  # install devtools first

install.packages('https://cran.r-project.org/src/contrib/Archive/Matrix/Matrix_1.6-4.tar.gz')
install.packages('https://cran.r-project.org/src/contrib/Archive/MASS/MASS_7.3-59.tar.gz')
install.packages('Seurat')  # 21: China (Nanjing) [https]

install.packages('BiocManager')

BiocManager::install("rhdf5")
BiocManager::install("GenomeInfoDbData")

BiocManager::install("chromVAR")
BiocManager::install("motifmatchr")


BiocManager::install("ComplexHeatmap")


1: In install.packages(...) :
  installation of package ‘DirichletMultinomial’ had non-zero exit status
2: In install.packages(...) :
  installation of package ‘TFBSTools’ had non-zero exit status
3: In install.packages(...) :
  installation of package ‘chromVAR’ had non-zero exit status


conda install -c conda-forge r-seurat 

r-devtools r-biocmanager
conda install -c bioconda bioconductor-rhdf5 \
      	      	          bioconductor-genomeinfodbdata \
                          bioconductor-chromvar \
                          bioconductor-motifmatchr \
                          bioconductor-complexheatmap
devtools::install_github("GreenleafLab/ArchR", ref="master", repos=BiocManager::repositories())






conda activate /data/home/zouqihang/miniconda3/envs/scarlink-env
## python predict_gex.py

import h5py
import numpy as np
from scipy.sparse import csr_matrix
import tensorflow as tf
import pandas as pd

def build_model(atac_shape, a):
    inputs = tf.keras.layers.Input(shape=(atac_shape,), name = 'inputA')
    out = tf.keras.layers.Dense(1, activation = tf.exp, name = 'rate', kernel_regularizer=tf.keras.regularizers.l2(a), kernel_constraint = tf.keras.constraints.NonNeg())(inputs)
    m = tf.keras.models.Model(inputs=inputs, outputs=out)
    return m

dat = h5py.File('coassay_matrix.h5')
cof = h5py.File('scarlink_out/coefficients_None.hd5')
gene_lst = list(cof['genes'].keys())

exp = pd.DataFrame()
for gene in gene_lst:
    atac_shape = dat[gene]['shape'][0]
    a = int(cof['genes'][gene].attrs['alpha'])  
    w = []
    w.append(cof['genes'][gene][:])
    w.append(np.array([cof['genes'][gene].attrs['intercept']]))
    
    x = csr_matrix((dat[gene]['data'][:], dat[gene]['indices'][:], dat[gene]['indptr'][:]), shape=dat[gene]['shape'][:][::-1])
    
    model = build_model(atac_shape, a)
    model.set_weights(w)
    pred_y = model.predict(x)
    exp[gene] = pred_y.flatten()
    print('Predict', gene, 'done#######')

exp.to_csv('predicted_gex.csv', index=False)
