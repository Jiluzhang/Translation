## cell line dataset

# wget -c https://static-content.springer.com/esm/art%3A10.1038%2Fs41467-018-08205-7/MediaObjects/41467_2018_8205_MOESM6_ESM.xls   # ATAC
# wget -c https://static-content.springer.com/esm/art%3A10.1038%2Fs41467-018-08205-7/MediaObjects/41467_2018_8205_MOESM7_ESM.xls   # RNA

## xls to h5ad
import pandas as pd
import scanpy as sc

rna_dat = pd.read_table('41467_2018_8205_MOESM7_ESM.xls')  # 49059 x 549
rna = sc.AnnData(rna_dat.values.T, var=pd.DataFrame({'gene_ids': rna_dat.index.values, 'feature_types': 'Peaks'}))
rna.obs.index = rna_dat.columns.values
rna.var.index = rna.var['gene_ids']
