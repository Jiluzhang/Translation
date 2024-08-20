## mouse kidney
## https://cf.10xgenomics.com/samples/cell-arc/2.0.2/M_Kidney_Chromium_Nuc_Isolation_vs_SaltyEZ_vs_ComplexTissueDP/\
##                            M_Kidney_Chromium_Nuc_Isolation_vs_SaltyEZ_vs_ComplexTissueDP_filtered_feature_bc_matrix.h5
## kidney.h5

dat = sc.read_10x_h5('kidney.h5', gex_only=False)           # 14652 × 97637
atac = dat[:, dat.var['feature_types']=='Peaks']            # 14652 × 65352
rna = dat[:, dat.var['feature_types']=='Gene Expression']   # 14652 × 32285


## Tabula Muris Senis: https://cellxgene.cziscience.com/collections/0b9d8a04-bb9d-44da-aa27-705bb65b54eb
# kidney dataset: wget -c https://datasets.cellxgene.cziscience.com/e896f2b3-e22b-42c0-81e8-9dd1b6e80d64.h5ad
# -> kidney_aging.h5ad
dat_all = sc.read_h5ad('kidney_aging.h5ad')
