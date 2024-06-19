############ BMMC dataset ############
# wget -c https://ftp.ncbi.nlm.nih.gov/geo/series/GSE194nnn/GSE194122/suppl/GSE194122_openproblems_neurips2021_multiome_BMMC_processed.h5ad.gz
import scanpy as sc
bmmc = sc.read_h5ad('GSE194122_openproblems_neurips2021_multiome_BMMC_processed.h5ad')
bmmc.obs.cell_type.value_counts()
# cell_type
# CD8+ T                 11589
# CD14+ Mono             10843
# NK                      6929
# CD4+ T activated        5526
# Naive CD20+ B           5052
# Erythroblast            4916
# CD4+ T naive            4398
# Transitional B          2810
# Proerythroblast         2300
# CD16+ Mono              1894
# B1 B                    1890
# Normoblast              1780
# Lymph prog              1779
# G/M prog                1203
# pDC                     1191
# HSC                     1072
# CD8+ T naive            1012
# MK/E prog                884
# cDC2                     859
# ILC                      835
# Plasma cell              379
# ID2-hi myeloid prog      108
