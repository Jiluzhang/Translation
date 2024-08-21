## mouse kidney
## https://cf.10xgenomics.com/samples/cell-arc/2.0.2/M_Kidney_Chromium_Nuc_Isolation_vs_SaltyEZ_vs_ComplexTissueDP/\
##                            M_Kidney_Chromium_Nuc_Isolation_vs_SaltyEZ_vs_ComplexTissueDP_filtered_feature_bc_matrix.h5
## kidney.h5

import scanpy as sc

dat = sc.read_10x_h5('kidney.h5', gex_only=False)                  # 14652 × 97637
dat.var_names_make_unique()

rna = dat[:, dat.var['feature_types']=='Gene Expression'].copy()   # 14652 × 32285
rna.write('rna.h5ad')

atac = dat[:, dat.var['feature_types']=='Peaks'].copy()            # 14652 × 65352
atac.X[atac.X!=0] = 1
atac.write('atac.h5ad')


python split_train_val.py --RNA rna.h5ad --ATAC atac.h5ad --train_pct 0.9
python data_preprocess.py -r rna_train.h5ad -a atac_train.h5ad -s preprocessed_data_train --dt train --config rna2atac_config_train.yaml
python data_preprocess.py -r rna_val.h5ad -a atac_val.h5ad -s preprocessed_data_val --dt val --config rna2atac_config_val.yaml
nohup accelerate launch --config_file accelerator_config_train.yaml --main_process_port 29823 rna2atac_train.py --config_file rna2atac_config_train.yaml \
                        --train_data_dir ./preprocessed_data_train --val_data_dir ./preprocessed_data_val -n rna2atac_train > 20240820.log &   # 42406

import scanpy as sc
import numpy as np
rna = sc.read_h5ad('rna_val.h5ad')
np.count_nonzero(rna.X.toarray(), axis=1).max()  # 13845

import snapatac2 as snap
import scanpy as sc
atac = sc.read_h5ad('atac_val.h5ad')
snap.pp.select_features(atac)
snap.tl.spectral(atac)
snap.tl.umap(atac)
snap.pp.knn(atac)
snap.tl.leiden(atac)
sc.pl.umap(atac, color='leiden', legend_fontsize='7', legend_loc='right margin', size=5,
           title='', frameon=True, save='_atac_val_true.pdf')
atac.write('atac_val_with_leiden.h5ad')


python data_preprocess.py -r rna_val.h5ad -a atac_val.h5ad -s preprocessed_data_test --dt test --config rna2atac_config_test.yaml
accelerate launch --config_file accelerator_config_test.yaml --main_process_port 29822 rna2atac_test.py \
                  -d ./preprocessed_data_test \
                  -l save/2024-08-20_rna2atac_train_50/pytorch_model.bin --config_file rna2atac_config_test.yaml
python npy2h5ad.py
python csr2array.py --pred rna2atac_scm2m_raw.h5ad --true atac_val_with_leiden.h5ad
python cal_auroc_auprc.py --pred rna2atac_scm2m.h5ad --true atac_val_with_leiden.h5ad
python cal_cluster_plot.py --pred rna2atac_scm2m.h5ad --true atac_val_with_leiden.h5ad


## Tabula Muris Senis: https://cellxgene.cziscience.com/collections/0b9d8a04-bb9d-44da-aa27-705bb65b54eb
# kidney dataset: wget -c https://datasets.cellxgene.cziscience.com/e896f2b3-e22b-42c0-81e8-9dd1b6e80d64.h5ad
# -> kidney_aging.h5ad

import scanpy as sc

dat = sc.read_h5ad('kidney_aging.h5ad')

## map rna
ref_rna = sc.read_h5ad('rna.h5ad')
genes = ref_rna.var[['gene_ids']]
genes['gene_name'] = genes.index.values
genes.reset_index(drop=True, inplace=True)

genes = pd.read_table('human_genes.txt', names=['gene_ids', 'gene_name'])

rna_exp = pd.concat([rna.var, pd.DataFrame(rna.X.T, index=rna.var.index.values)], axis=1)
X_new = pd.merge(genes, rna_exp, how='left', on='gene_ids').iloc[:, 3:].T
X_new.fillna(value=0, inplace=True)

rna_new = sc.AnnData(X_new.values, obs=rna.obs, var=pd.DataFrame({'gene_ids': genes['gene_ids'], 'feature_types': 'Gene Expression'}))
rna_new.var.index = genes['gene_name'].values
rna_new.X = csr_matrix(rna_new.X)
rna_new.write('rna.h5ad')



ref_atac = sc.read_h5ad('atac.h5ad')



















