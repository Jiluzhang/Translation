## workdir: /fs/home/jiluzhang/Nature_methods/Figure_2/pan_cancer/cisformer/CTHRC1

import scanpy as sc
import harmonypy as hm
import pandas as pd

rna = sc.read_h5ad('../rna.h5ad')                         # 144409 × 19160
fibro = rna[rna.obs['cell_anno']=='Fibroblasts'].copy()   # 28023 × 19160

sc.pp.normalize_total(fibro, target_sum=1e4)
sc.pp.log1p(fibro)
# fibro.obs['batch'] = [x.split('_')[0] for x in fibro.obs.index]  # exist samples with only one fibroblast
sc.pp.highly_variable_genes(fibro, batch_key='cancer_type')
fibro.raw = fibro
fibro = fibro[:, fibro.var.highly_variable]
sc.tl.pca(fibro)
ho = hm.run_harmony(fibro.obsm['X_pca'], fibro.obs, 'cancer_type')  # Converged after 2 iterations

fibro.obsm['X_pca_harmony'] = ho.Z_corr.T
sc.pp.neighbors(fibro, use_rep='X_pca_harmony')
sc.tl.umap(fibro)
sc.pl.umap(fibro, color='cancer_type', legend_fontsize='5', legend_loc='right margin', size=2, 
           title='', frameon=True, save='_batch_harmony_cancer_type.pdf')

sc.tl.leiden(fibro, resolution=0.25)
fibro.obs['leiden'].value_counts()
# 0    7571
# 1    7315
# 2    5481
# 3    3582
# 4    2297
# 5    1403
# 6     334
# 7      40

sc.tl.rank_genes_groups(fibro, 'leiden')
pd.DataFrame(fibro.uns['rank_genes_groups']['names']).head(20)

sc.pl.umap(fibro, color='CTHRC1', legend_fontsize='5', legend_loc='right margin', size=2,
           title='', frameon=True, save='_cthrc1.pdf')
sc.pl.umap(fibro, color='leiden', legend_fontsize='5', legend_loc='right margin', size=2,
           title='', frameon=True, save='_leiden.pdf')

fibro.write('fibro_res.h5ad')















