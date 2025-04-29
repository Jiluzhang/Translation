## workdir: /fs/home/jiluzhang/Nature_methods/Figure_1/ATAC2RNA

cp /fs/home/zouqihang/to_luz/k562_acc/CA_scatter_spearman.pdf_raw_data.tsv k562_archr_cisformer_spearman.txt
cp /fs/home/zouqihang/to_luz/k562_acc/CS_scatter_spearman.pdf_raw_data.tsv k562_scarlink_cisformer_spearman.txt

cp /fs/home/zouqihang/to_luz/pbmc_acc/CA_scatter_spearman.pdf_raw_data.tsv pbmc_archr_cisformer_spearman.txt
cp /fs/home/zouqihang/to_luz/pbmc_acc/CS_scatter_spearman.pdf_raw_data.tsv pbmc_scarlink_cisformer_spearman.txt

cp /fs/home/zouqihang/to_luz/tumor_b_acc/CA_scatter_spearman.pdf_raw_data.tsv tumor_b_archr_cisformer_spearman.txt
cp /fs/home/zouqihang/to_luz/tumor_b_acc/CS_scatter_spearman.pdf_raw_data.tsv tumor_b_scarlink_cisformer_spearman.txt

cp /fs/home/zouqihang/to_luz/k562_acc/CA_scatter_peason.pdf_raw_data.tsv k562_archr_cisformer_pearson.txt
cp /fs/home/zouqihang/to_luz/k562_acc/CS_scatter_peason.pdf_raw_data.tsv k562_scarlink_cisformer_pearson.txt

cp /fs/home/zouqihang/to_luz/pbmc_acc/CA_scatter_peason.pdf_raw_data.tsv pbmc_archr_cisformer_pearson.txt
cp /fs/home/zouqihang/to_luz/pbmc_acc/CS_scatter_peason.pdf_raw_data.tsv pbmc_scarlink_cisformer_pearson.txt

cp /fs/home/zouqihang/to_luz/tumor_b_acc/CA_scatter_peason.pdf_raw_data.tsv tumor_b_archr_cisformer_pearson.txt
cp /fs/home/zouqihang/to_luz/tumor_b_acc/CS_scatter_peason.pdf_raw_data.tsv tumor_b_scarlink_cisformer_pearson.txt


import pandas as pd
from plotnine import *
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['pdf.fonttype'] = 42

## spearman correlation
# k562 (not used)
k562_archr_cisformer = pd.read_table('k562_archr_cisformer_spearman.txt')
k562_archr_cisformer.columns = ['cisformer', 'archr', 'comp']
k562_archr_cisformer[k562_archr_cisformer['cisformer']!=0]['cisformer'].mean()              # 0.9996338235134935
k562_archr_cisformer[k562_archr_cisformer['archr']!=0]['archr'].mean()                      # 0.005798095658470096

k562_scarlink_cisformer = pd.read_table('k562_scarlink_cisformer_spearman.txt')
k562_scarlink_cisformer.columns = ['cisformer', 'scarlink', 'comp']
k562_scarlink_cisformer[k562_scarlink_cisformer['cisformer']!=0]['cisformer'].mean()        # 0.9996338235134935
k562_scarlink_cisformer[k562_scarlink_cisformer['scarlink']!=0]['scarlink'].mean()          # 0.04839020230554602

# pbmc
pbmc_archr_cisformer = pd.read_table('pbmc_archr_cisformer_spearman.txt')
pbmc_archr_cisformer.columns = ['cisformer', 'archr', 'comp']
pbmc_archr_cisformer[pbmc_archr_cisformer['cisformer']!=0]['cisformer'].mean()              # 0.9760733112541677
pbmc_archr_cisformer[pbmc_archr_cisformer['archr']!=0]['archr'].mean()                      # 0.04391130255656801

pbmc_scarlink_cisformer = pd.read_table('pbmc_scarlink_cisformer_spearman.txt')
pbmc_scarlink_cisformer.columns = ['cisformer', 'scarlink', 'comp']
pbmc_scarlink_cisformer[pbmc_scarlink_cisformer['cisformer']!=0]['cisformer'].mean()        # 0.976204637298592
pbmc_scarlink_cisformer[pbmc_scarlink_cisformer['scarlink']!=0]['scarlink'].mean()          # 0.33709043776532704

# tumor_b
tumor_b_archr_cisformer = pd.read_table('tumor_b_archr_cisformer_spearman.txt')
tumor_b_archr_cisformer.columns = ['cisformer', 'archr', 'comp']
tumor_b_archr_cisformer[tumor_b_archr_cisformer['cisformer']!=0]['cisformer'].mean()        # 0.950734398584293
tumor_b_archr_cisformer[tumor_b_archr_cisformer['archr']!=0]['archr'].mean()                # 0.06685202274617193

tumor_b_scarlink_cisformer = pd.read_table('tumor_b_scarlink_cisformer_spearman.txt')
tumor_b_scarlink_cisformer.columns = ['cisformer', 'scarlink', 'comp']
tumor_b_scarlink_cisformer[tumor_b_scarlink_cisformer['cisformer']!=0]['cisformer'].mean()  # 0.950734398584293
tumor_b_scarlink_cisformer[tumor_b_scarlink_cisformer['scarlink']!=0]['scarlink'].mean()    # 0.35068920573969

## barplot for average spearman correlation
dat = pd.DataFrame({'dataset': ['k562']*3+['pbmc']*3+['tumor_b']*3,
                    'algorithm': ['archr', 'scarlink', 'cisformer']*3,
                    'correlation': [0.0058, 0.0484, 0.9996, 0.0439, 0.3371, 0.9761, 0.0669, 0.3507, 0.9507]})
dat['dataset'] = pd.Categorical(dat['dataset'], categories=['k562', 'pbmc', 'tumor_b'])
dat['algorithm'] = pd.Categorical(dat['algorithm'], categories=['archr', 'scarlink', 'cisformer'])

p = ggplot(dat, aes(x='dataset', y='correlation', fill='algorithm')) + geom_bar(stat='identity', position=position_dodge(), width=0.75) +\
                                                                       scale_y_continuous(limits=[0, 1], breaks=np.arange(0, 1.1, 0.2)) + theme_bw()
p.save(filename='pbmc_tumor_b_spearman_archr_scarlink_cisformer.pdf', dpi=600, height=4, width=6)


## pearson correlation
# k562 (not used)
k562_archr_cisformer = pd.read_table('k562_archr_cisformer_pearson.txt')
k562_archr_cisformer.columns = ['cisformer', 'archr', 'comp']
k562_archr_cisformer[k562_archr_cisformer['cisformer']!=0]['cisformer'].mean()              # 0.9429113540264555
k562_archr_cisformer[k562_archr_cisformer['archr']!=0]['archr'].mean()                      # 0.000339627248936119

k562_scarlink_cisformer = pd.read_table('k562_scarlink_cisformer_pearson.txt')
k562_scarlink_cisformer.columns = ['cisformer', 'scarlink', 'comp']
k562_scarlink_cisformer[k562_scarlink_cisformer['cisformer']!=0]['cisformer'].mean()        # 0.9429113540264555
k562_scarlink_cisformer[k562_scarlink_cisformer['scarlink']!=0]['scarlink'].mean()          # 0.0808712320383338

# pbmc
pbmc_archr_cisformer = pd.read_table('pbmc_archr_cisformer_pearson.txt')
pbmc_archr_cisformer.columns = ['cisformer', 'archr', 'comp']
pbmc_archr_cisformer[pbmc_archr_cisformer['cisformer']!=0]['cisformer'].mean()              # 0.9394013579516567
pbmc_archr_cisformer[pbmc_archr_cisformer['archr']!=0]['archr'].mean()                      # 0.03008414050581014

pbmc_scarlink_cisformer = pd.read_table('pbmc_scarlink_cisformer_pearson.txt')
pbmc_scarlink_cisformer.columns = ['cisformer', 'scarlink', 'comp']
pbmc_scarlink_cisformer[pbmc_scarlink_cisformer['cisformer']!=0]['cisformer'].mean()        # 0.9395192286143785
pbmc_scarlink_cisformer[pbmc_scarlink_cisformer['scarlink']!=0]['scarlink'].mean()          # 0.35413627781392704

# tumor_b
tumor_b_archr_cisformer = pd.read_table('tumor_b_archr_cisformer_pearson.txt')
tumor_b_archr_cisformer.columns = ['cisformer', 'archr', 'comp']
tumor_b_archr_cisformer[tumor_b_archr_cisformer['cisformer']!=0]['cisformer'].mean()        # 0.8492236952119349
tumor_b_archr_cisformer[tumor_b_archr_cisformer['archr']!=0]['archr'].mean()                # 0.020564016117101236

tumor_b_scarlink_cisformer = pd.read_table('tumor_b_scarlink_cisformer_pearson.txt')
tumor_b_scarlink_cisformer.columns = ['cisformer', 'scarlink', 'comp']
tumor_b_scarlink_cisformer[tumor_b_scarlink_cisformer['cisformer']!=0]['cisformer'].mean()  # 0.8492236952119349
tumor_b_scarlink_cisformer[tumor_b_scarlink_cisformer['scarlink']!=0]['scarlink'].mean()    # 0.388961161420796

## barplot for average pearson correlation
dat = pd.DataFrame({'dataset': ['k562']*3+['pbmc']*3+['tumor_b']*3,
                    'algorithm': ['archr', 'scarlink', 'cisformer']*3,
                    'correlation': [0.0003, 0.0809, 0.9429, 0.0301, 0.3541, 0.9394, 0.0206, 0.3890, 0.8492]})
dat['dataset'] = pd.Categorical(dat['dataset'], categories=['k562', 'pbmc', 'tumor_b'])
dat['algorithm'] = pd.Categorical(dat['algorithm'], categories=['archr', 'scarlink', 'cisformer'])

p = ggplot(dat, aes(x='dataset', y='correlation', fill='algorithm')) + geom_bar(stat='identity', position=position_dodge(), width=0.75) +\
                                                                       scale_y_continuous(limits=[0, 1], breaks=np.arange(0, 1.1, 0.2)) + theme_bw()
p.save(filename='pbmc_tumor_b_pearson_archr_scarlink_cisformer.pdf', dpi=600, height=4, width=6)



## AMI & NMI & ARI & HOM
## PBMC
cp /fs/home/zouqihang/to_luz/pbmc_acc/groundtruth_of_pbmc_test_rna.h5ad rna_pbmc_true.h5ad
cp /fs/home/zouqihang/to_luz/pbmc_acc/archr_predicted_pbmc_test_rna.h5ad rna_pbmc_archr.h5ad
cp /fs/home/zouqihang/to_luz/pbmc_acc/scarlink_predicted_pbmc_test_rna.h5ad rna_pbmc_scarlink.h5ad
cp /fs/home/zouqihang/to_luz/pbmc_acc/cisformer_predicted_pbmc_test_rna.h5ad rna_pbmc_cisformer.h5ad

import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score, normalized_mutual_info_score, homogeneity_score

plt.rcParams['pdf.fonttype'] = 42

# rna_true = sc.read_h5ad('rna_pbmc_true.h5ad')
# rna_true.obs['cell_anno'].value_counts()
# # CD4T           698
# # CD14Mono       617
# # CD8T           561
# # B              192
# # CD16Mono       148
# # NK              83
# # pDC             20
# # Endothelial      9
# # Plasma           3

# rna_true.obs['cell_anno'][rna_true.obs['cell_anno']=='Plasma'] = 'B'
# rna_true = rna_true[rna_true.obs['cell_anno']!='Endothelial']
# # CD4T        698
# # CD14Mono    617
# # CD8T        561
# # B           195
# # CD16Mono    148
# # NK           83
# # pDC          20

rna_ref = sc.read_h5ad('/fs/home/jiluzhang/Nature_methods/Figure_1/scenario_1/rna.h5ad')
rna_true = sc.read_h5ad('rna_pbmc_true.h5ad')

idx = np.intersect1d(rna_ref.obs.index, rna_true.obs.index)

rna_true = rna_true[idx].copy()
rna_true.obs['cell_anno'] = rna_ref[idx].obs['cell_anno']  # 1954 × 38244
sc.pp.highly_variable_genes(rna_true)
rna_true = rna_true[:, rna_true.var.highly_variable]  # 1954 × 2857
sc.tl.pca(rna_true)
sc.pp.neighbors(rna_true)
sc.tl.umap(rna_true)

rcParams["figure.figsize"] = (2, 2)
sc.pl.umap(rna_true, color='cell_anno', legend_fontsize='5', legend_loc='right margin', size=2.5, 
           title='', frameon=True, save='_cell_anno_rna_true_pbmc.pdf')

## archr
rna_archr = sc.read_h5ad('rna_pbmc_archr.h5ad')
rna_archr = rna_archr[idx].copy()    # 1954 × 21369
sc.pp.highly_variable_genes(rna_archr)
rna_archr = rna_archr[:, rna_archr.var.highly_variable]  # 1954 × 5326
sc.tl.pca(rna_archr)
sc.pp.neighbors(rna_archr)
sc.tl.umap(rna_archr)
sc.tl.leiden(rna_archr)

rna_archr.obs['cell_anno'] = rna_true.obs['cell_anno']
sc.pl.umap(rna_archr, color='cell_anno', legend_fontsize='5', legend_loc='right margin', size=2.5, 
           title='', frameon=True, save='_cell_anno_rna_archr_pbmc.pdf')

adjusted_mutual_info_score(rna_true.obs['cell_anno'], rna_archr.obs['leiden'])     # 0.6702947344988958
adjusted_rand_score(rna_true.obs['cell_anno'], rna_archr.obs['leiden'])            # 0.527498181193253
homogeneity_score(rna_true.obs['cell_anno'], rna_archr.obs['leiden'])              # 0.6346569960985834
normalized_mutual_info_score(rna_true.obs['cell_anno'], rna_archr.obs['leiden'])   # 0.6743807458691784

## scarlink
rna_scarlink = sc.read_h5ad('rna_pbmc_scarlink.h5ad')
rna_scarlink = rna_scarlink[idx].copy()    # 1954 × 526
sc.pp.highly_variable_genes(rna_scarlink)
rna_scarlink = rna_scarlink[:, rna_scarlink.var.highly_variable]  # 1954 × 25
sc.tl.pca(rna_scarlink)
sc.pp.neighbors(rna_scarlink)
sc.tl.umap(rna_scarlink)
sc.tl.leiden(rna_scarlink)

rna_scarlink.obs['cell_anno'] = rna_true.obs['cell_anno']
sc.pl.umap(rna_scarlink, color='cell_anno', legend_fontsize='5', legend_loc='right margin', size=2.5, 
           title='', frameon=True, save='_cell_anno_rna_scarlink_pbmc.pdf')


adjusted_mutual_info_score(rna_true.obs['cell_anno'], rna_scarlink.obs['leiden'])     # 0.3389995946951028
adjusted_rand_score(rna_true.obs['cell_anno'], rna_scarlink.obs['leiden'])            # 0.27478068443487585
homogeneity_score(rna_true.obs['cell_anno'], rna_scarlink.obs['leiden'])              # 0.33714529881302363
normalized_mutual_info_score(rna_true.obs['cell_anno'], rna_scarlink.obs['leiden'])   # 0.3492819479771526

## cisformer
rna_cisformer = sc.read_h5ad('rna_pbmc_cisformer.h5ad')
rna_cisformer = rna_cisformer[idx].copy()    # 1954 × 38244

sc.pp.normalize_total(rna_cisformer, target_sum=1e4)
sc.pp.log1p(rna_cisformer)

sc.pp.highly_variable_genes(rna_cisformer)
rna_cisformer = rna_cisformer[:, rna_cisformer.var.highly_variable]  # 1954 × 2716
sc.tl.pca(rna_cisformer)
sc.pp.neighbors(rna_cisformer)
sc.tl.umap(rna_cisformer)
sc.tl.leiden(rna_cisformer)

rna_cisformer.obs['cell_anno'] = rna_true.obs['cell_anno']
sc.pl.umap(rna_cisformer, color='cell_anno', legend_fontsize='5', legend_loc='right margin', size=2.5, 
           title='', frameon=True, save='_cell_anno_rna_cisformer_pbmc.pdf')

adjusted_mutual_info_score(rna_true.obs['cell_anno'], rna_cisformer.obs['leiden'])     # 0.6945638077638026
adjusted_rand_score(rna_true.obs['cell_anno'], rna_cisformer.obs['leiden'])            # 0.608194907581094
homogeneity_score(rna_true.obs['cell_anno'], rna_cisformer.obs['leiden'])              # 0.6110892448421035
normalized_mutual_info_score(rna_true.obs['cell_anno'], rna_cisformer.obs['leiden'])   # 0.6980029861918517

#### plot metrics
from plotnine import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['pdf.fonttype'] = 42

dat = pd.DataFrame({'Method':['ArchR']*4 + ['SCARlink']*4 + ['Cisformer']*4,
                    'Metrics':['AMI', 'ARI', 'HOM', 'NMI']*3, 
                    'val':[0.670, 0.527, 0.635, 0.674,
                           0.339, 0.275, 0.337, 0.349,
                           0.695, 0.608, 0.611, 0.698]})

dat['Method'] = pd.Categorical(dat['Method'], categories=['ArchR', 'SCARlink', 'Cisformer'])
dat['Metrics'] = pd.Categorical(dat['Metrics'], categories=['AMI', 'NMI', 'ARI', 'HOM'])

p = ggplot(dat, aes(x='Metrics', y='val', fill='Method')) + geom_bar(stat='identity', position=position_dodge(), width=0.75) + ylab('') +\
                                                            scale_y_continuous(limits=[0, 0.8], breaks=np.arange(0, 0.8+0.1, 0.2)) + theme_bw()
p.save(filename='pbmc_ami_nmi_ari_hom.pdf', dpi=600, height=4, width=6)


## Tumor_B
cp /fs/home/zouqihang/to_luz/tumor_b_acc/groundtruth_of_tumor_B_test_rna.h5ad rna_tumor_b_true.h5ad
cp /fs/home/zouqihang/to_luz/tumor_b_acc/archr_predicted_tumor_B_test_rna.h5ad rna_tumor_b_archr.h5ad
cp /fs/home/zouqihang/to_luz/tumor_b_acc/scarlink_predicted_tumor_B_test_rna.h5ad rna_tumor_b_scarlink.h5ad
cp /fs/home/zouqihang/to_luz/tumor_b_acc/cisformer_predicted_tumor_B_test_rna.h5ad rna_tumor_b_cisformer.h5ad

import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score, normalized_mutual_info_score, homogeneity_score

plt.rcParams['pdf.fonttype'] = 42

rna_true = sc.read_h5ad('rna_tumor_b_true.h5ad')  # 2914 × 38244
# T cell           1796
# Tumor B cell      659
# Monocyte          319
# Normal B cell     119
# Other              21

rna_true = rna_true[rna_true.obs['cell_anno']!='Other'].copy()  # 2893 × 38244
rna_archr = sc.read_h5ad('rna_tumor_b_archr.h5ad')              # 2572 × 21369
rna_scarlink = sc.read_h5ad('rna_tumor_b_scarlink.h5ad')        # 12869 × 425
rna_cisformer = sc.read_h5ad('rna_tumor_b_cisformer.h5ad')      # 2914 × 38244

idx_1 = np.intersect1d(rna_true.obs.index, rna_archr.obs.index)           # 2555
idx_2 = np.intersect1d(rna_scarlink.obs.index, rna_cisformer.obs.index)   # 2572
idx = np.intersect1d(idx_1, idx_2)                                        # 2555

rna_true = rna_true[idx]

ax = sc.pl.violin(rna_true, keys='BANK1', groupby='cell_anno', rotation=90, stripplot=False, show=False)
ax.set_ylim(0-0.1, 4+0.1)
ax.figure.savefig('./figures/violin_BANK1_exp_true.pdf')

sc.pp.highly_variable_genes(rna_true)
rna_true = rna_true[:, rna_true.var.highly_variable]  # 2555 × 3881
sc.tl.pca(rna_true)
sc.pp.neighbors(rna_true)
sc.tl.umap(rna_true)

rcParams["figure.figsize"] = (2, 2)
sc.pl.umap(rna_true, color='cell_anno', legend_fontsize='5', legend_loc='right margin', size=2.5, 
           title='', frameon=True, save='_cell_anno_rna_true_tumor_b.pdf')


sc.pl.violin(rna_true, keys='MSI2', groupby='cell_anno', rotation=90, stripplot=False, save='_MSI2_exp.pdf')


## archr
rna_archr = rna_archr[idx].copy()    # 2555 × 21369
# sc.pp.highly_variable_genes(rna_archr)
# rna_archr = rna_archr[:, rna_archr.var.highly_variable]  # 1954 × 5326
sc.tl.pca(rna_archr)
sc.pp.neighbors(rna_archr)
sc.tl.umap(rna_archr)
sc.tl.leiden(rna_archr)

rna_archr.obs['cell_anno'] = rna_true.obs['cell_anno']
sc.pl.umap(rna_archr, color='cell_anno', legend_fontsize='5', legend_loc='right margin', size=2.5, 
           title='', frameon=True, save='_cell_anno_rna_archr_tumor_b.pdf')

ax = sc.pl.violin(rna_archr, keys='BANK1', groupby='cell_anno', rotation=90, stripplot=False, show=False)
ax.set_ylim(0-0.5, 12+0.5)
ax.figure.savefig('./figures/violin_BANK1_exp_archr.pdf')

adjusted_mutual_info_score(rna_true.obs['cell_anno'], rna_archr.obs['leiden'])     # 0.38189452322152706
adjusted_rand_score(rna_true.obs['cell_anno'], rna_archr.obs['leiden'])            # 0.19023829754875463
homogeneity_score(rna_true.obs['cell_anno'], rna_archr.obs['leiden'])              # 0.6400735035455338
normalized_mutual_info_score(rna_true.obs['cell_anno'], rna_archr.obs['leiden'])   # 0.3845730494963135

## scarlink
rna_scarlink = rna_scarlink[idx].copy()    # 2555 × 425
# sc.pp.highly_variable_genes(rna_scarlink)
# rna_scarlink = rna_scarlink[:, rna_scarlink.var.highly_variable]  # 1954 × 25
sc.tl.pca(rna_scarlink)
sc.pp.neighbors(rna_scarlink)
sc.tl.umap(rna_scarlink)
sc.tl.leiden(rna_scarlink)

rna_scarlink.obs['cell_anno'] = rna_true.obs['cell_anno']
sc.pl.umap(rna_scarlink, color='cell_anno', legend_fontsize='5', legend_loc='right margin', size=2.5, 
           title='', frameon=True, save='_cell_anno_rna_scarlink_tumor_b.pdf')

ax = sc.pl.violin(rna_scarlink, keys='BANK1', groupby='cell_anno', rotation=90, stripplot=False, show=False)
ax.set_ylim(2.15-0.025, 2.55+0.025)
ax.figure.savefig('./figures/violin_BANK1_exp_scarlink.pdf')

adjusted_mutual_info_score(rna_true.obs['cell_anno'], rna_scarlink.obs['leiden'])     # 0.149206539276603
adjusted_rand_score(rna_true.obs['cell_anno'], rna_scarlink.obs['leiden'])            # 0.11948664400466379
homogeneity_score(rna_true.obs['cell_anno'], rna_scarlink.obs['leiden'])              # 0.09591856009183418
normalized_mutual_info_score(rna_true.obs['cell_anno'], rna_scarlink.obs['leiden'])   # 0.15083369805862043

## cisformer
rna_cisformer = rna_cisformer[idx].copy()    # 2555 × 38244

sc.pp.normalize_total(rna_cisformer, target_sum=1e4)
sc.pp.log1p(rna_cisformer)

#sc.pp.highly_variable_genes(rna_cisformer)
#rna_cisformer = rna_cisformer[:, rna_cisformer.var.highly_variable]  # 1954 × 2716
sc.tl.pca(rna_cisformer)
sc.pp.neighbors(rna_cisformer)
sc.tl.umap(rna_cisformer)
sc.tl.leiden(rna_cisformer)

rna_cisformer.obs['cell_anno'] = rna_true.obs['cell_anno']
sc.pl.umap(rna_cisformer, color='cell_anno', legend_fontsize='5', legend_loc='right margin', size=2.5, 
           title='', frameon=True, save='_cell_anno_rna_cisformer_tumor_b.pdf')

ax = sc.pl.violin(rna_cisformer, keys='BANK1', groupby='cell_anno', rotation=90, stripplot=False, show=False)
ax.set_ylim(0-0.1, 5+0.1)
ax.figure.savefig('./figures/violin_BANK1_exp_cisformer.pdf')

adjusted_mutual_info_score(rna_true.obs['cell_anno'], rna_cisformer.obs['leiden'])     # 0.5433616233806915
adjusted_rand_score(rna_true.obs['cell_anno'], rna_cisformer.obs['leiden'])            # 0.2549840914883823
homogeneity_score(rna_true.obs['cell_anno'], rna_cisformer.obs['leiden'])              # 0.8860724898337444
normalized_mutual_info_score(rna_true.obs['cell_anno'], rna_cisformer.obs['leiden'])   # 0.5450310697696542

#### plot metrics
from plotnine import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['pdf.fonttype'] = 42

dat = pd.DataFrame({'Method':['ArchR']*4 + ['SCARlink']*4 + ['Cisformer']*4,
                    'Metrics':['AMI', 'ARI', 'HOM', 'NMI']*3, 
                    'val':[0.382, 0.190, 0.640, 0.385,
                           0.149, 0.119, 0.096, 0.151,
                           0.543, 0.255, 0.886, 0.545]})

dat['Method'] = pd.Categorical(dat['Method'], categories=['ArchR', 'SCARlink', 'Cisformer'])
dat['Metrics'] = pd.Categorical(dat['Metrics'], categories=['AMI', 'NMI', 'ARI', 'HOM'])

p = ggplot(dat, aes(x='Metrics', y='val', fill='Method')) + geom_bar(stat='identity', position=position_dodge(), width=0.75) + ylab('') +\
                                                            scale_y_continuous(limits=[0, 1.0], breaks=np.arange(0, 1.0+0.1, 0.2)) + theme_bw()
p.save(filename='tumor_b_ami_nmi_ari_hom.pdf', dpi=600, height=4, width=6)



############################################# set zero #############################################
## workdir: /fs/home/jiluzhang/Nature_methods/Figure_1/ATAC2RNA/set_zero

##### k562
cp /fs/home/zouqihang/to_luz/k562_acc/groundtruth_of_k562_test_rna.h5ad rna_k562_true.h5ad
cp /fs/home/zouqihang/to_luz/k562_acc/archr_predicted_k562_test_rna.h5ad rna_k562_archr.h5ad
cp /fs/home/zouqihang/to_luz/k562_acc/scarlink_predicted_k562_test_rna.h5ad rna_k562_scarlink.h5ad
cp /fs/home/zouqihang/to_luz/k562_acc/cisformer_predicted_k562_test_rna.h5ad rna_k562_cisformer.h5ad

import scanpy as sc
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import rcParams
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score, normalized_mutual_info_score, homogeneity_score

plt.rcParams['pdf.fonttype'] = 42
rcParams["figure.figsize"] = (2, 2)

rna_true = sc.read_h5ad('../rna_k562_true.h5ad')  # 1413 × 38244

rna_true = rna_true[rna_true.obs['cell_anno']!='Other'].copy()  # 1413 × 38244
rna_archr = sc.read_h5ad('../rna_k562_archr.h5ad')              # 1411 × 16473
rna_scarlink = sc.read_h5ad('../rna_k562_scarlink.h5ad')        # 7050 × 23
rna_cisformer = sc.read_h5ad('../rna_k562_cisformer.h5ad')      # 1413 × 38244

idx_1 = np.intersect1d(rna_true.obs.index, rna_archr.obs.index)           # 1411
idx_2 = np.intersect1d(rna_scarlink.obs.index, rna_cisformer.obs.index)   # 1411
idx = np.intersect1d(idx_1, idx_2)                                        # 1411

rna_true = rna_true[idx]  # 1411 × 38244

## archr
rna_archr = rna_archr[idx].copy()    # 1411 × 16473

var_idx = np.intersect1d(rna_archr.var.index, rna_true.var.index)

rna_true_archr = rna_true[:, var_idx]   # 1411 × 15380
rna_archr = rna_archr[:, var_idx]       # 1411 × 15380
rna_archr_X = rna_archr.X.toarray()
rna_archr_X[rna_true_archr.X.toarray()==0] = 0  # set non-zero to zero (based on the raw data)
rna_archr.X = csr_matrix(rna_archr_X)

archr_X = rna_archr.X.toarray()
true_X = rna_true_archr.X.toarray()

archr_pearson_r_lst = []
archr_spearman_r_lst = []
for i in tqdm(range(rna_archr.X.shape[1]), ncols=80):
    archr_i = archr_X[:, i]
    true_i = true_X[:, i]
    if len(set(true_i)) != 1:
        archr_pearson_r_lst.append(pearsonr(archr_X[:, i], true_X[:, i])[0])
        archr_spearman_r_lst.append(spearmanr(archr_X[:, i], true_X[:, i])[0])

archr_pearson_r_array = np.nan_to_num(np.array(archr_pearson_r_lst), nan=0)
archr_pearson_r_array.mean()                              # 0.10300548555056828
archr_pearson_r_array[archr_pearson_r_array!=0].mean()    # 0.27386225139265963

archr_spearman_r_array = np.nan_to_num(np.array(archr_spearman_r_lst), nan=0)
archr_spearman_r_array.mean()                              # 0.11457426978305996
archr_spearman_r_array[archr_spearman_r_array!=0].mean()   # 0.30462035402041426

## scarlink
rna_scarlink = sc.read_h5ad('../rna_k562_scarlink.h5ad')
rna_scarlink = rna_scarlink[idx].copy()    # 1411 × 23

var_idx = np.intersect1d(rna_scarlink.var.index, rna_true.var.index)

rna_true_scarlink = rna_true[:, var_idx]    # 1411 × 20
rna_scarlink = rna_scarlink[:, var_idx]     # 1411 × 20
rna_scarlink_X = rna_scarlink.X.toarray()
rna_scarlink_X[rna_true_scarlink.X.toarray()==0] = 0  # set non-zero to zero (based on the raw data)
rna_scarlink.X = csr_matrix(rna_scarlink_X)

scarlink_X = rna_scarlink.X.toarray()
true_X = rna_true_scarlink.X.toarray()

scarlink_pearson_r_lst = []
scarlink_spearman_r_lst = []
for i in tqdm(range(rna_scarlink.X.shape[1]), ncols=80):
    scarlink_i = scarlink_X[:, i]
    true_i = true_X[:, i]
    if len(set(true_i)) != 1:
        scarlink_pearson_r_lst.append(pearsonr(scarlink_i, true_i)[0])
        scarlink_spearman_r_lst.append(spearmanr(scarlink_i, true_i)[0])

scarlink_pearson_r_array = np.nan_to_num(np.array(scarlink_pearson_r_lst), nan=0)  # no nan
scarlink_pearson_r_array.mean()                                 # 0.8321069221484478
scarlink_pearson_r_array[scarlink_pearson_r_array!=0].mean()    # 0.8321069221484478

scarlink_spearman_r_array = np.nan_to_num(np.array(scarlink_spearman_r_lst), nan=0)
scarlink_spearman_r_array.mean()                                 # 0.9102476970829454
scarlink_spearman_r_array[scarlink_spearman_r_array!=0].mean()   # 0.9102476970829454

## cisformer
rna_cisformer = sc.read_h5ad('../rna_k562_cisformer.h5ad')
rna_cisformer = rna_cisformer[idx].copy()    # 1411 × 38244

var_idx = np.intersect1d(rna_cisformer.var.index, rna_true.var.index)

rna_true_cisformer = rna_true[:, var_idx]     # 1411 × 38244
rna_cisformer = rna_cisformer[:, var_idx]     # 1411 × 38244
rna_cisformer_X = rna_cisformer.X.toarray()
rna_cisformer_X[rna_true_cisformer.X.toarray()==0] = 0  # set non-zero to zero (based on the raw data)
rna_cisformer.X = csr_matrix(rna_cisformer_X)

cisformer_X = rna_cisformer.X.toarray()
true_X = rna_true_cisformer.X.toarray()

cisformer_pearson_r_lst = []
cisformer_spearman_r_lst = []
for i in tqdm(range(rna_cisformer.X.shape[1]), ncols=80):
    cisformer_i = cisformer_X[:, i]
    true_i = true_X[:, i]
    if len(set(true_i)) != 1:
        cisformer_pearson_r_lst.append(pearsonr(cisformer_i, true_i)[0])
        cisformer_spearman_r_lst.append(spearmanr(cisformer_i, true_i)[0])

cisformer_pearson_r_array = np.nan_to_num(np.array(cisformer_pearson_r_lst), nan=0)  
cisformer_pearson_r_array.mean()                                  # 0.9385810951699105
cisformer_pearson_r_array[cisformer_pearson_r_array!=0].mean()    # 0.9386545193344391

cisformer_spearman_r_array = np.nan_to_num(np.array(cisformer_spearman_r_lst), nan=0)
cisformer_spearman_r_array.mean()                                  # 0.9995556293783626
cisformer_spearman_r_array[cisformer_spearman_r_array!=0].mean()   # 0.9996338235134936

##### PBMC
import scanpy as sc
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import rcParams
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score, normalized_mutual_info_score, homogeneity_score

plt.rcParams['pdf.fonttype'] = 42
rcParams["figure.figsize"] = (2, 2)

rna_ref = sc.read_h5ad('/fs/home/jiluzhang/Nature_methods/Figure_1/scenario_1/rna.h5ad')
rna_true = sc.read_h5ad('../rna_pbmc_true.h5ad')

idx = np.intersect1d(rna_ref.obs.index, rna_true.obs.index)

rna_true = rna_true[idx].copy()
rna_true.obs['cell_anno'] = rna_ref[idx].obs['cell_anno']  # 1954 × 38244
sc.pp.highly_variable_genes(rna_true)
rna_true_hvg = rna_true[:, rna_true.var.highly_variable]  # 1954 × 2857
sc.tl.pca(rna_true_hvg)
sc.pp.neighbors(rna_true_hvg)
sc.tl.umap(rna_true_hvg)

sc.pl.umap(rna_true_hvg, color='cell_anno', legend_fontsize='5', legend_loc='right margin', size=2.5, 
           title='', frameon=True, save='_cell_anno_rna_true_pbmc.pdf')

## archr
rna_archr = sc.read_h5ad('../rna_pbmc_archr.h5ad')
rna_archr = rna_archr[idx].copy()    # 1954 × 21369

var_idx = np.intersect1d(rna_archr.var.index, rna_true.var.index)

rna_true_archr = rna_true[:, var_idx]   # 1954 × 20032
rna_archr = rna_archr[:, var_idx]       # 1954 × 20032
rna_archr_X = rna_archr.X.toarray()
rna_archr_X[rna_true_archr.X.toarray()==0] = 0  # set non-zero to zero (based on the raw data)
rna_archr.X = csr_matrix(rna_archr_X)

archr_X = rna_archr.X.toarray()
true_X = rna_true_archr.X.toarray()

archr_pearson_r_lst = []
archr_spearman_r_lst = []
for i in tqdm(range(rna_archr.X.shape[1]), ncols=80):
    archr_i = archr_X[:, i]
    true_i = true_X[:, i]
    if len(set(true_i)) != 1:
        archr_pearson_r_lst.append(pearsonr(archr_X[:, i], true_X[:, i])[0])
        archr_spearman_r_lst.append(spearmanr(archr_X[:, i], true_X[:, i])[0])

archr_pearson_r_array = np.nan_to_num(np.array(archr_pearson_r_lst), nan=0)
archr_pearson_r_array.mean()                              # 0.5695329945686082
archr_pearson_r_array[archr_pearson_r_array!=0].mean()    # 0.6351615575356329

archr_spearman_r_array = np.nan_to_num(np.array(archr_spearman_r_lst), nan=0)
archr_spearman_r_array.mean()                              # 0.6793824238309737
archr_spearman_r_array[archr_spearman_r_array!=0].mean()   # 0.7576691826426437

sc.pp.highly_variable_genes(rna_archr)
rna_archr = rna_archr[:, rna_archr.var.highly_variable]  # 1954 × 3437
sc.tl.pca(rna_archr)
sc.pp.neighbors(rna_archr)
sc.tl.umap(rna_archr)
sc.tl.leiden(rna_archr)

rna_archr.obs['cell_anno'] = rna_true.obs['cell_anno']
sc.pl.umap(rna_archr, color='cell_anno', legend_fontsize='5', legend_loc='right margin', size=2.5, 
           title='', frameon=True, save='_cell_anno_rna_archr_pbmc.pdf')

adjusted_mutual_info_score(rna_true.obs['cell_anno'], rna_archr.obs['leiden'])     # 0.6475745535954995
adjusted_rand_score(rna_true.obs['cell_anno'], rna_archr.obs['leiden'])            # 0.5121670959302606
homogeneity_score(rna_true.obs['cell_anno'], rna_archr.obs['leiden'])              # 0.6148277788204358
normalized_mutual_info_score(rna_true.obs['cell_anno'], rna_archr.obs['leiden'])   # 0.6525023664522298

## scarlink
rna_scarlink = sc.read_h5ad('../rna_pbmc_scarlink.h5ad')
rna_scarlink = rna_scarlink[idx].copy()    # 1954 × 526

var_idx = np.intersect1d(rna_scarlink.var.index, rna_true.var.index)

rna_true_scarlink = rna_true[:, var_idx]    # 1954 × 511
rna_scarlink = rna_scarlink[:, var_idx]     # 1954 × 511
rna_scarlink_X = rna_scarlink.X.toarray()
rna_scarlink_X[rna_true_scarlink.X.toarray()==0] = 0  # set non-zero to zero (based on the raw data)
rna_scarlink.X = csr_matrix(rna_scarlink_X)

scarlink_X = rna_scarlink.X.toarray()
true_X = rna_true_scarlink.X.toarray()

scarlink_pearson_r_lst = []
scarlink_spearman_r_lst = []
for i in tqdm(range(rna_scarlink.X.shape[1]), ncols=80):
    scarlink_i = scarlink_X[:, i]
    true_i = true_X[:, i]
    if len(set(true_i)) != 1:
        scarlink_pearson_r_lst.append(pearsonr(scarlink_i, true_i)[0])
        scarlink_spearman_r_lst.append(spearmanr(scarlink_i, true_i)[0])

scarlink_pearson_r_array = np.nan_to_num(np.array(scarlink_pearson_r_lst), nan=0)  # no nan
scarlink_pearson_r_array.mean()                                 # 0.9034644320076812
scarlink_pearson_r_array[scarlink_pearson_r_array!=0].mean()    # 0.9034644320076812

scarlink_spearman_r_array = np.nan_to_num(np.array(scarlink_spearman_r_lst), nan=0)
scarlink_spearman_r_array.mean()                                 # 0.9591336475270621
scarlink_spearman_r_array[scarlink_spearman_r_array!=0].mean()   # 0.9591336475270621

sc.pp.highly_variable_genes(rna_scarlink)
rna_scarlink = rna_scarlink[:, rna_scarlink.var.highly_variable]  # 1954 × 120
sc.tl.pca(rna_scarlink)
sc.pp.neighbors(rna_scarlink)
sc.tl.umap(rna_scarlink)
sc.tl.leiden(rna_scarlink)

rna_scarlink.obs['cell_anno'] = rna_true.obs['cell_anno']
sc.pl.umap(rna_scarlink, color='cell_anno', legend_fontsize='5', legend_loc='right margin', size=2.5, 
           title='', frameon=True, save='_cell_anno_rna_scarlink_pbmc.pdf')

adjusted_mutual_info_score(rna_true.obs['cell_anno'], rna_scarlink.obs['leiden'])     # 0.5276696872459717
adjusted_rand_score(rna_true.obs['cell_anno'], rna_scarlink.obs['leiden'])            # 0.3903706339968538
homogeneity_score(rna_true.obs['cell_anno'], rna_scarlink.obs['leiden'])              # 0.5490668421516376
normalized_mutual_info_score(rna_true.obs['cell_anno'], rna_scarlink.obs['leiden'])   # 0.5362409873308464

## cisformer
rna_cisformer = sc.read_h5ad('../rna_pbmc_cisformer.h5ad')
rna_cisformer = rna_cisformer[idx].copy()    # 1954 × 38244

var_idx = np.intersect1d(rna_cisformer.var.index, rna_true.var.index)

rna_true_cisformer = rna_true[:, var_idx]     # 1954 × 38244
rna_cisformer = rna_cisformer[:, var_idx]     # 1954 × 38244
rna_cisformer_X = rna_cisformer.X.toarray()
rna_cisformer_X[rna_true_cisformer.X.toarray()==0] = 0  # set non-zero to zero (based on the raw data)
rna_cisformer.X = csr_matrix(rna_cisformer_X)

cisformer_X = rna_cisformer.X.toarray()
true_X = rna_true_cisformer.X.toarray()

cisformer_pearson_r_lst = []
cisformer_spearman_r_lst = []
for i in tqdm(range(rna_cisformer.X.shape[1]), ncols=80):
    cisformer_i = cisformer_X[:, i]
    true_i = true_X[:, i]
    if len(set(true_i)) != 1:
        cisformer_pearson_r_lst.append(pearsonr(cisformer_i, true_i)[0])
        cisformer_spearman_r_lst.append(spearmanr(cisformer_i, true_i)[0])

cisformer_pearson_r_array = np.nan_to_num(np.array(cisformer_pearson_r_lst), nan=0)  
cisformer_pearson_r_array.mean()                                  # 0.959877108085182
cisformer_pearson_r_array[cisformer_pearson_r_array!=0].mean()    # 0.9599877376141494

cisformer_spearman_r_array = np.nan_to_num(np.array(cisformer_spearman_r_lst), nan=0)
cisformer_spearman_r_array.mean()                                  # 0.9882082012718311
cisformer_spearman_r_array[cisformer_spearman_r_array!=0].mean()   # 0.9883220960682667

sc.pp.normalize_total(rna_cisformer, target_sum=1e4)
sc.pp.log1p(rna_cisformer)

sc.pp.highly_variable_genes(rna_cisformer)
rna_cisformer = rna_cisformer[:, rna_cisformer.var.highly_variable]  # 1954 × 2716
sc.tl.pca(rna_cisformer)
sc.pp.neighbors(rna_cisformer)
sc.tl.umap(rna_cisformer)
sc.tl.leiden(rna_cisformer)

rna_cisformer.obs['cell_anno'] = rna_true.obs['cell_anno']
sc.pl.umap(rna_cisformer, color='cell_anno', legend_fontsize='5', legend_loc='right margin', size=2.5, 
           title='', frameon=True, save='_cell_anno_rna_cisformer_pbmc.pdf')

adjusted_mutual_info_score(rna_true.obs['cell_anno'], rna_cisformer.obs['leiden'])     # 0.6945638077638026
adjusted_rand_score(rna_true.obs['cell_anno'], rna_cisformer.obs['leiden'])            # 0.608194907581094
homogeneity_score(rna_true.obs['cell_anno'], rna_cisformer.obs['leiden'])              # 0.6110892448421035
normalized_mutual_info_score(rna_true.obs['cell_anno'], rna_cisformer.obs['leiden'])   # 0.6980029861918517

#### plot metrics
from plotnine import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['pdf.fonttype'] = 42

dat = pd.DataFrame({'Method':['ArchR']*4 + ['SCARlink']*4 + ['Cisformer']*4,
                    'Metrics':['AMI', 'ARI', 'HOM', 'NMI']*3, 
                    'val':[0.648, 0.512, 0.615, 0.653,
                           0.528, 0.390, 0.549, 0.536,
                           0.695, 0.608, 0.611, 0.698]})

dat['Method'] = pd.Categorical(dat['Method'], categories=['ArchR', 'SCARlink', 'Cisformer'])
dat['Metrics'] = pd.Categorical(dat['Metrics'], categories=['AMI', 'NMI', 'ARI', 'HOM'])

p = ggplot(dat, aes(x='Metrics', y='val', fill='Method')) + geom_bar(stat='identity', position=position_dodge(), width=0.75) + ylab('') +\
                                                            scale_y_continuous(limits=[0, 0.8], breaks=np.arange(0, 0.8+0.1, 0.2)) + theme_bw()
p.save(filename='pbmc_ami_nmi_ari_hom.pdf', dpi=600, height=4, width=6)


##### tumor_b
import scanpy as sc
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import rcParams
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score, normalized_mutual_info_score, homogeneity_score

plt.rcParams['pdf.fonttype'] = 42

rna_true = sc.read_h5ad('../rna_tumor_b_true.h5ad')  # 2914 × 38244

rna_true = rna_true[rna_true.obs['cell_anno']!='Other'].copy()     # 2893 × 38244
rna_archr = sc.read_h5ad('../rna_tumor_b_archr.h5ad')              # 2572 × 21369
rna_scarlink = sc.read_h5ad('../rna_tumor_b_scarlink.h5ad')        # 12869 × 425
rna_cisformer = sc.read_h5ad('../rna_tumor_b_cisformer.h5ad')      # 2914 × 38244

idx_1 = np.intersect1d(rna_true.obs.index, rna_archr.obs.index)           # 2555
idx_2 = np.intersect1d(rna_scarlink.obs.index, rna_cisformer.obs.index)   # 2572
idx = np.intersect1d(idx_1, idx_2)                                        # 2555

rna_true = rna_true[idx]

sc.pp.highly_variable_genes(rna_true)
rna_true_hvg = rna_true[:, rna_true.var.highly_variable]  # 2555 × 3881
sc.tl.pca(rna_true_hvg)
sc.pp.neighbors(rna_true_hvg)
sc.tl.umap(rna_true_hvg)

rcParams["figure.figsize"] = (2, 2)
sc.pl.umap(rna_true_hvg, color='cell_anno', legend_fontsize='5', legend_loc='right margin', size=2.5, 
           title='', frameon=True, save='_cell_anno_rna_true_tumor_b.pdf')

rcParams["figure.figsize"] = (4, 4)
ax = sc.pl.violin(rna_true, keys='BANK1', groupby='cell_anno', rotation=90, stripplot=False, show=False)
ax.set_ylim(0-0.1, 4+0.1)
ax.figure.savefig('./figures/violin_BANK1_exp_true.pdf')

rcParams["figure.figsize"] = (4, 4)
ax = sc.pl.violin(rna_true, keys='CD14', groupby='cell_anno', rotation=90, stripplot=False, show=False)
ax.set_ylim(0-0.1, 2+0.1)
ax.figure.savefig('./figures/violin_CD14_exp_true.pdf')

## archr
rna_archr = rna_archr[idx].copy()    # 2555 × 21369

var_idx = np.intersect1d(rna_archr.var.index, rna_true.var.index)

rna_true_archr = rna_true[:, var_idx]   # 2555 × 20032
rna_archr = rna_archr[:, var_idx]       # 2555 × 20032
rna_archr_X = rna_archr.X.toarray()
rna_archr_X[rna_true_archr.X.toarray()==0] = 0  # set non-zero to zero (based on the raw data)
rna_archr.X = csr_matrix(rna_archr_X)

archr_X = rna_archr.X.toarray()
true_X = rna_true_archr.X.toarray()

archr_pearson_r_lst = []
archr_spearman_r_lst = []
for i in tqdm(range(rna_archr.X.shape[1]), ncols=80):
    archr_i = archr_X[:, i]
    true_i = true_X[:, i]
    if len(set(true_i)) != 1:
        archr_pearson_r_lst.append(pearsonr(archr_X[:, i], true_X[:, i])[0])
        archr_spearman_r_lst.append(spearmanr(archr_X[:, i], true_X[:, i])[0])

archr_pearson_r_array = np.nan_to_num(np.array(archr_pearson_r_lst), nan=0)
archr_pearson_r_array.mean()                              # 0.5099477838977333
archr_pearson_r_array[archr_pearson_r_array!=0].mean()    # 0.5407221829899874

archr_spearman_r_array = np.nan_to_num(np.array(archr_spearman_r_lst), nan=0)
archr_spearman_r_array.mean()                              # 0.6889661200775031
archr_spearman_r_array[archr_spearman_r_array!=0].mean()   # 0.7305439423757933

sc.tl.pca(rna_archr)
sc.pp.neighbors(rna_archr)
sc.tl.umap(rna_archr)
sc.tl.leiden(rna_archr)

rna_archr.obs['cell_anno'] = rna_true.obs['cell_anno']

rcParams["figure.figsize"] = (2, 2)
sc.pl.umap(rna_archr, color='cell_anno', legend_fontsize='5', legend_loc='right margin', size=2.5, 
           title='', frameon=True, save='_cell_anno_rna_archr_tumor_b.pdf')

adjusted_mutual_info_score(rna_true.obs['cell_anno'], rna_archr.obs['leiden'])     # 0.49419923225287654
adjusted_rand_score(rna_true.obs['cell_anno'], rna_archr.obs['leiden'])            # 0.28518901678924535
homogeneity_score(rna_true.obs['cell_anno'], rna_archr.obs['leiden'])              # 0.7753650035606509
normalized_mutual_info_score(rna_true.obs['cell_anno'], rna_archr.obs['leiden'])   # 0.4961455282072116

rcParams["figure.figsize"] = (4, 4)
ax = sc.pl.violin(rna_archr, keys='BANK1', groupby='cell_anno', rotation=90, stripplot=False, show=False)
ax.set_ylim(0-0.5, 12+0.5)
ax.figure.savefig('./figures/violin_BANK1_exp_archr.pdf')

rcParams["figure.figsize"] = (4, 4)
ax = sc.pl.violin(rna_archr, keys='CD14', groupby='cell_anno', rotation=90, stripplot=False, show=False)
ax.set_ylim(0-0.5, 12+0.5)
ax.figure.savefig('./figures/violin_CD14_exp_archr.pdf')

## scarlink
rna_scarlink = sc.read_h5ad('../rna_tumor_b_scarlink.h5ad')
rna_scarlink = rna_scarlink[idx].copy()    # 2555 × 425

var_idx = np.intersect1d(rna_scarlink.var.index, rna_true.var.index)

rna_true_scarlink = rna_true[:, var_idx]    # 2555 × 400
rna_scarlink = rna_scarlink[:, var_idx]     # 2555 × 400
rna_scarlink_X = rna_scarlink.X.toarray()
rna_scarlink_X[rna_true_scarlink.X.toarray()==0] = 0  # set non-zero to zero (based on the raw data)
rna_scarlink.X = csr_matrix(rna_scarlink_X)

scarlink_X = rna_scarlink.X.toarray()
true_X = rna_true_scarlink.X.toarray()

scarlink_pearson_r_lst = []
scarlink_spearman_r_lst = []
for i in tqdm(range(rna_scarlink.X.shape[1]), ncols=80):
    scarlink_i = scarlink_X[:, i]
    true_i = true_X[:, i]
    if len(set(true_i)) != 1:
        scarlink_pearson_r_lst.append(pearsonr(scarlink_i, true_i)[0])
        scarlink_spearman_r_lst.append(spearmanr(scarlink_i, true_i)[0])

scarlink_pearson_r_array = np.nan_to_num(np.array(scarlink_pearson_r_lst), nan=0)  # no nan
scarlink_pearson_r_array.mean()                                 # 0.8835446845923399
scarlink_pearson_r_array[scarlink_pearson_r_array!=0].mean()    # 0.8835446845923399

scarlink_spearman_r_array = np.nan_to_num(np.array(scarlink_spearman_r_lst), nan=0)
scarlink_spearman_r_array.mean()                                 # 0.982475901435646
scarlink_spearman_r_array[scarlink_spearman_r_array!=0].mean()   # 0.982475901435646

sc.tl.pca(rna_scarlink)
sc.pp.neighbors(rna_scarlink)
sc.tl.umap(rna_scarlink)
sc.tl.leiden(rna_scarlink)

rna_scarlink.obs['cell_anno'] = rna_true.obs['cell_anno']

rcParams["figure.figsize"] = (2, 2)
sc.pl.umap(rna_scarlink, color='cell_anno', legend_fontsize='5', legend_loc='right margin', size=2.5, 
           title='', frameon=True, save='_cell_anno_rna_scarlink_tumor_b.pdf')

adjusted_mutual_info_score(rna_true.obs['cell_anno'], rna_scarlink.obs['leiden'])     # 0.149206539276603
adjusted_rand_score(rna_true.obs['cell_anno'], rna_scarlink.obs['leiden'])            # 0.11948664400466379
homogeneity_score(rna_true.obs['cell_anno'], rna_scarlink.obs['leiden'])              # 0.09591856009183418
normalized_mutual_info_score(rna_true.obs['cell_anno'], rna_scarlink.obs['leiden'])   # 0.15083369805862043

rcParams["figure.figsize"] = (4, 4)
ax = sc.pl.violin(rna_scarlink, keys='BANK1', groupby='cell_anno', rotation=90, stripplot=False, show=False)
ax.set_ylim(0-0.1, 3+0.1)
ax.figure.savefig('./figures/violin_BANK1_exp_scarlink.pdf')

rcParams["figure.figsize"] = (4, 4)
ax = sc.pl.violin(rna_scarlink, keys='CD14', groupby='cell_anno', rotation=90, stripplot=False, show=False)
ax.set_ylim(0-0.1, 3+0.1)
ax.figure.savefig('./figures/violin_CD14_exp_scarlink.pdf')  # CD14 is filtered out

## cisformer
rna_cisformer = sc.read_h5ad('../rna_tumor_b_cisformer.h5ad')
rna_cisformer = rna_cisformer[idx].copy()    # 2555 × 38244

var_idx = np.intersect1d(rna_cisformer.var.index, rna_true.var.index)

rna_true_cisformer = rna_true[:, var_idx]     # 2555 × 38244
rna_cisformer = rna_cisformer[:, var_idx]     # 2555 × 38244
rna_cisformer_X = rna_cisformer.X.toarray()
rna_cisformer_X[rna_true_cisformer.X.toarray()==0] = 0  # set non-zero to zero (based on the raw data)
rna_cisformer.X = csr_matrix(rna_cisformer_X)

cisformer_X = rna_cisformer.X.toarray()
true_X = rna_true_cisformer.X.toarray()

cisformer_pearson_r_lst = []
cisformer_spearman_r_lst = []
for i in tqdm(range(rna_cisformer.X.shape[1]), ncols=80):
    cisformer_i = cisformer_X[:, i]
    true_i = true_X[:, i]
    if len(set(true_i)) != 1:
        cisformer_pearson_r_lst.append(pearsonr(cisformer_i, true_i)[0])
        cisformer_spearman_r_lst.append(spearmanr(cisformer_i, true_i)[0])

cisformer_pearson_r_array = np.nan_to_num(np.array(cisformer_pearson_r_lst), nan=0)  
cisformer_pearson_r_array.mean()                                  # 0.8461461779019522
cisformer_pearson_r_array[cisformer_pearson_r_array!=0].mean()    # 0.8516131213870238

cisformer_spearman_r_array = np.nan_to_num(np.array(cisformer_spearman_r_lst), nan=0)
cisformer_spearman_r_array.mean()                                  # 0.9453532463992587
cisformer_spearman_r_array[cisformer_spearman_r_array!=0].mean()   # 0.9514611659366471

sc.pp.normalize_total(rna_cisformer, target_sum=1e4)
sc.pp.log1p(rna_cisformer)
sc.tl.pca(rna_cisformer)
sc.pp.neighbors(rna_cisformer)
sc.tl.umap(rna_cisformer)
sc.tl.leiden(rna_cisformer)

rna_cisformer.obs['cell_anno'] = rna_true.obs['cell_anno']

rcParams["figure.figsize"] = (2, 2)
sc.pl.umap(rna_cisformer, color='cell_anno', legend_fontsize='5', legend_loc='right margin', size=2.5, 
           title='', frameon=True, save='_cell_anno_rna_cisformer_tumor_b.pdf')

adjusted_mutual_info_score(rna_true.obs['cell_anno'], rna_cisformer.obs['leiden'])     # 0.5433616233806915
adjusted_rand_score(rna_true.obs['cell_anno'], rna_cisformer.obs['leiden'])            # 0.2549840914883823
homogeneity_score(rna_true.obs['cell_anno'], rna_cisformer.obs['leiden'])              # 0.8860724898337444
normalized_mutual_info_score(rna_true.obs['cell_anno'], rna_cisformer.obs['leiden'])   # 0.5450310697696542

rcParams["figure.figsize"] = (4, 4)
ax = sc.pl.violin(rna_cisformer, keys='BANK1', groupby='cell_anno', rotation=90, stripplot=False, show=False)
ax.set_ylim(0-0.1, 5+0.1)
ax.figure.savefig('./figures/violin_BANK1_exp_cisformer.pdf')

rcParams["figure.figsize"] = (4, 4)
ax = sc.pl.violin(rna_cisformer, keys='CD14', groupby='cell_anno', rotation=90, stripplot=False, show=False)
ax.set_ylim(0-0.1, 4+0.1)
ax.figure.savefig('./figures/violin_CD14_exp_cisformer.pdf')

#### plot metrics
from plotnine import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['pdf.fonttype'] = 42

dat = pd.DataFrame({'Method':['ArchR']*4 + ['SCARlink']*4 + ['Cisformer']*4,
                    'Metrics':['AMI', 'ARI', 'HOM', 'NMI']*3, 
                    'val':[0.494, 0.285, 0.775, 0.496,
                           0.149, 0.119, 0.096, 0.151,
                           0.543, 0.255, 0.886, 0.545]})

dat['Method'] = pd.Categorical(dat['Method'], categories=['ArchR', 'SCARlink', 'Cisformer'])
dat['Metrics'] = pd.Categorical(dat['Metrics'], categories=['AMI', 'NMI', 'ARI', 'HOM'])

p = ggplot(dat, aes(x='Metrics', y='val', fill='Method')) + geom_bar(stat='identity', position=position_dodge(), width=0.75) + ylab('') +\
                                                            scale_y_continuous(limits=[0, 1.0], breaks=np.arange(0, 1.0+0.1, 0.2)) + theme_bw()
p.save(filename='tumor_b_ami_nmi_ari_hom.pdf', dpi=600, height=4, width=6)


## barplot for average correlation
import pandas as pd
from plotnine import *
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['pdf.fonttype'] = 42

# pearson
dat = pd.DataFrame({'dataset': ['k562']*3+['pbmc']*3+['tumor_b']*3,
                    'algorithm': ['archr', 'scarlink', 'cisformer']*3,
                    'correlation': [0.274, 0.832, 0.939, 0.635, 0.903, 0.960, 0.541, 0.884, 0.852]})
dat['dataset'] = pd.Categorical(dat['dataset'], categories=['k562', 'pbmc', 'tumor_b'])
dat['algorithm'] = pd.Categorical(dat['algorithm'], categories=['archr', 'scarlink', 'cisformer'])

p = ggplot(dat, aes(x='dataset', y='correlation', fill='algorithm')) + geom_bar(stat='identity', position=position_dodge(), width=0.75) +\
                                                                       scale_y_continuous(limits=[0, 1], breaks=np.arange(0, 1.1, 0.2)) + theme_bw()
p.save(filename='k562_pbmc_tumor_b_pearson_archr_scarlink_cisformer.pdf', dpi=600, height=4, width=6)

# spearman
dat = pd.DataFrame({'dataset': ['k562']*3+['pbmc']*3+['tumor_b']*3,
                    'algorithm': ['archr', 'scarlink', 'cisformer']*3,
                    'correlation': [0.305, 0.910, 1.000, 0.758, 0.959, 0.988, 0.731, 0.982, 0.951]})
dat['dataset'] = pd.Categorical(dat['dataset'], categories=['k562', 'pbmc', 'tumor_b'])
dat['algorithm'] = pd.Categorical(dat['algorithm'], categories=['archr', 'scarlink', 'cisformer'])

p = ggplot(dat, aes(x='dataset', y='correlation', fill='algorithm')) + geom_bar(stat='identity', position=position_dodge(), width=0.75) +\
                                                                       scale_y_continuous(limits=[0, 1], breaks=np.arange(0, 1.1, 0.2)) + theme_bw()
p.save(filename='k562_pbmc_tumor_b_spearman_archr_scarlink_cisformer.pdf', dpi=600, height=4, width=6)


