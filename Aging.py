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
                        --train_data_dir ./preprocessed_data_train --val_data_dir ./preprocessed_data_val -n rna2atac_train > 20240821.log &   
# accuracy improves until about 10 epoch
# 3199536


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
                  -l save/2024-08-21_rna2atac_train_56/pytorch_model.bin --config_file rna2atac_config_test.yaml
python npy2h5ad.py
python csr2array.py --pred rna2atac_scm2m_raw.h5ad --true atac_val_with_leiden.h5ad
python cal_auroc_auprc.py --pred rna2atac_scm2m.h5ad --true atac_val_with_leiden.h5ad
python cal_cluster_plot.py --pred rna2atac_scm2m.h5ad --true atac_val_with_leiden.h5ad


## Tabula Muris Senis: https://cellxgene.cziscience.com/collections/0b9d8a04-bb9d-44da-aa27-705bb65b54eb
# kidney dataset: wget -c https://datasets.cellxgene.cziscience.com/e896f2b3-e22b-42c0-81e8-9dd1b6e80d64.h5ad
# gene expression: https://cellxgene.cziscience.com/gene-expression
# -> kidney_aging.h5ad

import scanpy as sc
import pandas as pd
from scipy.sparse import csr_matrix
import numpy as np

rna = sc.read_h5ad('kidney_aging.h5ad')

## map rna
ref_rna = sc.read_h5ad('rna.h5ad')
genes = ref_rna.var[['gene_ids']]
genes['gene_name'] = genes.index.values
genes.reset_index(drop=True, inplace=True)

rna_exp = pd.concat([rna.var['feature_name'], pd.DataFrame(rna.raw.X.toarray().T, index=rna.var.index.values)], axis=1)
rna_exp['gene_ids'] = rna_exp.index.values
rna_exp.rename(columns={'feature_name': 'gene_name'}, inplace=True)
X_new = pd.merge(genes, rna_exp, how='left', on='gene_ids').iloc[:, 3:].T
X_new.fillna(value=0, inplace=True)

rna_new = sc.AnnData(X_new.values, obs=rna.obs, var=pd.DataFrame({'gene_ids': genes['gene_ids'], 'feature_types': 'Gene Expression'}))
rna_new.var.index = genes['gene_name'].values
rna_new.X = csr_matrix(rna_new.X)
rna_new.write('rna_unpaired.h5ad')

## pseudo atac
ref_atac = sc.read_h5ad('atac.h5ad')

atac_new = sc.AnnData(np.zeros([rna.n_obs, ref_atac.n_vars]),
                      obs=rna.obs, 
                      var=ref_atac.var)
atac_new.X[:, 0] = 1
atac_new.X = csr_matrix(atac_new.X)
atac_new.write('atac_unpaired.h5ad')


import scanpy as sc
import numpy as np
rna = sc.read_h5ad('rna_unpaired.h5ad')
np.count_nonzero(rna.X.toarray(), axis=1).max()  # 7617



python data_preprocess_unpaired.py -r rna_unpaired.h5ad -a atac_unpaired.h5ad -s preprocessed_data_unpaired --dt test --config rna2atac_config_unpaired.yaml
accelerate launch --config_file accelerator_config_test.yaml --main_process_port 29822 rna2atac_test.py \
                  -d ./preprocessed_data_unpaired \
                  -l save/2024-08-21_rna2atac_train_55/pytorch_model.bin --config_file rna2atac_config_unpaired.yaml   # ~ 0.5h

import scanpy as sc
import numpy as np
from scipy.sparse import csr_matrix
import snapatac2 as snap
import pandas as pd
from tqdm import tqdm

dat = np.load('predict_unpaired.npy')
dat[dat>0.5] = 1
dat[dat<=0.5] = 0
atac = sc.read_h5ad('atac_unpaired.h5ad')
atac.X = csr_matrix(dat)

snap.pp.select_features(atac)
snap.tl.spectral(atac, n_comps=100)  # snap.tl.spectral(atac, n_comps=100, weighted_by_sd=False)
snap.tl.umap(atac)

# sc.pl.umap(atac[atac.obs['cell_type'].isin(atac.obs.cell_type.value_counts()[:5].index), :], color='cell_type', legend_fontsize='7', legend_loc='right margin', size=10,
#            title='', frameon=True, save='_unpaired_tmp.pdf')

sc.pl.umap(atac[atac.obs['cell_type'].isin(['fenestrated cell', 'B cell']), :], color='cell_type', legend_fontsize='7', legend_loc='right margin', size=10,
            title='', frameon=True, save='_unpaired_luz.pdf')

sc.pl.umap(atac, color='cell_type', legend_fontsize='7', legend_loc='right margin', size=10,
           title='', frameon=True, save='_unpaired_tmp.pdf')

sc.pl.umap(atac, color='cell_type', legend_fontsize='7', legend_loc='right margin', size=10,
           title='', frameon=True, save='_unpaired.pdf')

sc.pl.umap(atac, color='cell_type', legend_fontsize='7', legend_loc='right margin', size=10,
           title='', frameon=True, save='_unpaired_n_comps_200.pdf')


def out_pred_bedgraph(cell_type='T cell'):
    atac_X = atac[atac.obs['cell_type']==cell_type, :].X.toarray().mean(axis=0)
    df = pd.DataFrame({'chr': atac.var['gene_ids'].map(lambda x: x.split(':')[0]).values,
                       'start': atac.var['gene_ids'].map(lambda x: x.split(':')[1].split('-')[0]).values,
                       'end': atac.var['gene_ids'].map(lambda x: x.split(':')[1].split('-')[1]).values,
                       'val': atac_X})
    df.to_csv(cell_type.replace(' ', '_')+'_atac_pred.bedgraph', index=False, header=False, sep='\t')

for cell_type in set(atac.obs.cell_type.values):
    out_pred_bedgraph(cell_type=cell_type)


def out_pred_norm_bedgraph(cell_type='T cell'):
    # atac_X_raw = atac[atac.obs['cell_type']==cell_type, :].X.toarray()
    # atac_X = (atac_X_raw/(atac_X_raw.sum(axis=1)[:, np.newaxis])).mean(axis=0)*10000
    atac_X_raw = atac[atac.obs['cell_type']==cell_type, :].X.toarray().sum(axis=0)
    atac_X = atac_X_raw/atac_X_raw.sum()*10000
    df = pd.DataFrame({'chr': atac.var['gene_ids'].map(lambda x: x.split(':')[0]).values,
                       'start': atac.var['gene_ids'].map(lambda x: x.split(':')[1].split('-')[0]).values,
                       'end': atac.var['gene_ids'].map(lambda x: x.split(':')[1].split('-')[1]).values,
                       'val': atac_X})
    df.to_csv(cell_type.replace(' ', '_')+'_atac_pred_norm.bedgraph', index=False, header=False, sep='\t')

for cell_type in tqdm(set(atac.obs.cell_type.values), ncols=80):
    out_pred_norm_bedgraph(cell_type=cell_type)

# atac_raw = atac.copy()
# sc.pp.filter_cells(atac, min_genes=2000)

# cell_type = 'kidney proximal convoluted tubule epithelial cell'
# m_01 = atac[(atac.obs['cell_type']==cell_type) & (atac.obs['age']=='1m')].X.toarray()  # 938
# m_01 = (m_01.sum(axis=0))/(m_01.sum())*10000
# m_03 = atac[(atac.obs['cell_type']==cell_type) & (atac.obs['age']=='3m')].X.toarray()  # 681
# m_03 = (m_03.sum(axis=0))/(m_03.sum())*10000
# m_18 = atac[(atac.obs['cell_type']==cell_type) & (atac.obs['age']=='18m')].X.toarray() # 1117
# m_18 = (m_18.sum(axis=0))/(m_18.sum())*10000
# m_21 = atac[(atac.obs['cell_type']==cell_type) & (atac.obs['age']=='21m')].X.toarray()  # 860
# m_21 = (m_21.sum(axis=0))/(m_21.sum())*10000
# m_30 = atac[(atac.obs['cell_type']==cell_type) & (atac.obs['age']=='30m')].X.toarray()  # 864
# m_30 = (m_30.sum(axis=0))/(m_30.sum())*10000
# m_all = np.vstack((m_01, m_03, m_18, m_21, m_30))

## LR model
cell_type = 'kidney proximal convoluted tubule epithelial cell'
m_01 = pd.DataFrame(atac[(atac.obs['cell_type']==cell_type) & (atac.obs['age']=='1m')].X.toarray())  # 938
m_01.columns = atac.var.index.values
m_01['age'] = 1
m_03 = pd.DataFrame(atac[(atac.obs['cell_type']==cell_type) & (atac.obs['age']=='3m')].X.toarray())  # 681
m_03.columns = atac.var.index.values
m_03['age'] = 3
m_18 = pd.DataFrame(atac[(atac.obs['cell_type']==cell_type) & (atac.obs['age']=='18m')].X.toarray()) # 1117
m_18.columns = atac.var.index.values
m_18['age'] = 18
m_21 = pd.DataFrame(atac[(atac.obs['cell_type']==cell_type) & (atac.obs['age']=='21m')].X.toarray())  # 860
m_21.columns = atac.var.index.values
m_21['age'] = 21
m_30 = pd.DataFrame(atac[(atac.obs['cell_type']==cell_type) & (atac.obs['age']=='30m')].X.toarray())  # 864
m_30.columns = atac.var.index.values
m_30['age'] = 30
m_all = pd.concat((m_01, m_03, m_18, m_21, m_30))

from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(m_all.iloc[:, :-1], m_all.iloc[:, -1])

peak_sorted = m_all.columns[np.argsort(model.coef_)]

df = pd.DataFrame({'idx':m_all.columns[:-1]})
df_idx = pd.DataFrame({'chr': df['idx'].map(lambda x: x.split(':')[0]).values,
                       'start': df['idx'].map(lambda x: x.split(':')[1].split('-')[0]).values,
                       'end': df['idx'].map(lambda x: x.split(':')[1].split('-')[1]).values,
                       'val': model.coef_})
df_idx.to_csv('atac_age_kidney_proximal_convoluted_tubule_epithelial_cell.bedgraph', sep='\t', index=None, header=None)

df_pos = pd.DataFrame({'idx':peak_sorted[-100:]})
df_pos_idx = pd.DataFrame({'chr': df_pos['idx'].map(lambda x: x.split(':')[0]).values,
                           'start': df_pos['idx'].map(lambda x: x.split(':')[1].split('-')[0]).values,
                           'end': df_pos['idx'].map(lambda x: x.split(':')[1].split('-')[1]).values})
df_pos_idx.to_csv('atac_age_pos_kidney_proximal_convoluted_tubule_epithelial_cell.bed', sep='\t', index=None, header=None)

df_neg = pd.DataFrame({'idx':peak_sorted[:100]})
df_neg_idx = pd.DataFrame({'chr': df_neg['idx'].map(lambda x: x.split(':')[0]).values,
                           'start': df_neg['idx'].map(lambda x: x.split(':')[1].split('-')[0]).values,
                           'end': df_neg['idx'].map(lambda x: x.split(':')[1].split('-')[1]).values})
df_neg_idx.to_csv('atac_age_neg_kidney_proximal_convoluted_tubule_epithelial_cell.bed', sep='\t', index=None, header=None)






# import statsmodels.api as sm
# from scipy.stats import norm
# from statsmodels.stats.multitest import multipletests

# # np.argwhere((m_all[1, :]>m_all[0, :]) & (m_all[2, :]>m_all[1, :]) & (m_all[3, :]>m_all[2, :])).flatten()[:20]
# # # array([  7,  36,  65,  75,  94, 234, 238, 241, 304, 324, 356, 364, 400, 424, 428, 494, 517, 549, 623, 630])

# # np.argwhere((m_all[1, :]<m_all[0, :]) & (m_all[2, :]<m_all[1, :]) & (m_all[3, :]<m_all[2, :])).flatten()[:20]
# # # array([ 132,  142,  461,  548,  660,  695,  773,  817,  846,  911,  914, 915,  917,  958, 1059, 1144, 1807, 1841, 1857, 1917])


# coef_lst = []
# pval_lst = []
# X = [1, 3, 18, 21, 30]
# X = sm.add_constant(X)
# for i in tqdm(range(m_all.shape[1]), ncols=80):
#     Y = m_all[:, i]
#     model = sm.OLS(Y, X)
#     results = model.fit()
#     beta_hat = results.params[1]
#     se_beta_hat = results.bse[1]
#     wald_statistic = beta_hat / se_beta_hat
#     p_value = 2 * (1 - norm.cdf(abs(wald_statistic)))
#     coef_lst.append(beta_hat)
#     pval_lst.append(p_value)

# df = pd.DataFrame({'idx':atac.var.index.values, 'coef':coef_lst, 'pval':pval_lst})
# df.dropna(inplace=True)
# _, fdr, _, _ = multipletests(df['pval'].values, alpha=0.05, method='fdr_bh')
# df['fdr'] = fdr


# df_pos = df[(df['fdr']<0.0001) & (df['coef']>0.003)] # df_pos = df[(df['fdr']<0.0001) & (df['coef']>0.002)]
# df_pos.to_csv('atac_age_pos_kidney_proximal_convoluted_tubule_epithelial_cell.txt', sep='\t', index=None)   # 2184
# df_pos_idx = pd.DataFrame({'chr': df_pos['idx'].map(lambda x: x.split(':')[0]).values,
#                            'start': df_pos['idx'].map(lambda x: x.split(':')[1].split('-')[0]).values,
#                            'end': df_pos['idx'].map(lambda x: x.split(':')[1].split('-')[1]).values})
# df_pos_idx.to_csv('atac_age_pos_kidney_proximal_convoluted_tubule_epithelial_cell.bed', sep='\t', index=None, header=None)

# df_neg = df[(df['fdr']<0.05) & (df['coef']<0)] # df_neg = df[(df['fdr']<0.0001) & (df['coef']<-0.002)]
# df_neg.to_csv('atac_age_neg_kidney_proximal_convoluted_tubule_epithelial_cell.txt', sep='\t', index=None)   # 1854
# df_neg_idx = pd.DataFrame({'chr': df_neg['idx'].map(lambda x: x.split(':')[0]).values,
#                            'start': df_neg['idx'].map(lambda x: x.split(':')[1].split('-')[0]).values,
#                            'end': df_neg['idx'].map(lambda x: x.split(':')[1].split('-')[1]).values})
# df_neg_idx.to_csv('atac_age_neg_kidney_proximal_convoluted_tubule_epithelial_cell.bed', sep='\t', index=None, header=None)

# df_sig = df[df['fdr']<0.01]
# out_1 = df_sig.sort_values('coef', ascending=False)[:20]
# out_2 = df_sig.sort_values('coef', ascending=True)[:20]
# pd.concat([out_2, out_1]).to_csv('kidney_proximal_convoluted_tubule_epithelial_cell_atac_test.txt', sep='\t', index=None)


### comparison coef btw pos & neg
library(TxDb.Mmusculus.UCSC.mm10.knownGene)
library(ChIPseeker)

peaks_pos <- readPeakFile('atac_age_pos_kidney_proximal_convoluted_tubule_epithelial_cell.bed')
peaks_pos_anno <- annotatePeak(peak=peaks_pos, tssRegion=c(-3000, 3000), TxDb=TxDb.Mmusculus.UCSC.mm10.knownGene, annoDb='org.Mm.eg.db')
peaks_pos_gene <- as.data.frame(peaks_pos_anno)[c('seqnames', 'start', 'end', 'SYMBOL')]
pos_gene <- unique(peaks_pos_gene['SYMBOL'])
pos_gene['pos'] <- 'pos'
write.table(pos_gene['SYMBOL'], 'peaks_pos_gene.txt', quote=FALSE, row.names=FALSE, sep='\t')

peaks_neg <- readPeakFile('atac_age_neg_kidney_proximal_convoluted_tubule_epithelial_cell.bed')
peaks_neg_anno <- annotatePeak(peak=peaks_neg, tssRegion=c(-3000, 3000), TxDb=TxDb.Mmusculus.UCSC.mm10.knownGene, annoDb='org.Mm.eg.db')
peaks_neg_gene <- as.data.frame(peaks_neg_anno)[c('seqnames', 'start', 'end', 'SYMBOL')]
neg_gene <- unique(peaks_neg_gene['SYMBOL'])
neg_gene['neg'] <- 'neg'
write.table(neg_gene['SYMBOL'], 'peaks_neg_gene.txt', quote=FALSE, row.names=FALSE, sep='\t')

# coef <- read.table('atac_age_coef_kidney_proximal_convoluted_tubule_epithelial_cell_all.txt', col.names=c('gene_name', 'coef'))  # with 0
coef <- read.table('atac_age_coef_kidney_proximal_convoluted_tubule_epithelial_cell.txt', col.names=c('gene_name', 'coef'))    # without 0
rownames(coef) <- coef$gene_name

df <- merge(coef, pos_gene, by.x='gene_name', by.y='SYMBOL', all.x=TRUE)
df <- merge(df, neg_gene, by.x='gene_name', by.y='SYMBOL', all.x=TRUE)
df[is.na(df)] <- 'unknown'
df['idx'] = 'unknown'
df[(df['pos']=='pos')&(df['neg']=='unknown'), ]['idx'] <- 'pos'
df[(df['pos']=='unknown')&(df['neg']=='neg'), ]['idx'] <- 'neg'

median(df[df['idx']=='pos', 'coef'])       # -7.98e-05
median(df[df['idx']=='unknown', 'coef'])   # -4.4e-05
median(df[df['idx']=='neg', 'coef'])       # -0.000163779

mean(df[df['idx']=='pos', 'coef'])       # -7.98e-05
mean(df[df['idx']=='unknown', 'coef'])   # -4.4e-05
mean(df[df['idx']=='neg', 'coef'])       # -0.000163779

pdf('boxplot.pdf')
boxplot(coef~idx, df, outline=FALSE)
dev.off()

wilcox.test(df[df$idx=='pos', 'coef'], df[df$idx=='neg', 'coef'])



import scanpy as sc
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from plotnine import *

rna = sc.read_h5ad('../rna_unpaired.h5ad')
rna.layers['counts'] = rna.X.copy()
sc.pp.normalize_total(rna)
sc.pp.log1p(rna)

dat = rna[rna.obs['cell_type']=='kidney proximal convoluted tubule epithelial cell']
df = pd.DataFrame(dat.X.toarray())
df['age'] = dat.obs['age'].values
df['age'].replace({'1m':1, '3m':3, '18m':18, '21m':21, '30m':30}, inplace=True)

model = LinearRegression()
model.fit(df.iloc[:, :-1], df.iloc[:, -1])

(pd.DataFrame(dat.var.index[np.argsort(model.coef_)][-100:])).to_csv('Top100_coef_genes.txt', index=False, header=False)    # top 100 genes
(pd.DataFrame(dat.var.index[np.argsort(model.coef_)][:100])).to_csv('Bottom100_coef_genes.txt', index=False, header=False)  # bottom 100 genes

pos_gene = pd.read_table('../peaks_pos_gene.txt')
neg_gene = pd.read_table('../peaks_neg_gene.txt')
pos_coef = model.coef_[(dat.var.index).isin(pos_gene['SYMBOL']) & ~(dat.var.index).isin(neg_gene['SYMBOL'])]
neg_coef = model.coef_[(dat.var.index).isin(neg_gene['SYMBOL']) & ~(dat.var.index).isin(pos_gene['SYMBOL'])]
mid_coef = model.coef_[~(dat.var.index).isin(pos_gene['SYMBOL']) & ~(dat.var.index).isin(neg_gene['SYMBOL'])]
[pos_coef.mean(), mid_coef.mean(), neg_coef.mean()]  # [0.00832399, -0.0018646211, -0.01999282]

pos_df = pd.DataFrame({'idx':['pos'], 'avg':[np.mean(pos_coef)], 'std':[np.std(pos_coef)]})
mid_df = pd.DataFrame({'idx':['mid'], 'avg':[np.mean(mid_coef)], 'std':[np.std(mid_coef)]})
neg_df = pd.DataFrame({'idx':['neg'], 'avg':[np.mean(neg_coef)], 'std':[np.std(neg_coef)]})
pos_mid_neg = pd.concat([pos_df, mid_df, neg_df])
pos_mid_neg['idx'] = pd.Categorical(pos_mid_neg['idx'], categories=['pos', 'mid', 'neg'])

p = ggplot(pos_mid_neg, aes(x='idx', y='avg', fill='idx')) + geom_errorbar(aes(ymin='avg-std', ymax='avg+std'), width=0.1) + geom_point(aes(y='avg', color='idx'), size=4, show_legend=False) +\
                                                             xlab('') + ylab('Average coef') + scale_y_continuous(limits=[-0.3, 0.3], breaks=np.arange(-0.3, 0.3+0.01, 0.1)) + theme_bw()
p.save(filename='pos_mid_neg_coef_kidney_proximal_convoluted_tubule_epithelial_cell.pdf', dpi=300, height=4, width=4)




# from sklearn.linear_model import LinearRegression
# X = np.array([1, 2, 3]).reshape(-1, 1)
# y = [pos_coef.mean(), mid_coef.mean(), neg_coef.mean()]
# model = LinearRegression().fit(X, y)
# slope = model.coef_[0]
# intercept = model.intercept_
# r_value = model.score(X, y)
# print(f"Slope: {slope}, Intercept: {intercept}, R-squared: {r_value}")
# # 计算 p 值
# t_stat = slope / np.sqrt(np.var(y) / len(y))
# p_value = stats.t.sf(np.abs(t_stat), df=len(y)-2) * 2  # 双尾检验
# print(f"t-statistic: {t_stat}, p-value: {p_value}")


from scipy import stats
stats.ttest_ind(pos_coef, neg_coef) 


pos_gene = pd.read_table('../peaks_pos_gene.txt')
m_01 = pd.DataFrame({'age': '1m', 'exp': dat[dat.obs['age']=='1m', (dat.var.index).isin(pos_gene['SYMBOL'])].X.toarray().mean(axis=0)})
m_03 = pd.DataFrame({'age': '3m', 'exp': dat[dat.obs['age']=='3m', (dat.var.index).isin(pos_gene['SYMBOL'])].X.toarray().mean(axis=0)})
m_18 = pd.DataFrame({'age': '18m', 'exp': dat[dat.obs['age']=='18m', (dat.var.index).isin(pos_gene['SYMBOL'])].X.toarray().mean(axis=0)})
m_21 = pd.DataFrame({'age': '21m', 'exp': dat[dat.obs['age']=='21m', (dat.var.index).isin(pos_gene['SYMBOL'])].X.toarray().mean(axis=0)})
m_30 = pd.DataFrame({'age': '30m', 'exp': dat[dat.obs['age']=='30m', (dat.var.index).isin(pos_gene['SYMBOL'])].X.toarray().mean(axis=0)})
[m_01['exp'].mean(), m_03['exp'].mean(), m_18['exp'].mean(), m_21['exp'].mean(), m_30['exp'].mean()]  # [0.17875327, 0.18963073, 0.19132525, 0.19010606, 0.16647679]  
[m_01['exp'].median(), m_03['exp'].median(), m_18['exp'].median(), m_21['exp'].median(), m_30['exp'].median()]   # [0.031883623, 0.03255263, 0.033601325, 0.03184045, 0.033066757]


neg_gene = pd.read_table('../peaks_neg_gene.txt')
m_01 = pd.DataFrame({'age': '1m', 'exp': dat[dat.obs['age']=='1m', (dat.var.index).isin(neg_gene['SYMBOL'])].X.toarray().mean(axis=0)})
m_03 = pd.DataFrame({'age': '3m', 'exp': dat[dat.obs['age']=='3m', (dat.var.index).isin(neg_gene['SYMBOL'])].X.toarray().mean(axis=0)})
m_18 = pd.DataFrame({'age': '18m', 'exp': dat[dat.obs['age']=='18m', (dat.var.index).isin(neg_gene['SYMBOL'])].X.toarray().mean(axis=0)})
m_21 = pd.DataFrame({'age': '21m', 'exp': dat[dat.obs['age']=='21m', (dat.var.index).isin(neg_gene['SYMBOL'])].X.toarray().mean(axis=0)})
m_30 = pd.DataFrame({'age': '30m', 'exp': dat[dat.obs['age']=='30m', (dat.var.index).isin(neg_gene['SYMBOL'])].X.toarray().mean(axis=0)})
[m_01['exp'].mean(), m_03['exp'].mean(), m_18['exp'].mean(), m_21['exp'].mean(), m_30['exp'].mean()]  # [0.20810956, 0.2113044, 0.21379451, 0.20848018, 0.18522385]
[m_01['exp'].median(), m_03['exp'].median(), m_18['exp'].median(), m_21['exp'].median(), m_30['exp'].median()]   # [0.031883623, 0.03255263, 0.033601325, 0.03184045, 0.033066757]

gene_lst = random.sample(list(dat.var.index.values), 100)
gene_lst = ['Wdr59']
m_01 = pd.DataFrame({'age': '1m', 'exp': dat[dat.obs['age']=='1m', (dat.var.index).isin(gene_lst)].X.toarray().mean(axis=0)})
m_03 = pd.DataFrame({'age': '3m', 'exp': dat[dat.obs['age']=='3m', (dat.var.index).isin(gene_lst)].X.toarray().mean(axis=0)})
m_18 = pd.DataFrame({'age': '18m', 'exp': dat[dat.obs['age']=='18m', (dat.var.index).isin(gene_lst)].X.toarray().mean(axis=0)})
m_21 = pd.DataFrame({'age': '21m', 'exp': dat[dat.obs['age']=='21m', (dat.var.index).isin(gene_lst)].X.toarray().mean(axis=0)})
m_30 = pd.DataFrame({'age': '30m', 'exp': dat[dat.obs['age']=='30m', (dat.var.index).isin(gene_lst)].X.toarray().mean(axis=0)})
[m_01['exp'].mean(), m_03['exp'].mean(), m_18['exp'].mean(), m_21['exp'].mean(), m_30['exp'].mean()]  # [0.061513495, 0.05995603, 0.0603721, 0.059424985, 0.053931557]


## plot bar
import pandas as pd
from plotnine import *
import numpy as np

df = pd.read_table('kidney_proximal_convoluted_tubule_epithelial_cell_atac_test.txt')
df = df.sort_values('coef', ascending=True)
df['idx'] = pd.Categorical(df['idx'], categories=df['idx'])

p = ggplot(df, aes(x='idx', y='coef')) + geom_bar(stat='identity', fill='darkred', position=position_dodge()) + coord_flip() + \
                                                        xlab('Peaks') + ylab('Coef') + labs(title='Kidney proximal convoluted tubule epithelial cells')  + \
                                                        scale_y_continuous(limits=[-0.015, 0.015], breaks=np.arange(-0.015, 0.015+0.0001, 0.003)) + theme_bw() + theme(plot_title=element_text(hjust=0.5))
p.save(filename='kidney_proximal_convoluted_tubule_epithelial_cell_atac_test.pdf', dpi=600, height=8, width=15)




# kidney proximal convoluted tubule epithelial cell            4460
# B cell                                                       3124
# epithelial cell of proximal tubule                           3053
# kidney loop of Henle thick ascending limb epithelial cell    1554
# lymphocyte                                                   1536
# macrophage                                                   1407
# T cell                                                       1359
# fenestrated cell                                             1027
# kidney collecting duct principal cell                         789
# kidney distal convoluted tubule epithelial cell               744
# plasmatocyte                                                  496
# brush cell                                                    414
# kidney cortex artery cell                                     381
# plasma cell                                                   325
# mesangial cell                                                261
# kidney loop of Henle ascending limb epithelial cell           201
# kidney capillary endothelial cell                             161
# fibroblast                                                    161
# kidney proximal straight tubule epithelial cell                95
# natural killer cell                                            66
# kidney collecting duct epithelial cell                         25
# leukocyte                                                       5
# kidney cell                                                     3


# marker genes: https://drive.google.com/drive/u/0/folders/1JZgFDmdVlw-UN8Gb_lRxSAdCZfhlzdrH
# Single cell regulatory landscape of the mouse kidney highlights cellular differentiation programs and disease targets
# Differentail Expression: https://cellxgene.cziscience.com/differential-expression

## prob average seems almost same!!!
# dat = np.load('predict_unpaired.npy')
# atac = sc.read_h5ad('atac_unpaired.h5ad')
# atac.X = dat

# def out_pred_prob_bedgraph(cell_type='T cell'):
#     atac_X = atac[atac.obs['cell_type']==cell_type, :].X.mean(axis=0)
#     df = pd.DataFrame({'chr': atac.var['gene_ids'].map(lambda x: x.split(':')[0]).values,
#                        'start': atac.var['gene_ids'].map(lambda x: x.split(':')[1].split('-')[0]).values,
#                        'end': atac.var['gene_ids'].map(lambda x: x.split(':')[1].split('-')[1]).values,
#                        'val': atac_X})
#     df.to_csv(cell_type.replace(' ', '_')+'_atac_pred_prob.bedgraph', index=False, header=False, sep='\t')

# for cell_type in set(atac.obs.cell_type.values):
#     out_pred_prob_bedgraph(cell_type=cell_type)




######################################################################################################################################################
## RNA -> ATAC
# Top100_coef_genes.bed
# Bottom100_coef_genes.bed

grep -v GL scATAC_peaks.bed | grep -v JH > scATAC_peaks_filtered.bed

sort -k1,1 -k2,2n Top100_coef_genes.bed | bedtools closest -a stdin -b scATAC_peaks_filtered.bed | cut -f 5-7 | uniq > Top100_coef_genes_peaks.bed         # 99
sort -k1,1 -k2,2n Bottom100_coef_genes.bed | bedtools closest -a stdin -b scATAC_peaks_filtered.bed | cut -f 5-7 | uniq > Bottom100_coef_genes_peaks.bed   # 100

# sort -k1,1 -k2,2n Top100_coef_genes.bed | bedtools intersect -a stdin -b scATAC_peaks_filtered.bed -wa -wb | cut -f 5-7 | uniq > Top100_coef_genes_peaks.bed          # 79
# sort -k1,1 -k2,2n Bottom100_coef_genes.bed | bedtools intersect -a stdin -b scATAC_peaks_filtered.bed -wa -wb | cut -f 5-7 | uniq > Bottom100_coef_genes_peaks.bed    # 82

# sort -k1,1 -k2,2n Top100_coef_genes.bed | awk '{print($1 "\t" $2-100000 "\t" $3+100000 "\t" $4)}' | bedtools intersect -a stdin -b scATAC_peaks_filtered.bed -wa -wb |\
#                                           cut -f 5-7 | uniq > Top100_coef_genes_peaks.bed  # 1333
# sort -k1,1 -k2,2n Bottom100_coef_genes.bed | awk '{print($1 "\t" $2-100000 "\t" $3+100000 "\t" $4)}' | bedtools intersect -a stdin -b scATAC_peaks_filtered.bed -wa -wb |\
#                                              cut -f 5-7 | uniq  > Bottom100_coef_genes_peaks.bed  # 1419

import scanpy as sc
import numpy as np
from scipy.sparse import csr_matrix
import pandas as pd
from tqdm import tqdm
from plotnine import *

dat = np.load('predict_unpaired.npy')
dat[dat>0.5] = 1
dat[dat<=0.5] = 0
atac = sc.read_h5ad('atac_unpaired.h5ad')
atac.X = csr_matrix(dat)

# from sklearn.feature_extraction.text import TfidfTransformer
# tfidf_transformer = TfidfTransformer(norm='l2', use_idf=True)
# X_tfidf = tfidf_transformer.fit_transform(atac[atac.obs['cell_type']==cell_type].X.toarray())

cell_type = 'kidney proximal convoluted tubule epithelial cell'
atac_m_01 = atac[(atac.obs['cell_type']==cell_type) & (atac.obs['age']=='1m')].X.toarray()
atac_m_01 = atac_m_01.sum(axis=0) / atac_m_01.sum() * 10000
atac_m_03 = atac[(atac.obs['cell_type']==cell_type) & (atac.obs['age']=='3m')].X.toarray()
atac_m_03 = atac_m_03.sum(axis=0) / atac_m_03.sum() * 10000
atac_m_18 = atac[(atac.obs['cell_type']==cell_type) & (atac.obs['age']=='18m')].X.toarray()
atac_m_18 = atac_m_18.sum(axis=0) / atac_m_18.sum() * 10000
atac_m_21 = atac[(atac.obs['cell_type']==cell_type) & (atac.obs['age']=='21m')].X.toarray()
atac_m_21 = atac_m_21.sum(axis=0) / atac_m_21.sum() * 10000
atac_m_30 = atac[(atac.obs['cell_type']==cell_type) & (atac.obs['age']=='30m')].X.toarray()
atac_m_30 = atac_m_30.sum(axis=0) / atac_m_30.sum() * 10000
[atac_m_01.mean(), atac_m_03.mean(), atac_m_18.mean(), atac_m_21.mean(), atac_m_30.mean()]
# [0.15301749, 0.1530175, 0.1530175, 0.1530175, 0.1530175]
[np.median(atac_m_01), np.median(atac_m_03), np.median(atac_m_18), np.median(atac_m_21), np.median(atac_m_30)]
# [0.06276866, 0.052454103, 0.0598559, 0.06315869, 0.07356452]

# import random
# gene_lst = random.sample(list(atac.var.index.values), 100)
# atac_random_m_01 = atac_m_01[(atac.var.index).isin(gene_lst)]
# atac_random_m_03 = atac_m_03[(atac.var.index).isin(gene_lst)]
# atac_random_m_18 = atac_m_18[(atac.var.index).isin(gene_lst)]
# atac_random_m_21 = atac_m_21[(atac.var.index).isin(gene_lst)]
# atac_random_m_30 = atac_m_30[(atac.var.index).isin(gene_lst)]
# [atac_random_m_01.mean(), atac_random_m_03.mean(), atac_random_m_18.mean(), atac_random_m_21.mean(), atac_random_m_30.mean()]
# # [0.14785847, 0.14840893, 0.14927007, 0.14665449, 0.14645793]

up_peaks = pd.read_table('epitrace/Top100_coef_genes_peaks.bed', header=None, names=['chrom', 'start', 'end'])
up_peaks['idx'] = up_peaks['chrom']+':'+up_peaks['start'].astype(str)+'-'+up_peaks['end'].astype(str)
atac_up_m_01 = atac_m_01[(atac.var.index).isin(up_peaks['idx'])]
atac_up_m_03 = atac_m_03[(atac.var.index).isin(up_peaks['idx'])]
atac_up_m_18 = atac_m_18[(atac.var.index).isin(up_peaks['idx'])]
atac_up_m_21 = atac_m_21[(atac.var.index).isin(up_peaks['idx'])]
atac_up_m_30 = atac_m_30[(atac.var.index).isin(up_peaks['idx'])]
[atac_up_m_01.mean(), atac_up_m_03.mean(), atac_up_m_18.mean(), atac_up_m_21.mean(), atac_up_m_30.mean()]
# [0.34636426, 0.35574147, 0.33578697, 0.32938212, 0.32193443]
[np.median(atac_up_m_01), np.median(atac_up_m_03), np.median(atac_up_m_18), np.median(atac_up_m_21), np.median(atac_up_m_30)]
# [0.33527648, 0.31110707, 0.32920748, 0.323162, 0.30200174]

down_peaks = pd.read_table('epitrace/Bottom100_coef_genes_peaks.bed', header=None, names=['chrom', 'start', 'end'])
down_peaks['idx'] = down_peaks['chrom']+':'+down_peaks['start'].astype(str)+'-'+down_peaks['end'].astype(str)
atac_down_m_01 = atac_m_01[(atac.var.index).isin(down_peaks['idx'])]
atac_down_m_03 = atac_m_03[(atac.var.index).isin(down_peaks['idx'])]
atac_down_m_18 = atac_m_18[(atac.var.index).isin(down_peaks['idx'])]
atac_down_m_21 = atac_m_21[(atac.var.index).isin(down_peaks['idx'])]
atac_down_m_30 = atac_m_30[(atac.var.index).isin(down_peaks['idx'])]
[atac_down_m_01.mean(), atac_down_m_03.mean(), atac_down_m_18.mean(), atac_down_m_21.mean(), atac_down_m_30.mean()]
# [0.43388453, 0.45624214, 0.42539063, 0.41345787, 0.40415314]
[np.median(atac_down_m_01), np.median(atac_down_m_03), np.median(atac_down_m_18), np.median(atac_down_m_21), np.median(atac_down_m_30)]
# [0.42560214, 0.4729913, 0.44539833, 0.44158453, 0.4142844]

atac_other_m_01 = atac_m_01[~(atac.var.index).isin(up_peaks['idx'].tolist()+down_peaks['idx'].tolist())]
atac_other_m_03 = atac_m_03[~(atac.var.index).isin(up_peaks['idx'].tolist()+down_peaks['idx'].tolist())]
atac_other_m_18 = atac_m_18[~(atac.var.index).isin(up_peaks['idx'].tolist()+down_peaks['idx'].tolist())]
atac_other_m_21 = atac_m_21[~(atac.var.index).isin(up_peaks['idx'].tolist()+down_peaks['idx'].tolist())]
atac_other_m_30 = atac_m_30[~(atac.var.index).isin(up_peaks['idx'].tolist()+down_peaks['idx'].tolist())]
[atac_other_m_01.mean(), atac_other_m_03.mean(), atac_other_m_18.mean(), atac_other_m_21.mean(), atac_other_m_30.mean()]
# [0.15229264, 0.15224406, 0.15232174, 0.15234977, 0.15237537]
[np.median(atac_other_m_01), np.median(atac_other_m_03), np.median(atac_other_m_18), np.median(atac_other_m_21), np.median(atac_other_m_30)]
# [0.06123772, 0.052454103, 0.0598559, 0.062106047, 0.07356452]

## plot up peaks
df_up = pd.concat([pd.DataFrame({'idx':'1m', 'val':atac_up_m_01}),
                   pd.DataFrame({'idx':'3m', 'val':atac_up_m_03}),
                   pd.DataFrame({'idx':'18m', 'val':atac_up_m_18}),
                   pd.DataFrame({'idx':'21m', 'val':atac_up_m_21}),
                   pd.DataFrame({'idx':'30m', 'val':atac_up_m_30})])
df_up['idx'] = pd.Categorical(df_up['idx'], categories=['1m', '3m', '18m', '21m', '30m'])
p = ggplot(df_up, aes(x='idx', y='val', fill='idx')) + geom_boxplot(width=0.5, show_legend=False, outlier_shape='') +\
                                                       xlab('Age') + ylab('Normalized ATAC signal') +\
                                                       scale_y_continuous(limits=[0, 1.5], breaks=np.arange(0, 1.5+0.1, 0.3)) + theme_bw()
p.save(filename='Top100_up_genes_peaks_age_boxplot.pdf', dpi=100, height=4, width=4)

## plot down peaks
df_down = pd.concat([pd.DataFrame({'idx':'1m', 'val':atac_down_m_01}),
                     pd.DataFrame({'idx':'3m', 'val':atac_down_m_03}),
                     pd.DataFrame({'idx':'18m', 'val':atac_down_m_18}),
                     pd.DataFrame({'idx':'21m', 'val':atac_down_m_21}),
                     pd.DataFrame({'idx':'30m', 'val':atac_down_m_30})])
df_down['idx'] = pd.Categorical(df_down['idx'], categories=['1m', '3m', '18m', '21m', '30m'])
p = ggplot(df_down, aes(x='idx', y='val', fill='idx')) + geom_boxplot(width=0.5, show_legend=False, outlier_shape='') +\
                                                         xlab('Age') + ylab('Normalized ATAC signal') +\
                                                         scale_y_continuous(limits=[0, 1.5], breaks=np.arange(0, 1.5+0.1, 0.3)) + theme_bw()
p.save(filename='Top100_down_genes_peaks_age_boxplot.pdf', dpi=100, height=4, width=4)

## plot other peaks
df_other = pd.concat([pd.DataFrame({'idx':'1m', 'val':atac_other_m_01}),
                      pd.DataFrame({'idx':'3m', 'val':atac_other_m_03}),
                      pd.DataFrame({'idx':'18m', 'val':atac_other_m_18}),
                      pd.DataFrame({'idx':'21m', 'val':atac_other_m_21}),
                      pd.DataFrame({'idx':'30m', 'val':atac_other_m_30})])
df_other['idx'] = pd.Categorical(df_other['idx'], categories=['1m', '3m', '18m', '21m', '30m'])
p = ggplot(df_other, aes(x='idx', y='val', fill='idx')) + geom_boxplot(width=0.5, show_legend=False, outlier_shape='') +\
                                                          xlab('Age') + ylab('Normalized ATAC signal') +\
                                                          scale_y_continuous(limits=[0, 1.5], breaks=np.arange(0, 1.5+0.1, 0.3)) + theme_bw()
p.save(filename='Top100_other_genes_peaks_age_boxplot.pdf', dpi=100, height=4, width=4)

# from scipy import stats
# stats.ttest_ind(atac_down_m_03, atac_down_m_30) 

## LR model
cell_type = 'kidney proximal convoluted tubule epithelial cell'
m_01 = pd.DataFrame(atac[(atac.obs['cell_type']==cell_type) & (atac.obs['age']=='1m')].X.toarray())  # 938
m_01.columns = atac.var.index.values
m_01['age'] = 1
m_03 = pd.DataFrame(atac[(atac.obs['cell_type']==cell_type) & (atac.obs['age']=='3m')].X.toarray())  # 681
m_03.columns = atac.var.index.values
m_03['age'] = 3
m_18 = pd.DataFrame(atac[(atac.obs['cell_type']==cell_type) & (atac.obs['age']=='18m')].X.toarray()) # 1117
m_18.columns = atac.var.index.values
m_18['age'] = 18
m_21 = pd.DataFrame(atac[(atac.obs['cell_type']==cell_type) & (atac.obs['age']=='21m')].X.toarray())  # 860
m_21.columns = atac.var.index.values
m_21['age'] = 21
m_30 = pd.DataFrame(atac[(atac.obs['cell_type']==cell_type) & (atac.obs['age']=='30m')].X.toarray())  # 864
m_30.columns = atac.var.index.values
m_30['age'] = 30
m_all = pd.concat((m_01, m_03, m_18, m_21, m_30))

from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(m_all.iloc[:, :-1], m_all.iloc[:, -1])

model.coef_[np.argwhere(atac.var.index.isin(up_peaks['idx'])).flatten()].mean()                                           # -1.7344282
model.coef_[np.argwhere(atac.var.index.isin(down_peaks['idx'])).flatten()].mean()                                         # -4.6969366
model.coef_[np.argwhere(~atac.var.index.isin(up_peaks['idx'].tolist()+down_peaks['idx'].tolist())).flatten()].mean()      # -0.025189517

np.median(model.coef_[np.argwhere(atac.var.index.isin(up_peaks['idx'])).flatten()])                                       # -0.14838511
np.median(model.coef_[np.argwhere(atac.var.index.isin(down_peaks['idx'])).flatten()])                                     # -0.3813411
np.median(model.coef_[np.argwhere(~atac.var.index.isin(up_peaks['idx'].tolist()+down_peaks['idx'].tolist())).flatten()])  # 0.0

pos_coef = model.coef_[np.argwhere(atac.var.index.isin(up_peaks['idx'])).flatten()]
neg_coef = model.coef_[np.argwhere(atac.var.index.isin(down_peaks['idx'])).flatten()]
mid_coef = model.coef_[np.argwhere(~atac.var.index.isin(up_peaks['idx'].tolist()+down_peaks['idx'].tolist())).flatten()]
pos_df = pd.DataFrame({'idx':['pos'], 'avg':[np.mean(pos_coef)], 'std':[np.std(pos_coef)]})
mid_df = pd.DataFrame({'idx':['mid'], 'avg':[np.mean(mid_coef)], 'std':[np.std(mid_coef)]})
neg_df = pd.DataFrame({'idx':['neg'], 'avg':[np.mean(neg_coef)], 'std':[np.std(neg_coef)]})
pos_mid_neg = pd.concat([pos_df, mid_df, neg_df])
pos_mid_neg['idx'] = pd.Categorical(pos_mid_neg['idx'], categories=['pos', 'mid', 'neg'])

p = ggplot(pos_mid_neg, aes(x='idx', y='avg', fill='idx')) + geom_errorbar(aes(ymin='avg-std', ymax='avg+std'), width=0.1) + geom_point(aes(y='avg', color='idx'), size=4, show_legend=False) +\
                                                             xlab('') + ylab('Coef') + scale_y_continuous(limits=[-30, 30], breaks=np.arange(-30, 30+1, 10)) + theme_bw()
p.save(filename='pos_mid_neg_coef_gene_to_peak_kidney_proximal_convoluted_tubule_epithelial_cell.pdf', dpi=300, height=4, width=4)


## output bedgraph
pd.DataFrame({'chr': atac.var.index.map(lambda x: x.split(':')[0]).values,
              'start': atac.var.index.map(lambda x: x.split(':')[1].split('-')[0]).values,
              'end': atac.var.index.map(lambda x: x.split(':')[1].split('-')[1]).values,
              'val': atac_m_01}).to_csv('age_1m_atac_norm.bedgraph', index=False, header=False, sep='\t')
pd.DataFrame({'chr': atac.var.index.map(lambda x: x.split(':')[0]).values,
              'start': atac.var.index.map(lambda x: x.split(':')[1].split('-')[0]).values,
              'end': atac.var.index.map(lambda x: x.split(':')[1].split('-')[1]).values,
              'val': atac_m_03}).to_csv('age_3m_atac_norm.bedgraph', index=False, header=False, sep='\t')
pd.DataFrame({'chr': atac.var.index.map(lambda x: x.split(':')[0]).values,
              'start': atac.var.index.map(lambda x: x.split(':')[1].split('-')[0]).values,
              'end': atac.var.index.map(lambda x: x.split(':')[1].split('-')[1]).values,
              'val': atac_m_18}).to_csv('age_18m_atac_norm.bedgraph', index=False, header=False, sep='\t')
pd.DataFrame({'chr': atac.var.index.map(lambda x: x.split(':')[0]).values,
              'start': atac.var.index.map(lambda x: x.split(':')[1].split('-')[0]).values,
              'end': atac.var.index.map(lambda x: x.split(':')[1].split('-')[1]).values,
              'val': atac_m_21}).to_csv('age_21m_atac_norm.bedgraph', index=False, header=False, sep='\t')
pd.DataFrame({'chr': atac.var.index.map(lambda x: x.split(':')[0]).values,
              'start': atac.var.index.map(lambda x: x.split(':')[1].split('-')[0]).values,
              'end': atac.var.index.map(lambda x: x.split(':')[1].split('-')[1]).values,
              'val': atac_m_30}).to_csv('age_30m_atac_norm.bedgraph', index=False, header=False, sep='\t')
######################################################################################################################################################





###### EpiTrace - Estimating cell age from single-cell ATAC data
# https://epitrace.readthedocs.io/en/latest/
# https://epitrace.readthedocs.io/en/latest/MSscATAC.html
# EpiTrace application (Genome Biology): https://link.springer.com/article/10.1186/s13059-024-03265-z#article-info

################################ INSTALLATION ########################################
conda create -n epitrace conda-forge::r-base=4.1.3

install.packages('pak')
library(pak)
install.packages('https://cran.r-project.org/src/contrib/Archive/rjson/rjson_0.2.20.tar.gz')
install.packages('lattice')
install.packages('https://cran.r-project.org/src/contrib/Archive/Matrix/Matrix_1.5-0.tar.gz')
pak::pkg_install('caleblareau/easyLift')

conda install r-xml
Sys.setenv(XML_CONFIG="/data/home/jiluzhang/miniconda3/bin/xml2-config")
install.packages("XML")

conda create -n epitrace conda-forge::r-base=4.2.0
install.packages('pak')
library(pak)
install.packages('https://cran.r-project.org/src/contrib/Archive/rjson/rjson_0.2.20.tar.gz')
install.packages('lattice')
install.packages('https://cran.r-project.org/src/contrib/Archive/Matrix/Matrix_1.5-0.tar.gz')

pak::pkg_install('caleblareau/easyLift')
install.packages('https://cran.r-project.org/src/contrib/Archive/Matrix/Matrix_1.6-4.tar.gz')
install.packages('https://cran.r-project.org/src/contrib/Archive/MASS/MASS_7.3-55.tar.gz')
install.packages('testthat')

wget -c https://github.com/Kitware/CMake/releases/download/v3.30.3/cmake-3.30.3.tar.gz
tar -zvxf cmake-3.30.3.tar.gz
cd cmake-3.30.3
./bootstrap
./configure --prefix=/data/home/jiluzhang/biosoftware/cmake
make
make install
add env

install.packages('https://cran.r-project.org/src/contrib/Archive/nloptr/nloptr_2.0.0.tar.gz')
install.packages('https://www.bioconductor.org/packages/release/bioc/src/contrib/ggtree_3.12.0.tar.gz')

pak::pkg_install('MagpiePKU/EpiTrace')

install.packages('readr')
install.packages('openxlsx')

conda install conda-forge::r-devtools
ln -s libicuuc.so.70 libicuuc.so.58 (workdir:/data/home/jiluzhang/miniconda3/envs/epitrace/lib)
ln -s libicui18n.so.70 libicui18n.so.58
install.packages("https://cran.r-project.org/src/contrib/stringi_1.8.4.tar.gz")

install.packages("BiocManager")
devtools::install_github("GreenleafLab/ArchR", ref="master", repos = BiocManager::repositories())
install.packages('https://cran.r-project.org/src/contrib/Rcpp_1.0.13.tar.gz')

BiocManager::install("ChIPseeker")
###################################################################################################

wget -c https://github.com/MagpiePKU/EpiTrace/blob/master/data/mouse_clock_mm285_design_clock347_mm10.rds

###### transform h5ad to 10x
import numpy as np
import scanpy as sc
import scipy.io as sio
import pandas as pd
from scipy.sparse import csr_matrix

dat = np.load('../predict_unpaired.npy')
dat[dat>0.5] = 1
dat[dat<=0.5] = 0
atac = sc.read_h5ad('../atac_unpaired.h5ad')
atac.X = csr_matrix(dat)

## output peaks
peaks_df = pd.DataFrame({'chr': atac.var['gene_ids'].map(lambda x: x.split(':')[0]).values,
                         'start': atac.var['gene_ids'].map(lambda x: x.split(':')[1].split('-')[0]).values,
                         'end': atac.var['gene_ids'].map(lambda x: x.split(':')[1].split('-')[1]).values})
peaks_df.to_csv('scATAC_peaks.bed', sep='\t', header=False, index=False)

## output barcodes
pd.DataFrame(atac.obs.index).to_csv('scATAC_barcodes.tsv', sep='\t', header=False, index=False)

## output matrix
sio.mmwrite("scATAC_matrix.mtx", atac.X.T)  # peak*cell (not cell*peak)

## output meta
pd.DataFrame(atac.obs[['age', 'cell_type']]).to_csv('scATAC_meta.tsv', sep='\t', header=False, index=True)








library(readr)
library(EpiTrace)
library(dplyr)
library(GenomicRanges)
library(ggplot2)
library(Seurat)


## load the  “mouse clocks”
# https://github.com/MagpiePKU/EpiTrace/blob/master/data/mouse_clock_mm285_design_clock347_mm10.rds
mouse_clock_by_MM285 <- readRDS('mouse_clock_mm285_design_clock347_mm10.rds')

## prepare the scATAC data
peaks_df <- read_tsv('scATAC_peaks.bed',col_names = c('chr','start','end'))
cells <- read_tsv('scATAC_barcodes.tsv',col_names=c('cell'))
mm <- Matrix::readMM('scATAC_matrix.mtx')
peaks_df$peaks <- paste0(peaks_df$chr, '-', peaks_df$start, '-', peaks_df$end)
init_gr <- EpiTrace::Init_Peakset(peaks_df)
init_mm <- EpiTrace::Init_Matrix(peakname=peaks_df$peaks, cellname=cells$cell, matrix=mm)

## EpiTrace analysis
# trace(EpiTraceAge_Convergence, edit=TRUE)  ref_genome='hg19'???
epitrace_obj_age_conv_estimated_by_mouse_clock <- EpiTraceAge_Convergence(peakSet=init_gr, matrix=init_mm, ref_genome='mm10',
                                                                          clock_gr=mouse_clock_by_MM285, 
                                                                          iterative_time=5, min.cutoff=0, non_standard_clock=T,
                                                                          qualnum=10, ncore_lim=48, mean_error_limit=0.1)
## add meta info of cell types
meta <- as.data.frame(epitrace_obj_age_conv_estimated_by_mouse_clock@meta.data)
write.table(meta, 'epitrace_meta.txt', sep='\t', quote=FALSE)
cell_info <- read.table('scATAC_meta.tsv', sep='\t', col.names=c('cell', 'age', 'cell_type'))
meta_cell_info <- left_join(meta, cell_info) 
epitrace_obj_age_conv_estimated_by_mouse_clock <- AddMetaData(epitrace_obj_age_conv_estimated_by_mouse_clock, metadata=meta_cell_info[['cell_type']], col.name='cell_type')

epitrace_obj_age_conv_estimated_by_mouse_clock[['all']] <- Seurat::CreateAssayObject(counts=init_mm[,c(epitrace_obj_age_conv_estimated_by_mouse_clock$cell)], min.cells=0, min.features=0, check.matrix=F)
DefaultAssay(epitrace_obj_age_conv_estimated_by_mouse_clock) <- 'all'
saveRDS(epitrace_obj_age_conv_estimated_by_mouse_clock, file='epitrace_obj_age_conv_estimated_by_mouse_clock.rds')
epitrace_obj_age_conv_estimated_by_mouse_clock <- readRDS(file='epitrace_obj_age_conv_estimated_by_mouse_clock.rds')

## peak-age association (consider all cell types for comparison)
# asso_res_mouse_clock <- AssociationOfPeaksToAge(epitrace_obj_age_conv_estimated_by_mouse_clock,
#                                                 epitrace_age_name="EpiTraceAge_iterative", parallel=TRUE, peakSetName='all')
# asso_res_mouse_clock[order(asso_res_mouse_clock$correlation_of_EpiTraceAge, decreasing=TRUE), ][1:10, 1:2]
# asso_res_mouse_clock[order(asso_res_mouse_clock$scaled_correlation_of_EpiTraceAge, decreasing=TRUE), ][1:10, c(1,3)]

## plot top 20 up & down peaks
## kidney proximal convoluted tubule epithelial cell
asso_res_mouse_clock_01 <- AssociationOfPeaksToAge(subset(epitrace_obj_age_conv_estimated_by_mouse_clock, cell_type %in% c('kidney proximal convoluted tubule epithelial cell')),
                                                   epitrace_age_name="EpiTraceAge_iterative", parallel=T, peakSetName='all')
up <- asso_res_mouse_clock_01[order(asso_res_mouse_clock_01$scaled_correlation_of_EpiTraceAge, decreasing=TRUE), ][1:20, c(1,3)]
dw <- asso_res_mouse_clock_01[order(asso_res_mouse_clock_01$scaled_correlation_of_EpiTraceAge, decreasing=FALSE), ][1:20, c(1,3)]
dat <- rbind(up, dw)
rownames(dat) <- seq(1, nrow(dat))
colnames(dat) <- c('Peaks', 'Scaled_correlation')
dat <- dat[order(dat$Scaled_correlation, decreasing=FALSE), ]
dat$Peaks <- factor(dat$Peaks, levels=dat$Peaks)
p <- ggplot(dat, aes(x=Peaks, y=Scaled_correlation)) + geom_bar(stat='identity', fill='darkred', position=position_dodge()) + coord_flip() +
                                                       scale_y_continuous(limits=c(-2.5, 2.5), breaks=seq(-2.5, 2.5, 0.5)) + theme_bw() + 
                                                       ggtitle('Kidney proximal convoluted tubule epithelial cells') + theme(plot.title=element_text(hjust=0.5))
ggsave(p, filename='peaks_epitrace_corr_kidney_proximal_convoluted_tubule_epithelial_cell.pdf', dpi=300, height=10, width=6)

## annotate peaks
peaks <- as.data.frame(do.call(rbind, strsplit(as.vector(rev(dat$Peaks)), split='-'))) # use rev() to match ranking
colnames(peaks) <- c('chrom', 'start', 'end')
write.table(peaks, 'peaks_epitrace_top20_bot20_kidney_proximal_convoluted_tubule_epithelial_cell.bed',
            sep='\t', quote=FALSE, row.names=FALSE, col.names=FALSE)

library(TxDb.Mmusculus.UCSC.mm10.knownGene)
library(ChIPseeker)
peaks <- readPeakFile('peaks_epitrace_top20_bot20_kidney_proximal_convoluted_tubule_epithelial_cell.bed')
peaks_anno <- annotatePeak(peak=peaks, tssRegion=c(-3000, 3000), TxDb=TxDb.Mmusculus.UCSC.mm10.knownGene, annoDb='org.Mm.eg.db')
as.data.frame(peaks_anno)['SYMBOL']

epitrace_obj_age_conv_estimated_by_mouse_clock <- readRDS(file='epitrace_obj_age_conv_estimated_by_mouse_clock.rds')
epitrace_obj_age_conv_estimated_by_mouse_clock@assays$all@data

enriched.motifs <- FindMotifs(object=epitrace_obj_age_conv_estimated_by_mouse_clock,
                              features=rownames(asso_res_mouse_clock_01[order(asso_res_mouse_clock_01$scaled_correlation_of_EpiTraceAge, decreasing=TRUE), ][1:20, ]['locus']))
# Selecting background regions to match input sequence characteristics
# Error in match.arg(arg = layer) : 
#   'arg' should be one of “data”, “scale.data”, “counts”



# idx <- rownames(asso_res_mouse_clock_01[order(asso_res_mouse_clock_01$scaled_correlation_of_EpiTraceAge, decreasing=TRUE), ][1:100, ])
# peaks <- as.data.frame(do.call(rbind, strsplit(idx, split='-')))
# colnames(peaks) <- c('chrom', 'start', 'end')
# write.table(peaks, 'peaks_epitrace_top100_kidney_proximal_convoluted_tubule_epithelial_cell.bed',
#             sep='\t', quote=FALSE, row.names=FALSE, col.names=FALSE)

# # BiocManager::install("TxDb.Mmusculus.UCSC.mm10.knownGene")
# # BiocManager::install("org.Mm.eg.db")
# library(TxDb.Mmusculus.UCSC.mm10.knownGene)
# library(ChIPseeker)
# peaks <- readPeakFile('peaks_epitrace_top100_kidney_proximal_convoluted_tubule_epithelial_cell.bed')
# peaks_anno <- annotatePeak(peak=peaks, tssRegion=c(-3000, 3000), TxDb=TxDb.Mmusculus.UCSC.mm10.knownGene, annoDb='org.Mm.eg.db')
# as.data.frame(peaks_anno)['SYMBOL']


## B cell
asso_res_mouse_clock_01 <- AssociationOfPeaksToAge(subset(epitrace_obj_age_conv_estimated_by_mouse_clock, cell_type %in% c('B cell')),
                                                   epitrace_age_name="EpiTraceAge_iterative", parallel=T, peakSetName='all')
up <- asso_res_mouse_clock_01[order(asso_res_mouse_clock_01$scaled_correlation_of_EpiTraceAge, decreasing=TRUE), ][1:20, c(1,3)]
dw <- asso_res_mouse_clock_01[order(asso_res_mouse_clock_01$scaled_correlation_of_EpiTraceAge, decreasing=FALSE), ][1:20, c(1,3)]
dat <- rbind(up, dw)
rownames(dat) <- seq(1, nrow(dat))
colnames(dat) <- c('Peaks', 'Scaled_correlation')
dat <- dat[order(dat$Scaled_correlation, decreasing=FALSE), ]
dat$Peaks <- factor(dat$Peaks, levels=dat$Peaks)
p <- ggplot(dat, aes(x=Peaks, y=Scaled_correlation)) + geom_bar(stat='identity', fill='darkred', position=position_dodge()) + coord_flip() +
                                                       scale_y_continuous(limits=c(-2.5, 3.0), breaks=seq(-2.5, 3.0, 0.5)) + theme_bw() + 
                                                       ggtitle('B cells') + theme(plot.title=element_text(hjust=0.5))
ggsave(p, filename='peaks_epitrace_corr_B_cell.pdf', dpi=300, height=10, width=6)

## annotate peaks
peaks <- as.data.frame(do.call(rbind, strsplit(as.vector(rev(dat$Peaks)), split='-')))  # use rev() to match ranking
colnames(peaks) <- c('chrom', 'start', 'end')
write.table(peaks, 'peaks_epitrace_top20_bot20_B_cell.bed',
            sep='\t', quote=FALSE, row.names=FALSE, col.names=FALSE)
peaks <- readPeakFile('peaks_epitrace_top20_bot20_B_cell.bed')
peaks_anno <- annotatePeak(peak=peaks, tssRegion=c(-3000, 3000), TxDb=TxDb.Mmusculus.UCSC.mm10.knownGene, annoDb='org.Mm.eg.db')
as.data.frame(peaks_anno)['SYMBOL']



## epithelial cell of proximal tubule
asso_res_mouse_clock_01 <- AssociationOfPeaksToAge(subset(epitrace_obj_age_conv_estimated_by_mouse_clock, cell_type %in% c('epithelial cell of proximal tubule')),
                                                   epitrace_age_name="EpiTraceAge_iterative", parallel=T, peakSetName='all')
up <- asso_res_mouse_clock_01[order(asso_res_mouse_clock_01$scaled_correlation_of_EpiTraceAge, decreasing=TRUE), ][1:20, c(1,3)]
dw <- asso_res_mouse_clock_01[order(asso_res_mouse_clock_01$scaled_correlation_of_EpiTraceAge, decreasing=FALSE), ][1:20, c(1,3)]
dat <- rbind(up, dw)
rownames(dat) <- seq(1, nrow(dat))
colnames(dat) <- c('Peaks', 'Scaled_correlation')
dat <- dat[order(dat$Scaled_correlation, decreasing=FALSE), ]
dat$Peaks <- factor(dat$Peaks, levels=dat$Peaks)
p <- ggplot(dat, aes(x=Peaks, y=Scaled_correlation)) + geom_bar(stat='identity', fill='darkred', position=position_dodge()) + coord_flip() +
                                                       scale_y_continuous(limits=c(-2.5, 2.5), breaks=seq(-2.5, 2.5, 0.5)) + theme_bw() + 
                                                       ggtitle('Epithelial cell of proximal tubule') + theme(plot.title=element_text(hjust=0.5))
ggsave(p, filename='peaks_epitrace_corr_epithelial_cell_of_proximal_tubule.pdf', dpi=300, height=10, width=6)

## annotate peaks
peaks <- as.data.frame(do.call(rbind, strsplit(as.vector(rev(dat$Peaks)), split='-')))  # use rev() to match ranking
colnames(peaks) <- c('chrom', 'start', 'end')
write.table(peaks, 'peaks_epitrace_top20_bot20_epithelial_cell_of_proximal_tubule.bed',
            sep='\t', quote=FALSE, row.names=FALSE, col.names=FALSE)
peaks <- readPeakFile('peaks_epitrace_top20_bot20_epithelial_cell_of_proximal_tubule.bed')
peaks_anno <- annotatePeak(peak=peaks, tssRegion=c(-3000, 3000), TxDb=TxDb.Mmusculus.UCSC.mm10.knownGene, annoDb='org.Mm.eg.db')
as.data.frame(peaks_anno)['SYMBOL']



## T cell
asso_res_mouse_clock_01 <- AssociationOfPeaksToAge(subset(epitrace_obj_age_conv_estimated_by_mouse_clock, cell_type %in% c('T cell')),
                                                   epitrace_age_name="EpiTraceAge_iterative", parallel=T, peakSetName='all')
up <- asso_res_mouse_clock_01[order(asso_res_mouse_clock_01$scaled_correlation_of_EpiTraceAge, decreasing=TRUE), ][1:20, c(1,3)]
dw <- asso_res_mouse_clock_01[order(asso_res_mouse_clock_01$scaled_correlation_of_EpiTraceAge, decreasing=FALSE), ][1:20, c(1,3)]
dat <- rbind(up, dw)
rownames(dat) <- seq(1, nrow(dat))
colnames(dat) <- c('Peaks', 'Scaled_correlation')
dat <- dat[order(dat$Scaled_correlation, decreasing=FALSE), ]
dat$Peaks <- factor(dat$Peaks, levels=dat$Peaks)
p <- ggplot(dat, aes(x=Peaks, y=Scaled_correlation)) + geom_bar(stat='identity', fill='darkred', position=position_dodge()) + coord_flip() +
                                                       scale_y_continuous(limits=c(-2.5, 3.0), breaks=seq(-2.5, 3.0, 0.5)) + theme_bw() + 
                                                       ggtitle('T cells') + theme(plot.title=element_text(hjust=0.5))
ggsave(p, filename='peaks_epitrace_corr_T_cell.pdf', dpi=300, height=10, width=6)

## annotate peaks
peaks <- as.data.frame(do.call(rbind, strsplit(as.vector(rev(dat$Peaks)), split='-')))  # use rev() to match ranking
colnames(peaks) <- c('chrom', 'start', 'end')
write.table(peaks, 'peaks_epitrace_top20_bot20_T_cell.bed',
            sep='\t', quote=FALSE, row.names=FALSE, col.names=FALSE)
peaks <- readPeakFile('peaks_epitrace_top20_bot20_T_cell.bed')
peaks_anno <- annotatePeak(peak=peaks, tssRegion=c(-3000, 3000), TxDb=TxDb.Mmusculus.UCSC.mm10.knownGene, annoDb='org.Mm.eg.db')
as.data.frame(peaks_anno)['SYMBOL']

## macrophage
asso_res_mouse_clock_01 <- AssociationOfPeaksToAge(subset(epitrace_obj_age_conv_estimated_by_mouse_clock, cell_type %in% c('macrophage')),
                                                   epitrace_age_name="EpiTraceAge_iterative", parallel=T, peakSetName='all')
up <- asso_res_mouse_clock_01[order(asso_res_mouse_clock_01$scaled_correlation_of_EpiTraceAge, decreasing=TRUE), ][1:20, c(1,3)]
dw <- asso_res_mouse_clock_01[order(asso_res_mouse_clock_01$scaled_correlation_of_EpiTraceAge, decreasing=FALSE), ][1:20, c(1,3)]
dat <- rbind(up, dw)
rownames(dat) <- seq(1, nrow(dat))
colnames(dat) <- c('Peaks', 'Scaled_correlation')
dat <- dat[order(dat$Scaled_correlation, decreasing=FALSE), ]
dat$Peaks <- factor(dat$Peaks, levels=dat$Peaks)
p <- ggplot(dat, aes(x=Peaks, y=Scaled_correlation)) + geom_bar(stat='identity', fill='darkred', position=position_dodge()) + coord_flip() +
                                                       scale_y_continuous(limits=c(-2.5, 3.0), breaks=seq(-2.5, 3.0, 0.5)) + theme_bw() + 
                                                       ggtitle('Macrophages') + theme(plot.title=element_text(hjust=0.5))
ggsave(p, filename='peaks_epitrace_corr_macrophage.pdf', dpi=300, height=10, width=6)

## annotate peaks
peaks <- as.data.frame(do.call(rbind, strsplit(as.vector(rev(dat$Peaks)), split='-')))  # use rev() to match ranking
colnames(peaks) <- c('chrom', 'start', 'end')
write.table(peaks, 'peaks_epitrace_top20_bot20_macrophage.bed',
            sep='\t', quote=FALSE, row.names=FALSE, col.names=FALSE)
peaks <- readPeakFile('peaks_epitrace_top20_bot20_macrophage.bed')
peaks_anno <- annotatePeak(peak=peaks, tssRegion=c(-3000, 3000), TxDb=TxDb.Mmusculus.UCSC.mm10.knownGene, annoDb='org.Mm.eg.db')
as.data.frame(peaks_anno)['SYMBOL']

## kidney loop of Henle thick ascending limb epithelial cell
asso_res_mouse_clock_01 <- AssociationOfPeaksToAge(subset(epitrace_obj_age_conv_estimated_by_mouse_clock, cell_type %in% c('kidney loop of Henle thick ascending limb epithelial cell')),
                                                   epitrace_age_name="EpiTraceAge_iterative", parallel=T, peakSetName='all')
up <- asso_res_mouse_clock_01[order(asso_res_mouse_clock_01$scaled_correlation_of_EpiTraceAge, decreasing=TRUE), ][1:20, c(1,3)]
dw <- asso_res_mouse_clock_01[order(asso_res_mouse_clock_01$scaled_correlation_of_EpiTraceAge, decreasing=FALSE), ][1:20, c(1,3)]
dat <- rbind(up, dw)
rownames(dat) <- seq(1, nrow(dat))
colnames(dat) <- c('Peaks', 'Scaled_correlation')
dat <- dat[order(dat$Scaled_correlation, decreasing=FALSE), ]
dat$Peaks <- factor(dat$Peaks, levels=dat$Peaks)
p <- ggplot(dat, aes(x=Peaks, y=Scaled_correlation)) + geom_bar(stat='identity', fill='darkred', position=position_dodge()) + coord_flip() +
                                                       scale_y_continuous(limits=c(-2.0, 3.0), breaks=seq(-2.0, 3.0, 0.5)) + theme_bw() + 
                                                       ggtitle('Kidney loop of Henle thick ascending limb epithelial cell') + theme(plot.title=element_text(hjust=0.5))
ggsave(p, filename='peaks_epitrace_corr_kidney_loop_of_Henle_thick_ascending_limb_epithelial_cell.pdf', dpi=300, height=10, width=6)

## annotate peaks
peaks <- as.data.frame(do.call(rbind, strsplit(as.vector(rev(dat$Peaks)), split='-')))  # use rev() to match ranking
colnames(peaks) <- c('chrom', 'start', 'end')
write.table(peaks, 'peaks_epitrace_top20_bot20_kidney_loop_of_Henle_thick_ascending_limb_epithelial_cell.bed',
            sep='\t', quote=FALSE, row.names=FALSE, col.names=FALSE)
peaks <- readPeakFile('peaks_epitrace_top20_bot20_kidney_loop_of_Henle_thick_ascending_limb_epithelial_cell.bed')
peaks_anno <- annotatePeak(peak=peaks, tssRegion=c(-3000, 3000), TxDb=TxDb.Mmusculus.UCSC.mm10.knownGene, annoDb='org.Mm.eg.db')
as.data.frame(peaks_anno)['SYMBOL']


######### plot for each age
library(dplyr)
library(ggplot2)

meta <- read.table('epitrace_meta.txt', sep='\t')
cell_info <- read.table('scATAC_meta.tsv', sep='\t', col.names=c('cell', 'age', 'cell_type'))
meta_cell_info <- inner_join(meta, cell_info) 

sort(table(meta_cell_info['cell_type']), decreasing=TRUE)[1:10]
#         kidney proximal convoluted tubule epithelial cell 
#                                                      3179 
#                                                    B cell 
#                                                      2612 
#                        epithelial cell of proximal tubule 
#                                                      1691 
#                                                lymphocyte 
#                                                      1272 
#                                                    T cell 
#                                                      1168 
#                                                macrophage 
#                                                      1166 
# kidney loop of Henle thick ascending limb epithelial cell 
#                                                      1116 
#                     kidney collecting duct principal cell 
#                                                       627 
#                                          fenestrated cell 
#                                                       577 
#           kidney distal convoluted tubule epithelial cell 
#                                                       449

## kidney proximal convoluted tubule epithelial cell
dat <- meta_cell_info %>% filter(cell_type=='kidney proximal convoluted tubule epithelial cell') %>% 
       select(EpiTraceAge_iterative, age) %>% dplyr::rename(epitrace_age=EpiTraceAge_iterative)

# dat %>% filter(age=='1m') %>% summarize(mean_value=mean(epitrace_age))  # 0.5720584
# dat %>% filter(age=='3m') %>% summarize(mean_value=mean(epitrace_age))  # 0.5702354
# dat %>% filter(age=='18m') %>% summarize(mean_value=mean(epitrace_age))  # 0.6278309
# dat %>% filter(age=='21m') %>% summarize(mean_value=mean(epitrace_age))  # 0.6548316
# dat %>% filter(age=='30m') %>% summarize(mean_value=mean(epitrace_age))  # 0.5969956

# dat %>% filter(age=='1m') %>% summarize(mean_value=median(epitrace_age))  # 0.5757576
# dat %>% filter(age=='3m') %>% summarize(mean_value=median(epitrace_age))  # 0.5703463
# dat %>% filter(age=='18m') %>% summarize(mean_value=median(epitrace_age))  # 0.6645022
# dat %>% filter(age=='21m') %>% summarize(mean_value=median(epitrace_age))  # 0.6872294
# dat %>% filter(age=='30m') %>% summarize(mean_value=median(epitrace_age))  # 0.6093074

dat['age'] <- factor(dat[['age']], levels=c('1m', '3m', '18m', '21m', '30m'))
table(dat['age'])
#  1m  3m 18m 21m 30m 
# 543 488 864 683 601
x_labels = names(table(dat['age']))
x_count = as.vector(table(dat['age']))
p <- ggplot(dat, aes(x=age, y=epitrace_age, fill=age)) + geom_boxplot(width=0.5, show.legend=FALSE, outlier.alpha=0) +
                                                         scale_x_discrete(breaks=x_labels, labels=paste0(x_labels, '\n(N=', x_count, ')')) +
                                                         scale_y_continuous(limits=c(0, 1), breaks=seq(0, 1, 0.2)) + theme_bw() +
                                                         ggtitle('Kidney proximal convoluted tubule epithelial cells') + theme(plot.title=element_text(hjust = 0.5))
ggsave(p, filename='epitrace_age_kidney_proximal_convoluted_tubule_epithelial_cell.pdf', dpi=300, height=4, width=6)


## B cell
dat <- meta_cell_info %>% filter(cell_type=='B cell') %>% 
       select(EpiTraceAge_iterative, age) %>% dplyr::rename(epitrace_age=EpiTraceAge_iterative)

dat['age'] <- factor(dat[['age']], levels=c('1m', '3m', '18m', '21m', '24m', '30m'))
table(dat['age'])
# 1m   3m  18m  21m  24m  30m 
#  8    9   44   37 2447   67 
x_labels = names(table(dat['age']))
x_count = as.vector(table(dat['age']))
p <- ggplot(dat, aes(x=age, y=epitrace_age, fill=age)) + geom_boxplot(width=0.5, show.legend=FALSE, outlier.alpha=0) + 
                                                         scale_x_discrete(breaks=x_labels, labels=paste0(x_labels, '\n(N=', x_count, ')')) +
                                                         scale_y_continuous(limits=c(0, 1), breaks=seq(0, 1, 0.2)) + theme_bw() +
                                                         ggtitle('B cells') + theme(plot.title=element_text(hjust = 0.5))
ggsave(p, filename='epitrace_age_B_cell.pdf', dpi=300, height=4, width=6)


## epithelial cell of proximal tubule
dat <- meta_cell_info %>% filter(cell_type=='epithelial cell of proximal tubule') %>% 
       select(EpiTraceAge_iterative, age) %>% dplyr::rename(epitrace_age=EpiTraceAge_iterative)

dat['age'] <- factor(dat[['age']], levels=c('1m', '3m', '18m', '21m', '30m'))
table(dat['age'])
# 1m   3m  18m  21m  30m 
# 56  132  231   34 1238
x_labels = names(table(dat['age']))
x_count = as.vector(table(dat['age']))
p <- ggplot(dat, aes(x=age, y=epitrace_age, fill=age)) + geom_boxplot(width=0.5, show.legend=FALSE, outlier.alpha=0) + 
                                                         scale_x_discrete(breaks=x_labels, labels=paste0(x_labels, '\n(N=', x_count, ')')) +
                                                         scale_y_continuous(limits=c(0, 1), breaks=seq(0, 1, 0.2)) + theme_bw() +
                                                         ggtitle('Epithelial cells of proximal tubule') + theme(plot.title=element_text(hjust = 0.5))
ggsave(p, filename='epitrace_age_epithelial_cell_of_proximal_tubule.pdf', dpi=300, height=4, width=6)


##  T cell
dat <- meta_cell_info %>% filter(cell_type=='T cell') %>% 
       select(EpiTraceAge_iterative, age) %>% dplyr::rename(epitrace_age=EpiTraceAge_iterative)

dat['age'] <- factor(dat[['age']], levels=c('1m', '3m', '18m', '21m', '24m', '30m'))
table(dat['age'])
# 1m  3m 18m 21m 24m 30m 
# 10  17  54  41 868 178
x_labels = names(table(dat['age']))
x_count = as.vector(table(dat['age']))
p <- ggplot(dat, aes(x=age, y=epitrace_age, fill=age)) + geom_boxplot(width=0.5, show.legend=FALSE, outlier.alpha=0) + 
                                                         scale_x_discrete(breaks=x_labels, labels=paste0(x_labels, '\n(N=', x_count, ')')) +
                                                         scale_y_continuous(limits=c(0, 1), breaks=seq(0, 1, 0.2)) + theme_bw() +
                                                         ggtitle('T cells') + theme(plot.title=element_text(hjust = 0.5))
ggsave(p, filename='epitrace_age_T_cells.pdf', dpi=300, height=4, width=6)


##  macrophage
dat <- meta_cell_info %>% filter(cell_type=='macrophage') %>% 
       select(EpiTraceAge_iterative, age) %>% dplyr::rename(epitrace_age=EpiTraceAge_iterative)

dat['age'] <- factor(dat[['age']], levels=c('1m', '3m', '18m', '21m', '24m', '30m'))
table(dat['age'])
# 1m  3m 18m 21m 24m 30m 
# 52 114 225  88 244 443
x_labels = names(table(dat['age']))
x_count = as.vector(table(dat['age']))
p <- ggplot(dat, aes(x=age, y=epitrace_age, fill=age)) + geom_boxplot(width=0.5, show.legend=FALSE, outlier.alpha=0) + 
                                                         scale_x_discrete(breaks=x_labels, labels=paste0(x_labels, '\n(N=', x_count, ')')) +
                                                         scale_y_continuous(limits=c(0, 1), breaks=seq(0, 1, 0.2)) + theme_bw() +
                                                         ggtitle('Macrophages') + theme(plot.title=element_text(hjust = 0.5))
ggsave(p, filename='epitrace_age_macrophages.pdf', dpi=300, height=4, width=6)


## kidney loop of Henle thick ascending limb epithelial cell
dat <- meta_cell_info %>% filter(cell_type=='kidney loop of Henle thick ascending limb epithelial cell') %>% 
       select(EpiTraceAge_iterative, age) %>% dplyr::rename(epitrace_age=EpiTraceAge_iterative)

dat['age'] <- factor(dat[['age']], levels=c('1m', '3m', '18m', '21m', '30m'))
table(dat['age'])
#  1m  3m 18m 21m 30m 
# 329 218 175 160 234 
x_labels = names(table(dat['age']))
x_count = as.vector(table(dat['age']))
p <- ggplot(dat, aes(x=age, y=epitrace_age, fill=age)) + geom_boxplot(width=0.5, show.legend=FALSE, outlier.alpha=0) + 
                                                         scale_x_discrete(breaks=x_labels, labels=paste0(x_labels, '\n(N=', x_count, ')')) +
                                                         scale_y_continuous(limits=c(0, 1), breaks=seq(0, 1, 0.2)) + theme_bw() +
                                                         ggtitle('Kidney loop of Henle thick ascending limb epithelial cells') + theme(plot.title=element_text(hjust = 0.5))
ggsave(p, filename='epitrace_age_kidney_loop_of_Henle_thick_ascending_limb_epithelial_cell.pdf', dpi=300, height=4, width=6)



## plot expression level of certain genes
import scanpy as sc
from plotnine import *
import pandas as pd
import numpy as np

rna = sc.read_h5ad('../rna_unpaired.h5ad')
rna.layers['counts'] = rna.X.copy()
sc.pp.normalize_total(rna)
sc.pp.log1p(rna)

meta = pd.read_table('epitrace_meta.txt')
rna_meta = rna[rna.obs.index.isin(meta.index)]

dat = rna_meta[rna_meta.obs['cell_type']=='kidney proximal convoluted tubule epithelial cell']

df = pd.DataFrame({'age': dat.obs['age'], 'exp': dat[:, dat.var.index=='Cdkn1a'].X.toarray().flatten()})

# df = pd.DataFrame({'age': dat.obs['age'], 'exp': dat[:, dat.var.index=='Pantr2'].X.toarray().flatten()})
# [df[df['age']=='1m']['exp'].mean(), df[df['age']=='3m']['exp'].mean(), df[df['age']=='18m']['exp'].mean(), df[df['age']=='21m']['exp'].mean(), df[df['age']=='30m']['exp'].mean()]

df['age'] = pd.Categorical(df['age'], categories=['1m', '3m', '18m', '21m', '30m'])
p = ggplot(df, aes(x='age', y='exp', fill='age')) + geom_jitter(width=0.2, height=0, show_legend=False) + xlab('Age') + ylab('Cdkn1a expression level') +\
                                                    scale_y_continuous(limits=[0, 3], breaks=np.arange(0, 3+0.1, 0.5)) +\
                                                    stat_summary(fun_y=np.mean, geom='point', color='red', size=3, shape=0, stroke=1, show_legend=False) +\
                                                    stat_summary(fun_y=np.mean, geom='point', color='red', size=3, shape=1, stroke=1, show_legend=False) + theme_bw()
p.save(filename='gene_exp_cdkn1a_kidney_proximal_convoluted_tubule_epithelial_cell.pdf', dpi=300, height=4, width=4)

df = pd.DataFrame({'age': dat.obs['age'], 'exp': dat[:, dat.var.index=='Kcns3'].X.toarray().flatten()})
df['age'] = pd.Categorical(df['age'], categories=['1m', '3m', '18m', '21m', '30m'])
p = ggplot(df, aes(x='age', y='exp', fill='age')) + geom_jitter(width=0.2, height=0, show_legend=False) + xlab('Age') + ylab('Kcns3 expression level') +\
                                                    scale_y_continuous(limits=[0, 3], breaks=np.arange(0, 3+0.1, 0.5)) +\
                                                    stat_summary(fun_y=np.mean, geom='point', color='red', size=3, shape=0, stroke=1, show_legend=False) +\
                                                    stat_summary(fun_y=np.mean, geom='point', color='red', size=3, shape=1, stroke=1, show_legend=False) + theme_bw()
p.save(filename='gene_exp_orc1_kidney_proximal_convoluted_tubule_epithelial_cell.pdf', dpi=300, height=4, width=4)







# import statsmodels.api as sm
# from scipy.stats import norm

# X = [1, 3, 18, 21, 30]
# X = sm.add_constant(X)
# Y = [0.5721, 0.5702, 0.6278, 0.6548, 0.5970]
# model = sm.OLS(Y, X)
# results = model.fit()
# beta_hat = results.params[1]
# se_beta_hat = results.bse[1]
# wald_statistic = beta_hat / se_beta_hat
# p_value = 2 * (1 - norm.cdf(abs(wald_statistic)))


# development_stage & age








































