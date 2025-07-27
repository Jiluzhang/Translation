########################### scenario 1 ###########################
## cifm
## workdir: /fs/home/jiluzhang/Nature_methods/Figure_1/scenario_1/cisformer/res_mlt_40_large
import scanpy as sc
from scipy.stats import pearsonr
import numpy as np
from tqdm import tqdm

true = sc.read_h5ad('atac_test.h5ad')
cifm = sc.read_h5ad('atac_cisformer.h5ad')

cor_lst = []
for i in range(true.shape[0]):
    cor_lst.append(pearsonr(cifm.X[i], true.X[i].toarray().flatten())[0])

np.mean(cor_lst)    # 0.45932666743897077
np.median(cor_lst)  # 0.4797373276162471

cifm_X_0_1 = cifm.X
cifm_X_0_1[cifm_X_0_1>0.5]=1
cifm_X_0_1[cifm_X_0_1<=0.5]=0
np.save('cifm_X_0_1.npy', cifm_X_0_1)

cifm_X_0_1 = np.load('cifm_X_0_1.npy')
precision_lst = []
recall_lst = []
for i in tqdm(range(true.shape[0]), ncols=80):
    precision_lst.append(sum(cifm_X_0_1[i] * true.X[i].toarray().flatten()) / cifm_X_0_1[i].sum())
    recall_lst.append(sum(cifm_X_0_1[i] * true.X[i].toarray().flatten()) / true.X[i].toarray().sum())

np.mean(precision_lst)                                                                                            # 0.29239782112720863
np.mean(recall_lst)                                                                                               # 0.8448660998334653
np.mean([2*precision_lst[i]*recall_lst[i]/(precision_lst[i]+recall_lst[i]) for i in range(len(precision_lst))])   # 0.42622835660476416


## scbt
## workdir: /fs/home/jiluzhang/Nature_methods/Figure_1/scenario_1/scbt
import scanpy as sc
from scipy.stats import pearsonr
import numpy as np
from tqdm import tqdm

true = sc.read_h5ad('atac_test.h5ad')
scbt = sc.read_h5ad('atac_scbt.h5ad')

cor_lst = []
for i in range(true.shape[0]):
    cor_lst.append(pearsonr(scbt.X[i].toarray().flatten(), true.X[i].toarray().flatten())[0])

np.mean(cor_lst)    # 0.523705681809846
np.median(cor_lst)  # 0.5469777496617984

m = scbt.X.toarray()
scbt_X_0_1 = ((m.T>m.T.mean(axis=0)).T) & (m>m.mean(axis=0)).astype(int)
np.save('scbt_X_0_1.npy', scbt_X_0_1)

scbt_X_0_1 = np.load('scbt_X_0_1.npy')
precision_lst = []
recall_lst = []
for i in tqdm(range(true.shape[0]), ncols=80):
    precision_lst.append(sum(scbt_X_0_1[i] * true.X[i].toarray().flatten()) / scbt_X_0_1[i].sum())
    recall_lst.append(sum(scbt_X_0_1[i] * true.X[i].toarray().flatten()) / true.X[i].toarray().sum())

np.mean(precision_lst)                                                                                            # 0.3010578659342757
np.mean(recall_lst)                                                                                               # 0.5664364082373033
np.mean([2*precision_lst[i]*recall_lst[i]/(precision_lst[i]+recall_lst[i]) for i in range(len(precision_lst))])   # 0.381639430781429


## babel
## workdir: /fs/home/jiluzhang/Nature_methods/Figure_1/scenario_1/babel
import scanpy as sc
from scipy.stats import pearsonr
import numpy as np
from tqdm import tqdm

true = sc.read_h5ad('atac_test.h5ad')
babl = sc.read_h5ad('atac_babel.h5ad')

cor_lst = []
for i in range(true.shape[0]):
    cor_lst.append(pearsonr(babl.X[i].toarray().flatten(), true.X[i].toarray().flatten())[0])

np.mean(cor_lst)    # 0.5214679648469931
np.median(cor_lst)  # 0.5442229551965063

m = babl.X.toarray()
babl_X_0_1 = ((m.T>m.T.mean(axis=0)).T) & (m>m.mean(axis=0)).astype(int)
np.save('babl_X_0_1.npy', babl_X_0_1)

babl_X_0_1 = np.load('babl_X_0_1.npy')
precision_lst = []
recall_lst = []
for i in tqdm(range(true.shape[0]), ncols=80):
    precision_lst.append(sum(babl_X_0_1[i] * true.X[i].toarray().flatten()) / babl_X_0_1[i].sum())
    recall_lst.append(sum(babl_X_0_1[i] * true.X[i].toarray().flatten()) / true.X[i].toarray().sum())

np.mean(precision_lst)                                                                                            # 0.2798497160948761
np.mean(recall_lst)                                                                                               # 0.5474433598728647
np.mean([2*precision_lst[i]*recall_lst[i]/(precision_lst[i]+recall_lst[i]) for i in range(len(precision_lst))])   # 0.35146998351787256


## plot metrics
from plotnine import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['pdf.fonttype'] = 42

## Pearson correlation
dat = pd.DataFrame([0.521, 0.524, 0.459])
dat.columns = ['val']
dat['Method'] = ['BABEL', 'scButterfly', 'Cisformer']
dat['Method'] = pd.Categorical(dat['Method'], categories=['BABEL', 'scButterfly', 'Cisformer'])
p = ggplot(dat, aes(x='Method', y='val', fill='Method')) + geom_bar(stat='identity', position=position_dodge(), width=0.75) + ylab('') +\
                                                           scale_y_continuous(limits=[0, 0.6], breaks=np.arange(0, 0.6+0.05, 0.1)) + theme_bw()
p.save(filename='s_1_pearson.pdf', dpi=600, height=4, width=6)

## precision & recell & F1 score
dat = pd.DataFrame([0.280, 0.547, 0.351,
                    0.301, 0.566, 0.382,
                    0.292, 0.845, 0.426], columns=['val'])
dat['Method'] = ['BABEL']*3 + ['scButterfly']*3 + ['Cisformer']*3
dat['Metrics'] = ['Precision', 'Recall', 'F1 score']*3
dat['Method'] = pd.Categorical(dat['Method'], categories=['BABEL', 'scButterfly', 'Cisformer'])
dat['Metrics'] = pd.Categorical(dat['Metrics'], categories=['Precision', 'Recall', 'F1 score'])
p = ggplot(dat, aes(x='Metrics', y='val', fill='Method')) + geom_bar(stat='identity', position=position_dodge(), width=0.75) + ylab('') +\
                                                            scale_y_continuous(limits=[0, 0.9], breaks=np.arange(0, 0.9+0.1, 0.3)) + theme_bw()
p.save(filename='s_1_precision_recall_f1.pdf', dpi=600, height=4, width=6)




















