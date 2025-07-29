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
## workdir: /fs/home/jiluzhang/Nature_methods/Revision
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



########################### scenario 2 ###########################
## cifm
## workdir: /fs/home/jiluzhang/Nature_methods/Figure_1/scenario_2/cisformer
import scanpy as sc
from scipy.stats import pearsonr
import numpy as np
from tqdm import tqdm

true = sc.read_h5ad('atac_test.h5ad')
cifm = sc.read_h5ad('atac_cisformer.h5ad')

cor_lst = []
for i in range(true.shape[0]):
    cor_lst.append(pearsonr(cifm.X[i], true.X[i].toarray().flatten())[0])

np.mean(cor_lst)    # 0.46535761818774146
np.median(cor_lst)  # 0.49286890175895115

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

np.mean(precision_lst)                                                                                            # 0.29212144129011797
np.mean(recall_lst)                                                                                               # 0.8551389655999201
np.mean([2*precision_lst[i]*recall_lst[i]/(precision_lst[i]+recall_lst[i]) for i in range(len(precision_lst))])   # 0.42748705091119527


## scbt
## workdir: /fs/home/jiluzhang/Nature_methods/Figure_1/scenario_2/scbt
import scanpy as sc
from scipy.stats import pearsonr
import numpy as np
from tqdm import tqdm

true = sc.read_h5ad('atac_test.h5ad')
scbt = sc.read_h5ad('atac_scbt.h5ad')

cor_lst = []
for i in range(true.shape[0]):
    cor_lst.append(pearsonr(scbt.X[i].toarray().flatten(), true.X[i].toarray().flatten())[0])

np.mean(cor_lst)    # 0.5334896929888864
np.median(cor_lst)  # 0.5656176072671576

m = scbt.X.toarray()
scbt_X_0_1 = ((m.T>m.T.mean(axis=0)).T) & (m>m.mean(axis=0)).astype(int)
np.save('scbt_X_0_1.npy', scbt_X_0_1)

scbt_X_0_1 = np.load('scbt_X_0_1.npy')
precision_lst = []
recall_lst = []
for i in tqdm(range(true.shape[0]), ncols=80):
    precision_lst.append(sum(scbt_X_0_1[i] * true.X[i].toarray().flatten()) / scbt_X_0_1[i].sum())
    recall_lst.append(sum(scbt_X_0_1[i] * true.X[i].toarray().flatten()) / true.X[i].toarray().sum())

np.mean(precision_lst)                                                                                            # 0.2942102354593384
np.mean(recall_lst)                                                                                               # 0.5498630914735302
np.mean([2*precision_lst[i]*recall_lst[i]/(precision_lst[i]+recall_lst[i]) for i in range(len(precision_lst))])   # 0.36827096342979804


## babel
## workdir: /fs/home/jiluzhang/Nature_methods/Figure_1/scenario_2/babel
import scanpy as sc
from scipy.stats import pearsonr
import numpy as np
from tqdm import tqdm

true = sc.read_h5ad('atac_test.h5ad')
babl = sc.read_h5ad('atac_babel.h5ad')

cor_lst = []
for i in range(true.shape[0]):
    cor_lst.append(pearsonr(babl.X[i].toarray().flatten(), true.X[i].toarray().flatten())[0])

np.mean(cor_lst)    # 0.5296898969303845
np.median(cor_lst)  # 0.5611101927763856

m = babl.X.toarray()
babl_X_0_1 = ((m.T>m.T.mean(axis=0)).T) & (m>m.mean(axis=0)).astype(int)
np.save('babl_X_0_1.npy', babl_X_0_1)

babl_X_0_1 = np.load('babl_X_0_1.npy')
precision_lst = []
recall_lst = []
for i in tqdm(range(true.shape[0]), ncols=80):
    precision_lst.append(sum(babl_X_0_1[i] * true.X[i].toarray().flatten()) / babl_X_0_1[i].sum())
    recall_lst.append(sum(babl_X_0_1[i] * true.X[i].toarray().flatten()) / true.X[i].toarray().sum())

np.nanmean(precision_lst)                                                                                            # 0.25623182837112657
np.mean(recall_lst)                                                                                                  # 0.4934721088997334
np.nanmean([2*precision_lst[i]*recall_lst[i]/(precision_lst[i]+recall_lst[i]) for i in range(len(precision_lst))])   # 0.3045336424771947


## plot metrics
## workdir: /fs/home/jiluzhang/Nature_methods/Revision
from plotnine import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['pdf.fonttype'] = 42

## Pearson correlation
dat = pd.DataFrame([0.530, 0.533, 0.465])
dat.columns = ['val']
dat['Method'] = ['BABEL', 'scButterfly', 'Cisformer']
dat['Method'] = pd.Categorical(dat['Method'], categories=['BABEL', 'scButterfly', 'Cisformer'])
p = ggplot(dat, aes(x='Method', y='val', fill='Method')) + geom_bar(stat='identity', position=position_dodge(), width=0.75) + ylab('') +\
                                                           scale_y_continuous(limits=[0, 0.6], breaks=np.arange(0, 0.6+0.05, 0.1)) + theme_bw()
p.save(filename='s_2_pearson.pdf', dpi=600, height=4, width=6)

## precision & recell & F1 score
dat = pd.DataFrame([0.256, 0.493, 0.305,
                    0.294, 0.550, 0.368,
                    0.292, 0.855, 0.427], columns=['val'])
dat['Method'] = ['BABEL']*3 + ['scButterfly']*3 + ['Cisformer']*3
dat['Metrics'] = ['Precision', 'Recall', 'F1 score']*3
dat['Method'] = pd.Categorical(dat['Method'], categories=['BABEL', 'scButterfly', 'Cisformer'])
dat['Metrics'] = pd.Categorical(dat['Metrics'], categories=['Precision', 'Recall', 'F1 score'])
p = ggplot(dat, aes(x='Metrics', y='val', fill='Method')) + geom_bar(stat='identity', position=position_dodge(), width=0.75) + ylab('') +\
                                                            scale_y_continuous(limits=[0, 0.9], breaks=np.arange(0, 0.9+0.1, 0.3)) + theme_bw()
p.save(filename='s_2_precision_recall_f1.pdf', dpi=600, height=4, width=6)



########################### scenario 3 ###########################
## cifm
## workdir: /fs/home/jiluzhang/Nature_methods/Figure_1/scenario_3/cisformer
import scanpy as sc
from scipy.stats import pearsonr
import numpy as np
from tqdm import tqdm

true = sc.read_h5ad('atac_test.h5ad')
cifm = sc.read_h5ad('atac_cisformer.h5ad')

cor_lst = []
for i in range(true.shape[0]):
    cor_lst.append(pearsonr(cifm.X[i], true.X[i].toarray().flatten())[0])

np.mean(cor_lst)    # 0.281577309141212
np.median(cor_lst)  # 0.293931142300529

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

np.mean(precision_lst)                                                                                            # 0.12727379368173594
np.mean(recall_lst)                                                                                               # 0.8119704445071072
np.mean([2*precision_lst[i]*recall_lst[i]/(precision_lst[i]+recall_lst[i]) for i in range(len(precision_lst))])   # 0.21170098396915063


## scbt
## workdir: /fs/home/jiluzhang/Nature_methods/Figure_1/scenario_3/scbt
import scanpy as sc
from scipy.stats import pearsonr
import numpy as np
from tqdm import tqdm

true = sc.read_h5ad('atac_test.h5ad')
scbt = sc.read_h5ad('atac_scbt.h5ad')

cor_lst = []
for i in range(true.shape[0]):
    cor_lst.append(pearsonr(scbt.X[i].toarray().flatten(), true.X[i].toarray().flatten())[0])

np.mean(cor_lst)    # 0.3233229031569915
np.median(cor_lst)  # 0.33819520147794685

m = scbt.X.toarray()
scbt_X_0_1 = ((m.T>m.T.mean(axis=0)).T) & (m>m.mean(axis=0)).astype(int)
np.save('scbt_X_0_1.npy', scbt_X_0_1)

scbt_X_0_1 = np.load('scbt_X_0_1.npy')
precision_lst = []
recall_lst = []
for i in tqdm(range(true.shape[0]), ncols=80):
    precision_lst.append(sum(scbt_X_0_1[i] * true.X[i].toarray().flatten()) / scbt_X_0_1[i].sum())
    recall_lst.append(sum(scbt_X_0_1[i] * true.X[i].toarray().flatten()) / true.X[i].toarray().sum())

np.mean(precision_lst)                                                                                            # 0.11771718798454622
np.mean(recall_lst)                                                                                               # 0.506809532239657
np.mean([2*precision_lst[i]*recall_lst[i]/(precision_lst[i]+recall_lst[i]) for i in range(len(precision_lst))])   # 0.17636299958116106


## babel
## workdir: /fs/home/jiluzhang/Nature_methods/Figure_1/scenario_3/babel
import scanpy as sc
from scipy.stats import pearsonr
import numpy as np
from tqdm import tqdm

true = sc.read_h5ad('atac_test.h5ad')
babl = sc.read_h5ad('atac_babel.h5ad')

cor_lst = []
for i in range(true.shape[0]):
    cor_lst.append(pearsonr(babl.X[i].toarray().flatten(), true.X[i].toarray().flatten())[0])

np.mean(cor_lst)    # 0.322622401658141
np.median(cor_lst)  # 0.33825707271695243

m = babl.X.toarray()
babl_X_0_1 = ((m.T>m.T.mean(axis=0)).T) & (m>m.mean(axis=0)).astype(int)
np.save('babl_X_0_1.npy', babl_X_0_1)

babl_X_0_1 = np.load('babl_X_0_1.npy')
precision_lst = []
recall_lst = []
for i in tqdm(range(true.shape[0]), ncols=80):
    precision_lst.append(sum(babl_X_0_1[i] * true.X[i].toarray().flatten()) / babl_X_0_1[i].sum())
    recall_lst.append(sum(babl_X_0_1[i] * true.X[i].toarray().flatten()) / true.X[i].toarray().sum())

np.nanmean(precision_lst)                                                                                            # 0.10840122180565075
np.mean(recall_lst)                                                                                                  # 0.5219033202372002
np.nanmean([2*precision_lst[i]*recall_lst[i]/(precision_lst[i]+recall_lst[i]) for i in range(len(precision_lst))])   # 0.16500324688805224


## plot metrics
## workdir: /fs/home/jiluzhang/Nature_methods/Revision
from plotnine import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['pdf.fonttype'] = 42

## Pearson correlation
dat = pd.DataFrame([0.323, 0.323, 0.282])
dat.columns = ['val']
dat['Method'] = ['BABEL', 'scButterfly', 'Cisformer']
dat['Method'] = pd.Categorical(dat['Method'], categories=['BABEL', 'scButterfly', 'Cisformer'])
p = ggplot(dat, aes(x='Method', y='val', fill='Method')) + geom_bar(stat='identity', position=position_dodge(), width=0.75) + ylab('') +\
                                                           scale_y_continuous(limits=[0, 0.4], breaks=np.arange(0, 0.4+0.05, 0.1)) + theme_bw()
p.save(filename='s_3_pearson.pdf', dpi=600, height=4, width=6)

## precision & recell & F1 score
dat = pd.DataFrame([0.108, 0.522, 0.165,
                    0.118, 0.507, 0.176,
                    0.127, 0.812, 0.212], columns=['val'])
dat['Method'] = ['BABEL']*3 + ['scButterfly']*3 + ['Cisformer']*3
dat['Metrics'] = ['Precision', 'Recall', 'F1 score']*3
dat['Method'] = pd.Categorical(dat['Method'], categories=['BABEL', 'scButterfly', 'Cisformer'])
dat['Metrics'] = pd.Categorical(dat['Metrics'], categories=['Precision', 'Recall', 'F1 score'])
p = ggplot(dat, aes(x='Metrics', y='val', fill='Method')) + geom_bar(stat='identity', position=position_dodge(), width=0.75) + ylab('') +\
                                                            scale_y_continuous(limits=[0, 0.9], breaks=np.arange(0, 0.9+0.1, 0.3)) + theme_bw()
p.save(filename='s_3_precision_recall_f1.pdf', dpi=600, height=4, width=6)



########################### scenario 4 ###########################
## cifm
## workdir: /fs/home/jiluzhang/Nature_methods/Figure_1/scenario_4/cisformer
import scanpy as sc
from scipy.stats import pearsonr
import numpy as np
from tqdm import tqdm

true = sc.read_h5ad('atac_test.h5ad')
cifm = sc.read_h5ad('atac_cisformer.h5ad')

cor_lst = []
for i in range(true.shape[0]):
    cor_lst.append(pearsonr(cifm.X[i], true.X[i].toarray().flatten())[0])

np.mean(cor_lst)    # 0.31640261756970767
np.median(cor_lst)  # 0.3347213821530459

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

np.mean(precision_lst)                                                                                            # 0.17736152762212923
np.mean(recall_lst)                                                                                               # 0.7611214112853847
np.mean([2*precision_lst[i]*recall_lst[i]/(precision_lst[i]+recall_lst[i]) for i in range(len(precision_lst))])   # 0.2769516714707969


## scbt
## workdir: /fs/home/jiluzhang/Nature_methods/Figure_1/scenario_4/scbt
import scanpy as sc
from scipy.stats import pearsonr
import numpy as np
from tqdm import tqdm

true = sc.read_h5ad('atac_test.h5ad')
scbt = sc.read_h5ad('atac_scbt.h5ad')

cor_lst = []
for i in range(true.shape[0]):
    cor_lst.append(pearsonr(scbt.X[i].toarray().flatten(), true.X[i].toarray().flatten())[0])

np.mean(cor_lst)    # 0.3409780946017293
np.median(cor_lst)  # 0.37078905883053537

m = scbt.X.toarray()
scbt_X_0_1 = ((m.T>m.T.mean(axis=0)).T) & (m>m.mean(axis=0)).astype(int)
np.save('scbt_X_0_1.npy', scbt_X_0_1)

scbt_X_0_1 = np.load('scbt_X_0_1.npy')
precision_lst = []
recall_lst = []
for i in tqdm(range(true.shape[0]), ncols=80):
    precision_lst.append(sum(scbt_X_0_1[i] * true.X[i].toarray().flatten()) / scbt_X_0_1[i].sum())
    recall_lst.append(sum(scbt_X_0_1[i] * true.X[i].toarray().flatten()) / true.X[i].toarray().sum())

np.mean(precision_lst)                                                                                               # 0.15069134326302988
np.mean(recall_lst)                                                                                                  # 0.5391674987890309
np.nanmean([2*precision_lst[i]*recall_lst[i]/(precision_lst[i]+recall_lst[i]) for i in range(len(precision_lst))])   # 0.21480630560530967


## babel
## workdir: /fs/home/jiluzhang/Nature_methods/Figure_1/scenario_4/babel
import scanpy as sc
from scipy.stats import pearsonr
import numpy as np
from tqdm import tqdm

true = sc.read_h5ad('atac_test.h5ad')
babl = sc.read_h5ad('atac_babel.h5ad')

cor_lst = []
for i in range(true.shape[0]):
    cor_lst.append(pearsonr(babl.X[i].toarray().flatten(), true.X[i].toarray().flatten())[0])

np.mean(cor_lst)    # 0.3771810584270034
np.median(cor_lst)  # 0.4038159251894927

m = babl.X.toarray()
babl_X_0_1 = ((m.T>m.T.mean(axis=0)).T) & (m>m.mean(axis=0)).astype(int)
np.save('babl_X_0_1.npy', babl_X_0_1)

babl_X_0_1 = np.load('babl_X_0_1.npy')
precision_lst = []
recall_lst = []
for i in tqdm(range(true.shape[0]), ncols=80):
    precision_lst.append(sum(babl_X_0_1[i] * true.X[i].toarray().flatten()) / babl_X_0_1[i].sum())
    recall_lst.append(sum(babl_X_0_1[i] * true.X[i].toarray().flatten()) / true.X[i].toarray().sum())

np.nanmean(precision_lst)                                                                                            # 0.13921214240478877
np.mean(recall_lst)                                                                                                  # 0.410129800179374
np.nanmean([2*precision_lst[i]*recall_lst[i]/(precision_lst[i]+recall_lst[i]) for i in range(len(precision_lst))])   # 0.19260729305972357


## plot metrics
## workdir: /fs/home/jiluzhang/Nature_methods/Revision
from plotnine import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['pdf.fonttype'] = 42

## Pearson correlation
dat = pd.DataFrame([0.377, 0.341, 0.316])
dat.columns = ['val']
dat['Method'] = ['BABEL', 'scButterfly', 'Cisformer']
dat['Method'] = pd.Categorical(dat['Method'], categories=['BABEL', 'scButterfly', 'Cisformer'])
p = ggplot(dat, aes(x='Method', y='val', fill='Method')) + geom_bar(stat='identity', position=position_dodge(), width=0.75) + ylab('') +\
                                                           scale_y_continuous(limits=[0, 0.4], breaks=np.arange(0, 0.4+0.05, 0.1)) + theme_bw()
p.save(filename='s_4_pearson.pdf', dpi=600, height=4, width=6)

## precision & recell & F1 score
dat = pd.DataFrame([0.139, 0.410, 0.193,
                    0.151, 0.539, 0.215,
                    0.177, 0.761, 0.277], columns=['val'])
dat['Method'] = ['BABEL']*3 + ['scButterfly']*3 + ['Cisformer']*3
dat['Metrics'] = ['Precision', 'Recall', 'F1 score']*3
dat['Method'] = pd.Categorical(dat['Method'], categories=['BABEL', 'scButterfly', 'Cisformer'])
dat['Metrics'] = pd.Categorical(dat['Metrics'], categories=['Precision', 'Recall', 'F1 score'])
p = ggplot(dat, aes(x='Metrics', y='val', fill='Method')) + geom_bar(stat='identity', position=position_dodge(), width=0.75) + ylab('') +\
                                                            scale_y_continuous(limits=[0, 0.9], breaks=np.arange(0, 0.9+0.1, 0.3)) + theme_bw()
p.save(filename='s_4_precision_recall_f1.pdf', dpi=600, height=4, width=6)


######################## attention score comparison in tracks ########################
# batch_attn = F.softmax(batch_attn, dim=-1)  M2M.py (line 134)

import pandas as pd
import numpy as np
import random
from functools import partial
import sys
sys.path.append("M2Mmodel")
from utils import PairDataset
import scanpy as sc
import torch
from M2Mmodel.utils import *
from collections import Counter
import pickle as pkl
import yaml
from M2Mmodel.M2M import M2M_rna2atac

import h5py
from tqdm import tqdm

from multiprocessing import Pool
import pickle

with open("rna2atac_config_test.yaml", "r") as f:
    config = yaml.safe_load(f)

model = M2M_rna2atac(
            dim = config["model"].get("dim"),

            enc_num_gene_tokens = config["model"].get("total_gene") + 1, # +1 for <PAD>
            enc_num_value_tokens = config["model"].get('max_express') + 1, # +1 for <PAD>
            enc_depth = config["model"].get("enc_depth"),
            enc_heads = config["model"].get("enc_heads"),
            enc_ff_mult = config["model"].get("enc_ff_mult"),
            enc_dim_head = config["model"].get("enc_dim_head"),
            enc_emb_dropout = config["model"].get("enc_emb_dropout"),
            enc_ff_dropout = config["model"].get("enc_ff_dropout"),
            enc_attn_dropout = config["model"].get("enc_attn_dropout"),

            dec_depth = config["model"].get("dec_depth"),
            dec_heads = config["model"].get("dec_heads"),
            dec_ff_mult = config["model"].get("dec_ff_mult"),
            dec_dim_head = config["model"].get("dec_dim_head"),
            dec_emb_dropout = config["model"].get("dec_emb_dropout"),
            dec_ff_dropout = config["model"].get("dec_ff_dropout"),
            dec_attn_dropout = config["model"].get("dec_attn_dropout")
        )
model = model.half()
model.load_state_dict(torch.load('/fs/home/jiluzhang/Nature_methods/Figure_1/scenario_1/cisformer/save_mlt_40_large/2024-09-30_rna2atac_pbmc_34/pytorch_model.bin'))
device = torch.device('cuda:3')
model.to(device)
model.eval()

rna = sc.read_h5ad('rna_test.h5ad')     # /fs/home/jiluzhang/Nature_methods/Figure_1/scenario_4/cisformer/rna_test.h5ad
atac = sc.read_h5ad('atac_test.h5ad')   # /fs/home/jiluzhang/Nature_methods/Figure_1/scenario_4/cisformer/atac_test.h5ad

## Oligodendrocyte
random.seed(0)
oli_idx = list(np.argwhere(rna.obs['cell_anno']=='Oligodendrocyte').flatten())
oli_idx_10 = random.sample(list(oli_idx), 10)
rna[oli_idx_10].write('rna_oli_10.h5ad')
np.count_nonzero(rna[oli_idx_10].X.toarray(), axis=1).max()  # 2724

atac[oli_idx_10].write('atac_oli_10.h5ad')

# python data_preprocess.py -r rna_oli_10.h5ad -a atac_oli_10.h5ad -s pt_oli_10 --dt test -n oli_10 --config rna2atac_config_test_oli.yaml
# enc_max_len: 2724

oli_data = torch.load("./pt_oli_10/oli_10_0.pt")
dataset = PreDataset(oli_data)
dataloader_kwargs = {'batch_size': 1, 'shuffle': False}
loader = torch.utils.data.DataLoader(dataset, **dataloader_kwargs)

rna_oli = sc.read_h5ad('rna_oli_10.h5ad')
atac_oli = sc.read_h5ad('atac_oli_10.h5ad')

out = atac_oli.copy()
out.X = np.zeros(out.shape)
i = 0
for inputs in tqdm(loader, ncols=80, desc='output attention matrix with no norm'):
    rna_sequence, rna_value, atac_sequence, _, enc_pad_mask = [each.to(device) for each in inputs]
    attn_tmp = model.generate_attn_weight(rna_sequence, atac_sequence, rna_value, enc_mask=enc_pad_mask, which='decoder')[0]
    out.X[i, :] = np.nansum(attn_tmp/1000000, axis=1)
    torch.cuda.empty_cache()
    i += 1

out.write('attn_oli_10_peaks.h5ad')
















