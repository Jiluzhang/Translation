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
## workdir: /fs/home/jiluzhang/Nature_methods/Revision/precision_recall_f1
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

with open("rna2atac_config_test.yaml", "r") as f:   # enc_max_len: 10000
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

rna.obs['cell_anno'].value_counts()
# Oligodendrocyte    1595
# Astrocyte           402
# OPC                 174
# VIP                 115
# Microglia           109
# SST                 104
# IT                   83
# PVALB                15

peaks = atac.var[['gene_ids']]
peaks['chr'] = peaks['gene_ids'].map(lambda x: x.split(':')[0])
peaks['start'] = peaks['gene_ids'].map(lambda x: x.split(':')[1].split('-')[0])
peaks['end'] = peaks['gene_ids'].map(lambda x: x.split(':')[1].split('-')[1])
peaks['start'] = peaks['start'].astype(int)
peaks['end'] = peaks['end'].astype(int)

# chr17:57356784-57357132    MSI2
#out.X[:, (peaks['chr']=='chr17') & (peaks['start']>57226000) & (peaks['end']<57650000)].sum(axis=0)
#out.X[:, (peaks['chr']=='chr17') & (peaks['start']==57356784) & (peaks['end']==57357132)].sum()


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
    out.X[i, :] = np.nansum(attn_tmp/1000, axis=1)
    out.X[i, :] = out.X[i, :]/(np.median(out.X[i, :]))
    torch.cuda.empty_cache()
    i += 1

out.write('attn_oli_10_peaks.h5ad')


## Astrocyte
random.seed(0)
ast_idx = list(np.argwhere(rna.obs['cell_anno']=='Astrocyte').flatten())
ast_idx_10 = random.sample(list(ast_idx), 10)
rna[ast_idx_10].write('rna_ast_10.h5ad')
np.count_nonzero(rna[ast_idx_10].X.toarray(), axis=1).max()  # 3384

atac[ast_idx_10].write('atac_ast_10.h5ad')

# python data_preprocess.py -r rna_ast_10.h5ad -a atac_ast_10.h5ad -s pt_ast_10 --dt test -n ast_10 --config rna2atac_config_test_ast.yaml
# enc_max_len: 3384

ast_data = torch.load("./pt_ast_10/ast_10_0.pt")
dataset = PreDataset(ast_data)
dataloader_kwargs = {'batch_size': 1, 'shuffle': False}
loader = torch.utils.data.DataLoader(dataset, **dataloader_kwargs)

rna_ast = sc.read_h5ad('rna_ast_10.h5ad')
atac_ast = sc.read_h5ad('atac_ast_10.h5ad')

out = atac_ast.copy()
out.X = np.zeros(out.shape)
i = 0
for inputs in tqdm(loader, ncols=80, desc='output attention matrix with no norm'):
    rna_sequence, rna_value, atac_sequence, _, enc_pad_mask = [each.to(device) for each in inputs]
    attn_tmp = model.generate_attn_weight(rna_sequence, atac_sequence, rna_value, enc_mask=enc_pad_mask, which='decoder')[0]
    out.X[i, :] = np.nansum(attn_tmp/1000, axis=1)
    out.X[i, :] = out.X[i, :]/(np.median(out.X[i, :]))
    torch.cuda.empty_cache()
    i += 1

out.write('attn_ast_10_peaks.h5ad')


## OPC
random.seed(0)
opc_idx = list(np.argwhere(rna.obs['cell_anno']=='OPC').flatten())
opc_idx_10 = random.sample(list(opc_idx), 10)
rna[opc_idx_10].write('rna_opc_10.h5ad')
np.count_nonzero(rna[opc_idx_10].X.toarray(), axis=1).max()  # 4595

atac[opc_idx_10].write('atac_opc_10.h5ad')

# python data_preprocess.py -r rna_opc_10.h5ad -a atac_opc_10.h5ad -s pt_opc_10 --dt test -n opc_10 --config rna2atac_config_test_opc.yaml
# enc_max_len: 4595

opc_data = torch.load("./pt_opc_10/opc_10_0.pt")
dataset = PreDataset(opc_data)
dataloader_kwargs = {'batch_size': 1, 'shuffle': False}
loader = torch.utils.data.DataLoader(dataset, **dataloader_kwargs)

rna_opc = sc.read_h5ad('rna_opc_10.h5ad')
atac_opc = sc.read_h5ad('atac_opc_10.h5ad')

out = atac_opc.copy()
out.X = np.zeros(out.shape)
i = 0
for inputs in tqdm(loader, ncols=80, desc='output attention matrix with no norm'):
    rna_sequence, rna_value, atac_sequence, _, enc_pad_mask = [each.to(device) for each in inputs]
    attn_tmp = model.generate_attn_weight(rna_sequence, atac_sequence, rna_value, enc_mask=enc_pad_mask, which='decoder')[0]
    out.X[i, :] = np.nanmean(attn_tmp/1000, axis=1)
    out.X[i, :] = out.X[i, :]/(np.median(out.X[i, :]))
    torch.cuda.empty_cache()
    i += 1

out.write('attn_opc_10_peaks.h5ad')


## VIP
random.seed(0)
vip_idx = list(np.argwhere(rna.obs['cell_anno']=='VIP').flatten())
vip_idx_10 = random.sample(list(vip_idx), 10)
rna[vip_idx_10].write('rna_vip_10.h5ad')
np.count_nonzero(rna[vip_idx_10].X.toarray(), axis=1).max()  # 6166

atac[vip_idx_10].write('atac_vip_10.h5ad')

# python data_preprocess.py -r rna_vip_10.h5ad -a atac_vip_10.h5ad -s pt_vip_10 --dt test -n vip_10 --config rna2atac_config_test_vip.yaml
# enc_max_len: 6166

vip_data = torch.load("./pt_vip_10/vip_10_0.pt")
dataset = PreDataset(vip_data)
dataloader_kwargs = {'batch_size': 1, 'shuffle': False}
loader = torch.utils.data.DataLoader(dataset, **dataloader_kwargs)

rna_vip = sc.read_h5ad('rna_vip_10.h5ad')
atac_vip = sc.read_h5ad('atac_vip_10.h5ad')

out = atac_vip.copy()
out.X = np.zeros(out.shape)
i = 0
for inputs in tqdm(loader, ncols=80, desc='output attention matrix with no norm'):
    rna_sequence, rna_value, atac_sequence, _, enc_pad_mask = [each.to(device) for each in inputs]
    attn_tmp = model.generate_attn_weight(rna_sequence, atac_sequence, rna_value, enc_mask=enc_pad_mask, which='decoder')[0]
    out.X[i, :] = np.nanmean(attn_tmp/1000, axis=1)
    out.X[i, :] = out.X[i, :]/(np.median(out.X[i, :]))
    torch.cuda.empty_cache()
    i += 1

out.write('attn_vip_10_peaks.h5ad')


## Microglia
random.seed(0)
mic_idx = list(np.argwhere(rna.obs['cell_anno']=='Microglia').flatten())
mic_idx_10 = random.sample(list(mic_idx), 10)
rna[mic_idx_10].write('rna_mic_10.h5ad')
np.count_nonzero(rna[mic_idx_10].X.toarray(), axis=1).max()  # 5751

atac[mic_idx_10].write('atac_mic_10.h5ad')

# python data_preprocess.py -r rna_mic_10.h5ad -a atac_mic_10.h5ad -s pt_mic_10 --dt test -n mic_10 --config rna2atac_config_test_mic.yaml
# enc_max_len: 5751

mic_data = torch.load("./pt_mic_10/mic_10_0.pt")
dataset = PreDataset(mic_data)
dataloader_kwargs = {'batch_size': 1, 'shuffle': False}
loader = torch.utils.data.DataLoader(dataset, **dataloader_kwargs)

rna_mic = sc.read_h5ad('rna_mic_10.h5ad')
atac_mic = sc.read_h5ad('atac_mic_10.h5ad')

out = atac_mic.copy()
out.X = np.zeros(out.shape)
i = 0
for inputs in tqdm(loader, ncols=80, desc='output attention matrix with no norm'):
    rna_sequence, rna_value, atac_sequence, _, enc_pad_mask = [each.to(device) for each in inputs]
    attn_tmp = model.generate_attn_weight(rna_sequence, atac_sequence, rna_value, enc_mask=enc_pad_mask, which='decoder')[0]
    out.X[i, :] = np.nanmean(attn_tmp/1000, axis=1)
    out.X[i, :] = out.X[i, :]/(np.median(out.X[i, :]))
    torch.cuda.empty_cache()
    i += 1

out.write('attn_mic_10_peaks.h5ad')


## SST
random.seed(0)
sst_idx = list(np.argwhere(rna.obs['cell_anno']=='SST').flatten())
sst_idx_10 = random.sample(list(sst_idx), 10)
rna[sst_idx_10].write('rna_sst_10.h5ad')
np.count_nonzero(rna[sst_idx_10].X.toarray(), axis=1).max()  # 6240

atac[sst_idx_10].write('atac_sst_10.h5ad')

# python data_preprocess.py -r rna_sst_10.h5ad -a atac_sst_10.h5ad -s pt_sst_10 --dt test -n sst_10 --config rna2atac_config_test_sst.yaml
# enc_max_len: 6240

sst_data = torch.load("./pt_sst_10/sst_10_0.pt")
dataset = PreDataset(sst_data)
dataloader_kwargs = {'batch_size': 1, 'shuffle': False}
loader = torch.utils.data.DataLoader(dataset, **dataloader_kwargs)

rna_sst = sc.read_h5ad('rna_sst_10.h5ad')
atac_sst = sc.read_h5ad('atac_sst_10.h5ad')

out = atac_sst.copy()
out.X = np.zeros(out.shape)
i = 0
for inputs in tqdm(loader, ncols=80, desc='output attention matrix with no norm'):
    rna_sequence, rna_value, atac_sequence, _, enc_pad_mask = [each.to(device) for each in inputs]
    attn_tmp = model.generate_attn_weight(rna_sequence, atac_sequence, rna_value, enc_mask=enc_pad_mask, which='decoder')[0]
    out.X[i, :] = np.nanmean(attn_tmp/1000, axis=1)
    out.X[i, :] = out.X[i, :]/(np.median(out.X[i, :]))
    torch.cuda.empty_cache()
    i += 1

out.write('attn_sst_10_peaks.h5ad')


## IT
random.seed(0)
it_idx = list(np.argwhere(rna.obs['cell_anno']=='IT').flatten())
it_idx_10 = random.sample(list(it_idx), 10)
rna[it_idx_10].write('rna_it_10.h5ad')
np.count_nonzero(rna[it_idx_10].X.toarray(), axis=1).max()  # 9665

atac[it_idx_10].write('atac_it_10.h5ad')

# python data_preprocess.py -r rna_it_10.h5ad -a atac_it_10.h5ad -s pt_it_10 --dt test -n it_10 --config rna2atac_config_test_it.yaml
# enc_max_len: 9665

it_data = torch.load("./pt_it_10/it_10_0.pt")
dataset = PreDataset(it_data)
dataloader_kwargs = {'batch_size': 1, 'shuffle': False}
loader = torch.utils.data.DataLoader(dataset, **dataloader_kwargs)

rna_it = sc.read_h5ad('rna_it_10.h5ad')
atac_it = sc.read_h5ad('atac_it_10.h5ad')

out = atac_it.copy()
out.X = np.zeros(out.shape)
i = 0
for inputs in tqdm(loader, ncols=80, desc='output attention matrix with no norm'):
    rna_sequence, rna_value, atac_sequence, _, enc_pad_mask = [each.to(device) for each in inputs]
    attn_tmp = model.generate_attn_weight(rna_sequence, atac_sequence, rna_value, enc_mask=enc_pad_mask, which='decoder')[0]
    out.X[i, :] = np.nanmean(attn_tmp/1000, axis=1)
    out.X[i, :] = out.X[i, :]/(np.median(out.X[i, :]))
    torch.cuda.empty_cache()
    i += 1

out.write('attn_it_10_peaks.h5ad')


## PVALB
random.seed(0)
pva_idx = list(np.argwhere(rna.obs['cell_anno']=='PVALB').flatten())
pva_idx_10 = random.sample(list(pva_idx), 10)
rna[pva_idx_10].write('rna_pva_10.h5ad')
np.count_nonzero(rna[pva_idx_10].X.toarray(), axis=1).max()  # 5231

atac[pva_idx_10].write('atac_pva_10.h5ad')

# python data_preprocess.py -r rna_pva_10.h5ad -a atac_pva_10.h5ad -s pt_pva_10 --dt test -n pva_10 --config rna2atac_config_test_pva.yaml
# enc_max_len: 5231

pva_data = torch.load("./pt_pva_10/pva_10_0.pt")
dataset = PreDataset(pva_data)
dataloader_kwargs = {'batch_size': 1, 'shuffle': False}
loader = torch.utils.data.DataLoader(dataset, **dataloader_kwargs)

rna_pva = sc.read_h5ad('rna_pva_10.h5ad')
atac_pva = sc.read_h5ad('atac_pva_10.h5ad')

out = atac_pva.copy()
out.X = np.zeros(out.shape)
i = 0
for inputs in tqdm(loader, ncols=80, desc='output attention matrix with no norm'):
    rna_sequence, rna_value, atac_sequence, _, enc_pad_mask = [each.to(device) for each in inputs]
    attn_tmp = model.generate_attn_weight(rna_sequence, atac_sequence, rna_value, enc_mask=enc_pad_mask, which='decoder')[0]
    out.X[i, :] = np.nanmean(attn_tmp/1000, axis=1)
    out.X[i, :] = out.X[i, :]/(np.median(out.X[i, :]))
    torch.cuda.empty_cache()
    i += 1

out.write('attn_pva_10_peaks.h5ad')


## Plot attention score
from plotnine import *
import scanpy as sc
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

plt.rcParams['pdf.fonttype'] = 42

atac = sc.read_h5ad('atac_test.h5ad')
peaks = atac.var[['gene_ids']]
peaks['chr'] = peaks['gene_ids'].map(lambda x: x.split(':')[0])
peaks['start'] = peaks['gene_ids'].map(lambda x: x.split(':')[1].split('-')[0])
peaks['end'] = peaks['gene_ids'].map(lambda x: x.split(':')[1].split('-')[1])
peaks['start'] = peaks['start'].astype(int)
peaks['end'] = peaks['end'].astype(int)

# chr17:57356784-57357132    MSI2
idx = (peaks['chr']=='chr17') & (peaks['start']>57226000) & (peaks['end']<57650000)
peaks[(peaks['chr']=='chr17') & (peaks['start']>57226000) & (peaks['end']<57650000)][15:18]
#                                         gene_ids    chr     start       end
# chr17:57356784-57357132  chr17:57356784-57357132  chr17  57356784  57357132
# chr17:57357146-57357345  chr17:57357146-57357345  chr17  57357146  57357345
# chr17:57357455-57357802  chr17:57357455-57357802  chr17  57357455  57357802

hlt = [0]*85
hlt[15] = 1
hlt[16] = 1
hlt[17] = 1

def out_df(cell_anno='oli'):
    oli = sc.read_h5ad('attn_'+cell_anno+'_10_peaks.h5ad')
    oli_lst = oli.X[:, idx].mean(axis=0)
    oli_lst[~np.isfinite(oli_lst)] = oli_lst[np.isfinite(oli_lst)].max()
    oli_lst = (oli_lst-oli_lst.min())/(oli_lst.max()-oli_lst.min())    
    dat = pd.DataFrame({'cell_anno':cell_anno, 'attn':oli_lst, 'hlt':hlt})
    return dat

dat = pd.concat([out_df('oli'), out_df('ast'), out_df('opc'),
                 out_df('vip'), out_df('mic'), out_df('sst'),
                 out_df('it'),  out_df('pva')])

dat['cell_anno'] = pd.Categorical(dat['cell_anno'], categories=['ast', 'it', 'mic', 'opc', 'oli', 'pva', 'sst', 'vip'])

p = ggplot(dat, aes(x='cell_anno', y='attn', color='cell_anno')) + geom_boxplot(width=0.5, show_legend=False, outlier_shape='') + xlab('') +\
                                                                   coord_cartesian(ylim=(0, 0.6)) +\
                                                                   scale_y_continuous(breaks=np.arange(0, 1.0+0.1, 0.1)) +\
                                                                   geom_point(dat[dat['hlt']==1], color='red', size=1) + theme_bw()
p.save(filename='track_attn_box.pdf', dpi=600, height=4, width=4)

# Oligodendrocyte    1595
# Astrocyte           402
# OPC                 174
# VIP                 115
# Microglia           109
# SST                 104
# IT                   83
# PVALB                15


########################### pan-cancer ###########################
## cifm
## workdir: /fs/home/jiluzhang/Nature_methods/Figure_2/pan_cancer/cisformer
import scanpy as sc
from scipy.stats import pearsonr
import numpy as np
from tqdm import tqdm
import random

true = sc.read_h5ad('atac_test.h5ad')
cifm = sc.read_h5ad('atac_cisformer_umap.h5ad')

## all cell types
cifm_X_0_1 = cifm.X.toarray()

random.seed(0)
idx_lst = random.sample(list(range(true.shape[0])), 500)

## cell 0 -> cell 499
precision_lst = []
recall_lst = []
for i in tqdm(range(len(idx_lst)), ncols=80):
    precision_lst.append(sum(cifm_X_0_1[i] * true.X[i].toarray().flatten()) / cifm_X_0_1[i].sum())
    recall_lst.append(sum(cifm_X_0_1[i] * true.X[i].toarray().flatten()) / true.X[i].toarray().sum())

np.mean(precision_lst)                                                                                            # 0.062182749667773395
np.mean(recall_lst)                                                                                               # 0.8694642632272895
np.mean([2*precision_lst[i]*recall_lst[i]/(precision_lst[i]+recall_lst[i]) for i in range(len(precision_lst))])   # 0.11428709808962342

## random 500 cells
precision_lst = []
recall_lst = []
for i in tqdm(idx_lst, ncols=80):
    precision_lst.append(sum(cifm_X_0_1[i] * true.X[i].toarray().flatten()) / cifm_X_0_1[i].sum())
    recall_lst.append(sum(cifm_X_0_1[i] * true.X[i].toarray().flatten()) / true.X[i].toarray().sum())

np.mean(precision_lst)                                                                                            # 0.08207397206414342
np.mean(recall_lst)                                                                                               # 0.8389998535174341
np.mean([2*precision_lst[i]*recall_lst[i]/(precision_lst[i]+recall_lst[i]) for i in range(len(precision_lst))])   # 0.14587817733034122

## 'B-cells', 'Endothelial', 'Fibroblasts', 'Macrophages', 'T-cells'

## B cell
random.seed(0)
bcell_idx_lst = random.sample(list(np.where(cifm.obs['cell_anno']=='B-cells')[0]), 500)

precision_lst = []
recall_lst = []
for i in tqdm(bcell_idx_lst, ncols=80):
    precision_lst.append(sum(cifm_X_0_1[i] * true.X[i].toarray().flatten()) / cifm_X_0_1[i].sum())
    recall_lst.append(sum(cifm_X_0_1[i] * true.X[i].toarray().flatten()) / true.X[i].toarray().sum())

np.mean(precision_lst)                                                                                            # 0.08619832931655753
np.mean(recall_lst)                                                                                               # 0.8609875596239799
np.mean([2*precision_lst[i]*recall_lst[i]/(precision_lst[i]+recall_lst[i]) for i in range(len(precision_lst))])   # 0.15304619612167406

## Endothelial 
random.seed(0)
endo_idx_lst = random.sample(list(np.where(cifm.obs['cell_anno']=='Endothelial')[0]), 
                             min(sum(cifm.obs['cell_anno']=='Endothelial'), 500))

precision_lst = []
recall_lst = []
for i in tqdm(endo_idx_lst, ncols=80):
    precision_lst.append(sum(cifm_X_0_1[i] * true.X[i].toarray().flatten()) / cifm_X_0_1[i].sum())
    recall_lst.append(sum(cifm_X_0_1[i] * true.X[i].toarray().flatten()) / true.X[i].toarray().sum())

np.mean(precision_lst)                                                                                            # 0.06428872713702377
np.mean(recall_lst)                                                                                               # 0.8042201928981897
np.mean([2*precision_lst[i]*recall_lst[i]/(precision_lst[i]+recall_lst[i]) for i in range(len(precision_lst))])   # 0.11619799792714877

## Fibroblasts
random.seed(0)
fibro_idx_lst = random.sample(list(np.where(cifm.obs['cell_anno']=='Fibroblasts')[0]), 500)

precision_lst = []
recall_lst = []
for i in tqdm(fibro_idx_lst, ncols=80):
    precision_lst.append(sum(cifm_X_0_1[i] * true.X[i].toarray().flatten()) / cifm_X_0_1[i].sum())
    recall_lst.append(sum(cifm_X_0_1[i] * true.X[i].toarray().flatten()) / true.X[i].toarray().sum())

np.mean(precision_lst)                                                                                            # 0.07813243690980005
np.mean(recall_lst)                                                                                               # 0.8008469322864205
np.mean([2*precision_lst[i]*recall_lst[i]/(precision_lst[i]+recall_lst[i]) for i in range(len(precision_lst))])   # 0.13803724796444586

## Macrophages
random.seed(0)
macro_idx_lst = random.sample(list(np.where(cifm.obs['cell_anno']=='Macrophages')[0]), 500)

precision_lst = []
recall_lst = []
for i in tqdm(macro_idx_lst, ncols=80):
    precision_lst.append(sum(cifm_X_0_1[i] * true.X[i].toarray().flatten()) / cifm_X_0_1[i].sum())
    recall_lst.append(sum(cifm_X_0_1[i] * true.X[i].toarray().flatten()) / true.X[i].toarray().sum())

np.mean(precision_lst)                                                                                            # 0.06901650120364035
np.mean(recall_lst)                                                                                               # 0.8539058774796146
np.mean([2*precision_lst[i]*recall_lst[i]/(precision_lst[i]+recall_lst[i]) for i in range(len(precision_lst))])   # 0.125336972874082

## T cell
random.seed(0)
tcell_idx_lst = random.sample(list(np.where(cifm.obs['cell_anno']=='T-cells')[0]), 500)

precision_lst = []
recall_lst = []
for i in tqdm(tcell_idx_lst, ncols=80):
    precision_lst.append(sum(cifm_X_0_1[i] * true.X[i].toarray().flatten()) / cifm_X_0_1[i].sum())
    recall_lst.append(sum(cifm_X_0_1[i] * true.X[i].toarray().flatten()) / true.X[i].toarray().sum())

np.mean(precision_lst)                                                                                            # 0.10134401268883131
np.mean(recall_lst)                                                                                               # 0.8624758350115329
np.mean([2*precision_lst[i]*recall_lst[i]/(precision_lst[i]+recall_lst[i]) for i in range(len(precision_lst))])   # 0.1762997600622369


## scbt
## workdir: /fs/home/jiluzhang/Nature_methods/Figure_2/pan_cancer/scbt

true = sc.read_h5ad('atac_test.h5ad')
scbt = sc.read_h5ad('atac_scbt_umap.h5ad')

scbt_X_0_1 = scbt.X.toarray()

random.seed(0)
idx_lst = random.sample(list(range(true.shape[0])), 500)

## cell 0 -> cell 499
precision_lst = []
recall_lst = []
for i in tqdm(range(len(idx_lst)), ncols=80):
    precision_lst.append(sum(scbt_X_0_1[i] * true.X[i].toarray().flatten()) / scbt_X_0_1[i].sum())
    recall_lst.append(sum(scbt_X_0_1[i] * true.X[i].toarray().flatten()) / true.X[i].toarray().sum())

np.mean(precision_lst)                                                                                            # 0.05888187379402629
np.mean(recall_lst)                                                                                               # 0.47700629245772347
np.mean([2*precision_lst[i]*recall_lst[i]/(precision_lst[i]+recall_lst[i]) for i in range(len(precision_lst))])   # 0.1014446268123869

## random 500 cells
precision_lst = []
recall_lst = []
for i in tqdm(idx_lst, ncols=80):
    precision_lst.append(sum(scbt_X_0_1[i] * true.X[i].toarray().flatten()) / scbt_X_0_1[i].sum())
    recall_lst.append(sum(scbt_X_0_1[i] * true.X[i].toarray().flatten()) / true.X[i].toarray().sum())

np.mean(precision_lst)                                                                                            # 0.07912352160910399
np.mean(recall_lst)                                                                                               # 0.4470605507110836
np.mean([2*precision_lst[i]*recall_lst[i]/(precision_lst[i]+recall_lst[i]) for i in range(len(precision_lst))])   # 0.12649792856105027

## 'B-cells', 'Endothelial', 'Fibroblasts', 'Macrophages', 'T-cells'

## B cell
random.seed(0)
bcell_idx_lst = random.sample(list(np.where(scbt.obs['cell_anno']=='B-cells')[0]), 500)

precision_lst = []
recall_lst = []
for i in tqdm(bcell_idx_lst, ncols=80):
    precision_lst.append(sum(scbt_X_0_1[i] * true.X[i].toarray().flatten()) / scbt_X_0_1[i].sum())
    recall_lst.append(sum(scbt_X_0_1[i] * true.X[i].toarray().flatten()) / true.X[i].toarray().sum())

np.mean(precision_lst)                                                                                            # 0.09068317974373025
np.mean(recall_lst)                                                                                               # 0.5514114979090791
np.mean([2*precision_lst[i]*recall_lst[i]/(precision_lst[i]+recall_lst[i]) for i in range(len(precision_lst))])   # 0.14771314082675732

## Endothelial 
random.seed(0)
endo_idx_lst = random.sample(list(np.where(scbt.obs['cell_anno']=='Endothelial')[0]), 
                             min(sum(scbt.obs['cell_anno']=='Endothelial'), 500))

precision_lst = []
recall_lst = []
for i in tqdm(endo_idx_lst, ncols=80):
    precision_lst.append(sum(scbt_X_0_1[i] * true.X[i].toarray().flatten()) / scbt_X_0_1[i].sum())
    recall_lst.append(sum(scbt_X_0_1[i] * true.X[i].toarray().flatten()) / true.X[i].toarray().sum())

np.mean(precision_lst)                                                                                            # 0.05860204348276823
np.mean(recall_lst)                                                                                               # 0.29205919125581237
np.mean([2*precision_lst[i]*recall_lst[i]/(precision_lst[i]+recall_lst[i]) for i in range(len(precision_lst))])   # 0.09231573263681561

## Fibroblasts
random.seed(0)
fibro_idx_lst = random.sample(list(np.where(scbt.obs['cell_anno']=='Fibroblasts')[0]), 500)

precision_lst = []
recall_lst = []
for i in tqdm(fibro_idx_lst, ncols=80):
    precision_lst.append(sum(scbt_X_0_1[i] * true.X[i].toarray().flatten()) / scbt_X_0_1[i].sum())
    recall_lst.append(sum(scbt_X_0_1[i] * true.X[i].toarray().flatten()) / true.X[i].toarray().sum())

np.mean(precision_lst)                                                                                            # 0.07605530987012625
np.mean(recall_lst)                                                                                               # 0.4265941599625745
np.mean([2*precision_lst[i]*recall_lst[i]/(precision_lst[i]+recall_lst[i]) for i in range(len(precision_lst))])   # 0.12194487282153642

## Macrophages
random.seed(0)
macro_idx_lst = random.sample(list(np.where(scbt.obs['cell_anno']=='Macrophages')[0]), 500)

precision_lst = []
recall_lst = []
for i in tqdm(macro_idx_lst, ncols=80):
    precision_lst.append(sum(scbt_X_0_1[i] * true.X[i].toarray().flatten()) / scbt_X_0_1[i].sum())
    recall_lst.append(sum(scbt_X_0_1[i] * true.X[i].toarray().flatten()) / true.X[i].toarray().sum())

np.mean(precision_lst)                                                                                            # 0.05923610786969398
np.mean(recall_lst)                                                                                               # 0.394674591247737
np.mean([2*precision_lst[i]*recall_lst[i]/(precision_lst[i]+recall_lst[i]) for i in range(len(precision_lst))])   # 0.09853872547420323

## T cell
random.seed(0)
tcell_idx_lst = random.sample(list(np.where(scbt.obs['cell_anno']=='T-cells')[0]), 500)

precision_lst = []
recall_lst = []
for i in tqdm(tcell_idx_lst, ncols=80):
    precision_lst.append(sum(scbt_X_0_1[i] * true.X[i].toarray().flatten()) / scbt_X_0_1[i].sum())
    recall_lst.append(sum(scbt_X_0_1[i] * true.X[i].toarray().flatten()) / true.X[i].toarray().sum())

np.mean(precision_lst)                                                                                            # 0.09637860393530838
np.mean(recall_lst)                                                                                               # 0.3590045609516801
np.mean([2*precision_lst[i]*recall_lst[i]/(precision_lst[i]+recall_lst[i]) for i in range(len(precision_lst))])   # 0.14107506699406633


## babel
## workdir: /fs/home/jiluzhang/Nature_methods/Figure_2/pan_cancer/babel

true = sc.read_h5ad('atac_test.h5ad')
babl = sc.read_h5ad('atac_babel_umap.h5ad')

babl_X_0_1 = babl.X.toarray()

random.seed(0)
idx_lst = random.sample(list(range(true.shape[0])), 500)

## cell 0 -> cell 499
precision_lst = []
recall_lst = []
for i in tqdm(range(len(idx_lst)), ncols=80):
    precision_lst.append(sum(babl_X_0_1[i] * true.X[i].toarray().flatten()) / babl_X_0_1[i].sum())
    recall_lst.append(sum(babl_X_0_1[i] * true.X[i].toarray().flatten()) / true.X[i].toarray().sum())

np.mean(precision_lst)                                                                                            # 0.053536932950956344
np.mean(recall_lst)                                                                                               # 0.3590095262425303
np.mean([2*precision_lst[i]*recall_lst[i]/(precision_lst[i]+recall_lst[i]) for i in range(len(precision_lst))])   # 0.08751641766203117

## random 500 cells
precision_lst = []
recall_lst = []
for i in tqdm(idx_lst, ncols=80):
    precision_lst.append(sum(babl_X_0_1[i] * true.X[i].toarray().flatten()) / babl_X_0_1[i].sum())
    recall_lst.append(sum(babl_X_0_1[i] * true.X[i].toarray().flatten()) / true.X[i].toarray().sum())

np.mean(precision_lst)                                                                                            # 0.07687440691648656
np.mean(recall_lst)                                                                                               # 0.42256071761454234
np.mean([2*precision_lst[i]*recall_lst[i]/(precision_lst[i]+recall_lst[i]) for i in range(len(precision_lst))])   # 0.12137646158600708

## 'B-cells', 'Endothelial', 'Fibroblasts', 'Macrophages', 'T-cells'

## B cell
random.seed(0)
bcell_idx_lst = random.sample(list(np.where(babl.obs['cell_anno']=='B-cells')[0]), 500)

precision_lst = []
recall_lst = []
for i in tqdm(bcell_idx_lst, ncols=80):
    precision_lst.append(sum(babl_X_0_1[i] * true.X[i].toarray().flatten()) / babl_X_0_1[i].sum())
    recall_lst.append(sum(babl_X_0_1[i] * true.X[i].toarray().flatten()) / true.X[i].toarray().sum())

np.mean(precision_lst)                                                                                            # 0.07914512541009337
np.mean(recall_lst)                                                                                               # 0.3307198874540757
np.mean([2*precision_lst[i]*recall_lst[i]/(precision_lst[i]+recall_lst[i]) for i in range(len(precision_lst))])   # 0.12032509575739175

## Endothelial 
random.seed(0)
endo_idx_lst = random.sample(list(np.where(babl.obs['cell_anno']=='Endothelial')[0]), 
                             min(sum(babl.obs['cell_anno']=='Endothelial'), 500))

precision_lst = []
recall_lst = []
for i in tqdm(endo_idx_lst, ncols=80):
    precision_lst.append(sum(babl_X_0_1[i] * true.X[i].toarray().flatten()) / babl_X_0_1[i].sum())
    recall_lst.append(sum(babl_X_0_1[i] * true.X[i].toarray().flatten()) / true.X[i].toarray().sum())

np.mean(precision_lst)                                                                                            # 0.05717498386459103
np.mean(recall_lst)                                                                                               # 0.3546358870047395
np.mean([2*precision_lst[i]*recall_lst[i]/(precision_lst[i]+recall_lst[i]) for i in range(len(precision_lst))])   # 0.08863125262965604

## Fibroblasts
random.seed(0)
fibro_idx_lst = random.sample(list(np.where(babl.obs['cell_anno']=='Fibroblasts')[0]), 500)

precision_lst = []
recall_lst = []
for i in tqdm(fibro_idx_lst, ncols=80):
    precision_lst.append(sum(babl_X_0_1[i] * true.X[i].toarray().flatten()) / babl_X_0_1[i].sum())
    recall_lst.append(sum(babl_X_0_1[i] * true.X[i].toarray().flatten()) / true.X[i].toarray().sum())

np.mean(precision_lst)                                                                                            # 0.07097899824277375
np.mean(recall_lst)                                                                                               # 0.3909794943201093
np.mean([2*precision_lst[i]*recall_lst[i]/(precision_lst[i]+recall_lst[i]) for i in range(len(precision_lst))])   # 0.1114908501366938

## Macrophages
random.seed(0)
macro_idx_lst = random.sample(list(np.where(babl.obs['cell_anno']=='Macrophages')[0]), 500)

precision_lst = []
recall_lst = []
for i in tqdm(macro_idx_lst, ncols=80):
    precision_lst.append(sum(babl_X_0_1[i] * true.X[i].toarray().flatten()) / babl_X_0_1[i].sum())
    recall_lst.append(sum(babl_X_0_1[i] * true.X[i].toarray().flatten()) / true.X[i].toarray().sum())

np.mean(precision_lst)                                                                                            # 0.06608417450171004
np.mean(recall_lst)                                                                                               # 0.49220638593478255
np.mean([2*precision_lst[i]*recall_lst[i]/(precision_lst[i]+recall_lst[i]) for i in range(len(precision_lst))])   # 0.11090390025782483

## T cell
random.seed(0)
tcell_idx_lst = random.sample(list(np.where(babl.obs['cell_anno']=='T-cells')[0]), 500)

precision_lst = []
recall_lst = []
for i in tqdm(tcell_idx_lst, ncols=80):
    precision_lst.append(sum(babl_X_0_1[i] * true.X[i].toarray().flatten()) / babl_X_0_1[i].sum())
    recall_lst.append(sum(babl_X_0_1[i] * true.X[i].toarray().flatten()) / true.X[i].toarray().sum())

np.mean(precision_lst)                                                                                            # 0.10246069987713889
np.mean(recall_lst)                                                                                               # 0.49635852955421955
np.mean([2*precision_lst[i]*recall_lst[i]/(precision_lst[i]+recall_lst[i]) for i in range(len(precision_lst))])   # 0.16179700008933404


## plot metrics
## workdir: /fs/home/jiluzhang/Nature_methods/Revision/precision_recall_f1
from plotnine import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['pdf.fonttype'] = 42

## precision & recell & F1 score (cell 0 -> cell 499)
dat = pd.DataFrame([0.054, 0.359, 0.088,
                    0.059, 0.477, 0.101,
                    0.062, 0.869, 0.114], columns=['val'])
dat['Method'] = ['BABEL']*3 + ['scButterfly']*3 + ['Cisformer']*3
dat['Metrics'] = ['Precision', 'Recall', 'F1 score']*3
dat['Method'] = pd.Categorical(dat['Method'], categories=['BABEL', 'scButterfly', 'Cisformer'])
dat['Metrics'] = pd.Categorical(dat['Metrics'], categories=['Precision', 'Recall', 'F1 score'])

p = ggplot(dat, aes(x='Metrics', y='val', fill='Method')) + geom_bar(stat='identity', position=position_dodge(), width=0.75) + ylab('') +\
                                                            scale_y_continuous(limits=[0, 0.9], breaks=np.arange(0, 0.9+0.1, 0.3)) + theme_bw()
p.save(filename='pan_cancer_precision_recall_f1_cell_0_499.pdf', dpi=600, height=4, width=6)

## precision & recell & F1 score (random 500)
dat = pd.DataFrame([0.077, 0.423, 0.121,
                    0.079, 0.447, 0.126,
                    0.082, 0.839, 0.146], columns=['val'])
dat['Method'] = ['BABEL']*3 + ['scButterfly']*3 + ['Cisformer']*3
dat['Metrics'] = ['Precision', 'Recall', 'F1 score']*3
dat['Method'] = pd.Categorical(dat['Method'], categories=['BABEL', 'scButterfly', 'Cisformer'])
dat['Metrics'] = pd.Categorical(dat['Metrics'], categories=['Precision', 'Recall', 'F1 score'])

p = ggplot(dat, aes(x='Metrics', y='val', fill='Method')) + geom_bar(stat='identity', position=position_dodge(), width=0.75) + ylab('') +\
                                                            scale_y_continuous(limits=[0, 0.9], breaks=np.arange(0, 0.9+0.1, 0.3)) + theme_bw()
p.save(filename='pan_cancer_precision_recall_f1_cell_random_500.pdf', dpi=600, height=4, width=6)

## precision & recell & F1 score (B-cells)
dat = pd.DataFrame([0.079, 0.331, 0.120,
                    0.091, 0.551, 0.148,
                    0.086, 0.861, 0.153], columns=['val'])
dat['Method'] = ['BABEL']*3 + ['scButterfly']*3 + ['Cisformer']*3
dat['Metrics'] = ['Precision', 'Recall', 'F1 score']*3
dat['Method'] = pd.Categorical(dat['Method'], categories=['BABEL', 'scButterfly', 'Cisformer'])
dat['Metrics'] = pd.Categorical(dat['Metrics'], categories=['Precision', 'Recall', 'F1 score'])

p = ggplot(dat, aes(x='Metrics', y='val', fill='Method')) + geom_bar(stat='identity', position=position_dodge(), width=0.75) + ylab('') +\
                                                            scale_y_continuous(limits=[0, 0.9], breaks=np.arange(0, 0.9+0.1, 0.3)) + theme_bw()
p.save(filename='pan_cancer_precision_recall_f1_cell_bcells.pdf', dpi=600, height=4, width=6)

## precision & recell & F1 score (Endothelial)
dat = pd.DataFrame([0.057, 0.355, 0.089,
                    0.059, 0.292, 0.092,
                    0.064, 0.804, 0.116], columns=['val'])
dat['Method'] = ['BABEL']*3 + ['scButterfly']*3 + ['Cisformer']*3
dat['Metrics'] = ['Precision', 'Recall', 'F1 score']*3
dat['Method'] = pd.Categorical(dat['Method'], categories=['BABEL', 'scButterfly', 'Cisformer'])
dat['Metrics'] = pd.Categorical(dat['Metrics'], categories=['Precision', 'Recall', 'F1 score'])

p = ggplot(dat, aes(x='Metrics', y='val', fill='Method')) + geom_bar(stat='identity', position=position_dodge(), width=0.75) + ylab('') +\
                                                            scale_y_continuous(limits=[0, 0.9], breaks=np.arange(0, 0.9+0.1, 0.3)) + theme_bw()
p.save(filename='pan_cancer_precision_recall_f1_cell_endothelial.pdf', dpi=600, height=4, width=6)

## precision & recell & F1 score (Fibroblasts)
dat = pd.DataFrame([0.071, 0.391, 0.111,
                    0.076, 0.427, 0.122,
                    0.078, 0.801, 0.138], columns=['val'])
dat['Method'] = ['BABEL']*3 + ['scButterfly']*3 + ['Cisformer']*3
dat['Metrics'] = ['Precision', 'Recall', 'F1 score']*3
dat['Method'] = pd.Categorical(dat['Method'], categories=['BABEL', 'scButterfly', 'Cisformer'])
dat['Metrics'] = pd.Categorical(dat['Metrics'], categories=['Precision', 'Recall', 'F1 score'])

p = ggplot(dat, aes(x='Metrics', y='val', fill='Method')) + geom_bar(stat='identity', position=position_dodge(), width=0.75) + ylab('') +\
                                                            scale_y_continuous(limits=[0, 0.9], breaks=np.arange(0, 0.9+0.1, 0.3)) + theme_bw()
p.save(filename='pan_cancer_precision_recall_f1_cell_fibroblast.pdf', dpi=600, height=4, width=6)

## precision & recell & F1 score (Macrophages)
dat = pd.DataFrame([0.066, 0.492, 0.111,
                    0.059, 0.395, 0.099,
                    0.069, 0.854, 0.125], columns=['val'])
dat['Method'] = ['BABEL']*3 + ['scButterfly']*3 + ['Cisformer']*3
dat['Metrics'] = ['Precision', 'Recall', 'F1 score']*3
dat['Method'] = pd.Categorical(dat['Method'], categories=['BABEL', 'scButterfly', 'Cisformer'])
dat['Metrics'] = pd.Categorical(dat['Metrics'], categories=['Precision', 'Recall', 'F1 score'])

p = ggplot(dat, aes(x='Metrics', y='val', fill='Method')) + geom_bar(stat='identity', position=position_dodge(), width=0.75) + ylab('') +\
                                                            scale_y_continuous(limits=[0, 0.9], breaks=np.arange(0, 0.9+0.1, 0.3)) + theme_bw()
p.save(filename='pan_cancer_precision_recall_f1_cell_macrophage.pdf', dpi=600, height=4, width=6)

## precision & recell & F1 score (T-cells)
dat = pd.DataFrame([0.102, 0.496, 0.162,
                    0.096, 0.359, 0.141,
                    0.101, 0.862, 0.176], columns=['val'])
dat['Method'] = ['BABEL']*3 + ['scButterfly']*3 + ['Cisformer']*3
dat['Metrics'] = ['Precision', 'Recall', 'F1 score']*3
dat['Method'] = pd.Categorical(dat['Method'], categories=['BABEL', 'scButterfly', 'Cisformer'])
dat['Metrics'] = pd.Categorical(dat['Metrics'], categories=['Precision', 'Recall', 'F1 score'])

p = ggplot(dat, aes(x='Metrics', y='val', fill='Method')) + geom_bar(stat='identity', position=position_dodge(), width=0.75) + ylab('') +\
                                                            scale_y_continuous(limits=[0, 0.9], breaks=np.arange(0, 0.9+0.1, 0.3)) + theme_bw()
p.save(filename='pan_cancer_precision_recall_f1_cell_tcell.pdf', dpi=600, height=4, width=6)


########################### aging kidney ###########################
## no ground truth !!!








