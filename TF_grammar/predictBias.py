################## ecoli to ecoli ##################
#### convert bigwig to csv
import pyfaidx
import pyBigWig
import pandas as pd
import math
from tqdm import * 

genome = pyfaidx.Fasta("ecoli.fa")
bw = pyBigWig.open('ecoli_nakedDNA.bw')

t = 0
context_lst = []
obsBias_lst = []
idx_lst = []
start_lst = []

for i in tqdm(range(50, len(genome['NC_000913.3'][:].seq)-50), ncols=80):
    signal = bw.values('NC_000913.3', i, i+1)[0]
    if not math.isnan(signal) and signal!=0:
        context_lst.append(genome['NC_000913.3'][(i-50):(i+50+1)].seq)
        obsBias_lst.append(signal)
        ## 0:training  1:validation  2:testing
        if i<=3641652:
            idx_lst.append(0)
        elif i<=4141651:
            idx_lst.append(1)
        else:
            idx_lst.append(2)
        start_lst.append(i)
        t += 1

bw.close()

res = pd.DataFrame({'context':context_lst, 'obsBias':obsBias_lst, 'idx':idx_lst,
                    'chrom':'NC_000913.3', 'start':start_lst})   # 2289693
res['end'] = res['start']+1
res.to_csv('obsBias_ecoli.tsv', sep='\t', index=False)


#### train CNN model 
# sometimes predicted values not less than 0.1: overfitting
import os
import sys
import h5py
import pandas as pd
import numpy as np
import tqdm
import pickle
import matplotlib
import matplotlib.pyplot as plt
import multiprocessing as mp
import scipy.stats as ss
from datetime import datetime
from keras.models import load_model
from keras.models import Sequential
from keras.layers import *
from keras.models import Model
from keras import backend as K

import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = "6"
tf.config.set_logical_device_configuration(tf.config.experimental.list_physical_devices('GPU')[0], 
                                           [tf.config.LogicalDeviceConfiguration(memory_limit=40960)])

def onehot_encode(seq):
    mapping = pd.Series(index=["A", "C", "G", "T"], data=[0, 1, 2, 3])
    bases = [base for base in seq]
    base_inds = mapping[bases]
    onehot = np.zeros((len(bases), 4))
    onehot[np.arange(len(bases)), base_inds] = 1
    return onehot
 
# Load ground truth Tn5 bias data
bias_data = pd.read_csv('obsBias_ecoli.tsv', sep='\t')  # 'context' & 'obsBias' & 'idx' & 'chrom' & 'start' & 'end'

# One-hot encoding of sequence contexts
seqs = bias_data.loc[:, "context"].values
with mp.Pool(10) as pool:
   onehot_seqs = list(tqdm.tqdm(pool.imap(onehot_encode, seqs), total=len(seqs)))  # ~2 min
onehot_seqs = np.array(onehot_seqs)  # (2289693, 101, 4)

# get bias and idx
target = bias_data.loc[:, "obsBias"].values
idx = bias_data.loc[:, "idx"].values

# Split the encoded sequences and target values into training (0), validation (1) and test (1)
training_data = onehot_seqs[idx==0]
training_target = target[idx==0]
val_data = onehot_seqs[idx==1]
val_target = target[idx==1]
test_data = onehot_seqs[idx==2]
test_target = target[idx==2]

# Model definition
inputs = Input(shape=(np.shape(test_data[0])))
conv_1 = Conv1D(32, 5, padding='same', activation='relu', strides=1)(inputs)
maxpool_1 = MaxPooling1D()(conv_1)
conv_2 = Conv1D(32, 5, padding='same', activation='relu', strides=1)(maxpool_1)
maxpool_2 = MaxPooling1D()(conv_2)
conv_3 = Conv1D(32, 5, padding='same', activation='relu', strides=1)(maxpool_2)
maxpool_3 = MaxPooling1D()(conv_3)
flat = Flatten()(maxpool_3)
fc = Dense(32, activation="relu")(flat)
out = Dense(1, activation="linear")(fc)
model = Model(inputs=inputs, outputs=out)  
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse'])

# Model training
mse = tf.keras.losses.MeanSquaredError()
min_loss = np.inf
patience = 0
for n_epoch in range(100):
    
    # New training epoch
    model.fit(training_data, training_target, batch_size=256, epochs=1, 
              validation_data=(val_data, val_target))  

    # Get MSE loss on the valdation set after current epoch
    val_pred = np.transpose(model.predict(val_data))[0]
    mse_loss = mse(val_target, val_pred).numpy()
    print("MSE on validation set " + str(mse_loss))

    # Get pred-target correlation on the validation set after current epoch
    print("Pred-target correlation " + str(ss.pearsonr(val_target, val_pred)[0]))
    
    # stop training with valid loss not decrease until 10 epoches
    if mse_loss > min_loss:
        patience += 1
        if patience==10:
            break
    else:
        min_loss = mse_loss
        patience = 0
        print('save model for epoch '+str(n_epoch))
        model.save('Tn5_NN_model_epoch_'+str(n_epoch)+'.h5')


#### model evaluation
import os
import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = "7"
tf.config.set_logical_device_configuration(tf.config.experimental.list_physical_devices('GPU')[0], 
                                           [tf.config.LogicalDeviceConfiguration(memory_limit=40960)])

from keras.models import load_model
import numpy as np
import pandas as pd
import multiprocessing as mp
import tqdm
import matplotlib.pyplot as plt
import scipy.stats as ss
import pyBigWig

plt.rcParams['pdf.fonttype'] = 42

def onehot_encode(seq):
    mapping = pd.Series(index=["A", "C", "G", "T"], data=[0, 1, 2, 3])
    bases = [base for base in seq]
    base_inds = mapping[bases]
    onehot = np.zeros((len(bases), 4))
    onehot[np.arange(len(bases)), base_inds] = 1
    return onehot

bias_data = pd.read_csv('obsBias_ecoli.tsv', sep='\t')
bias_test = bias_data[bias_data['idx']==2]

seqs = bias_test.loc[:, "context"].values
with mp.Pool(10) as pool:
   onehot_seqs = list(tqdm.tqdm(pool.imap(onehot_encode, seqs), total=len(seqs)))
onehot_seqs = np.array(onehot_seqs)  # (249326, 101, 4)

test_target = bias_test.loc[:, "obsBias"].values
test_data = onehot_seqs

# Load last Tn5 bias model, which should be the currently best model
model = load_model('Tn5_NN_model_epoch_15.h5')

# Model evaluation on the test set
plt_ind = np.random.choice(np.arange(len(test_data)), 10000)
test_pred = np.transpose(model.predict(test_data))[0]

# Plot test results
plt.figure(figsize=(5, 5), dpi=100)
plt.scatter(test_target[plt_ind], test_pred[plt_ind], s=1)
plt.xlim(0-0.01, 0.8+0.01)
plt.ylim(0-0.01, 0.8+0.01)
plt.xlabel("Target label")
plt.ylabel("Target prediction")
plt.title("Pearson correlation = " + str(ss.pearsonr(test_target, test_pred)[0]))   # 0.917956254958737
plt.savefig('testing_correlation_epoch_15.pdf')

# Extract regions from bigwig files
bw = pyBigWig.open("regions_test_pred.bw", "w")
chroms = {"NC_000913.3": 4641652}
bw.addHeader(list(chroms.items()))
test_pred_trans = test_pred.astype(np.float64)  # avoid the bug
bw.addEntries(chroms=['NC_000913.3']*(bias_test.shape[0]),
              starts=list(bias_test['start'].values),
              ends=list(bias_test['end'].values),
              values=list(test_pred_trans))
bw.close()


# multiBigwigSummary bins -b regions_test_pred.bw regions_test.bw -o test.npz --outRawCounts test.tab -l pred raw -bs 1 -p 20
# grep -v nan test.tab | sed 1d > test_nonan.tab
# rm test.tab
# /fs/home/jiluzhang/TF_grammar/cnn_bias_model/data/cal_cor --file test_nonan.tab



########## Ecoli to human ##########
#### bigwig to tsv
import pyfaidx
import pyBigWig
import pandas as pd
import math
from tqdm import * 

genome = pyfaidx.Fasta("hg38.fa")
bw = pyBigWig.open('human_nakedDNA.bw')

t = 0
context_lst = []
obsBias_lst = []
idx_lst = []
start_lst = []

for i in tqdm(range(46679982, 46709983), ncols=80):
    signal = bw.values('chr21', i, i+1)[0]
    if not math.isnan(signal) and signal!=0:
        context_lst.append(genome['chr21'][(i-50):(i+50+1)].seq)
        obsBias_lst.append(signal)
        start_lst.append(i)
        t += 1

bw.close()

res = pd.DataFrame({'context':context_lst, 'obsBias':obsBias_lst,
                    'chrom':'chr21', 'start':start_lst})   # 7183
res['end'] = res['start']+1
res['idx'] = 2
res = res[['context', 'obsBias', 'idx', 'chrom', 'start', 'end']]
res.to_csv('obsBias_human_test.tsv', sep='\t', index=False)


#### test using human naked dna
import os
import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = "7"
tf.config.set_logical_device_configuration(tf.config.experimental.list_physical_devices('GPU')[0], 
                                           [tf.config.LogicalDeviceConfiguration(memory_limit=40960)])

from keras.models import load_model
import numpy as np
import pandas as pd
import multiprocessing as mp
import tqdm
import matplotlib.pyplot as plt
import scipy.stats as ss
import pyBigWig

plt.rcParams['pdf.fonttype'] = 42

def onehot_encode(seq):
    mapping = pd.Series(index=["A", "C", "G", "T"], data=[0, 1, 2, 3])
    bases = [base for base in seq]
    base_inds = mapping[bases]
    onehot = np.zeros((len(bases), 4))
    onehot[np.arange(len(bases)), base_inds] = 1
    return onehot

bias_data = pd.read_csv('obsBias_human_test.tsv', sep='\t')   # 7183
bias_test = bias_data[bias_data['idx']==2]
bias_test['context'] = [s.upper() for s in bias_test['context']]  # lower to upper
bias_test = bias_test[~bias_test['context'].str.contains('N')]   # remove seqs with 'N'

seqs = bias_test.loc[:, "context"].values
with mp.Pool(10) as pool:
   onehot_seqs = list(tqdm.tqdm(pool.imap(onehot_encode, seqs), total=len(seqs)))
onehot_seqs = np.array(onehot_seqs)  # (7177, 101, 4)

test_target = bias_test.loc[:, "obsBias"].values
test_data = onehot_seqs

# Load last Tn5 bias model, which should be the currently best model
model = load_model('../ecoli_to_ecoli/Tn5_NN_model_epoch_15.h5')

# Model evaluation on the test set
plt_ind = np.random.choice(np.arange(len(test_data)), 7177)
test_pred = np.transpose(model.predict(test_data))[0]

# Plot test results
plt.figure(figsize=(5, 5), dpi=100)
plt.scatter(test_target[plt_ind], test_pred[plt_ind], s=1)
plt.xlim(0-0.01, 0.8+0.01)
plt.ylim(0-0.01, 0.8+0.01)
plt.xlabel("Target label")
plt.ylabel("Target prediction")
plt.title("Pearson correlation = " + str(ss.pearsonr(test_target, test_pred)[0]))   # 0.9188055561249172
plt.savefig('testing_correlation_ecoli_to_human.pdf')

## extract regions from bigwig files
bw = pyBigWig.open("regions_test_ecoli_to_human_pred.bw", "w")
chroms = {"chr21": 46709983}
bw.addHeader(list(chroms.items()))
test_pred_trans = test_pred.astype(np.float64)  # avoid the bug
bw.addEntries(chroms=['chr21']*(bias_test.shape[0]),
              starts=list(bias_test['start'].values),
              ends=list(bias_test['end'].values),
              values=list(test_pred_trans))
bw.close()


#### get human chr21 seq
import pyfaidx
import pandas as pd
from tqdm import * 

genome = pyfaidx.Fasta("hg38.fa")

context_lst = []
start_lst = []

for i in tqdm(range(50, 46709983-50), ncols=80):
    context = genome['chr21'][(i-50):(i+50+1)].seq.upper()
    if (not 'N' in context) and (context[50] in ['G', 'C']):  # select G/C
        context_lst.append(context)
        start_lst.append(i)  # 4.5 min

res = pd.DataFrame({'context':context_lst, 'start':start_lst})   # 16409390
res.to_csv('human_chr21_seq_GC.tsv', sep='\t', index=False)


#### predict human chr21 bias (for G/C)  15 min
## python predict_chr21_GC.py
import os
import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = "7"
tf.config.set_logical_device_configuration(tf.config.experimental.list_physical_devices('GPU')[0], 
                                           [tf.config.LogicalDeviceConfiguration(memory_limit=40960)])

from keras.models import load_model
import numpy as np
import pandas as pd
import multiprocessing as mp
import tqdm
import matplotlib.pyplot as plt
import scipy.stats as ss
import pyBigWig
import math

def onehot_encode(seq):
    mapping = pd.Series(index=["A", "C", "G", "T"], data=[0, 1, 2, 3])
    bases = [base for base in seq]
    base_inds = mapping[bases]
    onehot = np.zeros((len(bases), 4))
    onehot[np.arange(len(bases)), base_inds] = 1
    return onehot

seq_data = pd.read_csv('human_chr21_seq_GC.tsv', sep='\t')   # 40083519

model = load_model('../ecoli_to_ecoli/Tn5_NN_model_epoch_15.h5')

chunk_size = math.ceil(seq_data.shape[0]/1000000)
for i in tqdm.tqdm(range(chunk_size), ncols=80):
    seqs = seq_data.loc[:, "context"].values[(i*1000000):(i*1000000+1000000)]
    with mp.Pool(10) as pool:
        onehot_seqs = np.array(pool.map(onehot_encode, seqs))   # onehot_seqs = list(tqdm.tqdm(pool.imap(onehot_encode, seqs), total=len(seqs)))
    if i==0:
        pred = np.transpose(model.predict(onehot_seqs, batch_size=128, verbose=0))[0]
    else:
        pred = np.concatenate([pred, np.transpose(model.predict(onehot_seqs, batch_size=128, verbose=0))[0]])

bw = pyBigWig.open("pred_chr21_GC.bw", "w")
chroms = {"chr21": 46709983}
bw.addHeader(list(chroms.items()))
pred_trans = pred.astype(np.float64)  # avoid the bug
bw.addEntries(chroms=['chr21']*(seq_data.shape[0]),
              starts=list(seq_data['start'].values),
              ends=list(seq_data['start'].values+1),
              values=list(pred_trans))
bw.close()


# ## select G/C from bw file
# import pyBigWig
# import pyfaidx
# from tqdm import tqdm

# genome = pyfaidx.Fasta('hg38.fa')

# with pyBigWig.open('pred_chr21.bw') as bw:
#     chroms = bw.chroms()
    
#     with pyBigWig.open('pred_chr21_corrected.bw', "w") as out_bw:
#         out_bw.addHeader(list(chroms.items()))
        
#         for chrom, length in chroms.items():
#             intervals = bw.intervals(chrom)    
#             new_intervals = []
#             for start, end, value in tqdm(intervals, ncols=80):
#                 seq = genome[chrom][start:end].seq.upper()
#                 if seq in ['G', 'C']:
#                     new_intervals.append((start, end, value))
            
#             starts = [x[0] for x in new_intervals]
#             ends = [x[1] for x in new_intervals]
#             values = [x[2] for x in new_intervals]
#             out_bw.addEntries([chrom] * len(starts), starts, ends=ends, values=values)


#### Bias correction
python /fs/home/jiluzhang/TF_grammar/ACCESS-ATAC/bias_correction/correct_bias.py --bw_raw HepG2_7.5U.bw --bw_bias pred_chr21_GC.bw \
                                                                                 --bed_file regions_test.bed --extend 0 --window 101 \
                                                                                 --pseudo_count 1 --out_dir . --out_name ecoli_to_human_naked_dna_corrected_GC_NA \
                                                                                 --chrom_size_file hg38.chrom.sizes
for tf in ctcf atf3 elf4 myc nfib pbx3 sox13 tcf7 tead3 yy1 atf7 cebpa elk4 foxa1 gata2 hoxa5 irf3 klf11 nfyc rara;do
    TOBIAS PlotAggregate --TFBS /fs/home/jiluzhang/TF_grammar/cnn_bias_model/data/bias_correction/$tf\_chip_motif.bed \
                                /fs/home/jiluzhang/TF_grammar/cnn_bias_model/data/bias_correction/$tf\_nochip_motif.bed \
                         --signals ecoli_to_human_naked_dna_corrected_GC_NA.norm.bw --output $tf\_ecoli_to_human_print_GC_NA.pdf > $tf\_ecoli_to_human_print_GC_NA.log
    TOBIAS Log2Table --logfiles $tf\_ecoli_to_human_print_GC_NA.log --outdir $tf\_ecoli_to_human_print_GC_NA_log2table_output
    echo $tf ecoli_to_human_print done
done

for tf in ctcf atf3 elf4 myc nfib pbx3 sox13 tcf7 tead3 yy1 atf7 cebpa elk4 foxa1 gata2 hoxa5 irf3 klf11 nfyc rara;do
    echo $tf >> tfs.txt
    awk '{if($3==20) print$6}' ./$tf\_ecoli_to_human_print_GC_NA_log2table_output/aggregate_FPD.txt | sed -n '1p' >> ecoli_to_human_print_GC_NA_chip_FPD.txt
    awk '{if($3==20) print$6}' ./$tf\_ecoli_to_human_print_GC_NA_log2table_output/aggregate_FPD.txt | sed -n '2p' >> ecoli_to_human_print_GC_NA_nochip_FPD.txt
    awk '{if($3==20) print$6}' /fs/home/jiluzhang/TF_grammar/cnn_bias_model/data/bias_correction/$tf\_raw_log2table_output/aggregate_FPD.txt | sed -n '1p' >> raw_chip_FPD.txt
    awk '{if($3==20) print$6}' /fs/home/jiluzhang/TF_grammar/cnn_bias_model/data/bias_correction/$tf\_raw_log2table_output/aggregate_FPD.txt | sed -n '2p' >> raw_nochip_FPD.txt
    awk '{if($3==20) print$6}' /fs/home/jiluzhang/TF_grammar/cnn_bias_model/data/bias_correction/$tf\_ecoli_to_human_log2table_output/aggregate_FPD.txt | sed -n '1p' >> ecoli_to_human_chip_FPD.txt
    awk '{if($3==20) print$6}' /fs/home/jiluzhang/TF_grammar/cnn_bias_model/data/bias_correction/$tf\_ecoli_to_human_log2table_output/aggregate_FPD.txt | sed -n '2p' >> ecoli_to_human_nochip_FPD.txt
    awk '{if($3==20) print$6}' /fs/home/jiluzhang/TF_grammar/cnn_bias_model/data/bias_correction/$tf\_human_to_human_log2table_output/aggregate_FPD.txt | sed -n '1p' >> human_to_human_chip_FPD.txt
    awk '{if($3==20) print$6}' /fs/home/jiluzhang/TF_grammar/cnn_bias_model/data/bias_correction/$tf\_human_to_human_log2table_output/aggregate_FPD.txt | sed -n '2p' >> human_to_human_nochip_FPD.txt
    awk '{if($3==20) print$6}' /fs/home/jiluzhang/TF_grammar/cnn_bias_model/data/bias_correction/$tf\_human_raw_log2table_output/aggregate_FPD.txt | sed -n '1p' >> human_raw_chip_FPD.txt
    awk '{if($3==20) print$6}' /fs/home/jiluzhang/TF_grammar/cnn_bias_model/data/bias_correction/$tf\_human_raw_log2table_output/aggregate_FPD.txt | sed -n '2p' >> human_raw_nochip_FPD.txt
    awk '{if($3==20) print$6}' /fs/home/jiluzhang/TF_grammar/cnn_bias_model/data/bias_correction/$tf\_FootTrack_log2table_output/aggregate_FPD.txt | sed -n '1p' >> FootTrack_chip_FPD.txt
    awk '{if($3==20) print$6}' /fs/home/jiluzhang/TF_grammar/cnn_bias_model/data/bias_correction/$tf\_FootTrack_log2table_output/aggregate_FPD.txt | sed -n '2p' >> FootTrack_nochip_FPD.txt
    awk '{if($3==20) print$6}' /fs/home/jiluzhang/TF_grammar/cnn_bias_model/data/bias_correction/$tf\_human_foottrack_local_log2table_output/aggregate_FPD.txt | sed -n '1p' >> human_foottrack_local_chip_FPD.txt
    awk '{if($3==20) print$6}' /fs/home/jiluzhang/TF_grammar/cnn_bias_model/data/bias_correction/$tf\_human_foottrack_local_log2table_output/aggregate_FPD.txt | sed -n '2p' >> human_foottrack_local_nochip_FPD.txt
    echo $tf done
done

paste tfs.txt raw_chip_FPD.txt raw_nochip_FPD.txt \
              ecoli_to_human_print_GC_NA_chip_FPD.txt ecoli_to_human_print_GC_NA_nochip_FPD.txt \
              ecoli_to_human_chip_FPD.txt ecoli_to_human_nochip_FPD.txt \
              human_to_human_chip_FPD.txt human_to_human_nochip_FPD.txt \
              human_raw_chip_FPD.txt human_raw_nochip_FPD.txt \
              FootTrack_chip_FPD.txt FootTrack_nochip_FPD.txt \
              human_foottrack_local_chip_FPD.txt human_foottrack_local_nochip_FPD.txt > tfs_FPD_GC_NA.txt


import pandas as pd
from plotnine import *
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['pdf.fonttype'] = 42

dat = pd.read_table('tfs_FPD_GC_NA.txt', header=None)

df = pd.DataFrame({'alg':['raw']*40+['ecoli_to_human_print']*40+['ecoli_to_human']*40+['human_to_human']*40+['human_raw']*40+['FootTrack_global']*40+['FootTrack_local']*40,
                   'tf_chip':(['chip']*20+['nochip']*20)*7,
                   'FPD':0})

df.loc[(df['alg']=='raw')&(df['tf_chip']=='chip'), 'FPD'] = dat[1].values
df.loc[(df['alg']=='raw')&(df['tf_chip']=='nochip'), 'FPD'] = dat[2].values
df.loc[(df['alg']=='ecoli_to_human_print')&(df['tf_chip']=='chip'), 'FPD'] = dat[3].values
df.loc[(df['alg']=='ecoli_to_human_print')&(df['tf_chip']=='nochip'), 'FPD'] = dat[4].values
df.loc[(df['alg']=='ecoli_to_human')&(df['tf_chip']=='chip'), 'FPD'] = dat[5].values
df.loc[(df['alg']=='ecoli_to_human')&(df['tf_chip']=='nochip'), 'FPD'] = dat[6].values
df.loc[(df['alg']=='human_to_human')&(df['tf_chip']=='chip'), 'FPD'] = dat[7].values
df.loc[(df['alg']=='human_to_human')&(df['tf_chip']=='nochip'), 'FPD'] = dat[8].values
df.loc[(df['alg']=='human_raw')&(df['tf_chip']=='chip'), 'FPD'] = dat[9].values
df.loc[(df['alg']=='human_raw')&(df['tf_chip']=='nochip'), 'FPD'] = dat[10].values
df.loc[(df['alg']=='FootTrack_global')&(df['tf_chip']=='chip'), 'FPD'] = dat[11].values
df.loc[(df['alg']=='FootTrack_global')&(df['tf_chip']=='nochip'), 'FPD'] = dat[12].values
df.loc[(df['alg']=='FootTrack_local')&(df['tf_chip']=='chip'), 'FPD'] = dat[13].values
df.loc[(df['alg']=='FootTrack_local')&(df['tf_chip']=='nochip'), 'FPD'] = dat[14].values

df['alg'] = pd.Categorical(df['alg'], categories=['raw', 'ecoli_to_human_print', 'human_raw', 'ecoli_to_human', 'human_to_human', 'FootTrack_global', 'FootTrack_local'])
p = ggplot(df, aes(x='alg', y='FPD', fill='tf_chip')) + geom_boxplot(width=0.5, show_legend=True, outlier_shape='') + xlab('') +\
                                                         coord_cartesian(ylim=(-0.08, 0.08)) +\
                                                         scale_y_continuous(breaks=np.arange(-0.08, 0.08+0.02, 0.04)) +\
                                                         theme_bw()+ theme(axis_text_x=element_text(angle=45, hjust=1))
p.save(filename='tfs_FPD_GC_NA_20.pdf', dpi=600, height=4, width=4)

## paired t test
stats.ttest_rel(df.loc[(df['alg']=='raw')&(df['tf_chip']=='chip'), 'FPD'], 
                df.loc[(df['alg']=='raw')&(df['tf_chip']=='nochip'), 'FPD'], alternative='less')[1]                   # 1.238310454214302e-06
stats.ttest_rel(df.loc[(df['alg']=='ecoli_to_human_print')&(df['tf_chip']=='chip'), 'FPD'], 
                df.loc[(df['alg']=='ecoli_to_human_print')&(df['tf_chip']=='nochip'), 'FPD'], alternative='less')[1]   # 0.0007637641145011214
stats.ttest_rel(df.loc[(df['alg']=='ecoli_to_human')&(df['tf_chip']=='chip'), 'FPD'], 
                df.loc[(df['alg']=='ecoli_to_human')&(df['tf_chip']=='nochip'), 'FPD'], alternative='less')[1]   # 0.0011192045793338558      
stats.ttest_rel(df.loc[(df['alg']=='human_to_human')&(df['tf_chip']=='chip'), 'FPD'], 
                df.loc[(df['alg']=='human_to_human')&(df['tf_chip']=='nochip'), 'FPD'], alternative='less')[1]   # 0.0002339827175529051
stats.ttest_rel(df.loc[(df['alg']=='human_raw')&(df['tf_chip']=='chip'), 'FPD'], 
                df.loc[(df['alg']=='human_raw')&(df['tf_chip']=='nochip'), 'FPD'], alternative='less')[1]        # 0.0017942748421807659
stats.ttest_rel(df.loc[(df['alg']=='FootTrack_global')&(df['tf_chip']=='chip'), 'FPD'], 
                df.loc[(df['alg']=='FootTrack_global')&(df['tf_chip']=='nochip'), 'FPD'], alternative='less')[1]        # 0.000145318602947415
stats.ttest_rel(df.loc[(df['alg']=='FootTrack_local')&(df['tf_chip']=='chip'), 'FPD'], 
                df.loc[(df['alg']=='FootTrack_local')&(df['tf_chip']=='nochip'), 'FPD'], alternative='less')[1]        # 0.0005022915052293431

#### Plot TF footprint
## KLF11
computeMatrix reference-point --referencePoint center -p 10 -S ecoli_to_human_naked_dna_corrected_GC_NA.norm.bw \
                              -R /fs/home/jiluzhang/TF_grammar/cnn_bias_model/data/bias_correction/klf11_chip_motif.bed \
                              -o klf11_ecoli_to_human_print_GC_NA.gz -a 50 -b 50 -bs 1
plotProfile -m klf11_ecoli_to_human_print_GC_NA.gz -out deeptools_plot/klf11_ecoli_to_human_print_GC_NA.pdf

## CEBPA
computeMatrix reference-point --referencePoint center -p 10 -S ecoli_to_human_naked_dna_corrected_GC_NA.norm.bw \
                              -R /fs/home/jiluzhang/TF_grammar/cnn_bias_model/data/bias_correction/cebpa_chip_motif.bed \
                              -o cebpa_ecoli_to_human_print_GC_NA.gz -a 50 -b 50 -bs 2
plotProfile -m cebpa_ecoli_to_human_print_GC_NA.gz -out deeptools_plot/cebpa_ecoli_to_human_print_GC_NA.pdf

## CTCF
computeMatrix reference-point --referencePoint center -p 10 -S ecoli_to_human_naked_dna_corrected_GC_NA.norm.bw \
                              -R /fs/home/jiluzhang/TF_grammar/cnn_bias_model/data/bias_correction/ctcf_chip_motif.bed \
                              -o ctcf_ecoli_to_human_print_GC_NA.gz -a 50 -b 50 -bs 1
plotProfile -m ctcf_ecoli_to_human_print_GC_NA.gz -out deeptools_plot/ctcf_ecoli_to_human_print_GC_NA.pdf



#### get chrM sequence
## training: 1-14569
## validation: 14570-15569
## testing: 15569-16569

import pyfaidx
import pyBigWig
import pandas as pd
import math
from tqdm import * 

genome = pyfaidx.Fasta("hg38.fa")
bw = pyBigWig.open('K562_ATAC_cFOOT_3w_cell_100Tn5_10U_methylation.bw')

context_lst = []
obsBias_lst = []
idx_lst = []
start_lst = []

for i in tqdm(range(50, len(genome['chrM'][:].seq)-50), ncols=80):
    seq = genome['chrM'][(i-50):(i+50+1)].seq.upper()
    signal = bw.values('chrM', i, i+1)[0]
    if (not math.isnan(signal)) and (signal!=0) and ('N' not in seq):
        context_lst.append(seq)
        obsBias_lst.append(signal)
        ## 0:training  1:validation  2:testing
        if i<=14569:
            idx_lst.append(0)
        elif i<=15569:
            idx_lst.append(1)
        else:
            idx_lst.append(2)
        start_lst.append(i)

bw.close()

res = pd.DataFrame({'context':context_lst, 'obsBias':obsBias_lst, 'idx':idx_lst,
                    'chrom':'chrM', 'start':start_lst})
res['end'] = res['start']+1
res.to_csv('obsBias_chrM.tsv', sep='\t', index=False)  # 7245

#### predict human testing chrM bias 
import os
import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = "7"
tf.config.set_logical_device_configuration(tf.config.experimental.list_physical_devices('GPU')[0], 
                                           [tf.config.LogicalDeviceConfiguration(memory_limit=40960)])

from keras.models import load_model
import numpy as np
import pandas as pd
import multiprocessing as mp
import tqdm
import matplotlib.pyplot as plt
import scipy.stats as ss
import pyBigWig
import math

def onehot_encode(seq):
    mapping = pd.Series(index=["A", "C", "G", "T"], data=[0, 1, 2, 3])
    bases = [base for base in seq]
    base_inds = mapping[bases]
    onehot = np.zeros((len(bases), 4))
    onehot[np.arange(len(bases)), base_inds] = 1
    return onehot

chrm_dat = pd.read_csv('obsBias_chrM.tsv', sep='\t') 
test_dat = chrm_dat[chrm_dat['idx']==2]
model = load_model('../ecoli_to_ecoli/Tn5_NN_model_epoch_15.h5')

seqs = test_dat.loc[:, "context"].values
onehot_seqs = np.array(list(map(onehot_encode, seqs)))
test_target = test_dat['obsBias'].values
test_pred = np.transpose(model.predict(onehot_seqs, batch_size=128, verbose=0))[0]

bw = pyBigWig.open("pred_chrM_test_PRINT_no_finetune.bw", "w")
chroms = {"chrM": 16569}
bw.addHeader(list(chroms.items()))
pred_trans = test_pred.astype(np.float64)  # avoid the bug
bw.addEntries(chroms=['chrM']*(test_dat.shape[0]),
              starts=list(test_dat['start'].values),
              ends=list(test_dat['start'].values+1),
              values=list(pred_trans))
bw.close()

# Plot test results
plt.figure(figsize=(5, 5), dpi=100)
plt.scatter(test_target, test_pred, s=1)
plt.xlim(0-0.01, 0.8+0.01)
plt.ylim(0-0.01, 0.8+0.01)
plt.xlabel("Target label")
plt.ylabel("Target prediction")
plt.title("Pearson correlation = " + str(ss.pearsonr(test_target, test_pred)[0]))   # 0.26343426393752606
plt.savefig('testing_correlation_chrM_PRINT_no_finetune.pdf')



# ##################### parameter setting (Luz) #####################
# import argparse

# parser = argparse.ArgumentParser(description='PRINT bias model')
# parser.add_argument('--main_dir', type=str, default='.', help='main directory')
# parser.add_argument('--context_radius', type=int, default=50, help='context radius')
# parser.add_argument('--n_jobs', type=int, default=5, help='core number')
# parser.add_argument('--chunk_size', type=int, default=5, help='chunk size')

# args = parser.parse_args()
# main_dir = args.main_dir
# context_radius = args.context_radius
# n_jobs = args.n_jobs
# chunk_size = args.chunk_size
# #################################################################


# # Data augmentation by taking the reverse complement sequence
# training_reverse_complement = np.flip(np.flip(training_data, axis=1), axis=2)
# training_data = np.concatenate([training_data, training_reverse_complement])
# training_target = np.concatenate([training_target, training_target])
# np.shape(training_data)

# # Randomly shuffle augmented data
# inds = np.arange(np.shape(training_data)[0])
# np.random.shuffle(inds)
# training_data = training_data[inds]
# training_target = training_target[inds]







