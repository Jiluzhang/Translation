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
    if not 'N' in context:
        context_lst.append(context)
        start_lst.append(i)  # 4.5 min

res = pd.DataFrame({'context':context_lst, 'start':start_lst})   # 40083519
res.to_csv('human_chr21_seq.tsv', sep='\t', index=False)


#### predict human chr21 bias
## python predict_chr21.py
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

seq_data = pd.read_csv('human_chr21_seq.tsv', sep='\t')   # 40083519

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

bw = pyBigWig.open("pred_chr21.bw", "w")
chroms = {"chr21": 46709983}
bw.addHeader(list(chroms.items()))
pred_trans = pred.astype(np.float64)  # avoid the bug
bw.addEntries(chroms=['chr21']*(seq_data.shape[0]),
              starts=list(seq_data['start'].values),
              ends=list(seq_data['start'].values+1),
              values=list(pred_trans))
bw.close()



















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







