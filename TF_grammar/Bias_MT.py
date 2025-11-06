## workdir: /fs/home/jiluzhang/TF_grammar/PRINT/cfoot/MT
scp -P 10022 u21509@logini.tongji.edu.cn:/share/home/u21509/workspace/wuang/04.tf_grammer/intermediate_process/01.bias_correct/human_nakedDNA/HepG2_gDNA_0.75U_conversion.bw .
scp -P 10022 u21509@logini.tongji.edu.cn:/share/home/u21509/workspace/wuang/04.tf_grammer/intermediate_process/01.bias_correct/HepG2/HepG2_7.5U_conversion.bw .
scp -P 10022 u21509@logini.tongji.edu.cn:/share/home/u21509/workspace/wuang/04.tf_grammer/intermediate_process/01.bias_correct/02.chrM/03.methratio/K562_ATAC_cFOOT_3w_cell_100Tn5_10U_methylation.bw .

#### get chrM bias for different samples
import pyfaidx
import pyBigWig
import pandas as pd
import math
from tqdm import * 

genome = pyfaidx.Fasta("hg38.fa")

def get_chrM_bias(bw='HepG2_7.5U_conversion.bw'):
    bw = pyBigWig.open(bw)
    
    context_lst = []
    obsBias_lst = []
    start_lst = []
    
    for i in tqdm(range(50, len(genome['chrM'][:].seq)-50), ncols=80):
        seq = genome['chrM'][(i-50):(i+50+1)].seq.upper()
        signal = bw.values('chrM', i, i+1)[0]
        if (not math.isnan(signal)) and ('N' not in seq):  #if (not math.isnan(signal)) and (signal!=0) and ('N' not in seq):
            context_lst.append(seq)
            obsBias_lst.append(signal)
            start_lst.append(i)
    
    bw.close()
    
    res = pd.DataFrame({'context':context_lst, 'obsBias':obsBias_lst,
                        'chrom':'chrM', 'start':start_lst})
    res['end'] = res['start']+1
    return res

get_chrM_bias(bw='HepG2_gDNA_0.75U_conversion.bw').to_csv('obsBias_chrM_human_naked_DNA_cfoot.tsv', sep='\t', index=False)                 # 7249
get_chrM_bias(bw='HepG2_7.5U_conversion.bw').to_csv('obsBias_chrM_HepG2_cfoot.tsv', sep='\t', index=False)                                 # 7251
get_chrM_bias(bw='K562_ATAC_cFOOT_3w_cell_100Tn5_10U_methylation.bw').to_csv('obsBias_chrM_k562_atac_cfoot.tsv', sep='\t', index=False)    # 7247


#### predict human testing chrM bias 
import os
import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
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

plt.rcParams['pdf.fonttype'] = 42

def onehot_encode(seq):
    mapping = pd.Series(index=["A", "C", "G", "T"], data=[0, 1, 2, 3])
    bases = [base for base in seq]
    base_inds = mapping[bases]
    onehot = np.zeros((len(bases), 4))
    onehot[np.arange(len(bases)), base_inds] = 1
    return onehot

model = load_model('../ecoli_to_ecoli/Tn5_NN_model_epoch_15.h5')

def out_pred_bw_plot(sample='obsBias_chrM_HepG2_cfoot.tsv'):
    chrm_dat = pd.read_csv(sample, sep='\t') 
    seqs = chrm_dat.loc[:, "context"].values
    onehot_seqs = np.array(list(map(onehot_encode, seqs)))
    test_target = chrm_dat['obsBias'].values
    test_pred = np.transpose(model.predict(onehot_seqs, batch_size=128))[0]

    ## Output bw
    bw = pyBigWig.open('Pred'+sample[3:-3]+'bw', "w")
    chroms = {"chrM": 16569}
    bw.addHeader(list(chroms.items()))
    pred_trans = test_pred.astype(np.float64)  # avoid the bug
    bw.addEntries(chroms=['chrM']*(chrm_dat.shape[0]),
                  starts=list(chrm_dat['start'].values),
                  ends=list(chrm_dat['start'].values+1),
                  values=list(pred_trans))
    bw.close()
    
    ## Plot test results
    plt.figure(figsize=(5, 5), dpi=100)
    plt.scatter(test_target, test_pred, s=1)
    plt.xlim(0-0.01, 0.8+0.01)
    plt.ylim(0-0.01, 0.8+0.01)
    plt.xlabel("Target label")
    plt.ylabel("Target prediction")
    plt.title("Pearson correlation = " + str(ss.pearsonr(test_target, test_pred)[0]))
    plt.savefig('Pred'+sample[3:-4]+'_correlation.pdf')

out_pred_bw_plot(sample='obsBias_chrM_human_naked_DNA_cfoot.tsv')
out_pred_bw_plot(sample='obsBias_chrM_HepG2_cfoot.tsv')
out_pred_bw_plot(sample='obsBias_chrM_k562_atac_cfoot.tsv')


############ split chrM based on position ############
#### train PRINT using only chrM from HepG2 cfoot data
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

os.environ["CUDA_VISIBLE_DEVICES"] = "4"
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
bias_data = pd.read_csv('obsBias_chrM_HepG2_cfoot.tsv', sep='\t')

# One-hot encoding of sequence contexts
seqs = bias_data.loc[:, "context"].values
with mp.Pool(10) as pool:
   onehot_seqs = list(tqdm.tqdm(pool.imap(onehot_encode, seqs), total=len(seqs)))
onehot_seqs = np.array(onehot_seqs)  # (7250, 101, 4)

# get bias and idx
target = bias_data.loc[:, "obsBias"].values

# Split the encoded sequences and target values into training (0), validation (1) and test (1)
training_data = onehot_seqs[:5000]
training_target = target[:5000]
val_data = onehot_seqs[5000:6250]
val_target = target[5000:6250]
test_data = onehot_seqs[6250:]
test_target = target[6250:]

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
    model.fit(training_data, training_target, batch_size=16, epochs=1, 
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
        model.save('Tn5_NN_model_only_chrM_position.h5')


#### model evaluation
import os
import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
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

bias_data = pd.read_csv('obsBias_chrM_HepG2_cfoot.tsv', sep='\t')
bias_test = bias_data[6250:]

seqs = bias_test.loc[:, "context"].values
with mp.Pool(10) as pool:
   onehot_seqs = list(tqdm.tqdm(pool.imap(onehot_encode, seqs), total=len(seqs)))
onehot_seqs = np.array(onehot_seqs)  # (1000, 101, 4)

test_target = bias_test.loc[:, "obsBias"].values
test_data = onehot_seqs

## model for only chrM
# Load last Tn5 bias model, which should be the currently best model
model = load_model('Tn5_NN_model_only_chrM_position.h5')

# Model evaluation on the test set
test_pred = np.transpose(model.predict(test_data))[0]

# Plot test results
plt.figure(figsize=(5, 5), dpi=100)
plt.scatter(test_target, test_pred, s=1)
plt.xlim(0-0.01, 0.8+0.01)
plt.ylim(0-0.01, 0.8+0.01)
plt.xlabel("Target label")
plt.ylabel("Target prediction")
plt.title("Pearson correlation = " + str(ss.pearsonr(test_target, test_pred)[0]))   # 0.6116053165848684
plt.savefig('test_correlation_only_chrM_position.pdf')

## model for ecoli naked DNA
# Load last Tn5 bias model, which should be the currently best model
model = load_model('../ecoli_to_ecoli/Tn5_NN_model_epoch_15.h5')

# Model evaluation on the test set
test_pred = np.transpose(model.predict(test_data))[0]

# Plot test results
plt.figure(figsize=(5, 5), dpi=100)
plt.scatter(test_target, test_pred, s=1)
plt.xlim(0-0.01, 0.8+0.01)
plt.ylim(0-0.01, 0.8+0.01)
plt.xlabel("Target label")
plt.ylabel("Target prediction")
plt.title("Pearson correlation = " + str(ss.pearsonr(test_target, test_pred)[0]))  # 0.6894182932563541
plt.savefig('test_correlation_ecoli_position.pdf')


#### train PRINT using ecoli plus chrM fine tune
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

os.environ["CUDA_VISIBLE_DEVICES"] = "4"
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
bias_data = pd.read_csv('obsBias_chrM_HepG2_cfoot.tsv', sep='\t')

# One-hot encoding of sequence contexts
seqs = bias_data.loc[:, "context"].values
with mp.Pool(10) as pool:
   onehot_seqs = list(tqdm.tqdm(pool.imap(onehot_encode, seqs), total=len(seqs)))
onehot_seqs = np.array(onehot_seqs)  # (7250, 101, 4)

# get bias and idx
target = bias_data.loc[:, "obsBias"].values

# Split the encoded sequences and target values into training, validation and testing
training_data = onehot_seqs[:5000]
training_target = target[:5000]
val_data = onehot_seqs[5000:6250]
val_target = target[5000:6250]
test_data = onehot_seqs[6250:]
test_target = target[6250:]

# Model definition
model = load_model('../ecoli_to_ecoli/Tn5_NN_model_epoch_15.h5') 
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse'])

# Model training
mse = tf.keras.losses.MeanSquaredError()
min_loss = np.inf
patience = 0
for n_epoch in range(100):
    
    # New training epoch
    model.fit(training_data, training_target, batch_size=16, epochs=1, 
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
        model.save('Tn5_NN_model_ecoli_chrM_position.h5')


#### model evaluation
import os
import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
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

bias_data = pd.read_csv('obsBias_chrM_HepG2_cfoot.tsv', sep='\t')
bias_test = bias_data[6250:]

seqs = bias_test.loc[:, "context"].values
with mp.Pool(10) as pool:
   onehot_seqs = list(tqdm.tqdm(pool.imap(onehot_encode, seqs), total=len(seqs)))
onehot_seqs = np.array(onehot_seqs)  # (1000, 101, 4)

test_target = bias_test.loc[:, "obsBias"].values
test_data = onehot_seqs

# Load last Tn5 bias model, which should be the currently best model
model = load_model('Tn5_NN_model_ecoli_chrM_position.h5')

# Model evaluation on the test set
test_pred = np.transpose(model.predict(test_data))[0]

# Plot test results
plt.figure(figsize=(5, 5), dpi=100)
plt.scatter(test_target, test_pred, s=1)
plt.xlim(0-0.01, 0.8+0.01)
plt.ylim(0-0.01, 0.8+0.01)
plt.xlabel("Target label")
plt.ylabel("Target prediction")
plt.title("Pearson correlation = " + str(ss.pearsonr(test_target, test_pred)[0]))   # 0.6932888904232268
plt.savefig('test_correlation_ecoli_chrM_position.pdf')


############ split chrM randomly ############
#### train PRINT using only chrM from HepG2 cfoot data
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
import random

import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = "4"
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
bias_data = pd.read_csv('obsBias_chrM_HepG2_cfoot.tsv', sep='\t')

# shuffle the data
random.seed(0)
idx_lst = list(range(bias_data.shape[0]))
random.shuffle(idx_lst)
bias_data = bias_data.iloc[idx_lst]

# One-hot encoding of sequence contexts
seqs = bias_data.loc[:, "context"].values
with mp.Pool(10) as pool:
   onehot_seqs = list(tqdm.tqdm(pool.imap(onehot_encode, seqs), total=len(seqs)))
onehot_seqs = np.array(onehot_seqs)  # (7250, 101, 4)

# get bias and idx
target = bias_data.loc[:, "obsBias"].values

# Split the encoded sequences and target values into training (0), validation (1) and test (1)
training_data = onehot_seqs[:5000]
training_target = target[:5000]
val_data = onehot_seqs[5000:6250]
val_target = target[5000:6250]
test_data = onehot_seqs[6250:]
test_target = target[6250:]

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
    model.fit(training_data, training_target, batch_size=16, epochs=1, 
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
        model.save('Tn5_NN_model_only_chrM_random.h5')


#### model evaluation
import os
import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
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
import random

plt.rcParams['pdf.fonttype'] = 42

def onehot_encode(seq):
    mapping = pd.Series(index=["A", "C", "G", "T"], data=[0, 1, 2, 3])
    bases = [base for base in seq]
    base_inds = mapping[bases]
    onehot = np.zeros((len(bases), 4))
    onehot[np.arange(len(bases)), base_inds] = 1
    return onehot

bias_data = pd.read_csv('obsBias_chrM_HepG2_cfoot.tsv', sep='\t')

# shuffle the data
random.seed(0)
idx_lst = list(range(bias_data.shape[0]))
random.shuffle(idx_lst)
bias_data = bias_data.iloc[idx_lst]

bias_test = bias_data[6250:]

seqs = bias_test.loc[:, "context"].values
with mp.Pool(10) as pool:
   onehot_seqs = list(tqdm.tqdm(pool.imap(onehot_encode, seqs), total=len(seqs)))
onehot_seqs = np.array(onehot_seqs)  # (1000, 101, 4)

test_target = bias_test.loc[:, "obsBias"].values
test_data = onehot_seqs

## model for only chrM
# Load last Tn5 bias model, which should be the currently best model
model = load_model('Tn5_NN_model_only_chrM_random.h5')

# Model evaluation on the test set
test_pred = np.transpose(model.predict(test_data))[0]

# Plot test results
plt.figure(figsize=(5, 5), dpi=100)
plt.scatter(test_target, test_pred, s=1)
plt.xlim(0-0.01, 0.8+0.01)
plt.ylim(0-0.01, 0.8+0.01)
plt.xlabel("Target label")
plt.ylabel("Target prediction")
plt.title("Pearson correlation = " + str(ss.pearsonr(test_target, test_pred)[0]))   # 0.7517524110074875
plt.savefig('test_correlation_only_chrM_random.pdf')

## model for ecoli naked DNA
# Load last Tn5 bias model, which should be the currently best model
model = load_model('../ecoli_to_ecoli/Tn5_NN_model_epoch_15.h5')

# Model evaluation on the test set
test_pred = np.transpose(model.predict(test_data))[0]

# Plot test results
plt.figure(figsize=(5, 5), dpi=100)
plt.scatter(test_target, test_pred, s=1)
plt.xlim(0-0.01, 0.8+0.01)
plt.ylim(0-0.01, 0.8+0.01)
plt.xlabel("Target label")
plt.ylabel("Target prediction")
plt.title("Pearson correlation = " + str(ss.pearsonr(test_target, test_pred)[0]))  # 0.7918957394746929
plt.savefig('test_correlation_ecoli_random.pdf')


#### train PRINT using ecoli plus chrM fine tune
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
import random

import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = "4"
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
bias_data = pd.read_csv('obsBias_chrM_HepG2_cfoot.tsv', sep='\t')

# shuffle the data
random.seed(0)
idx_lst = list(range(bias_data.shape[0]))
random.shuffle(idx_lst)
bias_data = bias_data.iloc[idx_lst]

# One-hot encoding of sequence contexts
seqs = bias_data.loc[:, "context"].values
with mp.Pool(10) as pool:
   onehot_seqs = list(tqdm.tqdm(pool.imap(onehot_encode, seqs), total=len(seqs)))
onehot_seqs = np.array(onehot_seqs)  # (7250, 101, 4)

# get bias and idx
target = bias_data.loc[:, "obsBias"].values

# Split the encoded sequences and target values into training, validation and testing
training_data = onehot_seqs[:5000]
training_target = target[:5000]
val_data = onehot_seqs[5000:6250]
val_target = target[5000:6250]
test_data = onehot_seqs[6250:]
test_target = target[6250:]

# Model definition
model = load_model('../ecoli_to_ecoli/Tn5_NN_model_epoch_15.h5') 
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse'])

# Model training
mse = tf.keras.losses.MeanSquaredError()
min_loss = np.inf
patience = 0
for n_epoch in range(100):
    
    # New training epoch
    model.fit(training_data, training_target, batch_size=16, epochs=1, 
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
        model.save('Tn5_NN_model_ecoli_chrM_random.h5')


#### model evaluation
import os
import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
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
import random

plt.rcParams['pdf.fonttype'] = 42

def onehot_encode(seq):
    mapping = pd.Series(index=["A", "C", "G", "T"], data=[0, 1, 2, 3])
    bases = [base for base in seq]
    base_inds = mapping[bases]
    onehot = np.zeros((len(bases), 4))
    onehot[np.arange(len(bases)), base_inds] = 1
    return onehot

bias_data = pd.read_csv('obsBias_chrM_HepG2_cfoot.tsv', sep='\t')

# shuffle the data
random.seed(0)
idx_lst = list(range(bias_data.shape[0]))
random.shuffle(idx_lst)
bias_data = bias_data.iloc[idx_lst]

bias_test = bias_data[6250:]

seqs = bias_test.loc[:, "context"].values
with mp.Pool(10) as pool:
   onehot_seqs = list(tqdm.tqdm(pool.imap(onehot_encode, seqs), total=len(seqs)))
onehot_seqs = np.array(onehot_seqs)  # (1000, 101, 4)

test_target = bias_test.loc[:, "obsBias"].values
test_data = onehot_seqs

# Load last Tn5 bias model, which should be the currently best model
model = load_model('Tn5_NN_model_ecoli_chrM_random.h5')

# Model evaluation on the test set
test_pred = np.transpose(model.predict(test_data))[0]

# Plot test results
plt.figure(figsize=(5, 5), dpi=100)
plt.scatter(test_target, test_pred, s=1)
plt.xlim(0-0.01, 0.8+0.01)
plt.ylim(0-0.01, 0.8+0.01)
plt.xlabel("Target label")
plt.ylabel("Target prediction")
plt.title("Pearson correlation = " + str(ss.pearsonr(test_target, test_pred)[0]))   # 0.8402885251401522
plt.savefig('test_correlation_ecoli_chrM_random.pdf')


#### correct bias using ecoli_chrM fine-tune model (for chr21)  15 min
import os
import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = "4"
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

seq_data = pd.read_csv('human_chr21_seq_GC.tsv', sep='\t')   # 16409390

model = load_model('./Tn5_NN_model_ecoli_chrM_random.h5')

chunk_size = math.ceil(seq_data.shape[0]/1000000)
for i in tqdm.tqdm(range(chunk_size), ncols=80):
    seqs = seq_data.loc[:, "context"].values[(i*1000000):(i*1000000+1000000)]
    with mp.Pool(10) as pool:
        onehot_seqs = np.array(pool.map(onehot_encode, seqs))   # onehot_seqs = list(tqdm.tqdm(pool.imap(onehot_encode, seqs), total=len(seqs)))
    if i==0:
        pred = np.transpose(model.predict(onehot_seqs, batch_size=128, verbose=0))[0]
    else:
        pred = np.concatenate([pred, np.transpose(model.predict(onehot_seqs, batch_size=128, verbose=0))[0]])

bw = pyBigWig.open("ecoli_chrM_pred_bias.bw", "w")
chroms = {"chr21": 46709983}
bw.addHeader(list(chroms.items()))
pred_trans = pred.astype(np.float64)  # avoid the bug
bw.addEntries(chroms=['chr21']*(seq_data.shape[0]),
              starts=list(seq_data['start'].values),
              ends=list(seq_data['start'].values+1),
              values=list(pred_trans))
bw.close()

## correct bias (10 min)
python /fs/home/jiluzhang/TF_grammar/ACCESS-ATAC/bias_correction/correct_bias.py --bw_raw HepG2_7.5U.bw --bw_bias ecoli_chrM_pred_bias.bw \
                                                                                 --bed_file regions_test.bed --extend 0 --window 101 \
                                                                                 --pseudo_count 1 --out_dir . --out_name ecoli_chrM_print \
                                                                                 --chrom_size_file hg38.chrom.sizes
