############################# Human naked DNA bias correction #############################
#### workdir: /fs/home/jiluzhang/TF_grammar/PRINT/cfoot_atac_cfoot/human_naked_DNA
#### ACCESS_ATAC ####
python /fs/home/jiluzhang/TF_grammar/ACCESS-ATAC/bias_correction/correct_bias.py --bw_raw human_nakedDNA.bw --bw_bias ecoli_to_human_access_atac.bw \
                                                                                 --bed_file regions_test.bed --extend 0 --window 101 \
                                                                                 --pseudo_count 1 --out_dir . --out_name human_naked_dna_access_atac_corrected \
                                                                                 --chrom_size_file hg38.chrom.sizes
#### PRINT_noft ####
python /fs/home/jiluzhang/TF_grammar/ACCESS-ATAC/bias_correction/correct_bias.py --bw_raw human_nakedDNA.bw --bw_bias ecoli_to_human_print_noft.bw \
                                                                                 --bed_file regions_test.bed --extend 0 --window 101 \
                                                                                 --pseudo_count 1 --out_dir . --out_name human_naked_dna_print_noft_corrected \
                                                                                 --chrom_size_file hg38.chrom.sizes
#### PRINT_ft ####
#### fine tune PRINT
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
bias_data = pd.read_csv('obsBias_chrM_human_naked_DNA_cfoot.tsv', sep='\t')  # 7248

# shuffle the data
random.seed(0)
idx_lst = list(range(bias_data.shape[0]))
random.shuffle(idx_lst)
bias_data = bias_data.iloc[idx_lst]

# One-hot encoding of sequence contexts
seqs = bias_data.loc[:, "context"].values
with mp.Pool(10) as pool:
   onehot_seqs = list(tqdm.tqdm(pool.imap(onehot_encode, seqs), total=len(seqs)))
onehot_seqs = np.array(onehot_seqs)  # (7248, 101, 4)

# get bias and idx
target = bias_data.loc[:, "obsBias"].values

# Split the encoded sequences and target values into training (0), validation (1) and test (1)
training_data = onehot_seqs[:5000]
training_target = target[:5000]
val_data = onehot_seqs[5000:6248]
val_target = target[5000:6248]
test_data = onehot_seqs[6248:]
test_target = target[6248:]

# Model definition
model = load_model('/fs/home/jiluzhang/TF_grammar/PRINT/cfoot/ecoli_to_ecoli/Tn5_NN_model_epoch_15.h5') 
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
        model.save('Tn5_NN_model_human_naked_dna_chrM_print_ft.h5')

#### correct bias using ecoli_chrM fine-tune model (for chr21)
import os
import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
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

model = load_model('./Tn5_NN_model_human_naked_dna_chrM_print_ft.h5')

chunk_size = math.ceil(seq_data.shape[0]/1000000)
for i in tqdm.tqdm(range(chunk_size), ncols=80):
    seqs = seq_data.loc[:, "context"].values[(i*1000000):(i*1000000+1000000)]
    with mp.Pool(5) as pool:
        onehot_seqs = np.array(pool.map(onehot_encode, seqs))   # onehot_seqs = list(tqdm.tqdm(pool.imap(onehot_encode, seqs), total=len(seqs)))
    if i==0:
        pred = np.transpose(model.predict(onehot_seqs, batch_size=16))[0]
    else:
        pred = np.concatenate([pred, np.transpose(model.predict(onehot_seqs, batch_size=128))[0]])

bw = pyBigWig.open("ecoli_to_human_print_ft.bw", "w")
chroms = {"chr21": 46709983}
bw.addHeader(list(chroms.items()))
pred_trans = pred.astype(np.float64)  # avoid the bug
bw.addEntries(chroms=['chr21']*(seq_data.shape[0]),
              starts=list(seq_data['start'].values),
              ends=list(seq_data['start'].values+1),
              values=list(pred_trans))
bw.close()

## correct bias
python /fs/home/jiluzhang/TF_grammar/ACCESS-ATAC/bias_correction/correct_bias.py --bw_raw human_nakedDNA.bw --bw_bias ecoli_to_human_print_ft.bw \
                                                                                 --bed_file regions_test.bed --extend 0 --window 101 \
                                                                                 --pseudo_count 1 --out_dir . --out_name human_naked_dna_print_ft_corrected \
                                                                                 --chrom_size_file hg38.chrom.sizes


############################# atac-cfoot bias correction #############################
#### workdir: /fs/home/jiluzhang/TF_grammar/PRINT/cfoot_atac_cfoot/atac_cfoot
# scp -P 10022 u21509@logini.tongji.edu.cn:/share/home/u21509/workspace/wuang/04.tf_grammer/intermediate_process/01.bias_correct/HepG2_ATAC-cFOOT/HepG2_Tn5100_SsdA5_ATAC.bw ATAC_cFOOT.bw
#### ACCESS_ATAC ####
python /fs/home/jiluzhang/TF_grammar/ACCESS-ATAC/bias_correction/correct_bias.py --bw_raw ATAC_cFOOT.bw --bw_bias ecoli_to_human_access_atac.bw \
                                                                                 --bed_file regions_test.bed --extend 0 --window 101 \
                                                                                 --pseudo_count 1 --out_dir . --out_name atac_cfoot_access_atac_corrected \
                                                                                 --chrom_size_file hg38.chrom.sizes
#### PRINT_noft ####
python /fs/home/jiluzhang/TF_grammar/ACCESS-ATAC/bias_correction/correct_bias.py --bw_raw ATAC_cFOOT.bw --bw_bias ecoli_to_human_print_noft.bw \
                                                                                 --bed_file regions_test.bed --extend 0 --window 101 \
                                                                                 --pseudo_count 1 --out_dir . --out_name atac_cfoot_print_noft_corrected \
                                                                                 --chrom_size_file hg38.chrom.sizes
#### PRINT_ft ####
#### fine tune PRINT
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

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
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
bias_data = pd.read_csv('obsBias_chrM_HepG2_atac_cfoot.tsv', sep='\t')  # 7252

# shuffle the data
random.seed(0)
idx_lst = list(range(bias_data.shape[0]))
random.shuffle(idx_lst)
bias_data = bias_data.iloc[idx_lst]

# One-hot encoding of sequence contexts
seqs = bias_data.loc[:, "context"].values
with mp.Pool(10) as pool:
   onehot_seqs = list(tqdm.tqdm(pool.imap(onehot_encode, seqs), total=len(seqs)))
onehot_seqs = np.array(onehot_seqs)  # (7252, 101, 4)

# get bias and idx
target = bias_data.loc[:, "obsBias"].values

# Split the encoded sequences and target values into training (0), validation (1) and test (1)
training_data = onehot_seqs[:5000]
training_target = target[:5000]
val_data = onehot_seqs[5000:6252]
val_target = target[5000:6252]
test_data = onehot_seqs[6252:]
test_target = target[6252:]

# Model definition
model = load_model('/fs/home/jiluzhang/TF_grammar/PRINT/cfoot/ecoli_to_ecoli/Tn5_NN_model_epoch_15.h5') 
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
        model.save('Tn5_NN_model_atac_cfoot_chrM_print_ft.h5')

#### correct bias using ecoli_chrM fine-tune model (for chr21)
import os
import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
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

model = load_model('./Tn5_NN_model_atac_cfoot_chrM_print_ft.h5')

chunk_size = math.ceil(seq_data.shape[0]/1000000)
for i in tqdm.tqdm(range(chunk_size), ncols=80):
    seqs = seq_data.loc[:, "context"].values[(i*1000000):(i*1000000+1000000)]
    with mp.Pool(5) as pool:
        onehot_seqs = np.array(pool.map(onehot_encode, seqs))   # onehot_seqs = list(tqdm.tqdm(pool.imap(onehot_encode, seqs), total=len(seqs)))
    if i==0:
        pred = np.transpose(model.predict(onehot_seqs, batch_size=16))[0]
    else:
        pred = np.concatenate([pred, np.transpose(model.predict(onehot_seqs, batch_size=128))[0]])

bw = pyBigWig.open("ecoli_to_human_print_ft.bw", "w")
chroms = {"chr21": 46709983}
bw.addHeader(list(chroms.items()))
pred_trans = pred.astype(np.float64)  # avoid the bug
bw.addEntries(chroms=['chr21']*(seq_data.shape[0]),
              starts=list(seq_data['start'].values),
              ends=list(seq_data['start'].values+1),
              values=list(pred_trans))
bw.close()

## correct bias
python /fs/home/jiluzhang/TF_grammar/ACCESS-ATAC/bias_correction/correct_bias.py --bw_raw ATAC_cFOOT.bw --bw_bias ecoli_to_human_print_ft.bw \
                                                                                 --bed_file regions_test.bed --extend 0 --window 101 \
                                                                                 --pseudo_count 1 --out_dir . --out_name atac_cfoot_print_ft_corrected \
                                                                                 --chrom_size_file hg38.chrom.sizes




############################# cfoot-atac bias correction #############################
#### workdir: /fs/home/jiluzhang/TF_grammar/PRINT/cfoot_atac_cfoot/cfoot_atac
# scp -P 10022 u21509@logini.tongji.edu.cn:/share/home/u21509/workspace/wuang/04.tf_grammer/intermediate_process/01.bias_correct/HepG2_cFOOT-ATAC/05.conversionProfile/ordinary/cFOOT_ATAC.bw .
#### ACCESS_ATAC ####
python /fs/home/jiluzhang/TF_grammar/ACCESS-ATAC/bias_correction/correct_bias.py --bw_raw cFOOT_ATAC.bw --bw_bias ecoli_to_human_access_atac.bw \
                                                                                 --bed_file regions_test.bed --extend 0 --window 101 \
                                                                                 --pseudo_count 1 --out_dir . --out_name cfoot_atac_access_atac_corrected \
                                                                                 --chrom_size_file hg38.chrom.sizes
#### PRINT_noft ####
python /fs/home/jiluzhang/TF_grammar/ACCESS-ATAC/bias_correction/correct_bias.py --bw_raw cFOOT_ATAC.bw --bw_bias ecoli_to_human_print_noft.bw \
                                                                                 --bed_file regions_test.bed --extend 0 --window 101 \
                                                                                 --pseudo_count 1 --out_dir . --out_name cfoot_atac_print_noft_corrected \
                                                                                 --chrom_size_file hg38.chrom.sizes
#### PRINT_ft ####
#### fine tune PRINT
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

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
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
bias_data = pd.read_csv('obsBias_chrM_HepG2_cfoot_atac.tsv', sep='\t')  # 7251

# shuffle the data
random.seed(0)
idx_lst = list(range(bias_data.shape[0]))
random.shuffle(idx_lst)
bias_data = bias_data.iloc[idx_lst]

# One-hot encoding of sequence contexts
seqs = bias_data.loc[:, "context"].values
with mp.Pool(10) as pool:
   onehot_seqs = list(tqdm.tqdm(pool.imap(onehot_encode, seqs), total=len(seqs)))
onehot_seqs = np.array(onehot_seqs)  # (7251, 101, 4)

# get bias and idx
target = bias_data.loc[:, "obsBias"].values

# Split the encoded sequences and target values into training (0), validation (1) and test (1)
training_data = onehot_seqs[:5000]
training_target = target[:5000]
val_data = onehot_seqs[5000:6251]
val_target = target[5000:6251]
test_data = onehot_seqs[6251:]
test_target = target[6251:]

# Model definition
model = load_model('/fs/home/jiluzhang/TF_grammar/PRINT/cfoot/ecoli_to_ecoli/Tn5_NN_model_epoch_15.h5') 
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
        model.save('Tn5_NN_model_cfoot_atac_chrM_print_ft.h5')

#### correct bias using ecoli_chrM fine-tune model (for chr21)
import os
import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
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

model = load_model('./Tn5_NN_model_cfoot_atac_chrM_print_ft.h5')

chunk_size = math.ceil(seq_data.shape[0]/1000000)
for i in tqdm.tqdm(range(chunk_size), ncols=80):
    seqs = seq_data.loc[:, "context"].values[(i*1000000):(i*1000000+1000000)]
    with mp.Pool(5) as pool:
        onehot_seqs = np.array(pool.map(onehot_encode, seqs))   # onehot_seqs = list(tqdm.tqdm(pool.imap(onehot_encode, seqs), total=len(seqs)))
    if i==0:
        pred = np.transpose(model.predict(onehot_seqs, batch_size=16))[0]
    else:
        pred = np.concatenate([pred, np.transpose(model.predict(onehot_seqs, batch_size=128))[0]])

bw = pyBigWig.open("ecoli_to_human_print_ft.bw", "w")
chroms = {"chr21": 46709983}
bw.addHeader(list(chroms.items()))
pred_trans = pred.astype(np.float64)  # avoid the bug
bw.addEntries(chroms=['chr21']*(seq_data.shape[0]),
              starts=list(seq_data['start'].values),
              ends=list(seq_data['start'].values+1),
              values=list(pred_trans))
bw.close()

## correct bias
python /fs/home/jiluzhang/TF_grammar/ACCESS-ATAC/bias_correction/correct_bias.py --bw_raw cFOOT_ATAC.bw --bw_bias ecoli_to_human_print_ft.bw \
                                                                                 --bed_file regions_test.bed --extend 0 --window 101 \
                                                                                 --pseudo_count 1 --out_dir . --out_name cfoot_atac_print_ft_corrected \
                                                                                 --chrom_size_file hg38.chrom.sizes
