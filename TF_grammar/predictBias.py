############################ convert bigwig to csv ############################
## python bw2csv.py  # ~4 min

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
res.to_csv('obsBias_cfoot.tsv', sep='\t', index=False)
########################################################################################


######################################################## train CNN model ########################################################
## python train.py
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

##############################################
# functions used #
##############################################

# One-hot encoding of a DNA sequence.
def onehot_encode(seq):
    mapping = pd.Series(index=["A", "C", "G", "T"], data=[0, 1, 2, 3])
    bases = [base for base in seq]
    base_inds = mapping[bases]
    onehot = np.zeros((len(bases), 4))
    onehot[np.arange(len(bases)), base_inds] = 1
    return onehot

# Running one-hot encoding along the sequence of a genomic region
def region_onehot_encode(region_seq, context_radius=50):
    
    # Calculate length of every local sub-sequence
    context_len = 2 * context_radius + 1
    
    # If region width is L, then region_seq should be a string of length L + 2 * context_len
    region_width = len(region_seq) - 2 * context_radius
    
    if "N" in region_seq:
        return np.zeros((region_width, 4))
    else:
        # First encode the whole region sequence 
        # This prevents repetitive computing for overlapping sub-sequences
        region_onehot = np.array(onehot_encode(region_seq))
        
        # Retrieve encoded sub-sequences by subsetting the larger encoded matrix
        region_onehot = np.array([region_onehot[i : (i + context_len), :] for i in range(region_width)])
        
        return region_onehot
    
##############################################
# Prepare training, validation and test data #
##############################################

# Load ground truth Tn5 bias data
print("Loading ground truth data")
bias_data = pd.read_csv('obsBias_cfoot.tsv', sep='\t')  # 'context' & 'obsBias' & 'idx' & 'chrom' & 'start' & 'end'

# One-hot encoding of sequence contexts
print("One-hot encoding of sequence contexts")
seqs = bias_data.loc[:, "context"].values
with mp.Pool(10) as pool:
   onehot_seqs = list(tqdm.tqdm(pool.imap(onehot_encode, seqs), total=len(seqs)))  # ~2 min
onehot_seqs = np.array(onehot_seqs)  # (2289693, 101, 4)

# Get target values
target = bias_data.loc[:, "obsBias"].values

# Get the indices of all regions
idx = bias_data.loc[:, "idx"].values

# Split the encoded sequences and target values into training (0), validation (1) and test (1)
training_data = onehot_seqs[idx==0]
training_target = target[idx==0]
val_data = onehot_seqs[idx==1]
val_target = target[idx==1]
test_data = onehot_seqs[idx==2]
test_target = target[idx==2]


#################################
# Model definition and training #
#################################

# Model definition
print("Training Tn5 bias model")
inputs = Input(shape = (np.shape(test_data[0])))
conv_1 = Conv1D(32, 5, padding = 'same', activation = 'relu', strides = 1)(inputs)
maxpool_1 = MaxPooling1D()(conv_1)
conv_2 = Conv1D(32, 5, padding = 'same', activation = 'relu', strides = 1)(maxpool_1)
maxpool_2 = MaxPooling1D()(conv_2)
conv_3 = Conv1D(32, 5, padding = 'same', activation = 'relu', strides = 1)(maxpool_2)
maxpool_3 = MaxPooling1D()(conv_3)
flat = Flatten()(maxpool_3)
fc = Dense(32, activation = "relu")(flat)
out = Dense(1, activation = "linear")(fc)
model = Model(inputs=inputs, outputs=out)  
model.summary()
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
        print('save model')
        model.save('Tn5_NN_model.h5')
#####################################################################################################################


######################################################## CNN model evaluation ########################################################
## python test.py

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

##############################################
# functions used #
##############################################

# One-hot encoding of a DNA sequence.
def onehot_encode(seq):
    mapping = pd.Series(index=["A", "C", "G", "T"], data=[0, 1, 2, 3])
    bases = [base for base in seq]
    base_inds = mapping[bases]
    onehot = np.zeros((len(bases), 4))
    onehot[np.arange(len(bases)), base_inds] = 1
    return onehot

# Running one-hot encoding along the sequence of a genomic region
def region_onehot_encode(region_seq, context_radius=50):
    
    # Calculate length of every local sub-sequence
    context_len = 2 * context_radius + 1
    
    # If region width is L, then region_seq should be a string of length L + 2 * context_len
    region_width = len(region_seq) - 2 * context_radius
    
    if "N" in region_seq:
        return np.zeros((region_width, 4))
    else:
        # First encode the whole region sequence 
        # This prevents repetitive computing for overlapping sub-sequences
        region_onehot = np.array(onehot_encode(region_seq))
        
        # Retrieve encoded sub-sequences by subsetting the larger encoded matrix
        region_onehot = np.array([region_onehot[i : (i + context_len), :] for i in range(region_width)])
        
        return region_onehot

bias_data = pd.read_csv('obsBias_cfoot.tsv', sep='\t')
bias_test = bias_data[bias_data['idx']==2]

print("One-hot encoding of sequence contexts")
seqs = bias_test.loc[:, "context"].values
with mp.Pool(10) as pool:
   onehot_seqs = list(tqdm.tqdm(pool.imap(onehot_encode, seqs), total=len(seqs)))
onehot_seqs = np.array(onehot_seqs)  # (249326, 101, 4)

test_target = bias_test.loc[:, "obsBias"].values
test_data = onehot_seqs

# Load last Tn5 bias model, which should be the currently best model
model = load_model('Tn5_NN_model.h5')

# Model evaluation on the test set
print("Evaluating performance on the test set")
plt_ind = np.random.choice(np.arange(len(test_data)), 1000)
test_pred = np.transpose(model.predict(test_data))[0]

# Plot test results
plt.figure(figsize=(5, 5), dpi=100)
plt.scatter(test_target[plt_ind], test_pred[plt_ind], s=1)
plt.xlim(0-0.01, 0.8+0.01)
plt.ylim(0-0.01, 0.8+0.01)
plt.xlabel("Target label")
plt.ylabel("Target prediction")
plt.title("Pearson correlation = " + str(ss.pearsonr(test_target, test_pred)[0]))
plt.savefig('testing_correlation.pdf')
print('Plot correlation plot done')

## extract regions from bigwig files
bw = pyBigWig.open("regions_test_pred.bw", "w")
chroms = {"NC_000913.3": 4641652}
bw.addHeader(list(chroms.items()))
test_pred_trans = test_pred.astype(np.float64)
bw.addEntries(chroms=['NC_000913.3']*(bias_test.shape[0]),
              starts=list(bias_test['start'].values),
              ends=list(bias_test['end'].values),
              values=list(test_pred_trans))
bw.close()
print('Output bigwig file done')
#######################################################################################################################################


#### !!!!!!!!!!!!! prediction 0.1  &  install deeptools on PRINT env !!!!!!!!!!!!!


multiBigwigSummary bins -b regions_test_pred.bw regions_test.bw -o test.npz --outRawCounts test.tab -l pred raw -bs 1 -p 20
grep -v nan test.tab | sed 1d > test_nonan.tab
rm test.tab
/fs/home/jiluzhang/TF_grammar/cnn_bias_model/data/cal_cor --file test_nonan.tab




###############################################
# Use the model to predict Tn5 bias for regions #
###############################################
    
# Load Tn5 bias model
model = load_model(main_dir + "/Tn5_NN_model.h5")

# Load sequences of regions
print("Loading sequences of regions")
region_seqs = pd.read_csv(data_dir + "/regionSeqs.txt", header = None)  # sequences with same length
region_seqs = np.transpose(region_seqs.values)[0]

# Specify the radius of sequence context and the width of the region
context_len = 2 * context_radius + 1
region_width = len(region_seqs[0]) - 2 * context_radius

# To reduce memory usage, we chunk the region list in to smaller chunks
starts = np.arange(len(region_seqs), step = chunk_size)
if len(starts) > 1:
    starts = starts[:(len(starts) - 1)]
    ends = starts + chunk_size
    ends[len(ends) - 1] = len(region_seqs) + 1
else:
    starts = [0]
    ends = [len(region_seqs) + 1]

# Create folder for storing intermediate results for each chunk
os.system("mkdir " + data_dir + "/chunked_bias_pred")

# Go through all chunks and predict Tn5 bias
print("Predicting Tn5 bias for regions")
for i in tqdm.tqdm(range(len(starts))):

    # if os.path.exists(data_dir + "/chunked_bias_pred/chunk_" + str(i) + ".pkl"):
    #     continue

    print("Processing chunk No." + str(i) + " " + datetime.now().strftime("%H:%M:%S"))
    chunk_seqs = region_seqs[starts[i]:ends[i]]
    
    # Encode sequences in the current chunk into matrices using one-hot encoding
    print("Encoding sequence contexts")
    with mp.Pool(2) as pool:
       chunk_onehot = list(tqdm.tqdm(pool.imap(region_onehot_encode, chunk_seqs), total = len(chunk_seqs)))
    
    # Use neural network to predict bias
    # For sequences containing "N", the encoded matrix will be a empty zero matrix,
    # in such cases we assign 1s as bias values
    print("Predicting bias")
    pred_bias = np.array([np.transpose(model.predict(chunk_onehot[j]))[0] \
                          if (np.sum(chunk_onehot[j]) > 1) else np.ones(region_width) \
                          for j in tqdm.tqdm(range(len(chunk_onehot)), total = chunk_size)])
    
    # Reverse transform the predicted values to the original scale
    pred_bias = np.power(10, (pred_bias - 0.5) * 2) - 0.01

    # Save intermediate result
    with open(data_dir + "/chunked_bias_pred/chunk_" + str(i) + ".pkl", "wb") as f:
        pickle.dump(pred_bias, f)

# Integrate results from all chunks
pred_bias_all = None
for i in tqdm.tqdm(range(len(starts))):

    # Load intermediate result
    with open(data_dir + "/chunked_bias_pred/chunk_" + str(i) + ".pkl", "rb") as f:
        pred_bias = pickle.load(f)

    # Reverse transform the predicted values to the original scale
    if pred_bias_all is None:
        pred_bias_all = pred_bias
    else:
        pred_bias_all = np.concatenate([pred_bias_all, pred_bias])

# Write results of the current batch to file
output_path = data_dir + "/predBias.h5"
hf = h5py.File(output_path, 'w')
hf.create_dataset("predBias", data = pred_bias_all)
hf.close()

# Remove intermediate files
os.system("rm -r " + data_dir + "/chunked_bias_pred")















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







