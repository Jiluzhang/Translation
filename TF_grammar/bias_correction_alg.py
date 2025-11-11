#### predict bias using PRINT
import os
import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = '2'
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

model = load_model('/fs/home/jiluzhang/TF_grammar/PRINT/cfoot/ecoli_to_ecoli/Tn5_NN_model_epoch_15.h5')

# os.makedirs('chunked_files', exist_ok=True)

chunk_size = math.ceil(seq_data.shape[0]/10000)
bw = pyBigWig.open('pred_chr21.bw', 'w')
chroms = {"chr21": 46709983}
bw.addHeader(list(chroms.items()))

for i in tqdm.tqdm(range(chunk_size), ncols=80):
    seqs = seq_data.loc[:, "context"].values[(i*10000):(i*10000+10000)]
    with mp.Pool(10) as pool:
        onehot_seqs = np.array(pool.map(onehot_encode, seqs))   # onehot_seqs = list(tqdm.tqdm(pool.imap(onehot_encode, seqs), total=len(seqs)))
    pred = model.predict(onehot_seqs, batch_size=64, verbose=0).flatten()  #pred = np.transpose(model.predict(onehot_seqs, batch_size=64, verbose=0))[0]

    pred_trans = pred.astype(np.float64)  # avoid the bug
    bw.addEntries(chroms=['chr21']*(seqs.shape[0]),
                  starts=list(seq_data['start'][(i*10000):(i*10000+10000)].values),
                  ends=list(seq_data['start'][(i*10000):(i*10000+10000)].values+1),
                  values=list(pred_trans))

bw.close()


#### merge bw file for different chromosomes
all_chroms = {}
all_data = {}

for i in tqdm.tqdm(range(21, 23), ncols=80):
    bw = pyBigWig.open('pred_chr'+str(i)+'.bw')
    
    chroms = bw.chroms()
    all_chroms.update(chroms)
    chrom = list(chroms.keys())[0]
    length = chroms[chrom]
    intervals = bw.intervals(chrom)
    all_data[chrom] = intervals
    
    bw.close()

bw_out = pyBigWig.open('pred.bw', "w")
bw_out.addHeader(list(all_chroms.items()))

for chrom, intervals in all_data.items():
    starts = [interval[0] for interval in intervals]
    ends = [interval[1] for interval in intervals]
    values = [interval[2] for interval in intervals]
    bw_out.addEntries([chrom]*len(starts), starts, ends=ends, values=values)

bw_out.close()






