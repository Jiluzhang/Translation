## workdir: /fs/home/jiluzhang/TF_grammar/PRINT/cfoot_atac_cfoot/pipeline

#### predict bias using PRINT
## python predict_bias.py
import os
import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = '2'
tf.config.set_logical_device_configuration(tf.config.experimental.list_physical_devices('GPU')[0], 
                                           [tf.config.LogicalDeviceConfiguration(memory_limit=40960)])

import pyfaidx
from keras.models import load_model
import numpy as np
import pandas as pd
import multiprocessing as mp
from tqdm import tqdm
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

genome = pyfaidx.Fasta("hg38.fa")
chrom_sizes = pd.read_table('hg38.chrom.sizes', header=None)
model = load_model('Tn5_NN_model_epoch_15.h5')

for chrom in ['chr'+str(i) for i in list(range(1, 23))]:
    ## extract seqs
    context_chrom = genome[chrom][:].seq.upper()  # read seqs into memory to speed up
    chrom_len = chrom_sizes[chrom_sizes[0]==chrom][1].item()
    context_lst = []
    start_lst = []
    for i in tqdm(range(50, chrom_len-50), ncols=100, desc='extract seqs '+chrom):
        context = context_chrom[(i-50):(i+50+1)]
        if (context[50] in ['G', 'C']) and (not 'N' in context):  # select G/C & delete N
            context_lst.append(context)
            start_lst.append(i)
    
    ## predict bias
    chunk_size = math.ceil(len(context_lst)/10000)
    bw = pyBigWig.open('pred_'+chrom+'.bw', 'w')
    bw.addHeader([(chrom, chrom_len)])
    with mp.Pool(20) as pool:
        for j in tqdm(range(chunk_size), ncols=100, desc='predict bias '+chrom):
            seqs = context_lst[(j*10000):(j*10000+10000)]
            onehot_seqs = np.array(pool.map(onehot_encode, seqs))
            
            pred = model.predict(onehot_seqs, batch_size=64, verbose=0).flatten()
            
            pred_trans = pred.astype(np.float64)  # avoid the bug
            bw.addEntries(chroms=[chrom]*len(seqs),
                          starts=start_lst[(j*10000):(j*10000+10000)],
                          ends=[start+1 for start in start_lst[(j*10000):(j*10000+10000)]],
                          values=list(pred_trans))
        
        bw.close()


#### merge bw file for different chromosomes
from tqdm import tqdm

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






