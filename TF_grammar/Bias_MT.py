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


# #### get chrM bias
# ## training: 1-14569
# ## validation: 14570-15569
# ## testing: 15569-16569

# import pyfaidx
# import pyBigWig
# import pandas as pd
# import math
# from tqdm import * 

# genome = pyfaidx.Fasta("hg38.fa")
# bw = pyBigWig.open('HepG2_7.5U_conversion.bw') # bw = pyBigWig.open('K562_ATAC_cFOOT_3w_cell_100Tn5_10U_methylation.bw')

# context_lst = []
# obsBias_lst = []
# idx_lst = []
# start_lst = []

# for i in tqdm(range(50, len(genome['chrM'][:].seq)-50), ncols=80):
#     seq = genome['chrM'][(i-50):(i+50+1)].seq.upper()
#     signal = bw.values('chrM', i, i+1)[0]
#     if (not math.isnan(signal)) and (signal!=0) and ('N' not in seq):
#         context_lst.append(seq)
#         obsBias_lst.append(signal)
#         ## 0:training  1:validation  2:testing
#         if i<=14569:
#             idx_lst.append(0)
#         elif i<=15569:
#             idx_lst.append(1)
#         else:
#             idx_lst.append(2)
#         start_lst.append(i)

# bw.close()

# res = pd.DataFrame({'context':context_lst, 'obsBias':obsBias_lst, 'idx':idx_lst,
#                     'chrom':'chrM', 'start':start_lst})
# res['end'] = res['start']+1
# res.to_csv('obsBias_chrM.tsv', sep='\t', index=False)  # 7245
