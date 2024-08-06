## cell line dataset
## reference: hg19 !!!!!!!!!!!!!!!!!!

# wget -c https://static-content.springer.com/esm/art%3A10.1038%2Fs41467-018-08205-7/MediaObjects/41467_2018_8205_MOESM6_ESM.xls   # ATAC
# wget -c https://static-content.springer.com/esm/art%3A10.1038%2Fs41467-018-08205-7/MediaObjects/41467_2018_8205_MOESM7_ESM.xls   # RNA

# wget -c https://hgdownload.soe.ucsc.edu/admin/exe/linux.x86_64/liftOver
# wget -c https://hgdownload.soe.ucsc.edu/goldenPath/hg19/liftOver/hg19ToHg38.over.chain.gz

# awk '{print $1}' 41467_2018_8205_MOESM6_ESM.xls | sed '1d' | sed 's/_/\t/g' | awk '{print $0 "\t" "peak_" NR}' > peaks_hg19.bed
# liftOver peaks_hg19.bed hg19ToHg38.over.chain.gz peaks_hg38.bed unmapped.bed

## hg19 -> hg38 for all peak sets
import pandas as pd

dat = pd.read_table('41467_2018_8205_MOESM6_ESM.xls')
dat['idx'] = ['peak_'+str(i) for i in range(1, dat.shape[0]+1)]

peaks_hg38 = pd.read_table('peaks_hg38.bed', header=None, names=['chr', 'start', 'end', 'idx'])
peaks_hg38['start'] = peaks_hg38['start'].astype('str')
peaks_hg38['end'] = peaks_hg38['end'].astype('str')
peaks_hg38['pos'] = peaks_hg38['chr']+'_'+peaks_hg38['start']+'_'+peaks_hg38['end']
peaks_hg38.drop(['chr', 'start', 'end'], axis=1, inplace=True)

out = pd.merge(peaks_hg38, dat, how='left', on='idx')
out.index = out['pos'].values
out.drop(['idx', 'pos'], axis=1, inplace=True)
out.to_csv('41467_2018_8205_MOESM6_ESM_hg38_raw.xls', sep='\t')


# grep -v GL 41467_2018_8205_MOESM6_ESM_hg38_raw.xls | grep -v alt | grep -v KI > 41467_2018_8205_MOESM6_ESM_hg38.xls


############ xls to h5ad ############
import scanpy as sc
import pandas as pd
from scipy.sparse import csr_matrix
import pybedtools
import numpy as np
from tqdm import tqdm

## rna
rna_dat = pd.read_table('41467_2018_8205_MOESM7_ESM.xls')  # 49059 x 549
rna = sc.AnnData(rna_dat.values.T, var=pd.DataFrame({'gene_ids': rna_dat.index.values, 'feature_types': 'Gene Expression'}))
rna.obs.index = rna_dat.columns.values
rna.var.index = rna.var['gene_ids']
rna.obs['cell_anno'] = rna.obs.index.map(lambda x: x.split('_')[1])

genes = pd.read_table('human_genes.txt', names=['gene_ids', 'gene_name'])

rna_exp = pd.concat([rna.var, pd.DataFrame(rna.X.T, index=rna.var.index.values)], axis=1)
X_new = pd.merge(genes, rna_exp, how='left', on='gene_ids').iloc[:, 3:].T
X_new.fillna(value=0, inplace=True)

rna_new = sc.AnnData(X_new.values, obs=rna.obs, var=pd.DataFrame({'gene_ids': genes['gene_ids'], 'feature_types': 'Gene Expression'}))
rna_new.var.index = genes['gene_name'].values
rna_new.X = csr_matrix(rna_new.X)
rna_new.write('rna.h5ad')

## atac
atac_dat = pd.read_table('41467_2018_8205_MOESM6_ESM_hg38.xls', index_col=0)  # 157358 x 549
atac = sc.AnnData(atac_dat.values.T, var=pd.DataFrame({'gene_ids': atac_dat.index.values, 'feature_types': 'Peaks'}))
atac.obs.index = atac_dat.columns.values
atac.var['gene_ids'] = atac.var['gene_ids'].map(lambda x: x.replace('_', ':', 1).replace('_', '-'))
atac.var.index = atac.var['gene_ids'].values
atac.obs['cell_anno'] = atac.obs.index.map(lambda x: x.split('_')[1])
atac.X[atac.X>0] = 1  # binarization

peaks = pd.DataFrame({'id': atac.var_names})
peaks['chr'] = peaks['id'].map(lambda x: x.split(':')[0])
peaks['start'] = peaks['id'].map(lambda x: x.split(':')[1].split('-')[0])
peaks['end'] = peaks['id'].map(lambda x: x.split(':')[1].split('-')[1])
peaks.drop(columns='id', inplace=True)
peaks['idx'] = range(peaks.shape[0])

cCREs = pd.read_table('human_cCREs.bed', names=['chr', 'start', 'end'])
cCREs['idx'] = range(cCREs.shape[0])
cCREs_bed = pybedtools.BedTool.from_dataframe(cCREs)
peaks_bed = pybedtools.BedTool.from_dataframe(peaks)
idx_map = peaks_bed.intersect(cCREs_bed, wa=True, wb=True).to_dataframe().iloc[:, [3, 7]]
idx_map.columns = ['peaks_idx', 'cCREs_idx']

m = np.zeros([atac.n_obs, cCREs_bed.to_dataframe().shape[0]], dtype='float32')
for i in tqdm(range(atac.X.shape[0]), ncols=80, desc='Aligning ATAC peaks'):
    m[i][idx_map[idx_map['peaks_idx'].isin(peaks.iloc[np.nonzero(atac.X[i])]['idx'])]['cCREs_idx']] = 1

atac_new = sc.AnnData(m, obs=atac.obs, var=pd.DataFrame({'gene_ids': cCREs['chr']+':'+cCREs['start'].map(str)+'-'+cCREs['end'].map(str), 'feature_types': 'Peaks'}))
atac_new.var.index = atac_new.var['gene_ids'].values
atac_new.X = csr_matrix(atac_new.X)
atac_new.write('atac.h5ad')

########## filter cells ##########
#################### no filtering !!!! (for model performance evaluation using different data quality) ####################
# sc.pp.filter_genes(rna_new, min_cells=5)
# sc.pp.filter_cells(rna_new, min_genes=500)
# sc.pp.filter_cells(rna_new, max_genes=10000)

# sc.pp.filter_genes(atac_new, min_cells=5)
# sc.pp.filter_cells(atac_new, min_genes=1000)
# sc.pp.filter_cells(atac_new, max_genes=50000)

# idx = np.intersect1d(rna_new.obs.index, atac_new.obs.index)  
# rna_new[idx, :].copy().write('rna.h5ad')     # 460 * 21965
# atac_new[idx, :].copy().write('atac.h5ad')   # 460 * 72149


######## max enc length ########
import scanpy as sc
import numpy as np
rna = sc.read_h5ad('rna.h5ad')
np.count_nonzero(rna.X.toarray(), axis=1).max()
# 13703


python split_train_val_test.py --RNA rna.h5ad --ATAC atac.h5ad --train_pct 0.7 --valid_pct 0.1
python data_preprocess.py -r rna_train.h5ad -a atac_train.h5ad -s preprocessed_data_train --dt train --config rna2atac_config_train.yaml   # list_digits: 6 -> 7
python data_preprocess.py -r rna_val.h5ad -a atac_val.h5ad -s preprocessed_data_val --dt val --config rna2atac_config_val.yaml
python data_preprocess.py -r rna_test.h5ad -a atac_test.h5ad -s preprocessed_data_test --dt test --config rna2atac_config_test.yaml

nohup accelerate launch --config_file accelerator_config_train.yaml --main_process_port 29823 rna2atac_train.py --config_file rna2atac_config_train.yaml \
                        --train_data_dir ./preprocessed_data_train --val_data_dir ./preprocessed_data_val -n rna2atac_train > 20240623.log &    
# self.iConv_enc = nn.Embedding(10, dim//6)  -> 7
# accelerator.print('#'*80)

accelerate launch --config_file accelerator_config_test.yaml --main_process_port 29822 rna2atac_test.py \
                  -d ./preprocessed_data_test \
                  -l save/2024-06-23_rna2atac_train_104/pytorch_model.bin --config_file rna2atac_config_test.yaml

python npy2h5ad.py
mv rna2atac_scm2m_raw.h5ad ../benchmark/


#################### scButterfly-B ####################
python scbt_b.py


########### BABLE ###########
concat_train_val.py
python h5ad2h5.py -n train_val
python h5ad2h5.py -n test

python /fs/home/jiluzhang/BABEL/bin/train_model.py --data train_val.h5 --outdir babel_train_out --batchsize 512 --earlystop 25 --device 6 --nofilter  
python /fs/home/jiluzhang/BABEL/bin/predict_model.py --checkpoint babel_train_out --data test.h5 --outdir babel_test_out --device 4 --nofilter --noplot --transonly



########### Benchmark ###########
python csr2array.py --pred rna2atac_scm2m_raw.h5ad  --true rna2atac_true.h5ad
python csr2array.py --pred rna2atac_babel_raw.h5ad  --true rna2atac_true.h5ad
python csr2array.py --pred rna2atac_scbt_raw.h5ad  --true rna2atac_true.h5ad

import argparse
import scanpy as sc
import numpy as np

parser = argparse.ArgumentParser(description='Calculate clustering metrics and plot umap')
parser.add_argument('--pred', type=str, help='prediction')
parser.add_argument('--true', type=str, help='ground truth')
args = parser.parse_args()
pred_file = args.pred
true_file = args.true

alg = pred_file.split('_')[1]

pred = sc.read_h5ad(pred_file)
true = sc.read_h5ad(true_file)

if type(pred.X) is not np.ndarray:
    pred.X = pred.X.toarray()

pred[true.obs.index.values, :].write('rna2atac_'+alg+'.h5ad')


python csr2array.py --pred rna2atac_scm2m_raw.h5ad  --true rna2atac_true.h5ad
python cal_auroc_auprc.py --pred rna2atac_scm2m.h5ad --true rna2atac_true.h5ad
python cal_cluster_plot.py --pred rna2atac_scm2m.h5ad --true rna2atac_true.h5ad
# AMI: [0.5732, 0.8516, 0.7297, 0.8516, 0.8473]
# ARI: [0.4715, 0.8618, 0.6112, 0.8618, 0.8536]
# HOM: [0.4249, 0.8046, 0.6081, 0.8046, 0.7989]
# NMI: [0.5806, 0.858, 0.7384, 0.858, 0.8538]

python cal_cluster_plot.py --pred rna2atac_scbt.h5ad --true rna2atac_true.h5ad
# AMI: [0.8435, 0.8435, 0.8435, 0.8435, 0.8435]
# ARI: [0.7894, 0.7894, 0.7894, 0.7894, 0.7894]
# HOM: [0.7363, 0.7363, 0.7363, 0.7363, 0.7363]
# NMI: [0.8481, 0.8481, 0.8481, 0.8481, 0.8481]

python cal_cluster_plot.py --pred rna2atac_babel.h5ad --true rna2atac_true.h5ad
# AMI: [0.7441, 0.7332, 0.7531, 0.7531, 0.7531]
# ARI: [0.7033, 0.6735, 0.7014, 0.7014, 0.7014]
# HOM: [0.6518, 0.6416, 0.6589, 0.6589, 0.6589]
# NMI: [0.7516, 0.7411, 0.7604, 0.7604, 0.7604]


# python csr2array.py --pred rna2atac_scm2malpha_raw.h5ad  --true rna2atac_true.h5ad
# python cal_auroc_auprc.py --pred rna2atac_scm2malpha.h5ad --true rna2atac_true.h5ad
# python cal_cluster_plot.py --pred rna2atac_scm2malpha.h5ad --true rna2atac_true.h5ad

# python csr2array.py --pred rna2atac_scm2malphaftdiff_raw.h5ad  --true rna2atac_true.h5ad
# python cal_auroc_auprc.py --pred rna2atac_scm2malphaftdiff.h5ad --true rna2atac_true.h5ad
# python cal_cluster_plot.py --pred rna2atac_scm2malphaftdiff.h5ad --true rna2atac_true.h5ad



################# scM2M PBMC #################
######## max enc length ########
import scanpy as sc
import numpy as np
rna = sc.read_h5ad('rna.h5ad')
np.count_nonzero(rna.X.toarray(), axis=1).max()
# 9744

python split_train_val_test.py --RNA rna.h5ad --ATAC atac.h5ad --train_pct 0.7 --valid_pct 0.1
python data_preprocess.py -r rna_train.h5ad -a atac_train.h5ad -s preprocessed_data_train --dt train --config rna2atac_config_train.yaml
python data_preprocess.py -r rna_val.h5ad -a atac_val.h5ad -s preprocessed_data_val --dt val --config rna2atac_config_val.yaml
python data_preprocess.py -r rna_test.h5ad -a atac_test.h5ad -s preprocessed_data_test --dt test --config rna2atac_config_test.yaml
# 122344
nohup accelerate launch --config_file accelerator_config_train.yaml --main_process_port 29823 rna2atac_train.py --config_file rna2atac_config_train.yaml \
                        --train_data_dir ./preprocessed_data_train --val_data_dir ./preprocessed_data_val -n rna2atac_train > 20240623.log &    

accelerate launch --config_file accelerator_config_test.yaml --main_process_port 29822 rna2atac_test.py \
                  -d ./preprocessed_data_test \
                  -l save/2024-06-23_rna2atac_train_27/pytorch_model.bin --config_file rna2atac_config_test.yaml
# 155117
python npy2h5ad.py
mv rna2atac_scm2m_raw.h5ad ../benchmark/


#################### scButterfly-B ####################
nohup python scbt_b.py > 20240624.log &  # 194227




############# Pan-cancer #############
# ## CE336E1-S1
# cp /fs/home/jiluzhang/2023_nature_LD/normal_h5ad/CE336E1-S1_rna.h5ad rna.h5ad
# cp /fs/home/jiluzhang/2023_nature_LD/normal_h5ad/CE336E1-S1_atac.h5ad atac.h5ad

# python ../split_train_val.py --RNA rna.h5ad --ATAC atac.h5ad --train_pct 0.9
# python ../data_preprocess.py -r rna_train.h5ad -a atac_train.h5ad -s preprocessed_data_train --dt train --config ../rna2atac_config_train.yaml
# python ../data_preprocess.py -r rna_val.h5ad -a atac_val.h5ad -s preprocessed_data_val --dt val --config ../rna2atac_config_val.yaml
# accelerate launch --config_file ../accelerator_config_train.yaml --main_process_port 29823 ../rna2atac_train.py --config_file ../rna2atac_config_train.yaml \
#                   --train_data_dir ./preprocessed_data_train --val_data_dir ./preprocessed_data_val -n rna2atac_train


# ## CE348E1-S1K1
# cp /fs/home/jiluzhang/2023_nature_LD/normal_h5ad/CE348E1-S1K1_rna.h5ad rna.h5ad
# cp /fs/home/jiluzhang/2023_nature_LD/normal_h5ad/CE348E1-S1K1_atac.h5ad atac.h5ad

# accelerate launch --config_file ../accelerator_config_train.yaml --main_process_port 29823 ../rna2atac_train.py --config_file ../rna2atac_config_train.yaml \
#                   --train_data_dir ./preprocessed_data_train --val_data_dir ./preprocessed_data_val 
#                   -l /fs/home/jiluzhang/scM2M_no_dec_attn/pan_cancer/CE336E1-S1/save/2024-06-24_rna2atac_train_35/pytorch_model.bin -n rna2atac_train






cp /fs/home/jiluzhang/scM2M_v2/rna_32124_shuf.h5ad rna.h5ad
cp /fs/home/jiluzhang/scM2M_v2/atac_32124_shuf.h5ad atac.h5ad

python split_train_val.py --RNA rna.h5ad --ATAC atac.h5ad --train_pct 0.9
python data_preprocess.py -r rna_train.h5ad -a atac_train.h5ad -s preprocessed_data_train --dt train --config rna2atac_config_train.yaml  # ~30 min
python data_preprocess.py -r rna_val.h5ad -a atac_val.h5ad -s preprocessed_data_val --dt val --config rna2atac_config_val.yaml
accelerate launch --config_file accelerator_config_train.yaml --main_process_port 29823 rna2atac_train.py --config_file rna2atac_config_train.yaml \
                  --train_data_dir ./preprocessed_data_train --val_data_dir ./preprocessed_data_val -n rna2atac_train



accelerate launch --config_file accelerator_config_train.yaml --main_process_port 29822 rna2atac_train.py --config_file rna2atac_config_train.yaml \
                  --train_data_dir ./preprocessed_data_train --val_data_dir ./preprocessed_data_val -n rna2atac_train -s save_ft \
                  -l /fs/home/jiluzhang/scM2M_no_dec_attn/pan_cancer/save_01/2024-06-24_rna2atac_train_7/pytorch_model.bin

accelerate launch --config_file accelerator_config_test.yaml --main_process_port 29822 rna2atac_test.py \
                  -d ./preprocessed_data_test \
                  -l save_ft/2024-06-24_rna2atac_train_57/pytorch_model.bin \
                  --config_file rna2atac_config_test.yaml

accelerate launch --config_file accelerator_config_test.yaml --main_process_port 29822 rna2atac_test.py \
                  -d ./preprocessed_data_test \
                  -l /fs/home/jiluzhang/scM2M_no_dec_attn/pan_cancer/save/2024-06-24_rna2atac_train_27/pytorch_model.bin \
                  --config_file rna2atac_config_test.yaml


python data_preprocess.py -r rna_val.h5ad -a atac_val.h5ad -s preprocessed_data_test --dt test --config rna2atac_config_test.yaml

accelerate launch --config_file accelerator_config_test.yaml --main_process_port 29822 rna2atac_test.py \
                  -d ./preprocessed_data_test \
                  -l /fs/home/jiluzhang/scM2M_no_dec_attn/pan_cancer/save/2024-06-24_rna2atac_train_27/pytorch_model.bin \
                  --config_file rna2atac_config_test.yaml



# import snapatac2 as snap
# import numpy as np
# from scipy.sparse import csr_matrix
# import scanpy as sc

# true = snap.read('atac_val_100.h5ad', backed=None)
# snap.pp.select_features(true)
# snap.tl.spectral(true)
# snap.tl.umap(true)
# snap.pp.knn(true)
# snap.tl.leiden(true)
# sc.pl.umap(true, color='leiden', legend_fontsize='7', legend_loc='right margin', size=5,
#            title='', frameon=True, save='_atac_100_true.pdf')

# m_raw = np.load('predict.npy')
# m = m_raw.copy()
# m[m>0.5]=1
# m[m<=0.5]=0

# pred = true.copy()
# pred.X = csr_matrix(m)

# snap.pp.select_features(pred)
# snap.tl.spectral(pred)
# snap.tl.umap(pred)
# sc.pl.umap(pred, color='leiden', legend_fontsize='7', legend_loc='right margin', size=5,
#            title='', frameon=True, save='_atac_100_predict.pdf')



python npy2h5ad.py
mv rna2atac_scm2m_raw.h5ad ../benchmark/rna2atac_scm2mpancancer_raw.h5ad
python csr2array.py --pred rna2atac_scm2mpancancer_raw.h5ad  --true rna2atac_true.h5ad
python cal_auroc_auprc.py --pred rna2atac_scm2mpancancer.h5ad --true rna2atac_true.h5ad
python cal_cluster_plot.py --pred rna2atac_scm2mpancancer.h5ad --true rna2atac_true.h5ad



## combine samples
import pandas as pd
import scanpy as sc
import anndata as ad
import numpy as np

samples = pd.read_table('files_all.txt', header=None)  # 129
rna_0 = sc.read_h5ad('/fs/home/jiluzhang/2023_nature_LD/normal_h5ad/'+samples[0][0]+'_rna.h5ad')
rna_0.obs.index = samples[0][0]+'_'+rna_0.obs.index 
atac_0 = sc.read_h5ad('/fs/home/jiluzhang/2023_nature_LD/normal_h5ad/'+samples[0][0]+'_atac.h5ad')
atac_0.obs.index = samples[0][0]+'_'+atac_0.obs.index 
rna = rna_0.copy()
atac = atac_0.copy()

for i in range(1, samples.shape[0]):
    rna_tmp = sc.read_h5ad('/fs/home/jiluzhang/2023_nature_LD/normal_h5ad/'+samples[0][i]+'_rna.h5ad')
    rna_tmp.obs.index = samples[0][i]+'_'+rna_tmp.obs.index 
    atac_tmp = sc.read_h5ad('/fs/home/jiluzhang/2023_nature_LD/normal_h5ad/'+samples[0][i]+'_atac.h5ad')
    atac_tmp.obs.index = samples[0][i]+'_'+atac_tmp.obs.index 
    rna = ad.concat([rna, rna_tmp])
    atac = ad.concat([atac, atac_tmp])
    print(samples[0][i], 'done')

rna.var = rna_0.var
atac.var = atac_0.var


## workdir: all_data
# python out_train_dataset.sh
for sample in `cat ../files_all.txt`;do
    mkdir $sample
    cp /fs/home/jiluzhang/2023_nature_LD/normal_h5ad/$sample\_rna.h5ad $sample/rna.h5ad
    cp /fs/home/jiluzhang/2023_nature_LD/normal_h5ad/$sample\_atac.h5ad $sample/atac.h5ad
    python data_preprocess.py -r $sample/rna.h5ad -a $sample/atac.h5ad -s $sample/preprocessed_data_train --dt train --config ../rna2atac_config_train.yaml
    echo $sample done
done


## list all pt files
import os
import random

train_files = []
for root, dirs, files in os.walk('./all_data/train_datasets'):
    for filename in files:
        if filename[-2:]=='pt':
            train_files.append(os.path.join(root, filename))

random.seed(0)
random.shuffle(train_files)


########## no fine tune for cl dataset ##########
accelerate launch --config_file accelerator_config_test.yaml --main_process_port 29823 rna2atac_test.py \
                  -d ./preprocessed_data_test \
                  -l /fs/home/jiluzhang/scM2M_no_dec_attn/pan_cancer/save/2024-06-25_rna2atac_train_3/pytorch_model.bin \
                  --config_file rna2atac_config_test.yaml

python npy2h5ad.py
mv rna2atac_scm2m_raw.h5ad ../benchmark/rna2atac_scm2mpancancernoft_raw.h5ad
python csr2array.py --pred rna2atac_scm2mpancancernoft_raw.h5ad  --true rna2atac_true.h5ad
python cal_auroc_auprc.py --pred rna2atac_scm2mpancancernoft.h5ad --true rna2atac_true.h5ad
python cal_cluster_plot.py --pred rna2atac_scm2mpancancernoft.h5ad --true rna2atac_true.h5ad
# AMI: [0.2027, 0.2392, 0.1969, 0.1883, 0.1899]
# ARI: [0.1369, 0.1468, 0.1293, 0.1106, 0.1206]
# HOM: [0.158, 0.2261, 0.154, 0.1765, 0.1491]
# NMI: [0.2164, 0.2619, 0.2107, 0.2146, 0.2038]

########## with fine tune for cl dataset ##########
accelerate launch --config_file accelerator_config_train.yaml --main_process_port 29822 rna2atac_train.py --config_file rna2atac_config_train.yaml \
                  --train_data_dir ./preprocessed_data_train --val_data_dir ./preprocessed_data_val -n rna2atac_train -s save_ft_1e-3 \
                  -l /fs/home/jiluzhang/scM2M_no_dec_attn/pan_cancer/save/2024-06-25_rna2atac_train_3/pytorch_model.bin

accelerate launch --config_file accelerator_config_test.yaml --main_process_port 29823 rna2atac_test.py \
                  -d ./preprocessed_data_test \
                  -l ./save_ft_1e-3/2024-06-26_rna2atac_train_72/pytorch_model.bin \
                  --config_file rna2atac_config_test.yaml

python npy2h5ad.py
mv rna2atac_scm2m_raw.h5ad ../benchmark/rna2atac_scm2mpancancerft_raw.h5ad
python csr2array.py --pred rna2atac_scm2mpancancerft_raw.h5ad  --true rna2atac_true.h5ad
python cal_auroc_auprc.py --pred rna2atac_scm2mpancancerft.h5ad --true rna2atac_true.h5ad
python cal_cluster_plot.py --pred rna2atac_scm2mpancancerft.h5ad --true rna2atac_true.h5ad
# AMI: [0.878, 0.8758, 0.878, 0.8785, 0.8811]
# ARI: [0.8574, 0.8457, 0.8574, 0.8647, 0.873]
# HOM: [0.8253, 0.819, 0.8253, 0.8278, 0.8321]
# NMI: [0.8832, 0.8812, 0.8832, 0.8836, 0.8862]




pred = sc.read_h5ad('rna2atac_scm2mpancancernoft.h5ad')
#pred.X = pred.X.toarray()
pred.X[pred.X>0.5]=1
pred.X[pred.X<=0.5]=0
#pred.X = ((pred.X.T > pred.X.T.mean(axis=0)).T) & (pred.X>pred.X.mean(axis=0)).astype(int)
round(pearsonr(pred.X.sum(axis=0), true.X.toarray().sum(axis=0))[0], 4)


## cell count stats
import scanpy as sc
import os

s_lst = os.listdir('.')
cell_cnt = 0
for s in s_lst:
    cell_cnt += sc.read_h5ad(s+'/rna.h5ad').n_obs
print(cell_cnt)
# 461604 (for training)
# 4061 (for validating)


python data_preprocess.py -r val_datasets/GBML018G1-M1/rna.h5ad -a val_datasets/GBML018G1-M1/atac.h5ad -s preprocessed_data_test --dt test --config ../rna2atac_config_test.yaml

accelerate launch --config_file accelerator_config_test.yaml --main_process_port 29823 rna2atac_test.py \
                  -d ./preprocessed_data_test \
                  -l ../save/2024-06-25_rna2atac_train_3/pytorch_model.bin \
                  --config_file rna2atac_config_test.yaml

python npy2h5ad.py
python csr2array.py --pred rna2atac_scm2m_raw.h5ad  --true val_datasets/GBML018G1-M1/atac.h5ad
python cal_auroc_auprc.py --pred rna2atac_scm2m.h5ad --true rna2atac_true.h5ad
python cal_cluster_plot.py --pred rna2atac_scm2m.h5ad --true rna2atac_true.h5ad
# AMI: [0.2586, 0.2621, 0.2654, 0.2628, 0.2818]
# ARI: [0.147, 0.1475, 0.1507, 0.1476, 0.1555]
# HOM: [0.3055, 0.3092, 0.3132, 0.31, 0.3296]
# NMI: [0.2683, 0.2718, 0.275, 0.2725, 0.2913]


## cluster & plot for true atac
import snapatac2 as snap
import scanpy as sc

true = snap.read('./val_datasets/GBML018G1-M1/atac.h5ad', backed=None)
snap.pp.select_features(true)
snap.tl.spectral(true)
snap.tl.umap(true)
snap.pp.knn(true)
snap.tl.leiden(true)
sc.pl.umap(true[:550, :], color='leiden', legend_fontsize='7', legend_loc='right margin', size=5,
           title='', frameon=True, save='_atac_true.pdf')
true[:550, :].write('rna2atac_true.h5ad')



################ DSRCT multiome dataset ################
wget -c https://ftp.ncbi.nlm.nih.gov/geo/series/GSE263nnn/GSE263522/suppl/GSE263522_RAW.tar

mv GSM8193977_GR11_Multiome_matrix.mtx matrix.mtx
gzip matrix.mtx
mv GSM8193977_GR11_Multiome_features.tsv features.tsv
gzip features.tsv
mv GSM8193977_GR11_Multiome_barcodes.tsv barcodes.tsv
gzip barcodes.tsv 

## RNA
import scanpy as sc

rna = sc.read_10x_mtx('./rna')  # 2916 × 36601
rna.write('rna_raw.h5ad')

## ATAC
import snapatac2 as snap
import pandas as pd

ccre = pd.read_table('/fs/home/jiluzhang/scM2M_no_dec_attn/scM2M_cl/data/human_cCREs.bed', header=None)
ccre[1] = ccre[1].astype(str)
ccre[2] = ccre[2].astype(str)
ccre['idx'] = ccre[0]+':'+ccre[1]+'-'+ccre[2]

atac = snap.pp.import_data(fragment_file='./GSM8193977_GR11_Multiome_atac_fragments.tsv', genome=snap.genome.hg38, file='atac_raw.h5ad', sorted_by_barcode=False)  # ~5 min
peak = snap.pp.make_peak_matrix(atac, use_rep=ccre['idx'])  # 15138 × 1033239
peak.write('atac_peak_raw.h5ad')


## Combine RNA & ATAC
import scanpy as sc
import pandas as pd
from scipy.sparse import csr_matrix
import numpy as np

rna = sc.read_h5ad('rna_raw.h5ad')
rna.X = rna.X.toarray()

genes = pd.read_table('/fs/home/jiluzhang/scM2M_no_dec_attn/scM2M_cl/data/human_genes.txt', names=['gene_ids', 'gene_name'])

rna_exp = pd.concat([rna.var, pd.DataFrame(rna.X.T, index=rna.var.index.values)], axis=1)
X_new = pd.merge(genes, rna_exp, how='left', on='gene_ids').iloc[:, 3:].T
X_new.fillna(value=0, inplace=True)

rna_new = sc.AnnData(X_new.values, obs=rna.obs, var=pd.DataFrame({'gene_ids': genes['gene_ids'], 'feature_types': 'Gene Expression'}))
rna_new.var.index = genes['gene_name'].values
rna_new.X = csr_matrix(rna_new.X)

sc.pp.filter_cells(rna_new, min_genes=500)
sc.pp.filter_cells(rna_new, max_genes=10000)

atac = sc.read_h5ad('atac_raw.h5ad')
atac.X[atac.X>1]=1

sc.pp.filter_cells(atac, min_genes=1000)
sc.pp.filter_cells(atac, max_genes=50000)

idx = np.intersect1d(rna_new.obs.index, atac.obs.index)  
rna_new[idx, :].copy().write('rna.h5ad')     # 2359 × 38244
atac[idx, :].copy().write('atac.h5ad')       # 2359 × 1033239


## fine tune & predict
python split_train_val.py --RNA rna.h5ad --ATAC atac.h5ad --train_pct 0.9
mv rna_train.h5ad rna_2123.h5ad
mv atac_train.h5ad atac_2123.h5ad
mv rna_val.h5ad rna_236.h5ad
mv atac_val.h5ad atac_236.h5ad

python data_preprocess.py -r rna_236.h5ad -a atac_236.h5ad -s preprocessed_data_test --dt test --config rna2atac_config_test.yaml

accelerate launch --config_file accelerator_config_test.yaml --main_process_port 29822 rna2atac_test.py \
                  -d ./preprocessed_data_test \
                  -l /fs/home/jiluzhang/scM2M_no_dec_attn/pan_cancer/save/2024-06-25_rna2atac_train_3/pytorch_model.bin \
                  --config_file rna2atac_config_test.yaml

python npy2h5ad.py
python csr2array.py --pred rna2atac_scm2m_raw.h5ad  --true rna2atac_true.h5ad
python cal_auroc_auprc.py --pred rna2atac_scm2m.h5ad --true rna2atac_true.h5ad
python cal_cluster_plot.py --pred rna2atac_scm2m.h5ad --true rna2atac_true.h5ad


## cluster for true atac
import snapatac2 as snap
import numpy as np
from scipy.sparse import csr_matrix
import scanpy as sc

true = snap.read('atac_236.h5ad', backed=None)
snap.pp.select_features(true)
snap.tl.spectral(true)
snap.tl.umap(true)
snap.pp.knn(true)
snap.tl.leiden(true)
sc.pl.umap(true, color='leiden', legend_fontsize='7', legend_loc='right margin', size=5,
           title='', frameon=True, save='_atac_236_true.pdf')
true.write('rna2atac_true.h5ad')

# sc.pl.umap(true, color='leiden', legend_fontsize='7', legend_loc='right margin', size=15,
#            title='', frameon=True, save='_atac_236_true_bigsize.pdf')

python data_preprocess.py -r rna_2123.h5ad -a atac_2123.h5ad -s preprocessed_data_train --dt train --config rna2atac_config_train.yaml

accelerate launch --config_file accelerator_config_train.yaml --main_process_port 29822 rna2atac_train.py --config_file rna2atac_config_train.yaml \
                  --train_data_dir ./preprocessed_data_train --val_data_dir ./preprocessed_data_test -n rna2atac_train -s save_ft_2123 \
                  -l /fs/home/jiluzhang/scM2M_no_dec_attn/pan_cancer/save/2024-06-25_rna2atac_train_3/pytorch_model.bin
# 352883

## find the most similar dataset for the target rna
# pan_cancer_samples.txt  127
import scanpy as sc
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances
import numpy as np
from tqdm import tqdm
from scipy.stats import pearsonr

tgt_rna = sc.read_h5ad('rna_236.h5ad')
tgt_rna_m = tgt_rna.X.toarray()

cor_lst = []
pan_cancer_samples = pd.read_table('pan_cancer_samples.txt', header=None)
for s in tqdm(pan_cancer_samples[0].values, ncols=80):
    rna_s_m = sc.read_h5ad('/fs/home/jiluzhang/scM2M_no_dec_attn/pan_cancer/all_data/train_datasets/'+s+'/rna.h5ad').X.toarray()
    cor_lst.append(round(pearsonr(tgt_rna_m.sum(axis=0), rna_s_m.sum(axis=0))[0], 4))

np.argmax(cor_lst)  # 122
pan_cancer_samples[0][np.argmax(cor_lst)]   # 'VF032V1-S1'


accelerate launch --config_file accelerator_config_train.yaml --main_process_port 29822 rna2atac_train.py --config_file rna2atac_config_train.yaml \
                  --train_data_dir /fs/home/jiluzhang/scM2M_no_dec_attn/pan_cancer/all_data/train_datasets/VF032V1-S1/preprocessed_data_train \
                  --val_data_dir ./preprocessed_data_test -n rna2atac_train -s save_ft_VF032V1_S1 \
                  -l /fs/home/jiluzhang/scM2M_no_dec_attn/pan_cancer/save/2024-06-25_rna2atac_train_3/pytorch_model.bin
# 557002



## cell mapping
# Using DenseFly algorithm for cell searching on massive scRNA-seq datasets
# scmap: https://github.com/hemberg-lab/scmap

import scanpy as sc
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors

tgt_rna = sc.read_h5ad('rna_236.h5ad')
tgt_rna_m = tgt_rna.X.toarray()
pan_cancer_samples = pd.read_table('pan_cancer_samples.txt', header=None)

for i in tqdm(range(pan_cancer_samples.shape[0]), ncols=80):
    if i==0:
        ref_rna = sc.read_h5ad('/fs/home/jiluzhang/scM2M_no_dec_attn/pan_cancer/all_data/train_datasets/'+pan_cancer_samples[0][0]+'/rna.h5ad')
        ref_rna_m = ref_rna.X.toarray()
        neigh = NearestNeighbors(n_neighbors=1, metric='cosine')
        neigh.fit(ref_rna_m)
        dist, idx = neigh.kneighbors(tgt_rna_m, 1, return_distance=True)
        dist = dist.flatten()
        idx = idx.flatten()
        obs_knn = ref_rna[idx, :].obs.index.values
        pt_lst = [[pan_cancer_samples[0][0], i] for i in obs_knn]
    else:
        ref_rna_tmp = sc.read_h5ad('/fs/home/jiluzhang/scM2M_no_dec_attn/pan_cancer/all_data/train_datasets/'+pan_cancer_samples[0][i]+'/rna.h5ad')
        ref_rna_m_tmp = ref_rna_tmp.X.toarray()
        neigh = NearestNeighbors(n_neighbors=1, metric='cosine')
        neigh.fit(ref_rna_m_tmp)
        dist_tmp, idx_tmp = neigh.kneighbors(tgt_rna_m, 1, return_distance=True)
        dist_tmp = dist_tmp.flatten()
        idx_tmp = idx_tmp.flatten()
        
        bool_tmp = dist_tmp<dist
        dist[bool_tmp] = dist_tmp[bool_tmp]
        obs_knn_tmp = ref_rna_tmp[idx_tmp, :].obs.index.values
        
        for t in range(len(bool_tmp)):
            if bool_tmp[t]:
                pt_lst[t] = [pan_cancer_samples[0][i], obs_knn_tmp[t]]

np.save('knn_res.npy', np.array(pt_lst))

################ extract certain cells' info from h5ad files !!!!!!!!!!!!!!!!!!

import numpy as np
import pandas as pd
from tqdm import tqdm
import scanpy as sc
from scipy.sparse import csr_matrix

idx = np.load('knn_res.npy')

pan_cancer_samples = pd.read_table('pan_cancer_samples.txt', header=None)
rna_dict = {}
atac_dict = {}
for i in tqdm(range(pan_cancer_samples.shape[0]), ncols=80):
    rna_dict[pan_cancer_samples[0][i]] = sc.read_h5ad('/fs/home/jiluzhang/scM2M_no_dec_attn/pan_cancer/all_data/train_datasets/'+pan_cancer_samples[0][i]+'/rna.h5ad')
    atac_dict[pan_cancer_samples[0][i]] = sc.read_h5ad('/fs/home/jiluzhang/scM2M_no_dec_attn/pan_cancer/all_data/train_datasets/'+pan_cancer_samples[0][i]+'/atac.h5ad')

rna_m = np.zeros(shape=[236, 38244])
atac_m = np.zeros(shape=[236, 1033239])
obs_lst = []
for j in tqdm(range(len(idx)), ncols=80):
    obs_lst.append(idx[j][0]+'_'+idx[j][1])
    rna_m[j] = rna_dict[idx[j][0]][idx[j][1]].X.toarray()
    atac_m[j] = atac_dict[idx[j][0]][idx[j][1]].X.toarray()

rna_out = sc.AnnData(csr_matrix(rna_m), obs=[], var=rna_dict[idx[0][0]][idx[0][1]].var)
rna_out.obs.index = obs_lst
rna_out.write('rna_236_knn.h5ad')

atac_out = sc.AnnData(csr_matrix(atac_m), obs=[], var=atac_dict[idx[0][0]][idx[0][1]].var)
atac_out.obs.index = obs_lst
atac_out.write('atac_236_knn.h5ad')  ## maybe with duplicates


python data_preprocess.py -r rna_236_knn.h5ad -a atac_236_knn.h5ad -s preprocessed_data_train_knn --dt train --config rna2atac_config_train.yaml

accelerate launch --config_file accelerator_config_train.yaml --main_process_port 29822 rna2atac_train.py --config_file rna2atac_config_train.yaml \
                  --train_data_dir ./preprocessed_data_train_knn \
                  --val_data_dir ./preprocessed_data_test -n rna2atac_train -s save_ft_knn \
                  -l /fs/home/jiluzhang/scM2M_no_dec_attn/pan_cancer/save/2024-06-25_rna2atac_train_3/pytorch_model.bin
# 1282803



## prediction by model_ft_2123
accelerate launch --config_file accelerator_config_test.yaml --main_process_port 29823 rna2atac_test.py \
                  -d ./preprocessed_data_test \
                  -l ./save_ft_2123/2024-07-01_rna2atac_train_40/pytorch_model.bin \
                  --config_file rna2atac_config_test.yaml

python npy2h5ad.py
python csr2array.py --pred rna2atac_scm2m_raw.h5ad  --true rna2atac_true.h5ad
python cal_auroc_auprc.py --pred rna2atac_scm2m.h5ad --true rna2atac_true.h5ad
python cal_cluster_plot.py --pred rna2atac_scm2m.h5ad --true rna2atac_true.h5ad
# AMI: [0.0557, 0.0657, 0.0671, 0.0686, 0.0602]
# ARI: [0.0183, 0.0305, 0.0308, 0.033, 0.0245]
# HOM: [0.0583, 0.0786, 0.0798, 0.0815, 0.0735]
# NMI: [0.0684, 0.0827, 0.0842, 0.0856, 0.0773]


## prediction by model_ft_VF032V1_S1
accelerate launch --config_file accelerator_config_test.yaml --main_process_port 29823 rna2atac_test.py \
                  -d ./preprocessed_data_test \
                  -l ./save_ft_VF032V1_S1/2024-07-01_rna2atac_train_7/pytorch_model.bin \
                  --config_file rna2atac_config_test.yaml

python npy2h5ad.py
python csr2array.py --pred rna2atac_scm2m_raw.h5ad  --true rna2atac_true.h5ad
python cal_auroc_auprc.py --pred rna2atac_scm2m.h5ad --true rna2atac_true.h5ad
python cal_cluster_plot.py --pred rna2atac_scm2m.h5ad --true rna2atac_true.h5ad
# AMI: [0.0239, 0.0308, 0.0272, 0.0239, 0.0296]
# ARI: [0.0054, 0.0104, 0.0082, 0.0055, 0.0094]
# HOM: [0.0396, 0.0461, 0.0427, 0.0396, 0.045]
# NMI: [0.0416, 0.0483, 0.0448, 0.0416, 0.0472]


## prediction by model_ft_knn
accelerate launch --config_file accelerator_config_test.yaml --main_process_port 29823 rna2atac_test.py \
                  -d ./preprocessed_data_test \
                  -l ./save_ft_knn/2024-07-03_rna2atac_train_38/pytorch_model.bin \
                  --config_file rna2atac_config_test.yaml

python npy2h5ad.py
python csr2array.py --pred rna2atac_scm2m_raw.h5ad  --true rna2atac_true.h5ad
python cal_auroc_auprc.py --pred rna2atac_scm2m.h5ad --true rna2atac_true.h5ad
python cal_cluster_plot.py --pred rna2atac_scm2m.h5ad --true rna2atac_true.h5ad
# AMI: [0.0083, 0.0095, 0.0361, 0.0256, 0.0278]
# ARI: [0.0003, 0.0024, 0.014, 0.0098, 0.0136]
# HOM: [0.0251, 0.0261, 0.0594, 0.041, 0.0433]
# NMI: [0.0264, 0.0276, 0.058, 0.0436, 0.0456]





############ motif enrichment analysis ############
######## True ########
import snapatac2 as snap
import scanpy as sc
import numpy as np
import random

atac = snap.read('atac_236.h5ad', backed=None)
snap.pp.select_features(atac)
snap.tl.spectral(atac)
snap.tl.umap(atac)
snap.pp.knn(atac)
snap.tl.leiden(atac, resolution=0.5)
atac.obs.leiden.value_counts()
# 0    174
# 1     62
# 0: malignant cells
# 1: non-malignant cells
sc.pl.umap(atac, color='leiden', legend_fontsize='7', legend_loc='right margin', size=5,
           title='', frameon=True, save='_atac_236_true_2_clusters.pdf')
atac.write('atac_236_2_clusters.h5ad')

atac = sc.read_h5ad('atac_236_2_clusters.h5ad')

atac_0 = atac[atac.obs.leiden=='0'].copy()
atac_1 = atac[atac.obs.leiden=='1'].copy()

snap.pp.select_features(atac_0)
snap.pp.select_features(atac_1)

peaks = atac_0.var.index.values

atac_0.var.index = list(range(atac_0.n_vars))
atac_0_peak_idx = atac_0.var[atac_0.var['count']>atac_0.n_obs*0.10].index.values  # 47883

atac_1.var.index = list(range(atac_1.n_vars))
atac_1_peak_idx = atac_1.var[atac_1.var['count']>atac_1.n_obs*0.02].index.values  # 43691

atac_0_sp = peaks[np.setdiff1d(atac_0_peak_idx, atac_1_peak_idx)]
atac_1_sp = peaks[np.setdiff1d(atac_1_peak_idx, atac_0_peak_idx)]

random.seed(0)
peaks_shuf = peaks.copy()
random.shuffle(peaks_shuf)
marker_peaks = {'0': atac_0_sp, '1': atac_1_sp}

motifs = snap.tl.motif_enrichment(motifs=snap.datasets.cis_bp(unique=True),
                                  regions=marker_peaks,
                                  genome_fasta=snap.genome.hg38,
                                  background=peaks_shuf[:50000],
                                  method='hypergeometric')  # ~6 min
# background   A list of regions to be used as the background. If None, the union of elements in `regions` will be used as the background.
# method  Statistical testing method: "binomial" or "hypergeometric".  To use "hypergeometric", the testing regions must be a subset of background regions.

motifs_0 = motifs['0'].to_pandas()
del motifs_0['family']
motifs_0.to_csv('atac_236_motif_cluster_0.txt', sep='\t', index=False)

motifs_1 = motifs['1'].to_pandas()
del motifs_1['family']
motifs_1.to_csv('atac_236_motif_cluster_1.txt', sep='\t', index=False)


## Plot top enriched motifs
import pandas as pd
from plotnine import *
import numpy as np

## cluster 0
motifs_0 = pd.read_table('atac_236_motif_cluster_0.txt')
motifs_0.index = motifs_0['name'].values
tfs = ['EGR3', 'ZNF740', 'EGR1', 'KLF16', 'MAZ', 'RREB1', 'ZNF148', 'SP1', 'SP2', 'ZNF263']
tfs.reverse()
motifs_0_stat = motifs_0.loc[tfs]
motifs_0_stat['name'] = pd.Categorical(motifs_0_stat['name'], categories=tfs)

dat = motifs_0_stat[['name', 'log2(fold change)']]

p = ggplot(dat, aes(x='name', y='log2(fold change)')) + geom_bar(stat='identity', fill='darkred', position=position_dodge()) + coord_flip() + \
                                                        xlab('Motif') + ylab('log2FC') + labs(title='Malignant cells')  + \
                                                        scale_y_continuous(limits=[0, 3], breaks=np.arange(0, 3.1, 0.5)) + theme_bw() + theme(plot_title=element_text(hjust=0.5))
p.save(filename='atac_236_motif_cluster_0.pdf', dpi=600, height=4, width=5)

## cluster 1
motifs_1 = pd.read_table('atac_236_motif_cluster_1.txt')
motifs_1.index = motifs_1['name'].values
tfs = ['FOSL2', 'JUND', 'JUNB', 'FOSL1', 'BATF3', 'BATF']
tfs.reverse()
motifs_1_stat = motifs_1.loc[tfs]
motifs_1_stat['name'] = pd.Categorical(motifs_1_stat['name'], categories=tfs)

dat = motifs_1_stat[['name', 'log2(fold change)']]

p = ggplot(dat, aes(x='name', y='log2(fold change)')) + geom_bar(stat='identity', fill='midnightblue', position=position_dodge()) + coord_flip() + \
                                                        xlab('Motif') + ylab('log2FC') + labs(title='Non-malignant cells')  + \
                                                        scale_y_continuous(limits=[-1.5, 1], breaks=np.arange(-1.5, 1.1, 0.5)) + theme_bw() + theme(plot_title=element_text(hjust=0.5))
p.save(filename='atac_236_motif_cluster_1.pdf', dpi=600, height=2.5, width=5)



######## scM2M predict (fine-tune by target dataset) ########
import snapatac2 as snap
import scanpy as sc
import numpy as np
import random

atac = snap.read('ft_2123/rna2atac_scm2m_binary_ft_2123.h5ad', backed=None)
snap.pp.select_features(atac)
snap.tl.spectral(atac)
snap.tl.umap(atac)
snap.pp.knn(atac)
snap.tl.leiden(atac, resolution=0.4)
atac.obs.leiden.value_counts()
# 0    132
# 1    104
# 0: malignant cells
# 1: non-malignant cells
sc.pl.umap(atac, color='leiden', legend_fontsize='7', legend_loc='right margin', size=5,
           title='', frameon=True, save='_atac_236_scm2mft2123_2_clusters.pdf')
atac.write('atac_236_scm2mft2123_2_clusters.h5ad')

atac = sc.read_h5ad('atac_236_scm2mft2123_2_clusters.h5ad')

atac_0 = atac[atac.obs.leiden=='0'].copy()
atac_1 = atac[atac.obs.leiden=='1'].copy()

snap.pp.select_features(atac_0)
snap.pp.select_features(atac_1)

peaks = atac_0.var.index.values

atac_0.var.index = list(range(atac_0.n_vars))
atac_0_peak_idx = atac_0.var[atac_0.var['count']>atac_0.n_obs*0.9].index.values  # 75208

atac_1.var.index = list(range(atac_1.n_vars))
atac_1_peak_idx = atac_1.var[atac_1.var['count']>atac_1.n_obs*0.9].index.values  # 76281

atac_0_sp = peaks[np.setdiff1d(atac_0_peak_idx, atac_1_peak_idx)]  # 2641
atac_1_sp = peaks[np.setdiff1d(atac_1_peak_idx, atac_0_peak_idx)]  # 3714

random.seed(0)
peaks_shuf = peaks.copy()
random.shuffle(peaks_shuf)
marker_peaks = {'0': atac_0_sp, '1': atac_1_sp}

motifs = snap.tl.motif_enrichment(motifs=snap.datasets.cis_bp(unique=True),
                                  regions=marker_peaks,
                                  genome_fasta=snap.genome.hg38,
                                  background=peaks_shuf[:50000],
                                  method='hypergeometric')  # ~6 min
# background   A list of regions to be used as the background. If None, the union of elements in `regions` will be used as the background.
# method  Statistical testing method: "binomial" or "hypergeometric".  To use "hypergeometric", the testing regions must be a subset of background regions.

motifs_0 = motifs['0'].to_pandas()
del motifs_0['family']
motifs_0.to_csv('atac_236_motif_scm2mft2123_cluster_0.txt', sep='\t', index=False)

motifs_1 = motifs['1'].to_pandas()
del motifs_1['family']
motifs_1.to_csv('atac_236_motif_scm2mft2123_cluster_1.txt', sep='\t', index=False)


## Plot top enriched motifs
import pandas as pd
from plotnine import *
import numpy as np

## cluster 0
motifs_0 = pd.read_table('atac_236_motif_scm2mft2123_cluster_0.txt')
motifs_0.index = motifs_0['name'].values
tfs = ['EGR3', 'ZNF740', 'EGR1', 'KLF16', 'MAZ', 'RREB1', 'ZNF148', 'SP1', 'SP2', 'ZNF263']
tfs.reverse()
motifs_0_stat = motifs_0.loc[tfs]
motifs_0_stat['name'] = pd.Categorical(motifs_0_stat['name'], categories=tfs)

dat = motifs_0_stat[['name', 'log2(fold change)']]

p = ggplot(dat, aes(x='name', y='log2(fold change)')) + geom_bar(stat='identity', fill='darkred', position=position_dodge()) + coord_flip() + \
                                                        xlab('Motif') + ylab('log2FC') + labs(title='Malignant cells')  + \
                                                        scale_y_continuous(limits=[0, 3], breaks=np.arange(0, 3.1, 0.5)) + theme_bw() + theme(plot_title=element_text(hjust=0.5))
p.save(filename='atac_236_motif_scm2mft2123_cluster_0.pdf', dpi=600, height=4, width=5)

## cluster 1
motifs_1 = pd.read_table('atac_236_motif_scm2mft2123_cluster_1.txt')
motifs_1.index = motifs_1['name'].values
tfs = ['FOSL2', 'JUND', 'JUNB', 'FOSL1', 'BATF3', 'BATF']
tfs.reverse()
motifs_1_stat = motifs_1.loc[tfs]
motifs_1_stat['name'] = pd.Categorical(motifs_1_stat['name'], categories=tfs)

dat = motifs_1_stat[['name', 'log2(fold change)']]

p = ggplot(dat, aes(x='name', y='log2(fold change)')) + geom_bar(stat='identity', fill='midnightblue', position=position_dodge()) + coord_flip() + \
                                                        xlab('Motif') + ylab('log2FC') + labs(title='Non-malignant cells')  + \
                                                        scale_y_continuous(limits=[-1.5, 1], breaks=np.arange(-1.5, 1.1, 0.5)) + theme_bw() + theme(plot_title=element_text(hjust=0.5))
p.save(filename='atac_236_motif_scm2mft2123_cluster_1.pdf', dpi=600, height=2.5, width=5)


## Comparison between predicted and true results
# total 1165 TFs
import pandas as pd
import numpy as np
#from plotnine import *

pred_0 = pd.read_table('atac_236_motif_scm2mft2123_cluster_0.txt')
pred_0.index = pred_0['name'].values
sum((pred_0['adjusted p-value']<0.01) & (pred_0['log2(fold change)']>0))  # 157
pred_0_tf = pred_0[(pred_0['adjusted p-value']<0.01) & (pred_0['log2(fold change)']>0)]['name'].values

true_0 = pd.read_table('atac_236_motif_cluster_0.txt')
true_0.index = true_0['name'].values
sum((true_0['adjusted p-value']<0.01) & (true_0['log2(fold change)']>0))  # 432
true_0_tf = true_0[(true_0['adjusted p-value']<0.01) & (true_0['log2(fold change)']>0)]['name'].values

pred_1 = pd.read_table('atac_236_motif_scm2mft2123_cluster_1.txt')
pred_1.index = pred_1['name'].values
sum((pred_1['adjusted p-value']<0.01) & (pred_1['log2(fold change)']>0))  # 197
pred_1_tf = pred_1[(pred_1['adjusted p-value']<0.01) & (pred_1['log2(fold change)']>0)]['name'].values

true_1 = pd.read_table('atac_236_motif_cluster_1.txt')
true_1.index = true_1['name'].values
sum((true_1['adjusted p-value']<0.01) & (true_1['log2(fold change)']>0))  # 346
true_1_tf = true_1[(true_1['adjusted p-value']<0.01) & (true_1['log2(fold change)']>0)]['name'].values

np.intersect1d(pred_0_tf, true_0_tf).shape[0]  # 156
np.intersect1d(pred_1_tf, true_1_tf).shape[0]  # 191

import matplotlib.pyplot as plt
from matplotlib_venn import venn2
plt.figure()
venn2(subsets = [set(pred_0_tf), set(true_0_tf)], set_labels=('Pred','True'), set_colors=('darkred','midnightblue'))
plt.savefig('pred_0_true_0_tf_venn.pdf')

plt.figure()
venn2(subsets = [set(pred_1_tf), set(true_1_tf)], set_labels=('Pred','True'), set_colors=('darkred','midnightblue'))
plt.savefig('pred_1_true_1_tf_venn.pdf')


############################### p-value=0 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

#snap.pl.motif_enrichment(motifs, max_fdr=0.00001, height=500, show=False, out_file='motif_enrichment_0.pdf')


## Paper in Cell Reports Medicine: https://www.cell.com/cell-reports-medicine/fulltext/S2666-3791(24)00274-X#supplementaryMaterial




setRepositories(ind=1:3) # needed to automatically install Bioconductor dependencies
install.packages("Signac")
install.packages("GenomicRanges")
install.packages("Rhtslib")
install.packages("Rsamtools")
install.packages("Signac")


# level 4 rds data

library(Seurat)
dat <- readRDS('CPT1541DU-T1.rds')
write.table(dat@meta.data['cell_type'], 'UCEC_cell_type.txt', row.names=TRUE, col.names=FALSE, quote=FALSE, sep='\t')


## python download_rds.py
import synapseclient
import pandas as pd
import os

syn = synapseclient.Synapse()
syn.login('Luz','jiluzhang_57888282')

ids = pd.read_table('rds_synapse_id.txt', header=None)
for id in ids[1].values:
    meta = syn.get(entity=id, downloadFile=True, downloadLocation='.')
    print(id, 'done')

############## CEAD ##############
library(Seurat)

cead_stats <- data.frame()
for (sample in c('CE336E1-S1', 'CE354E1-S1', 'CE357E1-S1')){
    dat <- readRDS(paste0(sample, '.rds'))
    cead_stats <- rbind(cead_stats, data.frame(table(dat@meta.data['cell_type'])))
    print(paste0(sample, ' done'))    
}
cead_stats <- aggregate(Freq~cell_type, stats, FUN=function(x) sum(x))
# cell_type Freq
# 1 Endothelial  114
# 2 Fibroblasts  115
# 3 Macrophages  114
# 4     T-cells   18
# 5       Tumor 9657
# 6     B-cells    4
# 7      Plasma   36
write.table(cead_stats, 'cead_stats.txt', row.names=FALSE, col.names=FALSE, quote=FALSE)

############## CESC ##############
cesc_stats <- data.frame()
for (sample in c('CE337E1-S1N1', 'CE338E1-S1N1', 'CE339E1-S1N1', 'CE340E1-S1N1', 'CE342E1-S1N1',
                 'CE346E1-S1',   'CE347E1-S1K1', 'CE348E1-S1K1', 'CE349E1-S1',   'CE352E1-S1K1',
                 'CE356E1-S1')){
    dat <- readRDS(paste0(sample, '.rds'))
    cesc_stats <- rbind(cesc_stats, data.frame(table(dat@meta.data['cell_type'])))
    print(paste0(sample, ' done'))    
}
cesc_stats <- aggregate(Freq~cell_type, cesc_stats, FUN=function(x) sum(x))
#                  cell_type  Freq
# 1                  B-cells   199
# 2              Endothelial   387
# 3              Fibroblasts  1365
# 4              Macrophages  1145
# 5  Normal epithelial cells   248
# 6    Normal squamous cells   143
# 7                   Plasma   452
# 8                  T-cells  1112
# 9                    Tumor 26584
# 10             Low quality   164
write.table(cesc_stats, 'cesc_stats.txt', row.names=FALSE, col.names=FALSE, quote=FALSE)

############## CRC ##############
crc_stats <- data.frame()
# SP819H1-Mc1 -> CRC_SP819H1-Mc1
for (sample in c('CM1563C1-S1Y1', 'CM1563C1-T1Y1', 'CM268C1-S1',    'CM268C1-T1',    'CM354C1-T1Y2',
                 'CM354C2-T1Y2',  'CM478C1-T1Y2',  'CM478C2-T1Y2',  'CM663C1-T1Y1',  'HT225C1-Th1', 
                 'HT230C1-Th1',   'HT250C1-Th1K1', 'HT253C1-Th1K1', 'HT260C1-Th1K1', 'HT291C1-M1A3',
                 'HT307C1-Th1K1', 'SP369H1-Mc1',   'CRC_SP819H1-Mc1')){
    dat <- readRDS(paste0(sample, '.rds'))
    crc_stats <- rbind(crc_stats, data.frame(table(dat@meta.data['cell_type'])))
    print(paste0(sample, ' done'))    
}
crc_stats <- aggregate(Freq~cell_type, crc_stats, FUN=function(x) sum(x))
#                       cell_type  Freq
# 1                       B-cells  1979
# 2             Distal Absorptive  1719
# 3            Distal Enterocytes  2145
# 4                   Endothelial  1436
# 5                   Fibroblasts  3508
# 6                        Goblet   875
# 7                   Hepatocytes  3128
# 8                   Low quality  1242
# 9                   Macrophages  8869
# 10                       Plasma  2644
# 11                      T-cells 10636
# 12                        Tumor 58538
# 13                     Alveolar  1761
# 14               Cholangiocytes   421
# 15 Absorptive BEST4/OTOP2 cells    44
# 16    Distal Mature Enterocytes   460
# 17            Distal Stem Cells   414
# 18                    Distal TA  1554
# 19               Other_doublets   190
# 20                           TA    93
write.table(crc_stats, 'crc_stats.txt', row.names=FALSE, col.names=FALSE, quote=FALSE)

############## UCEC ##############
ucec_stats <- data.frame()
for (sample in c('CPT1541DU-T1', 'CPT2373DU-S1', 'CPT704DU-M1', 'CPT704DU-S1')){
    dat <- readRDS(paste0(sample, '.rds'))
    ucec_stats <- rbind(ucec_stats, data.frame(table(dat@meta.data['cell_type'])))
    print(paste0(sample, ' done'))    
}
ucec_stats <- aggregate(Freq~cell_type, ucec_stats, FUN=function(x) sum(x))
#                                 cell_type  Freq
# 1                                 B-cells    51
# 2                             Endothelial    94
# 3                             Fibroblasts   287
# 4                             Low quality    39
# 5                             Macrophages  1158
# 6                                  Plasma    32
# 7                                 T-cells   532
# 8                                   Tumor 17186
# 9   Ciliated Endometrial epithelial cells   193
# 10 Secretory Endometrial epithelial cells   645
write.table(ucec_stats, 'ucec_stats.txt', row.names=FALSE, col.names=FALSE, quote=FALSE)


############## GBM ##############
## 2 normal samples
gbm_stats <- data.frame()
for (sample in c('GBML018G1-M1N2', 'GBML019G1-M1N1')){
    dat <- readRDS(paste0(sample, '.rds'))
    gbm_stats <- rbind(gbm_stats, data.frame(table(dat@meta.data['cell_type'])))
    print(paste0(sample, ' done'))    
}
gbm_stats <- aggregate(Freq~cell_type, gbm_stats, FUN=function(x) sum(x))
#          cell_type Freq
# 1       Astrocytes  833
# 2      Low quality  415
# 3      Macrophages   16
# 4        Microglia  673
# 5          Neurons  498
# 6 Oligodendrocytes 5378
# 7              OPC  418
# 8        Pericytes  127
# 9          T-cells   11
write.table(gbm_stats, 'gbm_stats.txt', row.names=FALSE, col.names=FALSE, quote=FALSE)


############## PDAC ##############
pdac_stats <- data.frame()
for (sample in c('HT090P1-T2A3',   'HT113P1-T2A3',    'HT181P1-T1A3',  'HT224P1-S1',    'HT231P1-S1H3',
                 'HT232P1-S1H1',   'HT242P1-S1H1',    'HT259P1-S1H1',  'HT264P1-S1H2',  'HT270P1-S1H2',
                 'HT270P2-Th1Fc1', 'HT288P1-S1H4',    'HT306P1-S1H1',  'HT341P1-S1H1',  'HT390P1-S1H1',
                 'HT412P1-S1H1',   'HT447P1-Th1K1A3', 'HT452P1-Th1K1', 'PM1380P1-T1Y2', 'PM1467P1-T1Y2',
                 'PM1532P1-T1Y2',  'PM185P1-T1N2',    'PM439P1-T1N1',  'PM565P1-T1N1',  'PM581P1-T1N1')){
    dat <- readRDS(paste0(sample, '.rds'))
    pdac_stats <- rbind(pdac_stats, data.frame(table(dat@meta.data['cell_type'])))
    print(paste0(sample, ' done'))    
}
pdac_stats <- aggregate(Freq~cell_type, pdac_stats, FUN=function(x) sum(x))
#         cell_type  Freq
# 1     Endothelial  1423
# 2     Fibroblasts  8140
# 3     Hepatocytes  2298
# 4     Low quality   672
# 5     Macrophages 14190
# 6            Mast   501
# 7  Other_doublets  1126
# 8          Plasma  2581
# 9         T-cells  8948
# 10          Tumor 49670
# 11     Adipocytes    41
# 12        B-cells  1293
# 13            ADM   195
# 14   Ductal-like1  4312
# 15   Ductal-like2  1426
# 16         Islets  4543
# 17         Acinar  2810
# 18    Acinar REG+   962
write.table(pdac_stats, 'pdac_stats.txt', row.names=FALSE, col.names=FALSE, quote=FALSE)

############## BRCA ##############
brca_stats <- data.frame()
for (sample in c('HT235B1-S1H1', 'HT243B1-S1H4', 'HT263B1-S1H1', 'HT297B1-S1H1', 'HT305B1-S1H1', 
                 'HT497B1-S1H1', 'HT514B1-S1H3', 'HT545B1-S1H1')){
    dat <- readRDS(paste0(sample, '.rds'))
    brca_stats <- rbind(brca_stats, data.frame(table(dat@meta.data['cell_type'])))
    print(paste0(sample, ' done'))    
}
brca_stats <- aggregate(Freq~cell_type, brca_stats, FUN=function(x) sum(x))
#             cell_type  Freq
# 1             B-cells   744
# 2    Basal progenitor   938
# 3         Endothelial   313
# 4         Fibroblasts  2951
# 5      Luminal mature  1258
# 6         Macrophages  4114
# 7           Pericytes   326
# 8              Plasma   599
# 9             T-cells  2163
# 10              Tumor 28431
# 11                 DC    92
# 12 Luminal progenitor   889
write.table(brca_stats, 'brca_stats.txt', row.names=FALSE, col.names=FALSE, quote=FALSE)

############## BRCA_Basal ##############
brcab_stats <- data.frame()
for (sample in c('HT268B1-Th1H3', 'HT271B1-S1H3', 'HT378B1-S1H1', 'HT378B1-S1H2', 'HT384B1-S1H1', 
                 'HT517B1-S1H1')){
    dat <- readRDS(paste0(sample, '.rds'))
    brcab_stats <- rbind(brcab_stats, data.frame(table(dat@meta.data['cell_type'])))
    print(paste0(sample, ' done'))    
}
brcab_stats <- aggregate(Freq~cell_type, brcab_stats, FUN=function(x) sum(x))
#             cell_type  Freq
# 1             B-cells   435
# 2                  DC    88
# 3         Endothelial   170
# 4         Fibroblasts   430
# 5      Luminal mature   395
# 6         Macrophages  3140
# 7           Pericytes   140
# 8              Plasma  1621
# 9             T-cells  3049
# 10              Tumor 21478
# 11   Basal progenitor   213
# 12 Luminal progenitor   278
write.table(brcab_stats, 'brcab_stats.txt', row.names=FALSE, col.names=FALSE, quote=FALSE)

############## HNSCC ##############
hnscc_stats <- data.frame()
for (sample in c('HTx002S1-S1',    'HTx003S1-S1A1', 'HTx004S1-S1A1', 'HTx006S1-S1', 'HTx007S1-S1',
                 'HTx009S1-S1A1',  'HTx011S1-S1',   'HTx012S1-S1',   'HTx013S1-S1', 'P5216-N1', 
                 'P5216-N2',       'P5296-1N1',     'P5296-1N2',     'P5379-N1',    'P5504-N1', 
                 'P5514-1N1',      'P5532-N1',      'P5539-N1',      'P5576-1N1',   'P5590-N1',
                 'HNSCC_P5794-N1', 'P5797-N1')){
    dat <- readRDS(paste0(sample, '.rds'))
    hnscc_stats <- rbind(hnscc_stats, data.frame(table(dat@meta.data['cell_type'])))
    print(paste0(sample, ' done'))    
}
hnscc_stats <- aggregate(Freq~cell_type, hnscc_stats, FUN=function(x) sum(x))
#          cell_type  Freq
# 1          B-cells   865
# 2               DC   934
# 3      Endothelial  2342
# 4      Fibroblasts  7919
# 5      Low quality   358
# 6      Macrophages  4052
# 7        Pericytes   660
# 8           Plasma  1485
# 9          T-cells  3461
# 10           Tumor 35605
# 11 Skeletal Muscle   174
write.table(hnscc_stats, 'hnscc_stats.txt', row.names=FALSE, col.names=FALSE, quote=FALSE)

############## SKCM ##############
skcm_stats <- data.frame()
for (sample in c('ML080M1-Ty1',  'ML1199M1-Ty1', 'ML1199M1-Ty2', 'ML1232M1-Ty1', 'ML1239M1-Tf1',
                 'ML123M1-Ty1',  'ML1254M1-Ty1', 'ML1294M1-Ty1', 'ML1329M1-S1',  'ML1332M1-Ty1',
                 'ML139M1-Tf1',  'ML1511M1-S1',  'ML1525M1-Ty1', 'ML1526M1-Ta1', 'ML1646M1-S1',
                 'ML1652M1-Ty1', 'ML1704M1-Ty1', 'ML1984M1-Ty1', 'ML225M1-Ty1',  'ML2356M1-S1',
                 'ML499M1-S1',   'SN001H1-Ms1')){
    dat <- readRDS(paste0(sample, '.rds'))
    skcm_stats <- rbind(skcm_stats, data.frame(table(dat@meta.data['cell_type'])))
    print(paste0(sample, ' done'))    
}
skcm_stats <- aggregate(Freq~cell_type, skcm_stats, FUN=function(x) sum(x))
#        cell_type  Freq
# 1        B-cells  2071
# 2             DC   332
# 3    Endothelial   911
# 4    Fibroblasts  2129
# 5    Macrophages 14514
# 6         Plasma  1284
# 7        T-cells  6256
# 8          Tumor 32533
# 9          vSMCs   692
# 10          Mast    84
# 11 Keratinocytes   641
# 12       Unknown   310
# 13   Melanocytes    20
# 14 Smooth muscle     2
write.table(skcm_stats, 'skcm_stats.txt', row.names=FALSE, col.names=FALSE, quote=FALSE)

############## OV ##############
ov_stats <- data.frame()
for (sample in c('VF026V1-S1_1N1', 'VF027V1-S1_1N1', 'VF027V1-S1Y1', 'VF032V1_S1N1', 'VF034V1-T1Y1',
                 'VF035V1_S1N1',   'VF044V1-S1N1',   'VF050V1-S2N1')){
    dat <- readRDS(paste0(sample, '.rds'))
    ov_stats <- rbind(ov_stats, data.frame(table(dat@meta.data['cell_type'])))
    print(paste0(sample, ' done'))    
}
ov_stats <- aggregate(Freq~cell_type, ov_stats, FUN=function(x) sum(x))
#     cell_type  Freq
# 1     B-cells   232
# 2 Endothelial   159
# 3 Fibroblasts  1706
# 4 Macrophages  1500
# 5      Plasma  1839
# 6     T-cells  1042
# 7       Tumor 22511
# 8 Low quality   316
write.table(ov_stats, 'ov_stats.txt', row.names=FALSE, col.names=FALSE, quote=FALSE)

library(data.table)
stats <- rbindlist(list(cead_stats, cesc_stats, crc_stats,   ucec_stats,  gbm_stats,
                        pdac_stats, brca_stats, brcab_stats, hnscc_stats, skcm_stats,
                        ov_stats))
stats <- aggregate(Freq~cell_type, stats, FUN=function(x) sum(x))
write.table(stats, 'all_stats.txt', row.names=FALSE, col.names=FALSE, quote=FALSE)


########## plot stacked bars ##########
# tumor:normal ~3:2
# Low quality & Other_doublets & Unknown

from plotnine import *
import pandas as pd
import numpy as np

dat = pd.read_table('stats_cell_types.txt')
dat.index = dat.iloc[:, 0].values
del dat['Unnamed: 0']
dat = dat.T
dat['cancer_type'] = dat.index
dat = pd.melt(dat, id_vars='cancer_type')
dat.columns = ['cancer_type', 'cell_type', 'ptg']
cell_type_lst = ['Tumor', 'Macrophages', 'T-cells', 'Fibroblasts', 'Plasma', 'B-cells', 'Endothelial', 'Other']
cell_type_lst.reverse()
dat['cell_type'] = pd.Categorical(dat['cell_type'], categories=cell_type_lst)

# color map: https://www.jianshu.com/p/b88a77a609da
p = ggplot(dat, aes(x='cancer_type', y='ptg', fill='cell_type')) + geom_bar(stat='identity', position='stack') + coord_flip() + \
                                                                   xlab('Cancer type') + ylab('Proportion') + scale_fill_brewer(palette="Set2", type='qual', direction=-1) +\
                                                                   scale_y_continuous(limits=[0, 1.01], breaks=np.arange(0, 1.1, 0.2)) + theme_bw()
p.save(filename='cancer_type_cell_type_ptg.pdf', dpi=600, height=4, width=5)


## generate info files (barcode & cell_anno)
# processbar: https://blog.csdn.net/ZaoJewin/article/details/133894474
# library(Seurat)
# library(Signac)

samples <- read.table('rds_samples_id.txt')
pb <- txtProgressBar(min=0, max=129, style=3)

for (i in 1:nrow(samples)){
    s_1 <- samples[i, 1]
    rds <- readRDS(paste0('../', s_1, '.rds'))
    out <- rds@meta.data[c('Original_barcode', 'cell_type')]
    out <- out[!(out$cell_type %in% c('Low quality', 'Unknown', 'Other_doublets')), ]

    s_2 <- samples[i, 2]
    dir.create(s_2)
    write.table(out[c('Original_barcode', 'cell_type')], paste0(s_2, '/', s_2, '_info.txt'), row.names=FALSE, col.names=FALSE, quote=FALSE, sep='\t')
    setTxtProgressBar(pb, i) 
}
close(pb)


## python filter_cells_with_cell_anno.py
# something wrong in rds files
# VF027V1-S1_1N1   VF027V1-S1
# VF027V1-S1Y1     VF027V1-S2
# ->
# VF027V1-S1_1N1   VF027V1-S2
# VF027V1-S1Y1     VF027V1-S1
# i=122 & i=123

import scanpy as sc
import pandas as pd
from tqdm import tqdm

samples = pd.read_table('rds_samples_id.txt', header=None)
for i in tqdm(range(samples.shape[0]), ncols=80):
    s_2= samples[1][i]
    info = pd.read_table(s_2+'/'+s_2+'_info.txt', header=None)
    info.columns = ['barcode', 'cell_anno']
    dat = sc.read_10x_mtx('/fs/home/jiluzhang/2023_nature_LD/'+s_2, gex_only=False)
    out = dat[info['barcode'], :].copy()
    out.obs['cell_anno'] = info['cell_anno'].values
    out.write(s_2+'/'+s_2+'_filtered.h5ad')




## python rna_atac_align.py
import scanpy as sc
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
import pybedtools
from tqdm import tqdm

samples = pd.read_table('rds_samples_id_ov_ucec.txt', header=None)
for s in tqdm(samples[1], ncols=80):
    dat = sc.read_h5ad(s+'/'+s+'_filtered.h5ad')
    
    ########## rna ##########
    rna = dat[:, dat.var['feature_types']=='Gene Expression'].copy()
    rna.X = rna.X.toarray()
    genes = pd.read_table('human_genes.txt', names=['gene_id', 'gene_name'])
    
    rna_genes = rna.var.copy()
    rna_genes['gene_id'] = rna_genes['gene_ids'].map(lambda x: x.split('.')[0])  # delete ensembl id with version number
    rna_exp = pd.concat([rna_genes, pd.DataFrame(rna.X.T, index=rna_genes.index)], axis=1)
    
    X_new = pd.merge(genes, rna_exp, how='left', on='gene_id').iloc[:, 4:].T
    X_new.fillna(value=0, inplace=True)
    
    rna_new = sc.AnnData(X_new.values, obs=rna.obs, var=pd.DataFrame({'gene_ids': genes['gene_id'], 'feature_types': 'Gene Expression'}))  # 5517*38244
    rna_new.var.index = genes['gene_name'].values
    rna_new.X = csr_matrix(rna_new.X)
    rna_new.write(s+'/'+s+'_rna.h5ad')
    
    ########## atac ##########
    atac = dat[:, dat.var['feature_types']=='Peaks'].copy()
    atac.X = atac.X.toarray()
    atac.X[atac.X>0] = 1  # binarization
    
    peaks = pd.DataFrame({'id': atac.var_names})
    peaks['chr'] = peaks['id'].map(lambda x: x.split(':')[0])
    peaks['start'] = peaks['id'].map(lambda x: x.split(':')[1].split('-')[0])
    peaks['end'] = peaks['id'].map(lambda x: x.split(':')[1].split('-')[1])
    peaks.drop(columns='id', inplace=True)
    peaks['idx'] = range(peaks.shape[0])
    
    cCREs = pd.read_table('human_cCREs.bed', names=['chr', 'start', 'end'])
    cCREs['idx'] = range(cCREs.shape[0])
    cCREs_bed = pybedtools.BedTool.from_dataframe(cCREs)
    peaks_bed = pybedtools.BedTool.from_dataframe(peaks)
    idx_map = peaks_bed.intersect(cCREs_bed, wa=True, wb=True).to_dataframe().iloc[:, [3, 7]]
    idx_map.columns = ['peaks_idx', 'cCREs_idx']
    
    m = np.zeros([atac.n_obs, cCREs_bed.to_dataframe().shape[0]], dtype='float32')
    for i in tqdm(range(atac.X.shape[0]), ncols=80, desc='Aligning ATAC peaks'):
        m[i][idx_map[idx_map['peaks_idx'].isin(peaks.iloc[np.nonzero(atac.X[i])]['idx'])]['cCREs_idx']] = 1
    
    atac_new = sc.AnnData(m, obs=atac.obs, var=pd.DataFrame({'gene_ids': cCREs['chr']+':'+cCREs['start'].map(str)+'-'+cCREs['end'].map(str), 'feature_types': 'Peaks'}))
    atac_new.var.index = atac_new.var['gene_ids'].values
    atac_new.X = csr_matrix(atac_new.X)
    atac_new.write(s+'/'+s+'_atac.h5ad')



# data_preprocess.py
#     dataset = PairDataset(rna.X, atac.X, enc_max_len, dec_max_len, rna_num_bin, multiple, atac2rna, is_raw_count)
# ->  dataset = PairDataset(rna, atac, enc_max_len, dec_max_len, rna_num_bin, multiple, atac2rna, is_raw_count)

# utils.py
#     num_one = round(self.atac_len * self.dec_atac_one_rate)
# ->  num_one = round(self.atac_len * 0.5)

# block.py
#     self_and_cross = True
# ->  only_cross_attn = False
#     self.self_and_cross = self_and_cross
# ->  self.only_cross_attn = only_cross_attn
#     def route_args(router, args, depth, cross_attend, self_and_cross):
# ->  def route_args(router, args, depth, cross_attend, only_cross_attn):
#     if cross_attend and self_and_cross:
# ->  if cross_attend and not only_cross_attn:
#     def __init__(self, layers, cross_attend, self_and_cross, args_route = {}):
# ->  def __init__(self, layers, cross_attend, only_cross_attn, args_route = {}):
#     args = route_args(self.args_route, kwargs, len(self.layers), self.cross_attend, self.self_and_cross)
# ->  args = route_args(self.args_route, kwargs, len(self.layers), self.cross_attend, self.only_cross_attn)
#     if self.cross_attend and self.self_and_cross:
# ->  if self.cross_attend and not self.only_cross_attn:
#     elif not self_and_cross:
# ->  elif only_cross_attn:
#     self.net = SequentialSequence(layers, cross_attend, self_and_cross, args_route = {**attn_route_map, **context_route_map})
# ->  self.net = SequentialSequence(layers, cross_attend, only_cross_attn, args_route = {**attn_route_map, **context_route_map})

# use block.py from QH (/data/home/zouqihang/desktop/project/M2M/version1.0.0/M2Mmodel/block.py)

./out_train_dataset_ov_ucec.sh

## training dataset:   CPT1541DU-T1  CPT2373DU-S1  CPT704DU-S1
## validation dataset: CPT704DU-M1
## testing dataset:    VF026V1-S1  VF027V1-S2  VF027V1-S1  VF032V1_S1  VF034V1-T1  VF035V1_S1  VF044V1-S1  VF050V1-S2
nohup accelerate launch --config_file accelerator_config_train.yaml --main_process_port 29823 rna2atac_train.py --config_file rna2atac_config_train.yaml \
                        --train_data_dir ./preprocessed_data_train --val_data_dir ./preprocessed_data_val -n rna2atac_train > 20240623.log & 

python data_preprocess.py -r ../VF026V1-S1/VF026V1-S1_rna.h5ad -a ../VF026V1-S1/VF026V1-S1_atac.h5ad -s ./preprocessed_data_test --dt test --config rna2atac_config_test.yaml

## max count of expressed genes
import scanpy as sc
import numpy as np
rna = sc.read_h5ad('VF026V1-S1_rna.h5ad')
np.count_nonzero(rna.X.toarray(), axis=1).max()
# 6814


accelerate launch --config_file accelerator_config_test.yaml --main_process_port 29822 rna2atac_test.py \
                  -d ./preprocessed_data_test \
                  -l save/2024-07-10_rna2atac_train_20/pytorch_model.bin --config_file rna2atac_config_test.yaml
python npy2h5ad.py
python csr2array.py --pred rna2atac_scm2m_raw.h5ad  --true VF026V1-S1_atac.h5ad
python cal_auroc_auprc.py --pred rna2atac_scm2m.h5ad --true VF026V1-S1_atac.h5ad
python cal_cluster_plot.py --pred rna2atac_scm2m.h5ad --true VF026V1-S1_atac.h5ad


accelerate launch --config_file accelerator_config_test.yaml --main_process_port 29822 rna2atac_test.py \
                  -d ./preprocessed_data_test \
                  -l save/2024-07-11_rna2atac_train_1/pytorch_model.bin --config_file rna2atac_config_test.yaml


## plot for true atac
import snapatac2 as snap
import scanpy as sc

true = snap.read('VF026V1-S1_atac.h5ad', backed=None)
snap.pp.select_features(true)
snap.tl.spectral(true)
snap.tl.umap(true)
sc.pl.umap(true, color='cell_anno', legend_fontsize='7', legend_loc='right margin', size=10,
           title='', frameon=True, save='_VF026V1-S1_atac_true.pdf')





## Attention score matrix
import pandas as pd
import numpy as np
import random
from functools import partial
import sys
sys.path.append("M2Mmodel")
from utils import PairDataset
import scanpy as sc
import torch
from utils import *
from collections import Counter
import pickle as pkl
import yaml
from M2M import M2M_rna2atac

data = torch.load("/fs/home/jiluzhang/scM2M_no_dec_attn/pan_cancer/all_data/data_with_annotation/h5ad/scM2M/preprocessed_data_test/preprocessed_data_0.pt")
# from config
with open("rna2atac_config_train.yaml", "r") as f:
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

model.load_state_dict(torch.load('./save/2024-07-11_rna2atac_train_1/pytorch_model.bin'))

dataset = PreDataset(data)
dataloader_kwargs = {'batch_size': 1, 'shuffle': False}
loader = torch.utils.data.DataLoader(dataset, **dataloader_kwargs)
device = torch.device('cuda:2') #device = select_least_used_gpu()
model.to(device)

rna  = sc.read_h5ad('VF026V1-S1_rna.h5ad')
atac = sc.read_h5ad('VF026V1-S1_atac.h5ad')

###### different tfs in same cell
i = 0
p = 0
#gene_lst = ['PAX8', 'MECOM', 'SOX17', 'WT1']
gene_lst = ['PAX8', 'MECOM', 'ZEB1', 'ZEB2']
zeros = torch.zeros([atac.shape[1], 1])

# ['B-cells', 'Endothelial', 'Fibroblasts', 'Macrophages', 'Plasma', 'T-cells', 'Tumor']

for inputs in loader:
    if rna.obs.cell_anno.values[p]=='Tumor':
        rna_sequence, rna_value, atac_sequence, _, enc_pad_mask = [each.to(device) for each in inputs]
        attn = model.generate_attn_weight(rna_sequence, atac_sequence, rna_value, enc_mask=enc_pad_mask, which='decoder')
        attn = (attn[0]-attn[0].min())/(attn[0].max()-attn[0].min())
        
        dat = torch.Tensor()
        for g in gene_lst:
            idx = torch.argwhere(rna_sequence[0]==(np.argwhere(rna.var.index==g)[0][0]+1))
            if len(idx)!=0:
                dat = torch.cat([dat, attn[:, [idx.item()]]], axis=1)
            else:
                dat = torch.cat([dat, zeros], axis=1)
        
        dat = pd.DataFrame(dat)
        dat.columns = gene_lst
        plt.figure(figsize=(5, 5))
        sns.heatmap(dat, cmap='Reds', vmin=0, vmax=0.01, yticklabels=False)
        plt.savefig('cell_'+str(i)+'_Tumor.png')
        plt.close()
        
        i += 1
        torch.cuda.empty_cache()
        print(str(i), 'cell done')
    
    if i==3:
        break

    p += 1



i = 0
p = 0
gene_lst = rna.var.iloc[np.argwhere(rna.X[0].toarray()[0]!=0).flatten()].index
zeros = torch.zeros([atac.shape[1], 1])

from tqdm import tqdm
for inputs in loader:
    if rna.obs.cell_anno.values[p]=='Tumor':
        rna_sequence, rna_value, atac_sequence, _, enc_pad_mask = [each.to(device) for each in inputs]
        attn = model.generate_attn_weight(rna_sequence, atac_sequence, rna_value, enc_mask=enc_pad_mask, which='decoder')
        attn = (attn[0]-attn[0].min())/(attn[0].max()-attn[0].min())
        
        dat = torch.Tensor()
        for g in tqdm(gene_lst, ncols=80):
            idx = torch.argwhere(rna_sequence[0]==(np.argwhere(rna.var.index==g)[0][0]+1))
            if len(idx)!=0:
                dat = torch.cat([dat, attn[:, [idx.item()]]], axis=1)
            else:
                dat = torch.cat([dat, zeros], axis=1)
    torch.cuda.empty_cache()
    break

        import random
        random.seed(0)
        rad_idx = random.sample(list(range(atac.shape[1])), k=10000)
        df = pd.DataFrame(dat[rad_idx, :][:, ([366, 698]+list(range(200)))])
        df.columns = gene_lst[[366, 698]+list(range(200))]
        # df_rank = df.rank(axis=0)

        from sklearn.metrics.pairwise import cosine_similarity
        cosine_similarity(df['PAX8'].values.reshape(1, -1), df['MECOM'].values.reshape(1, -1)).item()  # 0.038373351097106934
        pax8_lst = [cosine_similarity(df['PAX8'].values.reshape(1, -1), df.iloc[:, i].values.reshape(1, -1)).item() for i in range(100)]

        from scipy.stats import pearsonr
        pearsonr(df['PAX8'], df['MECOM'])[0]  # 0.018551660027068803
        pax8_lst = [pearsonr(df['PAX8'], df.iloc[:, i])[0] for i in range(100)]
        sum(np.array(pax8_lst)>pearsonr(df['PAX8'], df['MECOM'])[0])  # 2

        # from sklearn.cluster import KMeans
        # kmeans = KMeans(n_clusters=2)
        # clusters = kmeans.fit_predict(df)
        #plt.figure(figsize=(20, 5))
        sns.clustermap(df, metric='correlation', cmap='Reds', row_cluster=False, vmin=0, vmax=0.001, yticklabels=False, figsize=[40, 5])  #sns.heatmap(df, cmap='Reds', vmin=0, vmax=0.01, yticklabels=False)
        plt.savefig('tmp.png')
        plt.close()
        
        i += 1
        torch.cuda.empty_cache()
        print(str(i), 'cell done')
    
    if i==3:
        break

    p += 1





###### same tfs in different cell
i = 0
p = 0
dat = torch.Tensor()
zeros = torch.zeros([atac.shape[1], 1])
for inputs in loader:
    if rna.obs.cell_anno.values[p] == 'Tumor':
        rna_sequence, rna_value, atac_sequence, _, enc_pad_mask = [each.to(device) for each in inputs]
        attn = model.generate_attn_weight(rna_sequence, atac_sequence, rna_value, enc_mask=enc_pad_mask, which='decoder')
        attn = (attn[0]-attn[0].min())/(attn[0].max()-attn[0].min())
        
        idx = torch.argwhere(rna_sequence[0]==(np.argwhere(rna.var.index=='MECOM')[0][0]+1))
        if len(idx)!=0:
            dat = torch.cat([dat, attn[:, [idx.item()]]], axis=1)
        else:
            dat = torch.cat([dat, zeros], axis=1)
        
        i += 1
        torch.cuda.empty_cache()
        print(str(i), 'cell done')
    
    if i==20:
        break

    p += 1


dat = pd.DataFrame(dat)
dat.columns = ['cell_'+str(i) for i in range(6)]
plt.figure(figsize=(5, 5))
sns.heatmap(dat, cmap='Reds', vmin=0, vmax=0.01, yticklabels=False)
#sns.heatmap(np.log(dat), cmap='Reds', vmin=-10, vmax=0, yticklabels=False)
plt.savefig('Tumor_PAX8_2.png')
plt.close()


## PAX8 in tumor cells
# liftOver pax8_hg19.bed hg19ToHg38.over.chain.gz pax18_hg38.bed unmapped.bed

dat = pd.DataFrame(dat)
dat.index = atac.var.index.values
dat_agg = dat.sum(axis=1)

out_lst = dat_agg.nlargest(30000).index
chrom = out_lst.map(lambda x: x.split(':')[0])
start = out_lst.map(lambda x: x.split(':')[1].split('-')[0])
end = out_lst.map(lambda x: x.split(':')[1].split('-')[1])
out = pd.DataFrame({'chrom':chrom, 'start':start, 'end':end})
out.to_csv('PAX8/pax8_predicted_peaks_30000.bed', index=False, header=False, sep='\t')

awk '{print $1 "\t" $2-500 "\t" $2+500}' pax8_predicted_peaks_5000.bed | bedtools intersect -a stdin -b pax8_hg38.bed -wa | sort | uniq | wc -l  # 379 (379/5000=0.0758)
awk '{print $1 "\t" $2-500 "\t" $2+500}' pax8_predicted_peaks_10000.bed | bedtools intersect -a stdin -b pax8_hg38.bed -wa | sort | uniq | wc -l  # 748 (748/10000=0.0748)
awk '{print $1 "\t" $2-500 "\t" $2+500}' pax8_predicted_peaks_20000.bed | bedtools intersect -a stdin -b pax8_hg38.bed -wa | sort | uniq | wc -l  # 1419 (1419/20000=0.07095)
awk '{print $1 "\t" $2-500 "\t" $2+500}' pax8_predicted_peaks_30000.bed | bedtools intersect -a stdin -b pax8_hg38.bed -wa | sort | uniq | wc -l  # 2051 (2051/30000=0.06837)

# bedtools intersect -a pax8_hg38.bed -b human_cCREs.bed -wa | sort | uniq | wc -l  # 21220

## randomly select cCREs
for i in `seq 10`;do shuf human_cCREs.bed | head -n 5000 | awk '{print $1 "\t" $2-500 "\t" $2+500}' | bedtools intersect -a stdin -b pax8_hg38.bed -wa | sort | uniq | wc -l; done
# (297+344+296+313+332+305+325+289+275+313)/10/5000=0.0618
for i in `seq 10`;do shuf human_cCREs.bed | head -n 10000 | awk '{print $1 "\t" $2-500 "\t" $2+500}' | bedtools intersect -a stdin -b pax8_hg38.bed -wa | sort | uniq | wc -l; done
# (651+619+667+604+622+631+626+659+627+601)/10/10000=0.06307
for i in `seq 10`;do shuf human_cCREs.bed | head -n 20000 | awk '{print $1 "\t" $2-500 "\t" $2+500}' | bedtools intersect -a stdin -b pax8_hg38.bed -wa | sort | uniq | wc -l; done
# (1279+1273+1295+1253+1300+1234+1229+1314+1227+1229)/10/20000=0.063165
for i in `seq 10`;do shuf human_cCREs.bed | head -n 30000 | awk '{print $1 "\t" $2-500 "\t" $2+500}' | bedtools intersect -a stdin -b pax8_hg38.bed -wa | sort | uniq | wc -l; done
# (1922+1881+1842+1928+1838+1835+1932+1926+1837+1899)/10/30000=0.0628

for i in `seq 1000`;do shuf human_cCREs.bed | head -n 5000 | awk '{print $1 "\t" $2-500 "\t" $2+500}' | bedtools intersect -a stdin -b pax8_hg38.bed -wa | sort | uniq | wc -l >> random_5000.txt; done
awk '{if($1>379) print $0}' random_5000.txt | wc -l  # 0
for i in `seq 1000`;do shuf human_cCREs.bed | head -n 10000 | awk '{print $1 "\t" $2-500 "\t" $2+500}' | bedtools intersect -a stdin -b pax8_hg38.bed -wa | sort | uniq | wc -l >> random_10000.txt; done
awk '{if($1>748) print $0}' random_10000.txt | wc -l  # 0
for i in `seq 1000`;do shuf human_cCREs.bed | head -n 20000 | awk '{print $1 "\t" $2-500 "\t" $2+500}' | bedtools intersect -a stdin -b pax8_hg38.bed -wa | sort | uniq | wc -l >> random_20000.txt; done
awk '{if($1>1419) print $0}' random_20000.txt | wc -l  # 0
for i in `seq 1000`;do shuf human_cCREs.bed | head -n 30000 | awk '{print $1 "\t" $2-500 "\t" $2+500}' | bedtools intersect -a stdin -b pax8_hg38.bed -wa | sort | uniq | wc -l >> random_30000.txt; done
awk '{if($1>2051) print $0}' random_30000.txt | wc -l  # 0


## MECOM in tumor cells
# liftOver mecom_hg19.bed hg19ToHg38.over.chain.gz mecom_hg38.bed unmapped.bed

out_lst = dat_agg.nlargest(30000).index
chrom = out_lst.map(lambda x: x.split(':')[0])
start = out_lst.map(lambda x: x.split(':')[1].split('-')[0])
end = out_lst.map(lambda x: x.split(':')[1].split('-')[1])
out = pd.DataFrame({'chrom':chrom, 'start':start, 'end':end})
out.to_csv('MECOM/mecom_predicted_peaks_30000.bed', index=False, header=False, sep='\t')

awk '{print $1 "\t" $2-500 "\t" $2+500}' mecom_predicted_peaks_5000.bed | bedtools intersect -a stdin -b mecom_hg38.bed -wa | sort | uniq | wc -l  # 78 (78/5000=0.0156)
awk '{print $1 "\t" $2-500 "\t" $2+500}' mecom_predicted_peaks_10000.bed | bedtools intersect -a stdin -b mecom_hg38.bed -wa | sort | uniq | wc -l  # 190 (190/10000=0.019)
awk '{print $1 "\t" $2-500 "\t" $2+500}' mecom_predicted_peaks_20000.bed | bedtools intersect -a stdin -b mecom_hg38.bed -wa | sort | uniq | wc -l  # 422 (422/20000=0.0211)
awk '{print $1 "\t" $2-500 "\t" $2+500}' mecom_predicted_peaks_30000.bed | bedtools intersect -a stdin -b mecom_hg38.bed -wa | sort | uniq | wc -l  # 583 (583/30000=0.01943)

## randomly select cCREs
for i in `seq 10`;do shuf human_cCREs.bed | head -n 5000 | awk '{print $1 "\t" $2-500 "\t" $2+500}' | bedtools intersect -a stdin -b mecom_hg38.bed -wa | sort | uniq | wc -l; done
# (82+89+87+90+99+102+96+77+108+77)/10/5000=0.01814
for i in `seq 10`;do shuf human_cCREs.bed | head -n 10000 | awk '{print $1 "\t" $2-500 "\t" $2+500}' | bedtools intersect -a stdin -b mecom_hg38.bed -wa | sort | uniq | wc -l; done
# (151+161+166+168+155+175+170+157+148+144)/10/10000=0.01595
for i in `seq 10`;do shuf human_cCREs.bed | head -n 20000 | awk '{print $1 "\t" $2-500 "\t" $2+500}' | bedtools intersect -a stdin -b mecom_hg38.bed -wa | sort | uniq | wc -l; done
# (353+336+301+308+325+311+334+339+339+332)/10/20000=0.01639
for i in `seq 10`;do shuf human_cCREs.bed | head -n 30000 | awk '{print $1 "\t" $2-500 "\t" $2+500}' | bedtools intersect -a stdin -b mecom_hg38.bed -wa | sort | uniq | wc -l; done
# (524+500+485+502+487+459+500+504+484+512)/10/30000=0.01652

for i in `seq 1000`;do shuf human_cCREs.bed | head -n 5000 | awk '{print $1 "\t" $2-500 "\t" $2+500}' | bedtools intersect -a stdin -b mecom_hg38.bed -wa | sort | uniq | wc -l >> random_5000.txt; done
awk '{if($1>78) print $0}' random_5000.txt | wc -l  # 673
for i in `seq 1000`;do shuf human_cCREs.bed | head -n 10000 | awk '{print $1 "\t" $2-500 "\t" $2+500}' | bedtools intersect -a stdin -b mecom_hg38.bed -wa | sort | uniq | wc -l >> random_10000.txt; done
awk '{if($1>190) print $0}' random_10000.txt | wc -l  # 29
for i in `seq 1000`;do shuf human_cCREs.bed | head -n 20000 | awk '{print $1 "\t" $2-500 "\t" $2+500}' | bedtools intersect -a stdin -b mecom_hg38.bed -wa | sort | uniq | wc -l >> random_20000.txt; done
awk '{if($1>422) print $0}' random_20000.txt | wc -l  # 0
for i in `seq 1000`;do shuf human_cCREs.bed | head -n 30000 | awk '{print $1 "\t" $2-500 "\t" $2+500}' | bedtools intersect -a stdin -b mecom_hg38.bed -wa | sort | uniq | wc -l >> random_30000.txt; done
awk '{if($1>583) print $0}' random_30000.txt | wc -l  # 0




## Generate bedgraph files
import scanpy as sc
from tqdm import tqdm
import pandas as pd

true = sc.read_h5ad('VF026V1-S1_atac.h5ad')
for i in tqdm(range(10), ncols=80):
    chrom = true.var.index.map(lambda x: x.split(':')[0])
    start = true.var.index.map(lambda x: x.split(':')[1].split('-')[0])
    end = true.var.index.map(lambda x: x.split(':')[1].split('-')[1])
    val = true.X[i].toarray()[0]
    out = pd.DataFrame({'chrom':chrom, 'start':start, 'end':end, 'val':val})
    out.to_csv('Tracks/tumor_cell_'+str(i)+'_true.bedgraph', index=False, header=False, sep='\t')

pred = sc.read_h5ad('rna2atac_scm2m_binary_epoch_1.h5ad')
for i in tqdm(range(10), ncols=80):
    chrom = pred.var.index.map(lambda x: x.split(':')[0])
    start = pred.var.index.map(lambda x: x.split(':')[1].split('-')[0])
    end = pred.var.index.map(lambda x: x.split(':')[1].split('-')[1])
    val = pred.X[i].toarray()[0]
    out = pd.DataFrame({'chrom':chrom, 'start':start, 'end':end, 'val':val})
    out.to_csv('Tracks/tumor_cell_'+str(i)+'_pred.bedgraph', index=False, header=False, sep='\t')

pred = sc.read_h5ad('rna2atac_scm2m_epoch_1.h5ad')
for i in tqdm(range(10), ncols=80):
    chrom = pred.var.index.map(lambda x: x.split(':')[0])
    start = pred.var.index.map(lambda x: x.split(':')[1].split('-')[0])
    end = pred.var.index.map(lambda x: x.split(':')[1].split('-')[1])
    val = pred.X[i]
    out = pd.DataFrame({'chrom':chrom, 'start':start, 'end':end, 'val':val})
    out.to_csv('Tracks/tumor_cell_'+str(i)+'_pred_prob.bedgraph', index=False, header=False, sep='\t')


true = sc.read_h5ad('VF026V1-S1_atac.h5ad')
chrom = true.var.index.map(lambda x: x.split(':')[0])
start = true.var.index.map(lambda x: x.split(':')[1].split('-')[0])
end = true.var.index.map(lambda x: x.split(':')[1].split('-')[1])
val = true.X[true.obs.cell_anno=='Tumor'].toarray().mean(axis=0)
out = pd.DataFrame({'chrom':chrom, 'start':start, 'end':end, 'val':val})
out.to_csv('Tracks/tumor_cell_true.bedgraph', index=False, header=False, sep='\t')
val = true.X[true.obs.cell_anno=='Fibroblasts'].toarray().mean(axis=0)
out = pd.DataFrame({'chrom':chrom, 'start':start, 'end':end, 'val':val})
out.to_csv('Tracks/fibroblasts_true.bedgraph', index=False, header=False, sep='\t')

pred = sc.read_h5ad('rna2atac_scm2m_epoch_1.h5ad')
pred.X[pred.X>0.6]=1
pred.X[pred.X<=0.6]=0
chrom = pred.var.index.map(lambda x: x.split(':')[0])
start = pred.var.index.map(lambda x: x.split(':')[1].split('-')[0])
end = pred.var.index.map(lambda x: x.split(':')[1].split('-')[1])
val = pred.X[pred.obs.cell_anno=='Tumor'].mean(axis=0)
out = pd.DataFrame({'chrom':chrom, 'start':start, 'end':end, 'val':val})
out.to_csv('Tracks/tumor_cell_pred_0.6.bedgraph', index=False, header=False, sep='\t')
val = pred.X[pred.obs.cell_anno=='Fibroblasts'].mean(axis=0)
out = pd.DataFrame({'chrom':chrom, 'start':start, 'end':end, 'val':val})
out.to_csv('Tracks/fibroblasts_pred_0.6.bedgraph', index=False, header=False, sep='\t')


pred = sc.read_h5ad('rna2atac_scm2m_binary_epoch_1.h5ad')
chrom = pred.var.index.map(lambda x: x.split(':')[0])
start = pred.var.index.map(lambda x: x.split(':')[1].split('-')[0])
end = pred.var.index.map(lambda x: x.split(':')[1].split('-')[1])
val = pred.X.toarray()[pred.obs.cell_anno=='Tumor'].mean(axis=0)
out = pd.DataFrame({'chrom':chrom, 'start':start, 'end':end, 'val':val})
out.to_csv('Tracks/tumor_cell_pred_binary.bedgraph', index=False, header=False, sep='\t')
val = pred.X.toarray()[pred.obs.cell_anno=='Fibroblasts'].mean(axis=0)
out = pd.DataFrame({'chrom':chrom, 'start':start, 'end':end, 'val':val})
out.to_csv('Tracks/fibroblasts_pred_binary.bedgraph', index=False, header=False, sep='\t')


pred = sc.read_h5ad('rna2atac_scm2m_binary_epoch_1.h5ad')
chrom = pred.var.index.map(lambda x: x.split(':')[0])
start = pred.var.index.map(lambda x: x.split(':')[1].split('-')[0])
end = pred.var.index.map(lambda x: x.split(':')[1].split('-')[1])
val_tumor = pred.X.toarray()[pred.obs.cell_anno=='Tumor'].mean(axis=0)
val_fibro = pred.X.toarray()[pred.obs.cell_anno=='Fibroblasts'].mean(axis=0)
out = pd.DataFrame({'chrom':chrom, 'start':start, 'end':end, 'val_tumor':val_tumor, 'val_fibro':val_fibro})
out[(out['val_tumor']/out['val_fibro']>3) & (out['val_tumor']>0.97)]

## plot gene expression box
rna = sc.read_h5ad('VF026V1-S1_rna.h5ad')
sc.pp.normalize_total(rna, target_sum=1e4)
sc.pp.log1p(rna)
sc.pl.violin(rna[rna.obs.cell_anno.isin(['Tumor', 'Fibroblasts'])], keys=['PAX8', 'MECOM', 'ZEB2', 'COL1A1'], groupby="cell_anno", 
             save='_PAX8_MECOM_ZEB2_COL1A1.pdf')



## predict using model with more epoch (similar with epoch_1)
accelerate launch --config_file accelerator_config_test.yaml --main_process_port 29822 rna2atac_test.py \
                  -d ./preprocessed_data_test \
                  -l save/2024-07-11_rna2atac_train_5/pytorch_model.bin --config_file rna2atac_config_test.yaml
python npy2h5ad.py
python csr2array.py --pred rna2atac_scm2m_raw.h5ad  --true VF026V1-S1_atac.h5ad
#python cal_auroc_auprc.py --pred rna2atac_scm2m.h5ad --true VF026V1-S1_atac.h5ad
#python cal_cluster_plot.py --pred rna2atac_scm2m.h5ad --true VF026V1-S1_atac.h5ad



############ motif enrichment analysis ############
######## True ########
import snapatac2 as snap
import scanpy as sc
import numpy as np

atac = snap.read('VF026V1-S1_atac.h5ad', backed=None)
marker_peaks = snap.tl.marker_regions(atac, groupby='cell_anno', pvalue=0.05)
motifs = snap.tl.motif_enrichment(motifs=snap.datasets.cis_bp(unique=True),
                                  regions=marker_peaks,
                                  genome_fasta=snap.genome.hg38,
                                  background=None,
                                  method='hypergeometric')

for c in atac.obs.cell_anno:
    motifs_sp = motifs[c].to_pandas()
    del motifs_sp['family']
    motifs_sp.to_csv('Motifs/'+c+'_sp_true_motif.txt', sep='\t', index=False)


## Plot top enriched motifs
import pandas as pd
from plotnine import *
import numpy as np

## tumor cells specific
motifs_tumor_sp = pd.read_table('Tumor_sp_true_motif.txt')
motifs_tumor_sp.index = motifs_tumor_sp['name'].values
motifs_tumor_sp = motifs_tumor_sp.sort_values('log2(fold change)', ascending=False)
tfs = list(motifs_tumor_sp[motifs_tumor_sp['p-value']<0.05][:10].index.values)
tfs.reverse()
motifs_tumor_sp_stat = motifs_tumor_sp.loc[tfs]
motifs_tumor_sp_stat['name'] = pd.Categorical(motifs_tumor_sp_stat['name'], categories=tfs)
dat = motifs_tumor_sp_stat[['name', 'log2(fold change)', 'p-value']]
dat['-log(p-value)'] = -np.log(dat['p-value'])

p = ggplot(dat, aes(x='name', y='log2(fold change)', fill='-log(p-value)')) + geom_bar(stat='identity', position=position_dodge()) + coord_flip() + \
                                                        scale_fill_gradient(low='lightcoral', high='darkred', limits=[3, 21]) + xlab('Motif') + ylab('log2FC') + labs(title='Tumor cell specific')  + \
                                                        scale_y_continuous(limits=[0, 0.5], breaks=np.arange(0, 0.51, 0.1)) + theme_bw() + theme(plot_title=element_text(hjust=0.5))
p.save(filename='Tumor_sp_true_motif.pdf', dpi=600, height=4, width=5)

## fibroblasts specific
motifs_fibro_sp = pd.read_table('Fibroblasts_sp_true_motif.txt')
motifs_fibro_sp.index = motifs_fibro_sp['name'].values
motifs_fibro_sp = motifs_fibro_sp.sort_values('log2(fold change)', ascending=False)
tfs = list(motifs_fibro_sp[motifs_fibro_sp['p-value']<0.05][:11].index.values)
tfs.reverse()
motifs_fibro_sp_stat = motifs_fibro_sp.loc[tfs]
motifs_fibro_sp_stat['name'] = pd.Categorical(motifs_fibro_sp_stat['name'], categories=tfs)
dat = motifs_fibro_sp_stat[['name', 'log2(fold change)', 'p-value']]
dat['-log(p-value)'] = -np.log(dat['p-value'])

p = ggplot(dat, aes(x='name', y='log2(fold change)', fill='-log(p-value)')) + geom_bar(stat='identity', position=position_dodge()) + coord_flip() + \
                                                        scale_fill_gradient(low='lightblue', high='darkblue', limits=[3, 8]) + xlab('Motif') + ylab('log2FC') + labs(title='Fibroblast specific') + \
                                                        scale_y_continuous(limits=[0, 1], breaks=np.arange(0, 1.1, 0.2)) + theme_bw() + theme(plot_title=element_text(hjust=0.5))
p.save(filename='Fibroblasts_sp_true_motif.pdf', dpi=600, height=4, width=5)



######## Pred ########
import snapatac2 as snap
import scanpy as sc
import numpy as np
import random

atac = snap.read('rna2atac_scm2m_binary_epoch_1.h5ad', backed=None)
marker_peaks = snap.tl.marker_regions(atac, groupby='cell_anno', pvalue=0.05)
motifs = snap.tl.motif_enrichment(motifs=snap.datasets.cis_bp(unique=True),
                                 regions=marker_peaks,
                                 genome_fasta=snap.genome.hg38,
                                 background=None,
                                 method='hypergeometric')

for c in atac.obs.cell_anno:
    motifs_sp = motifs[c].to_pandas()
    del motifs_sp['family']
    motifs_sp.to_csv('Motifs/'+c+'_sp_pred_motif.txt', sep='\t', index=False)


## Plot top enriched motifs
import pandas as pd
from plotnine import *
import numpy as np

## tumor cells specific
motifs_tumor_sp = pd.read_table('Tumor_sp_pred_motif.txt')
motifs_tumor_sp.index = motifs_tumor_sp['name'].values
motifs_tumor_sp = motifs_tumor_sp.sort_values('log2(fold change)', ascending=False)
tfs = list(motifs_tumor_sp[motifs_tumor_sp['p-value']<0.05][:10].index.values)
tfs.reverse()
motifs_tumor_sp_stat = motifs_tumor_sp.loc[tfs]
motifs_tumor_sp_stat['name'] = pd.Categorical(motifs_tumor_sp_stat['name'], categories=tfs)
dat = motifs_tumor_sp_stat[['name', 'log2(fold change)', 'p-value']]
dat['-log(p-value)'] = -np.log(dat['p-value'])
dat.loc['ZBED1', '-log(p-value)'] = 10

p = ggplot(dat, aes(x='name', y='log2(fold change)', fill='-log(p-value)')) + geom_bar(stat='identity', position=position_dodge()) + coord_flip() + \
                                                        scale_fill_gradient(low='lightcoral', high='darkred', limits=[3, 10]) + xlab('Motif') + ylab('log2FC') + labs(title='Tumor cell specific')  + \
                                                        scale_y_continuous(limits=[0, 1], breaks=np.arange(0, 1.1, 0.2)) + theme_bw() + theme(plot_title=element_text(hjust=0.5))
p.save(filename='Tumor_sp_pred_motif.pdf', dpi=600, height=4, width=5)


## fibroblasts
motifs_fibro_sp = pd.read_table('Fibroblasts_sp_pred_motif.txt')
motifs_fibro_sp.index = motifs_fibro_sp['name'].values
motifs_fibro_sp = motifs_fibro_sp.sort_values('log2(fold change)', ascending=False)
tfs = list(motifs_fibro_sp[motifs_fibro_sp['p-value']<0.05][:10].index.values)
tfs.reverse()
motifs_fibro_sp_stat = motifs_fibro_sp.loc[tfs]
motifs_fibro_sp_stat['name'] = pd.Categorical(motifs_fibro_sp_stat['name'], categories=tfs)
dat = motifs_fibro_sp_stat[['name', 'log2(fold change)', 'p-value']]
dat['-log(p-value)'] = -np.log(dat['p-value'])

p = ggplot(dat, aes(x='name', y='log2(fold change)', fill='-log(p-value)')) + geom_bar(stat='identity', position=position_dodge()) + coord_flip() + \
                                                        scale_fill_gradient(low='lightblue', high='darkblue', limits=[3, 8]) + xlab('Motif') + ylab('log2FC') + labs(title='Fibroblast specific') + \
                                                        scale_y_continuous(limits=[0, 1.8], breaks=np.arange(0, 1.81, 0.3)) + theme_bw() + theme(plot_title=element_text(hjust=0.5))
p.save(filename='Fibroblasts_sp_pred_motif.pdf', dpi=600, height=4, width=5)


## Comparison between predicted and true results
# total 1165 TFs
import pandas as pd
import numpy as np

pred_tumor = pd.read_table('Tumor_sp_pred_motif.txt')
pred_tumor.index = pred_tumor['name'].values
sum((pred_tumor['p-value']<0.05) & (pred_tumor['log2(fold change)']>0.1))
pred_tumor_tf = pred_tumor[(pred_tumor['p-value']<0.05) & (pred_tumor['log2(fold change)']>0.1)]['name'].values

true_tumor = pd.read_table('Tumor_sp_true_motif.txt')
true_tumor.index = true_tumor['name'].values
sum((true_tumor['p-value']<0.05) & (true_tumor['log2(fold change)']>0.1)) 
true_tumor_tf = true_tumor[(true_tumor['p-value']<0.05) & (true_tumor['log2(fold change)']>0.1)]['name'].values

pred_fibro = pd.read_table('Fibroblasts_sp_pred_motif.txt')
pred_fibro.index = pred_fibro['name'].values
sum((pred_fibro['p-value']<0.05) & (pred_fibro['log2(fold change)']>0.1))
pred_fibro_tf = pred_fibro[(pred_fibro['p-value']<0.05) & (pred_fibro['log2(fold change)']>0.1)]['name'].values

true_fibro = pd.read_table('Fibroblasts_sp_true_motif.txt')
true_fibro.index = true_fibro['name'].values
sum((true_fibro['p-value']<0.05) & (true_fibro['log2(fold change)']>0.1)) 
true_fibro_tf = true_fibro[(true_fibro['p-value']<0.05) & (true_fibro['log2(fold change)']>0.1)]['name'].values

#np.intersect1d(pred_tumor_tf, true_tumor_tf).shape[0]
#np.intersect1d(pred_fibro_tf, true_fibro_tf).shape[0]

import matplotlib.pyplot as plt
from matplotlib_venn import venn2
plt.figure()
venn2(subsets = [set(pred_tumor_tf), set(true_tumor_tf)], set_labels=('Pred','True'), set_colors=('darkred','midnightblue'))
plt.savefig('pred_tumor_true_tumor_tf_venn.pdf')
plt.close()

plt.figure()
venn2(subsets = [set(pred_fibro_tf), set(true_fibro_tf)], set_labels=('Pred','True'), set_colors=('darkred','midnightblue'))
plt.savefig('pred_fibro_true_fibro_tf_venn.pdf')


## Plot scatter (correlation)
from plotnine import *
import pandas as pd
import numpy as np
from scipy.stats import pearsonr

## tumor cells
true_tumor = pd.read_table('Tumor_sp_true_motif.txt')
pred_tumor = pd.read_table('Tumor_sp_pred_motif.txt')
dat = pd.DataFrame({'True': true_tumor['log2(fold change)'],
                    'Pred': pred_tumor['log2(fold change)']})

p = ggplot(dat, aes(x='True', y='Pred')) + geom_point(size=0.02) + \
                                           scale_x_continuous(limits=[-1, 1], breaks=np.arange(-1, 1.1, 0.5)) + \
                                           scale_y_continuous(limits=[-1, 1], breaks=np.arange(-1, 1.1, 0.5))+ theme_bw()
p.save(filename='Tumor_true_pred_scatter.png', dpi=100, height=4, width=4)

pearsonr(dat['True'], dat['Pred'])  # PearsonRResult(statistic=0.29477783404747027, pvalue=8.685860568843931e-25)

## fibroblasts
true_fibro = pd.read_table('Fibroblasts_sp_true_motif.txt')
pred_fibro = pd.read_table('Fibroblasts_sp_pred_motif.txt')
dat = pd.DataFrame({'True': true_fibro['log2(fold change)'],
                    'Pred': pred_fibro['log2(fold change)']})
dat = dat[np.isfinite(dat.T).all()]

p = ggplot(dat, aes(x='True', y='Pred')) + geom_point(size=0.02) + \
                                           scale_x_continuous(limits=[-1.5, 1.5], breaks=np.arange(-1.5, 1.6, 0.5)) + \
                                           scale_y_continuous(limits=[-1.5, 1.5], breaks=np.arange(-1.5, 1.6, 0.5))+ theme_bw()
p.save(filename='Fibroblasts_true_pred_scatter.png', dpi=100, height=4, width=4)

pearsonr(dat['True'], dat['Pred'])  # PearsonRResult(statistic=0.0751142646604209, pvalue=0.010694796657114255)



## pychromVAR
# pip install pychromVAR
# pip install pyjaspar
# https://github.com/pinellolab/pychromVAR
# preprocessing.py: "reads_per_peak = np.log10(np.sum(adata.X, axis=0))" -> "reads_per_peak = np.log1p(adata.X.sum(axis=0)) / np.log(10)"
# import pychromvar as pc
# import snapatac2 as snap
# import scanpy as sc

# atac = snap.read('rna2atac_scm2m_epoch_1.h5ad', backed=None)[:, 800:900].copy()
# atac.X[atac.X>0.9]=1
# atac.X[atac.X<=0.9]=0

# atac_tumor = atac[atac.obs.cell_anno=='Tumor'].copy()
# atac_fibro = atac[atac.obs.cell_anno=='Fibroblasts'].copy()

# snap.pp.select_features(atac_tumor)  # 11857
# snap.pp.select_features(atac_fibro)  # 10245

# pc.add_peak_seq(atac, genome_file="Motifs/hg38.fa", delimiter=":|-")
# pc.add_gc_bias(atac)
# pc.get_bg_peaks(atac)

# jdb_obj = jaspardb(release='JASPAR2024')
# motifs = jdb_obj.fetch_motifs(collection='CORE', tax_group=['vertebrates'])

# pc.match_motif(atac, motifs=motifs, p_value=1)
# dev = pc.compute_deviations(atac)  # treat motif score as expression value & differential analysis



## Attention based chromatin accessibility regulator screening & ranking
import pandas as pd
import numpy as np
import random
from functools import partial
import sys
sys.path.append("M2Mmodel")
from utils import PairDataset
import scanpy as sc
import torch
from utils import *
from collections import Counter
import pickle as pkl
import yaml
from M2M import M2M_rna2atac

import h5py
from tqdm import tqdm

data = torch.load("/fs/home/jiluzhang/scM2M_no_dec_attn/pan_cancer/all_data/data_with_annotation/h5ad/scM2M/preprocessed_data_test/preprocessed_data_0.pt")
# from config
with open("rna2atac_config_train.yaml", "r") as f:
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

model.load_state_dict(torch.load('./save/2024-07-11_rna2atac_train_1/pytorch_model.bin'))

rna  = sc.read_h5ad('VF026V1-S1_rna.h5ad')
atac = sc.read_h5ad('VF026V1-S1_atac.h5ad')

## tumor cells
tumor_idx = np.argwhere(rna.obs.cell_anno=='Tumor')[:20].flatten()
tumor_data = []
for i in range(len(data)):
    tumor_data.append(data[i][tumor_idx])

dataset = PreDataset(tumor_data)
dataloader_kwargs = {'batch_size': 1, 'shuffle': False}
loader = torch.utils.data.DataLoader(dataset, **dataloader_kwargs)
device = select_least_used_gpu()
model.to(device)

###### same tfs in different cell
i = 0
for inputs in loader:
    rna_sequence, rna_value, atac_sequence, _, enc_pad_mask = [each.to(device) for each in inputs]
    attn = model.generate_attn_weight(rna_sequence, atac_sequence, rna_value, enc_mask=enc_pad_mask, which='decoder')
    attn = (attn[0]-attn[0].min())/(attn[0].max()-attn[0].min())
    # attn[attn<(1e-5)] = 0
    attn = attn.to(torch.float16)
    with h5py.File('attn_tumor_'+str(i)+'.h5', 'w') as f:
        f.create_dataset('attn', data=attn)
    
    i += 1
    torch.cuda.empty_cache()
    print(str(i), 'cell done')
    
    if i==20:
        break


rna_tumor_20 = rna[tumor_idx].copy()
nonzero = np.count_nonzero(rna_tumor_20.X.toarray(), axis=0)
tf_lst = rna.var.index[nonzero>=5]

tf_attn = {}
tf_exp_cnt = {}
for tf in tf_lst:
    tf_attn[tf] = np.zeros([atac.shape[1], 1], dtype='float16')
    tf_exp_cnt[tf] = 0

for i in range(20):
    rna_sequence = tumor_data[0][i].flatten()
    with h5py.File('attn_tumor_'+str(i)+'.h5', 'r') as f:
        attn = f['attn'][:]
        for tf in tqdm(tf_lst, ncols=80, desc='cell '+str(i)):
            idx = torch.argwhere(rna_sequence==(np.argwhere(rna.var.index==tf))[0][0]+1)
            if len(idx)!=0:
                tf_attn[tf] += attn[:, [idx.item()]]
                tf_exp_cnt[tf] += 1
            
from multiprocessing import Pool
def cnt(tf):
    return sum((tf_attn[tf].flatten())>(0.01*tf_exp_cnt[tf]))

with Pool(20) as p:
    tf_peak_cnt = p.map(cnt, tf_lst)

df = pd.DataFrame({'tf':tf_lst, 'cnt':tf_peak_cnt})
df.to_csv('tumor_tf_peak_0.01.txt', sep='\t', header=None, index=None)

## plot lollipop
import pandas as pd
import numpy as np
from plotnine import *

df = pd.read_csv('tumor_tf_peak_0.01.txt', header=None, sep='\t')
df.columns = ['tf', 'cnt']
df_top20 = df.sort_values('cnt', ascending=False)[:20]
tf_top20 = list(df_top20['tf'])
tf_top20.reverse()
df_top20['tf'] = pd.Categorical(df_top20['tf'], categories=tf_top20)

p = ggplot(df_top20) + aes(x='tf', y='cnt') + geom_segment(aes(x='tf', xend='tf', y=0, yend='cnt'), color='red', size=1) + \
                                              geom_point(color='red', size=3) + coord_flip() + \
                                              xlab('Genes') + ylab('Count') + labs(title='Tumor cells') + \
                                              scale_y_continuous(limits=[0, 50000], breaks=np.arange(0, 50000+1, 10000)) + \
                                              theme_bw() + theme(plot_title=element_text(hjust=0.5)) 
p.save(filename='tumor_top20_lollipp.pdf', dpi=600, height=4, width=5)


## fibroblasts
data = torch.load("/fs/home/jiluzhang/scM2M_no_dec_attn/pan_cancer/all_data/data_with_annotation/h5ad/scM2M/preprocessed_data_test/preprocessed_data_0.pt")
for i in range(1, 5):
    data_tmp = torch.load('/fs/home/jiluzhang/scM2M_no_dec_attn/pan_cancer/all_data/data_with_annotation/h5ad/scM2M/preprocessed_data_test/preprocessed_data_'+str(i)+'.pt')
    data[0] = torch.cat([data[0], data_tmp[0]], axis=0)
    data[1] = torch.cat([data[1], data_tmp[1]], axis=0)
    data[2] = torch.cat([data[2], data_tmp[2]], axis=0)
    data[3] = torch.cat([data[3], data_tmp[3]], axis=0)
    data[4] = torch.cat([data[4], data_tmp[4]], axis=0)
    
fibro_idx = np.argwhere(rna.obs.cell_anno=='Fibroblasts')[:20].flatten()
fibro_data = []
for i in range(len(data)):
    fibro_data.append(data[i][fibro_idx])

dataset = PreDataset(fibro_data)
dataloader_kwargs = {'batch_size': 1, 'shuffle': False}
loader = torch.utils.data.DataLoader(dataset, **dataloader_kwargs)
device = select_least_used_gpu()
model.to(device)

###### same tfs in different cell
i = 0
for inputs in loader:
    rna_sequence, rna_value, atac_sequence, _, enc_pad_mask = [each.to(device) for each in inputs]
    attn = model.generate_attn_weight(rna_sequence, atac_sequence, rna_value, enc_mask=enc_pad_mask, which='decoder')
    attn = (attn[0]-attn[0].min())/(attn[0].max()-attn[0].min())
    # attn[attn<(1e-5)] = 0
    attn = attn.to(torch.float16)
    with h5py.File('attn_fibro_'+str(i)+'.h5', 'w') as f:
        f.create_dataset('attn', data=attn)
    
    i += 1
    torch.cuda.empty_cache()
    print(str(i), 'cell done')
    
    if i==20:
        break


rna_fibro_20 = rna[fibro_idx].copy()
nonzero = np.count_nonzero(rna_fibro_20.X.toarray(), axis=0)
tf_lst = rna.var.index[nonzero>=5]

tf_attn = {}
tf_exp_cnt = {}
for tf in tf_lst:
    tf_attn[tf] = np.zeros([atac.shape[1], 1], dtype='float16')
    tf_exp_cnt[tf] = 0

for i in range(20):
    rna_sequence = fibro_data[0][i].flatten()
    with h5py.File('attn_fibro_'+str(i)+'.h5', 'r') as f:
        attn = f['attn'][:]
        for tf in tqdm(tf_lst, ncols=80, desc='cell '+str(i)):
            idx = torch.argwhere(rna_sequence==(np.argwhere(rna.var.index==tf))[0][0]+1)
            if len(idx)!=0:
                tf_attn[tf] += attn[:, [idx.item()]]
                tf_exp_cnt[tf] += 1
            
from multiprocessing import Pool
def cnt(tf):
    return sum((tf_attn[tf].flatten())>(0.01*tf_exp_cnt[tf]))

with Pool(20) as p:
    tf_peak_cnt = p.map(cnt, tf_lst)

df = pd.DataFrame({'tf':tf_lst, 'cnt':tf_peak_cnt})
df.to_csv('fibro_tf_peak_0.01.txt', sep='\t', header=None, index=None)

## plot lollipop
import pandas as pd
import numpy as np
from plotnine import *

df = pd.read_csv('fibro_tf_peak_0.01.txt', header=None, sep='\t')
df.columns = ['tf', 'cnt']
df_top20 = df.sort_values('cnt', ascending=False)[:20]
tf_top20 = list(df_top20['tf'])
tf_top20.reverse()
df_top20['tf'] = pd.Categorical(df_top20['tf'], categories=tf_top20)

p = ggplot(df_top20) + aes(x='tf', y='cnt') + geom_segment(aes(x='tf', xend='tf', y=0, yend='cnt'), color='blue', size=1) + \
                                              geom_point(color='blue', size=3) + coord_flip() + \
                                              xlab('Genes') + ylab('Count') + labs(title='Fibroblasts') + \
                                              scale_y_continuous(limits=[0, 50000], breaks=np.arange(0, 50000+1, 10000)) + \
                                              theme_bw() + theme(plot_title=element_text(hjust=0.5)) 
p.save(filename='fibro_top20_lollipp.pdf', dpi=600, height=4, width=5)


## plot sankey (holoviews not support cyclic graph)
# pip install holoviews selenium
# import pandas as pd
# import holoviews as hv
# from holoviews import opts
# hv.extension('bokeh')

# data = pd.DataFrame({
#     'source': ['A', 'B', 'C'],
#     'target': ['A_1', 'B_1', 'C_1'],
#     'value': [1, 1, 1]
# })

# sankey_data = hv.Sankey(data)
# sankey_plot = sankey_data.opts(labels='index', width=600, height=400, node_color='index', edge_color='source')
# hv.save(sankey_plot, 'sankey_plot.png', fmt='png')

## plot rank (too many points to show)
# https://plotnine.org/reference/examples/geom_segment-preview.html
# from plotnine import *
# import pandas as pd

# df_tumor = pd.read_table('tumor_tf_peak_0.01.txt', header=None)
# df_tumor.columns = ['tf', 'cnt_tumor']

# df_fibro = pd.read_table('fibro_tf_peak_0.01.txt', header=None)
# df_fibro.columns = ['tf', 'cnt_fibro']

# df = pd.merge(df_tumor, df_fibro, how='outer')
# df.fillna(0, inplace=True)

# df['rank_tumor'] = df['cnt_tumor'].rank(ascending=True, method='first').values
# df['rank_fibro'] = df['cnt_fibro'].rank(ascending=True, method='first').values

# df = df.sort_values('cnt_tumor', ascending=False)

# p = ggplot(df[:100]) + geom_text(aes(1, 'rank_tumor', label='tf'), nudge_x=-0.01, ha='right', size=5) + \
#                  geom_text(aes(2, 'rank_fibro', label='tf'), nudge_x=0.01, ha='left', size=5) + \
#                  geom_point(aes(1, 'rank_tumor')) + geom_point(aes(2, 'rank_fibro')) + \
#                  geom_segment(aes(x=1, y='rank_tumor', xend=2, yend='rank_fibro')) + theme_void()
# p.save(filename='tumor_fibro_tf_rank.pdf', dpi=600, height=8, width=8)


## plot sankey (bin-based seperation)
import pandas as pd
import holoviews as hv
from holoviews import opts
hv.extension('bokeh')

df_tumor = pd.read_table('tumor_tf_peak_0.01.txt', header=None)
df_tumor.columns = ['tf', 'cnt_tumor']

df_fibro = pd.read_table('fibro_tf_peak_0.01.txt', header=None)
df_fibro.columns = ['tf', 'cnt_fibro']

df = pd.merge(df_tumor, df_fibro, how='outer')
df.fillna(0, inplace=True)

df['rank_tumor'] = df['cnt_tumor'].rank(ascending=False, method='first').values
df['rank_fibro'] = df['cnt_fibro'].rank(ascending=False, method='first').values
df = df.sort_values('cnt_tumor', ascending=False)

df['label_tumor'] = pd.cut(df['rank_tumor'], bins=[0,1000,2000,3000,4000,5000,6153], labels=['0001_1000','1001_2000','2001_3000','3001_4000', '4001_5000', '5001_6153'])
df['label_fibro'] = pd.cut(df['rank_fibro'], bins=[0,1000,2000,3000,4000,5000,6153], labels=['0001_1000','1001_2000','2001_3000','3001_4000', '4001_5000', '5001_6153'])

stats = (df['label_tumor'].astype(str)+'-'+df['label_fibro'].astype(str)).value_counts().to_frame()
stats['source'] = stats.index.map(lambda x: 'tumor_'+x.split('-')[0])
stats['target'] = stats.index.map(lambda x: 'fibro_'+x.split('-')[1])
stats.rename(columns={'count': 'value'}, inplace=True)
stats = stats[['source', 'target', 'value']]
stats = stats.sort_values(['source', 'target'])

tumor_idx = ['tumor_'+i for i in ['0001_1000','1001_2000','2001_3000','3001_4000', '4001_5000', '5001_6153']]
fibro_idx = ['fibro_'+i for i in ['0001_1000','1001_2000','2001_3000','3001_4000', '4001_5000', '5001_6153']]
tumor_fibro_idx = pd.DataFrame([[i,j] for i in tumor_idx for j in fibro_idx])
tumor_fibro_idx.columns = ['source', 'target']

stats_all = pd.merge(tumor_fibro_idx, stats, how='left')
stats_all.fillna(0, inplace=True)

sankey_data = hv.Sankey(stats_all)
sankey_plot = sankey_data.opts(labels='index', width=700, height=400, node_color='index', edge_color='source', edge_line_width=0, node_sort=False)
hv.save(sankey_plot, 'tumor_fibro_all_sankey_plot.png', fmt='png')


## plot only for top100 genes
df_100 = df[:100].copy()
df_100['label_tumor'] = pd.cut(df_100['rank_tumor'], bins=[0,20,40,60,80,100], labels=['0001_0020','0021_0040','0041_0060','0061_0080', '0081_0100'])
df_100['label_fibro'] = pd.cut(df_100['rank_fibro'], bins=[0,20,40,60,80,100,6153], labels=['0001_0020','0021_0040','0041_0060','0061_0080', '0081_0100', '0100_6153'])

stats = (df_100['label_tumor'].astype(str)+'-'+df_100['label_fibro'].astype(str)).value_counts().to_frame()
stats['source'] = stats.index.map(lambda x: 'tumor_'+x.split('-')[0])
stats['target'] = stats.index.map(lambda x: 'fibro_'+x.split('-')[1])
stats.rename(columns={'count': 'value'}, inplace=True)
stats = stats[['source', 'target', 'value']]
stats = stats.sort_values(['source', 'target'])

tumor_idx = ['tumor_'+i for i in ['0001_0020','0021_0040','0041_0060','0061_0080', '0081_0100']]
fibro_idx = ['fibro_'+i for i in ['0001_0020','0021_0040','0041_0060','0061_0080', '0081_0100', '0100_6153']]
tumor_fibro_idx = pd.DataFrame([[i,j] for i in tumor_idx for j in fibro_idx])
tumor_fibro_idx.columns = ['source', 'target']

stats_top100 = pd.merge(tumor_fibro_idx, stats, how='left')
stats_top100.fillna(0, inplace=True)

sankey_data = hv.Sankey(stats_top100)
sankey_plot = sankey_data.opts(labels='index', width=700, height=400, node_color='index', edge_color='source', edge_line_width=0, node_sort=False)
# "edge_line_width=0" makes value=0 not show
# "node_sort=False" makes not changing node order
hv.save(sankey_plot, 'tumor_fibro_top100_sankey_plot.png', fmt='png')


## plot only for top20 genes
df_20 = df[:20].copy()
df_20['rank_tumor'] = (df_20['rank_tumor']).astype('int')
df_20['rank_fibro'] = (df_20['rank_fibro']).astype('int')
df_20['label_tumor'] = df_20['rank_tumor']
df_20['label_fibro'] = df_20['rank_fibro']

stats = (df_20['label_tumor'].astype(str)+'-'+df_20['label_fibro'].astype(str)).value_counts().to_frame()
stats['source'] = stats.index.map(lambda x: 'tumor_'+x.split('-')[0])
stats['target'] = stats.index.map(lambda x: 'fibro_'+x.split('-')[1])
stats.rename(columns={'count': 'value'}, inplace=True)
stats = stats[['source', 'target', 'value']]
stats = stats.sort_values(['source', 'target'])

tumor_idx = ['tumor_'+str(i) for i in np.sort(df_20['rank_tumor'].values)]
fibro_idx = ['fibro_'+str(i) for i in np.sort(df_20['rank_fibro'].values)]
tumor_fibro_idx = pd.DataFrame([[i,j] for i in tumor_idx for j in fibro_idx])
tumor_fibro_idx.columns = ['source', 'target']

stats_top20 = pd.merge(tumor_fibro_idx, stats, how='left')
stats_top20.fillna(0, inplace=True)

sankey_data = hv.Sankey(stats_top20)
sankey_plot = sankey_data.opts(labels='index', width=700, height=400, node_color='index', edge_color='source', edge_line_width=0, node_sort=False)
hv.save(sankey_plot, 'tumor_fibro_top20_sankey_plot.png', fmt='png')


########################################## value=0!!! ##########################################

## TF in silico perturbation






