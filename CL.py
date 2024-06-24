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
# 2666090



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





