## pan_cancer_process_strict_filter.py

import pandas as pd
import scanpy as sc
import numpy as np

ids = pd.read_table('files_all.txt', header=None)

for i in range(len(ids)):
    sample_id = ids.iloc[i, 0]
    dat = sc.read_10x_mtx(sample_id, gex_only=False)
    
    rna = dat[:, dat.var['feature_types']=='Gene Expression'].copy()
    sc.pp.filter_cells(rna, min_genes=1000)
    sc.pp.filter_cells(rna, max_genes=10000)
    
    atac = dat[:, dat.var['feature_types']=='Peaks'].copy()
    sc.pp.filter_cells(atac, min_genes=2000)
    sc.pp.filter_cells(atac, max_genes=50000)
    
    idx = np.intersect1d(rna.obs.index, atac.obs.index)
    rna[idx, :].copy().write('strict_filtered_out/'+sample_id+'_rna.h5ad')
    atac[idx, :].copy().write('strict_filtered_out/'+sample_id+'_atac.h5ad')
    print(i, sample_id, 'done')


## out_strict_paired_h5ad.py
import scanpy as sc
import pandas as pd
import numpy as np
import pybedtools
from scipy.sparse import csr_matrix
from tqdm import tqdm

ids = pd.read_table('files_all.txt', header=None)

for i in range(len(ids)):
    sample_id = ids.iloc[i, 0]
    
    ########## rna ##########
    rna = sc.read_h5ad('strict_filtered_out/'+sample_id+'_rna.h5ad')
    rna.X = np.array(rna.X.todense()) 
    genes = pd.read_table('cCRE_gene/human_genes.txt', names=['gene_id', 'gene_name'])
    
    rna_genes = rna.var.copy()
    rna_genes.rename(columns={'gene_ids': 'gene_id'}, inplace=True)
    rna_exp = pd.concat([rna_genes, pd.DataFrame(rna.X.T, index=rna_genes.index)], axis=1)
    
    X_new = pd.merge(genes, rna_exp, how='left', on='gene_id').iloc[:, 3:].T
    X_new = np.array(X_new, dtype='float32')
    X_new[np.isnan(X_new)] = 0
    
    rna_new_var = pd.DataFrame({'gene_ids': genes['gene_id'], 'feature_types': 'Gene Expression'})
    rna_new = sc.AnnData(X_new, obs=rna.obs, var=rna_new_var)  
    rna_new.var.index = genes['gene_name'].values  # set index for var
    rna_new.X = csr_matrix(rna_new.X)
    
    rna_new.write('strict_filtered_out/'+sample_id+'_rna_aligned.h5ad')
    
    
    ########## atac ##########
    atac = sc.read_h5ad('strict_filtered_out/'+sample_id+'_atac.h5ad')
    atac.X = np.array(atac.X.todense())
    atac.X[atac.X>1] = 1
    atac.X[atac.X<1] = 0
    
    peaks = pd.DataFrame({'id': atac.var_names})
    peaks['chr'] = peaks['id'].map(lambda x: x.split(':')[0])
    peaks['start'] = peaks['id'].map(lambda x: x.split(':')[1].split('-')[0])
    peaks['end'] = peaks['id'].map(lambda x: x.split(':')[1].split('-')[1])
    peaks.drop(columns='id', inplace=True)
    peaks['idx'] = range(peaks.shape[0])
    
    cCREs = pd.read_table('cCRE_gene/human_cCREs.bed', names=['chr', 'start', 'end'])
    cCREs['idx'] = range(cCREs.shape[0])
    cCREs_bed = pybedtools.BedTool.from_dataframe(cCREs)
    peaks_bed = pybedtools.BedTool.from_dataframe(peaks)
    idx_map = peaks_bed.intersect(cCREs_bed, wa=True, wb=True).to_dataframe().iloc[:, [3, 7]]
    idx_map.columns = ['peaks_idx', 'cCREs_idx']
    
    m = np.zeros([atac.n_obs, cCREs_bed.to_dataframe().shape[0]], dtype='float32')
    for j in tqdm(range(atac.X.shape[0])):
        m[j][idx_map[idx_map['peaks_idx'].isin(peaks.iloc[np.nonzero(atac.X[j])]['idx'])]['cCREs_idx']] = 1
    
    atac_new_var = pd.DataFrame({'gene_ids': cCREs['chr'] + ':' + cCREs['start'].map(str) + '-' + cCREs['end'].map(str), 'feature_types': 'Peaks'})
    atac_new = sc.AnnData(m, obs=atac.obs, var=atac_new_var)
    atac_new.var.index = atac_new.var['gene_ids'].values  # set index
    atac_new.X = csr_matrix(atac_new.X)
    
    atac_new.write('strict_filtered_out/'+sample_id+'_atac_aligned.h5ad')

    print(i, sample_id, 'done')


## merge_filter_genes_peaks.py
import scanpy as sc
import pandas as pd
import numpy as np
import anndata as ad

ids = pd.read_table('../files_all.txt', header=None)

## count for genes & peaks (time-consuming)
for i in range(len(ids)):
    sample_id = ids.iloc[i, 0]
    if i==0:
        rna_init = sc.read_h5ad(sample_id+'_rna_aligned.h5ad')
        rna_cnt = np.count_nonzero(rna_init.X.toarray(), axis=0)
        atac_init = sc.read_h5ad(sample_id+'_atac_aligned.h5ad')
        atac_cnt = np.count_nonzero(atac_init.X.toarray(), axis=0)
    else:
        rna_tmp = sc.read_h5ad(sample_id+'_rna_aligned.h5ad')
        rna_cnt = rna_cnt + np.count_nonzero(rna_tmp.X.toarray(), axis=0)
        atac_tmp = sc.read_h5ad(sample_id+'_atac_aligned.h5ad')
        atac_cnt = atac_cnt + np.count_nonzero(atac_tmp.X.toarray(), axis=0)
    print(i, sample_id, 'done')

np.save('rna_cnt_stats.npy', rna_cnt)
np.save('atac_cnt_stats.npy', atac_cnt)

## count cells
for i in range(len(ids)):
    sample_id = ids.iloc[i, 0]
    if i==0:
        rna_init = sc.read_h5ad(sample_id+'_rna_aligned.h5ad')
        cell_cnt = rna_init.n_obs
    else:
        rna_tmp = sc.read_h5ad(sample_id+'_rna_aligned.h5ad')
        cell_cnt = cell_cnt + rna_tmp.n_obs
    print(i, sample_id, 'done')

# cell_cnt 481,024

gene_idx = rna_cnt>481024*0.001   # 20,539
peak_idx = atac_cnt>481024*0.001  # 542,369


for sample_id in ['CE336E1-S1', 'CE348E1-S1K1', 'CPT2373DU-S1', 'HT181P1-T1A3', 'HT263B1-S1H1',
                  'ML123M1-Ty1', 'P5504-N1', 'PM581P1-T1', 'VF034V1-T1', 'GBML018G1-M1']:
    if sample_id=='CE336E1-S1':
        rna = sc.read_h5ad(sample_id+'_rna_aligned.h5ad')[:, gene_idx]
        rna.obs.index = sample_id+'-'+rna.obs.index
        atac = sc.read_h5ad(sample_id+'_atac_aligned.h5ad')[:, peak_idx]
        atac.obs.index = sample_id+'-'+atac.obs.index
    else:
        rna_tmp = sc.read_h5ad(sample_id+'_rna_aligned.h5ad')[:, gene_idx]
        rna_tmp.obs.index = sample_id+'-'+rna_tmp.obs.index
        rna = ad.concat([rna, rna_tmp])
        atac_tmp = sc.read_h5ad(sample_id+'_atac_aligned.h5ad')[:, peak_idx]
        atac_tmp.obs.index = sample_id+'-'+atac_tmp.obs.index
        atac = ad.concat([atac, atac_tmp])
    print(sample_id, 'done')

rna.var = rna_tmp.var    # 27845 × 20539
atac.var = atac_tmp.var  # 27845 × 542369

np.random.seed(0)
shuf_idx = np.arange(rna.shape[0])
np.random.shuffle(shuf_idx)
rna[shuf_idx[:25000], :].write('pan_cancer_rna_train_25000.h5ad')
atac[shuf_idx[:25000], :].write('pan_cancer_atac_train_25000.h5ad')
rna[shuf_idx[25000:], :].write('pan_cancer_rna_test_2845.h5ad')
atac[shuf_idx[25000:], :].write('pan_cancer_atac_test_2845.h5ad')


rna_cnt = np.load('rna_cnt_stats.npy')
atac_cnt = np.load('atac_cnt_stats.npy')
gene_idx = rna_cnt>481024*0.001  
peak_idx = atac_cnt>481024*0.001
crc_rna = sc.read_h5ad('CM354C2-T1_rna_aligned.h5ad')[:, gene_idx]
crc_rna.obs.index = 'CM354C2-T1-'+crc_rna.obs.index
crc_rna.write('pan_cancer_rna_test_crc.h5ad')
crc_atac = sc.read_h5ad('CM354C2-T1_atac_aligned.h5ad')[:, peak_idx]
crc_atac.obs.index = 'CM354C2-T1-'+crc_atac.obs.index
crc_atac.write('pan_cancer_atac_test_crc.h5ad')


# scM2M_v2_pro
accelerate launch --main_process_port 29501 --config_file default_config.yaml rna2atac_pre-train.py \
                  --atac pan_cancer_atac_train_25000.h5ad --rna pan_cancer_rna_train_25000.h5ad \
                  --save models --name pan_cancer --enc_max_len 20539 --dec_max_len 542369 --batch_size 10 --lr 0.001


## output h5ad files (5 samples per group)
import scanpy as sc
import pandas as pd
import numpy as np
import anndata as ad

rna_cnt = np.load('rna_cnt_stats.npy')
atac_cnt = np.load('atac_cnt_stats.npy')

gene_idx = rna_cnt>481024*0.001   # 20,539
peak_idx = atac_cnt>481024*0.001  # 542,369

sample_shuf = pd.read_table('sample_id_shuf.txt', header=None)
for i in range(25):
    rna_0 = sc.read_h5ad(sample_shuf[0][i*5]+'_rna_aligned.h5ad')[:, gene_idx]
    rna_0.obs.index = sample_shuf[0][i*5]+'-'+rna_0.obs.index
    rna_1 = sc.read_h5ad(sample_shuf[0][i*5+1]+'_rna_aligned.h5ad')[:, gene_idx]
    rna_1.obs.index = sample_shuf[0][i*5+1]+'-'+rna_1.obs.index
    rna_2 = sc.read_h5ad(sample_shuf[0][i*5+2]+'_rna_aligned.h5ad')[:, gene_idx]
    rna_2.obs.index = sample_shuf[0][i*5+2]+'-'+rna_2.obs.index
    rna_3 = sc.read_h5ad(sample_shuf[0][i*5+3]+'_rna_aligned.h5ad')[:, gene_idx]
    rna_3.obs.index = sample_shuf[0][i*5+3]+'-'+rna_3.obs.index
    rna_4 = sc.read_h5ad(sample_shuf[0][i*5+4]+'_rna_aligned.h5ad')[:, gene_idx]
    rna_4.obs.index = sample_shuf[0][i*5+4]+'-'+rna_4.obs.index
    rna = ad.concat([rna_0, rna_1, rna_2, rna_3, rna_4])
    rna.var = rna_0.var 
    print(i, 'merge rna done')
    
    atac_0 = sc.read_h5ad(sample_shuf[0][i*5]+'_atac_aligned.h5ad')[:, peak_idx]
    atac_0.obs.index = sample_shuf[0][i*5]+'-'+atac_0.obs.index
    atac_1 = sc.read_h5ad(sample_shuf[0][i*5+1]+'_atac_aligned.h5ad')[:, peak_idx]
    atac_1.obs.index = sample_shuf[0][i*5+1]+'-'+atac_1.obs.index
    atac_2 = sc.read_h5ad(sample_shuf[0][i*5+2]+'_atac_aligned.h5ad')[:, peak_idx]
    atac_2.obs.index = sample_shuf[0][i*5+2]+'-'+atac_2.obs.index
    atac_3 = sc.read_h5ad(sample_shuf[0][i*5+3]+'_atac_aligned.h5ad')[:, peak_idx]
    atac_3.obs.index = sample_shuf[0][i*5+3]+'-'+atac_3.obs.index
    atac_4 = sc.read_h5ad(sample_shuf[0][i*5+4]+'_atac_aligned.h5ad')[:, peak_idx]
    atac_4.obs.index = sample_shuf[0][i*5+4]+'-'+atac_4.obs.index
    atac = ad.concat([atac_0, atac_1, atac_2, atac_3, atac_4])
    atac.var = atac_0.var
    print(i, 'merge atac done')
    
    np.random.seed(0)
    shuf_idx = np.arange(rna.shape[0])
    np.random.shuffle(shuf_idx)
    rna[shuf_idx, :].write('pan_cancer_rna_dataset_'+str(i)+'.h5ad')
    atac[shuf_idx, :].write('pan_cancer_atac_dataset_'+str(i)+'.h5ad')
    print(i, 'output h5ad done')

## count cell
for sample_id in sample_shuf[0].values:
    print(sample_id, sc.read_h5ad(sample_id+'_rna_aligned.h5ad').n_obs)



## ./train_datasets.sh
accelerate launch --config_file default_config.yaml rna2atac_pre-train_p1.py --SEED 0 --epoch 1 \
                  --rna pan_cancer_rna_dataset_0.h5ad --atac pan_cancer_atac_dataset_0.h5ad \
                  --save models --name datasets --enc_max_len 20539 --dec_max_len 542369 --batch_size 10 --lr 0.001 > train_dataset_0_p1.log

accelerate launch --config_file default_config.yaml rna2atac_pre-train_p2.py --load models/datasets_epoch_1/pytorch_model.bin --SEED 0 --epoch 1 \
                  --rna pan_cancer_rna_dataset_0.h5ad --atac pan_cancer_atac_dataset_0.h5ad \
                  --save models --name datasets --enc_max_len 20539 --dec_max_len 542369 --batch_size 10 --lr 0.001 > train_dataset_0_p2.log

for i in `seq 1 24`;do
    accelerate launch --config_file default_config.yaml rna2atac_pre-train_p1.py --load models/datasets_epoch_1/pytorch_model.bin --SEED 0 --epoch 1 \
                      --rna pan_cancer_rna_dataset_$i.h5ad --atac pan_cancer_atac_dataset_$i.h5ad \
                      --save models --name datasets --enc_max_len 20539 --dec_max_len 542369 --batch_size 10 --lr 0.001 > train_dataset_$i\_p1.log
    
    accelerate launch --config_file default_config.yaml rna2atac_pre-train_p2.py --load models/datasets_epoch_1/pytorch_model.bin --SEED 0 --epoch 1 \
                      --rna pan_cancer_rna_dataset_$i.h5ad --atac pan_cancer_atac_dataset_$i.h5ad \
                      --save models --name datasets --enc_max_len 20539 --dec_max_len 542369 --batch_size 10 --lr 0.001 > train_dataset_$i\_p2.log
done


nohup ./train_datasets.sh > train_datasets.log &
# 2050491


accelerate launch --config_file default_config.yaml rna2atac_pre-train_p2.py --load models/datasets_epoch_1/pytorch_model.bin --SEED 0 --epoch 1 \
                  --rna pan_cancer_rna_dataset_12.h5ad --atac pan_cancer_atac_dataset_12.h5ad \
                  --save models --name datasets --enc_max_len 20539 --dec_max_len 542369 --batch_size 10 --lr 0.001 > train_dataset_12_p2.log



import scanpy as sc
import pandas as pd
import numpy as np
import anndata as ad

rna_cnt = np.load('rna_cnt_stats.npy')
atac_cnt = np.load('atac_cnt_stats.npy')

gene_idx = rna_cnt>481024*0.001   # 20,539
peak_idx = atac_cnt>481024*0.001  # 542,369

samples = pd.read_table('../files_all_cancer_type.txt', header=None, names=['sample_id', 'cancer_type'])
samples = samples[samples['cancer_type']!='UCEC']
samples.index = range(len(samples))

sample_id_lst = []
for i in range(len(samples['sample_id'])):
    sample_id_lst += list(samples['sample_id'][i]+'_'+samples['cancer_type'][i]+'_'+sc.read_h5ad(samples['sample_id'][i]+'_rna_aligned.h5ad').obs.index.values)
    print(i, samples['sample_id'][i], 'done')

np.random.seed(0)
shuf_idx = np.arange(len(sample_id_lst))
np.random.shuffle(shuf_idx)

def out_rna(p=sample_id_lst[:5000], cell_cnt=5000, dataset_idx=0):
    p1 = sorted(p)
    for i in range(cell_cnt):
        if i==0:
            sample_id = p1[i].split('_')[0]
            cancer_type = p1[i].split('_')[1]
            bc_lst = [p1[i].split('_')[2]]
        elif i!=(cell_cnt-1):
            if p1[i].split('_')[0]==sample_id:
                bc_lst.append(p1[i].split('_')[2])
            else:
                if 'rna_out' in locals():
                    rna_tmp = sc.read_h5ad(sample_id+'_rna_aligned.h5ad')[bc_lst, gene_idx].copy()
                    rna_tmp.obs.index = [sample_id+'_'+cancer_type+'_'+bc for bc in bc_lst]
                    rna_out = ad.concat([rna_out, rna_tmp])
                else:
                    rna_out = sc.read_h5ad(sample_id+'_rna_aligned.h5ad')[bc_lst, gene_idx].copy()
                    rna_out.obs.index = [sample_id+'_'+cancer_type+'_'+bc for bc in bc_lst]
                sample_id = p1[i].split('_')[0]
                cancer_type = p1[i].split('_')[1]
                bc_lst = [p1[i].split('_')[2]]
        else:
            if p1[i].split('_')[0]==sample_id:
                bc_lst.append(p1[i].split('_')[2])
                rna_tmp = sc.read_h5ad(sample_id+'_rna_aligned.h5ad')[bc_lst, gene_idx].copy()
                rna_tmp.obs.index = [sample_id+'_'+cancer_type+'_'+bc for bc in bc_lst]
                rna_out = ad.concat([rna_out, rna_tmp])
            else:
                # output the last one
                rna_tmp = sc.read_h5ad(sample_id+'_rna_aligned.h5ad')[bc_lst, gene_idx].copy()
                rna_tmp.obs.index = [sample_id+'_'+cancer_type+'_'+bc for bc in bc_lst]
                rna_out = ad.concat([rna_out, rna_tmp])
                # output the current one
                sample_id = p1[i].split('_')[0]
                bc_lst = [p1[i].split('_')[2]]
                cancer_type = p1[i].split('_')[1]
                rna_tmp = sc.read_h5ad(sample_id+'_rna_aligned.h5ad')[bc_lst, gene_idx].copy()
                rna_tmp.obs.index = [sample_id+'_'+cancer_type+'_'+bc for bc in bc_lst]
                rna_out = ad.concat([rna_out, rna_tmp])
    
    rna_out.var = rna_tmp.var
    rna_out[p, :].write('pan_cancer_rna_dataset_'+str(dataset_idx)+'.h5ad')


# 460187=5000*92+187
for i in range(93):
    if i!=92:
        out_rna(p=np.array(sample_id_lst)[shuf_idx[(5000*i):(5000*i+5000)]], cell_cnt=5000, dataset_idx=i)
        print(i, '*5000 cells done')
    else:
        out_rna(p=np.array(sample_id_lst)[shuf_idx[(5000*i):]], cell_cnt=187, dataset_idx=i)
        print('all cells done')


def out_atac(p=sample_id_lst[:5000], cell_cnt=5000, dataset_idx=0):
    p1 = sorted(p)
    for i in range(cell_cnt):
        if i==0:
            sample_id = p1[i].split('_')[0]
            cancer_type = p1[i].split('_')[1]
            bc_lst = [p1[i].split('_')[2]]
        elif i!=(cell_cnt-1):
            if p1[i].split('_')[0]==sample_id:
                bc_lst.append(p1[i].split('_')[2])
            else:
                if 'atac_out' in locals():
                    atac_tmp = sc.read_h5ad(sample_id+'_atac_aligned.h5ad')[bc_lst, peak_idx].copy()
                    atac_tmp.obs.index = [sample_id+'_'+cancer_type+'_'+bc for bc in bc_lst]
                    atac_out = ad.concat([atac_out, atac_tmp])
                else:
                    atac_out = sc.read_h5ad(sample_id+'_atac_aligned.h5ad')[bc_lst, peak_idx].copy()
                    atac_out.obs.index = [sample_id+'_'+cancer_type+'_'+bc for bc in bc_lst]
                sample_id = p1[i].split('_')[0]
                cancer_type = p1[i].split('_')[1]
                bc_lst = [p1[i].split('_')[2]]
        else:
            if p1[i].split('_')[0]==sample_id:
                bc_lst.append(p1[i].split('_')[2])
                atac_tmp = sc.read_h5ad(sample_id+'_atac_aligned.h5ad')[bc_lst, peak_idx].copy()
                atac_tmp.obs.index = [sample_id+'_'+cancer_type+'_'+bc for bc in bc_lst]
                atac_out = ad.concat([atac_out, atac_tmp])
            else:
                # output the last one
                atac_tmp = sc.read_h5ad(sample_id+'_atac_aligned.h5ad')[bc_lst, peak_idx].copy()
                atac_tmp.obs.index = [sample_id+'_'+cancer_type+'_'+bc for bc in bc_lst]
                atac_out = ad.concat([atac_out, atac_tmp])
                # output the current one
                sample_id = p1[i].split('_')[0]
                bc_lst = [p1[i].split('_')[2]]
                cancer_type = p1[i].split('_')[1]
                atac_tmp = sc.read_h5ad(sample_id+'_atac_aligned.h5ad')[bc_lst, peak_idx].copy()
                atac_tmp.obs.index = [sample_id+'_'+cancer_type+'_'+bc for bc in bc_lst]
                atac_out = ad.concat([atac_out, atac_tmp])
    
    atac_out.var = atac_tmp.var
    atac_out[p, :].write('pan_cancer_atac_dataset_'+str(dataset_idx)+'.h5ad')

# ~5 min per 5000 cells (time consuming)
for i in range(93):
    if i!=92:
        out_atac(p=np.array(sample_id_lst)[shuf_idx[(5000*i):(5000*i+5000)]], cell_cnt=5000, dataset_idx=i)
        print(i, '*5000 cells done')
    else:
        out_atac(p=np.array(sample_id_lst)[shuf_idx[(5000*i):]], cell_cnt=187, dataset_idx=i)
        print('all cells done')


for i in `seq 0 92`;do
    if [ $i = 0 ];then
        accelerate launch --config_file default_config.yaml rna2atac_pre-train.py --SEED 0 --epoch 1 \
                          --rna pan_cancer_rna_dataset_$i.h5ad --atac pan_cancer_atac_dataset_$i.h5ad \
                          --save model_$i --name datasets --enc_max_len 20539 --dec_max_len 542369 --batch_size 5 --lr 0.001 > train_dataset_$i.log
    else
        j=`expr $i - 1`
        accelerate launch --config_file default_config.yaml rna2atac_pre-train.py --load model_$j/datasets_epoch_1/pytorch_model.bin --SEED 0 --epoch 1 \
                          --rna pan_cancer_rna_dataset_$i.h5ad --atac pan_cancer_atac_dataset_$i.h5ad \
                          --save model_$i --name datasets --enc_max_len 20539 --dec_max_len 542369 --batch_size 10 --lr 0.001 > train_dataset_$i.log
    fi
done


## merge UCEC samples
import scanpy as sc
import pandas as pd
import numpy as np
import anndata as ad

rna_cnt = np.load('rna_cnt_stats.npy')
atac_cnt = np.load('atac_cnt_stats.npy')

gene_idx = rna_cnt>481024*0.001   # 20,539
peak_idx = atac_cnt>481024*0.001  # 542,369

samples = pd.read_table('../files_all_cancer_type.txt', header=None, names=['sample_id', 'cancer_type'])
samples = samples[samples['cancer_type']=='UCEC']
samples.index = range(len(samples))

rna_s0 = sc.read_h5ad(samples['sample_id'][0]+'_rna_aligned.h5ad')
rna_s0.obs.index = samples['sample_id'][0]+'_UCEC_'+rna_s0.obs.index
rna_s1 = sc.read_h5ad(samples['sample_id'][1]+'_rna_aligned.h5ad')
rna_s1.obs.index = samples['sample_id'][1]+'_UCEC_'+rna_s1.obs.index
rna_s2 = sc.read_h5ad(samples['sample_id'][2]+'_rna_aligned.h5ad')
rna_s2.obs.index = samples['sample_id'][2]+'_UCEC_'+rna_s2.obs.index
rna_s3 = sc.read_h5ad(samples['sample_id'][3]+'_rna_aligned.h5ad')
rna_s3.obs.index = samples['sample_id'][3]+'_UCEC_'+rna_s3.obs.index
rna = ad.concat([rna_s0, rna_s1, rna_s2, rna_s3])
rna.var = rna_s0.var
rna[:, gene_idx].write('pan_cancer_ucec_rna.h5ad')

atac_s0 = sc.read_h5ad(samples['sample_id'][0]+'_atac_aligned.h5ad')
atac_s0.obs.index = samples['sample_id'][0]+'_UCEC_'+atac_s0.obs.index
atac_s1 = sc.read_h5ad(samples['sample_id'][1]+'_atac_aligned.h5ad')
atac_s1.obs.index = samples['sample_id'][1]+'_UCEC_'+atac_s1.obs.index
atac_s2 = sc.read_h5ad(samples['sample_id'][2]+'_atac_aligned.h5ad')
atac_s2.obs.index = samples['sample_id'][2]+'_UCEC_'+atac_s2.obs.index
atac_s3 = sc.read_h5ad(samples['sample_id'][3]+'_atac_aligned.h5ad')
atac_s3.obs.index = samples['sample_id'][3]+'_UCEC_'+atac_s3.obs.index
atac = ad.concat([atac_s0, atac_s1, atac_s2, atac_s3])
atac.var = atac_s0.var
atac[:, peak_idx].write('pan_cancer_ucec_atac.h5ad')


## UCEC prediction evaluation
import sys
import os
sys.path.append("M2Mmodel")
import torch
from M2M import M2M_rna2atac
from utils import *
import scanpy as sc
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix

test_atac_file = 'pan_cancer_ucec_atac.h5ad'
test_rna_file = 'pan_cancer_ucec_rna.h5ad'

enc_max_len = 20539
dec_max_len = 542369

batch_size = 5
rna_max_value = 255
attn_window_size = 2*2048

device = torch.device("cuda:2")

test_atac = sc.read_h5ad(test_atac_file)[:100, :].copy()
test_atac.X = test_atac.X.toarray()

test_rna = sc.read_h5ad(test_rna_file)[:100, :].copy()
test_rna.X = test_rna.X.toarray()
test_rna.X = np.round(test_rna.X/(test_rna.X.sum(axis=1, keepdims=True))*10000)
test_rna.X[test_rna.X>255] = 255

test_kwargs = {'batch_size': batch_size, 'shuffle': False}  # test_kwargs = {'batch_size': batch_size, 'shuffle': True}
cuda_kwargs = {
    # 'pin_memory': True,  # 将加载的数据张量复制到 CUDA 设备的固定内存中，提高效率但会消耗更多内存
    "num_workers": 4,
    'prefetch_factor': 2
}  
test_kwargs.update(cuda_kwargs)

test_dataset = FullLenPairDataset(test_rna.X, test_atac.X, rna_max_value)
test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)

model = M2M_rna2atac(
            dim = 64, #Luz 128,
            enc_num_tokens = rna_max_value + 1,
            enc_seq_len = enc_max_len,
            enc_depth = 2, #2
            enc_heads = 2, #4
            enc_attn_window_size = attn_window_size,
            enc_dilation_growth_rate = 2,
            enc_ff_mult = 4,

            latent_len = 16, #Luz 256, 
            num_middle_MLP = 1, #Luz 2, 
            latent_dropout = 0.1, #Luz 0.1,
            mlp_init_weight = 0.08,

            dec_seq_len = dec_max_len,
            dec_depth = 2, #2
            dec_heads = 2, #8
            dec_attn_window_size = attn_window_size,
            dec_dilation_growth_rate = 5,
            dec_ff_mult = 4
)

model.load_state_dict(torch.load('model_1/datasets_epoch_1/pytorch_model.bin'))

model.to(device)
model.eval()

with torch.no_grad():
    data_iter = enumerate(test_loader)
    for i, (src, tgt) in data_iter:
        src = src.long()
        #tgt = tgt.float()
        logist = model(src)
        accuracy  = binary_auroc(torch.sigmoid(logist), tgt, num_tasks = logist.shape[0]).item()
        print(accuracy)
        precision = binary_auprc(torch.sigmoid(logist), tgt, num_tasks = logist.shape[0]).item()
        print(precision)



for i in `seq 0 92`;do
    echo -n $i' ' >> eval_stats.txt
    accelerate launch --main_process_port 29506 --config_file default_config_test.yaml rna2atac_test.py --load model_$i/datasets_epoch_1/pytorch_model.bin --SEED 0 --epoch 1 \
                      --rna pan_cancer_ucec_rna.h5ad --atac pan_cancer_ucec_atac.h5ad \
                      --enc_max_len 20539 --dec_max_len 542369 --batch_size 10 >> eval_stats.txt
done



accelerate launch --main_process_port 29506 --config_file default_config_test.yaml rna2atac_predict.py --load model_92/datasets_epoch_1/pytorch_model.bin --SEED 0 --epoch 1 \
                  --rna pan_cancer_ucec_rna.h5ad --atac pan_cancer_ucec_atac.h5ad \
                  --enc_max_len 20539 --dec_max_len 542369 --batch_size 10


##################################################################################################
## https://humantumoratlas.org/
library(Seurat)
dat <- readRDS('CPT1541DU-T1.rds')
write.table(dat@meta.data['cell_type'], 'UCEC_cell_type.txt', row.names=TRUE, col.names=FALSE, quote=FALSE, sep='\t')
##################################################################################################


import numpy as np
import scanpy as sc
import snapatac2 as snap
from scipy.sparse import csr_matrix
import pandas as pd

m_raw = np.load('pan_cancer_atac_ucec_predict_1000.npy')
#m = ((m_raw.T > m_raw.T.mean(axis=0)).T) & (m_raw>m_raw.mean(axis=0)).astype(int)

m = m_raw.copy()
m[m>0.3]=1
m[m<=0.3]=0

atac_true = snap.read('pan_cancer_ucec_atac.h5ad', backed=None)
atac_pred = atac_true[:1000, :].copy()
atac_pred.X = csr_matrix(m)

cell_anno = pd.read_table('UCEC_cell_type.txt', names=['cell_id', 'cell_type'])
cell_anno.index = ['CPT1541DU-T1_UCEC_'+x.split('_')[2] for x in cell_anno['cell_id']]
cell_anno.drop(['cell_id'], axis=1, inplace=True)

cell_anno_lst = []
for i in range(atac_pred.n_obs):
    if atac_pred.obs.index[i] in cell_anno.index:
        cell_anno_lst.append(cell_anno.loc[atac_pred.obs.index[i], 'cell_type'])
    else:
        cell_anno_lst.append('Unknown')

atac_pred.obs['cell_anno'] = cell_anno_lst

snap.pp.select_features(atac_pred, n_features=10000)
snap.tl.spectral(atac_pred) #snap.tl.spectral(atac_pred, n_comps=50)
snap.tl.umap(atac_pred)
snap.pl.umap(atac_pred, color='cell_anno', show=False, out_file='umap_tmp.pdf', height=500)
#snap.pl.umap(atac_pred, color='cell_anno', show=False, out_file='umap_cutoff_mean_nf_2000_color.pdf', height=500)

#snap.pl.umap(atac_pred, color='cell_anno', show=False, out_file='umap_true_173630_color.pdf', height=500)


m_raw[0]>np.sort(m_raw[0])[-10000]


true_0 = atac_true.X[1].toarray()[0]
for i in np.arange(0.1, 0.6, 0.1):
    pred_0 = m_raw[1].copy()
    pred_0[pred_0>i]=1
    pred_0[pred_0<i]=0
    print(i, sum(pred_0==1), sum(true_0[pred_0==1]==1)/sum(pred_0==1))



atac_pred = atac_true[:1000, :].copy()
atac_pred.obs['cell_anno'] = cell_anno_lst

from sklearn.metrics import precision_recall_curve, auc

for i in range(100):
    true_0 = atac_pred.X[i].toarray()[0]
    pred_0 = m_raw[i].copy()
    precision, recall, thresholds = precision_recall_curve(true_0, pred_0)
    print(i, atac_pred.obs['cell_anno'][i], auc(recall, precision))



#atac_pred.write('atac_pred_tumor_B_res_cutoff_0.9_new.h5ad')

atac_pred.obs.cell_anno = atac_true.obs.cell_anno.cat.set_categories(['T cell', 'Tumor B cell', 'Monocyte', 'Normal B cell', 'Other'])  # keep the color consistent

sc.pl.umap(atac_pred, color='cell_anno', legend_fontsize='7', legend_loc='right margin',
           size=3, title='', frameon=True, save='_atac_pred_cell_annotation_0.96.pdf')





for i in range(len(ids)):
    sample_id = ids.iloc[i, 0]
    if i==0:
        rna = sc.read_h5ad(sample_id+'_rna_aligned.h5ad')
        rna.obs.index = sample_id+'-'+rna.obs.index
        atac = sc.read_h5ad(sample_id+'_atac_aligned.h5ad')
        atac.obs.index = sample_id+'-'+atac.obs.index
    else:
        rna_tmp = sc.read_h5ad(sample_id+'_rna_aligned.h5ad')
        rna_tmp.obs.index = sample_id+'-'+rna_tmp.obs.index
        rna = ad.concat([rna, rna_tmp])
        atac_tmp = sc.read_h5ad(sample_id+'_atac_aligned.h5ad')
        atac_tmp.obs.index = sample_id+'-'+atac_tmp.obs.index
        atac = ad.concat([atac, atac_tmp])
    print(i, sample_id, 'done')

rna.var = rna_tmp.var
atac.var = atac_tmp.var

rna.write('pan_cancer_rna_head_20_samples.h5ad')
atac.write('pan_cancer_atac_head_20_samples.h5ad')

sc.pp.filter_genes(rna, min_cells=1)
sc.pp.filter_genes(atac, min_cells=1)

np.random.seed(0)
shuf_idx = np.arange(rna.shape[0])
np.random.shuffle(shuf_idx)
rna[shuf_idx[:10000], :].write('pan_cancer_rna_train_shuf.h5ad')     # 69577 × 23502 (head 20 samples)
atac[shuf_idx[:10000], :].write('pan_cancer_atac_train_shuf.h5ad')   # 69577 × 539927
rna[shuf_idx[-10000:], :].write('pan_cancer_rna_test_shuf.h5ad')
atac[shuf_idx[-10000:], :].write('pan_cancer_atac_test_shuf.h5ad')



sc.pp.filter_genes(rna, min_cells=100)
sc.pp.filter_genes(atac, min_cells=100)

np.random.seed(0)
shuf_idx = np.arange(rna.shape[0])
np.random.shuffle(shuf_idx)
rna[shuf_idx[:10000], :].write('pan_cancer_rna_train_shuf_2.h5ad')     # 69577 × 19546 (head 20 samples)
atac[shuf_idx[:10000], :].write('pan_cancer_atac_train_shuf_2.h5ad')   # 69577 × 471457
rna[shuf_idx[-10000:], :].write('pan_cancer_rna_test_shuf_2.h5ad')
atac[shuf_idx[-10000:], :].write('pan_cancer_atac_test_shuf_2.h5ad')


sc.pp.filter_genes(rna, min_cells=1)
sc.pp.filter_genes(atac, min_cells=100)

np.random.seed(0)
shuf_idx = np.arange(rna.shape[0])
np.random.shuffle(shuf_idx)
rna[shuf_idx[:10000], :].write('pan_cancer_rna_train_shuf_3.h5ad')     # 69577 × 23502 (head 20 samples)
atac[shuf_idx[:10000], :].write('pan_cancer_atac_train_shuf_3.h5ad')   # 69577 × 471457
rna[shuf_idx[-10000:], :].write('pan_cancer_rna_test_shuf_3.h5ad')
atac[shuf_idx[-10000:], :].write('pan_cancer_atac_test_shuf_3.h5ad')



accelerate launch --config_file default_config.yaml rna2atac_pre-train_v2.py --atac atac_32124_shuf.h5ad --rna rna_32124_shuf.h5ad --save models --name 32124_cells 

python rna2atac_pre-train_v3.py --atac pan_cancer_atac_train_shuf.h5ad --rna pan_cancer_rna_train_shuf.h5ad --save models --name pan_cancer \
                                --enc_max_len 23502 --dec_max_len 539927 --batch_size 5 --lr 0.001

python rna2atac_pre-train_v3.py --atac pan_cancer_atac_train_shuf.h5ad --rna pan_cancer_rna_train_shuf.h5ad --save models --name pan_cancer \
                                --enc_max_len 23502 --dec_max_len 539927 --batch_size 5 --lr 0.0001 --load models/pan_cancer_epoch_1_iter_159/pytorch_model.bin

accelerate launch --main_process_port 29501 --config_file default_config.yaml rna2atac_pre-train_v3.py \
                  --atac pan_cancer_atac_train_shuf_2.h5ad --rna pan_cancer_rna_train_shuf_2.h5ad \
                  --save models --name pan_cancer --enc_max_len 19546 --dec_max_len 471457 --batch_size 10 --lr 0.001

nohup accelerate launch --main_process_port 29501 --config_file default_config.yaml rna2atac_pre-train_v3.py \
                        --atac pan_cancer_atac_train_shuf_3.h5ad --rna pan_cancer_rna_train_shuf_3.h5ad \
                        --save models --name pan_cancer --enc_max_len 23502 --dec_max_len 471457 --batch_size 10 --lr 0.001 > training_20240328_3.log &

#latent_len: 32
