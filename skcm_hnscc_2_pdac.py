###### h5ad files
# from /fs/home/jiluzhang/scM2M_final_pan_cancer

import scanpy as sc
import anndata as ad

rna_train = sc.read_h5ad('rna_train.h5ad')
rna_train.obs['dt'] = 'train'
rna_val = sc.read_h5ad('rna_val.h5ad')
rna_val.obs['dt'] = 'val'
rna_test = sc.read_h5ad('rna_test.h5ad')
rna_test.obs['dt'] = 'test'
rna = ad.concat([rna_train, rna_val, rna_test])
rna.var = rna_train.var
del rna.var['n_cells']
rna.write('rna.h5ad')

atac_train = sc.read_h5ad('atac_train.h5ad')
atac_train.obs['dt'] = 'train'
atac_val = sc.read_h5ad('atac_val.h5ad')
atac_val.obs['dt'] = 'val'
atac_test = sc.read_h5ad('atac_test.h5ad')
atac_test.obs['dt'] = 'test'
atac = ad.concat([atac_train, atac_val, atac_test])
atac.var = atac_train.var
del atac.var['n_cells']
atac.write('atac.h5ad')


######## clustering using peaks ########
import scanpy as sc
import snapatac2 as snap

atac_val = sc.read_h5ad('atac_val.h5ad')
snap.pp.select_features(atac_val)
snap.tl.spectral(atac_val)
snap.tl.umap(atac_val)
snap.pp.knn(atac_val)
snap.tl.leiden(atac_val)
atac_val.write('atac_val_leiden.h5ad')

atac_test = sc.read_h5ad('atac_test.h5ad')
snap.pp.select_features(atac_test)
snap.tl.spectral(atac_test)
snap.tl.umap(atac_test)
snap.pp.knn(atac_test)
snap.tl.leiden(atac_test)
atac_test.write('atac_test_leiden.h5ad')

atac_val = sc.read_h5ad('atac_val_leiden.h5ad')
sc.pl.umap(atac_val, color='leiden', legend_fontsize='7', legend_loc='right margin', size=5,
           title='', frameon=True, save='_val_true.pdf')

atac_test = sc.read_h5ad('atac_test_leiden.h5ad')
sc.pl.umap(atac_test, color='leiden', legend_fontsize='7', legend_loc='right margin', size=5,
           title='', frameon=True, save='_test_true.pdf')

########## scm2m ##########
python data_preprocess.py -r rna_train.h5ad -a atac_train.h5ad -s preprocessed_data_train --dt train --config rna2atac_config_train.yaml
python data_preprocess.py -r rna_val.h5ad -a atac_val.h5ad -s preprocessed_data_val --dt val --config rna2atac_config_val_eval.yaml
python data_preprocess.py -r rna_test.h5ad -a atac_test.h5ad -s preprocessed_data_test --dt test --config rna2atac_config_val_eval.yaml

accelerate launch --config_file accelerator_config.yaml --main_process_port 29821 rna2atac_train.py --config_file rna2atac_config_train.yaml \
                  --train_data_dir ./preprocessed_data_train --val_data_dir ./preprocessed_data_val -n rna2atac_train

accelerate launch --config_file accelerator_config.yaml --main_process_port 29822 rna2atac_evaluate.py -d ./preprocessed_data_val \
                  -l save/2024-06-16_rna2atac_train_29/pytorch_model.bin --config_file rna2atac_config_val_eval.yaml
mv predict.npy val_predict.npy
python npy2h5ad_val.py 
mv rna2atac_scm2mval.h5ad ../benchmark

accelerate launch --config_file accelerator_config.yaml --main_process_port 29822 rna2atac_evaluate.py -d ./preprocessed_data_test \
                  -l save/2024-06-16_rna2atac_train_29/pytorch_model.bin --config_file rna2atac_config_val_eval.yaml
mv predict.npy test_predict.npy
python npy2h5ad_test.py 
mv rna2atac_scm2mtest.h5ad ../benchmark


########## scbutterfly ##########
python scbt_b.py
python scbt_b_predict.py


########## BABEL ##########
## concat train & valid dataset
import scanpy as sc
import anndata as ad

rna_train = sc.read_h5ad('rna_train.h5ad')
rna_val = sc.read_h5ad('rna_val.h5ad')
rna_train_val = ad.concat([rna_train, rna_val])
rna_train_val.var = rna_train.var[['gene_ids', 'feature_types']]
rna_train_val.write('rna_train_val.h5ad')

atac_train = sc.read_h5ad('atac_train.h5ad')
atac_val = sc.read_h5ad('atac_val.h5ad')
atac_train_val = ad.concat([atac_train, atac_val])
atac_train_val.var = atac_train.var[['gene_ids', 'feature_types']]
atac_train_val.write('atac_train_val.h5ad')

python h5ad2h5.py -n train_val
python h5ad2h5.py -n val
python h5ad2h5.py -n test
python /fs/home/jiluzhang/BABEL/bin/train_model.py --data train_val.h5 --outdir babel_train_out --batchsize 512 --earlystop 25 --device 3 --nofilter
python /fs/home/jiluzhang/BABEL/bin/predict_model.py --checkpoint babel_train_out --data val.h5 --outdir babel_val_out --device 2 \
                                                     --nofilter --noplot --transonly
python /fs/home/jiluzhang/BABEL/bin/predict_model.py --checkpoint babel_train_out --data test.h5 --outdir babel_test_out --device 3 \
                                                     --nofilter --noplot --transonly


python cal_auroc_auprc.py --pred rna2atac_babelval.h5ad --true atac_val_leiden.h5ad
python cal_cluster_plot.py --pred rna2atac_babelval.h5ad --true atac_val_leiden.h5ad
python cal_auroc_auprc.py --pred rna2atac_babeltest.h5ad --true atac_test_leiden.h5ad
python cal_cluster_plot.py --pred rna2atac_babeltest.h5ad --true atac_test_leiden.h5ad

python cal_auroc_auprc.py --pred rna2atac_scbtval.h5ad --true atac_val_leiden.h5ad
python cal_cluster_plot.py --pred rna2atac_scbtval.h5ad --true atac_val_leiden.h5ad
python cal_auroc_auprc.py --pred rna2atac_scbttest.h5ad --true atac_test_leiden.h5ad
python cal_cluster_plot.py --pred rna2atac_scbttest.h5ad --true atac_test_leiden.h5ad

python cal_auroc_auprc.py --pred rna2atac_scm2mval.h5ad --true atac_val_leiden.h5ad
python cal_cluster_plot.py --pred rna2atac_scm2mval.h5ad --true atac_val_leiden.h5ad
python cal_auroc_auprc.py --pred rna2atac_scm2mtest.h5ad --true atac_test_leiden.h5ad
python cal_cluster_plot.py --pred rna2atac_scm2mtest.h5ad --true atac_test_leiden.h5ad



