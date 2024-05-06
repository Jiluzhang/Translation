## rna2atac_config.yaml  rna2atac_data_preprocess.py  rna2atac_evaluate.py  rna2atac_pretrain.py

# pan_cancer_atac_dataset_0.h5ad  head 1000
# pan_cancer_rna_dataset_0.h5ad   head 1000

python rna2atac_data_preprocess.py  # ~6.5 min
accelerate launch --config_file accelerator_config.yaml rna2atac_pretrain.py -d ./preprocessed_data -n rna2atac_test
