## rna2atac_config.yaml  rna2atac_data_preprocess.py  rna2atac_evaluate.py  rna2atac_pretrain.py

# pan_cancer_atac_dataset_0.h5ad  head 1000
# pan_cancer_rna_dataset_0.h5ad   head 1000

python rna2atac_data_preprocess.py  # ~6.5 min
accelerate launch --config_file accelerator_config.yaml rna2atac_pretrain.py -d ./preprocessed_data -n rna2atac_test

python rna2atac_data_preprocess.py --config_file rna2atac_config_whole.yaml
accelerate launch --config_file accelerator_config.yaml rna2atac_evaluate.py -d ./preprocessed_data_whole -n rna2atac_test -l save_01/2024-05-06_rna2atac_test/pytorch_model.bin
