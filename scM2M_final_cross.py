# CM3542C2-T1  CRC  6,831

# Tumor B         ->  Tumor B
# Tumor B + CRC  ->  Tumor B

# /fs/home/jiluzhang/2023_nature_LD/strict_filtered_out/CM354C2-T1_rna_aligned.h5ad
# /fs/home/jiluzhang/2023_nature_LD/strict_filtered_out/CM354C2-T1_atac_aligned.h5ad

######## select genes & peaks ########
import scanpy as sc

tumor_B_train_rna = sc.read_h5ad('rna_tumor_B_train_filtered_0.h5ad')
crc_rna = sc.read_h5ad('/fs/home/jiluzhang/2023_nature_LD/strict_filtered_out/CM354C2-T1_rna_aligned.h5ad')
crc_rna[:, tumor_B_train_rna.var.index].write('crc_rna_0.h5ad')

tumor_B_train_atac = sc.read_h5ad('atac_tumor_B_train_filtered_0.h5ad')
crc_atac = sc.read_h5ad('/fs/home/jiluzhang/2023_nature_LD/strict_filtered_out/CM354C2-T1_atac_aligned.h5ad')
crc_atac[:, tumor_B_train_atac.var.index].write('crc_atac_0.h5ad')


######## preprocess & training & predicting ########
python rna2atac_data_preprocess.py --config_file rna2atac_config.yaml 
accelerate launch --config_file accelerator_config.yaml rna2atac_pretrain.py --config_file rna2atac_config.yaml -d ./preprocessed_data_tumor_B_crc -n rna2atac_tumor_B_crc

# python rna2atac_data_preprocess_whole.py --config_file rna2atac_config_whole.yaml
accelerate launch --config_file accelerator_config.yaml --main_process_port 29821 rna2atac_evaluate.py -d ./preprocessed_data_whole_test_all -l save/2024-05-21_rna2atac_tumor_B_crc_15/pytorch_model.bin --config_file rna2atac_config_whole.yaml
