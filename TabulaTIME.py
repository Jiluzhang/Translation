## Previous pan-cancer model: /fs/home/jiluzhang/scM2M_no_dec_attn/pan_cancer/save/2024-06-25_rna2atac_train_3/pytorch_model.bin
## Current model with only decoder

## workdir: /fs/home/jiluzhang/scM2M_no_dec_attn/pan_cancer/all_data/data_with_annotation/h5ad/scM2M/TabulaTIME
## copy files
for sample in `ls /fs/home/jiluzhang/scM2M_no_dec_attn/pan_cancer/all_data/data_with_annotation/h5ad/scM2M/preprocessed_data_train`;do
    cp /fs/home/jiluzhang/scM2M_no_dec_attn/pan_cancer/all_data/data_with_annotation/h5ad/scM2M/preprocessed_data_train/$sample/*filtered.h5ad .
    cp /fs/home/jiluzhang/scM2M_no_dec_attn/pan_cancer/all_data/data_with_annotation/h5ad/scM2M/preprocessed_data_train/$sample/*info.txt .
done

for sample in `ls /fs/home/jiluzhang/scM2M_no_dec_attn/pan_cancer/all_data/data_with_annotation/h5ad | grep VF`;do
    cp /fs/home/jiluzhang/scM2M_no_dec_attn/pan_cancer/all_data/data_with_annotation/h5ad/$sample/*filtered.h5ad .
    cp /fs/home/jiluzhang/scM2M_no_dec_attn/pan_cancer/all_data/data_with_annotation/h5ad/$sample/*info.txt .
done

cp /fs/home/jiluzhang/scM2M_no_dec_attn/pan_cancer/all_data/data_with_annotation/h5ad/scM2M/preprocessed_data_val/CPT704DU-M1/*filtered.h5ad .
cp /fs/home/jiluzhang/scM2M_no_dec_attn/pan_cancer/all_data/data_with_annotation/h5ad/scM2M/preprocessed_data_val/CPT704DU-M1/*info.txt .
