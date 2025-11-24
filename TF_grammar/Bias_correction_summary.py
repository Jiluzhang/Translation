############## ACCESS_ATAC ##############
## train CNN model
python ../access_atac/cnn_bias_model/train.py --bw_file ecoli_nakedDNA.bw --train_regions regions_train.bed --valid_regions regions_valid.bed \
                                              --ref_fasta ecoli.fa --k 64 --epochs 200 --out_dir . --out_name ecoli_bias --seed 0 --batch_size 512
## predict ecoli testing bias
python ../access_atac/cnn_bias_model/predict.py --regions regions_test.bed --ref_fasta ecoli.fa --k 64 --model_path ecoli_bias.pth \
                                                --chrom_size_file ecoli_chrom.sizes --out_dir . --out_name ecoli_test_pred

## calculate correlation between prediction and ground truth
./multiBigwigSummary bins -b ecoli_test_pred.bw ecoli_test_true.bw -o ecoli_test_pred_true.npz --outRawCounts ecoli_test_pred_true.tab \
                          -l pred true -bs 1 -p 10
grep -v nan ecoli_test_pred_true.tab | sed 1d > ecoli_test_pred_true_nonan.tab
rm ecoli_test_pred_true.tab
./cal_cor --file ecoli_test_pred_true_nonan.tab
# Pearson R: 0.8809306932571868
# Spearman R: 0.863951765993428

## predict human bias
python ../access_atac/cnn_bias_model/predict.py --regions regions_test_human.bed --ref_fasta hg38.fa --k 64 --model_path ecoli_bias.pth \
                                                --chrom_size_file hg38_chrom.sizes --out_dir . --out_name human_bias
  
## correct bias
python ../access_atac/bias_correction/correct_bias.py --bw_raw HepG2_7.5U.bw --bw_bias human_bias.bw \
                                                      --bed_file regions_test.bed --extend 0 --window 101 \
                                                      --out_dir . --out_name ecoli_to_human_naked_dna_corrected \
                                                      --chrom_size_file ../hg38.chrom.sizes  # ~5 min for chr21
