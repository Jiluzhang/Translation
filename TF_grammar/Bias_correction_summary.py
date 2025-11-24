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
