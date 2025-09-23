## workdir: /fs/home/jiluzhang/TF_grammar/cnn_bias_model
## human nakedDNA: https://github.com/ZhangLab-TJ/FootTrack/blob/main/bias/human/nakedDNA.pickle

conda create --name ACCESS_ATAC python=3.9
conda install numpy
conda install -c conda-forge pytorch-gpu
pip install pyBigWig pyranges pysam pyfaidx
pip install deeptools

wget -c http://hgdownload.cse.ucsc.edu/admin/exe/linux.x86_64/wigToBigWig

scp -P 10022 u21509@logini.tongji.edu.cn:/share/home/u21509/workspace/wuang/01.cFOOT-seq/final_data/human/HepG2_naked_DNA/rawdata/rep1/ordinary/nakedDNA.bw \
             /fs/home/jiluzhang/TF_grammar/cnn_bias_model/data/human/human_nakedDNA.bw
scp -P 10022 u21509@logini.tongji.edu.cn:/share/home/u21509/workspace/wangheng/105.JM110_gDNA_cFOOT_2/add/05.conversionProfile/ordinary/JM110_gDNA_0.5U.bw \
             /fs/home/jiluzhang/TF_grammar/cnn_bias_model/data/ecoli/ecoli_nakedDNA.bw
scp -P 10022 u21509@logini.tongji.edu.cn:/share/home/u21509/workspace/reference/e_coli/genome.fa \
             /fs/home/jiluzhang/TF_grammar/cnn_bias_model/data/ecoli
scp -P 10022 u21509@logini.tongji.edu.cn:/share/home/u21509/workspace/wuang/01.cFOOT-seq/final_data/human/HepG2/rawdata/rep1/ordinary/HepG2_7.5U.bw \
             /fs/home/jiluzhang/TF_grammar/cnn_bias_model/data/ecoli
scp -P 10022 u21509@logini.tongji.edu.cn:/share/home/u21509/workspace/reference/Homo_sapiens/GRCh38/cell/HepG2/tf/cor/ctcf/ctcf_raw.bed\
             /fs/home/jiluzhang/TF_grammar/cnn_bias_model/data/ecoli
scp -P 10022 u21509@logini.tongji.edu.cn:/share/home/u21509/workspace/reference/Homo_sapiens/GRCh38/cell/HepG2/tf/cor/atf3/atf3_raw.bed\
             /fs/home/jiluzhang/TF_grammar/cnn_bias_model/data/ecoli

# import pickle

# with open('nakedDNA_human.pickle', 'rb') as file:
#     df = pickle.load(file)

## /share/home/u21509/workspace/wuang/04.tf_grammer/01.bias_correct
## /share/home/u21509/workspace/reference

##############################################################################################################################
## extract regions from bigwig files
import pyBigWig
import os

def extract_region_to_bigwig(input_bw, output_bw, chromosome, start, end):
    """
    提取特定区域并生成新的BigWig文件
    
    Args:
        input_bw (str): 输入BigWig文件路径
        output_bw (str): 输出BigWig文件路径
        chromosome (str): 染色体名称，如 'chr1'
        start (int): 起始位置
        end (int): 结束位置
    """
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_bw) if os.path.dirname(output_bw) else '.', exist_ok=True)
    
    try:
        with pyBigWig.open(input_bw) as bw_in:
            # 检查染色体是否存在
            if chromosome not in bw_in.chroms():
                raise ValueError(f"染色体 {chromosome} 不在文件中")
            
            # 检查区域有效性
            chrom_length = bw_in.chroms()[chromosome]
            if start < 0 or end > chrom_length or start >= end:
                raise ValueError(f"无效的区域范围: {start}-{end} (染色体长度: {chrom_length})")
            
            # 获取特定区域的区间数据
            intervals = bw_in.intervals(chromosome, start, end)
            
            if not intervals:
                print(f"警告: 区域 {chromosome}:{start}-{end} 没有数据")
                return False
            
            print(f"找到 {len(intervals)} 个区间在区域 {chromosome}:{start}-{end}")
            
            # 创建新的BigWig文件
            with pyBigWig.open(output_bw, "w") as bw_out:
                # 添加染色体头信息（使用原始长度或区域长度）
                bw_out.addHeader([(chromosome, chrom_length)])
                
                # 添加提取的区间数据
                for interval in intervals:
                    bw_out.addEntries(
                        [chromosome],
                        [interval[0]],
                        ends=[interval[1]],
                        values=[interval[2]]
                    )
            
            print(f"成功生成: {output_bw}")
            return True
            
    except Exception as e:
        print(f"处理失败: {e}")
        return False

# 使用示例
extract_region_to_bigwig(input_bw='ecoli_nakedDNA.bw', output_bw='regions_train.bw', 
                         chromosome='NC_000913.3', start=1, end=3641652)
extract_region_to_bigwig(input_bw='ecoli_nakedDNA.bw', output_bw='regions_valid.bw', 
                         chromosome='NC_000913.3', start=3641652, end=4141651)
extract_region_to_bigwig(input_bw='ecoli_nakedDNA.bw', output_bw='regions_test.bw', 
                         chromosome='NC_000913.3', start=4141652, end=4641651)
extract_region_to_bigwig(input_bw='human_nakedDNA.bw', output_bw='regions_test_human.bw', 
                         chromosome='chr21', start=1, end=46709983)
##############################################################################################################################

## /fs/home/jiluzhang/TF_grammar/cnn_bias_model/data/ecoli
## epoch=100
python /fs/home/jiluzhang/TF_grammar/cnn_bias_model/train.py --bw_file ecoli_nakedDNA.bw --train_regions regions_train.bed --valid_regions regions_valid.bed \
                                                             --ref_fasta genome.fa --k 128 --epochs 100 --out_dir . --out_name epoch_100 --seed 0 --batch_size 512

python /fs/home/jiluzhang/TF_grammar/cnn_bias_model/predict.py --regions regions_valid.bed --ref_fasta genome.fa --k 128 --model_path ./epoch_100.pth \
                                                               --chrom_size_file chrom.sizes --out_dir . --out_name epoch_100_valid
multiBigwigSummary bins -b epoch_100_valid.bw regions_valid.bw -o epoch_100_valid.npz --outRawCounts epoch_100_valid.tab \
                        -l pred raw -bs 1 -p 10
grep -v nan epoch_100_valid.tab | sed 1d > epoch_100_valid_nonan.tab
rm epoch_100_valid.tab

python /fs/home/jiluzhang/TF_grammar/cnn_bias_model/predict.py --regions regions_test.bed --ref_fasta genome.fa --k 128 --model_path ./epoch_100.pth \
                                                               --chrom_size_file chrom.sizes --out_dir . --out_name epoch_100_test
multiBigwigSummary bins -b epoch_100_test.bw regions_test.bw -o epoch_100_test.npz --outRawCounts epoch_100_test.tab \
                        -l pred raw -bs 1 -p 10
grep -v nan epoch_100_test.tab | sed 1d > epoch_100_test_nonan.tab
rm epoch_100_test.tab

## epoch=200
python /fs/home/jiluzhang/TF_grammar/cnn_bias_model/train.py --bw_file ecoli_nakedDNA.bw --train_regions regions_train.bed --valid_regions regions_valid.bed \
                                                             --ref_fasta genome.fa --k 128 --epochs 200 --out_dir . --out_name epoch_200 --seed 0 --batch_size 512

python /fs/home/jiluzhang/TF_grammar/cnn_bias_model/predict.py --regions regions_valid.bed --ref_fasta genome.fa --k 128 --model_path ./epoch_200.pth \
                                                               --chrom_size_file chrom.sizes --out_dir . --out_name epoch_200_valid
multiBigwigSummary bins -b epoch_200_valid.bw regions_valid.bw -o epoch_200_valid.npz --outRawCounts epoch_200_valid.tab \
                        -l pred raw -bs 1 -p 10  # ~2 min
grep -v nan epoch_200_valid.tab | sed 1d > epoch_200_valid_nonan.tab
rm epoch_200_valid.tab

python /fs/home/jiluzhang/TF_grammar/cnn_bias_model/predict.py --regions regions_test.bed --ref_fasta genome.fa --k 128 --model_path ./epoch_200.pth \
                                                               --chrom_size_file chrom.sizes --out_dir . --out_name epoch_200_test
multiBigwigSummary bins -b epoch_200_test.bw regions_test.bw -o epoch_200_test.npz --outRawCounts epoch_200_test.tab \
                        -l pred raw -bs 1 -p 10  # ~2 min
grep -v nan epoch_200_test.tab | sed 1d > epoch_200_test_nonan.tab
rm epoch_200_test.tab

## epoch=500
python /fs/home/jiluzhang/TF_grammar/cnn_bias_model/train.py --bw_file ecoli_nakedDNA.bw --train_regions regions_train.bed --valid_regions regions_valid.bed \
                                                             --ref_fasta genome.fa --k 128 --epochs 500 --out_dir . --out_name epoch_500 --seed 0 --batch_size 512

python /fs/home/jiluzhang/TF_grammar/cnn_bias_model/predict.py --regions regions_valid.bed --ref_fasta genome.fa --k 128 --model_path ./epoch_500.pth \
                                                               --chrom_size_file chrom.sizes --out_dir . --out_name epoch_500_valid
multiBigwigSummary bins -b epoch_500_valid.bw regions_valid.bw -o epoch_500_valid.npz --outRawCounts epoch_500_valid.tab \
                        -l pred raw -bs 1 -p 10
grep -v nan epoch_500_valid.tab | sed 1d > epoch_500_valid_nonan.tab
rm epoch_500_valid.tab

python /fs/home/jiluzhang/TF_grammar/cnn_bias_model/predict.py --regions regions_test.bed --ref_fasta genome.fa --k 128 --model_path ./epoch_500.pth \
                                                               --chrom_size_file chrom.sizes --out_dir . --out_name epoch_500_test
multiBigwigSummary bins -b epoch_500_test.bw regions_test.bw -o epoch_500_test.npz --outRawCounts epoch_500_test.tab \
                        -l pred raw -bs 1 -p 10
grep -v nan epoch_500_test.tab | sed 1d > epoch_500_test_nonan.tab
rm epoch_500_test.tab

## human naked DNA chr21 test (chr21:46209983-46709983)
python /fs/home/jiluzhang/TF_grammar/cnn_bias_model/predict.py --regions regions_test_human.bed --ref_fasta ../human/hg38.fa --k 128 --model_path ./epoch_500.pth \
                                                               --chrom_size_file ../human/hg38.chrom.sizes --out_dir . --out_name epoch_500_test_human
multiBigwigSummary bins -b epoch_500_test_human.bw regions_test_human.bw -o epoch_500_test_human.npz --outRawCounts epoch_500_test_human.tab \
                        -l pred raw -bs 1 -p 10
grep -v nan epoch_500_test_human.tab | sed 1d > epoch_500_test_human_nonan.tab
rm epoch_500_test_human.tab

## human HEG2 chr21 test (chr21:46209983-46709983)
## predict.py (line 15  os.environ["CUDA_VISIBLE_DEVICES"] = '1')
python /fs/home/jiluzhang/TF_grammar/cnn_bias_model/predict.py --regions regions_test_human.bed --ref_fasta ../human/hg38.fa --k 128 --model_path ./epoch_500.pth \
                                                               --chrom_size_file ../human/hg38.chrom.sizes --out_dir . --out_name epoch_500_test_heg2  # ~6 min

# np.divide -> np.subtract
python /fs/home/jiluzhang/TF_grammar/ACCESS-ATAC/bias_correction/correct_bias.py --bw_raw HepG2_7.5U.bw --bw_bias epoch_500_test_heg2.bw \
                                                                                 --bed_file regions_test_human_2.bed --extend 0 --window 101 \
                                                                                 --pseudo_count 0.001 --out_dir . --out_name epoch_500_test_heg2_corrected \
                                                                                 --chrom_size_file ../human/hg38.chrom.sizes  # ~10.5 min

python /fs/home/jiluzhang/TF_grammar/ACCESS-ATAC/bias_correction/correct_bias.py --bw_raw HepG2_7.5U.bw --bw_bias human_nakedDNA.bw \
                                                                                 --bed_file regions_test_human_2.bed --extend 0 --window 101 \
                                                                                 --pseudo_count 0.001 --out_dir . --out_name epoch_500_test_heg2_corrected \
                                                                                 --chrom_size_file ../human/hg38.chrom.sizes  # ~1min / 10Mb

grep chr21 ctcf_raw.bed | awk '{if($2>30000000 && $3<40000000) print$0}' > ctcf_raw_chr21.bed
computeMatrix reference-point --referencePoint center -p 10 -S HepG2_7.5U.bw epoch_500_test_heg2_corrected.norm.bw epoch_500_test_heg2_corrected.exp.bw human_nakedDNA.bw \
                              -R ctcf_raw_chr21.bed -o raw_corrected.gz -a 50 -b 50 -bs 1
plotProfile -m raw_corrected.gz --yMin -0.2 --yMax 0.6 --perGroup -out raw_corrected.pdf  # the norm all equal to zero!!!

grep chr21 atf3_raw.bed | awk '{if($2>30000000 && $3<40000000) print$0}' > atf3_raw_chr21.bed
computeMatrix reference-point --referencePoint center -p 10 -S HepG2_7.5U.bw epoch_500_test_heg2_corrected.norm.bw epoch_500_test_heg2_corrected.exp.bw human_nakedDNA.bw \
                              -R atf3_raw.bed -o raw_corrected.gz -a 100 -b 100 -bs 5
plotProfile -m raw_corrected.gz --yMin -0.2 --yMax 0.6 --perGroup -out raw_corrected.pdf  # the norm all equal to zero!!!

##############################################################################################################################
## ../cal_cor --file epoch_100_valid_nonan.tab          # 0.7081576742099914  0.6798156516658299
## ../cal_cor --file epoch_100_test_nonan.tab           # 0.7082189028293715  0.6790170844020186
## ../cal_cor --file epoch_200_valid_nonan.tab          # 0.7395414336137928  0.7112723570920626
## ../cal_cor --file epoch_200_test_nonan.tab           # 0.7403168268301135  0.7110146291063021
## ../cal_cor --file epoch_500_valid_nonan.tab          # 0.7416945008705992  0.7134883683857012
## ../cal_cor --file epoch_500_test_nonan.tab           # 0.7417070864468944  0.7126470645540367
## ../cal_cor --file epoch_500_test_human_nonan.tab     # 0.2263957077458701  0.22646666457620462

#!/fs/home/jiluzhang/softwares/miniconda3/envs/ACCESS_ATAC/bin/python
import pandas as pd
from scipy import stats
import argparse

parser = argparse.ArgumentParser(description='Calculate correlation between prediction and ground truth')
parser.add_argument('--file', type=str, help='tab file')

args = parser.parse_args()
tab_file = args.file

def main():
    df = pd.read_csv(tab_file, sep='\t', header=None)
    df.columns = ['chrom', 'start', 'end', 'pred', 'raw']
    pearson_corr, _ = stats.pearsonr(df['pred'], df['raw'])
    spearman_corr, _ = stats.spearmanr(df['pred'], df['raw'])
    print(f"Pearson R: {pearson_corr}")
    print(f"Spearman R: {spearman_corr}")

if __name__ == "__main__":
    main()
##############################################################################################################################























extract_region_to_bigwig(input_file, output_file, chromosome, start_pos, end_pos)
