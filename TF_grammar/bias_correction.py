## workdir: /fs/home/jiluzhang/TF_grammar/cnn_bias_model
## human nakedDNA: https://github.com/ZhangLab-TJ/FootTrack/blob/main/bias/human/nakedDNA.pickle

conda create --name ACCESS_ATAC python=3.9
conda install numpy
conda install -c conda-forge pytorch-gpu
pip install pyBigWig pyranges pysam pyfaidx
pip install deeptools
conda install bioconda::bedtools


wget -c http://hgdownload.cse.ucsc.edu/admin/exe/linux.x86_64/wigToBigWig

## files download
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
scp -P 10022 u21509@logini.tongji.edu.cn:/share/home/u21509/workspace/reference/Homo_sapiens/GRCh38/cell/HepG2/tf/cor/atf2/atf2_raw.bed\
             /fs/home/jiluzhang/TF_grammar/cnn_bias_model/data/ecoli

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
                         chromosome='NC_000913.3', start=1, end=4581651)
extract_region_to_bigwig(input_bw='ecoli_nakedDNA.bw', output_bw='regions_valid.bw', 
                         chromosome='NC_000913.3', start=4581652, end=4611651)
extract_region_to_bigwig(input_bw='ecoli_nakedDNA.bw', output_bw='regions_test.bw', 
                         chromosome='NC_000913.3', start=4611652, end=4641651)
extract_region_to_bigwig(input_bw='human_nakedDNA.bw', output_bw='regions_test_human.bw', 
                         chromosome='chr21', start=46679983, end=46709983)
##############################################################################################################################

##############################################################################################################################
## ../cal_cor --file epoch_100_valid_nonan.tab          # 0.7081576742099914  0.6798156516658299

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


## human to human (/fs/home/jiluzhang/TF_grammar/cnn_bias_model/data/human_to_human)
# correct_bias.py (line 107: np.divide -> np.subtract)
python /fs/home/jiluzhang/TF_grammar/ACCESS-ATAC/bias_correction/correct_bias.py --bw_raw HepG2_7.5U.bw --bw_bias human_nakedDNA.bw \
                                                                                 --bed_file regions_test.bed --extend 0 --window 101 \
                                                                                 --pseudo_count 1 --out_dir . --out_name human_naked_dna_corrected \
                                                                                 --chrom_size_file hg38.chrom.sizes  # ~5 min for chr21
## All peaks
## CTCF
scp -P 10022 u21509@logini.tongji.edu.cn:/share/home/u21509/workspace/reference/Homo_sapiens/GRCh38/cell/HepG2/tf/chip/ctcf/ctcf.bed\
             /fs/home/jiluzhang/TF_grammar/cnn_bias_model/data/human_to_human
awk '{if($1=="chr21") print($1"\t"$2"\t"$3)}' ctcf.bed | sort -k1,1 -k2,2n > ctcf_chr21_raw.bed
bedtools merge -i ctcf_chr21_raw.bed > ctcf_chr21.bed  # 1346
rm ctcf_chr21_raw.bed

computeMatrix reference-point --referencePoint center -p 10 -S HepG2_7.5U.bw -R ctcf_chr21.bed -o HepG2_7.5U_ctcf.gz -a 50 -b 50 -bs 1
plotProfile -m HepG2_7.5U_ctcf.gz --yMin 0.31 --yMax 0.41 -out HepG2_7.5U_ctcf.pdf

computeMatrix reference-point --referencePoint center -p 10 -S human_nakedDNA.bw -R ctcf_chr21.bed -o human_nakedDNA_ctcf.gz -a 50 -b 50 -bs 1
plotProfile -m human_nakedDNA_ctcf.gz --yMin 0.35 --yMax 0.45 -out human_nakedDNA_ctcf.pdf

computeMatrix reference-point --referencePoint center -p 10 -S human_naked_dna_corrected.norm.bw -R ctcf_chr21.bed -o human_naked_dna_corrected_ctcf.gz -a 50 -b 50 -bs 1
plotProfile -m human_naked_dna_corrected_ctcf.gz --yMin -0.05 --yMax 0.05 -out human_naked_dna_corrected_ctcf.pdf

## ATF3
scp -P 10022 u21509@logini.tongji.edu.cn:/share/home/u21509/workspace/reference/Homo_sapiens/GRCh38/cell/HepG2/tf/chip/atf3/atf3.bed\
             /fs/home/jiluzhang/TF_grammar/cnn_bias_model/data/human_to_human
awk '{if($1=="chr21") print($1"\t"$2"\t"$3)}' atf3.bed | sort -k1,1 -k2,2n > atf3_chr21_raw.bed
bedtools merge -i atf3_chr21_raw.bed > atf3_chr21.bed  # 826
rm atf3_chr21_raw.bed

computeMatrix reference-point --referencePoint center -p 10 -S HepG2_7.5U.bw -R atf3_chr21.bed -o HepG2_7.5U_atf3.gz -a 50 -b 50 -bs 1
plotProfile -m HepG2_7.5U_atf3.gz --yMin 0.27 --yMax 0.37 -out HepG2_7.5U_atf3.pdf

computeMatrix reference-point --referencePoint center -p 10 -S human_nakedDNA.bw -R atf3_chr21.bed -o human_nakedDNA_atf3.gz -a 50 -b 50 -bs 1
plotProfile -m human_nakedDNA_atf3.gz --yMin 0.35 --yMax 0.45 -out human_nakedDNA_atf3.pdf

computeMatrix reference-point --referencePoint center -p 10 -S human_naked_dna_corrected.norm.bw -R atf3_chr21.bed -o human_naked_dna_corrected_atf3.gz -a 50 -b 50 -bs 1
plotProfile -m human_naked_dna_corrected_atf3.gz --yMin -0.025 --yMax 0.025 -out human_naked_dna_corrected_atf3.pdf

## ELF4
scp -P 10022 u21509@logini.tongji.edu.cn:/share/home/u21509/workspace/reference/Homo_sapiens/GRCh38/cell/HepG2/tf/chip/elf4/elf4.bed\
             /fs/home/jiluzhang/TF_grammar/cnn_bias_model/data/human_to_human
awk '{if($1=="chr21") print($1"\t"$2"\t"$3)}' elf4.bed | sort -k1,1 -k2,2n > elf4_chr21_raw.bed
bedtools merge -i elf4_chr21_raw.bed > elf4_chr21.bed  # 410
rm elf4_chr21_raw.bed

computeMatrix reference-point --referencePoint center -p 10 -S HepG2_7.5U.bw -R elf4_chr21.bed -o HepG2_7.5U_elf4.gz -a 50 -b 50 -bs 1
plotProfile -m HepG2_7.5U_elf4.gz --yMin 0.27 --yMax 0.37 -out HepG2_7.5U_elf4.pdf

computeMatrix reference-point --referencePoint center -p 10 -S human_nakedDNA.bw -R elf4_chr21.bed -o human_nakedDNA_elf4.gz -a 50 -b 50 -bs 1
plotProfile -m human_nakedDNA_elf4.gz --yMin 0.33 --yMax 0.46 -out human_nakedDNA_elf4.pdf

computeMatrix reference-point --referencePoint center -p 10 -S human_naked_dna_corrected.norm.bw -R elf4_chr21.bed -o human_naked_dna_corrected_elf4.gz -a 50 -b 50 -bs 1
plotProfile -m human_naked_dna_corrected_elf4.gz --yMin -0.025 --yMax 0.035 -out human_naked_dna_corrected_elf4.pdf


## Peaks with motif
## CTCF
scp -P 10022 u21509@logini.tongji.edu.cn:/share/home/u21509/workspace/reference/Homo_sapiens/GRCh38/cell/HepG2/tf/cor/ctcf/ctcf_raw.bed\
             /fs/home/jiluzhang/TF_grammar/cnn_bias_model/data/human_to_human
awk '{if($1=="chr21") print($1"\t"$2"\t"$3)}' ctcf_raw.bed | sort -k1,1 -k2,2n > ctcf_chr21_raw.bed
bedtools merge -i ctcf_chr21_raw.bed > ctcf_chr21.bed  # 664
rm ctcf_chr21_raw.bed

computeMatrix reference-point --referencePoint center -p 10 -S HepG2_7.5U.bw -R ctcf_chr21.bed -o HepG2_7.5U_ctcf.gz -a 50 -b 50 -bs 1
plotProfile -m HepG2_7.5U_ctcf.gz --yMin 0.25 --yMax 0.55 -out HepG2_7.5U_ctcf.pdf

computeMatrix reference-point --referencePoint center -p 10 -S human_nakedDNA.bw -R ctcf_chr21.bed -o human_nakedDNA_ctcf.gz -a 50 -b 50 -bs 1
plotProfile -m human_nakedDNA_ctcf.gz --yMin 0.33 --yMax 0.52 -out human_nakedDNA_ctcf.pdf

computeMatrix reference-point --referencePoint center -p 10 -S human_naked_dna_corrected.norm.bw -R ctcf_chr21.bed -o human_naked_dna_corrected_ctcf.gz -a 50 -b 50 -bs 1
plotProfile -m human_naked_dna_corrected_ctcf.gz --yMin -0.14 --yMax 0.09 -out human_naked_dna_corrected_ctcf.pdf

## ATF3
scp -P 10022 u21509@logini.tongji.edu.cn:/share/home/u21509/workspace/reference/Homo_sapiens/GRCh38/cell/HepG2/tf/cor/atf3/atf3_raw.bed\
             /fs/home/jiluzhang/TF_grammar/cnn_bias_model/data/human_to_human
awk '{if($1=="chr21") print($1"\t"$2"\t"$3)}' atf3_raw.bed | sort -k1,1 -k2,2n > atf3_chr21_raw.bed
bedtools merge -i atf3_chr21_raw.bed > atf3_chr21.bed  # 24
rm atf3_chr21_raw.bed

computeMatrix reference-point --referencePoint center -p 10 -S HepG2_7.5U.bw -R atf3_chr21.bed -o HepG2_7.5U_atf3.gz -a 50 -b 50 -bs 2  # bin_size=2
plotProfile -m HepG2_7.5U_atf3.gz --yMin 0.08 --yMax 0.52 -out HepG2_7.5U_atf3.pdf

computeMatrix reference-point --referencePoint center -p 10 -S human_nakedDNA.bw -R atf3_chr21.bed -o human_nakedDNA_atf3.gz -a 50 -b 50 -bs 2
plotProfile -m human_nakedDNA_atf3.gz --yMin 0.18 --yMax 0.62 -out human_nakedDNA_atf3.pdf

computeMatrix reference-point --referencePoint center -p 10 -S human_naked_dna_corrected.norm.bw -R atf3_chr21.bed -o human_naked_dna_corrected_atf3.gz -a 50 -b 50 -bs 2
plotProfile -m human_naked_dna_corrected_atf3.gz --yMin -0.12 --yMax 0.12 -out human_naked_dna_corrected_atf3.pdf

## ELF4
scp -P 10022 u21509@logini.tongji.edu.cn:/share/home/u21509/workspace/reference/Homo_sapiens/GRCh38/cell/HepG2/tf/cor/elf4/elf4_raw.bed\
             /fs/home/jiluzhang/TF_grammar/cnn_bias_model/data/human_to_human
awk '{if($1=="chr21") print($1"\t"$2"\t"$3)}' elf4_raw.bed | sort -k1,1 -k2,2n > elf4_chr21_raw.bed
bedtools merge -i elf4_chr21_raw.bed > elf4_chr21.bed  # 108
rm elf4_chr21_raw.bed

computeMatrix reference-point --referencePoint center -p 10 -S HepG2_7.5U.bw -R elf4_chr21.bed -o HepG2_7.5U_elf4.gz -a 50 -b 50 -bs 1
plotProfile -m HepG2_7.5U_elf4.gz --yMin 0.28 --yMax 0.52 -out HepG2_7.5U_elf4.pdf

computeMatrix reference-point --referencePoint center -p 10 -S human_nakedDNA.bw -R elf4_chr21.bed -o human_nakedDNA_elf4.gz -a 50 -b 50 -bs 1
plotProfile -m human_nakedDNA_elf4.gz --yMin 0.28 --yMax 0.58 -out human_nakedDNA_elf4.pdf

computeMatrix reference-point --referencePoint center -p 10 -S human_naked_dna_corrected.norm.bw -R elf4_chr21.bed -o human_naked_dna_corrected_elf4.gz -a 50 -b 50 -bs 1
plotProfile -m human_naked_dna_corrected_elf4.gz --yMin -0.17 --yMax 0.12 -out human_naked_dna_corrected_elf4.pdf



############################################################################################################################################
## ecoli to human
## /fs/home/jiluzhang/TF_grammar/cnn_bias_model/data/ecoli_to_human
# train.py (line 18  os.environ["CUDA_VISIBLE_DEVICES"] = '2')
# model.py (line 29: 'nn.Linear(928, 1024),' -> 'nn.Linear(int(32*((self.seq_len/2-2)/2-2)), 1024),')
# predict.py (line 15  os.environ["CUDA_VISIBLE_DEVICES"] = '2')

## k=32 (valid_loss: 0.00496)
python /fs/home/jiluzhang/TF_grammar/cnn_bias_model/train.py --bw_file ecoli_nakedDNA.bw --train_regions regions_train.bed --valid_regions regions_valid.bed \
                                                             --ref_fasta genome.fa --k 32 --epochs 200 --out_dir . --out_name epoch_200_k_32 --seed 0 --batch_size 512
python /fs/home/jiluzhang/TF_grammar/cnn_bias_model/predict.py --regions regions_test.bed --ref_fasta genome.fa --k 32 --model_path ./epoch_200_k_32.pth \
                                                               --chrom_size_file chrom.sizes --out_dir . --out_name epoch_200_k_32_test
multiBigwigSummary bins -b epoch_200_k_32_test.bw regions_test.bw -o epoch_200_k_32_test.npz --outRawCounts epoch_200_k_32_test.tab \
                        -l pred raw -bs 1 -p 10  # ~1 min
grep -v nan epoch_200_k_32_test.tab | sed 1d > epoch_200_k_32_test_nonan.tab
rm epoch_200_k_32_test.tab
../cal_cor --file epoch_200_k_32_test_nonan.tab          # 0.7071614613301089  0.6771305801143438

python /fs/home/jiluzhang/TF_grammar/cnn_bias_model/predict.py --regions regions_test_human.bed --ref_fasta hg38.fa --k 32 --model_path ./epoch_200_k_32.pth \
                                                               --chrom_size_file hg38.chrom.sizes --out_dir . --out_name epoch_200_k_32_test_human
multiBigwigSummary bins -b epoch_200_k_32_test_human.bw regions_test_human.bw -o epoch_200_k_32_test_human.npz --outRawCounts epoch_200_k_32_test_human.tab \
                        -l pred raw -bs 1 -p 10  # ~2.5 min
grep -v nan epoch_200_k_32_test_human.tab | sed 1d > epoch_200_k_32_test_human_nonan.tab
rm epoch_200_k_32_test_human.tab
../cal_cor --file epoch_200_k_32_test_human_nonan.tab          # 0.374381308004477  0.3648879178344095

## k=64 (valid_loss: 0.00488)
python /fs/home/jiluzhang/TF_grammar/cnn_bias_model/train.py --bw_file ecoli_nakedDNA.bw --train_regions regions_train.bed --valid_regions regions_valid.bed \
                                                             --ref_fasta genome.fa --k 64 --epochs 200 --out_dir . --out_name epoch_200_k_64 --seed 0 --batch_size 512
python /fs/home/jiluzhang/TF_grammar/cnn_bias_model/predict.py --regions regions_test.bed --ref_fasta genome.fa --k 64 --model_path ./epoch_200_k_64.pth \
                                                               --chrom_size_file chrom.sizes --out_dir . --out_name epoch_200_k_64_test
multiBigwigSummary bins -b epoch_200_k_64_test.bw regions_test.bw -o epoch_200_k_64_test.npz --outRawCounts epoch_200_k_64_test.tab \
                        -l pred raw -bs 1 -p 10  # ~1 min
grep -v nan epoch_200_k_64_test.tab | sed 1d > epoch_200_k_64_test_nonan.tab
rm epoch_200_k_64_test.tab
../cal_cor --file epoch_200_k_64_test_nonan.tab          # 0.746015664021949  0.7183493816590069

python /fs/home/jiluzhang/TF_grammar/cnn_bias_model/predict.py --regions regions_test_human.bed --ref_fasta hg38.fa --k 64 --model_path ./epoch_200_k_64.pth \
                                                               --chrom_size_file hg38.chrom.sizes --out_dir . --out_name epoch_200_k_64_test_human
multiBigwigSummary bins -b epoch_200_k_64_test_human.bw regions_test_human.bw -o epoch_200_k_64_test_human.npz --outRawCounts epoch_200_k_64_test_human.tab \
                        -l pred raw -bs 1 -p 10  # ~2.5 min
grep -v nan epoch_200_k_64_test_human.tab | sed 1d > epoch_200_k_64_test_human_nonan.tab
rm epoch_200_k_64_test_human.tab
../cal_cor --file epoch_200_k_64_test_human_nonan.tab          # 0.3933705624395759  0.3808398389366913
  
## k=128 (valid_loss: 0.00582)
python /fs/home/jiluzhang/TF_grammar/cnn_bias_model/train.py --bw_file ecoli_nakedDNA.bw --train_regions regions_train.bed --valid_regions regions_valid.bed \
                                                             --ref_fasta genome.fa --k 128 --epochs 200 --out_dir . --out_name epoch_200_k_128 --seed 0 --batch_size 512
python /fs/home/jiluzhang/TF_grammar/cnn_bias_model/predict.py --regions regions_test.bed --ref_fasta genome.fa --k 128 --model_path ./epoch_200_k_128.pth \
                                                               --chrom_size_file chrom.sizes --out_dir . --out_name epoch_200_k_128_test
multiBigwigSummary bins -b epoch_200_k_128_test.bw regions_test.bw -o epoch_200_k_128_test.npz --outRawCounts epoch_200_k_128_test.tab \
                        -l pred raw -bs 1 -p 10  # ~1 min
grep -v nan epoch_200_k_128_test.tab | sed 1d > epoch_200_k_128_test_nonan.tab
rm epoch_200_k_128_test.tab
../cal_cor --file epoch_200_k_128_test_nonan.tab          # 0.7416587365621455  0.7147805662432811

python /fs/home/jiluzhang/TF_grammar/cnn_bias_model/predict.py --regions regions_test_human.bed --ref_fasta hg38.fa --k 128 --model_path ./epoch_200_k_128.pth \
                                                               --chrom_size_file hg38.chrom.sizes --out_dir . --out_name epoch_200_k_128_test_human
multiBigwigSummary bins -b epoch_200_k_128_test_human.bw regions_test_human.bw -o epoch_200_k_128_test_human.npz --outRawCounts epoch_200_k_128_test_human.tab \
                        -l pred raw -bs 1 -p 10  # ~2.5 min
grep -v nan epoch_200_k_128_test_human.tab | sed 1d > epoch_200_k_128_test_human_nonan.tab
rm epoch_200_k_128_test_human.tab
../cal_cor --file epoch_200_k_128_test_human_nonan.tab          # 0.3745718735095749  0.367340934181349

## k=256 (valid_loss: 0.00839)
python /fs/home/jiluzhang/TF_grammar/cnn_bias_model/train.py --bw_file ecoli_nakedDNA.bw --train_regions regions_train.bed --valid_regions regions_valid.bed \
                                                             --ref_fasta genome.fa --k 256 --epochs 200 --out_dir . --out_name epoch_200_k_256 --seed 0 --batch_size 512
python /fs/home/jiluzhang/TF_grammar/cnn_bias_model/predict.py --regions regions_test.bed --ref_fasta genome.fa --k 256 --model_path ./epoch_200_k_256.pth \
                                                               --chrom_size_file chrom.sizes --out_dir . --out_name epoch_200_k_256_test
multiBigwigSummary bins -b epoch_200_k_256_test.bw regions_test.bw -o epoch_200_k_256_test.npz --outRawCounts epoch_200_k_256_test.tab \
                        -l pred raw -bs 1 -p 10  # ~1 min
grep -v nan epoch_200_k_256_test.tab | sed 1d > epoch_200_k_256_test_nonan.tab
rm epoch_200_k_256_test.tab
../cal_cor --file epoch_200_k_256_test_nonan.tab          # 0.6981378483939527  0.674795830345938

python /fs/home/jiluzhang/TF_grammar/cnn_bias_model/predict.py --regions regions_test_human.bed --ref_fasta hg38.fa --k 256 --model_path ./epoch_200_k_256.pth \
                                                               --chrom_size_file hg38.chrom.sizes --out_dir . --out_name epoch_200_k_256_test_human
multiBigwigSummary bins -b epoch_200_k_256_test_human.bw regions_test_human.bw -o epoch_200_k_256_test_human.npz --outRawCounts epoch_200_k_256_test_human.tab \
                        -l pred raw -bs 1 -p 10  # ~2.5 min
grep -v nan epoch_200_k_256_test_human.tab | sed 1d > epoch_200_k_256_test_human_nonan.tab
rm epoch_200_k_256_test_human.tab
../cal_cor --file epoch_200_k_256_test_human_nonan.tab          # 0.3289853953013008  0.32988449274807413

############################################################################################################################################

## Peaks with motif (ecoli_to_human)
python /fs/home/jiluzhang/TF_grammar/cnn_bias_model/predict.py --regions regions_test.bed --ref_fasta ../hg38.fa --k 64 --model_path ../k_64/epoch_200_k_64.pth \
                                                               --chrom_size_file ../hg38.chrom.sizes --out_dir . --out_name epoch_200_k_64_test_human  # ~8.5 min
python /fs/home/jiluzhang/TF_grammar/ACCESS-ATAC/bias_correction/correct_bias.py --bw_raw HepG2_7.5U.bw --bw_bias epoch_200_k_64_test_human.bw \
                                                                                 --bed_file regions_test.bed --extend 0 --window 101 \
                                                                                 --pseudo_count 1 --out_dir . --out_name ecoli_to_human_naked_dna_corrected \
                                                                                 --chrom_size_file ../hg38.chrom.sizes  # ~5 min for chr21

## CTCF
# computeMatrix reference-point --referencePoint center -p 10 -S HepG2_7.5U.bw -R ctcf_chr21.bed -o HepG2_7.5U_ctcf.gz -a 50 -b 50 -bs 1
# plotProfile -m HepG2_7.5U_ctcf.gz --yMin 0.25 --yMax 0.55 -out HepG2_7.5U_ctcf.pdf

computeMatrix reference-point --referencePoint center -p 10 -S epoch_200_k_64_test_human.bw -R ctcf_chr21.bed -o epoch_200_k_64_test_human_ctcf.gz -a 50 -b 50 -bs 1
plotProfile -m epoch_200_k_64_test_human_ctcf.gz --yMin 0.08 --yMax 0.37 -out epoch_200_k_64_test_human_ctcf.pdf

computeMatrix reference-point --referencePoint center -p 10 -S ecoli_to_human_naked_dna_corrected.norm.bw -R ctcf_chr21.bed \
                              -o ecoli_to_human_naked_dna_corrected_ctcf.gz -a 50 -b 50 -bs 1
plotProfile -m ecoli_to_human_naked_dna_corrected_ctcf.gz --yMin -0.18 --yMax 0.08 -out ecoli_to_human_naked_dna_corrected_ctcf.pdf

## ATF3
# computeMatrix reference-point --referencePoint center -p 10 -S HepG2_7.5U.bw -R atf3_chr21.bed -o HepG2_7.5U_atf3.gz -a 50 -b 50 -bs 1
# plotProfile -m HepG2_7.5U_atf3.gz --yMin 0.25 --yMax 0.55 -out HepG2_7.5U_atf3.pdf

computeMatrix reference-point --referencePoint center -p 10 -S epoch_200_k_64_test_human.bw -R atf3_chr21.bed -o epoch_200_k_64_test_human_atf3.gz -a 50 -b 50 -bs 2
plotProfile -m epoch_200_k_64_test_human_atf3.gz --yMin 0.02 --yMax 0.18 -out epoch_200_k_64_test_human_atf3.pdf

computeMatrix reference-point --referencePoint center -p 10 -S ecoli_to_human_naked_dna_corrected.norm.bw -R atf3_chr21.bed \
                              -o ecoli_to_human_naked_dna_corrected_atf3.gz -a 50 -b 50 -bs 2
plotProfile -m ecoli_to_human_naked_dna_corrected_atf3.gz --yMin -0.07 --yMax 0.07 -out ecoli_to_human_naked_dna_corrected_atf3.pdf
    
## ELF4
# computeMatrix reference-point --referencePoint center -p 10 -S HepG2_7.5U.bw -R elf4_chr21.bed -o HepG2_7.5U_elf4.gz -a 50 -b 50 -bs 1
# plotProfile -m HepG2_7.5U_elf4.gz --yMin 0.25 --yMax 0.55 -out HepG2_7.5U_elf4.pdf

computeMatrix reference-point --referencePoint center -p 10 -S epoch_200_k_64_test_human.bw -R elf4_chr21.bed -o epoch_200_k_64_test_human_elf4.gz -a 50 -b 50 -bs 1
plotProfile -m epoch_200_k_64_test_human_elf4.gz --yMin 0.01 --yMax 0.39 -out epoch_200_k_64_test_human_elf4.pdf

computeMatrix reference-point --referencePoint center -p 10 -S ecoli_to_human_naked_dna_corrected.norm.bw -R elf4_chr21.bed \
                              -o ecoli_to_human_naked_dna_corrected_elf4.gz -a 50 -b 50 -bs 1
plotProfile -m ecoli_to_human_naked_dna_corrected_elf4.gz --yMin -0.19 --yMax 0.09 -out ecoli_to_human_naked_dna_corrected_elf4.pdf


#### TOBIAS installation
conda create --name TOBIAS python=3.9
#conda remove --name TOBIAS --all
conda activate TOBIAS
#conda install tobias -c bioconda  # speed is too slow
pip install tobias -i https://pypi.tuna.tsinghua.edu.cn/simple/

/share/home/u21509/workspace/wuang/04.tf_grammer/01.bias_correct/FootTrack
/share/home/u21509/workspace/reference/Homo_sapiens/GRCh38/cell/HepG2/tf/chip/
/share/home/u21509/workspace/reference/Homo_sapiens/GRCh38/motif/bed/

## CTCF
scp -P 10022 u21509@logini.tongji.edu.cn:/share/home/u21509/workspace/reference/Homo_sapiens/GRCh38/motif/bed/ctcf/ctcf.bed \
             /fs/home/jiluzhang/TF_grammar/cnn_bias_model/data/tf_info_hepg2/ctcf_motif.bed
scp -P 10022 u21509@logini.tongji.edu.cn:/share/home/u21509/workspace/reference/Homo_sapiens/GRCh38/cell/HepG2/tf/chip/ctcf/ctcf.bed \
             /fs/home/jiluzhang/TF_grammar/cnn_bias_model/data/tf_info_hepg2/ctcf_chip.bed
scp -P 10022 u21509@logini.tongji.edu.cn:/share/home/u21509/workspace/reference/Homo_sapiens/GRCh38/motif/bed/atf3/atf3.bed \
             /fs/home/jiluzhang/TF_grammar/cnn_bias_model/data/tf_info_hepg2/atf3_motif.bed
scp -P 10022 u21509@logini.tongji.edu.cn:/share/home/u21509/workspace/reference/Homo_sapiens/GRCh38/cell/HepG2/tf/chip/atf3/atf3.bed \
             /fs/home/jiluzhang/TF_grammar/cnn_bias_model/data/tf_info_hepg2/atf3_chip.bed
scp -P 10022 u21509@logini.tongji.edu.cn:/share/home/u21509/workspace/reference/Homo_sapiens/GRCh38/motif/bed/elf4/elf4.bed \
             /fs/home/jiluzhang/TF_grammar/cnn_bias_model/data/tf_info_hepg2/elf4_motif.bed
scp -P 10022 u21509@logini.tongji.edu.cn:/share/home/u21509/workspace/reference/Homo_sapiens/GRCh38/cell/HepG2/tf/chip/elf4/elf4.bed \
             /fs/home/jiluzhang/TF_grammar/cnn_bias_model/data/tf_info_hepg2/elf4_chip.bed

#### transfer all motif and chip files
scp -P 10022 -r u21509@logini.tongji.edu.cn:/share/home/u21509/workspace/reference/Homo_sapiens/GRCh38/pre_motif/motif/* \
             /fs/home/jiluzhang/TF_grammar/cnn_bias_model/data/tf_info_hepg2/all_motif
scp -P 10022 -r u21509@logini.tongji.edu.cn:/share/home/u21509/workspace/reference/Homo_sapiens/GRCh38/cell/HepG2/tf/chip/* \
             /fs/home/jiluzhang/TF_grammar/cnn_bias_model/data/tf_info_hepg2/all_chip

grep chr ctcf_motif.bed | grep -v chrY | grep -v chrM > ctcf_motif_chrom.bed
grep chr ctcf_chip.bed | grep -v chrY | grep -v chrM > ctcf_chip_chrom.bed

bedtools intersect -a ctcf_motif_chrom.bed -b ctcf_chip_chrom.bed -wa | uniq > ctcf_chip_motif.bed        # 67720
bedtools intersect -a ctcf_motif_chrom.bed -b ctcf_chip_chrom.bed -wa -v | uniq > ctcf_nochip_motif.bed   # 891952

TOBIAS PlotAggregate --TFBS ctcf_chip_motif.bed ctcf_nochip_motif.bed --signals ../human_naked_dna_corrected.norm.bw --output test.pdf > test.log

TOBIAS Log2Table --logfiles test.log --outdir log2table_output
















