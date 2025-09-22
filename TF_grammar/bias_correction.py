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
input_file = "ecoli_nakedDNA.bw"
output_file = "regions_test.bw"
chromosome = "chr1"
start_pos = 3000000
end_pos = 4000000

extract_region_to_bigwig(input_bw='ecoli_nakedDNA.bw', output_bw='regions_train.bw', 
                         chromosome='NC_000913.3', start=1, end=3641652)
extract_region_to_bigwig(input_bw='ecoli_nakedDNA.bw', output_bw='regions_valid.bw', 
                         chromosome='NC_000913.3', start=3641652, end=4641651)
##############################################################################################################################

## /fs/home/jiluzhang/TF_grammar/cnn_bias_model/data/ecoli
## epoch=100
python /fs/home/jiluzhang/TF_grammar/cnn_bias_model/train.py --bw_file ecoli_nakedDNA.bw --train_regions regions_train.bed --valid_regions regions_valid.bed \
                                                             --ref_fasta genome.fa --k 128 --epochs 100 --out_dir . --out_name epoch_100 --seed 0 --batch_size 512

python /fs/home/jiluzhang/TF_grammar/cnn_bias_model/predict.py --regions regions_train.bed --ref_fasta genome.fa --k 128 --model_path ./epoch_100.pth \
                                                               --chrom_size_file chrom.sizes --out_dir . --out_name epoch_100_train
multiBigwigSummary bins -b epoch_100_train.bw regions_train.bw -o epoch_100_train.npz --outRawCounts epoch_100_train.tab \
                        -l pred raw -bs 1 -p 10  # ~2 min
grep -v nan epoch_100_train.tab | sed 1d > epoch_100_train_nonan.tab
rm epoch_100_train.tab

python /fs/home/jiluzhang/TF_grammar/cnn_bias_model/predict.py --regions regions_valid.bed --ref_fasta genome.fa --k 128 --model_path ./epoch_100.pth \
                                                               --chrom_size_file chrom.sizes --out_dir . --out_name epoch_100_valid
multiBigwigSummary bins -b epoch_100_valid.bw regions_valid.bw -o epoch_100_valid.npz --outRawCounts epoch_100_valid.tab \
                        -l pred raw -bs 1 -p 10  # ~2 min
grep -v nan epoch_100_valid.tab | sed 1d > epoch_100_valid_nonan.tab
rm epoch_100_valid.tab

##############################################################################################################################
## ../cal_cor --file epoch_100_train_nonan.tab    # 0.7100750860662395
## ../cal_cor --file epoch_100_valid_nonan.tab    # 0.7082189028293715

#!/fs/home/jiluzhang/softwares/miniconda3/envs/ACCESS_ATAC/bin/python
import pandas as pd
from scipy import stats
import argparse

parser = argparse.ArgumentParser(description='Calculate Pearson correlation between prediction and ground truth')
parser.add_argument('--file', type=str, help='tab file')

args = parser.parse_args()
tab_file = args.file

def main():
    df = pd.read_csv(tab_file, sep='\t', header=None)
    df.columns = ['chrom', 'start', 'end', 'pred', 'raw']
    pearson_corr, _ = stats.pearsonr(df['pred'], df['raw'])
    print(f"Pearson R: {pearson_corr}")

if __name__ == "__main__":
    main()
##############################################################################################################################























extract_region_to_bigwig(input_file, output_file, chromosome, start_pos, end_pos)
