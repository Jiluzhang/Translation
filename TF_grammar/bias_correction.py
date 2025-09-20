## workdir: /fs/home/jiluzhang/TF_grammar/cnn_bias_model
## human nakedDNA: https://github.com/ZhangLab-TJ/FootTrack/blob/main/bias/human/nakedDNA.pickle

conda create --name ACCESS_ATAC python=3.9
conda install numpy
conda install -c conda-forge pytorch-gpu
pip install pyBigWig pyranges pysam pyfaidx
pip install deeptools

wget -c http://hgdownload.cse.ucsc.edu/admin/exe/linux.x86_64/wigToBigWig

scp -P 10022 u21509@logini.tongji.edu.cn:/share/home/u21509/workspace/wuang/01.cFOOT-seq/final_data/human/HepG2_naked_DNA/rawdata/rep1/ordinary/nakedDNA.bw \
             /fs/home/jiluzhang/TF_grammar/cnn_bias_model/data

# import pickle

# with open('nakedDNA_human.pickle', 'rb') as file:
#     df = pickle.load(file)

python ../train.py --bw_file human_nakedDNA.bw --train_regions regions_train.bed --valid_regions regions_valid.bed --ref_fasta hg38.fa \
                   --k 128 --epochs 200 --out_dir . --out_name test --seed 0 --batch_size 512

python ../predict.py --regions regions_valid.bed --ref_fasta hg38.fa --k 128 --model_path ./test.pth --chrom_size_file hg38.chrom.sizes --out_dir . --out_name test


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
input_file = "input.bw"
output_file = "chr1_region.bw"
chromosome = "chr1"
start_pos = 1000000
end_pos = 2000000


multiBigwigSummary bins -b chr1_regions_valid.bw test.bw -o raw_pred_signal.npz --outRawCounts raw_pred_signal.tab \
                        -l raw pred -bs 10000 -p 8

#################################################### all nan in the tab files ####################################################


import pandas as pd
from scipy import stats

df = pd.read_csv('counts.tab', sep='\t', comment='#')
pearson_corr, _ = stats.pearsonr(df['sample1'], df['sample2'])
spearman_corr, _ = stats.spearmanr(df['sample1'], df['sample2'])
print(f"Pearson R: {pearson_corr}")
print(f"Spearman R: {spearman_corr}")

























extract_region_to_bigwig(input_file, output_file, chromosome, start_pos, end_pos)
