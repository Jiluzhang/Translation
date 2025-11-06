#### Bias correction
python /fs/home/jiluzhang/TF_grammar/ACCESS-ATAC/bias_correction/correct_bias.py --bw_raw HepG2_7.5U.bw --bw_bias pred_chr21_GC.bw \
                                                                                 --bed_file regions_test.bed --extend 0 --window 101 \
                                                                                 --pseudo_count 1 --out_dir . --out_name ecoli_to_human_naked_dna_corrected_GC_NA \
                                                                                 --chrom_size_file hg38.chrom.sizes
for tf in ctcf atf3 elf4 myc nfib pbx3 sox13 tcf7 tead3 yy1 atf7 cebpa elk4 foxa1 gata2 hoxa5 irf3 klf11 nfyc rara;do
    TOBIAS PlotAggregate --TFBS /fs/home/jiluzhang/TF_grammar/cnn_bias_model/data/bias_correction/$tf\_chip_motif.bed \
                                /fs/home/jiluzhang/TF_grammar/cnn_bias_model/data/bias_correction/$tf\_nochip_motif.bed \
                         --signals ecoli_to_human_naked_dna_corrected_GC_NA.norm.bw --output $tf\_ecoli_to_human_print_GC_NA.pdf > $tf\_ecoli_to_human_print_GC_NA.log
    TOBIAS Log2Table --logfiles $tf\_ecoli_to_human_print_GC_NA.log --outdir $tf\_ecoli_to_human_print_GC_NA_log2table_output
    echo $tf ecoli_to_human_print done
done

for tf in ctcf atf3 elf4 myc nfib pbx3 sox13 tcf7 tead3 yy1 atf7 cebpa elk4 foxa1 gata2 hoxa5 irf3 klf11 nfyc rara;do
    echo $tf >> tfs.txt
    awk '{if($3==20) print$6}' ./$tf\_ecoli_to_human_print_GC_NA_log2table_output/aggregate_FPD.txt | sed -n '1p' >> ecoli_to_human_print_GC_NA_chip_FPD.txt
    awk '{if($3==20) print$6}' ./$tf\_ecoli_to_human_print_GC_NA_log2table_output/aggregate_FPD.txt | sed -n '2p' >> ecoli_to_human_print_GC_NA_nochip_FPD.txt
    awk '{if($3==20) print$6}' /fs/home/jiluzhang/TF_grammar/cnn_bias_model/data/bias_correction/$tf\_raw_log2table_output/aggregate_FPD.txt | sed -n '1p' >> raw_chip_FPD.txt
    awk '{if($3==20) print$6}' /fs/home/jiluzhang/TF_grammar/cnn_bias_model/data/bias_correction/$tf\_raw_log2table_output/aggregate_FPD.txt | sed -n '2p' >> raw_nochip_FPD.txt
    awk '{if($3==20) print$6}' /fs/home/jiluzhang/TF_grammar/cnn_bias_model/data/bias_correction/$tf\_ecoli_to_human_log2table_output/aggregate_FPD.txt | sed -n '1p' >> ecoli_to_human_chip_FPD.txt
    awk '{if($3==20) print$6}' /fs/home/jiluzhang/TF_grammar/cnn_bias_model/data/bias_correction/$tf\_ecoli_to_human_log2table_output/aggregate_FPD.txt | sed -n '2p' >> ecoli_to_human_nochip_FPD.txt
    awk '{if($3==20) print$6}' /fs/home/jiluzhang/TF_grammar/cnn_bias_model/data/bias_correction/$tf\_human_to_human_log2table_output/aggregate_FPD.txt | sed -n '1p' >> human_to_human_chip_FPD.txt
    awk '{if($3==20) print$6}' /fs/home/jiluzhang/TF_grammar/cnn_bias_model/data/bias_correction/$tf\_human_to_human_log2table_output/aggregate_FPD.txt | sed -n '2p' >> human_to_human_nochip_FPD.txt
    awk '{if($3==20) print$6}' /fs/home/jiluzhang/TF_grammar/cnn_bias_model/data/bias_correction/$tf\_human_raw_log2table_output/aggregate_FPD.txt | sed -n '1p' >> human_raw_chip_FPD.txt
    awk '{if($3==20) print$6}' /fs/home/jiluzhang/TF_grammar/cnn_bias_model/data/bias_correction/$tf\_human_raw_log2table_output/aggregate_FPD.txt | sed -n '2p' >> human_raw_nochip_FPD.txt
    awk '{if($3==20) print$6}' /fs/home/jiluzhang/TF_grammar/cnn_bias_model/data/bias_correction/$tf\_FootTrack_log2table_output/aggregate_FPD.txt | sed -n '1p' >> FootTrack_chip_FPD.txt
    awk '{if($3==20) print$6}' /fs/home/jiluzhang/TF_grammar/cnn_bias_model/data/bias_correction/$tf\_FootTrack_log2table_output/aggregate_FPD.txt | sed -n '2p' >> FootTrack_nochip_FPD.txt
    awk '{if($3==20) print$6}' /fs/home/jiluzhang/TF_grammar/cnn_bias_model/data/bias_correction/$tf\_human_foottrack_local_log2table_output/aggregate_FPD.txt | sed -n '1p' >> human_foottrack_local_chip_FPD.txt
    awk '{if($3==20) print$6}' /fs/home/jiluzhang/TF_grammar/cnn_bias_model/data/bias_correction/$tf\_human_foottrack_local_log2table_output/aggregate_FPD.txt | sed -n '2p' >> human_foottrack_local_nochip_FPD.txt
    echo $tf done
done

paste tfs.txt raw_chip_FPD.txt raw_nochip_FPD.txt \
              ecoli_to_human_print_GC_NA_chip_FPD.txt ecoli_to_human_print_GC_NA_nochip_FPD.txt \
              ecoli_to_human_chip_FPD.txt ecoli_to_human_nochip_FPD.txt \
              human_to_human_chip_FPD.txt human_to_human_nochip_FPD.txt \
              human_raw_chip_FPD.txt human_raw_nochip_FPD.txt \
              FootTrack_chip_FPD.txt FootTrack_nochip_FPD.txt \
              human_foottrack_local_chip_FPD.txt human_foottrack_local_nochip_FPD.txt > tfs_FPD_GC_NA.txt


import pandas as pd
from plotnine import *
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['pdf.fonttype'] = 42

dat = pd.read_table('tfs_FPD_GC_NA.txt', header=None)

df = pd.DataFrame({'alg':['raw']*40+['ecoli_to_human_print']*40+['ecoli_to_human']*40+['human_to_human']*40+['human_raw']*40+['FootTrack_global']*40+['FootTrack_local']*40,
                   'tf_chip':(['chip']*20+['nochip']*20)*7,
                   'FPD':0})

df.loc[(df['alg']=='raw')&(df['tf_chip']=='chip'), 'FPD'] = dat[1].values
df.loc[(df['alg']=='raw')&(df['tf_chip']=='nochip'), 'FPD'] = dat[2].values
df.loc[(df['alg']=='ecoli_to_human_print')&(df['tf_chip']=='chip'), 'FPD'] = dat[3].values
df.loc[(df['alg']=='ecoli_to_human_print')&(df['tf_chip']=='nochip'), 'FPD'] = dat[4].values
df.loc[(df['alg']=='ecoli_to_human')&(df['tf_chip']=='chip'), 'FPD'] = dat[5].values
df.loc[(df['alg']=='ecoli_to_human')&(df['tf_chip']=='nochip'), 'FPD'] = dat[6].values
df.loc[(df['alg']=='human_to_human')&(df['tf_chip']=='chip'), 'FPD'] = dat[7].values
df.loc[(df['alg']=='human_to_human')&(df['tf_chip']=='nochip'), 'FPD'] = dat[8].values
df.loc[(df['alg']=='human_raw')&(df['tf_chip']=='chip'), 'FPD'] = dat[9].values
df.loc[(df['alg']=='human_raw')&(df['tf_chip']=='nochip'), 'FPD'] = dat[10].values
df.loc[(df['alg']=='FootTrack_global')&(df['tf_chip']=='chip'), 'FPD'] = dat[11].values
df.loc[(df['alg']=='FootTrack_global')&(df['tf_chip']=='nochip'), 'FPD'] = dat[12].values
df.loc[(df['alg']=='FootTrack_local')&(df['tf_chip']=='chip'), 'FPD'] = dat[13].values
df.loc[(df['alg']=='FootTrack_local')&(df['tf_chip']=='nochip'), 'FPD'] = dat[14].values

df['alg'] = pd.Categorical(df['alg'], categories=['raw', 'ecoli_to_human_print', 'human_raw', 'ecoli_to_human', 'human_to_human', 'FootTrack_global', 'FootTrack_local'])
p = ggplot(df, aes(x='alg', y='FPD', fill='tf_chip')) + geom_boxplot(width=0.5, show_legend=True, outlier_shape='') + xlab('') +\
                                                         coord_cartesian(ylim=(-0.08, 0.08)) +\
                                                         scale_y_continuous(breaks=np.arange(-0.08, 0.08+0.02, 0.04)) +\
                                                         theme_bw()+ theme(axis_text_x=element_text(angle=45, hjust=1))
p.save(filename='tfs_FPD_GC_NA_20.pdf', dpi=600, height=4, width=4)

## paired t test
stats.ttest_rel(df.loc[(df['alg']=='raw')&(df['tf_chip']=='chip'), 'FPD'], 
                df.loc[(df['alg']=='raw')&(df['tf_chip']=='nochip'), 'FPD'], alternative='less')[1]                   # 1.238310454214302e-06
stats.ttest_rel(df.loc[(df['alg']=='ecoli_to_human_print')&(df['tf_chip']=='chip'), 'FPD'], 
                df.loc[(df['alg']=='ecoli_to_human_print')&(df['tf_chip']=='nochip'), 'FPD'], alternative='less')[1]   # 0.0007637641145011214
stats.ttest_rel(df.loc[(df['alg']=='ecoli_to_human')&(df['tf_chip']=='chip'), 'FPD'], 
                df.loc[(df['alg']=='ecoli_to_human')&(df['tf_chip']=='nochip'), 'FPD'], alternative='less')[1]   # 0.0011192045793338558      
stats.ttest_rel(df.loc[(df['alg']=='human_to_human')&(df['tf_chip']=='chip'), 'FPD'], 
                df.loc[(df['alg']=='human_to_human')&(df['tf_chip']=='nochip'), 'FPD'], alternative='less')[1]   # 0.0002339827175529051
stats.ttest_rel(df.loc[(df['alg']=='human_raw')&(df['tf_chip']=='chip'), 'FPD'], 
                df.loc[(df['alg']=='human_raw')&(df['tf_chip']=='nochip'), 'FPD'], alternative='less')[1]        # 0.0017942748421807659
stats.ttest_rel(df.loc[(df['alg']=='FootTrack_global')&(df['tf_chip']=='chip'), 'FPD'], 
                df.loc[(df['alg']=='FootTrack_global')&(df['tf_chip']=='nochip'), 'FPD'], alternative='less')[1]        # 0.000145318602947415
stats.ttest_rel(df.loc[(df['alg']=='FootTrack_local')&(df['tf_chip']=='chip'), 'FPD'], 
                df.loc[(df['alg']=='FootTrack_local')&(df['tf_chip']=='nochip'), 'FPD'], alternative='less')[1]        # 0.0005022915052293431

#### Plot TF footprint
## KLF11
computeMatrix reference-point --referencePoint center -p 10 -S ecoli_to_human_naked_dna_corrected_GC_NA.norm.bw \
                              -R /fs/home/jiluzhang/TF_grammar/cnn_bias_model/data/bias_correction/klf11_chip_motif.bed \
                              -o klf11_ecoli_to_human_print_GC_NA.gz -a 50 -b 50 -bs 1
plotProfile -m klf11_ecoli_to_human_print_GC_NA.gz -out deeptools_plot/klf11_ecoli_to_human_print_GC_NA.pdf

## CEBPA
computeMatrix reference-point --referencePoint center -p 10 -S ecoli_to_human_naked_dna_corrected_GC_NA.norm.bw \
                              -R /fs/home/jiluzhang/TF_grammar/cnn_bias_model/data/bias_correction/cebpa_chip_motif.bed \
                              -o cebpa_ecoli_to_human_print_GC_NA.gz -a 50 -b 50 -bs 2
plotProfile -m cebpa_ecoli_to_human_print_GC_NA.gz -out deeptools_plot/cebpa_ecoli_to_human_print_GC_NA.pdf

## CTCF
computeMatrix reference-point --referencePoint center -p 10 -S ecoli_to_human_naked_dna_corrected_GC_NA.norm.bw \
                              -R /fs/home/jiluzhang/TF_grammar/cnn_bias_model/data/bias_correction/ctcf_chip_motif.bed \
                              -o ctcf_ecoli_to_human_print_GC_NA.gz -a 50 -b 50 -bs 1
plotProfile -m ctcf_ecoli_to_human_print_GC_NA.gz -out deeptools_plot/ctcf_ecoli_to_human_print_GC_NA.pdf
