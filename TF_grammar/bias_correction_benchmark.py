# import pandas as pd
# import numpy as np

# dat = pd.read_table('../ctcf_print_ft.gz', skiprows=1, header=None).iloc[:, 6:]
# [dat.iloc[:, 45:55].mean(axis=0).mean().item(), dat.iloc[:, np.r_[0:45, 55:100]].mean(axis=0).mean().item()]

# dat = pd.read_table('../ctcf_print_noft.gz', skiprows=1, header=None).iloc[:, 6:]
# [dat.iloc[:, 45:55].mean(axis=0).mean().item(), dat.iloc[:, np.r_[0:45, 55:100]].mean(axis=0).mean().item()]


## workdir: /fs/home/jiluzhang/TF_grammar/PRINT/cfoot/bias_correction_benchmark
#### Bias correction
for tf in ctcf atf3 elf4 myc nfib pbx3 sox13 tcf7 tead3 yy1 atf7 cebpa elk4 foxa1 gata2 hoxa5 irf3 klf11 nfyc rara;do
    TOBIAS PlotAggregate --TFBS /fs/home/jiluzhang/TF_grammar/cnn_bias_model/data/bias_correction/$tf\_chip_motif.bed \
                                /fs/home/jiluzhang/TF_grammar/cnn_bias_model/data/bias_correction/$tf\_nochip_motif.bed \
                         --signals HepG2_7.5U.bw --output $tf\_raw.pdf > $tf\_raw.log
    TOBIAS Log2Table --logfiles $tf\_raw.log --outdir $tf\_raw_log2table_output
    TOBIAS PlotAggregate --TFBS /fs/home/jiluzhang/TF_grammar/cnn_bias_model/data/bias_correction/$tf\_chip_motif.bed \
                                /fs/home/jiluzhang/TF_grammar/cnn_bias_model/data/bias_correction/$tf\_nochip_motif.bed \
                         --signals human_footrack.bw --output $tf\_footrack.pdf > $tf\_footrack.log
    TOBIAS Log2Table --logfiles $tf\_footrack.log --outdir $tf\_footrack_log2table_output
    TOBIAS PlotAggregate --TFBS /fs/home/jiluzhang/TF_grammar/cnn_bias_model/data/bias_correction/$tf\_chip_motif.bed \
                                /fs/home/jiluzhang/TF_grammar/cnn_bias_model/data/bias_correction/$tf\_nochip_motif.bed \
                         --signals ecoli_access_atac.norm.bw --output $tf\_access_atac.pdf > $tf\_access_atac.log
    TOBIAS Log2Table --logfiles $tf\_access_atac.log --outdir $tf\_access_atac_log2table_output
    TOBIAS PlotAggregate --TFBS /fs/home/jiluzhang/TF_grammar/cnn_bias_model/data/bias_correction/$tf\_chip_motif.bed \
                                /fs/home/jiluzhang/TF_grammar/cnn_bias_model/data/bias_correction/$tf\_nochip_motif.bed \
                         --signals ecoli_print.norm.bw --output $tf\_print_noft.pdf > $tf\_print_noft.log
    TOBIAS Log2Table --logfiles $tf\_print_noft.log --outdir $tf\_print_noft_log2table_output
    TOBIAS PlotAggregate --TFBS /fs/home/jiluzhang/TF_grammar/cnn_bias_model/data/bias_correction/$tf\_chip_motif.bed \
                                /fs/home/jiluzhang/TF_grammar/cnn_bias_model/data/bias_correction/$tf\_nochip_motif.bed \
                         --signals ecoli_chrM_print.norm.bw --output $tf\_print_ft.pdf > $tf\_print_ft.log
    TOBIAS Log2Table --logfiles $tf\_print_ft.log --outdir $tf\_print_ft_log2table_output
    echo $tf done
done

for tf in ctcf atf3 elf4 myc nfib pbx3 sox13 tcf7 tead3 yy1 atf7 cebpa elk4 foxa1 gata2 hoxa5 irf3 klf11 nfyc rara;do
    echo $tf >> tfs.txt
    awk '{if($3==20) print$6}' ./$tf\_raw_log2table_output/aggregate_FPD.txt | sed -n '1p' >> raw_chip_FPD.txt
    awk '{if($3==20) print$6}' ./$tf\_raw_log2table_output/aggregate_FPD.txt | sed -n '2p' >> raw_nochip_FPD.txt
    awk '{if($3==20) print$6}' ./$tf\_footrack_log2table_output/aggregate_FPD.txt | sed -n '1p' >> footrack_chip_FPD.txt
    awk '{if($3==20) print$6}' ./$tf\_footrack_log2table_output/aggregate_FPD.txt | sed -n '2p' >> footrack_nochip_FPD.txt
    awk '{if($3==20) print$6}' ./$tf\_access_atac_log2table_output/aggregate_FPD.txt | sed -n '1p' >> access_atac_chip_FPD.txt
    awk '{if($3==20) print$6}' ./$tf\_access_atac_log2table_output/aggregate_FPD.txt | sed -n '2p' >> access_atac_nochip_FPD.txt
    awk '{if($3==20) print$6}' ./$tf\_print_noft_log2table_output/aggregate_FPD.txt | sed -n '1p' >> print_noft_chip_FPD.txt
    awk '{if($3==20) print$6}' ./$tf\_print_noft_log2table_output/aggregate_FPD.txt | sed -n '2p' >> print_noft_nochip_FPD.txt
    awk '{if($3==20) print$6}' ./$tf\_print_ft_log2table_output/aggregate_FPD.txt | sed -n '1p' >> print_ft_chip_FPD.txt
    awk '{if($3==20) print$6}' ./$tf\_print_ft_log2table_output/aggregate_FPD.txt | sed -n '2p' >> print_ft_nochip_FPD.txt
    echo $tf done
done

paste tfs.txt raw_chip_FPD.txt raw_nochip_FPD.txt \
              footrack_chip_FPD.txt footrack_nochip_FPD.txt \
              access_atac_chip_FPD.txt access_atac_nochip_FPD.txt \
              print_noft_chip_FPD.txt print_noft_nochip_FPD.txt \
              print_ft_chip_FPD.txt print_ft_nochip_FPD.txt > tfs_FPD.txt


import pandas as pd
from plotnine import *
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['pdf.fonttype'] = 42

dat = pd.read_table('tfs_FPD.txt', header=None)

df = pd.DataFrame({'alg':['raw']*40+['footrack']*40+['access_atac']*40+['print_noft']*40+['print_ft']*40,
                   'tf_chip':(['chip']*20+['nochip']*20)*5,
                   'FPD':0})

df.loc[(df['alg']=='raw')&(df['tf_chip']=='chip'), 'FPD'] = dat[1].values
df.loc[(df['alg']=='raw')&(df['tf_chip']=='nochip'), 'FPD'] = dat[2].values
df.loc[(df['alg']=='footrack')&(df['tf_chip']=='chip'), 'FPD'] = dat[3].values
df.loc[(df['alg']=='footrack')&(df['tf_chip']=='nochip'), 'FPD'] = dat[4].values
df.loc[(df['alg']=='access_atac')&(df['tf_chip']=='chip'), 'FPD'] = dat[5].values
df.loc[(df['alg']=='access_atac')&(df['tf_chip']=='nochip'), 'FPD'] = dat[6].values
df.loc[(df['alg']=='print_noft')&(df['tf_chip']=='chip'), 'FPD'] = dat[7].values
df.loc[(df['alg']=='print_noft')&(df['tf_chip']=='nochip'), 'FPD'] = dat[8].values
df.loc[(df['alg']=='print_ft')&(df['tf_chip']=='chip'), 'FPD'] = dat[9].values
df.loc[(df['alg']=='print_ft')&(df['tf_chip']=='nochip'), 'FPD'] = dat[10].values

df['alg'] = pd.Categorical(df['alg'], categories=['raw', 'footrack', 'access_atac', 'print_noft', 'print_ft'])
p = ggplot(df, aes(x='alg', y='FPD', fill='tf_chip')) + geom_boxplot(width=0.5, show_legend=True, outlier_shape='') + xlab('') +\
                                                        coord_cartesian(ylim=(-0.08, 0.08)) +\
                                                        scale_y_continuous(breaks=np.arange(-0.08, 0.08+0.02, 0.04)) +\
                                                        theme_bw()+ theme(axis_text_x=element_text(angle=45, hjust=1))
p.save(filename='tfs_FPD_20.pdf', dpi=600, height=4, width=4)

## paired t test
stats.ttest_rel(df.loc[(df['alg']=='raw')&(df['tf_chip']=='chip'), 'FPD'], 
                df.loc[(df['alg']=='raw')&(df['tf_chip']=='nochip'), 'FPD'], alternative='less')[1]          # 1.238310454214302e-06
stats.ttest_rel(df.loc[(df['alg']=='footrack')&(df['tf_chip']=='chip'), 'FPD'], 
                df.loc[(df['alg']=='footrack')&(df['tf_chip']=='nochip'), 'FPD'], alternative='less')[1]     # 0.0005022915052293431
stats.ttest_rel(df.loc[(df['alg']=='access_atac')&(df['tf_chip']=='chip'), 'FPD'], 
                df.loc[(df['alg']=='access_atac')&(df['tf_chip']=='nochip'), 'FPD'], alternative='less')[1]  # 0.000995774936298764 
stats.ttest_rel(df.loc[(df['alg']=='print_noft')&(df['tf_chip']=='chip'), 'FPD'], 
                df.loc[(df['alg']=='print_noft')&(df['tf_chip']=='nochip'), 'FPD'], alternative='less')[1]   # 0.0007637641145011214
stats.ttest_rel(df.loc[(df['alg']=='print_ft')&(df['tf_chip']=='chip'), 'FPD'], 
                df.loc[(df['alg']=='print_ft')&(df['tf_chip']=='nochip'), 'FPD'], alternative='less')[1]     # 0.0027624589109376493


## footprint width=motif lenght
# for tf in ctcf atf3 elf4 myc nfib pbx3 sox13 tcf7 tead3 yy1 atf7 cebpa elk4 foxa1 gata2 hoxa5 irf3 klf11 nfyc rara;do
#     echo $tf >> tfs.txt
#     grep chip_motif ../$tf\_raw_log2table_output/aggregate_FPD.txt | head -n 1 | awk '{print$6}' >> raw_chip_FPD.txt
#     grep nochip_motif ../$tf\_raw_log2table_output/aggregate_FPD.txt | head -n 1 | awk '{print$6}' >> raw_nochip_FPD.txt
#     grep chip_motif ../$tf\_footrack_log2table_output/aggregate_FPD.txt | head -n 1 | awk '{print$6}' >> footrack_chip_FPD.txt
#     grep nochip_motif ../$tf\_footrack_log2table_output/aggregate_FPD.txt | head -n 1 | awk '{print$6}' >> footrack_nochip_FPD.txt
#     grep chip_motif ../$tf\_access_atac_log2table_output/aggregate_FPD.txt | head -n 1 | awk '{print$6}' >> access_atac_chip_FPD.txt
#     grep nochip_motif ../$tf\_access_atac_log2table_output/aggregate_FPD.txt | head -n 1 | awk '{print$6}' >> access_atac_nochip_FPD.txt
#     grep chip_motif ../$tf\_print_noft_log2table_output/aggregate_FPD.txt | head -n 1 | awk '{print$6}' >> print_noft_chip_FPD.txt
#     grep nochip_motif ../$tf\_print_noft_log2table_output/aggregate_FPD.txt | head -n 1 | awk '{print$6}' >> print_noft_nochip_FPD.txt
#     grep chip_motif ../$tf\_print_ft_log2table_output/aggregate_FPD.txt | head -n 1 | awk '{print$6}' >> print_ft_chip_FPD.txt
#     grep nochip_motif ../$tf\_print_ft_log2table_output/aggregate_FPD.txt | head -n 1 | awk '{print$6}' >> print_ft_nochip_FPD.txt
#     echo $tf done
# done

# paste tfs.txt raw_chip_FPD.txt raw_nochip_FPD.txt \
#               footrack_chip_FPD.txt footrack_nochip_FPD.txt \
#               access_atac_chip_FPD.txt access_atac_nochip_FPD.txt \
#               print_noft_chip_FPD.txt print_noft_nochip_FPD.txt \
#               print_ft_chip_FPD.txt print_ft_nochip_FPD.txt > tfs_FPD.txt


# import pandas as pd
# from plotnine import *
# from scipy import stats
# import numpy as np
# import matplotlib.pyplot as plt

# plt.rcParams['pdf.fonttype'] = 42

# dat = pd.read_table('tfs_FPD.txt', header=None)

# df = pd.DataFrame({'alg':['raw']*40+['footrack']*40+['access_atac']*40+['print_noft']*40+['print_ft']*40,
#                    'tf_chip':(['chip']*20+['nochip']*20)*5,
#                    'FPD':0})

# df.loc[(df['alg']=='raw')&(df['tf_chip']=='chip'), 'FPD'] = dat[1].values
# df.loc[(df['alg']=='raw')&(df['tf_chip']=='nochip'), 'FPD'] = dat[2].values
# df.loc[(df['alg']=='footrack')&(df['tf_chip']=='chip'), 'FPD'] = dat[3].values
# df.loc[(df['alg']=='footrack')&(df['tf_chip']=='nochip'), 'FPD'] = dat[4].values
# df.loc[(df['alg']=='access_atac')&(df['tf_chip']=='chip'), 'FPD'] = dat[5].values
# df.loc[(df['alg']=='access_atac')&(df['tf_chip']=='nochip'), 'FPD'] = dat[6].values
# df.loc[(df['alg']=='print_noft')&(df['tf_chip']=='chip'), 'FPD'] = dat[7].values
# df.loc[(df['alg']=='print_noft')&(df['tf_chip']=='nochip'), 'FPD'] = dat[8].values
# df.loc[(df['alg']=='print_ft')&(df['tf_chip']=='chip'), 'FPD'] = dat[9].values
# df.loc[(df['alg']=='print_ft')&(df['tf_chip']=='nochip'), 'FPD'] = dat[10].values

# df['alg'] = pd.Categorical(df['alg'], categories=['raw', 'footrack', 'access_atac', 'print_noft', 'print_ft'])
# p = ggplot(df, aes(x='alg', y='FPD', fill='tf_chip')) + geom_boxplot(width=0.5, show_legend=True, outlier_shape='') + xlab('') +\
#                                                         coord_cartesian(ylim=(-0.08, 0.08)) +\
#                                                         scale_y_continuous(breaks=np.arange(-0.08, 0.08+0.02, 0.04)) +\
#                                                         theme_bw()+ theme(axis_text_x=element_text(angle=45, hjust=1))
# p.save(filename='tfs_FPD_20.pdf', dpi=600, height=4, width=4)

# ## paired t test
# stats.ttest_rel(df.loc[(df['alg']=='raw')&(df['tf_chip']=='chip'), 'FPD'], 
#                 df.loc[(df['alg']=='raw')&(df['tf_chip']=='nochip'), 'FPD'], alternative='less')[1]          # 9.341739731206926e-07
# stats.ttest_rel(df.loc[(df['alg']=='footrack')&(df['tf_chip']=='chip'), 'FPD'], 
#                 df.loc[(df['alg']=='footrack')&(df['tf_chip']=='nochip'), 'FPD'], alternative='less')[1]     # 0.0011960241179037572
# stats.ttest_rel(df.loc[(df['alg']=='access_atac')&(df['tf_chip']=='chip'), 'FPD'], 
#                 df.loc[(df['alg']=='access_atac')&(df['tf_chip']=='nochip'), 'FPD'], alternative='less')[1]  # 0.0015954335415008813
# stats.ttest_rel(df.loc[(df['alg']=='print_noft')&(df['tf_chip']=='chip'), 'FPD'], 
#                 df.loc[(df['alg']=='print_noft')&(df['tf_chip']=='nochip'), 'FPD'], alternative='less')[1]   # 0.001200690751976006
# stats.ttest_rel(df.loc[(df['alg']=='print_ft')&(df['tf_chip']=='chip'), 'FPD'], 
#                 df.loc[(df['alg']=='print_ft')&(df['tf_chip']=='nochip'), 'FPD'], alternative='less')[1]     # 0.007372477944269592


#### Plot TF footprint
## KLF11
computeMatrix reference-point --referencePoint center -p 10 -S HepG2_7.5U.bw \
                              -R /fs/home/jiluzhang/TF_grammar/cnn_bias_model/data/bias_correction/klf11_chip_motif.bed \
                              -o klf11_raw.gz -a 50 -b 50 -bs 1
computeMatrix reference-point --referencePoint center -p 10 -S human_footrack.bw \
                              -R /fs/home/jiluzhang/TF_grammar/cnn_bias_model/data/bias_correction/klf11_chip_motif.bed \
                              -o klf11_footrack.gz -a 50 -b 50 -bs 1
computeMatrix reference-point --referencePoint center -p 10 -S ecoli_access_atac.norm.bw \
                              -R /fs/home/jiluzhang/TF_grammar/cnn_bias_model/data/bias_correction/klf11_chip_motif.bed \
                              -o klf11_access_atac.gz -a 50 -b 50 -bs 1
computeMatrix reference-point --referencePoint center -p 10 -S ecoli_print.norm.bw \
                              -R /fs/home/jiluzhang/TF_grammar/cnn_bias_model/data/bias_correction/klf11_chip_motif.bed \
                              -o klf11_print_noft.gz -a 50 -b 50 -bs 1
computeMatrix reference-point --referencePoint center -p 10 -S ecoli_chrM_print.norm.bw \
                              -R /fs/home/jiluzhang/TF_grammar/cnn_bias_model/data/bias_correction/klf11_chip_motif.bed \
                              -o klf11_print_ft.gz -a 50 -b 50 -bs 1
plotProfile -m klf11_raw.gz -out deeptools_plot/klf11_raw.pdf
plotProfile -m klf11_footrack.gz -out deeptools_plot/klf11_footrack.pdf
plotProfile -m klf11_access_atac.gz -out deeptools_plot/klf11_access_atac.pdf
plotProfile -m klf11_print_noft.gz -out deeptools_plot/klf11_print_noft.pdf
plotProfile -m klf11_print_ft.gz -out deeptools_plot/klf11_print_ft.pdf

## CEBPA
computeMatrix reference-point --referencePoint center -p 10 -S HepG2_7.5U.bw \
                              -R /fs/home/jiluzhang/TF_grammar/cnn_bias_model/data/bias_correction/cebpa_chip_motif.bed \
                              -o cebpa_raw.gz -a 50 -b 50 -bs 2
computeMatrix reference-point --referencePoint center -p 10 -S human_footrack.bw \
                              -R /fs/home/jiluzhang/TF_grammar/cnn_bias_model/data/bias_correction/cebpa_chip_motif.bed \
                              -o cebpa_footrack.gz -a 50 -b 50 -bs 2
computeMatrix reference-point --referencePoint center -p 10 -S ecoli_access_atac.norm.bw \
                              -R /fs/home/jiluzhang/TF_grammar/cnn_bias_model/data/bias_correction/cebpa_chip_motif.bed \
                              -o cebpa_access_atac.gz -a 50 -b 50 -bs 2
computeMatrix reference-point --referencePoint center -p 10 -S ecoli_print.norm.bw \
                              -R /fs/home/jiluzhang/TF_grammar/cnn_bias_model/data/bias_correction/cebpa_chip_motif.bed \
                              -o cebpa_print_noft.gz -a 50 -b 50 -bs 2
computeMatrix reference-point --referencePoint center -p 10 -S ecoli_chrM_print.norm.bw \
                              -R /fs/home/jiluzhang/TF_grammar/cnn_bias_model/data/bias_correction/cebpa_chip_motif.bed \
                              -o cebpa_print_ft.gz -a 50 -b 50 -bs 2
plotProfile -m cebpa_raw.gz -out deeptools_plot/cebpa_raw.pdf
plotProfile -m cebpa_footrack.gz -out deeptools_plot/cebpa_footrack.pdf
plotProfile -m cebpa_access_atac.gz -out deeptools_plot/cebpa_access_atac.pdf
plotProfile -m cebpa_print_noft.gz -out deeptools_plot/cebpa_print_noft.pdf
plotProfile -m cebpa_print_ft.gz -out deeptools_plot/cebpa_print_ft.pdf

## CTCF
computeMatrix reference-point --referencePoint center -p 10 -S HepG2_7.5U.bw \
                              -R /fs/home/jiluzhang/TF_grammar/cnn_bias_model/data/bias_correction/ctcf_chip_motif.bed \
                              -o ctcf_raw.gz -a 50 -b 50 -bs 1
computeMatrix reference-point --referencePoint center -p 10 -S human_footrack.bw \
                              -R /fs/home/jiluzhang/TF_grammar/cnn_bias_model/data/bias_correction/ctcf_chip_motif.bed \
                              -o ctcf_footrack.gz -a 50 -b 50 -bs 1
computeMatrix reference-point --referencePoint center -p 10 -S ecoli_access_atac.norm.bw \
                              -R /fs/home/jiluzhang/TF_grammar/cnn_bias_model/data/bias_correction/ctcf_chip_motif.bed \
                              -o ctcf_access_atac.gz -a 50 -b 50 -bs 1
computeMatrix reference-point --referencePoint center -p 10 -S ecoli_print.norm.bw \
                              -R /fs/home/jiluzhang/TF_grammar/cnn_bias_model/data/bias_correction/ctcf_chip_motif.bed \
                              -o ctcf_print_noft.gz -a 50 -b 50 -bs 1
computeMatrix reference-point --referencePoint center -p 10 -S ecoli_chrM_print.norm.bw \
                              -R /fs/home/jiluzhang/TF_grammar/cnn_bias_model/data/bias_correction/ctcf_chip_motif.bed \
                              -o ctcf_print_ft.gz -a 50 -b 50 -bs 1
plotProfile -m ctcf_raw.gz -out deeptools_plot/ctcf_raw.pdf
plotProfile -m ctcf_footrack.gz -out deeptools_plot/ctcf_footrack.pdf
plotProfile -m ctcf_access_atac.gz -out deeptools_plot/ctcf_access_atac.pdf
plotProfile -m ctcf_print_noft.gz -out deeptools_plot/ctcf_print_noft.pdf
plotProfile -m ctcf_print_ft.gz -out deeptools_plot/ctcf_print_ft.pdf
