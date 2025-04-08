## workdir: /fs/home/jiluzhang/Nature_methods/Figure_1/ATAC2RNA

cp /fs/home/zouqihang/to_luz/k562_acc/CA_scatter_spearman.pdf_raw_data.tsv k562_archr_cisformer_spearman.txt
cp /fs/home/zouqihang/to_luz/k562_acc/CS_scatter_spearman.pdf_raw_data.tsv k562_scarlink_cisformer_spearman.txt

cp /fs/home/zouqihang/to_luz/pbmc_acc/CA_scatter_spearman.pdf_raw_data.tsv pbmc_archr_cisformer_spearman.txt
cp /fs/home/zouqihang/to_luz/pbmc_acc/CS_scatter_spearman.pdf_raw_data.tsv pbmc_scarlink_cisformer_spearman.txt

cp /fs/home/zouqihang/to_luz/tumor_b_acc/CA_scatter_spearman.pdf_raw_data.tsv tumor_b_archr_cisformer_spearman.txt
cp /fs/home/zouqihang/to_luz/tumor_b_acc/CS_scatter_spearman.pdf_raw_data.tsv tumor_b_scarlink_cisformer_spearman.txt

cp /fs/home/zouqihang/to_luz/k562_acc/CA_scatter_peason.pdf_raw_data.tsv k562_archr_cisformer_pearson.txt
cp /fs/home/zouqihang/to_luz/k562_acc/CS_scatter_peason.pdf_raw_data.tsv k562_scarlink_cisformer_pearson.txt

cp /fs/home/zouqihang/to_luz/pbmc_acc/CA_scatter_peason.pdf_raw_data.tsv pbmc_archr_cisformer_pearson.txt
cp /fs/home/zouqihang/to_luz/pbmc_acc/CS_scatter_peason.pdf_raw_data.tsv pbmc_scarlink_cisformer_pearson.txt

cp /fs/home/zouqihang/to_luz/tumor_b_acc/CA_scatter_peason.pdf_raw_data.tsv tumor_b_archr_cisformer_pearson.txt
cp /fs/home/zouqihang/to_luz/tumor_b_acc/CS_scatter_peason.pdf_raw_data.tsv tumor_b_scarlink_cisformer_pearson.txt


import pandas as pd
from plotnine import *
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['pdf.fonttype'] = 42

## spearman correlation
# k562 (not used)
k562_archr_cisformer = pd.read_table('k562_archr_cisformer_spearman.txt')
k562_archr_cisformer.columns = ['cisformer', 'archr', 'comp']
k562_archr_cisformer[k562_archr_cisformer['cisformer']!=0]['cisformer'].mean()              # 0.9996338235134935
k562_archr_cisformer[k562_archr_cisformer['archr']!=0]['archr'].mean()                      # 0.005798095658470096

k562_scarlink_cisformer = pd.read_table('k562_scarlink_cisformer_spearman.txt')
k562_scarlink_cisformer.columns = ['cisformer', 'scarlink', 'comp']
k562_scarlink_cisformer[k562_scarlink_cisformer['cisformer']!=0]['cisformer'].mean()        # 0.9996338235134935
k562_scarlink_cisformer[k562_scarlink_cisformer['scarlink']!=0]['scarlink'].mean()          # 0.04839020230554602

# pbmc
pbmc_archr_cisformer = pd.read_table('pbmc_archr_cisformer_spearman.txt')
pbmc_archr_cisformer.columns = ['cisformer', 'archr', 'comp']
pbmc_archr_cisformer[pbmc_archr_cisformer['cisformer']!=0]['cisformer'].mean()              # 0.9760733112541677
pbmc_archr_cisformer[pbmc_archr_cisformer['archr']!=0]['archr'].mean()                      # 0.04391130255656801

pbmc_scarlink_cisformer = pd.read_table('pbmc_scarlink_cisformer_spearman.txt')
pbmc_scarlink_cisformer.columns = ['cisformer', 'scarlink', 'comp']
pbmc_scarlink_cisformer[pbmc_scarlink_cisformer['cisformer']!=0]['cisformer'].mean()        # 0.976204637298592
pbmc_scarlink_cisformer[pbmc_scarlink_cisformer['scarlink']!=0]['scarlink'].mean()          # 0.33709043776532704

# tumor_b
tumor_b_archr_cisformer = pd.read_table('tumor_b_archr_cisformer_spearman.txt')
tumor_b_archr_cisformer.columns = ['cisformer', 'archr', 'comp']
tumor_b_archr_cisformer[tumor_b_archr_cisformer['cisformer']!=0]['cisformer'].mean()        # 0.950734398584293
tumor_b_archr_cisformer[tumor_b_archr_cisformer['archr']!=0]['archr'].mean()                # 0.06685202274617193

tumor_b_scarlink_cisformer = pd.read_table('tumor_b_scarlink_cisformer_spearman.txt')
tumor_b_scarlink_cisformer.columns = ['cisformer', 'scarlink', 'comp']
tumor_b_scarlink_cisformer[tumor_b_scarlink_cisformer['cisformer']!=0]['cisformer'].mean()  # 0.950734398584293
tumor_b_scarlink_cisformer[tumor_b_scarlink_cisformer['scarlink']!=0]['scarlink'].mean()    # 0.35068920573969

## barplot for average spearman correlation
dat = pd.DataFrame({'dataset': ['k562']*3+['pbmc']*3+['tumor_b']*3,
                    'algorithm': ['archr', 'scarlink', 'cisformer']*3,
                    'correlation': [0.0058, 0.0484, 0.9996, 0.0439, 0.3371, 0.9761, 0.0669, 0.3507, 0.9507]})
dat['dataset'] = pd.Categorical(dat['dataset'], categories=['k562', 'pbmc', 'tumor_b'])
dat['algorithm'] = pd.Categorical(dat['algorithm'], categories=['archr', 'scarlink', 'cisformer'])

p = ggplot(dat, aes(x='dataset', y='correlation', fill='algorithm')) + geom_bar(stat='identity', position=position_dodge(), width=0.75) +\
                                                                       scale_y_continuous(limits=[0, 1], breaks=np.arange(0, 1.1, 0.2)) + theme_bw()
p.save(filename='pbmc_tumor_b_spearman_archr_scarlink_cisformer.pdf', dpi=600, height=4, width=6)


## pearson correlation
# k562 (not used)
k562_archr_cisformer = pd.read_table('k562_archr_cisformer_pearson.txt')
k562_archr_cisformer.columns = ['cisformer', 'archr', 'comp']
k562_archr_cisformer[k562_archr_cisformer['cisformer']!=0]['cisformer'].mean()              # 0.9429113540264555
k562_archr_cisformer[k562_archr_cisformer['archr']!=0]['archr'].mean()                      # 0.000339627248936119

k562_scarlink_cisformer = pd.read_table('k562_scarlink_cisformer_pearson.txt')
k562_scarlink_cisformer.columns = ['cisformer', 'scarlink', 'comp']
k562_scarlink_cisformer[k562_scarlink_cisformer['cisformer']!=0]['cisformer'].mean()        # 0.9429113540264555
k562_scarlink_cisformer[k562_scarlink_cisformer['scarlink']!=0]['scarlink'].mean()          # 0.0808712320383338

# pbmc
pbmc_archr_cisformer = pd.read_table('pbmc_archr_cisformer_pearson.txt')
pbmc_archr_cisformer.columns = ['cisformer', 'archr', 'comp']
pbmc_archr_cisformer[pbmc_archr_cisformer['cisformer']!=0]['cisformer'].mean()              # 0.9394013579516567
pbmc_archr_cisformer[pbmc_archr_cisformer['archr']!=0]['archr'].mean()                      # 0.03008414050581014

pbmc_scarlink_cisformer = pd.read_table('pbmc_scarlink_cisformer_pearson.txt')
pbmc_scarlink_cisformer.columns = ['cisformer', 'scarlink', 'comp']
pbmc_scarlink_cisformer[pbmc_scarlink_cisformer['cisformer']!=0]['cisformer'].mean()        # 0.9395192286143785
pbmc_scarlink_cisformer[pbmc_scarlink_cisformer['scarlink']!=0]['scarlink'].mean()          # 0.35413627781392704

# tumor_b
tumor_b_archr_cisformer = pd.read_table('tumor_b_archr_cisformer_pearson.txt')
tumor_b_archr_cisformer.columns = ['cisformer', 'archr', 'comp']
tumor_b_archr_cisformer[tumor_b_archr_cisformer['cisformer']!=0]['cisformer'].mean()        # 0.8492236952119349
tumor_b_archr_cisformer[tumor_b_archr_cisformer['archr']!=0]['archr'].mean()                # 0.020564016117101236

tumor_b_scarlink_cisformer = pd.read_table('tumor_b_scarlink_cisformer_pearson.txt')
tumor_b_scarlink_cisformer.columns = ['cisformer', 'scarlink', 'comp']
tumor_b_scarlink_cisformer[tumor_b_scarlink_cisformer['cisformer']!=0]['cisformer'].mean()  # 0.8492236952119349
tumor_b_scarlink_cisformer[tumor_b_scarlink_cisformer['scarlink']!=0]['scarlink'].mean()    # 0.388961161420796

## barplot for average pearson correlation
dat = pd.DataFrame({'dataset': ['k562']*3+['pbmc']*3+['tumor_b']*3,
                    'algorithm': ['archr', 'scarlink', 'cisformer']*3,
                    'correlation': [0.0003, 0.0809, 0.9429, 0.0301, 0.3541, 0.9394, 0.0206, 0.3890, 0.8492]})
dat['dataset'] = pd.Categorical(dat['dataset'], categories=['k562', 'pbmc', 'tumor_b'])
dat['algorithm'] = pd.Categorical(dat['algorithm'], categories=['archr', 'scarlink', 'cisformer'])

p = ggplot(dat, aes(x='dataset', y='correlation', fill='algorithm')) + geom_bar(stat='identity', position=position_dodge(), width=0.75) +\
                                                                       scale_y_continuous(limits=[0, 1], breaks=np.arange(0, 1.1, 0.2)) + theme_bw()
p.save(filename='pbmc_tumor_b_pearson_archr_scarlink_cisformer.pdf', dpi=600, height=4, width=6)


