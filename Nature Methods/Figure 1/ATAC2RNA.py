## workdir: /fs/home/jiluzhang/Nature_methods/Figure_1/ATAC2RNA

cp /fs/home/zouqihang/to_luz/k562_acc/CA_scatter_spearman.pdf_raw_data.tsv k562_archr_cisformer_spearman.txt
cp /fs/home/zouqihang/to_luz/k562_acc/CS_scatter_spearman.pdf_raw_data.tsv k562_scarlink_cisformer_spearman.txt

cp /fs/home/zouqihang/to_luz/pbmc_acc/CA_scatter_spearman.pdf_raw_data.tsv pbmc_archr_cisformer_spearman.txt
cp /fs/home/zouqihang/to_luz/pbmc_acc/CS_scatter_spearman.pdf_raw_data.tsv pbmc_scarlink_cisformer_spearman.txt

cp /fs/home/zouqihang/to_luz/tumor_b_acc/CA_scatter_spearman.pdf_raw_data.tsv tumor_b_archr_cisformer_spearman.txt
cp /fs/home/zouqihang/to_luz/tumor_b_acc/CS_scatter_spearman.pdf_raw_data.tsv tumor_b_scarlink_cisformer_spearman.txt


awk '{if($2!=0) print $0}' k562_archr_cisformer_spearman.txt | sed 1d | wc -l   # 12540


import pandas as pd

k562_archr_cisformer = pd.read_table('k562_archr_cisformer_spearman.txt')
k562_archr_cisformer.columns = ['cisformer', 'archr', 'comp']
k562_archr_cisformer['cisformer'].mean()   # 0.9995556293783626
k562_archr_cisformer['archr'].mean()       # 0.005687431129319071

pbmc_archr_cisformer = pd.read_table('pbmc_archr_cisformer_spearman.txt')
pbmc_archr_cisformer.columns = ['cisformer', 'archr', 'comp']
pbmc_archr_cisformer['cisformer'].mean()   # 0.9750859934906061
pbmc_archr_cisformer['archr'].mean()       # 0.041026654470665905
pbmc_archr_cisformer[pbmc_archr_cisformer['archr']!=0]['archr'].mean()   # 0.04391130255656801


pbmc_scarlink_cisformer = pd.read_table('pbmc_scarlink_cisformer_spearman.txt')
pbmc_scarlink_cisformer.columns = ['cisformer', 'scarlink', 'comp']
pbmc_scarlink_cisformer['cisformer'].mean()                                           # 0.976204637298592
pbmc_scarlink_cisformer['scarlink'].mean()                                            # 0.009690757451368895
pbmc_scarlink_cisformer[pbmc_scarlink_cisformer['scarlink']!=0]['cisformer'].mean()   # 0.9470517619327977
pbmc_scarlink_cisformer[pbmc_scarlink_cisformer['scarlink']!=0]['scarlink'].mean()    # 0.33709043776532704







