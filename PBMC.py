## pbmc dataset

## raw data process
import scanpy as sc

dat = sc.read_10x_h5('/fs/home/jiluzhang/scTranslator_new/datasets/raw/10x/pbmc/pbmc_granulocyte_sorted_10k_filtered_feature_bc_matrix.h5', gex_only=False)  # 11898 × 180488
dat.var_names_make_unique()
rna = dat[:, dat.var['feature_types']=='Gene Expression'].copy()  # 11898 × 36601
sc.pp.filter_genes(rna, min_cells=10)  # 11898 × 22017
rna.write('pbmc_rna.h5ad')

atac = dat[:, dat.var['feature_types']=='Peaks'].copy()  # 11898 × 143887
atac.X[atac.X>0]=1  # binarization
sc.pp.filter_genes(atac, min_cells=10)   # 11898 × 143883
atac.write('pbmc_atac.h5ad')


sc.pp.normalize_total(rna, target_sum=1e4)
sc.pp.log1p(rna)

sc.pp.highly_variable_genes(rna)
rna.raw = rna
rna = rna[:, rna.var.highly_variable]
sc.tl.pca(rna)
sc.pp.neighbors(rna)
sc.tl.umap(rna, min_dist=0.4)
sc.tl.leiden(rna, resolution=0.1)
sc.pl.umap(rna, color='leiden', legend_fontsize='5', legend_loc='on data',
           title='', frameon=True, save='.pdf')

sc.tl.rank_genes_groups(adata, 'leiden', method='wilcoxon')

pd.DataFrame(adata.uns['rank_genes_groups']['names']).head(10)
#            0          1       2        3        4
# 0      PRKCH       TCF4  SLC8A1    BANK1    FBXL7
# 1        FYN       PAX5   RBM47  RALGPS2     SVIL
# 2     THEMIS   ARHGAP24   LRMDA   CCSER1     NFIB
# 3     BCL11B       AFF3    TYMP    MEF2C  PPFIBP1
# 4     INPP4B      FCRL5   DMXL2    FCRL1      APP
# 5      CELF2      GPM6A    DPYD   FCHSD2    ZFPM2
# 6      CD247  LINC01320    TNS3    MS4A1    DOCK1
# 7      MBNL1     NIBAN3   MCTP1     AFF3    MAGI1
# 8       CBLB      FOXP1  PLXDC2    CDK14   SPTBN1
# 9  LINC01934     RUBCNL   FMNL2   RIPOR2    RBPMS

# cluster_0:PRKCH,FYN,THEMIS,BCL11B,INPP4B,CELF2,CD247,MBNL1,CBLB,LINC01934
# cluster_1:TCF4,PAX5,ARHGAP24,AFF3,FCRL5,GPM6A,LINC01320,NIBAN3,FOXP1,RUBCNL
# cluster_2:SLC8A1,RBM47,LRMDA,TYMP,DMXL2,DPYD,TNS3,MCTP1,PLXDC2,FMNL2
# cluster_3:BANK1,RALGPS2,CCSER1,MEF2C,FCRL1,FCHSD2,MS4A1,AFF3,CDK14,RIPOR2
# cluster_4:FBXL7,SVIL,NFIB,PPFIBP1,APP,ZFPM2,DOCK1,MAGI1,SPTBN1,RBPMS

# cluster_0: T cell
# cluster_1: Tumor B cell
# cluster_2: Monocyte
# cluster_3: Normal B cell
# cluster_4: Other

## cell type annotation tool: http://xteam.xbio.top/ACT
## ref: https://www.nature.com/articles/s41467-023-36559-0/figures/5
cell_anno_dict = {'0':'T cell', '1':'Tumor B cell', '2':'Monocyte', '3':'Normal B cell', '4':'Other'}
adata.obs['cell_anno'] = adata.obs['leiden']
adata.obs['cell_anno'].replace(cell_anno_dict, inplace=True)

sc.pl.umap(adata, color='cell_anno', legend_fontsize='7', legend_loc='on data', size=3,
           title='', frameon=True, save='_cell_annotation.pdf')

adata.write('rna_tumor_B_res.h5ad')
