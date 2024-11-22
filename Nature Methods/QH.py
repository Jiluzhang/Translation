import scanpy as sc
import pandas as pd
from scipy.sparse import csr_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr


def intersect_adata(adata1, adata2):
    """intersect two adata by their obs and var indices, and drop NA in obs and var

    Args:
        adata1 (scanpy_adata): _description_
        adata2 (scanpy_adata): _description_

    Returns:
        scanpy_adata: copy of adata1
        scanpy_adata: copy of adata2
    """
    adata1 = adata1.copy()
    adata2 = adata2.copy()
    cell_set1 = set(adata1.obs.dropna().index)
    gene_set1 = set(adata1.var.dropna().index)
    cell_set2 = set(adata2.obs.dropna().index)
    gene_set2 = set(adata2.var.dropna().index)
    common_cell = list(cell_set1 & cell_set2)
    common_gene = list(gene_set1 & gene_set2)
    adata1 = adata1[common_cell, common_gene]
    adata2 = adata2[common_cell, common_gene]
    return adata1, adata2

raw_rna = sc.read_h5ad("/data/home/zouqihang/desktop/project/M2M/dataset/Tumor_B/test_rna_tumor_B.h5ad")
raw_atac = sc.read_h5ad("/data/home/zouqihang/desktop/project/M2M/dataset/Tumor_B/test_atac_tumor_B.h5ad")
predict_rna = sc.read_h5ad("/data/home/zouqihang/desktop/project/M2M/version1.0.0/output/Tumor_B/predicted_tumor_B_test_rna.h5ad")
cell_info = pd.read_csv("/data/home/zouqihang/desktop/project/M2M/dataset/Tumor_B/celltype_info.tsv", header=None, index_col=0, sep="\t", names=["cell_anno"])

raw_rna.obs['cell_anno'] = cell_info.loc[raw_rna.obs_names, :]
predict_rna.obs['cell_anno'] = cell_info.loc[predict_rna.obs_names, :]

## raw
sc.pp.log1p(raw_rna)
sc.tl.pca(raw_rna)
sc.pp.neighbors(raw_rna)
sc.tl.umap(raw_rna)
sc.tl.leiden(raw_rna, n_iterations=2)
raw_rna.obs['cell_anno'] = pd.Categorical(raw_rna.obs['cell_anno'], categories=['Normal B cell', 'Tumor B cell', 'T cell', 'Monocyte', 'Other'])
sc.pl.umap(raw_rna, color='cell_anno', legend_fontsize='7', legend_loc='right margin', size=5,
           title='', frameon=True, save='_tumor_b_raw.pdf')

## archr
archr_rna = pd.read_csv("/data/home/zouqihang/desktop/project/M2M/ArchR/Tumor_B/ArchR-GeneScore_matrix.csv", index_col=0)
archr_rna = sc.AnnData(archr_rna)
archr_rna.X = csr_matrix(archr_rna.X)
archr_rna.obs.index = [each.split("#")[1] for each in archr_rna.obs.index]
f_archr_rna, Af_raw_rna = intersect_adata(archr_rna, raw_rna)
f_archr_rna.obs['cell_anno'] = cell_info.loc[f_archr_rna.obs_names, :]
f_archr_rna.X[Af_raw_rna.X == 0] = 0

sc.tl.pca(f_archr_rna)
sc.pp.neighbors(f_archr_rna)
sc.tl.umap(f_archr_rna)
sc.tl.leiden(f_archr_rna, n_iterations=2)
f_archr_rna.obs['cell_anno'] = pd.Categorical(f_archr_rna.obs['cell_anno'], categories=['Normal B cell', 'Tumor B cell', 'T cell', 'Monocyte', 'Other'])
sc.pl.umap(f_archr_rna, color='cell_anno', legend_fontsize='7', legend_loc='right margin', size=5,
           title='', frameon=True, save='_tumor_b_archr.pdf')

## scarlink
scarlink_rna = pd.read_csv("/data/home/zouqihang/desktop/project/M2M/SCARlink/tumor_B_output/predicted_gex.csv", index_col=0)
scarlink_rna = sc.AnnData(scarlink_rna)
scarlink_rna.X = csr_matrix(scarlink_rna.X)
f_scarlink_rna, scarf_raw_rna = intersect_adata(scarlink_rna, raw_rna)
f_scarlink_rna.obs['cell_anno'] = cell_info.loc[scarf_raw_rna.obs_names, :]
f_scarlink_rna.X[scarf_raw_rna.X == 0] = 0

sc.tl.pca(f_scarlink_rna)
sc.pp.neighbors(f_scarlink_rna)
sc.tl.umap(f_scarlink_rna)
sc.tl.leiden(f_scarlink_rna, n_iterations=2)
f_scarlink_rna.obs['cell_anno'] = pd.Categorical(f_scarlink_rna.obs['cell_anno'], categories=['Normal B cell', 'Tumor B cell', 'T cell', 'Monocyte', 'Other'])
sc.pl.umap(f_scarlink_rna, color='cell_anno', legend_fontsize='7', legend_loc='right margin', size=5,
           title='', frameon=True, save='_tumor_b_scarlink.pdf')

## cisformer
sc.pp.log1p(predict_rna)
sc.tl.pca(predict_rna)
sc.pp.neighbors(predict_rna)
sc.tl.umap(predict_rna)
sc.tl.leiden(predict_rna, n_iterations=2)
predict_rna.obs['cell_anno'] = pd.Categorical(predict_rna.obs['cell_anno'], categories=['Normal B cell', 'Tumor B cell', 'T cell', 'Monocyte', 'Other'])
sc.pl.umap(predict_rna, color='cell_anno', legend_fontsize='7', legend_loc='right margin', size=5,
           title='', frameon=True, save='_tumor_b_cisformer.pdf')

def gene_wise_correlations(predict_rna, raw_rna, plot=False):
    predict_array = predict_rna.X.toarray()
    pred_df = pd.DataFrame(predict_array, index=predict_rna.obs.index, columns=predict_rna.var.index)
    raw_array = raw_rna.X.toarray()
    raw_df = pd.DataFrame(raw_array, index=raw_rna.obs.index, columns=raw_rna.var.index)

    # Peason Correlation
    raw_gp_corr_value = [np.corrcoef(predict_array[:, j], raw_array[:, j])[0,1] for j in range(raw_array.shape[1])]
    predict_rna.var["peason_corr"] = raw_gp_corr_value
    # plot peason correlation
    raw_gp_corr = predict_rna.var.copy()
    raw_gp_corr = raw_gp_corr.dropna()

    # Spearman Correlation
    raw_gs_corr_value = [spearmanr(predict_array[:, j], raw_array[:, j]) for j in range(raw_array.shape[1])]
    predict_rna.var["spearman_corr"] = [each[0] for each in raw_gs_corr_value]
    predict_rna.var["spearman_p_value"] = [each[1] for each in raw_gs_corr_value]
    # plot spearman correlation
    raw_gs_corr = predict_rna.var.copy()
    raw_gs_corr = raw_gs_corr.dropna()
    
    return predict_rna

def plot_comparing_scatter_plot(
    xdata,
    ydata,
    title = "Comparing Scatter",
    xlabel = "x",
    ylabel = "y",
    save_name = None
):
    
    data = pd.DataFrame({'x': xdata, 'y': ydata})
    data['Color'] = np.where(data['x'] > data['y'], 'Better', 'Worse')
    palette = {'Better': 'orangered', 'Worse': 'dodgerblue'}
    plt.figure(figsize=(8, 8))  # set width equal to height
    sns.scatterplot(data=data, x='x', y='y', hue='Color', palette=palette, edgecolor=None, alpha=1, s=5)  # 's' is to set size
    plt.plot([0, 1], [0, 1], linestyle='--', color='black')
    plt.title(title)
    plt.xlabel(r'$\rho_{' + xlabel + '}$')
    plt.ylabel(r'$\rho_{' + ylabel + '}$')
    if save_name:
        plt.savefig(save_name, dpi=600, bbox_inches='tight')
    else:
        plt.show()
    
    return None


## archr
predict_rna_0, archr_rna_0 = intersect_adata(predict_rna, f_archr_rna)
predict_rna_0, raw_rna_0 = intersect_adata(predict_rna_0, raw_rna)
predict_rna_1 = gene_wise_correlations(predict_rna_0, raw_rna_0)
archr_rna_1 = gene_wise_correlations(archr_rna_0, raw_rna_0)
plot_comparing_scatter_plot(predict_rna_1.var["peason_corr"].to_list(), archr_rna_1.var["peason_corr"].to_list(), title="", xlabel='cisformer', ylabel='archr', save_name='CA_pearson_scatter.pdf')

## scarlink
predict_rna_0, scarlink_rna_0 = intersect_adata(predict_rna, scarlink_rna)  # not use f_scarlink_rna???
predict_rna_0, raw_rna_0 = intersect_adata(predict_rna_0, raw_rna)
# scarlink_rna_0.X[raw_rna_0.X == 0] = 0
predict_rna_1 = gene_wise_correlations(predict_rna_0, raw_rna_0)
scarlink_rna_1 = gene_wise_correlations(scarlink_rna_0, raw_rna_0)
# plot_comparing_scatter_plot(predict_rna_1.var["peason_corr"].to_list(), scarlink_rna_1.var["peason_corr"].to_list(), title="", xlabel='cisformer', ylabel='scarlink', save_name='test_pearson.pdf')
# plot_comparing_scatter_plot(predict_rna_1.var["spearman_corr"].to_list(), scarlink_rna_1.var["spearman_corr"].to_list(), title="", xlabel='cisformer', ylabel='scarlink', save_name='test_spearman.pdf')
plot_comparing_scatter_plot(predict_rna_1.var["peason_corr"].to_list(), scarlink_rna_1.var["peason_corr"].to_list(), title="", xlabel='cisformer', ylabel='scarlink', save_name='CS_pearson_scatter.pdf')
plot_comparing_scatter_plot(predict_rna_1.var["spearman_corr"].to_list(), scarlink_rna_1.var["spearman_corr"].to_list(), title="", xlabel='cisformer', ylabel='scarlink', save_name='CS_spearman_scatter.pdf')
