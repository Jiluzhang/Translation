#################### calculate Voronoi area for each cell ####################
## workdir: /fs/home/jiluzhang/spatial_mechano/pipeline_2
import pandas as pd
import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d
import matplotlib.pyplot as plt
from tqdm import *
from scipy import stats
import scanpy as sc

df = pd.read_csv('R1_R77_4C4_cell_metadata.csv')
df = df[df['status']=='ok']  # 79088

vor= Voronoi(df[['global_x', 'global_y']])

# fig = voronoi_plot_2d(vor, point_size=1, show_vertices=False, line_width=0.5)
# plt.savefig('all_fov.pdf')
# plt.close()

# df_sub = df[(df['global_x']>1000) & (df['global_x']<1500) & (df['global_y']>1000) & (df['global_y']<1500)]
# vor_sub= Voronoi(df_sub[['global_x', 'global_y']])
# fig = voronoi_plot_2d(vor_sub, point_size=1, show_vertices=False, line_width=0.5)
# plt.savefig('fov_1000_1500.pdf')
# plt.close()

# vertices should be ranked based on plot
def poly_area(vertices):
    x = vertices[:, 0]
    y = vertices[:, 1]
    return (0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))).item()

## calculate area
areas = []
region_indices = []
for idx, region in enumerate(vor.regions):
    # delete blank or no-border regions
    if not region or -1 in region:
        continue
    
    polygon_vertices = vor.vertices[region]
    area = poly_area(polygon_vertices)
    areas.append(area)
    region_indices.append(idx)

## assign area for each point
# 输入点 i → vor.point_region[i] → 区域索引 R → vor.regions[R] → 顶点坐标 → 计算面积 → 存储到 point_areas[i]
# ~ 3 min
point_areas = np.full(df.shape[0], np.nan)
for point_idx, region_idx in enumerate(tqdm(vor.point_region, ncols=80)):
    if region_idx in region_indices:
        area_idx = region_indices.index(region_idx)
        point_areas[point_idx] = areas[area_idx]

df['areas'] = point_areas


adata_raw = sc.read_h5ad('../cytospace_Luz/overall_merfish.h5ad')
adata = adata_raw[adata_raw.obs['sample_id']=='R77_4C4'].copy()  # 72962 × 238
adata.obs.index = [i.split('-')[0] for i in adata.obs.index]

## renormalization
# raw_cnt = pd.read_csv('../cytospace_Luz/R1_R77_4C4_single_cell_raw_counts.csv', index_col=0)
# adata.X = np.array(raw_cnt.loc[adata.obs.index])
# sc.pp.normalize_total(adata, target_sum=1e4)
# sc.pp.log1p(adata)

adata.obs['areas'] = np.nan
df['cell_id'] = df['cell_id'].astype(str)
idx =  np.intersect1d(df['cell_id'], adata.obs.index)
adata.obs.loc[idx, 'areas'] = df[df['cell_id'].isin(idx)]['areas'].values
adata.write('overall_merfish_areas.h5ad')

#### specifically for PIEZO2 (median is not significant)
ct_lst = []
s_gex_lst = []
b_gex_lst = []
p_value_lst = []
for ct in adata.obs['populations'].drop_duplicates():
    ct_lst.append(ct)
    adata_ct = adata[(adata.obs['populations']==ct) & (adata.obs['areas']<1000), :].copy()
    area_mid = adata_ct.obs['areas'].median().item()
    s_gex_lst.append(adata_ct.X[adata_ct.obs['areas']<adata_ct.obs['areas'].quantile(0.2).item(), adata_ct.var.index=='PIEZO2'].mean().item())
    b_gex_lst.append(adata_ct.X[adata_ct.obs['areas']>adata_ct.obs['areas'].quantile(0.8).item(), adata_ct.var.index=='PIEZO2'].mean().item())
    p_value_lst.append(stats.ttest_ind(adata_ct.X[adata_ct.obs['areas']<adata_ct.obs['areas'].quantile(0.2).item(), adata_ct.var.index=='PIEZO2'],
                                       adata_ct.X[adata_ct.obs['areas']>adata_ct.obs['areas'].quantile(0.8).item(), adata_ct.var.index=='PIEZO2'])[1].item())

p_value_lst[0]   # 0.010926409206141654
p_value_lst[-4]  # 0.011584681009410555
ct_lst[0]        # VIC
ct_lst[-4]       # Epicardial

#### test for all genes
ct_lst = []
genes_lst = []
fc_lst = []
p_lst = []
for ct in tqdm(adata.obs['populations'].drop_duplicates(), ncols=80):
    adata_ct = adata[(adata.obs['populations']==ct) & (adata.obs['areas']<1000), :].copy()
    for gene in adata_ct.var.index:
        ct_lst.append(ct)
        genes_lst.append(gene)
        s_gex = adata_ct.X[adata_ct.obs['areas']<adata_ct.obs['areas'].quantile(0.2).item(), adata_ct.var.index==gene].mean().item()
        b_gex = adata_ct.X[adata_ct.obs['areas']>adata_ct.obs['areas'].quantile(0.8).item(), adata_ct.var.index==gene].mean().item()
        fc_lst.append(np.log2((s_gex+0.001)/(b_gex+0.001)).item())
        p_value = stats.ttest_ind(adata_ct.X[adata_ct.obs['areas']<adata_ct.obs['areas'].quantile(0.2).item(), adata_ct.var.index==gene],
                                  adata_ct.X[adata_ct.obs['areas']>adata_ct.obs['areas'].quantile(0.8).item(), adata_ct.var.index==gene])[1].item() 
        p_lst.append(p_value)

res = pd.DataFrame({'ct':ct_lst, 'gene':genes_lst, 'log2fc':fc_lst, 'pvalue':p_lst})
res.to_csv('log2fc_pvalue.txt', sep='\t', index=False)

## plot volcano
import pandas as pd
import numpy as np
from plotnine import *
import matplotlib.pyplot as plt

plt.rcParams['pdf.fonttype'] = 42

df = pd.read_table('log2fc_pvalue.txt')
df['-log10(pvalue)'] = -np.log10(df['pvalue'])
df['idx'] = 'not_sig'
df.loc[(df['log2fc']>np.log2(1.2))  &  (df['-log10(pvalue)']>-np.log10(0.05)),  'idx'] = 'up'
df.loc[(df['log2fc']<-np.log2(1.2))  &  (df['-log10(pvalue)']>-np.log10(0.05)), 'idx'] = 'down'
df['idx'] = pd.Categorical(df['idx'], categories=['up', 'down', 'not_sig'])

df['idx'].value_counts()
# not_sig    6011
# up          225
# down        190

for ct in df['ct'].drop_duplicates():
    df_ct = df[df['ct']==ct]
    p = ggplot(df_ct, aes(x='log2fc', y='-log10(pvalue)', color='idx')) + geom_point(size=0.5) +\
                                                                          scale_color_manual(values={'up':'red', 'down':'blue', 'not_sig':'gray'}) +\
                                                                          geom_text(data=df_ct[df_ct['idx']!='not_sig'], mapping=aes(label='gene'), 
                                                                                    size=8, color='black', nudge_x=0.1, nudge_y=0.1) + theme_bw()
                                                                          # scale_x_continuous(limits=[-1, 1], breaks=np.arange(-1, 1+0.1, 0.4)) +\
                                                                          # scale_y_continuous(limits=[0, 3], breaks=np.arange(0, 3+0.1, 0.5)) +\
                                                                          # geom_vline(xintercept=[-np.log2(1.5), np.log2(1.5)], linetype='dashed', color='black', size=0.5) +\
                                                                          # geom_hline(yintercept=-np.log10(0.05), linetype='dashed', color='black', size=0.5)
    p.save(filename=ct+'_volcano.pdf', dpi=600, height=4, width=5)

## VIC
df_ct = df[df['ct']=='VIC']
p = ggplot(df_ct, aes(x='log2fc', y='-log10(pvalue)', color='idx')) + geom_point(size=0.5) +\
                                                                      scale_color_manual(values={'up':'red', 'down':'blue', 'not_sig':'gray'}) +\
                                                                      scale_x_continuous(limits=[-1.5, 1.5], breaks=np.arange(-1.5, 1.5+0.1, 0.5)) +\
                                                                      scale_y_continuous(limits=[0, 4], breaks=np.arange(0, 4+0.1, 1.0)) +\
                                                                      geom_text(data=df_ct[df_ct['idx']!='not_sig'], mapping=aes(label='gene'), 
                                                                                size=8, color='black', nudge_x=0.1, nudge_y=0.1) + theme_bw()
                                                                      # geom_vline(xintercept=[-np.log2(1.5), np.log2(1.5)], linetype='dashed', color='black', size=0.5) +\
                                                                      # geom_hline(yintercept=-np.log10(0.05), linetype='dashed', color='black', size=0.5)
p.save(filename='VIC_volcano_mod.pdf', dpi=600, height=4, width=5)

## Epicardial
df_ct = df[df['ct']=='Epicardial']
p = ggplot(df_ct, aes(x='log2fc', y='-log10(pvalue)', color='idx')) + geom_point(size=0.5) +\
                                                                      scale_color_manual(values={'up':'red', 'down':'blue', 'not_sig':'gray'}) +\
                                                                      scale_x_continuous(limits=[-2, 2], breaks=np.arange(-2, 2+0.1, 1.0)) +\
                                                                      scale_y_continuous(limits=[0, 5], breaks=np.arange(0, 5+0.1, 1.0)) +\
                                                                      geom_text(data=df_ct[df_ct['idx']!='not_sig'], mapping=aes(label='gene'), 
                                                                                size=8, color='black', nudge_x=0.1, nudge_y=0.1) + theme_bw()
                                                                      # geom_vline(xintercept=[-np.log2(1.5), np.log2(1.5)], linetype='dashed', color='black', size=0.5) +\
                                                                      # geom_hline(yintercept=-np.log10(0.05), linetype='dashed', color='black', size=0.5)
p.save(filename='Epicardial_volcano_mod.pdf', dpi=600, height=4, width=5)

## BEC: blood endothelial cell
df_ct = df[df['ct']=='BEC']
p = ggplot(df_ct, aes(x='log2fc', y='-log10(pvalue)', color='idx')) + geom_point(size=0.5) +\
                                                                      scale_color_manual(values={'up':'red', 'down':'blue', 'not_sig':'gray'}) +\
                                                                      scale_x_continuous(limits=[-1, 1], breaks=np.arange(-1, 1+0.1, 0.5)) +\
                                                                      scale_y_continuous(limits=[0, 3], breaks=np.arange(0, 3+0.1, 0.5)) +\
                                                                      geom_text(data=df_ct[df_ct['idx']!='not_sig'], mapping=aes(label='gene'), 
                                                                                size=8, color='black', nudge_x=0.1, nudge_y=0.1) + theme_bw()
                                                                      # geom_vline(xintercept=[-np.log2(1.5), np.log2(1.5)], linetype='dashed', color='black', size=0.5) +\
                                                                      # geom_hline(yintercept=-np.log10(0.05), linetype='dashed', color='black', size=0.5)
p.save(filename='BEC_volcano_mod.pdf', dpi=600, height=4, width=5)

## aFibro: atria firoblast
df_ct = df[df['ct']=='aFibro']
p = ggplot(df_ct, aes(x='log2fc', y='-log10(pvalue)', color='idx')) + geom_point(size=0.5) +\
                                                                      scale_color_manual(values={'up':'red', 'down':'blue', 'not_sig':'gray'}) +\
                                                                      scale_x_continuous(limits=[-1.25, 1.25], breaks=np.arange(-1.25, 1.25+0.1, 0.25)) +\
                                                                      scale_y_continuous(limits=[0, 9], breaks=np.arange(0, 9+0.1, 3)) +\
                                                                      geom_text(data=df_ct[df_ct['idx']!='not_sig'], mapping=aes(label='gene'), 
                                                                                size=8, color='black', nudge_x=0.1, nudge_y=0.1) + theme_bw()
                                                                      # geom_vline(xintercept=[-np.log2(1.5), np.log2(1.5)], linetype='dashed', color='black', size=0.5) +\
                                                                      # geom_hline(yintercept=-np.log10(0.05), linetype='dashed', color='black', size=0.5)
p.save(filename='aFibro_volcano_mod.pdf', dpi=600, height=4, width=5)
