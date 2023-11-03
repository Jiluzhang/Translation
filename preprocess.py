import scanpy as sc

dat = sc.read_10x_h5('/fs/home/jiluzhang/scMOG_raw/scMOG_modified/dataset/DM_rep4.h5', gex_only=False)

rna = dat[dat.var['feature_types']=='Peaks']
