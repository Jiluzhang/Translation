## SEE github: https://github.com/4dglab/SEE

# conda env create -f environment.yml
# conda activate see

#### pairs to cool
for file in *pairs.gz; do
    s="${file%.pairs.gz}"
    cooler cload pairs -c1 2 -p1 3 -c2 4 -p2 5 mm10.chrom.sizes:10000 $file $s.cool
    echo "$s" done
done

#### cool to h5ad
import cooler
import pandas as pd
import os
from joblib import Parallel, delayed
import anndata

folder_path = '/fs/home/jiluzhang/scHiC_integration/HiRES/cools'

parallel = Parallel(n_jobs=3, backend='loky', verbose=1)

## sum contact by bins & parallel
def load_coolers(folder_path):
    def load_cooler(folder_path, file_name):
        c = cooler.Cooler(os.path.join(folder_path, file_name))
        contact = c.pixels(join=True)[:]
        contact = contact[contact['start1']!=contact['start2']]
        binsize, chromsizes = c.binsize, c.chromsizes

        _1 = contact.groupby(['chrom1', 'start1'])['count'].sum()
        _2 = contact.groupby(['chrom2', 'start2'])['count'].sum()
        _1.index.names = _2.index.names = ['chrom', 'start']
        _1, _2 = _1[_1!=0], _2[_2!=0]
        info = pd.concat([_1, _2], axis=1).fillna(0).sum(axis=1).sort_index()
        
        _indexs = set([(chrom, int(i * binsize))
                   for chrom in chromsizes.keys()
                   for i in range(int(chromsizes[chrom]/binsize)+1)])
        _indexs -= set(info.index)
        info = pd.concat([info, pd.Series([0]*len(_indexs), index=list(_indexs))]).sort_index()

        return info.to_frame().astype('float16').rename(columns={0:file_name})

    joblist = []
    for root, dirs, files in os.walk(folder_path, topdown=False):
        for file_name in files:
            joblist.append(delayed(load_cooler)(folder_path, file_name))

    infos = parallel(joblist)
    infos = pd.concat(infos, axis=1).fillna(0).sort_index()
    return infos

infos = load_coolers(folder_path)

## write to h5ad
obs = pd.DataFrame(infos.T.index, columns=['cells'])
obs.insert(obs.shape[1] - 1, 'domain', 'scHiC')
obs = obs.set_index('cells')
var = infos.reset_index()[['chrom', 'start']].set_index(infos.index.map('{0[0]}_{0[1]}'.format))
infos.index = infos.index.map('{0[0]}_{0[1]}'.format)
infos = anndata.AnnData(X=infos.T, obs=obs, var=var)
infos.write("/fs/home/jiluzhang/scHiC_integration/HiRES/scHiC.h5ad", compression="gzip")


#### scRNA preprocessing
## wget -c https://ftp.ncbi.nlm.nih.gov/geo/series/GSE223nnn/GSE223917/suppl/GSE223917%5FHiRES%5Femb%5Fmetadata%2Exlsx            # GSE223917_HiRES_emb_metadata.xlsx
## wget -c https://ftp.ncbi.nlm.nih.gov/geo/series/GSE223nnn/GSE223917/suppl/GSE223917%5FHiRES%5Femb%2Erna%2Eumicount%2Etsv%2Egz  # GSE223917_HiRES_emb.rna.umicount.tsv
## pip install openpyxl

import pandas as pd
import anndata

scRNA_metadata = pd.read_excel('/fs/home/jiluzhang/scHiC_integration/HiRES/GSE223917_HiRES_emb_metadata.xlsx')   # 7469*24
#           Cellname  Rawreads  DNAreads  RNAreads       y/x  Raw contacts  ...  Stage             Celltype   RMSD 20k  CDPS cluster  Sub_k_cluster  Cellcycle phase
# 0      GasaE751001  0.788963  0.704156  0.073214  0.010131        293142  ...    E75         ExE ectoderm  22.312863             9              2                M
# 1      GasaE751002  1.194513  1.117651  0.065995  0.028088        593838  ...    E75      neural ectoderm   0.891031            11              3           Late-S

## set index
target_scRNA_metadata = scRNA_metadata.copy()
target_scRNA_metadata = target_scRNA_metadata.set_index('Cellname')
target_scRNA_metadata.index.name = 'sample_name'

## read gex data
infos = pd.DataFrame()
for chunk in pd.read_table('/fs/home/jiluzhang/scHiC_integration/HiRES/GSE223917_HiRES_emb.rna.umicount.tsv.gz', chunksize=10000):
    infos = pd.concat([infos, chunk])

infos = infos.set_index('gene')  # 50463*7469
#                GasaE751001  GasaE751002  GasaE751003  GasaE751004  GasaE751005  ...  OrgeEX053380  OrgeEX053381  OrgeEX053382  OrgeEX053383  OrgeEX053384
# gene                                                                            ...                                                                      
# 0610005C13Rik          0.0          0.0          0.0          0.0          0.0  ...           0.0           0.0           0.0           0.0           0.0
# 0610006L08Rik          0.0          0.0          0.0          0.0          0.0  ...           0.0           0.0           0.0           0.0           0.0

infos = infos.T
infos.index.name = 'sample_name'  # 7469*50463

## write to h5ad
infos = anndata.AnnData(X=infos)

_target_scRNA_metadata = target_scRNA_metadata.copy()
_target_scRNA_metadata = _target_scRNA_metadata.loc[infos.obs.index]
infos.obs["cell_type"] = _target_scRNA_metadata['Celltype']
infos.obs['domain'] = 'scRNA'

infos.write("/fs/home/jiluzhang/scHiC_integration/HiRES/scRNA.h5ad", compression="gzip")







#### BandNorm github: https://github.com/sshen82/BandNorm
#### BandNorm tutorial: https://sshen82.github.io/BandNorm/articles/BandNorm-tutorial.html
conda create -n BandNorm
conda activate BandNorm

conda install conda-forge::r-devtools
conda install -c conda-forge r-rcpparmadillo r-rcppeigen r-harmony r-rtsne r-umap r-strawr
conda install -c conda-forge mamba
mamba install bioconda::bioconductor-genomicinteractions
mamba install conda-forge::r-seurat

devtools::install_github('sshen82/BandNorm', build_vignettes=FALSE)
















