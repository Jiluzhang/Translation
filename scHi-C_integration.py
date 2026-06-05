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





