## workdir: /fs/home/jiluzhang/Nature_methods/Figure_2/pan_cancer/cisformer/Fibroblast/subtype

import scanpy as sc
import harmonypy as hm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import rcParams

plt.rcParams['pdf.fonttype'] = 42

rna_raw = sc.read_h5ad('cancer_fibro_annotation.h5ad')           # 19138 × 37982
rna = rna_raw[rna_raw.obs['cell_type']!='iFibro_FBLN1'].copy()   # 14738 × 37982
sc.pp.filter_genes(rna, min_cells=10)                            # 14738 × 19400
rna.obs['cell_type'].value_counts()
# eFibro_COL11A1    8675
# apFibro_CCL5      2736
# qFibro_SAA1       1304
# eFibro_CTHRC1     1260
# MyoFibro_MYH11     763

np.percentile(np.count_nonzero(rna.X.toarray(), axis=1), 80)   # 2455.0
np.percentile(np.count_nonzero(rna.X.toarray(), axis=1), 95)   # 3580.0
np.percentile(np.count_nonzero(rna.X.toarray(), axis=1), 100)  # 8406.0

rna.write('rna.h5ad')

atac = sc.read_h5ad('cancer_fibro_atac_annotation.h5ad')       # 19138 × 1033239
atac = atac[rna_raw.obs['cell_type']!='iFibro_FBLN1'].copy()   # 14738 × 1033239
sc.pp.filter_genes(atac, min_cells=10)                         # 14738 × 368527
atac.write('atac.h5ad')

sc.pp.normalize_total(rna, target_sum=1e4)
sc.pp.log1p(rna)
sc.pp.highly_variable_genes(rna, batch_key='cancer_type')
rna.raw = rna
rna = rna[:, rna.var.highly_variable]   # 19138 × 2239
sc.tl.pca(rna)
ho = hm.run_harmony(rna.obsm['X_pca'], rna.obs, 'cancer_type')  # Converged after 2 iterations

rna.obsm['X_pca_harmony'] = ho.Z_corr.T
sc.pp.neighbors(rna, use_rep='X_pca_harmony')
sc.tl.umap(rna)

# sc.tl.leiden(rna, resolution=0.75)
# rna.obs['leiden'].value_counts()
# # 0     3575
# # 1     2887
# # 2     2293
# # 3     2238
# # 4     1914
# # 5     1461
# # 6     1164
# # 7      988
# # 8      824
# # 9      747
# # 10     537
# # 11     357
# # 12     153

# sc.pl.matrixplot(rna, ['MYH11', 'RGS5', 'NDUFA4L2', 'ADIRF', 
#                        'CCL5', 'HLA-DRA', 'LYZ', 'TYROBP', 'HLA-DRB1', 'HLA-DPB1',
#                        'COL11A1',
#                        'CTHRC1', 'MMP11', 'COL1A1',
#                        'FBLN1', 'FBLN2', 'CFD', 'GSN',
#                        'SAA1'],
#                  dendrogram=True, groupby='leiden', cmap='coolwarm', standard_scale='var', save='leiden_heatmap.pdf')

rna.obs['cell_type'] = pd.Categorical(rna.obs['cell_type'], categories=['qFibro_SAA1', 'eFibro_CTHRC1', 'eFibro_COL11A1', 'apFibro_CCL5', 'MyoFibro_MYH11'])

sc.settings.set_figure_params(dpi_save=600)
rcParams["figure.figsize"] = (2, 2)

sc.pl.umap(rna, color='cell_type', legend_fontsize='5', legend_loc='right margin', size=5,
           title='', frameon=True, save='_cell_type_fibro.pdf')

sc.pl.matrixplot(rna, ['SAA1',
                       'CTHRC1', 'MMP11', 'COL1A1',
                       'COL11A1',
                       'CCL5', 'HLA-DRA', 'LYZ',
                       'MYH11', 'RGS5', 'ADIRF'],
                 dendrogram=False, groupby='cell_type', cmap='coolwarm', standard_scale='var', save='cell_type_heatmap.pdf')

rna.write('rna_res.h5ad')

## train Cisformer
python split_train_val.py --RNA rna.h5ad --ATAC atac.h5ad --train_pct 0.9
python data_preprocess.py -r rna_train.h5ad -a atac_train.h5ad -s train_pt --dt train -n train --config rna2atac_config_train.yaml
python data_preprocess.py -r rna_val.h5ad -a atac_val.h5ad -s val_pt --dt val -n val --config rna2atac_config_val.yaml
nohup accelerate launch --config_file accelerator_config_train.yaml --main_process_port 29823 rna2atac_train.py --config_file rna2atac_config_train.yaml \
                        --train_data_dir train_pt --val_data_dir val_pt -s save -n rna2atac > 20250402.log &    # 2426617



######################################################################## qFibro_SAA1 ########################################################################
import pandas as pd
import numpy as np
import random
from functools import partial
import sys
sys.path.append("M2Mmodel")
from utils import PairDataset
import scanpy as sc
import torch
from M2Mmodel.utils import *
from collections import Counter
import pickle as pkl
import yaml
from M2Mmodel.M2M import M2M_rna2atac

import h5py
from tqdm import tqdm

from multiprocessing import Pool
import pickle

rna = sc.read_h5ad('rna.h5ad')
random.seed(0)
saa1_idx = list(np.argwhere(rna.obs['cell_type']=='qFibro_SAA1').flatten())
saa1_idx_100 = random.sample(list(saa1_idx), 100)
out = sc.AnnData(rna[saa1_idx_100].X, obs=rna[saa1_idx_100].obs, var=rna[saa1_idx_100].var)
np.count_nonzero(out.X.toarray(), axis=1).max()  # 5681
out.write('rna_saa1_100.h5ad')

atac = sc.read_h5ad('atac.h5ad')
atac_saa1 = atac[rna.obs.index[saa1_idx_100]].copy()
atac_saa1.write('atac_saa1_100.h5ad')

# python data_preprocess.py -r rna_saa1_100.h5ad -a atac_saa1_100.h5ad -s saa1_pt_100 --dt test -n saa1 --config rna2atac_config_test_saa1.yaml   # enc_max_len: 5681 

with open("rna2atac_config_test_saa1.yaml", "r") as f:
    config = yaml.safe_load(f)

model = M2M_rna2atac(
            dim = config["model"].get("dim"),

            enc_num_gene_tokens = config["model"].get("total_gene") + 1, # +1 for <PAD>
            enc_num_value_tokens = config["model"].get('max_express') + 1, # +1 for <PAD>
            enc_depth = config["model"].get("enc_depth"),
            enc_heads = config["model"].get("enc_heads"),
            enc_ff_mult = config["model"].get("enc_ff_mult"),
            enc_dim_head = config["model"].get("enc_dim_head"),
            enc_emb_dropout = config["model"].get("enc_emb_dropout"),
            enc_ff_dropout = config["model"].get("enc_ff_dropout"),
            enc_attn_dropout = config["model"].get("enc_attn_dropout"),

            dec_depth = config["model"].get("dec_depth"),
            dec_heads = config["model"].get("dec_heads"),
            dec_ff_mult = config["model"].get("dec_ff_mult"),
            dec_dim_head = config["model"].get("dec_dim_head"),
            dec_emb_dropout = config["model"].get("dec_emb_dropout"),
            dec_ff_dropout = config["model"].get("dec_ff_dropout"),
            dec_attn_dropout = config["model"].get("dec_attn_dropout")
        )
model = model.half()
model.load_state_dict(torch.load('./save/2025-04-02_rna2atac_30/pytorch_model.bin'))
device = torch.device('cuda:6')
model.to(device)
model.eval()

saa1_data = torch.load("./saa1_pt_100/saa1_0.pt")
dataset = PreDataset(saa1_data)
dataloader_kwargs = {'batch_size': 1, 'shuffle': False}
loader = torch.utils.data.DataLoader(dataset, **dataloader_kwargs)

rna_saa1 = sc.read_h5ad('rna_saa1_100.h5ad')
atac_saa1 = sc.read_h5ad('atac_saa1_100.h5ad')
out = rna_saa1.copy()
out.X = np.zeros(out.shape)
i = 0
for inputs in tqdm(loader, ncols=80, desc='output attention matrix'):
    rna_sequence, rna_value, atac_sequence, _, enc_pad_mask = [each.to(device) for each in inputs]
    out.X[i, (rna_sequence[rna_sequence!=0]-1).cpu()] = model.generate_attn_weight(rna_sequence, atac_sequence, rna_value, enc_mask=enc_pad_mask, which='decoder')[0].sum(axis=0)
    torch.cuda.empty_cache()
    i += 1

out.write('attn_saa1_100.h5ad')

######################################################################## eFibro_CTHRC1 ########################################################################
import pandas as pd
import numpy as np
import random
from functools import partial
import sys
sys.path.append("M2Mmodel")
from utils import PairDataset
import scanpy as sc
import torch
from M2Mmodel.utils import *
from collections import Counter
import pickle as pkl
import yaml
from M2Mmodel.M2M import M2M_rna2atac

import h5py
from tqdm import tqdm

from multiprocessing import Pool
import pickle

rna = sc.read_h5ad('rna.h5ad')
random.seed(0)
cthrc1_idx = list(np.argwhere(rna.obs['cell_type']=='eFibro_CTHRC1').flatten())
cthrc1_idx_100 = random.sample(list(cthrc1_idx), 100)
out = sc.AnnData(rna[cthrc1_idx_100].X, obs=rna[cthrc1_idx_100].obs, var=rna[cthrc1_idx_100].var)
np.count_nonzero(out.X.toarray(), axis=1).max()  # 5324
out.write('rna_cthrc1_100.h5ad')

atac = sc.read_h5ad('atac.h5ad')
atac_cthrc1 = atac[rna.obs.index[cthrc1_idx_100]].copy()
atac_cthrc1.write('atac_cthrc1_100.h5ad')

# python data_preprocess.py -r rna_cthrc1_100.h5ad -a atac_cthrc1_100.h5ad -s cthrc1_pt_100 --dt test -n cthrc1 --config rna2atac_config_test_cthrc1.yaml   # enc_max_len: 5324 

with open("rna2atac_config_test_cthrc1.yaml", "r") as f:
    config = yaml.safe_load(f)

model = M2M_rna2atac(
            dim = config["model"].get("dim"),

            enc_num_gene_tokens = config["model"].get("total_gene") + 1, # +1 for <PAD>
            enc_num_value_tokens = config["model"].get('max_express') + 1, # +1 for <PAD>
            enc_depth = config["model"].get("enc_depth"),
            enc_heads = config["model"].get("enc_heads"),
            enc_ff_mult = config["model"].get("enc_ff_mult"),
            enc_dim_head = config["model"].get("enc_dim_head"),
            enc_emb_dropout = config["model"].get("enc_emb_dropout"),
            enc_ff_dropout = config["model"].get("enc_ff_dropout"),
            enc_attn_dropout = config["model"].get("enc_attn_dropout"),

            dec_depth = config["model"].get("dec_depth"),
            dec_heads = config["model"].get("dec_heads"),
            dec_ff_mult = config["model"].get("dec_ff_mult"),
            dec_dim_head = config["model"].get("dec_dim_head"),
            dec_emb_dropout = config["model"].get("dec_emb_dropout"),
            dec_ff_dropout = config["model"].get("dec_ff_dropout"),
            dec_attn_dropout = config["model"].get("dec_attn_dropout")
        )
model = model.half()
model.load_state_dict(torch.load('./save/2025-04-02_rna2atac_30/pytorch_model.bin'))
device = torch.device('cuda:6')
model.to(device)
model.eval()

cthrc1_data = torch.load("./cthrc1_pt_100/cthrc1_0.pt")
dataset = PreDataset(cthrc1_data)
dataloader_kwargs = {'batch_size': 1, 'shuffle': False}
loader = torch.utils.data.DataLoader(dataset, **dataloader_kwargs)

rna_cthrc1 = sc.read_h5ad('rna_cthrc1_100.h5ad')
atac_cthrc1 = sc.read_h5ad('atac_cthrc1_100.h5ad')
out = rna_cthrc1.copy()
out.X = np.zeros(out.shape)
i = 0
for inputs in tqdm(loader, ncols=80, desc='output attention matrix'):
    rna_sequence, rna_value, atac_sequence, _, enc_pad_mask = [each.to(device) for each in inputs]
    out.X[i, (rna_sequence[rna_sequence!=0]-1).cpu()] = model.generate_attn_weight(rna_sequence, atac_sequence, rna_value, enc_mask=enc_pad_mask, which='decoder')[0].sum(axis=0)
    torch.cuda.empty_cache()
    i += 1

out.write('attn_cthrc1_100.h5ad')


######################################################################## eFibro_COL11A1 ########################################################################
import pandas as pd
import numpy as np
import random
from functools import partial
import sys
sys.path.append("M2Mmodel")
from utils import PairDataset
import scanpy as sc
import torch
from M2Mmodel.utils import *
from collections import Counter
import pickle as pkl
import yaml
from M2Mmodel.M2M import M2M_rna2atac

import h5py
from tqdm import tqdm

from multiprocessing import Pool
import pickle

rna = sc.read_h5ad('rna.h5ad')
random.seed(0)
col11a1_idx = list(np.argwhere(rna.obs['cell_type']=='eFibro_COL11A1').flatten())
col11a1_idx_100 = random.sample(list(col11a1_idx), 100)
out = sc.AnnData(rna[col11a1_idx_100].X, obs=rna[col11a1_idx_100].obs, var=rna[col11a1_idx_100].var)
np.count_nonzero(out.X.toarray(), axis=1).max()  # 3553
out.write('rna_col11a1_100.h5ad')

atac = sc.read_h5ad('atac.h5ad')
atac_col11a1 = atac[rna.obs.index[col11a1_idx_100]].copy()
atac_col11a1.write('atac_col11a1_100.h5ad')

# python data_preprocess.py -r rna_col11a1_100.h5ad -a atac_col11a1_100.h5ad -s col11a1_pt_100 --dt test -n col11a1 --config rna2atac_config_test_col11a1.yaml   # enc_max_len: 3553 

with open("rna2atac_config_test_col11a1.yaml", "r") as f:
    config = yaml.safe_load(f)

model = M2M_rna2atac(
            dim = config["model"].get("dim"),

            enc_num_gene_tokens = config["model"].get("total_gene") + 1, # +1 for <PAD>
            enc_num_value_tokens = config["model"].get('max_express') + 1, # +1 for <PAD>
            enc_depth = config["model"].get("enc_depth"),
            enc_heads = config["model"].get("enc_heads"),
            enc_ff_mult = config["model"].get("enc_ff_mult"),
            enc_dim_head = config["model"].get("enc_dim_head"),
            enc_emb_dropout = config["model"].get("enc_emb_dropout"),
            enc_ff_dropout = config["model"].get("enc_ff_dropout"),
            enc_attn_dropout = config["model"].get("enc_attn_dropout"),

            dec_depth = config["model"].get("dec_depth"),
            dec_heads = config["model"].get("dec_heads"),
            dec_ff_mult = config["model"].get("dec_ff_mult"),
            dec_dim_head = config["model"].get("dec_dim_head"),
            dec_emb_dropout = config["model"].get("dec_emb_dropout"),
            dec_ff_dropout = config["model"].get("dec_ff_dropout"),
            dec_attn_dropout = config["model"].get("dec_attn_dropout")
        )
model = model.half()
model.load_state_dict(torch.load('./save/2025-04-02_rna2atac_30/pytorch_model.bin'))
device = torch.device('cuda:6')
model.to(device)
model.eval()

col11a1_data = torch.load("./col11a1_pt_100/col11a1_0.pt")
dataset = PreDataset(col11a1_data)
dataloader_kwargs = {'batch_size': 1, 'shuffle': False}
loader = torch.utils.data.DataLoader(dataset, **dataloader_kwargs)

rna_col11a1 = sc.read_h5ad('rna_col11a1_100.h5ad')
atac_col11a1 = sc.read_h5ad('atac_col11a1_100.h5ad')
out = rna_col11a1.copy()
out.X = np.zeros(out.shape)
i = 0
for inputs in tqdm(loader, ncols=80, desc='output attention matrix'):
    rna_sequence, rna_value, atac_sequence, _, enc_pad_mask = [each.to(device) for each in inputs]
    out.X[i, (rna_sequence[rna_sequence!=0]-1).cpu()] = model.generate_attn_weight(rna_sequence, atac_sequence, rna_value, enc_mask=enc_pad_mask, which='decoder')[0].sum(axis=0)
    torch.cuda.empty_cache()
    i += 1

out.write('attn_col11a1_100.h5ad')


######################################################################## apFibro_CCL5 ########################################################################
import pandas as pd
import numpy as np
import random
from functools import partial
import sys
sys.path.append("M2Mmodel")
from utils import PairDataset
import scanpy as sc
import torch
from M2Mmodel.utils import *
from collections import Counter
import pickle as pkl
import yaml
from M2Mmodel.M2M import M2M_rna2atac

import h5py
from tqdm import tqdm

from multiprocessing import Pool
import pickle

rna = sc.read_h5ad('rna.h5ad')
random.seed(0)
ccl5_idx = list(np.argwhere(rna.obs['cell_type']=='apFibro_CCL5').flatten())
ccl5_idx_100 = random.sample(list(ccl5_idx), 100)
out = sc.AnnData(rna[ccl5_idx_100].X, obs=rna[ccl5_idx_100].obs, var=rna[ccl5_idx_100].var)
np.count_nonzero(out.X.toarray(), axis=1).max()  # 5740
out.write('rna_ccl5_100.h5ad')

atac = sc.read_h5ad('atac.h5ad')
atac_ccl5 = atac[rna.obs.index[ccl5_idx_100]].copy()
atac_ccl5.write('atac_ccl5_100.h5ad')

# python data_preprocess.py -r rna_ccl5_100.h5ad -a atac_ccl5_100.h5ad -s ccl5_pt_100 --dt test -n ccl5 --config rna2atac_config_test_ccl5.yaml   # enc_max_len: 5740 

with open("rna2atac_config_test_ccl5.yaml", "r") as f:
    config = yaml.safe_load(f)

model = M2M_rna2atac(
            dim = config["model"].get("dim"),

            enc_num_gene_tokens = config["model"].get("total_gene") + 1, # +1 for <PAD>
            enc_num_value_tokens = config["model"].get('max_express') + 1, # +1 for <PAD>
            enc_depth = config["model"].get("enc_depth"),
            enc_heads = config["model"].get("enc_heads"),
            enc_ff_mult = config["model"].get("enc_ff_mult"),
            enc_dim_head = config["model"].get("enc_dim_head"),
            enc_emb_dropout = config["model"].get("enc_emb_dropout"),
            enc_ff_dropout = config["model"].get("enc_ff_dropout"),
            enc_attn_dropout = config["model"].get("enc_attn_dropout"),

            dec_depth = config["model"].get("dec_depth"),
            dec_heads = config["model"].get("dec_heads"),
            dec_ff_mult = config["model"].get("dec_ff_mult"),
            dec_dim_head = config["model"].get("dec_dim_head"),
            dec_emb_dropout = config["model"].get("dec_emb_dropout"),
            dec_ff_dropout = config["model"].get("dec_ff_dropout"),
            dec_attn_dropout = config["model"].get("dec_attn_dropout")
        )
model = model.half()
model.load_state_dict(torch.load('./save/2025-04-02_rna2atac_30/pytorch_model.bin'))
device = torch.device('cuda:6')
model.to(device)
model.eval()

ccl5_data = torch.load("./ccl5_pt_100/ccl5_0.pt")
dataset = PreDataset(ccl5_data)
dataloader_kwargs = {'batch_size': 1, 'shuffle': False}
loader = torch.utils.data.DataLoader(dataset, **dataloader_kwargs)

rna_ccl5 = sc.read_h5ad('rna_ccl5_100.h5ad')
atac_ccl5 = sc.read_h5ad('atac_ccl5_100.h5ad')
out = rna_ccl5.copy()
out.X = np.zeros(out.shape)
i = 0
for inputs in tqdm(loader, ncols=80, desc='output attention matrix'):
    rna_sequence, rna_value, atac_sequence, _, enc_pad_mask = [each.to(device) for each in inputs]
    out.X[i, (rna_sequence[rna_sequence!=0]-1).cpu()] = model.generate_attn_weight(rna_sequence, atac_sequence, rna_value, enc_mask=enc_pad_mask, which='decoder')[0].sum(axis=0)
    torch.cuda.empty_cache()
    i += 1

out.write('attn_ccl5_100.h5ad')


######################################################################## MyoFibro_MYH11 ########################################################################
import pandas as pd
import numpy as np
import random
from functools import partial
import sys
sys.path.append("M2Mmodel")
from utils import PairDataset
import scanpy as sc
import torch
from M2Mmodel.utils import *
from collections import Counter
import pickle as pkl
import yaml
from M2Mmodel.M2M import M2M_rna2atac

import h5py
from tqdm import tqdm

from multiprocessing import Pool
import pickle

rna = sc.read_h5ad('rna.h5ad')
random.seed(0)
myh11_idx = list(np.argwhere(rna.obs['cell_type']=='MyoFibro_MYH11').flatten())
myh11_idx_100 = random.sample(list(myh11_idx), 100)
out = sc.AnnData(rna[myh11_idx_100].X, obs=rna[myh11_idx_100].obs, var=rna[myh11_idx_100].var)
np.count_nonzero(out.X.toarray(), axis=1).max()  # 4482
out.write('rna_myh11_100.h5ad')

atac = sc.read_h5ad('atac.h5ad')
atac_myh11 = atac[rna.obs.index[myh11_idx_100]].copy()
atac_myh11.write('atac_myh11_100.h5ad')

# python data_preprocess.py -r rna_myh11_100.h5ad -a atac_myh11_100.h5ad -s myh11_pt_100 --dt test -n myh11 --config rna2atac_config_test_myh11.yaml   # enc_max_len: 4482 

with open("rna2atac_config_test_myh11.yaml", "r") as f:
    config = yaml.safe_load(f)

model = M2M_rna2atac(
            dim = config["model"].get("dim"),

            enc_num_gene_tokens = config["model"].get("total_gene") + 1, # +1 for <PAD>
            enc_num_value_tokens = config["model"].get('max_express') + 1, # +1 for <PAD>
            enc_depth = config["model"].get("enc_depth"),
            enc_heads = config["model"].get("enc_heads"),
            enc_ff_mult = config["model"].get("enc_ff_mult"),
            enc_dim_head = config["model"].get("enc_dim_head"),
            enc_emb_dropout = config["model"].get("enc_emb_dropout"),
            enc_ff_dropout = config["model"].get("enc_ff_dropout"),
            enc_attn_dropout = config["model"].get("enc_attn_dropout"),

            dec_depth = config["model"].get("dec_depth"),
            dec_heads = config["model"].get("dec_heads"),
            dec_ff_mult = config["model"].get("dec_ff_mult"),
            dec_dim_head = config["model"].get("dec_dim_head"),
            dec_emb_dropout = config["model"].get("dec_emb_dropout"),
            dec_ff_dropout = config["model"].get("dec_ff_dropout"),
            dec_attn_dropout = config["model"].get("dec_attn_dropout")
        )
model = model.half()
model.load_state_dict(torch.load('./save/2025-04-02_rna2atac_30/pytorch_model.bin'))
device = torch.device('cuda:6')
model.to(device)
model.eval()

myh11_data = torch.load("./myh11_pt_100/myh11_0.pt")
dataset = PreDataset(myh11_data)
dataloader_kwargs = {'batch_size': 1, 'shuffle': False}
loader = torch.utils.data.DataLoader(dataset, **dataloader_kwargs)

rna_myh11 = sc.read_h5ad('rna_myh11_100.h5ad')
atac_myh11 = sc.read_h5ad('atac_myh11_100.h5ad')
out = rna_myh11.copy()
out.X = np.zeros(out.shape)
i = 0
for inputs in tqdm(loader, ncols=80, desc='output attention matrix'):
    rna_sequence, rna_value, atac_sequence, _, enc_pad_mask = [each.to(device) for each in inputs]
    out.X[i, (rna_sequence[rna_sequence!=0]-1).cpu()] = model.generate_attn_weight(rna_sequence, atac_sequence, rna_value, enc_mask=enc_pad_mask, which='decoder')[0].sum(axis=0)
    torch.cuda.empty_cache()
    i += 1

out.write('attn_myh11_100.h5ad')




###### cell type specific factor heatmap
import scanpy as sc
import pandas as pd
import numpy as np
from plotnine import *
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['pdf.fonttype'] = 42

cr = ['SMARCA4', 'SMARCA2', 'ARID1A', 'ARID1B', 'SMARCB1',
      'CHD1', 'CHD2', 'CHD3', 'CHD4', 'CHD5', 'CHD6', 'CHD7', 'CHD8', 'CHD9',
      'BRD2', 'BRD3', 'BRD4', 'BRDT',
      'SMARCA5', 'SMARCA1', 'ACTL6A', 'ACTL6B',
      'SSRP1', 'SUPT16H',
      'EP400',
      'SMARCD1', 'SMARCD2', 'SMARCD3']   # 28
tf_lst = pd.read_table('TF_jaspar.txt', header=None)
tf = list(tf_lst[0].values)   # 735
cr_tf = cr + tf   # 763

attn_saa1 = sc.read_h5ad('attn_saa1_100.h5ad')
attn_cthrc1 = sc.read_h5ad('attn_cthrc1_100.h5ad')
attn_col11a1 = sc.read_h5ad('attn_col11a1_100.h5ad')
attn_ccl5 = sc.read_h5ad('attn_ccl5_100.h5ad')
attn_myh11 = sc.read_h5ad('attn_myh11_100.h5ad')

df_fibro = pd.DataFrame({'gene':attn_saa1.var.index.values,
                         'attn_norm_saa1':(attn_saa1.X.sum(axis=0) / (attn_saa1.X.sum(axis=0).max())),
                         'attn_norm_cthrc1':(attn_cthrc1.X.sum(axis=0) / (attn_cthrc1.X.sum(axis=0).max())),
                         'attn_norm_col11a1':(attn_col11a1.X.sum(axis=0) / (attn_col11a1.X.sum(axis=0).max())),
                         'attn_norm_ccl5':(attn_ccl5.X.sum(axis=0) / (attn_ccl5.X.sum(axis=0).max())),
                         'attn_norm_myh11':(attn_myh11.X.sum(axis=0) / (attn_myh11.X.sum(axis=0).max()))})
df_fibro = df_fibro[df_fibro['gene'].isin(cr_tf)]

saa1_sp = df_fibro[(df_fibro['attn_norm_saa1']>df_fibro['attn_norm_cthrc1']) & 
                   (df_fibro['attn_norm_saa1']>df_fibro['attn_norm_col11a1']) &
                   (df_fibro['attn_norm_saa1']>df_fibro['attn_norm_ccl5']) &
                   (df_fibro['attn_norm_saa1']>df_fibro['attn_norm_myh11'])]
saa1_sp = saa1_sp.sort_values('attn_norm_saa1', ascending=False)
saa1_sp = saa1_sp[saa1_sp['attn_norm_saa1']>(1e-5)]
saa1_sp.index = range(saa1_sp.shape[0])

cthrc1_sp = df_fibro[(df_fibro['attn_norm_cthrc1']>df_fibro['attn_norm_saa1']) & 
                     (df_fibro['attn_norm_cthrc1']>df_fibro['attn_norm_col11a1']) &
                     (df_fibro['attn_norm_cthrc1']>df_fibro['attn_norm_ccl5']) &
                     (df_fibro['attn_norm_cthrc1']>df_fibro['attn_norm_myh11'])]
cthrc1_sp = cthrc1_sp.sort_values('attn_norm_cthrc1', ascending=False)
cthrc1_sp = cthrc1_sp[cthrc1_sp['attn_norm_cthrc1']>(1e-5)]
cthrc1_sp.index = range(cthrc1_sp.shape[0])

col11a1_sp = df_fibro[(df_fibro['attn_norm_col11a1']>df_fibro['attn_norm_saa1']) & 
                      (df_fibro['attn_norm_col11a1']>df_fibro['attn_norm_cthrc1']) &
                      (df_fibro['attn_norm_col11a1']>df_fibro['attn_norm_ccl5']) &
                      (df_fibro['attn_norm_col11a1']>df_fibro['attn_norm_myh11'])]
col11a1_sp = col11a1_sp.sort_values('attn_norm_col11a1', ascending=False)
col11a1_sp = col11a1_sp[col11a1_sp['attn_norm_col11a1']>(1e-5)]
col11a1_sp.index = range(col11a1_sp.shape[0])

ccl5_sp = df_fibro[(df_fibro['attn_norm_ccl5']>df_fibro['attn_norm_saa1']) & 
                   (df_fibro['attn_norm_ccl5']>df_fibro['attn_norm_cthrc1']) &
                   (df_fibro['attn_norm_ccl5']>df_fibro['attn_norm_col11a1']) &
                   (df_fibro['attn_norm_ccl5']>df_fibro['attn_norm_myh11'])]
ccl5_sp = ccl5_sp.sort_values('attn_norm_ccl5', ascending=False)
ccl5_sp = ccl5_sp[ccl5_sp['attn_norm_ccl5']>(1e-5)]
ccl5_sp.index = range(ccl5_sp.shape[0])

myh11_sp = df_fibro[(df_fibro['attn_norm_myh11']>df_fibro['attn_norm_saa1']) & 
                    (df_fibro['attn_norm_myh11']>df_fibro['attn_norm_cthrc1']) &
                    (df_fibro['attn_norm_myh11']>df_fibro['attn_norm_col11a1']) &
                    (df_fibro['attn_norm_myh11']>df_fibro['attn_norm_ccl5'])]
myh11_sp = myh11_sp.sort_values('attn_norm_myh11', ascending=False)
myh11_sp = myh11_sp[myh11_sp['attn_norm_myh11']>(1e-5)]
myh11_sp.index = range(myh11_sp.shape[0])

sp = pd.concat([saa1_sp, cthrc1_sp, col11a1_sp, ccl5_sp, myh11_sp])
sp.index = sp['gene'].values

p = sns.clustermap(sp[['attn_norm_saa1', 'attn_norm_cthrc1', 'attn_norm_col11a1', 'attn_norm_ccl5', 'attn_norm_myh11']],
                   cmap='coolwarm', row_cluster=False, col_cluster=False, z_score=0, vmin=-1, vmax=1, figsize=(5, 150)) 

# ['RBPJ', 'TCF7', 'LEF1', 'BACH2',
#  'PRDM1', 'EOMES', 'TBX21', 'YY1', 'FOXO3', 'TFAP4', 'STAT5A',
#  'TFAP4', 'NR3C1', 'MAF', 'NFKB2', 'NFKB1', 'BATF', 'NFATC1']

yticklabels = p.ax_heatmap.get_yticklabels()
for label in yticklabels:
    text = label.get_text()
    if text not in ['FOXP1', 'SMAD3', 'KLF2', 'BACH2',  'TCF7',
                    'ESR2', 'IKZF2', 'MLX', 'JUNB', 'BATF3', 
                    'CHD2', 'NFKB2', 'NR3C1', 'BATF', 'NFKB1']:
        label.set_text(f"")

p.ax_heatmap.set_yticklabels(yticklabels, rotation=0)
p.savefig('fibro_specific_factor_heatmap.pdf')
plt.close()











