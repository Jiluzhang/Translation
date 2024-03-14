# mkdir normal_h5ad

# cCREs: wget -c https://downloads.wenglab.org/V3/GRCh38-cCREs.bed
awk '{if($1!="chrX" && $1!="chrY") print($1 "\t" $2 "\t" $3)}' GRCh38-cCREs.bed | sort -k1,1V -k2,2n | bedtools merge -i stdin > human_cCREs.bed  # 1,033,239

# human gtf file: wget -c https://ftp.ensembl.org/pub/release-110/gtf/homo_sapiens/Homo_sapiens.GRCh38.110.chr.gtf.gz
# gunzip Homo_sapiens.GRCh38.110.chr.gtf.gz
## cetered by gene_name
grep gene_name Homo_sapiens.GRCh38.110.chr.gtf | awk -F "\t" '{if($1!="X" && $1!="Y" && $1!="MT" && $3=="gene") print$9}' | awk '{print $2}' | sed 's/"//g' | sed 's/;//g' > gene_id.txt
grep gene_name Homo_sapiens.GRCh38.110.chr.gtf | awk -F "\t" '{if($1!="X" && $1!="Y" && $1!="MT" && $3=="gene") print$9}' | awk '{print $6}' | sed 's/"//g' | sed 's/;//g' > gene_name.txt
grep gene_name Homo_sapiens.GRCh38.110.chr.gtf | awk -F "\t" '{if($1!="X" && $1!="Y" && $1!="MT" && $3=="gene") print("chr" $1 "\t" $4 "\t" $5 "\t" $7)}' > gene_pos.txt
awk '{if($4=="+") print($1 "\t" $2); else print($1 "\t" $3)}' gene_pos.txt > gene_tss.txt
paste gene_id.txt gene_name.txt gene_tss.txt | sort -k3,3V -k4,4n | awk '{print $1 "\t" $2}' > human_genes_raw.txt
rm gene_id.txt gene_name.txt gene_pos.txt gene_tss.txt

## remove duplicates
import pandas as pd
dat = pd.read_table('human_genes_raw.txt', names=['gene_id', 'gene_name'])
gene_cnt = dat.groupby('gene_name').agg('count').reset_index()
gene_cnt.columns = ['gene_name', 'cnt']
out = pd.merge(dat, gene_cnt)
out = out[out['cnt']==1][['gene_id', 'gene_name']]
out.to_csv('human_genes.txt', sep='\t', index=False, header=False)  # 38,244

rm human_genes_raw.txt





## cetered by gene_name
grep gene_name Homo_sapiens.GRCh38.110.chr.gtf | awk -F "\t" '{if($1!="X" && $1!="Y" && $1!="MT" && $3=="gene") print$9}' | awk '{print $2}' | sed 's/"//g' | sed 's/;//g' > gene_id.txt
grep gene_name Homo_sapiens.GRCh38.110.chr.gtf | awk -F "\t" '{if($1!="X" && $1!="Y" && $1!="MT" && $3=="gene") print$9}' | awk '{print $6}' | sed 's/"//g' | sed 's/;//g' > gene_name.txt
grep gene_name Homo_sapiens.GRCh38.110.chr.gtf | awk -F "\t" '{if($1!="X" && $1!="Y" && $1!="MT" && $3=="gene") print("chr" $1 "\t" $4 "\t" $5 "\t" $7)}' > gene_pos.txt
awk '{if($4=="+") print($1 "\t" $2); else print($1 "\t" $3)}' gene_pos.txt > gene_tss.txt
paste gene_id.txt gene_name.txt gene_tss.txt | sort -k3,3V -k4,4n | awk '{print $3 "\t" $4 "\t" $4+1 "\t" $2 "\t" $1}' > gene_pos_id.txt
rm gene_id.txt gene_name.txt gene_pos.txt gene_tss.txt

## remove duplicates
import pandas as pd
# output unique gene name
gene_pos_id = pd.read_table('gene_pos_id.txt', names=['chr', 'tss', 'tss_1', 'gene_name', 'gene_id'])
gene_cnt = gene_pos_id[['gene_name', 'gene_id']].groupby('gene_name').agg('count').reset_index()
gene_cnt.columns = ['gene_name', 'cnt']
out = pd.merge(gene_pos_id[['gene_name', 'gene_id']], gene_cnt, how='left')
out = out[out['cnt']==1][['gene_id', 'gene_name']]
out.to_csv('human_genes.txt', sep='\t', index=False, header=False)  # 38,244
# output unique gene name with pos info
out_2 = pd.merge(out, gene_pos_id, how='left')
out_2 = out_2[['chr', 'tss', 'tss_1', 'gene_name', 'gene_id']]
out_2.to_csv('human_genes_pos_id.txt', sep='\t', index=False, header=False)  # 38,244

rm human_genes_raw.txt gene_pos_id.txt 


##########################################################
import scanpy as sc
import numpy as np
import pandas as pd
import pybedtools

rna = sc.read_h5ad('rna_10.h5ad')
atac = sc.read_h5ad('atac_10.h5ad')

gene_pos_id = pd.read_table('human_genes_pos_id.txt', names=['chr', 'tss', 'tss_1', 'gene_name', 'gene_id'])
gene_pos_id.index = gene_pos_id['gene_id'].values

exp_gene = gene_pos_id.loc[rna.var['gene_ids'].tolist()][['chr', 'tss', 'tss_1']]
exp_gene['start'] = exp_gene['tss']-500000
exp_gene['start'][exp_gene['start']<0] = 0
exp_gene['end'] = exp_gene['tss']+500000
exp_gene = exp_gene[['chr', 'start', 'end']]
exp_gene['gene_idx'] = range(exp_gene.shape[0])

cCREs = pd.DataFrame({'chr': atac.var['gene_ids'].map(lambda x: x.split(':')[0]).values,
                      'start': atac.var['gene_ids'].map(lambda x: x.split(':')[1].split('-')[0]).values,
                      'end': atac.var['gene_ids'].map(lambda x: x.split(':')[1].split('-')[1]).values})
cCREs['peak_idx'] = range(cCREs.shape[0])

cCREs_bed = pybedtools.BedTool.from_dataframe(cCREs)
genes_bed = pybedtools.BedTool.from_dataframe(exp_gene)
idx_out = genes_bed.intersect(cCREs_bed, wa=True, wb=True).to_dataframe()

idx_dict = dict(idx_out.groupby(['name'], as_index=True)['thickEnd'].apply(lambda x: [i for i in x]))  # 38086
for i in range(38244):
    if i not in idx_dict.keys():
        idx_dict[i] = [] # 38244

pre_seq = np.zeros(atac.X.shape)
for i in range(pre_seq.shape[0]):
    exp_idx = np.where(rna.X[i]!=0)[0]  # exp_idx = np.where(rna.X[i]>1)[0]
    l = []
    for j in exp_idx:
        l += idx_dict[j] # maybe duplicated (it does not matter)
    pre_seq[i][l] = 1  # pre_seq[i][l] = np.round(np.random.random((len(pre_seq[i][l]))), 3)
    
    # idx_0 = np.where(pre_seq[i]==0)[0]
    # pre_seq[i][np.random.choice(idx_0, int(len(idx_0)*0.2))] = 1
    # idx_1 = np.where(pre_seq[i]==1)[0]
    # pre_seq[i][np.random.choice(idx_1, int(len(idx_1)*0.2))] = 0
    
    print('cell_'+str(i)+' done')

# pre_seq = np.random.randint(0, 2, atac.X.shape)
pre_seq_out = atac.copy()
pre_seq_out.X = pre_seq
pre_seq_out.write('atac_10_pre_seq_500k_2.h5ad')


predict_0 = sigmoid(predict.X[0])
predict_0[predict_0>0.36]=1
predict_0[predict_0<=0.36]=0
#len(np.intersect1d(np.where(predict_0==1)[0], np.where(true_0==1)))/sum(true_0==1)
len(np.intersect1d(np.where(predict_0==1)[0], np.where(true_0==1)))/sum(predict_0==1)




import torch
import numpy
import random
import torch.nn.functional as F
import torch.optim as optim
from torchmetrics import AUROC; from torcheval.metrics.functional import binary_auprc
 
x = torch.tensor(rna.X[0])
y = torch.tensor(atac.X[0])

class Net(torch.nn.Module):
    def __init__(self,n_feature,n_hidden,n_output):
        super(Net,self).__init__()
        #两层感知机
        self.hidden = torch.nn.Linear(n_feature,n_hidden)
        self.predict = torch.nn.Linear(n_hidden,n_output)
 
    def forward(self,x):
        x = F.relu(self.hidden(x))
        x = self.predict(x)
        return x

net = Net(38244, 1, 1033239)
optimizer = optim.Adam(net.parameters(),lr = 0.01)
loss_func = torch.nn.MSELoss()

auroc = AUROC(task='binary')
for t in range(200):
    prediction = net(x)
    loss = loss_func(torch.sigmoid(prediction),y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print('loss:', loss, 'auroc:', auroc(prediction, y).item())
    #print('auprc', binary_auprc(torch.sigmoid(prediction), y, num_tasks = prediction.shape[0]).mean().item())








import torch
import numpy as np
import random
import torch.nn.functional as F
import torch.optim as optim
from torchmetrics import AUROC; from torcheval.metrics.functional import binary_auroc, binary_auprc
from torch.utils.data import DataLoader, TensorDataset
import scanpy as sc

device = 'cuda:7'
rna = sc.read_h5ad('rna_32124.h5ad')
atac = sc.read_h5ad('atac_32124.h5ad')


rna_idx = np.where(np.count_nonzero(rna.X[:10000].toarray(), axis=1)>2000)[0]
atac_idx = np.where(np.count_nonzero(atac.X[:10000].toarray(), axis=1)>10000)[0]
idx = np.intersect1d(rna_idx, atac_idx)


x = torch.tensor(rna.X[idx].toarray()).to(device)
y = torch.tensor(atac.X[idx].toarray()).to(device)

dataset = TensorDataset(x, y)
dataloader = DataLoader(dataset, batch_size=10, shuffle=False)

class Net(torch.nn.Module):
    def __init__(self,n_feature,n_hidden,n_output):
        super(Net,self).__init__()
        #两层感知机
        self.hidden = torch.nn.Linear(n_feature,n_hidden)
        self.predict = torch.nn.Linear(n_hidden,n_output)
 
    def forward(self,x):
        x = F.relu(self.hidden(x))
        x = self.predict(x)
        return x


net = Net(38244, 64, 1033239).to(device)
#optimizer = optim.Adam(net.parameters(),lr = 0.002)
#loss_func = torch.nn.MSELoss()

lr_init = 0.001
for t in range(100):
    optimizer = optim.Adam(net.parameters(),lr = lr_init*pow(0.9, t+1))
    epoch_loss = 0
    for batch_idx, (X, Y) in enumerate(dataloader):
        prediction = net(X)
        #weight = Y.clone()
        #weight[weight==0]=1
        #atac_cnt = Y.sum(axis=1)
        #for i in range(10):
        #    weight[i] = 1/atac_cnt[i]*10000
        #weight[weight==1]=50
        #weight[weight<1]=1
        loss = F.binary_cross_entropy(torch.sigmoid(prediction), Y, reduction = "mean") #loss = loss_func(torch.sigmoid(prediction),Y)
        #loss = F.binary_cross_entropy(torch.sigmoid(prediction), Y, weight, reduction = "mean") #loss = loss_func(torch.sigmoid(prediction),Y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    #print('lr:', optimizer.state_dict()['param_groups'][0]['lr'])
        #if batch_idx % 1000 ==0:
        #  print(batch_idx)
    #print(binary_auprc(torch.sigmoid(prediction), Y, num_tasks = prediction.shape[0]))
    print('epoch:', t,
          'loss:', round(epoch_loss/(batch_idx+1)*100, 4),
          'auroc:', round(binary_auroc(torch.sigmoid(prediction), Y, num_tasks = prediction.shape[0]).mean().item(), 4),
          'auprc', round(binary_auprc(torch.sigmoid(prediction), Y, num_tasks = prediction.shape[0]).mean().item(), 4))


param_sum = 0
for name, param in model.named_parameters():
    if param.requires_grad:
        param_sum += np.prod(list(param.shape))

for name, param in model.named_parameters():
    if param.requires_grad:
        print(param.shape)

print(param_sum)



##########################################################



## out_paired_h5ad.py
import scanpy as sc
import pandas as pd
import numpy as np
import pybedtools
from tqdm import tqdm
from scipy.sparse import csr_matrix
from datetime import datetime

ids = pd.read_table('files_all.txt', header=None)

for i in range(len(ids)):
    print(ids[0][i], 'start', datetime.now().replace(microsecond=0))
    
    dat = sc.read_10x_mtx(ids[0][i]+'_out/normal', gex_only=False)
    
    ########## rna ##########
    rna = dat[:, dat.var['feature_types']=='Gene Expression'].copy()
    rna.X = np.array(rna.X.todense()) 
    genes = pd.read_table('cCRE_gene/human_genes.txt', names=['gene_id', 'gene_name'])
    
    rna_genes = rna.var.copy()
    rna_genes['gene_id'] = rna_genes['gene_ids'].map(lambda x: x.split('.')[0])  # delete ensembl id with version number
    rna_exp = pd.concat([rna_genes, pd.DataFrame(rna.X.T, index=rna_genes.index)], axis=1)
    
    X_new = pd.merge(genes, rna_exp, how='left', on='gene_id').iloc[:, 4:].T
    X_new = np.array(X_new, dtype='float32')
    X_new[np.isnan(X_new)] = 0
    
    rna_new_var = pd.DataFrame({'gene_ids': genes['gene_id'], 'feature_types': 'Gene Expression'})
    rna_new = sc.AnnData(X_new, obs=rna.obs, var=rna_new_var)  
    rna_new.var.index = genes['gene_name'].values  # set index for var
    rna_new.X = csr_matrix(rna_new.X)
    
    rna_new.write('normal_h5ad/'+ids[0][i]+'_rna.h5ad')
    
    
    ########## atac ##########
    atac = dat[:, dat.var['feature_types']=='Peaks'].copy()
    atac.X = np.array(atac.X.todense())
    atac.X[atac.X>1] = 1
    atac.X[atac.X<1] = 0
    
    peaks = pd.DataFrame({'id': atac.var_names})
    peaks['chr'] = peaks['id'].map(lambda x: x.split(':')[0])
    peaks['start'] = peaks['id'].map(lambda x: x.split(':')[1].split('-')[0])
    peaks['end'] = peaks['id'].map(lambda x: x.split(':')[1].split('-')[1])
    peaks.drop(columns='id', inplace=True)
    peaks['idx'] = range(peaks.shape[0])
    
    cCREs = pd.read_table('cCRE_gene/human_cCREs.bed', names=['chr', 'start', 'end'])
    cCREs['idx'] = range(cCREs.shape[0])
    cCREs_bed = pybedtools.BedTool.from_dataframe(cCREs)
    peaks_bed = pybedtools.BedTool.from_dataframe(peaks)
    idx_map = peaks_bed.intersect(cCREs_bed, wa=True, wb=True).to_dataframe().iloc[:, [3, 7]]
    idx_map.columns = ['peaks_idx', 'cCREs_idx']
    
    m = np.zeros([atac.n_obs, cCREs_bed.to_dataframe().shape[0]], dtype='float32')
    for j in tqdm(range(atac.X.shape[0])):
        m[j][idx_map[idx_map['peaks_idx'].isin(peaks.iloc[np.nonzero(atac.X[j])]['idx'])]['cCREs_idx']] = 1
    
    atac_new_var = pd.DataFrame({'gene_ids': cCREs['chr'] + ':' + cCREs['start'].map(str) + '-' + cCREs['end'].map(str), 'feature_types': 'Peaks'})
    atac_new = sc.AnnData(m, obs=atac.obs, var=atac_new_var)
    atac_new.var.index = atac_new.var['gene_ids'].values  # set index
    atac_new.X = csr_matrix(atac_new.X)
    
    atac_new.write('normal_h5ad/'+ids[0][i]+'_atac.h5ad')

    print(ids[0][i], 'end', datetime.now().replace(microsecond=0))


## plot cell stats
#grep from pan_cancer_process_20240108_2.log | grep QC | sed 's/.*: //g' | sed 's/ cells.*//g' > cells_qc.txt
#grep from pan_cancer_process_20240108_2.log | grep QC | sed 's/.*from //g' | sed 's/ cells.*//g' > cells_all.txt
#grep from pan_cancer_process_20240108_2.log | grep % | grep -v QC | sed -n '1~2p' | sed 's/.*: //g' | sed 's/ cells.*//g' > cells_normal.txt
#grep from pan_cancer_process_20240108_2.log | grep % | grep -v QC | sed -n '2~2p' | sed 's/.*: //g' | sed 's/ cells.*//g' > cells_tumor.txt

from plotnine import *
import pandas as pd
import numpy as np

## RAW
dat = pd.read_table('cells_all.txt', header=None)
dat.columns = ['count']
dat['idx'] = 'RAW'
np.median(dat['count'])  # 710663.0
p = ggplot(dat, aes(x='idx', y='count')) + geom_boxplot(width=0.5, color='black', fill='brown', outlier_shape='') + xlab('') + \
                                           scale_y_continuous(limits=[600000, 800000], breaks=range(600000, 800000+1, 50000)) + theme_bw()
p.save(filename='cells_all.pdf', dpi=600, height=5, width=4)

## QC
dat = pd.read_table('cells_qc.txt', header=None)
dat.columns = ['count']
dat['idx'] = 'QC'
np.median(dat['count'])  # 9828.0
p = ggplot(dat, aes(x='idx', y='count')) + geom_boxplot(width=0.5, color='black', fill='orange', outlier_shape='') + xlab('') + \
                                           scale_y_continuous(limits=[0, 20000], breaks=range(0, 20000+1, 5000)) + theme_bw()
p.save(filename='cells_qc.pdf', dpi=600, height=5, width=4)

## Normal
dat = pd.read_table('cells_normal.txt', header=None)
dat.columns = ['count']
dat['idx'] = 'Normal'
np.median(dat['count'])  # 3398.5
p = ggplot(dat, aes(x='idx', y='count')) + geom_boxplot(width=0.5, color='black', fill='cornflowerblue', outlier_shape='') + xlab('') + \
                                           scale_y_continuous(limits=[0, 10000], breaks=range(0, 10000+1, 2000)) + theme_bw()
p.save(filename='cells_normal.pdf', dpi=600, height=5, width=4)

## Tumor
dat = pd.read_table('cells_tumor.txt', header=None)
dat.columns = ['count']
dat['idx'] = 'Tumor'
np.median(dat['count'])  # 2515.0
p = ggplot(dat, aes(x='idx', y='count')) + geom_boxplot(width=0.5, color='black', fill='darkolivegreen', outlier_shape='') + xlab('') + \
                                           scale_y_continuous(limits=[0, 10000], breaks=range(0, 10000+1, 2000)) + theme_bw()
p.save(filename='cells_tumor.pdf', dpi=600, height=5, width=4)



############################################ plot histogram for gene/peak count ############################################
from plotnine import *
import pandas as pd
import numpy as np
import scanpy as sc

ids = pd.read_table('normal_samples_id.txt', header=None)

rna_cnt = []
for i in range(len(ids)):
    rna = sc.read_h5ad(ids[0][i]+'_rna.h5ad')
    rna_cnt = rna_cnt + list(np.count_nonzero(rna.X.toarray(), axis=1))
    print(i, ids[0][i], 'RNA done')

rna_cnt_df = pd.DataFrame({'Genes': rna_cnt})
p = ggplot(rna_cnt_df)+ geom_histogram(aes(x='Genes'), color='black', fill='orange', bins=100) + \
                        scale_x_continuous(limits=[0, 9000], breaks=range(0, 9000+1, 1000)) + \
                        scale_y_continuous(limits=[0, 30000], breaks=range(0, 30000+1, 5000)) + \
                        theme_bw() + theme(panel_grid=element_blank())   # "y=after_stat('density')" for density plot
p.save(filename='gene_cnt_stats.pdf', dpi=600, height=5, width=5)

rna_cnt_df.to_csv('gene_cnt_stats.txt', index=False, header=False)


atac_cnt = []
for i in range(len(ids)):
    atac = sc.read_h5ad(ids[0][i]+'_atac.h5ad')
    atac_cnt = atac_cnt + list(np.count_nonzero(atac.X.toarray(), axis=1))
    print(i, ids[0][i], 'ATAC done')
    
atac_cnt_df = pd.DataFrame({'Peaks': atac_cnt})
p = ggplot(atac_cnt_df)+ geom_histogram(aes(x='Peaks'), color='black', fill='cornflowerblue', bins=100) + \
                         scale_x_continuous(limits=[0, 80000], breaks=range(0, 80000+1, 10000)) + \
                         scale_y_continuous(limits=[0, 50000], breaks=range(0, 50000+1, 10000)) + \
                         theme_bw() + theme(panel_grid=element_blank()) 
p.save(filename='peak_cnt_stats.pdf', dpi=600, height=5, width=5)

atac_cnt_df.to_csv('peak_cnt_stats.txt', index=False, header=False)


## normal samples
# CPT704DU-M1
# GBML018G1-M1N2
# GBML019G1-M1N1
# HT291C1-M1A3
# SN001H1-Ms1
# SP369H1-Mc1
# SP819H1-Mc1



