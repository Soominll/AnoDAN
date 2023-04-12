#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import pandas as pd
import keras
import keras.backend as K
import pickle


# #### Overall sen & res score

# In[ ]:


#importing some variables

with open('processed_subpath.p','rb') as f:
    pdata = pickle.load(f)

varnamelist = ['all_exp', 'all_path', 'Total_gene_list', 'globals_0', 'senders_0', 'receivers_0', 'edges_0',
               'Total_gene', 'Total_data', 'gene_path_idx', 's_drug_CL', 'r_drug_CL', 'real_dt', 'real_resi_dt']
for vn in varnamelist:
    exec("%s=pdata['%s']"%(vn,vn))


# In[ ]:


# importing sen & res anomaly scores

with open('processed_subpath_scores.p','rb') as f:
    pdata = pickle.load(f)

varnamelist = ['sen_score', 'sen_score2', 'sen_gene', 'res_score', 'res_score2', 'res_gene']
for vn in varnamelist:
    exec("%s=pdata['%s']"%(vn,vn))


# In[ ]:


a = pd.DataFrame({'sens': sen_score})
b = pd.DataFrame(res_score)
b = b.rename(columns={0: 'res'})
sens_res_score = pd.concat([a,b], axis=1)


# #### sen_total vs. res_total

# In[ ]:


sen_total = []   #median gene scores in each pathway for total cell line (1 drug)
for i in range(len(sen_gene)):
    ge = sen_gene[i].ravel()
    sen_total.append([np.median(ge[gp]) for gp in gene_path_idx])
sen_total = np.array(sen_total)

res_total = []
for i in range(len(res_gene)):
    ge = res_gene[i].ravel()
    res_total.append([np.median(ge[gp]) for gp in gene_path_idx])
res_total = np.array(res_total)


# In[ ]:


df = pd.DataFrame(sen_total.mean(axis=0)- res_total.mean(axis=0)) #difference between sens & res total gene score 
sns.barplot(x='index',y=0,data=df.reset_index())


# In[ ]:


np.sort(sen_total.mean(axis=0)- res_total.mean(axis=0))[:5]


# In[ ]:


print(sen_total.mean())
print(res_total.mean())
print(abs(sen_total.mean()-res_total.mean()))


# In[ ]:


paths = pd.read_csv('Data/List_pathways_34.csv')
paths = paths.loc[paths['num']!=114]


# In[ ]:


paths.loc[np.argsort(sen_total.mean(axis=0)- res_total.mean(axis=0))[:5]] #top 5 pathways with the largest difference between sens & res total gene score


# #### All tissue type

# In[ ]:


topgenes = [Total_gene[np.argsort(i[0])[::-1][:int(Total_gene.size*0.01)]] for i in res_gene] #Top 1%
topscores = [np.sort(i[0])[::-1][:int(Total_gene.size*0.01)] for i in res_gene]


# In[ ]:


Total_topgenes, count = np.unique(topgenes, return_counts=True) 


# In[ ]:


count_sort_ind = np.argsort(-count) #sort by count


# In[ ]:


Total_topgenes = Total_topgenes[count_sort_ind] #genes sort by count


# In[ ]:


co = count[count_sort_ind] #sort by count


# In[ ]:


plt.bar(np.arange(0,co.shape[0]), co, color='dodgerblue')


# In[ ]:


# Total pathways - enriched pathways

gcontain = []
for tgl in Total_gene_list:
    gcontain.append(np.intersect1d(Total_topgenes, tgl).size / tgl.size) #calculate proportion of enriched top genes for each pathway


# In[ ]:


paths.iloc[np.argsort(gcontain)[::-1][:5]] #top 5 enriched pathways


# In[ ]:


np.sort(gcontain)[::-1][:5]


# In[ ]:


# Top genes in the top 1 pathway

e = 'Gene_expression/Mapped_gene_expression118.csv' #P53 signaling pathway
Expression_dt = pd.read_csv(e)
Expression_dt


# #### LUAD

# In[ ]:


resistantCL = 'Data/List_resistantCL_TP53mt.csv'
resistantCL = pd.read_csv(resistantCL)
resistantCL = resistantCL.loc[resistantCL['Z_IC50']>0.5]
resistantCL = resistantCL.loc[resistantCL['Pubchem.ID']== 52918385]


# In[ ]:


rtra = (resistantCL['TCGA_Desc'] == 'LUAD').values


# In[ ]:


r_drug_CL = resistantCL.iloc[:,[0,2,5]]


# In[ ]:


res_gene_luad = [i for (i, v) in zip(res_gene, rtra) if v]


# In[ ]:


avg_gene_score = np.mean(res_gene_luad, axis=0).ravel()


# In[ ]:


topgenes = [Total_gene[np.argsort(i[0])[::-1][:int(Total_gene.size*0.01)]] for i in res_gene_luad] #Top 1%
topscores = [np.sort(i[0])[::-1][:int(Total_gene.size*0.01)] for i in res_gene_luad]


# In[ ]:


# Most enriched pathways

gcontain = []
for tgl in Total_gene_list:
    gcontain.append(np.intersect1d(topgenes, tgl).size / tgl.size) #calculate proportion of enriched top genes for each pathway


# In[ ]:


paths.iloc[np.argsort(gcontain)[::-1][:5]] #top 5 enriched pathways


# In[ ]:


# Top genes in the top 1 pathway

e = 'Gene_expression/Mapped_gene_expression160.csv' #P53 signaling pathway
e2 = 'Gene_expression/Mapped_gene_expression149.csv' #TGFB signaling pathway
Expression_dt = pd.read_csv(e)
Expression_dt2 = pd.read_csv(e2)


# In[ ]:


## Gene scores

Expression_dt.index = Expression_dt['ENTREZID']
subgex = Expression_dt.loc[np.intersect1d(topgenes, Total_gene_list[20])] #top genes in the selected pathway
Expression_dt2.index = Expression_dt2['ENTREZID']
subgex2 = Expression_dt2.loc[np.intersect1d(topgenes, Total_gene_list[29])] 


# In[ ]:


np.intersect1d(subgex['ENTREZID'].values, subgex2['ENTREZID'].values) #overlapping top genes between two pathways


# In[ ]:


######## Total_topgenes

Total_topgenes, count = np.unique(topgenes, return_counts=True) 


# In[ ]:


count_sort_ind = np.argsort(-count) #sort by count
Total_topgenes = Total_topgenes[count_sort_ind] #genes sort by count


# In[ ]:


co = count[count_sort_ind]


# In[ ]:


Total_topgenes[count_sort_ind]


# In[ ]:


plt.bar(np.arange(0,co.shape[0]), co, color='dodgerblue')


# In[ ]:


##### Most enriched pathways

gcontain = []
for tgl in Total_gene_list:
    gcontain.append(np.intersect1d(Total_topgenes, tgl).size / tgl.size) #calculate proportion of enriched top genes for each pathway


# In[ ]:


paths.iloc[np.argsort(gcontain) [::-1][:5]] #top 5 enriched pathways


# In[ ]:


paths.iloc[np.argsort(gcontain)[::-1]] #pathway rankings


# In[ ]:


x, x_ind, _ = np.intersect1d(Total_topgenes, Total_gene_list[0], return_indices=True)


# In[ ]:


##### pathway score with weight (count)

gcontain = []
for tgl in Total_gene_list:
    x, x_ind, _ = np.intersect1d(Total_topgenes, tgl, return_indices=True)
    gcontain.append(np.intersect1d(Total_topgenes, tgl).size / tgl.size*np.mean(count[x_ind])) #with count


# In[ ]:


##### pathway score with weight (ave_gene_score)

gcontain = []
for tgl in Total_gene_list:
    x, x_ind, _ = np.intersect1d(Total_topgenes, tgl, return_indices=True)
    idx = np.in1d(Total_gene, x).nonzero()[0]
    gcontain.append(np.intersect1d(Total_topgenes, tgl).size / tgl.size*np.mean(avg_gene_score[idx])) #with avg_gene_score


# In[ ]:


paths.iloc[np.argsort(gcontain)[::-1][:5]] #top 5 enriched pathways


# In[ ]:


paths.iloc[np.argsort(gcontain)[::-1]] #top pathways


# In[ ]:


## Gene scores

Expression_dt.index = Expression_dt['ENTREZID']
subgex = Expression_dt.loc[np.intersect1d(Total_topgenes[:25], Total_gene_list[20])] #top genes in the selected pathway
Expression_dt2.index = Expression_dt2['ENTREZID']
subgex2 = Expression_dt2.loc[np.intersect1d(Total_topgenes[:25], Total_gene_list[29])] 


# In[ ]:


np.intersect1d(subgex['ENTREZID'].values, subgex2['ENTREZID'].values)

