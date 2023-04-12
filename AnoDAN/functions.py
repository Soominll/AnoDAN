#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import umap.umap_ as umap
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import pandas as pd
import keras
import keras.backend as K
from tensorflow.keras.utils import to_categorical


# In[ ]:


### Edges

def edges(Pathway_data):
    Pathway_dt0 = np.array(Pathway_data.loc[:,'subtype']).reshape(-1,1)
    a = to_categorical([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15], num_classes=16) 
    Pathway_dt = []
    for i in range(len(Pathway_dt0)):
        if (Pathway_dt0[i]=="compound"):
            Pathway_dt.append(a[0])
        elif (Pathway_dt0[i]=="hidden compound"):
            Pathway_dt.append(a[1])
        elif (Pathway_dt0[i]=="activation"):
            Pathway_dt.append(a[2])
        elif (Pathway_dt0[i]=="inhibition"):
            Pathway_dt.append(a[3])
        elif (Pathway_dt0[i]=="expression"):
            Pathway_dt.append(a[4])
        elif (Pathway_dt0[i]=="repression"):
            Pathway_dt.append(a[5])
        elif (Pathway_dt0[i]=="indirect effect" or Pathway_dt0[i]=="indirect"):
            Pathway_dt.append(a[6])
        elif (Pathway_dt0[i]=="state change"):
            Pathway_dt.append(a[7])
        elif (Pathway_dt0[i]=="binding/association"):
            Pathway_dt.append(a[8])
        elif (Pathway_dt0[i]=="dissociation"):
            Pathway_dt.append(a[9])
        elif (Pathway_dt0[i]=="missing interaction"):
            Pathway_dt.append(a[10])
        elif (Pathway_dt0[i]=="phosphorylation"):
            Pathway_dt.append(a[11])
        elif (Pathway_dt0[i]=="dephosphorylation"):
            Pathway_dt.append(a[12])
        elif (Pathway_dt0[i]=="glycosylation"):
            Pathway_dt.append(a[13])
        elif (Pathway_dt0[i]=="ubiquitination"):
            Pathway_dt.append(a[14])
        elif (Pathway_dt0[i]=="methylation"):
            Pathway_dt.append(a[15])

    Pathway_dt0 = np.array(Pathway_data.loc[:,'type']).reshape(-1,1)
    b = to_categorical([0,1,2,3,4], num_classes=5) 
    Pathway_dt2 = []
    for i in range(len(Pathway_dt0)):
        if (Pathway_dt0[i]=="ECrel"):
            Pathway_dt2.append(b[0])
        elif (Pathway_dt0[i]=="PPrel"):
            Pathway_dt2.append(b[1])
        elif (Pathway_dt0[i]=="GErel"):
            Pathway_dt2.append(b[2])
        elif (Pathway_dt0[i]=="PCrel"):
            Pathway_dt2.append(b[3])
        elif (Pathway_dt0[i]=="maplink"):
            Pathway_dt2.append(b[4])

    edges = np.hstack((Pathway_dt, Pathway_dt2))

    return edges


# In[ ]:


### Senders & receivers & globals

def numextract(string):
    number = int(''.join(filter(str.isdigit,string)))
    return number

def others(Pathway_data, Expression_dt):
    globals_ = [0.]

    senders = []
    receivers = []
    e_from = np.array(Pathway_data.loc[:,'from']).reshape(-1,1)
    e_to = np.array(Pathway_data.loc[:,'to']).reshape(-1,1)

    expression = Expression_dt[1:,2].reshape(-1,1)

    for i in range(len(e_from)):
        for j in range(len(expression)):
            if (float(numextract(str(e_from[i]))) == float(expression[j])):
                senders.append(float(j))
            if (float(numextract(str(e_to[i]))) == float(expression[j])):
                receivers.append(float(j))
    globals_ = np.array(globals_)
    senders = np.array(senders)
    receivers = np.array(receivers)

    return globals_, senders, receivers


# In[ ]:


def create_data(mb_idx, drug_CL, Total_data):
    x_data = []

    for i in range(0,len(mb_idx)):
        CL = drug_CL.iloc[mb_idx[i],].values[1]
        x_data.append(list(Total_data.loc[CL].values))
    x_data = np.array(x_data, dtype=np.float32)

    return x_data


# In[ ]:


# using seaborn to plot UMAP

def umapplot(dataset, dataset_resi, csample, epoch):
    chunksize = 32
    fake = []
    for shard in range(dataset.shape[0] // chunksize):
        z = np.random.normal(size=(chunksize, z_dim))
        gen = G(z, csample[shard*chunksize:(shard+1)*chunksize], istraining=False)
        fake.append(gen.numpy())
    gen = G(np.random.normal(size=(csample[(shard+1)*chunksize:].shape[0], z_dim)), csample[(shard+1)*chunksize:], istraining=False)
    fake.append(gen.numpy())
    fake = np.concatenate(fake, axis=0)

    real = dataset
    real_r = dataset_resi
    
    realfake = np.concatenate((real, fake), axis=0)
    label = ['Sensitive cell']* real.shape[0]
    label.extend(['Generated cell']*fake.shape[0])

    fit = umap.UMAP(n_neighbors=15, min_dist=0.1) #, metric='correlation'
    u = fit.fit_transform(realfake)
    
    df =pd.DataFrame({'UMAP1':u[:,0], 'UMAP2':u[:,1], '':label})
    sns.set_context("paper")

    a = sns.relplot(
    data=df, x="UMAP1", y="UMAP2",
    hue='',
    kind="scatter",alpha=0.6, s=15
)
    sns.despine()
    a.set(xticklabels=[],yticklabels=[])
    axes = a.axes.flat[0]
    axes.tick_params(left=False, bottom=False)
    
   # plt.savefig('umapplot_final.png', dpi=1200)
   # plt.show()

