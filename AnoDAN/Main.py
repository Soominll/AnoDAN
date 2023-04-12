#!/usr/bin/env python
# coding: utf-8


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import umap.umap_ as umap
import seaborn as sns
import matplotlib.pyplot as plt
import time
import networkx as nx
import numpy as np
import sonnet as snt
import tensorflow as tf
import pandas as pd
import os
import keras
import keras.backend as K

from tensorflow.keras.utils import to_categorical
from tensorflow.python.keras.callbacks import TensorBoard

from graph_nets import blocks
from graph_nets import graphs
from graph_nets import modules
from graph_nets import utils_np
from graph_nets import utils_tf

import sonnet as snt
import datetime 
import pickle

import functions as fn


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    for g in gpus:
        tf.config.experimental.set_memory_growth(g, True)
  except RuntimeError as e:
    print(e)

mirrored_strategy = tf.distribute.MirroredStrategy()


### Hyperparameters

learning_rate = 0.0001  # alpha
gp_lambda = 10          # gradient penalty coefficient
n_critic = 5
b_1 = 0.9             # Adam arg beta1 
b_2 = 0.999               # Adam arg beta2 
epochs = 100000
batch_size = 4 # training
global_batch_size = mirrored_strategy.num_replicas_in_sync * batch_size
num_recurrent_passes = 3
z_dim = 128
kappa = 1.0

exp_lst = [3,21,24,26,30,34,44,45,46,49,69,71,77,79,86,91,95,98,99,100,106,107,111,116,118,123,134,137,145,149,150,152,160,163]


# ### Graph net


NUM_LAYERS = 1  # Hard-code number of layers in the edge/node/global models.
OUTPUT_EDGE_SIZE = 4 
OUTPUT_NODE_SIZE = 4 
OUTPUT_GLOBAL_SIZE = 32 

class MLPGraphIndependent(snt.Module):
  """GraphIndependent with MLP edge, node, and global models."""

  def __init__(self, name="MLPGraphIndependent"):
    super(MLPGraphIndependent, self).__init__(name=name)
    self._network = modules.GraphIndependent(
        edge_model_fn=lambda: snt.Sequential([
      snt.nets.MLP([OUTPUT_EDGE_SIZE] * NUM_LAYERS, activate_final=True),
  ]),
        node_model_fn=lambda: snt.Sequential([
      snt.nets.MLP([OUTPUT_NODE_SIZE] * NUM_LAYERS, activate_final=True),
  ]),
        global_model_fn=lambda: snt.Sequential([
      snt.nets.MLP([OUTPUT_GLOBAL_SIZE] * NUM_LAYERS, activate_final=True),
  ]))

  def __call__(self, inputs):
    return self._network(inputs)


class MLPGraphNetwork(snt.Module):
  """GraphNetwork with MLP edge, node, and global models."""

  def __init__(self, name="MLPGraphNetwork"):
    super(MLPGraphNetwork, self).__init__(name=name)
    self._network = modules.GraphNetwork(
        edge_model_fn=lambda: snt.Sequential([
      snt.nets.MLP([OUTPUT_EDGE_SIZE] * NUM_LAYERS, activate_final=True),
  ]),
        node_model_fn=lambda: snt.Sequential([
      snt.nets.MLP([OUTPUT_NODE_SIZE] * NUM_LAYERS, activate_final=True),
  ]),
        global_model_fn=lambda: snt.Sequential([
      snt.nets.MLP([OUTPUT_GLOBAL_SIZE] * NUM_LAYERS, activate_final=True),
  ]),
        reducer=tf.math.unsorted_segment_mean)

  def __call__(self, inputs):
    return self._network(inputs)


class EncodeProcess(snt.Module):
  """
  - An "Encoder" graph net, which independently encodes the edge, node, and
    global attributes (does not compute relations etc.).
  - A "Core" graph net, which performs N rounds of processing (message-passing)
    steps. The input to the Core is the concatenation of the Encoder's output
    and the previous output of the Core (labeled "Hidden(t)" below, where "t" is
    the processing step).
  """
  def __init__(self,
               edge_output_size=None,
               node_output_size=None,
               global_output_size=None,
               name="EncodeProcess"):
    super(EncodeProcess, self).__init__(name=name)
    self._encoder = MLPGraphIndependent()
    self._core = MLPGraphNetwork()

  def __call__(self, input_op, num_processing_steps):
    latent = self._encoder(input_op)
    latent0 = latent
    for _ in range(num_recurrent_passes):
      core_input = utils_tf.concat([latent0, latent], axis=1)
      latent = self._core(core_input)
    output_ops = latent

    return output_ops


# ### Real data

### Real data list

# Expression & pathway data
all_exp = []
all_path = []
Total_gene_list = []
globals_0, senders_0, receivers_0, edges_0 = [], [], [], []

for i in exp_lst:
    if i == 114: #no edges with the nodes that are included in the expression data exist
        continue
    e = 'Gene_expression/Mapped_gene_expression'+str(i)+'.csv'
    p = 'Pathway/Pathway'+str(i)+'.csv'
    Expression_dt = pd.read_csv(e)
    Expression_dt = np.array(Expression_dt)
    all_exp.append(Expression_dt)
    
    Pathway_data = pd.read_csv(p)
    all_path.append(Pathway_data)
    
    glo_t, sen_t, rec_t = fn.others(Pathway_data, Expression_dt)
    edg_t = fn.edges(Pathway_data)
    
    globals_0.append(glo_t)
    senders_0.append(sen_t)
    receivers_0.append(rec_t)
    edges_0.append(edg_t)

    lst = []
    for j in range(1,len(Expression_dt[1:,2])+1):
        lst.append(int(Expression_dt[j,2]))
    Total_gene_list.append(np.array(lst))
    
Total_gene = Total_gene_list[0] 
for a in range(len(Total_gene_list)):
    new = Total_gene_list[a]
    Total_gene = np.concatenate((Total_gene, new))

Total_gene = np.unique(Total_gene) # List of unique genes 
print(Total_gene.shape)


### Total gene expression values for each cell line (entrez id ascending order)

dt = 'Data/Gene_expression_rmNA.csv'
dt = pd.read_csv(dt)
index = list(dt.columns)
del index[:3]

Total_data_dict_list = []

for i in range(0,len(Total_gene)):
    for j in range(1, len(dt['ENTREZID'])):
        if (int(dt['ENTREZID'][j]) == Total_gene[i]):
            Total_data_dict_list.append(list(dt.iloc[j,3:]))
Total_data = np.array(Total_data_dict_list, dtype=np.float32)
Total_data = Total_data.T
Total_data = pd.DataFrame(Total_data, index=index)


### Total drug-CL list

sensitiveCL = 'Data/List_sensitiveCL.csv'
sensitiveCL = pd.read_csv(sensitiveCL)
resistantCL = 'Data/List_resistantCL_TP53mt.csv'
resistantCL = pd.read_csv(resistantCL)
resistantCL = resistantCL.loc[resistantCL['Z_IC50']>0.5]
s_drug_CL = sensitiveCL.iloc[:,[0,2]]
r_drug_CL = resistantCL.iloc[:,[0,2]]
idx = list(range(0,s_drug_CL.shape[0]))
total_real_dt = fn.create_data(idx, s_drug_CL, Total_data)
idx_r = list(range(0,r_drug_CL.shape[0]))
total_real_resi_dt = fn.create_data(idx_r, r_drug_CL, Total_data)


# Data for ARP-246 (p53 reactivator)

real_dt = total_real_dt[s_drug_CL['Pubchem.ID'] == 52918385,:] 
real_resi_dt = total_real_resi_dt[r_drug_CL['Pubchem.ID'] == 52918385,:]


# ### Fake data

gene_path_idx = []

for i in exp_lst:
    if i == 114: #no edges with the nodes that are included in the expression data exist
        continue
    e = 'Gene_expression/Mapped_gene_expression'+str(i)+'.csv'
    Expression_dt = pd.read_csv(e)
    gid = Expression_dt.iloc[1:,2].values
    gene_path_idx.append(np.array([np.where(g == Total_gene)[0][0] for g in gid])) #index of gid in total_gene


varnamelist = ['all_exp', 'all_path', 'Total_gene_list', 'globals_0', 'senders_0', 'receivers_0', 'edges_0',
               'Total_gene', 'Total_data', 'gene_path_idx', 's_drug_CL', 'r_drug_CL', 'real_dt', 'real_resi_dt']
savedict = {}
for vn in varnamelist:
    savedict[vn] = locals()[vn]
with open('processed_subpath.p','wb') as f:
    pickle.dump(savedict,f)
    
# converting data for graph network

def graph_net(data):
    data = tf.transpose(data)
    Data_dict_list = []

    for h in range(0, batch_size):
    
        data_dict_list = []

        for i in range(0, len(gene_path_idx)):
            nodes_0 = tf.gather(data[:,h], gene_path_idx[i])
            nodes_0 = tf.reshape(nodes_0, [-1,1])

            data_dict_0 = {
                "globals": globals_0[i].astype(np.float32),
                "nodes": nodes_0,
                "edges": edges_0[i].astype(np.float32),
                "senders": senders_0[i].astype(np.float32),
                "receivers": receivers_0[i].astype(np.float32)
            }
            data_dict_list.append(data_dict_0)

        Data_dict_list.append(utils_tf.data_dicts_to_graphs_tuple(data_dict_list)) #convert to graphs tuple
        
    return Data_dict_list


  
# ### Models

### Generator

class Generator(snt.Module):
    def __init__(self):
        super(Generator, self).__init__()
        
        with mirrored_strategy.scope():
            self._Dense = tf.keras.layers.Dense(512, activation = None, kernel_initializer='he_normal') #Linear (default)512
            self._Dense2 = tf.keras.layers.Dense(1024, activation = None, kernel_initializer='he_normal') #1024
            self._Dense_logits = tf.keras.layers.Dense(real_dt.shape[1], activation = None, kernel_initializer='he_normal') #Total_gene.shape[0]
            self._LayerNorm = tf.keras.layers.LayerNormalization()
            self._LayerNorm2 = tf.keras.layers.LayerNormalization()
            self._LReLU = tf.keras.layers.LeakyReLU(alpha=0.1)
            self._Dropout = tf.keras.layers.Dropout(.2)
            self._Dropout2 = tf.keras.layers.Dropout(.2)
        
    def __call__(self, z, istraining=True):
        
        joint = z
        
        hidden = self._Dense(joint)
        hidden = self._LayerNorm(hidden)
        hidden = self._LReLU(hidden)
        hidden = self._Dropout(hidden, training=istraining)
        
        hidden = self._Dense2(hidden)
        hidden = self._LayerNorm2(hidden)
        hidden = self._LReLU(hidden)
        hidden = self._Dropout2(hidden, training=istraining)

        output = self._Dense_logits(hidden)
        
        return output


### Discriminator

class Discriminator(snt.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        
        with mirrored_strategy.scope():
            self._Process = EncodeProcess()
            self._Dense_node = tf.keras.layers.Dense(256, activation = None, kernel_initializer='he_normal')
            self._Dense_edge = tf.keras.layers.Dense(256, activation = None, kernel_initializer='he_normal')
            self._Dense_global = tf.keras.layers.Dense(128, activation = None, kernel_initializer='he_normal')
            self._Dense2 = tf.keras.layers.Dense(512, activation = None, kernel_initializer='he_normal', kernel_regularizer=tf.keras.regularizers.l2(5e-4)) #512. 1024
            self._Dense3 = tf.keras.layers.Dense(128, activation = None, kernel_initializer='he_normal', kernel_regularizer=tf.keras.regularizers.l2(5e-4)) #512
            self._Dense_logits = tf.keras.layers.Dense (1, activation = None, kernel_initializer='he_normal')
            self._Flatten = tf.keras.layers.Flatten()
            self._LReLU = tf.keras.layers.LeakyReLU(alpha=0.1)
            self._ReLU = tf.keras.layers.ReLU()
        
    def __call__(self, input_, num_recurrent_passes):
       
        input_ = graph_net(input_)

        stacked_nodes, stacked_edges, stacked_globals = [], [], []
        for i in range(0,batch_size):
            previous_graphs = self._Process(input_[i], num_recurrent_passes)
            stacked_nodes.append(tf.math.reduce_mean(previous_graphs.nodes, axis=1))
            stacked_edges.append(tf.math.reduce_mean(previous_graphs.edges, axis=1))
            stacked_globals.append(tf.math.reduce_mean(previous_graphs.globals, axis=1))
        
        stacked_nodes = tf.stack(stacked_nodes, axis=0)
        stacked_edges = tf.stack(stacked_edges, axis=0)
        stacked_globals = tf.stack(stacked_globals, axis=0)
        
        stacked_nodes = self._Dense_node(stacked_nodes)
        stacked_edges = self._Dense_edge(stacked_edges)
        stacked_globals = self._Dense_global(stacked_globals)
        
        stacked = tf.concat([stacked_nodes, stacked_edges, stacked_globals], axis=-1)

        joint = stacked      
        hidden = joint

        hidden = self._Dense2(hidden)
        hidden = self._LReLU(hidden)
        
        hidden = self._Dense3(hidden)
        hidden = self._ReLU(hidden)
        
        output = self._Dense_logits(hidden)
        
        return output, hidden


### Encoder

class Encoder(snt.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        
        with mirrored_strategy.scope():
            self._Dense = tf.keras.layers.Dense(1024, activation = None, kernel_initializer='he_normal')
            self._Dense2 = tf.keras.layers.Dense(512, activation = None, kernel_initializer='he_normal') 
            self._Dense_logits = tf.keras.layers.Dense(z_dim, activation = None, kernel_initializer='he_normal')
            self._LReLU = tf.keras.layers.LeakyReLU(alpha=0.1)
            self._ReLU = tf.keras.layers.ReLU()
            self._Dropout = tf.keras.layers.Dropout(.2)
            self._Dropout2 = tf.keras.layers.Dropout(.2)
            
            self._LayerNorm = tf.keras.layers.LayerNormalization()
            self._LayerNorm2 = tf.keras.layers.LayerNormalization()
        
    def __call__(self, input_, istraining=True):
        
        joint = input_
        
        hidden = self._Dense(joint)
        hidden = self._LayerNorm(hidden)
        hidden = self._LReLU(hidden)
        hidden = self._Dropout(hidden)
        
        hidden = self._Dense2(hidden)
        hidden = self._LayerNorm2(hidden)
        hidden = self._LReLU(hidden)
        hidden = self._Dropout2(hidden)
        
        output = self._Dense_logits(hidden)
        
        return output


      
# ### Generator & discriminator training


### Optimizers

with mirrored_strategy.scope():
    generator_optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1 = b_1, beta_2 = b_2)
    discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1 = b_1, beta_2 = b_2)


### Discriminator training

G = Generator()
D = Discriminator()
@tf.function
def discriminator_train_step(dt):
    def step_fn(inputs):
        input_ = inputs
        len_batch = len(input_) # for the last batch that has different length

        with tf.GradientTape() as disc_tape:
            z = tf.random.normal((input_.shape[0], z_dim))
            generated_data = G(z)

            real_output, _ = D(input_, num_recurrent_passes)
            fake_output, _ = D(generated_data, num_recurrent_passes)

            #wgan loss
            disc_loss = K.mean(fake_output) - K.mean(real_output)

            eps = tf.random.uniform(shape=[len_batch, 1])
            x_hat = eps*input_ + (1 - eps)*generated_data

            with tf.GradientTape() as t:
                t.watch(x_hat)
                d_hat, _ = D(x_hat, num_recurrent_passes)

            gradients = t.gradient(d_hat, [x_hat])[0]  # gradients computation
            slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1]))
            gradient_penalty = tf.reduce_mean(tf.square((slopes-1.)))
            disc_loss += gp_lambda*gradient_penalty
            loss = tf.reduce_sum(disc_loss) * (1./global_batch_size)

        gradients_of_discriminator = disc_tape.gradient(loss, D.trainable_variables)
        discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, D.trainable_variables))
        
        return disc_loss
    
    per_replica_losses = mirrored_strategy.run(step_fn, args=(dt,))
    mean_loss = mirrored_strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_losses, axis=None)
    
    return mean_loss


### Generator training

@tf.function
def generator_train_step(dt):
    def step_fn(inputs):
        input_ = inputs
        with tf.GradientTape() as gen_tape:
            z = tf.random.normal((input_.shape[0], z_dim))
            generated_data = G(z)
            fake_output, _ = D(generated_data, num_recurrent_passes)

            #wgan loss
            gen_loss = - K.mean(fake_output)
            loss = tf.reduce_sum(gen_loss) * (1./global_batch_size)

        gradients_of_generator = gen_tape.gradient(loss, G.trainable_variables)
        generator_optimizer.apply_gradients(zip(gradients_of_generator, G.trainable_variables)) 
        return gen_loss
    
    per_replica_losses = mirrored_strategy.run(step_fn, args=(dt,))
    mean_loss = mirrored_strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_losses, axis=None)
    
    return mean_loss


### Training

base_dir = 'logs/'
log_dir = base_dir+'GAN_subpath_apr246' 
summary_writer = tf.summary.create_file_writer(logdir=log_dir)
ckptG = tf.train.Checkpoint(model=G)
ckptD = tf.train.Checkpoint(model=D)
ckptG_manager = tf.train.CheckpointManager(ckptG, log_dir+'/G', max_to_keep=5)
ckptD_manager = tf.train.CheckpointManager(ckptD, log_dir+'/D', max_to_keep=5)
def train(dataset, epochs):
    for epoch in range(epochs):
        start = time.time()
        gen_loss_list = []
        disc_loss_list = []
        
        with mirrored_strategy.scope():
            for dt in dataset:
                WGANGP_loss_d_ = 0
                for i in range(n_critic):
                    WGANGP_loss_d_ += discriminator_train_step(dt)

                gen_loss_list.append(generator_train_step(dt))
                disc_loss_list.append(WGANGP_loss_d_ / n_critic)
   
        WGANGP_loss_g = np.mean(gen_loss_list)
        WGANGP_loss_d = np.mean(disc_loss_list)

        # loss & time printing

        with summary_writer.as_default():
            tf.summary.scalar('WGANGP_loss_g', WGANGP_loss_g, step=epoch)
            tf.summary.scalar('WGANGP_loss_d', WGANGP_loss_d, step=epoch)
            
        if epoch % 5 == 0:
            ckptG_manager.save()
            ckptD_manager.save()
            
        print('Time for step {} is {} sec'.format(epoch + 1, time.time()-start))
        print('G_Loss is {}, D_Loss is {}'.format(WGANGP_loss_g,WGANGP_loss_d))

        if epoch % 20 == 0:
            fn.umapplot(real_dt, real_resi_dt, c, epoch)
    plt.show()


get_ipython().run_cell_magic('time', '', 'with mirrored_strategy.scope():\n    dataset = tf.data.Dataset.from_tensor_slices(real_dt).shuffle(real_dt.shape[0]).batch(global_batch_size, drop_remainder=True)\n    dist_dataset = mirrored_strategy.experimental_distribute_dataset(dataset)\n    train(dist_dataset, epochs)')


### Encoder optimizer

with mirrored_strategy.scope():
    optimizer = tf.keras.optimizers.RMSprop(learning_rate)


### Encoder training

kappa = 0.01
@tf.function
def encoder_train_step(dt):
    def step_fn(inputs):
        x_data = inputs
        with tf.GradientTape() as tape:
            z = E(x_data, istraining=True)
            generated_data = G(z, istraining=False)
            _, fake_output = D(generated_data, num_recurrent_passes)
            _, real_output = D(x_data, num_recurrent_passes)

            loss_data = tf.reduce_mean(tf.pow(x_data-generated_data, 2))
            loss_features = tf.reduce_mean(tf.pow(real_output-fake_output, 2))
            loss = loss_data + (kappa * loss_features)
            l = loss * (1./global_batch_size)

        gradients = tape.gradient(l, E.trainable_variables)
        optimizer.apply_gradients(zip(gradients, E.trainable_variables)) 
        return loss, loss_data, loss_features
    
    per_replica_losses, l_dt, l_ft = mirrored_strategy.run(step_fn, args=(dt,))
    mean_loss = mirrored_strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_losses, axis=None)
    mean_l_dt = mirrored_strategy.reduce(tf.distribute.ReduceOp.MEAN, l_dt, axis=None)
    mean_l_ft = mirrored_strategy.reduce(tf.distribute.ReduceOp.MEAN, l_ft, axis=None)
    
    return mean_loss, mean_l_dt, mean_l_ft


def anomaly_scoring(dt):
    gen, fake_hid, real_hid = [], [], []
    for shard in range(dt.shape[0] // batch_size):
        current_d = dt[shard*batch_size:(shard+1)*batch_size]
        z = E(current_d, istraining=False)
        generated_data = G(z, istraining=False)
        _, fake_output = D(generated_data, num_recurrent_passes)
        _, real_output = D(current_d, num_recurrent_passes)
        
        gen.append(generated_data.numpy())
        fake_hid.append(fake_output.numpy())
        real_hid.append(real_output.numpy())
    
    gen = np.concatenate(gen, axis=0)
    fake_hid = np.concatenate(fake_hid, axis=0)
    real_hid = np.concatenate(real_hid, axis=0)
    
    loss_data = tf.reduce_mean(tf.pow(dt-gen, 2))
    loss_features = tf.reduce_mean(tf.pow(real_hid-fake_hid, 2))
    sample_score = loss_data  #Sample-level score
    
    gene_score = abs(dt - gen)
    
    return sample_score, gene_score


E = Encoder()
ckptE = tf.train.Checkpoint(model=E)
ckpt_manager = tf.train.CheckpointManager(ckptE, log_dir+'/E/kappa', max_to_keep=5)


### training

def e_train(dataset, epochs):
    for epoch in range(epochs+1):
        start = time.time()
        mapping_losses = []
        mapping_losses_dt = []
        mapping_losses_fts = []
        
        with mirrored_strategy.scope():
            for dt in dataset:
                loss, loss_data, loss_features = encoder_train_step(dt)
                mapping_losses.append(loss)
                mapping_losses_dt.append(loss_data)
                mapping_losses_fts.append(loss_features)
        
        mloss = np.mean(mapping_losses)
        mloss_dt = np.mean(mapping_losses_dt)
        mloss_fts = np.mean(mapping_losses_fts)
        
        with summary_writer.as_default():
            tf.summary.scalar('loss/Encoder_loss', mloss, step=epoch)
            tf.summary.scalar('loss/Encoder_loss_dt', mloss_dt, step=epoch)
            tf.summary.scalar('loss/Encoder_loss_fts', mloss_fts, step=epoch)
        
        if epoch % 20 == 0:
            if epoch % 10 == 0:
                ckpt_manager.save()
            idx = np.random.choice(real_dt.shape[0], batch_size*70, replace=False)
            s = real_dt[idx]
            sa, _ = anomaly_scoring(s)
            idx = np.random.choice(real_resi_dt.shape[0], batch_size*60, replace=False)
            r = real_resi_dt[idx]
            ra, _ = anomaly_scoring(r)
            print('ano score sen:', sa.numpy(), ' res:',ra.numpy())
            with summary_writer.as_default():
                tf.summary.scalar('score/sens_ascore', sa, step=epoch)
                tf.summary.scalar('score/resi_ascore', ra, step=epoch)
        
        print('Time for step {} is {} sec'.format(epoch + 1, time.time()-start))
        print('Encoder loss is {}'.format(mloss))

get_ipython().run_cell_magic('time', '', 'with mirrored_strategy.scope():\n    dataset = tf.data.Dataset.from_tensor_slices(real_dt).shuffle(real_dt.shape[0]).batch(global_batch_size, drop_remainder=True)\n    dist_dataset = mirrored_strategy.experimental_distribute_dataset(dataset)\n    e_train(dist_dataset, epochs)')


# ### Anomaly scoring

### Anomaly detection

batch_size = 1
def anomaly_scoring(dt):
    sample_score, sample_score2, gene_score = [], [], []
    
    for shard in range(dt.shape[0] // batch_size):
        if ((shard % 100) == 0):
            print(shard, '/', dt.shape[0] // batch_size)
        current_d = dt[shard*batch_size:(shard+1)*batch_size]
        z = E(current_d, istraining=False)
        generated_data = G(z, istraining=False)
        _, fake_output = D(generated_data, num_recurrent_passes)
        _, real_output = D(current_d, num_recurrent_passes)
        
        loss_data = tf.reduce_mean(tf.pow(current_d-generated_data, 2), axis=-1)
        loss_features = tf.reduce_mean(tf.pow(real_output-fake_output, 2), axis=-1)
    
        sample_score.append(loss_data.numpy()) #Sample-level score
        sample_score2.append(loss_features.numpy()) #Sample-level score
        gene_score.append(abs(current_d - generated_data.numpy()))
    
    sample_score = np.concatenate(sample_score)
    sample_score2 = np.concatenate(sample_score2)
    
    return sample_score, sample_score2, gene_score


np.set_printoptions(precision=6, suppress=True)
sen_score, sen_score2, sen_gene = anomaly_scoring(real_dt)
res_score, res_score2, res_gene = anomaly_scoring(real_resi_dt)


varnamelist = ['sen_score', 'sen_score2', 'sen_gene', 'res_score', 'res_score2', 'res_gene']
savedict = {}
for vn in varnamelist:
    savedict[vn] = locals()[vn]
with open('processed_subpath_scores.p','wb') as f:
    pickle.dump(savedict,f)

    
    
