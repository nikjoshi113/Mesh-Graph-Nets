# %% [markdown]
# # Acessing Pre-Processed Datasets

# %%
FOLDERNAME = r"C:\Users\nikjo\exercise_cie\Mesh Refinment in PBF process\meshgraphnets_heatmodel-lpbf"
# %%
import torch
import random
import pandas as pd
import torch_scatter
import torch.nn as nn
from torch.nn import Linear, Sequential, LayerNorm, ReLU
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.loader import DataLoader
import os
import numpy as np
import time
import torch.optim as optim
from tqdm import trange
import pandas as pd
import copy
import matplotlib.pyplot as plt

print("PyTorch has version {}".format(torch.__version__))

# %% [markdown]
# # Preparing and Loading the Dataset

# %%
root_dir = os.path.join(FOLDERNAME)
dataset_dir = os.path.join(root_dir, 'datasets')
checkpoint_dir = os.path.join(root_dir, 'models')
postprocess_dir = os.path.join(root_dir, 'evaluation')

# %%
print("dataset_dir {}".format(dataset_dir))

# %% [markdown]
# **Data Pre-Processing Code**

# %%
import os
import numpy as np
import torch
import h5py
import tensorflow as tf
import functools
import json
from torch_geometric.data import Data
import enum

# %% [markdown]
# #Utility functions 
# 
# Here we define the functions that are needed for assisting in data processing.
# 
# triangle_to_edges:  decomposes 2D triangular meshes to edges and returns the undirected graph nodes. 
# 
# NodeType: is subclass of enum with unique and unchanging integer valued attributes over instances in order to make sure values are unchanged
#from prepare_config import prepare_config
#CONFIG = prepare_config()

# %%
#Utility functions, provided in the release of the code from the original MeshGraphNets study:
#https://github.com/deepmind/deepmind-research/tree/master/meshgraphnets

def triangles_to_edges(faces):
  """Computes mesh edges from triangles.
     Note that this triangles_to_edges method was provided as part of the
     code release for the MeshGraphNets paper by DeepMind, available here:
     https://github.com/deepmind/deepmind-research/tree/master/meshgraphnets
  """
  # collect edges from triangles
  edges = tf.concat([faces[:, 0:2],
                     faces[:, 1:3],
                     tf.stack([faces[:, 2], faces[:, 0]], axis=1)], axis=0)
  # those edges are sometimes duplicated (within the mesh) and sometimes
  # single (at the mesh boundary).
  # sort & pack edges as single tf.int64
  receivers = tf.reduce_min(edges, axis=1)
  senders = tf.reduce_max(edges, axis=1)
  packed_edges = tf.bitcast(tf.stack([senders, receivers], axis=1), tf.int64)
  # remove duplicates and unpack
#   print(packed_edges.shape)
  unique_edges = tf.bitcast(tf.unique(packed_edges)[0], tf.int32)
  senders, receivers = tf.unstack(unique_edges, axis=1)
  # create two-way connectivity
  return (tf.concat([senders, receivers], axis=0),
          tf.concat([receivers, senders], axis=0))



class NodeType(enum.IntEnum):
    """
    Define the code for the one-hot vector representing the node types.
    Note that this is consistent with the codes provided in the original
    MeshGraphNets study: 
    https://github.com/deepmind/deepmind-research/tree/master/meshgraphnets
    """
    NORMAL = 0
    BOTTOM = 1
    BUILDPLATE_BOUNDARY = 2
    ZERO = 3
    PART_BOUNDARY = 4
    TOP_LAYER = 5
    SIZE = 6

# %%
from helpers import (Struct, PyJSON)

with open("config\config_additve.json", "r") as f: #C:\Users\nikjo\exercise_cie\Mesh Refinment in PBF process\meshgraphnets_heatmodel-lpbf\
    # CONFIG = Struct(json.load(f))
    CONFIG = PyJSON(json.load(f))
f.close()


# %%
#Define the data folder and data file name
datafile = os.path.join(dataset_dir + '/dataset_cone_2.h5')
data = h5py.File(datafile, 'r')

#Define the list that will return the data graphs
data_list = []
cells_list = []

#define the time difference between the graphs
dt=2 #0.01   #A constant: do not change!

#define the number of trajectories and time steps within each to process.
#note that here we only include 2 of each for a toy example.
number_trajectories = CONFIG.dataset.specs.number_trajectories
number_ts = CONFIG.dataset.specs.number_timesteps

#name of the dataset
PREPROCESSED_DATASET = "mgn_dataset_cone_2_{}traj_{}ts_vis.pt".format(number_trajectories,number_ts) #meshgraphnets_miniset30traj5ts_vis

# %%
print(data.keys(), str("Len: {}".format(len(data.keys()))))
print(data['dataset_0'].keys(), str("Len: {}".format(len(data['dataset_0'].keys()))))
print(data['dataset_0']["cells"][0])
print(data['dataset_0']["cells"], str("Len: {}".format(len(data['dataset_0']["cells"]))))
#print(data['dataset_0']["cells"][0][7209][2])

# %%
# Inspect data shapes
for item in data['dataset_0'].keys():
  print('{} : {}'.format(item, data['dataset_0'][item].shape))
print(np.unique( data['dataset_0']["laser"]))
print(np.unique( data['dataset_0']["layer"]))
print(np.unique( data['dataset_0']["heat_conductivity"]))
print(np.unique( data['dataset_0']["node_type"]))

# %%
ioffset = 0
ff = dataset_dir+'/'+PREPROCESSED_DATASET
if os.path.exists(ff): os.remove(ff)

with h5py.File(datafile, 'r') as data:

    for i,trajectory in enumerate(data.keys()):  
      i += ioffset

      if(i==(number_trajectories+ioffset)):   #condition for no. of trajectories
        break

      print("Trajectory: ",i)

      #We iterate over all the time steps to produce an example graph except
      #for the last one, which does not have a following time step to produce
      #node output values
      for ts in range(0,len(data[trajectory]['temperature'])-1):   
          
          if(ts==number_ts):      #condition for no. of timesteps
              break

          #Get node features

          #Note that it's faster to convert to numpy then to torch than to
          #import to torch from h5 format directly
          temperature = torch.tensor(np.array(data[trajectory]['temperature'][ts]))  #ts or ts-1??
          #laser = torch.tensor(np.array(tf.one_hot(tf.convert_to_tensor(data[trajectory]['laser'][ts+1]), 2, on_value=1, off_value=0))).squeeze(1)
          layer = torch.tensor(np.array(data[trajectory]['layer'][ts+1]))
          #heat_conductivity = torch.tensor(np.array(tf.one_hot(tf.convert_to_tensor(data[trajectory]['heat_conductivity'][ts]), 2, on_value=1, off_value=0))).squeeze(1)
          #heat_conductivity =  torch.tensor(np.array(tf.one_hot(tf.convert_to_tensor(data[trajectory]['heat_conductivity'][ts]), NodeType.SIZE))).squeeze(1)
          #node_type = torch.tensor(np.array(data[trajectory]['node_type'][ts]))
          node_type = torch.tensor(np.array(tf.one_hot(tf.convert_to_tensor(data[trajectory]['node_type'][ts]), NodeType.SIZE))).squeeze(1)
          x = torch.cat((temperature,layer,node_type),dim=-1).type(torch.float)  #

          #Get edge indices in COO format
          edges = triangles_to_edges(tf.convert_to_tensor(np.array(data[trajectory]['cells'][ts])))

          edge_index = torch.cat( (torch.tensor(edges[0].numpy()).unsqueeze(0) ,
                        torch.tensor(edges[1].numpy()).unsqueeze(0)), dim=0).type(torch.long)

          #Get edge features
          u_i = torch.tensor(np.array(data[trajectory]['mesh_pos'][ts]))[edge_index[0]]
          u_j = torch.tensor(np.array(data[trajectory]['mesh_pos'][ts]))[edge_index[1]]
          u_ij= u_i - u_j
          u_ij_norm = torch.norm(u_ij,p=2,dim=1,keepdim=True)
          edge_attr = torch.cat((u_ij,u_ij_norm),dim=-1).type(torch.float)

          #Node outputs, for training (temperature)
          x_0 = torch.tensor(np.array(data[trajectory]['temperature'][ts]))
          x_1 = torch.tensor(np.array(data[trajectory]['temperature'][ts+1]))
          t_0 = torch.tensor(np.array(data[trajectory]['time'][ts]))
          t_1 = torch.tensor(np.array(data[trajectory]['time'][ts+1]))
          dt = 1 #t_1-t_0 #(t1-t0)    #dt disabled during error 0/0
          y = ((x_1-x_0)/dt).type(torch.float)
          #print(y)
          # print(y.min(),y.max())
          #Node outputs, for testing integrator (temperature)
          #p = torch.tensor(np.array(data[trajectory]['temperature'][ts]))

          #Data needed for visualization code
          cells = torch.tensor(np.array(data[trajectory]['cells'][ts]))
          mesh_pos = torch.tensor(np.array(data[trajectory]['mesh_pos'][ts]))

          data_list.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr,y=y,dt=dt,cells=cells,mesh_pos=mesh_pos))
          #cells_list.append(Data(cells=cells,mesh_pos=mesh_pos))



print("Done collecting data!")

#os.path.join(data_folder + '/test.h5')
# torch.save(data_list,os.path.join(data_folder + '/test_processed_set.pt'))


torch.save(data_list,ff)

print("Done saving data!")
print("Output Location: ", dataset_dir+'/'+PREPROCESSED_DATASET)

# %% [markdown]
# **Loading a Pre-Processed Dataset**
# 
# ***NOTE*** Run the cell below if you would like to use a pre-processed dataset and test that loading works. If not, follow the instructions in the first cell at the top of this Colab.

# %%
file_path=os.path.join(dataset_dir, PREPROCESSED_DATASET)
dataset_full_timesteps = torch.load(file_path)
dataset = torch.load(file_path)[:1]

print(dataset)

# %%
len(dataset_full_timesteps)/number_ts


