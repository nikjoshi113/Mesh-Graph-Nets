#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Title:		  
    Version:	  2.0
    Institute: 	Digital Additive Production, RWTH Aachen University
    Authors: 	  Jan Theunissen and Nikhil Joshi
    Contact: 	  jan.theunissen@dap.rwth-aachen.de and nikhil.joshi@rwth-aachen.de
"""

import enum
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pyvista as pv
from tqdm import tqdm, trange

from helpers import query_yes_no


class NodeType(enum.IntEnum):
  """
  Define the code for the one-hot vector representing the node types.
  """
  NORMAL = 0
  BOTTOM = 1
  BUILDPLATE_BOUNDARY = 2
  ZERO = 3
  PART_BOUNDARY = 4
  TOP_LAYER = 5
  NOT_IN_USE = 9
  NEXT_LAYER = 1



DATASET_KEYS_AND_PADDING_VALUE = {
  "cells": np.zeros((3,0), np.int32), # this array will overwrite with the data from the highest dimension
  "mesh_pos": np.zeros((2,1), np.float32), # this array will overwrite with the data from the highest dimension
  "node_type": int(NodeType.NORMAL),
  "time": -1.0, # this vector contains only one value -> padding with actual value
  "layer": 0, # this vector contains only one value -> padding with actual value
  "laser": 0,
  "heat_conductivity": 0,
  "temperature": 25.0,
}

class VTKtoH5():
  """
  """
  def __init__(self, settings):
    """
    Args:
        settings (_type_): _description_
        path (_type_): _description_
    """
    self.settings = settings
    self.allocate_values()
    
    
  def allocate_values(self):      
    self.dataset = {key: [] for key in DATASET_KEYS_AND_PADDING_VALUE.keys()}
    self.pointdata = {key: [] for key in DATASET_KEYS_AND_PADDING_VALUE.keys()}
    self.max_dim = {key: {"index": 0 , "maxdim": 0} for key in DATASET_KEYS_AND_PADDING_VALUE.keys()}
    self.boundary_indicies = {}
    self.boundary_points = {}
    self.pad = {key: [] for key in DATASET_KEYS_AND_PADDING_VALUE.keys()}
    
  
  def _load_vtk(self, path):
    """
    """
    self.vtk_path = path
    self.mesh_celldata = pv.read(path)
    self.mesh_pointdata  = self.mesh_celldata.cell_data_to_point_data()
    self.mesh_feature_edges = pv.PolyData(self.mesh_pointdata.extract_feature_edges(20).points)
    
  def _get_boundary_boundary_indicies_from_mesh_points(self): 
    """ Use only boundary nodes with feature extraction from original mesh set 
    """
     # ------------------------------------------------------------
    msh = self.mesh_feature_edges
    x, y, z = msh.points[:, 0], msh.points[:, 1], msh.points[:, 2]
    node_coordinates = np.vstack([x, y, z]).T
    mn, mx = np.min(node_coordinates, axis=0), np.max(node_coordinates, axis=0)
    
     # ------------------------------------------------------------
    ## get node boundary_indicies for every area of the mesh
    self.boundary_indicies["BOUNDARY"] = np.where(node_coordinates[:, 1] > -1000)[0]
    self.boundary_indicies["BUILDPLATE_BOUNDARY"] = np.where(node_coordinates[:, 1] <= 0)[0]
    self.boundary_indicies["BOTTOM"] = np.where(node_coordinates[:, 1] == mn[1])[0]
    self.boundary_indicies["ZERO"] = np.where(node_coordinates[:, 1] == 0)[0]
    self.boundary_indicies["BUILDPLATE_BOUNDARY_SIDE"] = np.setdiff1d(self.boundary_indicies["BUILDPLATE_BOUNDARY"], 
                                              np.concatenate((self.boundary_indicies["BOTTOM"], self.boundary_indicies["ZERO"])))  
    self.boundary_indicies["PART_BOUNDARY"] = np.where(node_coordinates[:, 1] > 0)[0]
    self.boundary_indicies["TOP_LAYER"] = np.where(node_coordinates[:, 1] == mx[1])[0]
    self.boundary_indicies["PART_BOUNDARY"] = np.setdiff1d(self.boundary_indicies["PART_BOUNDARY"], self.boundary_indicies["TOP_LAYER"]) 
        
    ## get node points 
    self.boundary_points = {
        key: np.array(np.take(self.mesh_feature_edges.points, self.boundary_indicies[key], axis=0)
                    ) for key in list(self.boundary_indicies.keys())}

     # ------------------------------------------------------------
    if self.settings["Verbose"] > 1:
      tqdm.write("Boundary mesh nodes [X Y Z] | Min: {} | Max: {}".format(mn, mx))
      for key in list(self.boundary_indicies.keys()):
          tqdm.write("Key : {:20} \t|  Points : {} \t|  boundary_indicies : {}".format(
              key, np.shape(self.boundary_points[key]), np.shape(self.boundary_indicies[key])))

  def _set_boundary_labels(self):
    """
    """
    msh = self.mesh_feature_edges
    x, y, z = msh.points[:, 0], msh.points[:, 1], msh.points[:, 2]
    node_coordinates = np.vstack([x, y, z]).T
    mx = np.min(node_coordinates, axis=0) #,mn = np.max(node_coordinates, axis=0)
    
    self.boundary_labels = np.zeros(self.mesh_pointdata.n_points)
    self.labels_laser = np.zeros(self.mesh_pointdata.n_points)
    
    np.put(self.boundary_labels, np.argwhere(node_coordinates[:, 1] <= 0)[0], NodeType.BUILDPLATE_BOUNDARY)
    np.put(self.labels_laser, np.argwhere(node_coordinates[:, 1]) == mx[1], 1)
    
    for key in NodeType:
      if key.name in self.boundary_indicies:
        a = [np.argwhere((f == self.mesh_pointdata.points).all(1))[0][0] for f in self.boundary_points[key.name]]
        [np.put(self.boundary_labels, i, key.value) for i in a]
        
        if self.settings["Verbose"] > 1:
          unique, counts = np.unique(self.boundary_labels, return_counts=True)
          tqdm.write("{} {} {}".format(key.name, key.value, dict(zip(unique, counts))))

  def _set_pointdata(self):
    """
    """
    #t = int(self.vtk_path[self.vtk_path.rfind("_z_")+7:self.vtk_path.rfind(".vtu")])
    #z = float(self.vtk_path[self.vtk_path.rfind("_z_")+3:self.vtk_path.rfind("_"+str(t)+".vtu")])
    
    npoints = len(self.mesh_pointdata.point_data["Temperature"])

    self.pointdata = {
      "cells": np.array(self.mesh_pointdata.cells_dict[5], np.int32),
      "mesh_pos": np.array(self.mesh_pointdata.points, np.float32),
      "node_type": np.array(self.boundary_labels, np.int32),
      "time": np.array(self.mesh_pointdata.point_data["Time"], np.float32),
      "layer": np.array(self.mesh_pointdata.point_data["Layer"], np.int32),
      #"laser": np.array(self.boundary_labels, np.int32), #np.zeros(len(self.mesh_pointdata.point_data["layer"]), np.int32),
      "heat_conductivity": np.ones(npoints, np.int32),
      "temperature": np.array(self.mesh_pointdata.point_data["Temperature"], np.int32).round(2)
    }

  def padding_and_reformat_dataset(self):
    """
    """
     # ------------------------------------------------------------
    # overwrite "cells" and "mesh_pos"
    # -> only possible with datasets using the same mesh
    for key in ["cells", "mesh_pos"]:
      arr = [self.dataset[key][self.max_dim[key]["index"]][0] for _ in range(len(self.dataset[key]))]
      self.pad[key] = np.stack(arr, axis=0)
      #self.dataset[key] = self.pad[key]  
    # ------------------------------------------------------------
    # padding datasets with fixed value  
    for key in ["temperature","time","layer"]:  #
      arr = [self.dataset[key][self.max_dim[key]["index"]][0] for _ in range(len(self.dataset[key]))]
      arr = [np.expand_dims(f, axis=1) for f in arr]
      self.pad[key] = np.stack(arr, axis=0)
      
      vr = []
      #print(self.pad[key][0][0])
      for i in range(len(self.dataset[key])):   # i: time step
        ind = (np.array(self.pad["mesh_pos"][i])[:, None] == np.array(self.dataset["mesh_pos"][i])).all(-1).any(-1)
        np.set_printoptions(threshold=np.inf)
        #print(self.dataset[key][i][0][0].shape)
        #break
        ar = np.full(shape = (len(self.pad["mesh_pos"][0]),1),fill_value = DATASET_KEYS_AND_PADDING_VALUE[key],dtype=np.int32)
        id = np.array((ind==True).nonzero())[0]
        
        #r_id = id
        for k in range(len(id)):       
          #if (self.pad["mesh_pos"][i][id[k]][:] == self.dataset["mesh_pos"][i][0][k][:]).all():
          ar[id[k],0] = self.dataset[key][i][0][k]
            #r_id = np.delete(r_id, np.where(r_id == id[k]))
        #print(r_id.shape)
        vr.append(ar)
      self.dataset[key] = np.stack(vr, axis=0)
      
    arr = None
    vr = []

    for key in ["node_type","heat_conductivity"]:  #
      for i in range(len(self.dataset[key])):
        ar = np.full(shape = (len(self.pad["mesh_pos"][0]),1),fill_value = DATASET_KEYS_AND_PADDING_VALUE[key],dtype=np.int32)
        y_coord = np.array(self.dataset["mesh_pos"][i][0])[:,1]
        mx = np.max(y_coord, axis=0)
        y_pad = np.array(self.pad["mesh_pos"][i])[:,1]
        if key=='node_type':
          ind = ((y_pad>=mx) & (y_pad<=(mx+0.5)))             #.all(-1).any(-1)     o.5 layer height 
          ar = np.where(ind, NodeType.NEXT_LAYER , DATASET_KEYS_AND_PADDING_VALUE[key])
        else:
          indx = (y_pad<=mx)
          ar = np.where(indx, 1 , DATASET_KEYS_AND_PADDING_VALUE[key]) 
        vr.append(ar)
      self.dataset[key] = np.stack(vr, axis=0)
      
    #  
    # ------------------------------------------------------------
    # set laser values = 1 where node_type = top
    self.dataset["cells"] = self.pad["cells"]
    self.dataset["mesh_pos"] = self.pad["mesh_pos"]
    self.dataset["laser"] = np.where(self.dataset["laser"] == NodeType.TOP_LAYER, 1, 0)
    
    # overwrite nodes with last time step -> graph has constant node types per time step
    self.dataset["node_type"] = [self.dataset["node_type"][-1] for i in range(len(self.dataset[key]))]
    #self.dataset = self.pad
    
    self.pad = None
   
  def create_dataset(self, source_files):   
    """
    Args:
        source_files (_type_): _description_
    """
    # ------------------------------------------------------------
    # loop over alls files and add to self.dataset dict list
    for i in trange(np.shape(source_files)[0], leave=False):
      # load vtk
      self._load_vtk(path=source_files[i])
      
      # extract feature shape as point cloud
      self._get_boundary_boundary_indicies_from_mesh_points()
      
      # set boundary labels based on boundary point cloud
      self._set_boundary_labels()
      
      # create an pointdata dict containing all neccessary data relatet to this vtk
      self._set_pointdata()  
      
      for key in self.pointdata:
        # append point data to dataset list list
        #print("pointdata: ",len(self.pointdata[key]))
        self.dataset[key].append([self.pointdata[key]])
        # extract max dim and save with index
        if (self.pointdata[key].shape[0] > self.max_dim[key]["maxdim"]):
          self.max_dim[key] = {
            "index": len(self.dataset[key])-1,
            "maxdim": self.pointdata[key].shape[0]
          }
        #print("dataset: ",len(self.dataset[key]))
    
    # ------------------------------------------------------------
    # pad the data to the maximum shape
    # and convert self.dataset list to 3 dim array
    #print("before padding dataset: ",np.shape(self.dataset[key]))
    self.padding_and_reformat_dataset()
    #print("after padding dataset: ",np.shape(self.dataset[key]))    
    # ------------------------------------------------------------     
    if self.settings["Verbose"] > 0:
      for key in list(self.dataset.keys()):
        tqdm.write("Key : {:20} \t|  Shape : {} ".format(
            key, np.shape(self.dataset[key])))
        
    

  def create_empty_h5(self, path, name):
    """

    Args:
        path (_type_): _description_
        name (_type_): _description_
    """
    # ------------------------------------------------------------
    self.file = str(str(path) + "\\" + name + ".h5")
    delete, create = False, True
    
    # ------------------------------------------------------------
    if Path(self.file).is_file(): 
      delete, create = False, False     
      if self.settings["Overwrite"] == 1: 
        delete, create = True, True 
      elif query_yes_no("h5 file found. Overwrite?"): 
        delete, create = False, False
      else:
        delete, create = True, True 
    
    # ------------------------------------------------------------
    if delete:
      try:
        Path(self.file).unlink()
        Path(str(str(path) + "\\" + name)).unlink()
      except:
        pass
    if create:  
      hf = h5py.File(self.file, "w")
      hf.close()
      if self.settings["Verbose"] > 1:
        tqdm.write("Created h5 file: {}".format(self.file))
    
    
  def write_dataset_h5(self, path, name, number, reopen=False):
    """
    Args:
        path (_type_): _description_
        iter (_type_): _description_
    """
    # ------------------------------------------------------------
    # writing data
    hf = h5py.File(self.file, "a")
    dict_group = hf.create_group('/dataset_{}'.format(number))
    
    for k, v in self.dataset.items():
        dict_group[k] = v
        
    hf.close()
    # ------------------------------------------------------------
    # Reading data
    if reopen:
      hf1 = h5py.File(self.file, "r")
      for name in hf1.keys():
          tqdm.write(hf1[name])

      tqdm.write(hf1.attrs.keys())
      hf1.close()

  
  def plot(self, mesh, type=["pointdata", "celldata"], scalar_name=""):
    """
    Args:
        mesh (_type_): _description_
        type (list, optional): _description_. Defaults to ["pointdata", "celldata"].
        scalar_name (str, optional): _description_. Defaults to "".
    """

    if mesh is not None:
      msh = mesh
      if type == "pointdata":
        msh.point_data[scalar_name] = self.data[scalar_name]
      else:
        msh.cell_data[scalar_name] = self.data[scalar_name]
      msh.active_scalars_name = scalar_name

      c_map=['white', 'blue', 'green', 'yellow', "red"]
      
      p = pv.Plotter()
      p.add_mesh(msh, show_edges=True, line_width=.1,cmap=c_map)
      p.camera.zoom(10.5)
      p.camera_position = 'xy'
      p.show()





