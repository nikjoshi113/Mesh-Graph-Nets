#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Title:		create a .h5 files for an existing folder with data files.
    Version:	2.0
    Institute: 	Digital Additive Production, RWTH Aachen University
    Authors: 	Jan Theunissen and Nikhil Joshi
    Contact: 	jan.theunissen@dap.rwth-aachen.de, nikhil.joshi@rwth-aachen.de
"""
import pandas as pd
from pathlib import Path
from VTKtoH5_v3 import NodeType, VTKtoH5
from tqdm import tqdm, trange


# ---------------------------------------------------------------------------
# -- SETTINGS -------------------------------------------------------------------
SETTINGS = {
    "Verbose": 0,
    "Overwrite": 1
}

DATASET = {
    "source_datafolder": "src_datasets",
    "target_datafolder": "datasets",
    "dataset_name": "dataset_cone_2",
    "source_file_extension": ".vtk"
}

PATH = {
    "script_directory": Path(__file__).parent.absolute(),
    "folder_directory": Path(__file__).parent.absolute(),
    "source_directory": Path(Path(__file__).parent.absolute() / DATASET["source_datafolder"] / DATASET["dataset_name"]).absolute(),
    "source_filename": "",
    "source_file_extension": DATASET["source_file_extension"],
    "target_directory": Path(Path(__file__).parent.absolute() / DATASET["target_datafolder"]),
    "target_filename": "",
    "target_file_extension": "",
}


# ---------------------------------------------------------------------------
# -- MAIN -------------------------------------------------------------------

def _scandir(
    source_datafolder_path=str(""),
    datafile_extension=str("*"),
    rename_target_datafolder=["msh\dataset", "fem\dataset"]):
    """find recursively every file along the folder structure with given file extension
    
    Args:
        source_datafolder_path (_type_, optional): _description_. Defaults to str("").
        datafile_extension (_type_, optional): _description_. Defaults to str("*").
        rename_target_datafolder (list, optional): _description_. Defaults to ["msh\dataset", "fem\dataset"].

    Returns:
        source_folder (list): list with every dataset path
        source_files (list): array of string file for each dataset, already sorted
        target_folder (list): list with creted path name to save final dataset 
    """
    
    if Path(source_datafolder_path).exists():
        source_folder = [f for f in Path(source_datafolder_path).iterdir()]

        source_files = []
        for ff in source_folder:
            # get all data files in folder and sort it
            lst = [sorted((f.absolute() for f in Path(ff).rglob('*{}*'.format(datafile_extension)) if f.is_file()),
            key=lambda path: int(path.stem.rsplit("_", 1)[1])
            )][0]
            # convert from path to str
            source_files.append([str(lst[i]) for i in range(0,len(lst))])   
        
        #convert path folder to str
        source_folder = [str(f) for f in source_folder]
            
        #create target folder    
        target_folder = [f.replace(rename_target_datafolder[0], rename_target_datafolder[1]) for f in source_folder]
        
        return source_folder, source_files, target_folder
    else:
        return None, None, None


def main():
    
    DATASET["source_datafolder_list"], DATASET["source_datafiles_list"], DATASET["target_datafiles_list"] = _scandir(
        source_datafolder_path = PATH["source_directory"],
        datafile_extension = DATASET["source_file_extension"],
        rename_target_datafolder = [
            DATASET["source_datafolder"]+"/"+DATASET["dataset_name"],
            DATASET["target_datafolder"]+"/"+DATASET["dataset_name"]
            ])

    # for index, row in df_datafolder.iterrows():
    #     f=str(row["source_full_path"])
    
    
    dataset_container = VTKtoH5(SETTINGS)
    dataset_container.create_empty_h5(
        path=PATH["target_directory"],
        name=DATASET["dataset_name"]
        )
    
    for src, fls, trgt in tqdm(zip(
        DATASET["source_datafolder_list"], 
        DATASET["source_datafiles_list"], 
        DATASET["target_datafiles_list"]),
        total=len(DATASET["source_datafolder_list"]), leave=False):
        
        dataset_container.allocate_values()
        dataset_container.create_dataset(fls)
        dataset_container.write_dataset_h5(
            path=PATH["target_directory"],
            name=DATASET["dataset_name"],
            number = DATASET["source_datafolder_list"].index(src),
            reopen=False
        )
        
    dataset_container = None
        

if __name__ == '__main__':
    main()
    print("Dataset Created")
