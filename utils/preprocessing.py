import georasters as gr
import pandas as pd
import numpy as np
import zipfile
import torch
import glob
import os

from torch.utils.data import TensorDataset
from torchvision.datasets.utils import download_url
from sklearn.preprocessing import StandardScaler

from utils.pseudo_abscence import *

#URLs
class URLs():
    url600s='https://biogeo.ucdavis.edu/data/worldclim/v2.1/base/wc2.1_10m_bio.zip' 
    url300s='https://biogeo.ucdavis.edu/data/worldclim/v2.1/base/wc2.1_5m_bio.zip'
    url150s='https://biogeo.ucdavis.edu/data/worldclim/v2.1/base/wc2.1_2.5m_bio.zip'
    url30s='https://biogeo.ucdavis.edu/data/worldclim/v2.1/base/wc2.1_30s_bio.zip'
#Get presence

def get_presence_dependent(independent, dependent):
    
    data = match_variables(independent,dependent)

    presence = data.loc[data.iloc[:, -1] == 1, :]

    return presence, data

#Download

def from_url_tif(url: str):
    foldername = url.split('_')[1]
    extension = foldername + '.zip'
    folder = f'variables/{foldername}'
    os.makedirs(folder, exist_ok=True)
    download_url(url, folder, extension)
    path = os.path.join(folder, extension)
    with zipfile.ZipFile(path, 'r') as zip_ref:
        zip_ref.extractall(folder)
    os.remove(path)
    return folder

#transform to dataframe

def tif_to_dataframe(tif_path):
    for idx, file in enumerate(os.listdir(tif_path)):
        path = os.path.join(tif_path, file)
        variable = gr.from_file(path).to_pandas()
        if idx == 0:
            dataframe = variable.loc[:, ['x', 'y']].rename(columns = {'x':'Longitude', 'y': 'Latitude'})
        dataframe = pd.concat([dataframe,variable[True].rename(file.split('_')[2] + file.split('_')[-1][:-4])], axis = 1)
    return dataframe

# Targets

def create_path():
    print('Select the dataset: \n')
    file_list = glob.glob('data/*.{xlsx,csv}')
    while True:
        for idx, file in file_list:
            print(f'{idx+1}. {file}\n')
        ans = input('(Input the index number)=============================>(q: quit): ')
        if ans.isnumeric():
            idx = int(ans) - 1
            file = file_list[idx]
        else:
            if ans=='q':
                break
            else:
                print('Try again')
    path = os.path.join('data', file)
    return path

def import_targets(path):
    if 'csv' in path.split('.'):
        presence = pd.read_csv(path)
    else:
        presence = pd.read_excel(path)
    #Including the expert-based pseudo-absence data

    return presence

#
def match_variables(independent, dependent):
    #match variables
    #pd.merge
    dataset = pd.merge(independent, dependent, 'right', left_index=False, right_index=False)
    return dataset

#Data preprocessing
def dataframe_to_torch(dataframe, input_cols, output_cols):
    # Make a copy of the original dataframe
    dataframe1 = dataframe.copy(deep=True)
    # Extract input & outupts as numpy arrays
    inputs_array = dataframe1[input_cols].to_numpy()
    targets_array = dataframe1[output_cols].to_numpy()
    #Normalizing the dataset
    inputs_norm = StandardScaler().fit_transform(inputs_array)
    #Creating torch tensors
    inputs = torch.from_numpy(inputs_norm.astype(np.float32()))
    targets = torch.from_numpy(targets_array.astype(np.float32()))
    #Create dataset
    dataset = TensorDataset(inputs, targets)
    return dataset

def dataframe_to_numpy(dataframe, input_cols, output_cols):
    # Make a copy of the original dataframe
    dataframe1 = dataframe.copy(deep=True)
    # Extract input & outupts as numpy arrays
    inputs_array = dataframe1[input_cols].to_numpy()
    targets_array = dataframe1[output_cols].to_numpy()
    #Normalizing the dataset
    inputs_norm = StandardScaler().fit_transform(inputs_array)
    return inputs_norm, targets_array 

def scaler(dataframe, input_cols):
    # Make a copy of the original dataframe
    dataframe1 = dataframe.copy(deep=True)
    # Extract input & outupts as numpy arrays
    inputs_array = dataframe1[input_cols].to_numpy()
    #Normalizing the dataset
    preprocess = StandardScaler().fit(inputs_array)
    return preprocess

def transform(scaler, data):
    out = scaler(data)
    out = torch.from_numpy(out.astype(np.float32()))
    return out

def data_preprocess(url: str, presence, independent, down_boundary: int, up_boundary: int, bounding_box: boxes=None):
    folder = from_url_tif(url)
    independent = tif_to_dataframe(folder)
    path = create_path()
    dependent = import_targets(path)
    presence, dependent = get_presence_dependent(independent, dependent) 
    dataframe = absence_generator(presence, dependent, independent, down_boundary, up_boundary, bounding_box)
    dataset_torch = dataframe_to_torch(dataframe, dataframe.columns.values[:-1], dataframe.columns.values[-1])
    x,y = dataframe_to_numpy(dataframe, dataframe.columns.values[:-1], dataframe.columns.values[-1])
    return [dataset_torch, (x,y)]
    

