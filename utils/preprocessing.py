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
from tqdm import tqdm

from utils.pseudo_abscence import *

#URLs
class URLs():
    url600s='https://biogeo.ucdavis.edu/data/worldclim/v2.1/base/wc2.1_10m_bio.zip' 
    url300s='https://biogeo.ucdavis.edu/data/worldclim/v2.1/base/wc2.1_5m_bio.zip'
    url150s='https://biogeo.ucdavis.edu/data/worldclim/v2.1/base/wc2.1_2.5m_bio.zip'
    url30s='https://biogeo.ucdavis.edu/data/worldclim/v2.1/base/wc2.1_30s_bio.zip'
#Get presence

def get_presence_dependent(dependent):
    
    presence = dependent.loc[dependent.iloc[:, -1] == 1, :]

    return presence

#Download

def from_url_tif(url: str):
    foldername = url.split('_')[1]
    extension = foldername + '.zip'
    folder = f'variables/{foldername}'
    try:
        os.makedirs(folder)
        download_url(url, folder, extension)
        path = os.path.join(folder, extension)
        with zipfile.ZipFile(path, 'r') as zip_ref:
            zip_ref.extractall(folder)
        os.remove(path)
    except FileExistsError:
        pass
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
    file_list = glob.glob(os.path.join('data/', "*.csv")) + glob.glob(os.path.join('data/', "*.xlsx"))
    while True:
        for idx, file in enumerate(file_list):
            print(f'{idx+1}. {file}\n')
        ans = input('(Input the index number)=============================>(q: quit): ')
        if ans.isnumeric():
            idx = int(ans) - 1
            file = file_list[idx]
            break
        else:
            if ans=='q':
                break
            else:
                print('Try again')
    return file

def import_targets(path):
    if 'csv' in path.split('.'):
        dependent = pd.read_csv(path)
    else:
        dependent = pd.read_excel(path)
    #Including the expert-based pseudo-absence data

    return dependent

#
def match_variables(independent, dependent):
    independent = boxes(independent,dependent, 10).restrict()
    #match variables
    for idx, isolated_vector in tqdm(enumerate(dependent.values), desc='Analyzing locations...', total=len(dependent)):
    
        independent['distance'] = independent.apply(lambda row: haversine(isolated_vector[1], isolated_vector[0], row['Latitude'], row['Longitude']), axis=1)
        
        nearest = independent[independent['distance'] == independent['distance'].min()]

        dependent.iloc[idx,:] = [nearest.iloc[0,0], nearest.iloc[0,1], isolated_vector[2]]

    independent = independent.drop('distance', axis = 1)

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
    scaler = StandardScaler()
    inputs_norm = scaler.fit_transform(inputs_array)
    #Creating torch tensors
    inputs = torch.from_numpy(inputs_norm.astype(np.float32()))
    targets = torch.from_numpy(targets_array.astype(np.float32()))
    #Create dataset
    dataset = TensorDataset(inputs, targets)
    return scaler, dataset

def dataframe_to_numpy(dataframe, input_cols, output_cols):
    # Make a copy of the original dataframe
    dataframe1 = dataframe.copy(deep=True)
    # Extract input & outupts as numpy arrays
    inputs_array = dataframe1[input_cols].to_numpy()
    targets_array = dataframe1[output_cols].to_numpy()
    #Normalizing the dataset
    scaler = StandardScaler()
    inputs_norm = scaler.fit_transform(inputs_array)
    return scaler, inputs_norm, targets_array 

def transform(scaler, data, is_torch: bool = True):
    out = scaler.transform(data)
    if is_torch:
        out = torch.from_numpy(out.astype(np.float32())).unsqueeze(0)
    return out

def import_data(url: str):
    path = create_path()
    dependent = import_targets(path)
    folder = from_url_tif(url)
    independent = tif_to_dataframe(folder)
    dependent = match_variables(independent, dependent)
    presence = get_presence_dependent(dependent)
    return dependent, independent, presence

def data_preprocess_with_pseudo(dependent, independent, presence, up_boundary: float, down_boundary: float = None):
    independent = boxes(independent, dependent,up_boundary).restrict()
    if down_boundary is not None:
        dataframe = MinMax_Env(presence, dependent, independent, down_boundary, up_boundary)
    else:
        dataframe = TSKM(presence, dependent, independent, up_boundary)
    scaler_torch, dataset_torch = dataframe_to_torch(dataframe, dataframe.columns.values[2:-1], dataframe.columns.values[-1])
    scaler_numpy, x,y = dataframe_to_numpy(dataframe, dataframe.columns.values[2:-1], dataframe.columns.values[-1])
    return dataset_torch, x,y, scaler_torch, scaler_numpy
    

def data_preprocess_without_pseudo(dependent):
    scaler_torch, dataset_torch = dataframe_to_torch(dependent, dependent.columns.values[2:-1], dependent.columns.values[-1])
    scaler_numpy, x,y = dataframe_to_numpy(dependent, dependent.columns.values[2:-1], dependent.columns.values[-1])
    return dataset_torch, x,y, scaler_torch, scaler_numpy