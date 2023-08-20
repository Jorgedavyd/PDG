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

def haversine(lat1, lon1, lat2, lon2):
    """
    Calculate the great-circle distance between two points 
    on the Earth's surface using the Haversine formula.
    """
    R = 6371.0  # Earth radius in kilometers

    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)

    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    distance = R * c
    return distance

#Bounding boxes
class boxes:
    
    North_America= {
        'latitude': {'min': 10.0, 'max': 80.0},
        'longitude': {'min': -180.0, 'max': -35.0}
    }

    Central_America= {
        'latitude': {'min': 5.0, 'max': 35.0},
        'longitude': {'min': -95.0, 'max': -70.0}
    }

    South_America = {
        'latitude': {'min': -55.0, 'max': 12.0},
        'longitude': {'min': -80.0, 'max': -35.0}
    }

    Europe = {
        'latitude': {'min': 35.0, 'max': 72.0},
        'longitude': {'min': -25.0, 'max': 65.0}
    }

    Asia = {
        'latitude': {'min': -10.0, 'max': 75.0},
        'longitude': {'min': 25.0, 'max': 180.0}
    }

    Africa = {
        'latitude': {'min': -40.0, 'max': 37.0},
        'longitude': {'min': -25.0, 'max': 52.0}
    }

    Oceania = {
        'latitude': {'min': -55.0, 'max': 0.0},
        'longitude': {'min': 85.0, 'max': 180.0}
    }

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

def tif_to_dataframe(tif_path, bounding_box = None):
    for idx, file in enumerate(os.listdir(tif_path)):
        path = os.path.join(tif_path, file)
        variable = gr.from_file(path).to_pandas()
        if idx == 0:
            dataframe = variable.loc[:, ['x', 'y']].rename(columns = {'x':'Longitude', 'y': 'Latitude'})
        dataframe = pd.concat([dataframe,variable[True].rename(file.split('_')[2] + file.split('_')[-1][:-4])], axis = 1)
        ## Domain

    if bounding_box is not None:
        dataframe = dataframe[
            (dataframe['Latitude'] >= bounding_box['latitude']['min']) &
            (dataframe['Latitude'] <= bounding_box['latitude']['max']) &
            (dataframe['Longitude'] >= bounding_box['longitude']['min']) &
            (dataframe['Longitude'] <= bounding_box['longitude']['max'])
        ].reset_index(drop = True)

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
        presence = pd.read_csv(path)
    else:
        presence = pd.read_excel(path)
    #Including the expert-based pseudo-absence data

    return presence

#
def match_variables(independent, dependent):
    #match variables
    for idx, isolated_vector in tqdm(enumerate(dependent.values), desc='Analyzing locations...', total=len(dependent)):
    
        independent['distance'] = independent.apply(lambda row: haversine(isolated_vector[1], isolated_vector[0], row['Latitude'], row['Longitude']), axis=1)
        
        nearest = independent[independent['distance'] == independent['distance'].min()]

        dependent.iloc[idx,:] = (float(nearest['Longitude']), float(nearest['Latitude']), isolated_vector[2])

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

def data_preprocess(url: str, down_boundary: int, up_boundary: int, bounding_box: boxes=None):
    folder = from_url_tif(url)
    independent = tif_to_dataframe(folder, bounding_box)
    path = create_path()
    dependent = import_targets(path)
    presence, dependent = get_presence_dependent(independent, dependent) 
    dataframe = absence_generator(presence, dependent, independent, down_boundary, up_boundary)
    dataset_torch = dataframe_to_torch(dataframe, dataframe.columns.values[:-1], dataframe.columns.values[-1])
    x,y = dataframe_to_numpy(dataframe, dataframe.columns.values[:-1], dataframe.columns.values[-1])
    return dataset_torch, (x,y)
    

