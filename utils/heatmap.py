import matplotlib.pyplot as plt
import geopandas as gpd
from scipy.interpolate import griddata
import pandas as pd
from utils.random_utils import Project
import os
from torchvision.datasets.utils import download_url
import numpy as np

class Map(Project):
    def __init__(self, model_name: str, name: str):
        super().__init__(name)
        global country
        self.model_name = model_name
        self.country = gpd.read_file(self.get_country(country))
        # Read the longitude, latitude, probability csv
        self.data = pd.read_csv('projects/'+self.name+f'/inference/{model_name}.csv')
        self.longitude = self.data['Longitude']
        self.latitude = self.data['Latitude']
        self.probability = self.data['Probability']
        self.geometry = gpd.points_from_xy(self.longitude, self.latitude)
        self.geo_data = gpd.GeoDataFrame(self.data, geometry=self.geometry)
    
    def get_country(self):
        global country
        try:
            with open('utils/map_dependencies/countries.geojson', 'r') as attempt:
                attempt.close()
        except FileNotFoundError:
            os.makedirs('utils/map_dependencies')
            url='https://datahub.io/core/geo-countries/r/countries.geojson'
            root = 'utils/map_dependencies'
            filename = 'countries.geojson'
            download_url(url, root, filename)

        while True:
            with open('utils/map_dependencies/countries.geojson', 'r') as file:
                lines = file.readlines()
                data = None
                for line in lines:
                    if country in line:
                        data = line
                        break
                if data is None:
                    print('Put again the country name\n')
                    country = input('Put the country: ')
                    continue
                else:
                    break
        path = f'utils/map_dependencies/{country}.geojson'
        with open(path, 'w') as file:
            file.write(data)

        return path

    def get_map(self, model_name: str):
        ## Interpolation
        min_lon, max_lon, min_lat, max_lat = self.country  ##cambiar
        
        grid_lon, grid_lat = np.meshgrid(np.linspace(min_lon, max_lon, 200), np.linspace(min_lat, max_lat, 200))
        
        grid_probabilities = griddata((self.longitude, self.latitude), self.probability, (grid_lon, grid_lat), method='linear')
        
        ##plot
        _, ax = plt.subplots(1, 1, figsize=(10, 10))
        
        heatmap = ax.imshow(grid_probabilities, extent=(min_lon, max_lon, min_lat, max_lat), origin='lower', cmap='OrRd')
        
        ax.set_title('Species Probability Heatmap')

        plt.colorbar(heatmap, ax=ax, label='Probability')

        plt.savefig('projects/'+self.name+'/heatmap/'+model_name+'.png')


