import matplotlib.pyplot as plt
import geopandas as gpd
from scipy.interpolate import griddata
import pandas as pd
from utils.random_utils import Project
import os
from torchvision.datasets.utils import download_url
import numpy as np
import folium
from folium.plugins import HeatMap
from shapely.geometry import Point

class Map(Project):
    def __init__(self, country: str, model_name: str, name: str):
        super().__init__(name)
        self.model_name = model_name
        self.country_name = country
        self.country = gpd.read_file(self.get_country())
        # Read the longitude, latitude, probability csv

    def get_country(self):
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
                    if self.country_name in line:
                        data = line
                        break
                if data is None:
                    print('Put again the country name\n')
                    self.country_name = input('Put the country: ')
                    continue
                else:
                    break
        path = f'utils/map_dependencies/{self.country_name}.geojson'
        with open(path, 'w') as file:
            file.write(data[:-1])

        return path

    def get_map(self):
        self.data = pd.read_csv('projects/'+self.name+f'/inference/{self.model_name}.csv')

        latitudes=list(self.data['Latitude'])
        longitudes=list(self.data['Longitude'])

        geometry = [Point(xy) for xy in zip(longitudes, latitudes)]
        geo_df = gpd.GeoDataFrame(self.data, geometry=geometry)

        # Create a base map centered around the mean of latitude and longitude
        m = folium.Map(location=[geo_df['Latitude'].mean(), geo_df['Longitude'].mean()], zoom_start=10)

        # Convert GeoDataFrame to a list of (latitude, longitude, weight) for HeatMap
        heat_data = [(row['Latitude'], row['Longitude'], row['Probability']) for index, row in geo_df.iterrows()]

        # Create a HeatMap layer
        HeatMap(heat_data, radius=15).add_to(m)

        # Display the map
        os.makedirs('projects/'+self.name+'/heatmap/', exist_ok=True)
        m.save('projects/'+self.name+'/heatmap/'+self.model_name+'_heatmap.html')  # Save the map as an HTML file

