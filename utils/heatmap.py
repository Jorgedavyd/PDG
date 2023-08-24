import matplotlib.pyplot as plt
import geopandas as gpd
import pandas as pd
from random_utils import Project
import os
from torchvision.datasets.utils import download_url

class Map(Project):
    def __init__(self, country: str, model_name: str, name: str):
        super().__init__(name)
        self.model_name = model_name
        self.country = gpd.read_file(self.get_country(country))
        # Read the longitude, latitude, probability csv
        self.data = pd.read_csv('projects/'+self.name+f'/inference/{model_name}.csv')
        self.longitude = self.data['Longitude']
        self.latitude = self.data['Latitude']
        self.geometry = gpd.points_from_xy(self.longitude, self.latitude)
        self.geo_data = gpd.GeoDataFrame(self.data, geometry=self.geometry)
    
    def get_country(self, country: str):
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
                    print('Put again the country name')
                    continue
                else:
                    break
        path = f'utils/map_dependencies/{country}.geojson'
        with open(path, 'w') as file:
            file.write(data)

        return path

    def get_map(self, model_name: str):
        _, ax = plt.subplots(1, 1, figsize=(10, 10))
        self.geo_data.plot(column='Specie Distribution', cmap='OrRd', markersize=10, ax=ax, legend=True)
        ax.set_title('Species Probability Heatmap (Within Country Boundary)')
        plt.savefig('projects/'+self.name+'/heatmap/'+model_name+'.png')
        plt.show()

