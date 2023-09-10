from sklearn.decomposition import PCA
from geopy.distance import geodesic
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import numpy as np
import math


from utils.models import OneClassSVMClassifier, KMeansCluster

def haversine(lat1, lon1, lat2, lon2):
    """
    Calculate the great-circle distance between two points 
    on the Earth's surface using the Haversine formula.
    """
    R = 6371.0  # Earth radius in kilometers

    # Convert input degrees to radians
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)

    # Differences in radians
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    # Haversine formula
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    # Calculate distance
    distance = R * c
    return distance

#Bounding boxes
class boxes:
    def __init__(self, independent, dependent, up_boundary: float):
        self.data = independent
        self.longitude = dependent['Longitude']
        self.latitude = dependent['Latitude']
        self.up_boundary = up_boundary
    def transformation(self, lat, lon, distance):

        # Earth's radius in kilometers
        earth_radius = 6371.0

        delta_lat =  distance/ earth_radius
        delta_lon = distance / (earth_radius * math.cos(math.radians(lat)))

        # Calculate new coordinates
        new_lat = lat + math.degrees(delta_lat)
        new_lon = lon + math.degrees(delta_lon)

        return new_lon, new_lat
    def restrict(self):
        min_lat = self.latitude.min()
        max_lat = self.latitude.max()
        min_lon = self.longitude.min()
        max_lon = self.longitude.max()
        min_lon_bound, min_lat_bound = self.transformation(min_lat, min_lon, -self.up_boundary)
        max_lon_bound, max_lat_bound = self.transformation(max_lat, max_lon, self.up_boundary)
        bounding_box = {
            'latitude': {'min': min_lat_bound, 'max': max_lat_bound},
            'longitude': {'min': min_lon_bound, 'max': max_lon_bound}
        }
        self.data = self.data[
            (self.data['Latitude'] >= bounding_box['latitude']['min']) &
            (self.data['Latitude'] <= bounding_box['latitude']['max']) &
            (self.data['Longitude'] >= bounding_box['longitude']['min']) &
            (self.data['Longitude'] <= bounding_box['longitude']['max'])
        ].reset_index(drop = True)
        return self.data

### Interface for TSKM
def TSKM(presence, dependent, independent, up_boundary):
    #getting the data
    pseudo_absences = simple_distance(presence, independent, up_boundary)

    #OCSVM
    pseudo_absences = OCSVM_step(pseudo_absences, presence)
    
    #K-means
    n_clusters = len(presence)

    pseudo_absences = Kmeans_step(pseudo_absences, n_clusters)
    #create dataframe
    
    df = pd.concat([dependent,pseudo_absences], ignore_index = True)
    
    return df
##OCSVM step
def OCSVM_step(pseudo_absences, presence):
    #Define data
    X = presence.drop(['Longitude', 'Latitude', 'Presence'], axis = 1).values
    #Define one class classifier
    OneClassSVM = OneClassSVMClassifier()
    #train oneclass support vector machine
    OneClassSVM.train(X)
    #inference similarity with presence
    pseudo_absences['prob'] = OneClassSVM.predict(pseudo_absences.iloc[:, 2:])
    #take 0 similarity
    pseudo_absences = pseudo_absences.loc[pseudo_absences['prob']==-1, :].drop('prob', axis = 1)
    return pseudo_absences

def Kmeans_step(pseudo_absences, n_clusters):
    #Define the data
    X = pseudo_absences.drop(['Longitude', 'Latitude'], axis = 1)
    columns = X.columns.values
    #Define the clusterer
    kmeans = KMeansCluster(n_clusters=n_clusters)
    kmeans.fit(X.values)
    centroid_dataframe = kmeans.get_centroids(columns)
    centroid_dataframe['Presence'] = 0
    return centroid_dataframe

def OptimunDistance_interface(dataframe):
    print('\n'*10)
    print(dataframe)
    while True:
        try:    
            option = int(input('\n1. Add radius.\n2.Plot.\n3.Quit optimized distance mode.\n================>'))
            if option>3 or option<1:
                raise TypeError
            break
        except TypeError:
            print('Try again')
            continue
    return option

def OptimunDistance(presence, dependent, independent, radius_dataframe): #whole independent
    while True:
        #show options
        option = OptimunDistance_interface(radius_dataframe)
        if option == 1:
            while True:
                try:
                    radius = float(input('Radius: '))
                    break
                except TypeError:
                    print('Try again')
                    continue
            row = PCA_analysis(presence, dependent, independent, radius)
            radius_dataframe.loc[len(radius_dataframe)] = row
            radius_dataframe = radius_dataframe.sort_values(by=['Radius'])
            continue
        elif option == 2:
            print(radius_dataframe)
            var = input('\nVariable name: ')
            _, ax = plt.subplots(figsize = (10,10), dpi = 100)
            ax.plot(radius_dataframe.Radius.values, radius_dataframe[var].values, c = 'r')
            plt.show()
            continue
        elif option == 3:
            return radius_dataframe


### Function f

def dataframe_distance(presence, independent, down_boundary: float, up_boundary: float):
    print('\n\n1. Generating pseudo-absence points:\n')
    print('1.1. Min-Max radius analysis\n')
    #presence points
    presence_points = presence.loc[:, ['Latitude', 'Longitude']].values

    ## Union of collection A and B
    
    A_alpha = []
    B_alpha = []

    for point in tqdm(presence_points, desc='Generating Min-Max radius area ...', total=len(presence_points)):
        filter_ =lambda x: independent.apply(lambda row: haversine(*point, row['Latitude'], row['Longitude']), axis = 1)<x
        A_alpha.append(independent[filter_(down_boundary)])
        B_alpha.append(independent[filter_(up_boundary)])

    Union_A_alpha = pd.concat(A_alpha, ignore_index=True).drop_duplicates()
    Union_B_alpha = pd.concat(B_alpha, ignore_index=True).drop_duplicates()
    

    #Complement B/A
    complement_df = pd.concat([Union_B_alpha, Union_A_alpha, Union_A_alpha]).drop_duplicates(keep=False).reset_index(drop=True)

    print('Done!')

    return complement_df

def simple_distance(presence, independent, up_boundary: float):

    presence_points = presence.loc[:, ['Latitude', 'Longitude']].values
    constrained_data = []
    for point in tqdm(presence_points, desc='Generating radius area ...', total=len(presence_points)):
        filter_ =lambda x: independent.apply(lambda row: haversine(*point, row['Latitude'], row['Longitude']), axis = 1)<x
        constrained_data.append(independent[filter_(up_boundary)])
    constrained_dataframe = pd.concat(constrained_data, ignore_index=True).drop_duplicates()
    
    return constrained_dataframe

### g function

def variable_analysis(dependent, presence, filtered_dataframe):
    print('1.2. Bioclim variable analysis ...\n')

    ##Variables analysis    (presence, independent)

    main_bio = ['bio1', 'bio3', 'bio5', 'bio6', 'bio7', 'bio12'] ##You can try changing these values, or use all variables

    analysis = presence.describe().loc[['mean'], ].loc[:, main_bio].T.reset_index()


    filtered_dataframe['distance'] = 0

    for variable, mean in tqdm(analysis.values, desc = 'Statistical analysis ...'):
        filtered_dataframe[f'distance_{variable}'] = filtered_dataframe.apply(lambda row: float(np.linalg.norm(mean-row[variable])), axis = 1)
        max_value = filtered_dataframe[f'distance_{variable}'].max()
        filtered_dataframe['distance'] = filtered_dataframe.apply(lambda row: row['distance'] + row[f'distance_{variable}']/max_value , axis = 1)
        filtered_dataframe = filtered_dataframe.drop(f'distance_{variable}', axis = 1)

    filtered_dataframe['distance'] = filtered_dataframe.apply(lambda row: row['distance']/6, axis = 1)

    filtered_dataframe = filtered_dataframe.sort_values('distance', ascending = False).head(100*(2*len(presence) - len(dependent))).drop('distance', axis = 1)

    print('Done!')
    
    return filtered_dataframe.reset_index(drop=True)

### h function

def unified_dataframe(dependent, filtered_dataframe):
    print('1.3. Picking random pseudo-absence points\n')
    filtered_dataframe['Presence'] = 0

    dataframe = pd.concat([filtered_dataframe, dependent]).sample(frac = 1.0).reset_index(drop=True)

    print('Done!')
    return dataframe

def MinMax_Env(presence, dependent, independent, down_boundary: float, up_boundary: float):
    ## Bounding data
    independent = boxes(independent, dependent, up_boundary).restrict()
    ### h(g(f(x)))---->dataframe

    complement_df = dataframe_distance(presence, independent, down_boundary, up_boundary)

    filtered_dataframe = variable_analysis(dependent, presence, complement_df)

    dataframe = unified_dataframe(dependent, filtered_dataframe)

    return dataframe

#PCA analysis
def PCA_analysis(presence, dependent,independent, up_boundary: float): #we need the whole independent
    independent = boxes(independent,dependent,up_boundary).restrict()
    dataframe = simple_distance(presence, independent, up_boundary)
    dataframe = dataframe.drop(['Longitude', 'Latitude'], axis = 1)#drop the longitude and latitude
    pca = PCA()
    pca.fit(dataframe)
    coef_pc1 = pca.components_[0]
    total_variance_pc1 = np.sum(np.square(coef_pc1))
    percentage_contribution = (np.square(coef_pc1)/total_variance_pc1)
    area = area_analysed(presence, up_boundary)
    return [up_boundary, area, *percentage_contribution]

def init_analysis(presence, dependent, independent, distances:list=[50,100,200,250]):
    columns = ['Radius','Area', *independent.drop(['Longitude', 'Latitude'], axis = 1).columns.values]
    radius_dataframe = pd.DataFrame(columns = columns)
    for radius in distances:
        row = PCA_analysis(presence, dependent, independent, radius)
        radius_dataframe.loc[len(radius_dataframe)] = row
        radius_dataframe = radius_dataframe.sort_values(by=['Radius'])
    return radius_dataframe

def area_analysed(presence, up_boundary: float):
    area = len(presence)*(np.pi*(up_boundary**2))
    return area