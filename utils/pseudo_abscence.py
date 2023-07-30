import pandas as pd
from geopy.distance import geodesic
import random

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

### Function f

def dataframe_distance(presence, dependent, independent, down_boundary: int, up_boundary: int, bounding_box: boxes=None):
    #presence points
    presence_points = presence.loc[:, ['Latitude', 'Longitude']].values
    
    ## Domain

    if bounding_box is not None:
        independent = independent[
            (independent['Latitude'] >= bounding_box['latitude']['min']) &
            (independent['Latitude'] <= bounding_box['latitude']['max']) &
            (independent['Longitude'] >= bounding_box['longitude']['min']) &
            (independent['Longitude'] <= bounding_box['longitude']['max'])
        ]

    ## Union of collection A and B
    
    A_alpha = []
    B_alpha = []

    for point in presence_points:
        filter_ =lambda x: independent.apply(lambda row: geodesic(point, (row['Latitude'], row['Longitude'])), axis = 1)<x
        A_alpha.append(independent[filter_(down_boundary)])
        B_alpha.append(independent[filter_(up_boundary)])

    Union_A_alpha = pd.concat(A_alpha, ignore_index=True).drop_duplicates()
    Union_B_alpha = pd.concat(B_alpha, ignore_index=True).drop_duplicates()
    

    #Complement B/A
    complement_df = pd.concat([Union_B_alpha, Union_A_alpha, Union_A_alpha]).drop_duplicates(keep=False).reset_index(drop=True)

    return dependent, presence, complement_df

### g function

def variable_analysis(dependent, presence, filtered_dataframe):
    
    ##Variables analysis    (presence, independent)

    analysis = presence.describe().loc[['min', 'max'], ].iloc[:, 2:-1].T.reset_index()

    ##Combine
    for variable, min_value, max_value in analysis.values:
            filtered_dataframe = filtered_dataframe[(filtered_dataframe[variable]<min_value) & (filtered_dataframe[variable]>max_value)]
    
    return dependent, presence, filtered_dataframe.reset_index(drop=True)

### h function

def random_picking(dependent, presence, filtered_dataframe):

    filtered_dataframe['Presence'] = 0

    for i in range(len(presence)):
        i = random.randint(0, len(filtered_dataframe))
        element = pd.DataFrame(filtered_dataframe.reset_index(drop=True).iloc[i,:]).T
        dependent = dependent.append(element)

    return dependent.reset_index(drop=True)

def absence_generator(presence, dependent, independent, down_boundary: int, up_boundary: int, bounding_box: boxes=None):
    
    ### h(g(f(x)))---->dataframe

    dependent, presence, complement_df = dataframe_distance(presence, dependent, independent, down_boundary, up_boundary, bounding_box)

    dependent, presence, filtered_dataframe = variable_analysis(dependent, presence, complement_df)

    dataframe = random_picking(dependent, presence, filtered_dataframe)

    return dataframe