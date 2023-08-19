import pandas as pd
from geopy.distance import geodesic
import random
from tqdm import tqdm



### Function f

def dataframe_distance(presence, dependent, independent, down_boundary: int, up_boundary: int):
    print('1. Generating pseudo-absence points:\n')
    print('1.1. Min-Max radius analysis')
    #presence points
    presence_points = presence.loc[:, ['Latitude', 'Longitude']].values

    ## Union of collection A and B
    
    A_alpha = []
    B_alpha = []

    for point in tqdm(presence_points, desc='Generating Min-Max radius area ...', total=len(presence_points)):
        filter_ =lambda x: independent.apply(lambda row: geodesic(point, (row['Latitude'], row['Longitude'])), axis = 1)<x
        A_alpha.append(independent[filter_(down_boundary)])
        B_alpha.append(independent[filter_(up_boundary)])

    Union_A_alpha = pd.concat(A_alpha, ignore_index=True).drop_duplicates()
    Union_B_alpha = pd.concat(B_alpha, ignore_index=True).drop_duplicates()
    

    #Complement B/A
    complement_df = pd.concat([Union_B_alpha, Union_A_alpha, Union_A_alpha]).drop_duplicates(keep=False).reset_index(drop=True)

    print('Done!')

    return dependent, presence, complement_df

### g function

def variable_analysis(dependent, presence, filtered_dataframe):
    print('1.2. Bioclim variable analysis ...')
    ##Variables analysis    (presence, independent)

    analysis = presence.describe().loc[['min', 'max'], ].loc[:, ['bio1', 'bio3', 'bio5', 'bio6', 'bio7', 'bio12']].T.reset_index()

    ##Combine
    for variable, min_value, max_value in tqdm(analysis.values, desc = 'Filtering recurrent variable zones ...'):
            filtered_dataframe = filtered_dataframe[(filtered_dataframe[variable]<min_value) & (filtered_dataframe[variable]>max_value)]
    
    print('Done!')
    
    return dependent, presence, filtered_dataframe.reset_index(drop=True)

### h function

def random_picking(dependent, presence, filtered_dataframe):
    print('1.3. Picking random pseudo-absence points')
    filtered_dataframe['Presence'] = 0

    for i in tqdm(range(len(presence))):
        i = random.randint(0, len(filtered_dataframe))
        element = pd.DataFrame(filtered_dataframe.reset_index(drop=True).iloc[i,:]).T
        dependent = dependent.append(element)

    return dependent.reset_index(drop=True)

def absence_generator(presence, dependent, independent, down_boundary: int, up_boundary: int):
    
    ### h(g(f(x)))---->dataframe

    dependent, presence, complement_df = dataframe_distance(presence, dependent, independent, down_boundary, up_boundary)

    dependent, presence, filtered_dataframe = variable_analysis(dependent, presence, complement_df)

    dataframe = random_picking(dependent, presence, filtered_dataframe)

    return dataframe