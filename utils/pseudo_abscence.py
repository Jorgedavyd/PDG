import pandas as pd
from geopy.distance import geodesic
from tqdm import tqdm
import numpy as np



### Function f

def dataframe_distance(presence, independent, down_boundary: int, up_boundary: int):
    print('\n\n1. Generating pseudo-absence points:\n')
    print('1.1. Min-Max radius analysis\n')
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

    return complement_df

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

    filtered_dataframe = filtered_dataframe.sort_values('distance', ascending = False).head(10*(2*len(presence) - len(dependent))).drop('distance', axis = 1)

    print('Done!')
    
    return filtered_dataframe.reset_index(drop=True)

### h function

def unified_dataframe(dependent, filtered_dataframe):
    print('1.3. Picking random pseudo-absence points\n')
    filtered_dataframe['Presence'] = 0

    dataframe = pd.concat([filtered_dataframe, dependent]).sample(frac = 1.0).reset_index(drop=True)

    print('Done!')
    return dataframe

def absence_generator(presence, dependent, independent, down_boundary: int, up_boundary: int):
    
    ### h(g(f(x)))---->dataframe

    complement_df = dataframe_distance(presence, independent, down_boundary, up_boundary)

    filtered_dataframe = variable_analysis(dependent, presence, complement_df)

    dataframe = unified_dataframe(dependent, filtered_dataframe)

    return dataframe