import pandas as pd


def absence_generator(data):
    presence = data.loc[data['Presence'] == 1, :]
    
    ...
    


    pseudo = ...

    dataframe = pd.concat([data, pseudo], axis = 0)

    return dataframe