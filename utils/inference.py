## Create csv with all models
from utils.heatmap import Map
from utils.preprocessing import *
import joblib
import geopandas as gpd
"""
for i in output:
    inferentiate the data into probabilities df.apply(lambda row: model(1, row[i]) for i in bioclim_variables)
    drops bioclim variables(longitude, latitude, probabilities)
    create csv Longitude, Latitude, Probabilities on 'projects/inference'
    
"""
class Model(Map):
    def __init__(self,country:str, model, model_name: str, name: str, independent, scaler):
        super().__init__(country, model_name, name)
        self.model = model
        if model_name == 'Neural_Network':
            self.torch_inference(independent, scaler)
        else:
            self.general_inference(independent, scaler)
    def restrict_zone(self, df):
        geo_df = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df['Longitude'], df['Latitude']))
        geo_df.crs = 'EPSG:4326'
        data = gpd.sjoin(geo_df, self.country, how='inner', op='within')
        data = data.drop(['geometry', 'index_right','ADMIN','ISO_A3'], axis = 1)
        return data
    
    def torch_inference(self, independent, scaler):
        data = self.restrict_zone(independent)
        predictions = []

        for _, row in data.iterrows():
            input_data = transform(scaler , row.values[2:].reshape(1,-1), is_torch = True)
            prediction = self.model(input_data)
            predictions.append(prediction.item())

        data['Probability'] = predictions


        os.makedirs('projects/'+self.name+'/inference', exist_ok=True)

        data.loc[:, ['Longitude', 'Latitude', 'Probability']].to_csv('projects/'+self.name+'/inference/'+self.model_name+'.csv', index = False)
    def general_inference(self, independent, scaler):
        data = self.restrict_zone(independent)
        predictions = []

        for _, row in data.iterrows():
            input_data = transform(scaler, row.values[2:].reshape(1,-1), is_torch = False)
            prediction = self.model.predict(input_data)
            predictions.append(prediction.item())

        data['Probability'] = predictions
        
        os.makedirs('projects/'+self.name+'/inference', exist_ok=True)

        data.loc[:, ['Longitude', 'Latitude', 'Probability']].to_csv('projects/'+self.name+'/inference/'+self.model_name+'.csv', index = False)
    def save_torch(self):
        os.makedirs('projects/'+self.name+'/models', exist_ok=True)
        torch.save(self.model.state_dict(), 'projects/'+self.name+'/models/'+self.model_name+'.pth')
    def save_joblib(self):
        os.makedirs('projects/'+self.name+'/models', exist_ok=True)
        joblib.dump(self.model, 'projects/'+self.name+'/models/'+self.model_name+'.pkl')