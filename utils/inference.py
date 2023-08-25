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
    def __init__(self, model, model_name: str, name: str):
        super().__init__(model_name, name)
        self.model = model
    def restrict_zone(self, independent):
        return gpd.sjoin(independent, self.country, how='inner', op='within')
    def torch_inference(self, independent):
        data = self.restrict_zone(independent)
        def perform_inference(row):
            input_data = torch.tensor(row.values, dtype=torch.float32)
            input_data = input_data.unsqueeze(0)  # Add a batch dimension
            prediction = self.model(input_data)
            return prediction.item()
        data['Probability'] = data.apply(perform_inference, axis = 1)
        data.loc[:, ['Longitude', 'Latitude', 'Probability']].to_csv('projects/'+self.name+'/inference/'+self.model_name+'.csv')
    def general_inference(self, independent):
        longitude = independent['Longitude']
        latitude = independent['Latitude']
        geometry = gpd.points_from_xy(longitude, latitude)
        geo_data = gpd.GeoDataFrame(independent, geometry=geometry)
        data = self.restrict_zone(geo_data)
        def perform_inference(row):
            input_data = row.values
            prediction = self.model(input_data)
            return prediction[0]
        data['Probability'] = data.apply(perform_inference, axis = 1)
        data.loc[:, ['Longitude', 'Latitude', 'Probability']].to_csv('projects/'+self.name+'/inference/'+self.model_name+'.csv')
    def save_torch(self):
        torch.save(self.model.state_dict(), 'projects/'+self.name+'/models/'+self.model_name+'.pth')
    def save_joblib(self):
        joblib.dump(self.model, 'projects/'+self.name+'/models/'+self.model_name+'.pkl')