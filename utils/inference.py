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
    def __init__(self,country:str, model, model_name: str, name: str, independent):
        super().__init__(country, model_name, name)
        if model_name == 'Neural_Network':
            self.torch_inference(independent)
        else:
            self.general_inference(independent)
        self.model = model
    def restrict_zone(self, independent):
        longitude = independent['Longitude']
        latitude = independent['Latitude']
        geometry = gpd.points_from_xy(longitude, latitude)
        geo_data = gpd.GeoDataFrame(independent, geometry=geometry)        
        return gpd.sjoin(geo_data, self.country, how='inner', op='within')
    def torch_inference(self, independent):
        data = self.restrict_zone(independent)

        predictions = []

        for _, row in data.iterrows():
            input_data = torch.tensor(row.values, dtype=torch.float32)
            input_data = input_data.unsqueeze(0)  # Add a batch dimension
            prediction = self.model(input_data)
            predictions.append(prediction.item())

        data['Probability'] = predictions

        os.makedirs('projects/'+self.name+'/inference', exist_ok=True)

        data.loc[:, ['Longitude', 'Latitude', 'Probability']].to_csv('projects/'+self.name+'/inference/'+self.model_name+'.csv', index = False)
    def general_inference(self, independent):
        data = self.restrict_zone(independent)
        predictions = []

        for _, row in data.iterrows():
            input_data = row.values
            prediction = self.model(input_data)
            predictions.append(prediction.item())

        data['Probability'] = predictions
        
        os.makedirs('projects/'+self.name+'/inference', exist_ok=True)

        data.loc[:, ['Longitude', 'Latitude', 'Probability']].to_csv('projects/'+self.name+'/inference/'+self.model_name+'.csv')
    def save_torch(self):
        os.makedirs('projects/'+self.name+'/models', exist_ok=True)
        torch.save(self.model.state_dict(), 'projects/'+self.name+'/models/'+self.model_name+'.pth')
    def save_joblib(self):
        os.makedirs('projects/'+self.name+'/models', exist_ok=True)
        joblib.dump(self.model, 'projects/'+self.name+'/models/'+self.model_name+'.pkl')