from utils.preprocessing import *
from utils.pseudo_abscence import *
from utils.heatmap import *
from utils.random_utils import *
from utils.models import *
from utils.inference import *

def program_init():
    while True:
        ans = input('Do you want to create you model with program generated pseudo-absence data?(if you dont make sure you have absence data points on your dataset):\n(Y/n):').lower()
        if ans == 'y':
            ans = 1
            break
        elif ans == 'n':
            ans = 0
            break
        else:
            print('Try again')
    return ans
def measurement():
    while True:
        ans = input('Data accuracy:\n1. 10m\n2. 5m\n3. 2.5 m\n4. 30 s\n===================>')
        if ans == '1':
            ans = URLs.url600s
            break
        elif ans == '2':
            ans= URLs.url300s
            break
        elif ans =='3':
            ans= URLs.url150s
            break
        elif ans == '4':
            ans= URLs.url30s
            break
        else:
            print('Try again')
            continue
    return ans

if __name__ == '__main__':
    # General dependencies of the app
    name = input('Name of the project: ')
    country = input('On which country do you want your map plot?: ')
    url = measurement()
    # Ask for mode, perform data preprocessing
    
    if program_init():
        down_boundary = float(input('Min radius:'))
        up_boundary = float(input('Max radius:'))
        torch_dataset, x, y, scaler_torch, scaler_numpy= data_preprocess_with_pseudo(url, down_boundary, up_boundary)
    else:
        torch_dataset, x,y, scaler_torch, scaler_numpy = data_preprocess_without_pseudo(url)
    
    # Training phase

    ## Neural network parameters
    epochs = 500
    lr = 1e-4
    weight_decay = 1e-5
    grad_clip = 1e-2
    opt_func = torch.optim.Adam
    
    ## training...
    models = train_phase(name, torch_dataset, x,y, epochs, lr, weight_decay, grad_clip, opt_func)

    
    # Model inference
    folder = from_url_tif(url)

    independent = tif_to_dataframe(folder, up_boundary, inference = True)
    
    LR_model_name = 'Maximum_entropy'
    RF_model_name = 'Random_forest'
    NN_model_name = 'Neural_Network'
    
    LR_model = Model(country, models[0],LR_model_name, name, independent, scaler_numpy)
    RF_model = Model(country, models[1],RF_model_name, name, independent, scaler_numpy)
    NN_model = Model(country, models[2],NN_model_name, name, independent, scaler_torch)

    LR_model.get_map()
    RF_model.get_map()
    NN_model.get_map()

    LR_model.save_joblib()
    RF_model.save_joblib()
    NN_model.save_torch()

    print('All the files are stored inside projects file with the respective project name')
