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
        if ans == 'n':
            ans = 0
            break
        else:
            print('Try again')
    return ans

if __name__ == '__main__':
    # General dependencies of the app
    
    country = input('On which country do you want your map plot?: ')
    url = URLs.url600s
    
    # Ask for mode, perform data preprocessing

    
    if program_init():
        down_boundary = float(input('Min radius:'))
        up_boundary = float(input('Max radius:'))
        torch_dataset, (x,y) = data_preprocess_with_pseudo(url, down_boundary, up_boundary)
    else:
        torch_dataset, (x,y) = data_preprocess_without_pseudo(url)
    
    # Training phase

    ## Neural network parameters
    batch_size = 1
    epochs = 250
    lr = 1e-3
    weight_decay = 1e-4
    grad_clip = 1e-2
    opt_func = torch.optim.Adam
    
    ## training...
    name, models = train_phase(torch_dataset, batch_size, x,y, epochs, lr, weight_decay, grad_clip, opt_func)

    
    # Model inference
    folder = from_url_tif(url)

    independent = tif_to_dataframe(folder, up_boundary, inference = True)
    
    LR_model_name = 'Maximum_entropy'
    RF_model_name = 'Random_forest'
    NN_model_name = 'Neural_Network'
    
    LR_model = Model(country, models[0],LR_model_name, name, independent)
    RF_model = Model(country, models[1],RF_model_name, name, independent)
    NN_model = Model(country, models[2],NN_model_name, name, independent)

    LR_model.get_map()
    RF_model.get_map()
    NN_model.get_map()

    LR_model.save_joblib()
    RF_model.save_joblib()
    NN_model.save_torch()

    print('All the files are stored inside projects file with the respective project name')
