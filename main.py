from utils.preprocessing import *
from utils.pseudo_abscence import *
from utils.heatmap import *
from utils.random_utils import *
from utils.models import *
from utils.inference import *

def program_init():
    while True:
        ans = input('Do you want to create you model with pseudo-absence data or you already have absence points?:\n(Y/n)').lower()
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
        independent, torch_dataset, numpy_data = data_preprocess_with_pseudo(url, down_boundary, up_boundary)
    else:
        independent, torch_dataset, numpy_data = data_preprocess_without_pseudo(url)
    
    # Training phase

    ## Neural network parameters
    batch_size = 16
    epochs = 250
    lr = 1e-4
    weight_decay = 1e-5
    grad_clip = None
    opt_func = torch.optim.Adam
    
    ## training...
    models = train_phase(torch_dataset, batch_size, numpy_data, epochs, lr, weight_decay, grad_clip, opt_func)

    
    # Model inference
    
    LR_model_name = 'Maximum_entropy'
    RF_model_name = 'Random_forest'
    NN_model_name = 'Neural_Network'
    
    LR_model = Model(models[0],LR_model_name, name)
    RF_model = Model(models[1],RF_model_name, name)
    NN_model = Model(models[2],NN_model_name, name)

    LR_model.general_inference(independent)
    RF_model.general_inference(independent)
    NN_model.torch_inference(independent)
    
    LR_model.get_map(LR_model_name)
    RF_model.get_map(RF_model_name)
    NN_model.get_map(NN_model_name)

    print('All the files are stored inside projects file with the respective project name')





    
    