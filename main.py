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

def pseudo_interface(presence, dependent, independent):
    
    radius_dataframe = init_analysis(presence, dependent,independent)
    print('\n'*100)
    print('-'*100)
    print('PSEUDO-ABSENCE ANALYSIS\n First, find an optimimal radius, the distance at which the contribution of the most important variables decline or stop increasing should be chosen as the optimal limit to bound background data.')
    print('\nNow, you have to choose your pseudo-absence generation method, feel free to experiment with both.(TSKM state-of-the-art)')
    print('\n')
    while True:
        while True:
            print(radius_dataframe)    
            try:
                mode = int(input('\n1.Optimized radius finder.\n2.Three step Min-Max analysis + Environmental variable analysis.\n3.TSKM(Three step K-means).\n4. Random sampling.\n================>'))
                if 0>mode or 3<mode:
                    raise TypeError
                else:
                    break
            except TypeError:
                print('Try again')
                continue
        if mode == 1:
            print(radius_dataframe)
            radius_dataframe = OptimunDistance(presence, dependent, independent, radius_dataframe)
            continue
        elif mode ==2:
            print(radius_dataframe)
            while True:
                try:
                    down_boundary = float(input('Min radius:'))
                    up_boundary = float(input('Max radius:'))
                    if down_boundary+10>=up_boundary:
                        raise TypeError
                    break
                except TypeError:
                    print('Try again')
                    continue
            torch_dataset, x, y, scaler_torch, scaler_numpy= data_preprocess_with_pseudo(dependent,independent,presence,up_boundary, down_boundary)
            return torch_dataset, x, y, scaler_torch, scaler_numpy
        elif mode ==3:
            print(radius_dataframe)
            while True:
                try:
                    radius = float(input('Radius: '))
                    break
                except TypeError:
                    print('Try again')
                    continue    
            torch_dataset, x, y, scaler_torch, scaler_numpy = data_preprocess_with_pseudo(dependent, independent, presence,radius)
            return torch_dataset, x, y, scaler_torch, scaler_numpy
        elif mode ==4:
            print(radius_dataframe)
            while True:
                try:
                    radius = float(input('Radius: '))
                    break
                except TypeError:
                    print('Try again')
                    continue    
            torch_dataset, x, y, scaler_torch, scaler_numpy = random_sampling(dependent, independent, presence,radius)
            return torch_dataset, x, y, scaler_torch, scaler_numpy
        
if __name__ == '__main__':
    print('\n'*100)
    # General dependencies of the app
    name = input('Name of the project: ')
    country = input('On which country do you want your map plot?: ')
    url = measurement()

    dependent, independent, presence = import_data(url)
    # Ask for mode, perform data preprocessing
    
    if program_init():
        torch_dataset, x, y, scaler_torch, scaler_numpy = pseudo_interface(presence, dependent, independent)
    else:
        torch_dataset, x,y, scaler_torch, scaler_numpy = data_preprocess_without_pseudo(dependent)
    
    # Training phase

    ## Neural network parameters
    epochs = 500
    lr = 1e-4
    weight_decay = 1e-5
    grad_clip = 1e-2
    opt_func = torch.optim.Adam
    
    ## training...
    models = train_phase(name, torch_dataset, x,y, epochs, lr, weight_decay, grad_clip, opt_func)

    
    LR_model_name = 'Maximum_entropy'
    RF_model_name = 'Random_forest'
    NN_model_name = 'Neural_Network'
    Bnv_model_name = 'Bernoulli Naive Bayes'
    Mnv_model_name = 'Multinomial Naive Bayes'
    SVM_model_name = 'Support Vector Machines'
    
    LR_model = Model(country, models[0],LR_model_name, name, independent, scaler_numpy)
    RF_model = Model(country, models[1],RF_model_name, name, independent, scaler_numpy)
    NN_model = Model(country, models[2],NN_model_name, name, independent, scaler_torch)
    Bnv_model = Model(country, models[3],Bnv_model_name, name, independent, scaler_numpy)
    Mnv_model = Model(country, models[4],Mnv_model_name, name, independent, scaler_numpy)
    SVM_model = Model(country, models[5],SVM_model_name, name, independent, scaler_numpy)


    LR_model.get_map()
    RF_model.get_map()
    Bnv_model.get_map()
    Mnv_model.get_map()
    SVM_model.get_map()
    NN_model.get_map()

    LR_model.save_joblib()
    RF_model.save_joblib()
    Bnv_model.save_joblib()
    Mnv_model.save_joblib()
    SVM_model.save_joblib()
    NN_model.save_torch()

    print('All the files are stored inside projects file with the respective project name')
