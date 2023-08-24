from utils.preprocessing import *
from utils.pseudo_abscence import *
from utils.heatmap import *
from utils.random_utils import *
from utils.models import *

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
    # Ask for mode, perform data preprocessing
    url = URLs.url600s
    if program_init():
        down_boundary = float(input('Min radius:'))
        up_boundary = float(input('Max radius:'))
        torch_dataset, numpy_data = data_preprocess_with_pseudo(url, down_boundary, up_boundary)
    else:
        torch_dataset, numpy_data = data_preprocess_without_pseudo(url)
    
    # Training phase

    # Neural network parameters
    batch_size = ...
    epochs = ...
    lr = ...
    weight_decay = ...
    grad_clip = ...
    opt_func = ...

    models = train_phase(torch_dataset, batch_size, numpy_data, epochs, lr, weight_decay, grad_clip, opt_func)
    
    ...