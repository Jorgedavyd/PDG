import torch.nn as nn
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np 
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

##data preprocessing
def dataframe_to_torch(dataframe, input_cols, output_cols):
    # Make a copy of the original dataframe
    dataframe1 = dataframe.copy(deep=True)
    # Extract input & outupts as numpy arrays
    inputs_array = dataframe1[input_cols].to_numpy()
    targets_array = dataframe1[output_cols].to_numpy()
    #Normalizing the dataset
    inputs_norm = StandardScaler().fit_transform(inputs_array)
    #Creating torch tensors
    inputs = torch.from_numpy(inputs_norm.astype(np.float32()))
    targets = torch.from_numpy(targets_array.astype(np.float32()))
    return inputs, targets

def scaler(dataframe, input_cols):
    # Make a copy of the original dataframe
    dataframe1 = dataframe.copy(deep=True)
    # Extract input & outupts as numpy arrays
    inputs_array = dataframe1[input_cols].to_numpy()
    #Normalizing the dataset
    preprocess = StandardScaler().fit(inputs_array)
    return preprocess

def transform(scaler, data):
    out = scaler(data)
    out = torch.from_numpy(out.astype(np.float32()))
    return out

## GPU usage

def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu') #utilizar la gpu si está disponible

def to_device(data, device):
    """Move tensor(s) to chosen device"""
    #Determina el tipo de estructura de dato, si es una lista o tupla la secciona en su subconjunto para mandar toda la información a la GPU
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl:
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl) #Mandar los data_loader que tienen todos los batch hacia la GPU


##Matplotlib plotting
#Loss
def plot_losses(history):
    losses_val = [x['val_loss'] for x in history]
    losses_train = [x['train_loss'] for x in history]
    fig, ax = plt.subplots(figsize = (7,7), dpi = 100)
    ax.plot(losses_val, marker = 'x', color = 'r', label = 'Cross-Validation' )
    ax.plot(losses_train, marker = 'o', color = 'g', label = 'Training' )
    ax.set(ylabel = 'Loss', xlabel = 'Epoch', title = 'Loss vs. No. of epochs')
    plt.legend()
    plt.show()

#AUC and ROC
def plot_roc(targets, predictions):
    fpr, tpr, thresholds = roc_curve(targets, predictions)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()

#Accuracy metric
def accuracy(outputs, targets):
    predictions = torch.round(outputs)
    accuracy_ = torch.from_numpy(np.asarray(accuracy_score(targets, predictions)).astype(np.float32()))
    return accuracy_

@torch.no_grad()

#Validation process
def evaluate(model, val_loader):
    model.eval()
    outputs = [model.validation_step(batch) for batch in val_loader]

    return model.validation_epoch_end(outputs)

def fit(epochs, lr, model, train_loader, val_loader,
                  weight_decay=0, grad_clip=False, opt_func=torch.optim.Adam):
    torch.cuda.empty_cache()
    history = [] # Seguimiento de entrenamiento

    # Poner el método de minimización personalizado
    optimizer = opt_func(model.parameters(), lr, weight_decay=weight_decay)

    for epoch in range(epochs):
        # Training Phase
        model.train()  #Activa calcular los vectores gradiente
        train_losses = []
        for batch in train_loader:
            # Calcular el costo
            loss = model.training_step(batch)
            #Seguimiento
            train_losses.append(loss)
            #Calcular las derivadas parciales
            loss.backward()

            # Gradient clipping, para que no ocurra el exploding gradient
            if grad_clip:
                nn.utils.clip_grad_value_(model.parameters(), grad_clip)

            #Efectuar el descensod e gradiente y borrar el historial
            optimizer.step()
            optimizer.zero_grad()

        # Fase de validación
        result = evaluate(model, val_loader)
        result['train_loss'] = torch.stack(train_losses).mean().item() #Stackea todos los costos de las iteraciones sobre los batches y los guarda como la pérdida general de la época
        model.epoch_end(epoch, result) #imprimir en pantalla el seguimiento
        history.append(result) # añadir a la lista el diccionario de resultados
    return history