import torch.nn as nn
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np 
from sklearn.metrics import accuracy_score
from sklearn import metrics
from tqdm import tqdm
import os

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


#Accuracy metric
def accuracy(outputs, targets):
    predictions = torch.round(outputs)
    accuracy_ = torch.from_numpy(np.asarray(accuracy_score(targets, predictions)).astype(np.float32()))
    return accuracy_
#Jaccard metric
def jaccard(outputs, targets):
    predictions = torch.round(outputs)
    jaccard_ = torch.from_numpy(np.asarray(metrics.jaccard_score(targets, predictions)).astype(np.float32()))
    return jaccard_
#f1 metric
def f1(outputs, targets):
    predictions = torch.round(outputs)
    f1_ = torch.from_numpy(np.asarray(metrics.f1_score(targets,predictions)).astype(np.float32()))
    return f1_
#AUC 
def AUC(outputs, targets):
    predictions = torch.round(outputs)
    fpr, tpr, _ = metrics.roc_curve(targets,predictions)
    auc_ = torch.from_numpy(np.asarray(metrics.auc(fpr, tpr)).astype(np.float32()))
    return auc_

@torch.no_grad()
#Validation process
def evaluate(model, val_loader):
    model.eval()
    outputs = [model.validation_step(batch) for batch in val_loader]

    return model.validation_epoch_end(outputs)
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr'] # Seguimiento del learning rate
    
def fit(epochs, lr, model, train_loader, val_loader,
                  weight_decay=0, grad_clip=False, opt_func=torch.optim.Adam):
    torch.cuda.empty_cache()
    history = [] # Seguimiento de entrenamiento
    
    optimizer = opt_func(model.parameters(), lr, weight_decay=weight_decay)
    
    sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, lr, epochs=epochs,
                                                steps_per_epoch=len(train_loader))

    for epoch in tqdm(range(epochs), total = epochs, desc = 'Training Neural Network ...'):
        # Training Phase
        model.train()  
        train_losses = []
        lrs = []
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

            lrs.append(get_lr(optimizer))
            sched.step()


        # Fase de validación
        result = evaluate(model, val_loader)
        result['train_loss'] = torch.stack(train_losses).mean().item() #Stackea todos los costos de las iteraciones sobre los batches y los guarda como la pérdida general de la época
        result['lrs'] = lrs
        history.append(result) # añadir a la lista el diccionario de resultados
    
    return history
class Project():
    def __init__(self, name: str):
        self.name = name
    def save_metrics(self,history, results_list):
        os.makedirs('projects/'+self.name+'/metrics/', exist_ok=True)
        with open('projects/'+self.name+'/metrics/neural_network.csv', 'w') as file:
            file.write('Epoch,Training_loss,Validation_loss,jaccard,f1_score,accuracy,auc\n')
            for epoch, data in tqdm(enumerate(history), desc = 'Saving Neural Network Metrics ...', total = len(history)):
                training_loss = data['train_loss']
                validation_loss = data['val_loss']
                jaccard_metric = data['val_jac']
                f1_score_ = data['val_f1']
                accuracy_ = data['val_acc']
                area_under_curve = data['val_auc']
                file.write(f'{epoch+1},{training_loss},{validation_loss},{jaccard_metric},{f1_score_},{accuracy_},{area_under_curve}\n')
            print('Done!')
        with open('projects/'+self.name+'/metrics/maximum_entropy.csv', 'w') as file:
            print('Saving Maximum Entropy metrics ...')
            loss_ = results_list[0]['loss']
            jaccard_ = results_list[0]['jaccard']
            f1_ = results_list[0]['f1']
            acc_ = results_list[0]['accuracy']
            auc_ = results_list[0]['auc']
            file.write(f'loss,jaccard,f1_score,accuracy,auc\n{loss_},{jaccard_},{f1_},{acc_},{auc_}')
            print('Done!')
        with open('projects/'+self.name+'/metrics/random_forest.csv', 'w') as file:
            print('Saving Random Forest metrics ...')
            loss_ = results_list[1]['loss']
            jaccard_ = results_list[1]['jaccard']
            f1_ = results_list[1]['f1']
            acc_ = results_list[1]['accuracy']
            auc_ = results_list[1]['auc']
            file.write(f'loss,jaccard,f1_score,accuracy,auc\n{loss_},{jaccard_},{f1_},{acc_},{auc_}')
            print('Done!')
    def loss_plot(self, history):
        losses_val = [x['val_loss'] for x in history]
        losses_train = [x['train_loss'] for x in history]
        _, ax = plt.subplots(figsize = (7,7), dpi = 100)
        ax.plot(losses_val, marker = 'x', color = 'r', label = 'Cross-Validation' )
        ax.plot(losses_train, marker = 'o', color = 'g', label = 'Training' )
        ax.set(ylabel = 'Loss', xlabel = 'Epoch', title = 'Loss vs. No. of epochs')
        plt.legend()
        plt.savefig('projects/'+self.name+'/metrics/nn_model_loss.png')
        plt.close()


class ROCplots(Project):
    def __init__(self, name: str):
        super().__init__(name)
    def torch_roc(self, test_loader, device=get_default_device()):
        self.eval()
        y_true = []
        y_scores = []

        with torch.no_grad():
            for data, labels in test_loader:
                data, labels = to_device(data, device), to_device(labels, device)
                outputs = self(data)
                y_true.extend(labels.cpu().numpy())
                y_scores.extend(outputs.cpu().numpy())

        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)

        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")

        plt.savefig('projects/'+self.name+'/metrics/nn_model_roc.png')
        plt.close()
    def roc(self, x, targets, model_name:str):
        predictions = self.predict(x)
        fpr, tpr, _ = roc_curve(targets, predictions)
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
        plt.savefig('projects/'+self.name+'/metrics/'+model_name+'_roc.png')
        plt.close()
    
