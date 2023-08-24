#classification module
from random_utils import *
import xgboost as xgb
from torch.nn.functional import binary_cross_entropy 
from sklearn.linear_model import LogisticRegression
from torch.optim import Adam
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.model_selection import train_test_split
from torch.utils.data import random_split
from torch.utils.data import DataLoader

class Classification(nn.Module):
    def training_step(self, batch):
        inputs, targets = batch
        # Reshape target tensor to match the input size
        target_tensor = targets.unsqueeze(1)  # Add a new dimension
        target_tensor = target_tensor.expand(-1, 1)  # Duplicate values across second dimension
        out = self(inputs)                  # Generar predicciones
        loss = binary_cross_entropy(out, target_tensor) # Calcular el costo
        return loss

    def validation_step(self, batch):
        inputs, targets = batch
        target_tensor = targets.unsqueeze(1)  # Add a new dimension
        target_tensor = target_tensor.expand(-1, 1)  # Duplicate values across second dimension
        out = self(inputs)                    # Generar predicciones
        loss = binary_cross_entropy(out, target_tensor)   # Calcular el costo
        acc = accuracy(out, targets) #Calcular la precisión
        jac = jaccard(out, targets) #jaccard metric
        arm_score = f1(out, targets) #f1_score
        area = AUC(out, targets) # Area under the curve
        return {'val_loss': loss.detach(), 'val_acc': acc, 'val_jac': jac, 'val_f1': arm_score, 'val_auc': area}

    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Sacar el valor expectado de todo el conjunto de costos
        batch_acc = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_acc).mean()   # Sacar el valor expectado de todo el conjunto de precisión
        batch_jaccard = [x['val_jac'] for x in outputs]
        epoch_jaccard = torch.stack(batch_jaccard).mean()   # Sacar el valor expectado de todo el conjunto de precisión
        batch_f1 = [x['val_f1'] for x in outputs]
        epoch_f1 = torch.stack(batch_f1).mean()   # Sacar el valor expectado de todo el conjunto de precisión
        batch_auc = [x['val_auc'] for x in outputs]
        epoch_auc = torch.stack(batch_auc).mean()   # Sacar el valor expectado de todo el conjunto de precisión
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item(), 'val_jac': epoch_jaccard.item(), 'val_f1': epoch_f1.item(), 'val_auc': epoch_auc.item()}

## Model module

# Here we can change the model architecture.
def  SingularLayer(input_size, output):
    out = nn.Sequential(
        nn.Linear(input_size, output),
        nn.ReLU(True)
    )
    return out

class NeuralNetwork(Classification, ROCplots):
    def __init__(self, input_size = 21, *args):
        super(NeuralNetwork, self).__init__()
        
        self.overall_structure = nn.Sequential()
        #Model input and hidden layer
        for num, output in enumerate(args):
            self.overall_structure.add_module(name = f'layer_{num+1}', module = SingularLayer(input_size, output))
            input_size = output

        #Model output layer
        self.output_layer = nn.Sequential(
                nn.Linear(input_size, 1),
                nn.Sigmoid()
            )
        
    def forward(self, xb):
        out = self.overall_structure(xb)
        out = self.output_layer(out)
        return out
    def predict(self, xb):
        return self(xb)

class RandomForest(ROCplots):
    def __init__(self, n_estimators=100, random_state=None):
        self(RandomForest, self).__init__()
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.model = None

    def fit(self, X, y):
        dtrain = xgb.DMatrix(X, label=y)
        self.model = xgb.train({'objective': 'binary:logistic',
                                'eval_metric': 'logloss',
                                'seed': self.random_state},
                               dtrain,
                               num_boost_round=self.n_estimators)

    def predict(self, X):
        dtest = xgb.DMatrix(X)
        predictions = self.model.predict(dtest)
        return np.mean(predictions)

class MaximumEntropy(ROCplots):
    def __init__(self, C=1.0, random_state=None):
        super(MaximumEntropy, self).__init__()
        self.C = C
        self.random_state = random_state
        self.model = LogisticRegression(C=C, random_state=random_state, solver='lbfgs', max_iter=1000)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

def test_phase(x_test,y_test, model):
    predictions = model.predict(x_test)
    
    #metrics and losses
    Accuracy_Score = metrics.accuracy_score(y_test, predictions)
    JaccardIndex = metrics.jaccard_score(y_test, predictions)
    F1_Score = metrics.f1_score(y_test, predictions)
    Log_Loss = metrics.log_loss(y_test, predictions)
    fpr, tpr, _ = metrics.roc_curve(y_test, predictions)
    auc_=metrics.auc(fpr, tpr)
    results_dict = {'accuracy': Accuracy_Score,
                 'jaccard': JaccardIndex,
                 'f1': F1_Score,
                 'loss': Log_Loss,
                 'auc': auc_}
    #ROC figure
    return results_dict

###Define the generalized hyperparameters

def train_phase(torch_data, batch_size, numpy_data, epochs, lr,
                  weight_decay=0, grad_clip=False, opt_func=torch.optim.Adam):
    results_list = []
    # data preparation
    ## Numpy based
    x_train, y_train, x_test, y_test = train_test_split(*numpy_data, test_size=0.2, shuffle = True)
    
    ## Pytorch based
    ###Generating dataset
    batch_size = batch_size
    val_size = round(0.2*len(torch_data))
    train_size = len(torch_data) - val_size
    
    train_ds, val_ds = random_split(torch_data, [train_size, val_size])

    ### Generating dataloaders
    device = get_default_device()
    
    train_loader = DataLoader(train_ds, batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size*2)
    train_loader = DeviceDataLoader(train_loader, device)
    val_loader = DeviceDataLoader(val_loader, device)

    # train phase
    ## 1.Maximun entropy
    print(f'\nTraining Maximum Entropy algorithm ...')
    me_model = MaximumEntropy().fit(x_train, y_train)
    result_dict = test_phase(x_test, y_test, me_model)
    results_list.append(result_dict)
    print('\nDone!')
    
    ## 2. Random Forest
    print(f'\nTraining Random Forest algorithm ...')
    rf_model = RandomForest().fit(x_train, y_train)
    result_dict = test_phase(x_test, y_test, rf_model)
    results_list.append(result_dict)
    print('\nDone!')
    
    ## 3. Neural Network
    nn_model = to_device(NeuralNetwork(19, (3,4,5)), device) ##define through jupyter notebooks
    history = fit(epochs, lr, nn_model, train_loader, val_loader, weight_decay, grad_clip, opt_func)
    print('\nDone!')
    
    #Defining project
    project = Project(input('Name of this project: '))
    #Save metrics
    project.save_metrics(history, results_list)
    nn_model.torch_roc(val_loader, device)
    project.loss_plot(history)
    rf_model.roc(x_test, y_test, 'random_forest')
    me_model.roc(x_test, y_test, 'maximum_entropy')

    return [me_model, rf_model, nn_model]