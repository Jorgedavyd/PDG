#classification module
from utils.random_utils import *
import xgboost as xgb
from torch.nn.functional import binary_cross_entropy 
from sklearn.linear_model import LogisticRegression
from torch.optim import Adam
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.model_selection import train_test_split
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from sklearn import svm
import pandas as pd

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
    def epoch_end(self, epoch, result): # Seguimiento del entrenamiento
        print("Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f},jaccard: {:.4f}, f1_score: {:.4f}, AUC: {:.4f}".format(
            epoch, result['train_loss'], result['val_loss'], result['val_acc'], result['val_jac'],result['val_f1'], result['val_auc']))
## Model module

# Here we can change the model architecture.
def  SingularLayer(input_size, output):
    out = nn.Sequential(
        nn.Linear(input_size, output),
        nn.ReLU(True)
    )
    return out

class OneClassSVMClassifier:
    def __init__(self, nu=0.05, kernel="rbf"):
        self.nu = nu  # The nu hyperparameter controls the trade-off between false positives and false negatives.
        self.kernel = kernel  # The kernel function to use (e.g., 'rbf', 'linear', 'poly')

    def train(self, X):
        # Create an OCSVM model with the specified parameters
        self.model = svm.OneClassSVM(nu=self.nu, kernel=self.kernel)

        # Fit the model on the training data (X contains only examples from the normal class)
        self.model.fit(X)

    def predict(self, X):
        # Predict whether each data point is an outlier or not
        predictions = self.model.predict(X)
        return predictions

class KMeansCluster:
    def __init__(self, n_clusters=8, max_iterations=300):
        self.n_clusters = n_clusters
        self.max_iterations = max_iterations

    def initialize_centroids(self, X):
        # Randomly select 'n_clusters' data points as initial centroids
        np.random.seed(0)  # For reproducibility
        indices = np.random.choice(X.shape[0], self.n_clusters, replace=False)
        centroids = X[indices]
        return centroids

    def assign_clusters(self, X, centroids):
        # Compute distances from each data point to centroids
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
        # Assign each data point to the nearest centroid
        labels = np.argmin(distances, axis=1)
        return labels

    def update_centroids(self, X, labels):
        # Compute new centroids as the mean of the data points in each cluster
        centroids = np.array([X[labels == k].mean(axis=0) for k in range(self.n_clusters)])
        return centroids

    def fit(self, X):
        # Step 1: Initialize centroids
        self.centroids = self.initialize_centroids(X)

        for _ in range(self.max_iterations):
            # Step 2: Assign data points to clusters
            labels = self.assign_clusters(X, self.centroids)

            # Step 3: Update centroids
            new_centroids = self.update_centroids(X, labels)

            # Check for convergence
            if np.all(new_centroids == self.centroids):
                break

            self.centroids = new_centroids

        self.labels = labels  # Store cluster assignments
        return labels

    def get_centroids(self, columns):
        return pd.DataFrame(self.centroids, columns=columns)


class NeuralNetwork(Classification, ROCplots):
    def __init__(self, name, input_size = 21, *args):
        super(NeuralNetwork, self).__init__()
        super(ROCplots, self).__init__(name)
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
    def __init__(self,name:str, n_estimators=100, random_state=43):
        super().__init__(name)
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
        return predictions

class MaximumEntropy(ROCplots):
    def __init__(self, name:str, C=1.0, random_state=42):
        super().__init__(name)
        self.C = C
        self.random_state = random_state
        self.model = LogisticRegression(C=C, random_state=random_state, solver='lbfgs', max_iter=1000)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

def test_phase(x_test,y_test, model):
    predictions = model.predict(x_test)
    predictions = np.round(predictions)
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

def train_phase(name: str, torch_data, x , y, epochs:int, lr: float,
                  weight_decay: float=0.0, grad_clip=False, opt_func=torch.optim.Adam):
    results_list = []
    # data preparation
    ## Numpy based
    x_train, x_test, y_train, y_test = train_test_split(x , y, test_size=0.2, shuffle = True, stratify=y)
    ## Pytorch based
    ###Generating dataset
    val_size = round(0.2*len(torch_data))
    train_size = len(torch_data) - val_size
    
    train_ds, val_ds = random_split(torch_data, [train_size, val_size])

    ### Generating dataloaders
    device = get_default_device()
    batch_size_train = 1
    batch_size_val = len(val_ds)

    train_loader = DataLoader(train_ds, batch_size_train, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size_val)
    train_loader = DeviceDataLoader(train_loader, device)
    val_loader = DeviceDataLoader(val_loader, device)

    # train phase
    ## 1.Maximun entropy
    print(f'\nTraining Maximum Entropy algorithm ...')
    me_model = MaximumEntropy(name)
    me_model.fit(x_train, y_train)
    result_dict = test_phase(x_test, y_test, me_model)
    results_list.append(result_dict)
    print('\nDone!')
    
    ## 2. Random Forest
    print(f'\nTraining Random Forest algorithm ...')
    rf_model = RandomForest(name)
    rf_model.fit(x_train, y_train)
    result_dict = test_phase(x_test, y_test, rf_model)
    results_list.append(result_dict)
    print('\nDone!')
    
    ## 3. Neural Network
    architecture = (64,32,16,8)
    nn_model = to_device(NeuralNetwork(name, 19, *architecture), device) ##define through jupyter notebooks
    history = fit(epochs, lr, nn_model, train_loader, val_loader, weight_decay, grad_clip, opt_func)
    print('\nDone!')
    
    #Defining project
    project = Project(name)
    #Save metrics
    project.save_metrics(history, results_list)
    nn_model.torch_roc(val_loader, device)
    project.loss_plot(history)
    rf_model.roc(x_test, y_test, 'random_forest')
    me_model.roc(x_test, y_test, 'maximum_entropy')

    return [me_model, rf_model, nn_model]