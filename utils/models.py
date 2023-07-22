#classification module
from random_utils import *
import xgboost as xgb
from torch.nn.functional import binary_cross_entropy 
from sklearn.linear_model import LogisticRegression
from torch.optim import Adam
from sklearn.metrics import accuracy_score

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
        return {'val_loss': loss.detach(), 'val_acc': acc}

    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Sacar el valor expectado de todo el conjunto de costos
        batch_acc = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_acc).mean()   # Sacar el valor expectado de todo el conjunto de precisión
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

    def epoch_end(self, epoch, result): # Seguimiento del entrenamiento
        print("Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch, result['train_loss'], result['val_loss'], result['val_acc']))
## Model module

# Here we can change the model architecture.
def  SingularLayer(input_size, output):
    out = nn.Sequential(
        nn.Linear(input_size, output),
        nn.ReLU(True)
    )
    return out

class NeuralNetwork(Classification):
    def __init__(self, input_size = 19, *args):
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
    
    def train(self,epochs, lr, loader, opt_func = Adam):
        opt = opt_func(self.parameters(), lr = lr)
        for epoch in range(epochs):
            for batch in loader:    
                #get the loss
                loss = self.training_step(batch)
                #take gradients
                loss.backward()
                #do the gradient descent step
                opt.step()
                #clear the gradients
                opt.zero_grad()

class RandomForest:
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

    def evaluate(self, X, y):
        predictions = self.predict(X)
        accuracy = accuracy_score(y, predictions)
        return accuracy
    

class MaximumEntropy:
    def __init__(self, C=1.0, random_state=None):
        super(MaximumEntropy, self).__init__()
        self.C = C
        self.random_state = random_state
        self.model = LogisticRegression(C=C, random_state=random_state, solver='lbfgs', max_iter=1000)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, X, y):
        predictions = self.model.predict(X)
        accuracy = accuracy_score(y, predictions)
        return accuracy
    
