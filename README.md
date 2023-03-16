#Useful Packages

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

#Constants

weight = 0.7
bias = 0.3

start = 0
end = 1
step = 0.02

#Last points in a data set

X = torch.arange(start,end,step).unsqueeze(dim=1)
y = weight * X + bias
X[:10], y[:10]

#Length of the data sets

len(X), len(y)

#The training data

train_split = int(0.8*len(X))
train_split

#Train and testing data

X_train, y_train = X[:train_split], y[:train_split]
X_test, y_test = X[train_split:], y[train_split:]

#Defining plot function

def plot_predictions(train_data = X_train, train_labels = y_train, test_data = X_test, test_labels = y_test, 
                     predictions= None):
    
    plt.figure(figsize=(10,7))
    
    plt.scatter(train_data, train_labels, c='b', s=4, label = "Training data")
    plt.scatter(test_data, test_labels, c='g', s=4, label = "Testing Data" )
    
    if predictions is not None:
        plt.scatter(test_data, predictions, c='r', s=4, label = "Prediction")
    plt.legend()
    
    
#ploting


plot_predictions()



#Creating class

class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(1,requires_grad=True,dtype=torch.float))
        self.bias = nn.Parameter(torch.randn(1,requires_grad=True,dtype=torch.float))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weights * x + self.bias 
        
        


#Model parameters

torch.manual_seed(42)

model_0 = LinearRegressionModel()

list(model_0.parameters())


#y predicted
y_preds = model_0(X_test)
y_preds

#Using inference

with torch.inference_mode():
    y_preds = model_0(X_test)
    
y_preds


#Plot with random tensor data

plot_predictions(predictions = y_preds)

#Loss and optimization

loss_fn = nn.L1Loss()

optimizer = torch.optim.SGD(params = model_0.parameters(), lr=0.000001)

#The learning process

torch.manual_seed(42)

epochs = 1000

for epochs in range(epochs):
    model_0.train()
    
    y_pred = model_0(X_test)
    
    loss = loss_fn(y_pred, y_test)
    
    optimizer.zero_grad()
    
    loss.backward()
    
    optimizer.step()
    
    model_0.eval
    
    print(model_0.state_dict())
    
  
  

#New predicted y values
  
with torch.inference_mode():
    y_preds_new = model_0(X_test)
    
    
 
 
#New plots
plot_predictions(predictions = y_preds_new)
    
   
   
