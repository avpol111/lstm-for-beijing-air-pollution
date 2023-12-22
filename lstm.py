import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
 
df = pd.read_csv('Beijing_PM.csv')
dataset = df[["pm2.5"]]
dataset.fillna(0, inplace=True)
dataset = dataset[24:]
timeseries = dataset.values.astype('float32')

train_size = int(len(timeseries) * 0.67)
test_size = len(timeseries) - train_size
train, test = timeseries[:train_size], timeseries[train_size:]

def create_dataset(dataset, lookback):
    X, y = [], []
    for i in range(len(dataset)-lookback):
        feature = dataset[i:i+lookback]
        target = dataset[i+1:i+lookback+1]
        X.append(feature)
        y.append(target)
    return torch.tensor(X), torch.tensor(y)

lookback = 5 # a window with the best RMSE
X_train, y_train = create_dataset(train, lookback=lookback)
X_test, y_test = create_dataset(test, lookback=lookback)

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=50, num_layers=1, batch_first=True)
        self.linear = nn.Linear(50, 1)
    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.linear(x)
        return x

model = Net()
optimizer = optim.Adam(model.parameters())
loss_fn = nn.MSELoss()
loader = data.DataLoader(data.TensorDataset(X_train, y_train), shuffle=True, batch_size=8)

n_epochs = 100 # 50 would suffice, too, but just in case
for epoch in range(n_epochs):
    model.train()
    for X_batch, y_batch in loader:
        y_pred = model(X_batch)
        loss = loss_fn(y_pred, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    if epoch % 10 != 0:
        continue
    model.eval()
    with torch.no_grad():
        y_pred = model(X_train)
        train_rmse = np.sqrt(loss_fn(y_pred, y_train))
        y_pred = model(X_test)
        test_rmse = np.sqrt(loss_fn(y_pred, y_test))
    print("Epoch %d: train RMSE %.4f, test RMSE %.4f" % (epoch, train_rmse, test_rmse))
    
'''
Epoch 0: train RMSE 67.9972, test RMSE 62.0409
Epoch 10: train RMSE 30.4910, test RMSE 25.1017
Epoch 20: train RMSE 28.3982, test RMSE 23.4498
Epoch 30: train RMSE 29.0897, test RMSE 24.0010
Epoch 40: train RMSE 28.5213, test RMSE 23.5750
Epoch 50: train RMSE 29.3950, test RMSE 24.3000
Epoch 60: train RMSE 29.2136, test RMSE 24.0882
Epoch 70: train RMSE 28.6480, test RMSE 23.7887
Epoch 80: train RMSE 28.6550, test RMSE 23.8411
Epoch 90: train RMSE 28.0076, test RMSE 23.1665
'''