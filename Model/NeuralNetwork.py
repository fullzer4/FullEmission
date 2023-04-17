import pandas as pd
import torch
import torch.nn as nn
from torch import device
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from torch.utils.data import TensorDataset, DataLoader
import json

# ativar modo gpu comentar linha 12 e descomentar a abaixo
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
print(f'Using: {device}')

# preparando dados e separando valores para o modelo
df = pd.read_csv("./data/data.csv") 
x = df[['m (kg)','Mt','ec (cm3)','ep (KW)','Fuel consumption']].values
y = df[["Ewltp (g/km)", 'Enedc (g/km)']].values

# separando os dados em ocnjuntos de teste e treino
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# tirando a media, o desvio padrao e normalizando os valores
x_train_mean = x_train.mean(axis=0)
x_train_std = x_train.std(axis=0)
x_train = (x_train - x_train_mean) / x_train_std
x_test = (x_test - x_train_mean) / x_train_std

# alocando os nossos dados em tensores
x_train = torch.from_numpy(x_train).float().to(device)
y_train = torch.from_numpy(y_train).float().to(device)
x_test = torch.from_numpy(x_test).float().to(device)
y_test = torch.from_numpy(y_test).float().to(device)

# salvar dados no json da media e do desvio padrao para usar futuramente
vars = {"x_train_mean": x_train_mean.tolist(), "x_train_std": x_train_std.tolist()}

with open("../backend/values.json", "w") as f:
    json.dump(vars, f)

# classe da rede neural
class NeuralNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_dim2, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x

# definindo super parametros
input_dim = 5
hidden_dim1 = 64
hidden_dim2 = 64
output_dim = 2
model = NeuralNetwork(input_dim, hidden_dim1, hidden_dim2, output_dim).to(device)

# definindo parametros para avaliacao do modelo
criterion = nn.MSELoss()
reg = nn.L1Loss()

def regularization(model, reg_lambda):
    reg_loss = 0
    for param in model.parameters():
        reg_loss += reg(param, torch.zeros_like(param))
    return reg_loss * reg_lambda

# taxa de aprendizado
reg_lambda = 0.001
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

train_data = TensorDataset(x_train, y_train)
test_data = TensorDataset(x_test, y_test)

batch_size = 64 # 32

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

#numero de epochs ou seja vezes que vai treinar a rede
num_epochs = 15000
for epoch in range(num_epochs):
    for i, (x_batch, y_batch) in enumerate(train_loader):

        outputs = model(x_batch)
        loss = criterion(outputs, y_batch)
        reg_loss = regulariz=ation(model, reg_lambda)
        loss += reg_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

with torch.no_grad():
    total_loss = 0
    total_mse = 0
    total_r2 = 0
    total_len = len(test_loader)
    for i, (x_batch, y_batch) in enumerate(test_loader):
        y_batch = y_batch.reshape(-1,2)
        outputs = model(x_batch)
        loss = criterion(outputs, y_batch)
        total_loss += loss.item()
        total_mse += mean_squared_error(y_batch.cpu(), outputs.cpu())
        total_r2 += r2_score(y_batch.cpu(), outputs.cpu())
    print(f'R²: {total_r2/total_len:.4f}, MSE: {total_mse/total_len:.4f} and Loss: {total_loss/total_len:.4f}')
    
torch.save(model.state_dict(), 'model.pt')