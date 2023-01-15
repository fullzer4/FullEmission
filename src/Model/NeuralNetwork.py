import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# Preparando os dados
df = pd.read_csv("./FuelConsumption.csv")

x_train = df[["ENGINESIZE", "CYLINDERS", "FUELCONSUMPTION_COMB"]].values
y_train = df["CO2EMISSIONS"].values

# Normalizando os dados
x_train = (x_train - x_train.mean()) / x_train.std()
y_train = (y_train - y_train.mean()) / y_train.std()

# Convertendo para tensores
x_train = torch.from_numpy(x_train).float()
y_train = torch.from_numpy(y_train).float()

# Input e Output
input_dim = 3
output_dim = 1

# Model neural
class RegressaoLinear(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(RegressaoLinear, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        return self.linear(x)

model = RegressaoLinear(input_dim, output_dim)

# Loss / OLptimizer
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

# Loop de treino do modelo
epochs = 500000

for epoch in range(epochs):
    optimizer.zero_grad()
    y_pred = model(x_train)
    loss = criterion(y_pred, y_train)
    loss.backward()
    optimizer.step()
    if (epoch+1) % 10 == 0:
        print(f'epoch {epoch+1}, loss = {loss.item()}')
    if torch.isnan(loss):
        break

# Plotando os dados
plt.scatter(x_train[:, 0], y_train)
plt.xlabel("Engine size")
plt.ylabel("CO2 emissions")

# Regression line
with torch.no_grad():
    w = model.linear.weight.view(-1)
    b = model.linear.bias
    x = torch.linspace(min(x_train[:,0]), max(x_train[:,0]), 100)
    y = x * w[0] + b
    plt.plot(x, y, 'r')

plt.show()

torch.save(model.state_dict(), 'model.pt')