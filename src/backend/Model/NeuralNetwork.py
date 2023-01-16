import pandas as pd
import torch
import torch.nn as nn

# Preparando os dados
df = pd.read_csv("./Model/data/data4.csv") # Avg Engine Power (kW) Avg Engine Capacity (cm3)

# Remover linhas com valores nulos
df = df.dropna()
df['Cn'] = df['Cn'].factorize()[0]
df['Ct'] = df['Ct'].factorize()[0]
x_train = df[["ec (cm3)", "Cn", "Ct"]].values
y_train = df["Ewltp (g/km)"].values

y_train_mean = y_train.mean()
y_train_std = y_train.std()

# Normalizando os dados
x_train = (x_train - x_train.mean()) / x_train.std()
y_train = (y_train - y_train.mean()) / y_train.std()

# Convertendo para tensores
x_train = torch.from_numpy(x_train).float()
y_train = torch.from_numpy(y_train).float()

# Input e Output
input_dim = 4
output_dim = 1

# Model neural
class RegressaoLinear(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(RegressaoLinear, self).__init__()
        self.linear1 = nn.Linear(input_dim, 8)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(8, 8)
        self.relu2 = nn.ReLU()
        self.linear3 = nn.Linear(8, output_dim)
    def forward(self, x):
        x = self.linear1(x)
        x = self.relu1(x)
        x = self.linear2(x)
        x = self.relu2(x)
        x = self.linear3(x)
        return x

model = RegressaoLinear(input_dim, output_dim)

# Loss / OLptimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Loop de treino do modelo
epochs = 1500

for epoch in range(epochs):
    optimizer.zero_grad()
    y_pred = model(x_train)
    loss = criterion(y_pred, y_train)
    loss.backward()
    optimizer.step()
    if (epoch+1) % 10 == 0:
        print(f'epoch {epoch+1}, loss = {loss.item()}')

torch.save(model.state_dict(), 'model.pt')