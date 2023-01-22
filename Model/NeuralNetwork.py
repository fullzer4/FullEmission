import pandas as pd
import torch
import torch.nn as nn
from torch import device
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from torch.utils.data import TensorDataset, DataLoader

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using: {device}')

# Prepare the data
df = pd.read_csv("./data/data.csv") 
x = df[['m (kg)','Mt','ec (cm3)','ep (KW)','Fuel consumption']].values
y = df[["Ewltp (g/km)", 'Enedc (g/km)']].values

# Splitting the data into training, validation, and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Normalizing the data
x_train_mean = x_train.mean(axis=0)
x_train_std = x_train.std(axis=0)
x_train = (x_train - x_train_mean) / x_train_std
x_test = (x_test - x_train_mean) / x_train_std

# Convert to tensors
x_train = torch.from_numpy(x_train).float().to(device)
y_train = torch.from_numpy(y_train).float().to(device)
x_test = torch.from_numpy(x_test).float().to(device)
y_test = torch.from_numpy(y_test).float().to(device)

# Define the model
class NeuralNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Initialize the model
input_dim = 5
hidden_dim = 64
output_dim = 2
model = NeuralNetwork(input_dim, hidden_dim, output_dim).to(device)

# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

# Create TensorDatasets for training and testing
train_data = TensorDataset(x_train, y_train)
test_data = TensorDataset(x_test, y_test)

# Define the batch size
batch_size = 32

# Create DataLoaders for training and testing
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

# Train the model
num_epochs = 1500
for epoch in range(num_epochs):
    for i, (x_batch, y_batch) in enumerate(train_loader):
        # Forward pass
        outputs = model(x_batch)
        loss = criterion(outputs, y_batch)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
            
# Test the model
with torch.no_grad():
    total_loss = 0
    total_mse = 0
    total_r2 = 0
    total_len = len(test_loader)
    for i, (x_batch, y_batch) in enumerate(test_loader):
        # reshape y_batch to [batch_size, 1]
        y_batch = y_batch.reshape(-1,2)
        outputs = model(x_batch)
        loss = criterion(outputs, y_batch)
        total_loss += loss.item()
        total_mse += mean_squared_error(y_batch.cpu(), outputs.cpu())
        total_r2 += r2_score(y_batch.cpu(), outputs.cpu())
    print(f'R²: {total_r2/total_len:.4f}, MSE: {total_mse/total_len:.4f} and Loss: {total_loss/total_len:.4f}')
    
torch.save(model.state_dict(), 'model.pt')