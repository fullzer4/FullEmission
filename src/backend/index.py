import torch
import torch.nn as nn
from flask  import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd

app = Flask(__name__)
CORS(app)

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

# Load the saved model

df = pd.read_csv("./Model/data/data.csv") 
x_train_mean = df[['m (kg)','Mt','ec (cm3)','ep (KW)','Fuel consumption']].mean()
x_train_std = df[['m (kg)','Mt','ec (cm3)','ep (KW)','Fuel consumption']].std()

x_train_mean = torch.tensor(x_train_mean, dtype=torch.float)
x_train_std = torch.tensor(x_train_std, dtype=torch.float)

input_dim = 5
hidden_dim = 64
output_dim = 2
model = NeuralNetwork(input_dim, hidden_dim, output_dim)
model.load_state_dict(torch.load('model.pt'))
model.eval()

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    inputs = torch.tensor(data['inputs']).float()
    inputs = (inputs - x_train_mean.unsqueeze(0)) / x_train_std.unsqueeze(0)
    prediction = model(inputs)
    output = {'prediction': prediction.tolist()}
    return jsonify(output)

if __name__ == '__main__':
    app.run(debug=True)