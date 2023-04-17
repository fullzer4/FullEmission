import torch
import torch.nn as nn
from flask  import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import json

# criando servidor flask
app = Flask(__name__)
CORS(app)

# modelo da rede neural
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

# abrindo os valores salvos anteriomente e retratando eles
with open('values.json', 'r') as f:
    data = json.load(f)
x_train_mean = torch.tensor(data['x_train_mean'], dtype=torch.float)
x_train_std = torch.tensor(data['x_train_std'], dtype=torch.float)

#super parametros
input_dim = 5
hidden_dim = 64
output_dim = 2
model = NeuralNetwork(input_dim, hidden_dim, output_dim)
model.load_state_dict(torch.load('model.pt'))
model.eval()

# rota para prever
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    inputs = torch.tensor(data['inputs']).float()
    inputs = (inputs - x_train_mean.unsqueeze(0)) / x_train_std.unsqueeze(0)
    prediction = model(inputs)
    output = {'prediction': prediction.tolist()}
    return jsonify(output)

# abrir o app para o host
if __name__ == '__main__':
    app.run(host='0.0.0.0')