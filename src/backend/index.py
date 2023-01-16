from flask import Flask
import pandas as pd
import torch
import torch.nn as nn
import Model.NeuralNetwork as nt

df = pd.read_csv("./Model/data/data4.csv") 

# Input e Output
input_dim = 4
output_dim = 1

# Modelo
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
    
# Loading model
    
modelo = RegressaoLinear(input_dim, output_dim)
modelo.load_state_dict(torch.load("./Model/model.pt"))
modelo.eval()

app = Flask(__name__)

@app.route('/previsao/<string:val>')
def previsao(val):
    
    global df
    
    res = [n for n in val[1:-1].split(",")]
    cn = res[0]
    ct = res[1]
    ec = res[2]
    ep = res[3]
    input_tensor = torch.FloatTensor([df.loc[(df['Cn'] == cn) & (df['Ct'] == ct), 'Cn'].values[0], df.loc[(df['Cn'] == cn) & (df['Ct'] == ct), 'Ct'].values[0]], ec, ep)
    value = modelo.forward(input_tensor).item()
    original_value = value * nt.y_train_std + nt.y_train_mean
    return str(original_value)

app.run(debug=True)