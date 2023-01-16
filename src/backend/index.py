from flask import Flask
import torch
import torch.nn as nn
import Model.NeuralNetwork as nt

# Input e Output
input_dim = 3
output_dim = 1

# Modelo
class RegressaoLinear(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(RegressaoLinear, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        return self.linear(x)
    
# Loading model
    
modelo = RegressaoLinear(input_dim, output_dim)
modelo.load_state_dict(torch.load("./Model/model.pt"))
modelo.eval()

app = Flask(__name__)

@app.route('/previsao/<val>')
def previsao(val):
    
    global input_dim, output_dim
    
    res = [float(n) for n in val[1:-1].split(",")]
    tensor = torch.FloatTensor(res)
    
    value = modelo.forward(tensor).item()
    original_value = value * nt.y_train_std + nt.y_train_mean
    return str(original_value)

app.run()