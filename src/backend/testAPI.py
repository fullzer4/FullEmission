import requests

data = {'inputs': [1875.0, 2022.0, 1598.0, 147.0, 5.5]}
response = requests.post('http://localhost:5000/predict', json=data)
print(response.json())