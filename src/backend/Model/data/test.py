import pandas as pd

df = pd.read_csv("./data5.csv")

df = df.sample(frac=0.30) #fracao dos dados
#df = df.drop(columns=['ID', 'Country', 'VFN', 'Mp', 'Mh', 'Man', 'MMS', 'Tan', 'T', 'Va', 'Ve', 'Mk', 'Cn', 'Ct', 'Cr', 'r', 'Enedc (g/km)', 'Ft', 'Fm', 'z (Wh/km)', 'IT', 'Ernedc (g/km)', 'Erwltp (g/km)', 'De', 'Vf', 'Status', 'year', 'Date of registration', 'Electric range (km)']) # remover colunas
#df = df.dropna() # limpa linhas vazias
df.to_csv("data6.csv", index=False) # salvar

print(df.head())

#['ID', 'Country', 'VFN', 'Mp', 'Mh', 'Man', 'MMS', 'Tan', 'T', 'Va', 'Ve', 'Mk', 'Cn', 'Ct', 'Cr', 'r', 'm (kg)', 'Mt', 'Enedc (g/km)', 'Ewltp (g/km)', 'W (mm)', 'At1 (mm)', 'At2 (mm)', 'Ft', 'Fm', 'ec (cm3)', 'ep (KW)', 'z (Wh/km)', 'IT', 'Ernedc (g/km)', 'Erwltp (g/km)', 'De', 'Vf', 'Status', 'year', 'Date of registration', 'Fuel consumption ', 'Electric range (km)']

# para usar: m (kg) Mt  Ewltp (g/km)  W (mm)  At1 (mm)  At2 (mm)  ec (cm3)  ep (KW)  Fuel consumption 

