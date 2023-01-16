import pandas as pd

df = pd.read_csv("E:\dataFiles\CO2EmissionFiles\data.csv")

df = df.sample(frac=0.5) #fracao dos dados
#df = df.drop(columns=[]) # remover colunas

df.to_csv("data2.csv", index=False)

print(df.head())

#['ID', 'Country', 'VFN', 'Mp', 'Mh', 'Man', 'MMS', 'Tan', 'T', 'Va', 'Ve', 'Mk', 'Cn', 'Ct', 'Cr', 'r', 'm (kg)', 'Mt', 'Enedc (g/km)', 'Ewltp (g/km)', 'W (mm)', 'At1 (mm)', 'At2 (mm)', 'Ft', 'Fm', 'ec (cm3)', 'ep (KW)', 'z (Wh/km)', 'IT', 'Ernedc (g/km)', 'Erwltp (g/km)', 'De', 'Vf', 'Status', 'year', 'Date of registration', 'Fuel consumption ', 'Electric range (km)']