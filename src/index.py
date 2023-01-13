import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader as web
import datetime as dt

from sklearn.preprocessing import minmax_scale
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM

empresa = 'FB'

inicio = dt.datetime(2012,1,1)
final = dt.datetime(2020,1,1)

dados = web.DataReader(empresa, 'yahoo', inicio, final)

print(dados)