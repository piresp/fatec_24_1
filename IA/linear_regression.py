from matplotlib import pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np
from sklearn import linear_model
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from math import sqrt

df = pd.read_csv('database.csv')

print(df.head())

print(df.describe())

motores = df[['area']]
co2 = df[['preco']]

motores_treino, motores_test, co2_treino, co2_teste = train_test_split(motores, co2, test_size=0.2, random_state=42)

plt.scatter(motores_treino, co2_treino, color= 'blue')
plt.xlabel('Motor')
plt.ylabel('Emissão de CO2')
plt.show()

modelo = linear_model.LinearRegression()

modelo.fit(motores_treino, co2_treino)

print('(A) Intercepto: ', modelo.Intercept_)
print('(B) Inclinação: ', modelo.coef_)

plt.scatter(motores_treino, co2_treino, color='blue')
plt.plot(motores_treino, modelo.coef_[0][0]*motores_treino + modelo.intercept_[0], '-r')
plt.ylabel('Emissão de CO2')
plt.xlabel('Motores')
plt.show

predicoesCo2 = modelo.predict(motores_test)

plt.scatter(motores_teste, co2_teste, color='blue')
plt.plot(motores_teste, modelo.coef_[0][0]*motores_teste + modelo.intercept_[0], '-r')
plt.ylabel('Emissão de CO2')
plt.xlabel('Motores')
plt.show()
