######### MBA PROJECT
######### Upload to modelo, teste e validação

# Importando as libraries

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle

# Read the data from the CSV file and assign it to the variable honey_database
honey_database = pd.read_csv(r"C:\Users\millx\OneDrive\MBA DSA\honey_purity_dataset fixed.csv", delimiter=';')

# Separar as características (features) e o alvo (target)
features = honey_database.drop(columns='Price')
target = honey_database['Price']

# Dividir os dados em conjunto de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=20)

model_pkl_file = r"C:\Users\millx\OneDrive\MBA DSA\modelo_de_RF.pkl"

# load model from pickle file
with open(model_pkl_file, 'rb') as file:  
    rf_regressor = pickle.load(file)

# Realizar previsões com os dados de teste
y_pred = rf_regressor.predict(X_test)

# Calcular as métricas de avaliação
mae = mean_absolute_error(y_pred, y_test)
mse = mean_squared_error(y_pred, y_test)
r2 = r2_score(y_pred, y_test)

print("Mean Absolute Error:", mae)
print("Mean Squared Error:", mse)
print("R^2 Score:", r2)

###################################################   CRIAÇÃO DOS GRÁFICOS ABAIXO #############################################################
################################ Gráfico dos valores previstos vs reais

import matplotlib.pyplot as plt
import numpy as np

## Definição do percentual de pontos a serem exibidos (1%) para facilitação da visualização gráfica 
percentual = 0.01
num_pontos = int(len(y_test) * percentual)

## Amostrar aleatoriamente os dados para plotagem
indices_amostrados = np.random.choice(len(y_test), num_pontos, replace=False)

## Plotar um gráfico de dispersão para y_pred e y_test com cores diferentes e legendas e tamanho pré-definido
plt.figure(figsize=(10, 6))
plt.scatter(y_test.iloc[indices_amostrados], y_pred[indices_amostrados], color='blue', alpha=0.5, label='Valores Preditos (y_pred)')
plt.scatter(y_test.iloc[indices_amostrados], y_test.iloc[indices_amostrados], color='yellow', alpha=0.2, label='Valores Reais (y_test)')  
plt.title('Comparação entre y_test e y_pred (Amostra)')
plt.xlabel('y_test')
plt.ylabel('y_pred')
plt.legend()

## Salvar o gráfico como imagem e abrir
plt.savefig('comparacao_y_test_y_pred_amostra.png')
plt.show()

################################ Fim do gráfico de valores previstos vs reais

################################ Gráfico de Resíduos

import matplotlib.pyplot as plt

## Cálculo dos resíduos
residuos = y_test - y_pred

## Criação do gráfico de dispersão dos resíduos com rótulos e título 
plt.figure(figsize=(10, 6))
plt.scatter(y_pred, residuos, color='blue', alpha=0.7)
plt.axhline(y=0, color='red', linestyle='--')
plt.title('Gráfico de Resíduos')
plt.xlabel('Valores Previstos')
plt.ylabel('Resíduos')
plt.show()

################################ Fim do Gráfico de Resíduos

################################ Gráfico da Matrix de Correlação

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Carregar os dados
honey_database = pd.read_csv(r"C:\Users\millx\OneDrive\MBA DSA\honey_purity_dataset fixed.csv",delimiter=';')

# Calcular a matriz de correlação
correlation_matrix = honey_database.corr()

# Visualizar a matriz de correlação em um mapa de calor
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Matriz de Correlação')
plt.show()

################################ Fim do Gráfico da Matrix de Correlação

################################ Gráfico de barras
import pandas as pd
import matplotlib.pyplot as plt

# Carregar os dados
honey_database = pd.read_csv(r"C:\Users\millx\OneDrive\MBA DSA\honey_purity_dataset fixed.csv",delimiter=';')

# Calcular a correlação entre cada variável independente e o preço
correlation_with_price = honey_database.drop(columns='Price').corrwith(honey_database['Price'])

# Ordenar as correlações em ordem decrescente
correlation_with_price = correlation_with_price.abs().sort_values(ascending=False)

# Plotar o gráfico de barras
plt.figure(figsize=(10, 6))
correlation_with_price.plot(kind='bar', color='blue')
plt.title('Correlação entre Variáveis e Preço do Mel')
plt.xlabel('Variável Independente')
plt.ylabel('Correlação Absoluta com o Preço')
plt.xticks(rotation=45)
plt.show()
################################ fim do gráfico de barras