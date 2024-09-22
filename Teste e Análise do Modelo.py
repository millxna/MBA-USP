######### MBA PROJECT
######### Upload do modelo, teste e validação

# Importando as libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
import numpy as np

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

# Gerar a curva de aprendizado
train_sizes, train_scores, test_scores = learning_curve(
    rf_regressor, X_train, y_train, cv=5, scoring='r2', n_jobs=-1, 
    train_sizes=np.linspace(0.1, 1.0, 10)
)

train_scores_mean = np.mean(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)

plt.plot(train_sizes, train_scores_mean, label="Training Score")
plt.plot(train_sizes, test_scores_mean, label="Cross-Validation Score")
plt.xlabel("Training Set Size")
plt.ylabel("R^2 Score")
plt.legend()
plt.title("Learning Curve")
plt.show()










#############   CRIAÇÃO DOS GRÁFICOS ABAIXO 
########### Gráfico dos valores previstos vs reais

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

######## Fim do gráfico de valores previstos vs reais

########Gráfico de Resíduos

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

############ Fim do Gráfico de Resíduos

########### Gráfico da Matrix de Correlação

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

########### Fim do Gráfico da Matrix de Correlação

############Gráfico de barras
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
########### fim do gráfico de barras






################## ANALISE PREDITIVA SEM A CRIAÇÃO DE CENÁRIOS - Visualizar o que o modelo prevê
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 1. Carregar a base de dados original
honey_database = pd.read_csv(r"C:\Users\millx\OneDrive\MBA DSA\honey_purity_dataset fixed.csv", delimiter=';')

# 2. Separar as características (features) e o alvo (target)
features = honey_database.drop(columns='Price')

# 3. Carregar o modelo de Random Forest treinado
with open(r'C:\Users\millx\OneDrive\MBA DSA\modelo_de_RF.pkl', 'rb') as file:
    modelo = pickle.load(file)

# 4. Prever os preços para a base de dados original
honey_database['Predicted_Price'] = modelo.predict(features)

# 5. Identificar a linha com o maior preço previsto
max_price_idx = honey_database['Predicted_Price'].idxmax()
max_price_scenario = honey_database.iloc[max_price_idx]

# Exibir a combinação ótima e o maior preço previsto
print(f"Melhor combinação de variáveis na base original:\n{max_price_scenario}")

# 6. Criar o gráfico 3D para visualizar a previsão com Pureza, Tipo de Flor e Preço Previsto
fig = plt.figure(figsize=(15, 10))

# Plotar Pureza vs Tipo de Flor vs Preço Previsto
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter(honey_database['Purity'], honey_database['FlowerTypeNumber'], honey_database['Predicted_Price'], 
                c=honey_database['Predicted_Price'], cmap='viridis')

ax.set_xlabel('Purity')
ax.set_ylabel('Flower Type Number')
ax.set_zlabel('Predicted Price')
ax.set_title('Preço vs. Pureza e Tipo de Flor')

# Adicionar a combinação ótima ao gráfico
ax.scatter(max_price_scenario['Purity'], max_price_scenario['FlowerTypeNumber'], max_price_scenario['Predicted_Price'], 
           color='r', s=100, label='Ótima Combinação', edgecolor='k')

# Anotação para o ponto ótimo
ax.text(max_price_scenario['Purity'], max_price_scenario['FlowerTypeNumber'], max_price_scenario['Predicted_Price'],
        f'Preço: {max_price_scenario["Predicted_Price"]:.2f}\n'
        f'Pureza: {max_price_scenario["Purity"]:.2f}\n'
        f'Tipo de Flor: {int(max_price_scenario["FlowerTypeNumber"])}\n'
        f'pH: {max_price_scenario["pH"]:.2f}',
        color='black', fontsize=9, bbox=dict(boxstyle="round,pad=0.3", edgecolor='black', facecolor='white'))

# Adicionar legenda
ax.legend()

# Exibir o gráfico
plt.tight_layout()
plt.colorbar(sc)
plt.show()

############ ANALISE DO PONTO OTIMO E DO GRAFICO 3D SEM CENÁRIOS

import pandas as pd
import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 1. Carregar a base de dados original
honey_database = pd.read_csv(r"C:\Users\millx\OneDrive\MBA DSA\honey_purity_dataset fixed.csv", delimiter=';')

# 2. Separar as características (features) e o alvo (target)
features = honey_database.drop(columns='Price')

# 3. Carregar o modelo de Random Forest treinado
with open(r'C:\Users\millx\OneDrive\MBA DSA\modelo_de_RF.pkl', 'rb') as file:
    modelo = pickle.load(file)

# 4. Prever os preços para a base de dados original
honey_database['Predicted_Price'] = modelo.predict(features)

# 5. Identificar a linha com o maior preço previsto
max_price_idx = honey_database['Predicted_Price'].idxmax()
max_price_scenario = honey_database.iloc[max_price_idx]

# Exibir a combinação ótima e o maior preço previsto
print(f"Melhor combinação de variáveis na base original:\n{max_price_scenario}")

# 6. Criar o gráfico de superfície 3D para Pureza, Tipo de Flor e Preço
fig = plt.figure(figsize=(15, 10))

# Subplot 1: Pureza vs. Tipo de Flor
ax1 = fig.add_subplot(121, projection='3d')
sc1 = ax1.scatter(honey_database['Purity'], honey_database['FlowerTypeNumber'], honey_database['Predicted_Price'], 
                  c=honey_database['Predicted_Price'], cmap='viridis')
ax1.set_xlabel('Purity')
ax1.set_ylabel('Flower Type Number')
ax1.set_zlabel('Predicted Price')
ax1.set_title('Preço vs. Pureza e Tipo de Flor')
fig.colorbar(sc1, ax=ax1, label='Preço Previsto')

# Subplot 2: Pureza vs. pH
ax2 = fig.add_subplot(122, projection='3d')
sc2 = ax2.scatter(honey_database['Purity'], honey_database['pH'], honey_database['Predicted_Price'], 
                  c=honey_database['Predicted_Price'], cmap='viridis')
ax2.set_xlabel('Purity')
ax2.set_ylabel('pH')
ax2.set_zlabel('Predicted Price')
ax2.set_title('Preço vs. Pureza e pH')
fig.colorbar(sc2, ax=ax2, label='Preço Previsto')

# Adicionar a combinação ótima ao gráfico
ax1.scatter(max_price_scenario['Purity'], max_price_scenario['FlowerTypeNumber'], max_price_scenario['Predicted_Price'], 
            color='r', s=100, label='Ótima Combinação', edgecolor='k')
ax1.text(max_price_scenario['Purity'], max_price_scenario['FlowerTypeNumber'], max_price_scenario['Predicted_Price'],
         f'Preço: {max_price_scenario["Predicted_Price"]:.2f}\n'
         f'Pureza: {max_price_scenario["Purity"]:.2f}\n'
         f'Tipo de Flor: {int(max_price_scenario["FlowerTypeNumber"])}\n'
         f'pH: {max_price_scenario["pH"]:.2f}',
         color='black', fontsize=9, bbox=dict(boxstyle="round,pad=0.3", edgecolor='black', facecolor='white'))
ax1.legend()

ax2.scatter(max_price_scenario['Purity'], max_price_scenario['pH'], max_price_scenario['Predicted_Price'], 
            color='r', s=100, label='Ótima Combinação', edgecolor='k')
ax2.text(max_price_scenario['Purity'], max_price_scenario['pH'], max_price_scenario['Predicted_Price'],
         f'Preço: {max_price_scenario["Predicted_Price"]:.2f}\n'
         f'Pureza: {max_price_scenario["Purity"]:.2f}\n'
         f'Tipo de Flor: {int(max_price_scenario["FlowerTypeNumber"])}\n'
         f'pH: {max_price_scenario["pH"]:.2f}',
         color='black', fontsize=9, bbox=dict(boxstyle="round,pad=0.3", edgecolor='black', facecolor='white'))
ax2.legend()

# Ajustar o layout e exibir o gráfico
plt.tight_layout()
plt.show()

############  Analise gráfica de sensibilidade
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

# 1. Carregar a base de dados original
honey_database = pd.read_csv(r"C:\Users\millx\OneDrive\MBA DSA\honey_purity_dataset fixed.csv", delimiter=';')

# 2. Carregar o modelo treinado
with open(r'C:\Users\millx\OneDrive\MBA DSA\modelo_de_RF.pkl', 'rb') as file:
    modelo = pickle.load(file)

# 3. Definir as colunas esperadas pelo modelo
colunas_esperadas = ['CS', 'Density', 'WC', 'pH', 'EC', 'F', 'G', 'Viscosity', 'Purity', 'FlowerTypeNumber']
honey_database = honey_database[colunas_esperadas]

# 4. Definir a função para criar novos cenários de sensibilidade
def criar_cenarios(df, variavel, intervalo):
    df_exp = pd.DataFrame(np.repeat(df.values, len(intervalo), axis=0), columns=df.columns)
    df_exp[variavel] = np.tile(intervalo, len(df))
    return df_exp

# 5. Definir intervalos para a variação das variáveis
variacao_purity = np.linspace(honey_database['Purity'].min(), honey_database['Purity'].max(), 100)
variacao_flower_type = np.arange(honey_database['FlowerTypeNumber'].min(), honey_database['FlowerTypeNumber'].max() + 1)
variacao_ph = np.linspace(honey_database['pH'].min(), honey_database['pH'].max(), 100)

# 6. Criar os cenários de sensibilidade para cada variável
cenarios_purity = criar_cenarios(honey_database, 'Purity', variacao_purity)
cenarios_flower_type = criar_cenarios(honey_database, 'FlowerTypeNumber', variacao_flower_type)
cenarios_ph = criar_cenarios(honey_database, 'pH', variacao_ph)

# 7. Prever os preços em lote
precos_purity = modelo.predict(cenarios_purity)
precos_flower_type = modelo.predict(cenarios_flower_type)
precos_ph = modelo.predict(cenarios_ph)

# 8. Plotar gráficos de sensibilidade
plt.figure(figsize=(18, 5))

# Sensibilidade de Pureza
plt.subplot(1, 3, 1)
plt.plot(variacao_purity, precos_purity[:len(variacao_purity)], marker='o', color='b')
plt.title('Sensibilidade ao Pureza')
plt.xlabel('Pureza')
plt.ylabel('Preço Previsto')
plt.grid(True)

# Sensibilidade ao Tipo de Flor
plt.subplot(1, 3, 2)
plt.plot(variacao_flower_type, precos_flower_type[:len(variacao_flower_type)], marker='o', color='r')
plt.title('Sensibilidade ao Tipo de Flor')
plt.xlabel('FlowerTypeNumber')
plt.ylabel('Preço Previsto')
plt.grid(True)

# Sensibilidade ao pH
plt.subplot(1, 3, 3)
plt.plot(variacao_ph, precos_ph[:len(variacao_ph)], marker='o', color='g')
plt.title('Sensibilidade ao pH')
plt.xlabel('pH')
plt.ylabel('Preço Previsto')
plt.grid(True)

plt.tight_layout()
plt.show()








################## ANALISE PREDITIVA COM A CRIAÇÃO DE CENÁRIOS - Visualizar a capacidade de generalização do modelo
##### Analise preditiva com cenários - PARTE 1
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Carregar o modelo treinado
with open(r'C:\Users\millx\OneDrive\MBA DSA\modelo_de_RF.pkl', 'rb') as file:
    modelo = pickle.load(file)


# 2. Definir intervalos para Pureza, Tipo de Flor e pH
pureza_interval = (0.16, 1.00)  # Exemplo de intervalo para Pureza
flower_type_interval = (1, 19)  # Intervalo de tipos de flor
ph_interval = (2.5, 7.5)  # Intervalo para pH

# 3. Gerar valores aleatórios para os cenários
np.random.seed(21)  # Garantir reprodutibilidade
cenarios = pd.DataFrame({


    'CS': [5] * 100,  # Fixo
    'Density': [1.5] * 100,  # Fixo
    'WC': [18] * 100,  # Fixo
    'pH': np.random.uniform(ph_interval[0], ph_interval[1], 100),
    'EC': [0.5] * 100,  # Fixo
    'F': [35] * 100,  # Fixo
    'G': [30] * 100,  # Fixo
    'Density': [1.5] * 100,  # Fixo
    'Viscosity': [6000] * 100,
    'Purity': np.random.uniform(pureza_interval[0], pureza_interval[1], 100), 
    'FlowerTypeNumber': np.random.randint(flower_type_interval[0], flower_type_interval[1], 100) # Fixo Isso é usado para gerar as colunas fixas do DataFrame (CS, WC, EC, F, G, Viscosity) onde cada cenário (ou linha) tem o mesmo valor para essas variáveis, repetido 4 vezes, pois o DataFrame terá 4 linhas (um para cada cenário gerado).
})

# 4. Fazer as previsões de preço com o modelo
cenarios['Predicted_Price'] = modelo.predict(cenarios)

# Criar o gráfico de superfície 3D para Purity, FlowerTypeNumber e Preço
fig = plt.figure(figsize=(15, 5))

# Subplot 1: Purity vs. FlowerTypeNumber
ax1 = fig.add_subplot(121, projection='3d')
sc = ax1.scatter(cenarios['Purity'], cenarios['FlowerTypeNumber'], cenarios['Predicted_Price'], c=cenarios['Predicted_Price'], cmap='viridis')
ax1.set_xlabel('Purity')
ax1.set_ylabel('FlowerTypeNumber')
ax1.set_zlabel('Predicted Price')
ax1.set_title('Preço vs. Pureza e Tipo de Flor')
fig.colorbar(sc, ax=ax1, label='Preço Previsto')

# Subplot 2: Purity vs. pH
ax2 = fig.add_subplot(122, projection='3d')
sc = ax2.scatter(cenarios['Purity'], cenarios['pH'], cenarios['Predicted_Price'], c=cenarios['Predicted_Price'], cmap='viridis')
ax2.set_xlabel('Purity')
ax2.set_ylabel('pH')
ax2.set_zlabel('Predicted Price')
ax2.set_title('Preço vs. Pureza e pH')
fig.colorbar(sc, ax=ax2, label='Preço Previsto')

plt.tight_layout()
plt.show()














########### Grafico de sensibilidade - PARTE 2

import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

# 1. Carregar o modelo treinado
with open(r'C:\Users\millx\OneDrive\MBA DSA\modelo_de_RF.pkl', 'rb') as file:
    modelo = pickle.load(file)

# 2. Definir as colunas esperadas pelo modelo
colunas_esperadas = ['CS', 'Density', 'WC', 'pH', 'EC', 'F', 'G', 'Viscosity', 'Purity', 'FlowerTypeNumber']

# 3. Gerar um cenário base com valores médios e garantir a ordem correta das colunas
cenario_base = pd.DataFrame({
    'CS': [5],
    'Density': [1.5],
    'WC': [18],
    'pH': [4.0],
    'EC': [0.5],
    'F': [35],
    'G': [30],
    'Viscosity': [6000],
    'Purity': [0.8],
    'FlowerTypeNumber': [10]
}, columns=colunas_esperadas)

# 4. Definir intervalos para a variação das variáveis
variacao_purity = np.linspace(0.16, 1.00, 100)
variacao_flower_type = np.arange(1, 19)
variacao_ph = np.linspace(2.5, 7.5, 100)

# Função para calcular o preço previsto dado um cenário com uma variável alterada
def calcular_preco_sensibilidade(modelo, base_cenario, variacao, variavel):
    cenarios = base_cenario.copy()
    resultados = []
    for valor in variacao:
        cenarios[variavel] = valor
        resultados.append(modelo.predict(cenarios)[0])
    return resultados

# Calcular preços previstos para cada variação
precos_purity = calcular_preco_sensibilidade(modelo, cenario_base, variacao_purity, 'Purity')
precos_flower_type = calcular_preco_sensibilidade(modelo, cenario_base, variacao_flower_type, 'FlowerTypeNumber')
precos_ph = calcular_preco_sensibilidade(modelo, cenario_base, variacao_ph, 'pH')

# Plotar gráficos de sensibilidade
plt.figure(figsize=(18, 5))

# Sensibilidade de Pureza
plt.subplot(1, 3, 1)
plt.plot(variacao_purity, precos_purity, marker='o', color='b')
plt.title('Sensibilidade ao Pureza')
plt.xlabel('Pureza')
plt.ylabel('Preço Previsto')
plt.grid(True)

# Sensibilidade ao Tipo de Flor
plt.subplot(1, 3, 2)
plt.plot(variacao_flower_type, precos_flower_type, marker='o', color='r')
plt.title('Sensibilidade ao Tipo de Flor')
plt.xlabel('FlowerTypeNumber')
plt.ylabel('Preço Previsto')
plt.grid(True)

# Sensibilidade ao pH
plt.subplot(1, 3, 3)
plt.plot(variacao_ph, precos_ph, marker='o', color='g')
plt.title('Sensibilidade ao pH')
plt.xlabel('pH')
plt.ylabel('Preço Previsto')
plt.grid(True)

plt.tight_layout()
plt.show()


######### ANALISE DE CENÁRISO - 100 GERADOS - P EXTRAPOLAR O MODELO E GENERALIZAR 

#### Combinação de cenários
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 1. Carregar o modelo treinado
with open(r'C:\Users\millx\OneDrive\MBA DSA\modelo_de_RF.pkl', 'rb') as file:
    modelo = pickle.load(file)

# 2. Definir intervalos para Pureza, Tipo de Flor e pH
pureza_interval = (0.16, 1.00)
flower_type_interval = (1, 19)
ph_interval = (2.5, 7.5)

# 3. Gerar cenários
pureza_values = np.linspace(pureza_interval[0], pureza_interval[1], 10)
flower_type_values = np.arange(flower_type_interval[0], flower_type_interval[1] + 1)
ph_values = np.linspace(ph_interval[0], ph_interval[1], 10)

# Gerar todos os cenários possíveis
from itertools import product

cenarios = pd.DataFrame(
    product(pureza_values, flower_type_values, ph_values),
    columns=['Purity', 'FlowerTypeNumber', 'pH']
)

# Adicionar variáveis fixas
cenarios['CS'] = 5
cenarios['Density'] = 1.5
cenarios['WC'] = 18
cenarios['EC'] = 0.5
cenarios['F'] = 35
cenarios['G'] = 30
cenarios['Viscosity'] = 6000

# Obter a ordem das características do modelo
try:
    feature_names = modelo.feature_names_in_  # Para scikit-learn 1.0+
except AttributeError:
    feature_names = modelo.get_feature_names_out()  # Para versões mais recentes

# Garantir que o DataFrame de entrada tenha a mesma ordem de características
cenarios = cenarios[feature_names]

# 4. Fazer as previsões de preço com o modelo
cenarios['Predicted_Price'] = modelo.predict(cenarios)

# 5. Encontrar a combinação ótima
max_price_idx = cenarios['Predicted_Price'].idxmax()
max_price_scenario = cenarios.iloc[max_price_idx]

# 6. Visualizar a combinação ótima com gráficos 3D

# Aumentar o tamanho da figura
fig = plt.figure(figsize=(12, 8))

# Subplot 1: Pureza vs. Tipo de Flor
ax1 = fig.add_subplot(121, projection='3d')
sc1 = ax1.scatter(cenarios['Purity'], cenarios['FlowerTypeNumber'], cenarios['Predicted_Price'], c=cenarios['Predicted_Price'], cmap='viridis')
ax1.set_xlabel('Pureza', fontsize=10)
ax1.set_ylabel('Tipo de Flor', fontsize=10)
ax1.set_zlabel('Preço Previsto', fontsize=10)
ax1.set_title('Preço vs. Pureza e Tipo de Flor', fontsize=12)

# Adicionar ponto ótimo ao gráfico
ax1.scatter(max_price_scenario['Purity'], max_price_scenario['FlowerTypeNumber'], max_price_scenario['Predicted_Price'], color='r', s=100, label='Ótima Combinação', edgecolor='k')
ax1.legend(fontsize=9)

# Anotação para o ponto ótimo
ax1.text(max_price_scenario['Purity'], max_price_scenario['FlowerTypeNumber'], max_price_scenario['Predicted_Price'],
          f'Preço: {max_price_scenario["Predicted_Price"]:.2f}\n'
          f'Pureza: {max_price_scenario["Purity"]:.2f}\n'
          f'Tipo de Flor: {int(max_price_scenario["FlowerTypeNumber"])}\n'
          f'pH: {max_price_scenario["pH"]:.2f}',
          color='black', fontsize=8, bbox=dict(boxstyle="round,pad=0.3", edgecolor='black', facecolor='white'))

# Subplot 2: Pureza vs. pH
ax2 = fig.add_subplot(122, projection='3d')
sc2 = ax2.scatter(cenarios['Purity'], cenarios['pH'], cenarios['Predicted_Price'], c=cenarios['Predicted_Price'], cmap='viridis')
ax2.set_xlabel('Pureza', fontsize=10)
ax2.set_ylabel('pH', fontsize=10)
ax2.set_zlabel('Preço Previsto', fontsize=10)
ax2.set_title('Preço vs. Pureza e pH', fontsize=12)

# Adicionar ponto ótimo ao gráfico
ax2.scatter(max_price_scenario['Purity'], max_price_scenario['pH'], max_price_scenario['Predicted_Price'], color='r', s=100, label='Ótima Combinação', edgecolor='k')
ax2.legend(fontsize=9)

# Anotação para o ponto ótimo
ax2.text(max_price_scenario['Purity'], max_price_scenario['pH'], max_price_scenario['Predicted_Price'],
          f'Preço: {max_price_scenario["Predicted_Price"]:.2f}\n'
          f'Pureza: {max_price_scenario["Purity"]:.2f}\n'
          f'Tipo de Flor: {int(max_price_scenario["FlowerTypeNumber"])}\n'
          f'pH: {max_price_scenario["pH"]:.2f}',
          color='black', fontsize=8, bbox=dict(boxstyle="round,pad=0.3", edgecolor='black', facecolor='white'))

# Ajustar o espaçamento entre os subplots
plt.subplots_adjust(wspace=0.3, hspace=0.3)
plt.tight_layout()
plt.show()