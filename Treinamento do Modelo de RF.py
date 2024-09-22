

######### MBA PROJECT
######### Criar e treinar o Modelo utilizando o pickle

#########  Importing the libraries 

import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
import pickle
import datetime
print(datetime.datetime.now())
#########  Read the data and view it

# Read the data from the CSV file and assign it to the variable honey_database
honey_database = pd.read_csv(r"C:\Users\millx\OneDrive\MBA DSA\honey_purity_dataset fixed.csv",delimiter=';')

# View the first few rows of the DataFrame
print(honey_database.head())
print("Colunas disponíveis na base de dados:", honey_database.columns)

# Separar as características (features) e o alvo (target)
features = honey_database.drop(columns='Price')
target = honey_database['Price']
print(target)
print(features)

# Dividir os dados em conjunto de treinamento e teste - Random State 42 ajuda os resultados do run de treinamento e teste serem o mesmo, sem random state, os resultados seriam diferentes a cada run
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=20)

# Criar o modelo Random Forest Regressor
rf_regressor = RandomForestRegressor(random_state=20)

# Realizar a validação cruzada com 5 folds
cv_scores = cross_val_score(rf_regressor, X_train, y_train, cv=5, scoring='r2')

# Exibir os resultados da validação cruzada

print("Pontuações R^2 da Validação Cruzada:", cv_scores)
print("Média da Pontuação R^2 da Validação Cruzada:", cv_scores.mean())
print("Desvio Padrão das Pontuações R^2 da Validação Cruzada:", cv_scores.std()) 

# Treinar o modelo com os dados de treinamento
model = rf_regressor.fit(X_train, y_train)

# save the RF classification model as a pickle file
model_pkl_file = "modelo_de_RF.pkl"  

with open(model_pkl_file, 'wb') as file:  
    pickle.dump(model, file)

print(datetime.datetime.now())