#Debe hacer 3 modelos de regresión usando knn, uno con k igual a la raiz cuadrada de la cantidad de filas, 
#otro con la misma k pero validación cruzada, y un tercero con el tuneo del valor de k. El conjunto de datos
#a usar es cars.csv y la variable respuesta es city_mpg. Codifique las variables categóricas y normalice las
#numéricas
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, mean_squared_error, root_mean_squared_error, r2_score
from sklearn.compose import make_column_selector as selector
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

carros = pd.read_csv("cars.csv")

#NA
carros["bore"] = carros["bore"].fillna(carros["bore"].median())
carros["stroke"] = carros["stroke"].fillna(carros["stroke"].median())
carros["horsepower"] = carros["horsepower"].fillna(carros["horsepower"].median())
carros["peak_rpm"] = carros["peak_rpm"].fillna(carros["peak_rpm"].median())
carros["price"] = carros["price"].fillna(carros["price"].median())

carros.pop("normalized_losses")

X = carros.drop(columns=['city_mpg'])
y = carros['city_mpg']

categorical_columns = X.select_dtypes(include=['object']).columns
numerical_columns = X.select_dtypes(include=['number']).columns

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numerical_columns),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_columns)
])

#MODELO 1: k raiz cuadrada de la cantidd de filas
k_1 = int(np.sqrt(len(carros)))

X_train, X_test,y_train, y_test = train_test_split(X, y,test_size=0.3,train_size=0.7,random_state=42)

raiz = Pipeline([
    ('preprocessing', preprocessor),
    ('knn', KNeighborsRegressor(n_neighbors=k_1))
])
raiz.fit(X_train,y_train)

y_pred_1 = raiz.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred_1))
r2 = r2_score(y_test, y_pred_1)

print("Modelo 1 (k=√n)")
print(f"K = {k_1}")
print(f"RMSE = {round(rmse, 2)}")

print("-----------------")

#MODELO 2: validación cruzada con raiz
val_cruzada_raiz = Pipeline([
    ('preprocessing', preprocessor),
    ('knn', KNeighborsRegressor(n_neighbors=k_1))
])
# -a
val_cruz_scores = cross_val_score(val_cruzada_raiz, X, y, cv = 5, scoring='neg_mean_squared_error')

rmse_val_cruz = np.sqrt(-val_cruz_scores.mean())
print("Modelo 2 (k=√n, validacion cruzada)")
print(f"K = {k_1}")
print(f"RMSE = {round(rmse_val_cruz, 2)}")

print("-----------------")
#MODELO 3: Tuneado

modelo3_preprocessor = Pipeline([
    ('preprocessing', preprocessor),
    ('knn', KNeighborsRegressor())
])

pram_tune = {
    'knn__n_neighbors': [3, 5, 7, 9, 11],
    'knn__weights': ['uniform', 'distance'],
    'knn__p': [1, 2]
}

grid_search = GridSearchCV(modelo3_preprocessor, pram_tune, cv = 5, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(X_train, y_train)
tuned_knn = grid_search.best_estimator_

y_pred_2 = tuned_knn.predict(X_test)
rmse_tuned = root_mean_squared_error(y_test, y_pred_2)
r2_tuned = r2_score(y_test, y_pred_2)

print("Modelo 3 (k tuneado)")
print(f"Mejor KNN:  {grid_search.best_params_}")
print(f"RMSE = {round(rmse_val_cruz, 2)}")