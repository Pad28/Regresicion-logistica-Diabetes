import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns

data = pd.read_csv("diabetes.csv");

#Se cambia el nombre de las columnas
nuevos_nombres = ['Embarazos', 'Glucosa', 'Presión arterial', 'Grosor de la piel', 'Insulina', 'IMC', 'DiabetesPedigríFunción', 'Edad', 'Resultado']
data.columns = nuevos_nombres

#Se eliminan las columnas que no son significantes para la prediccion
columnas_a_eliminar = ['Embarazos', 'Edad']
data = data.drop(columns=columnas_a_eliminar)

#Ahora cargamos las variables de las 6 columnas de entrada en X excluyendo la columna "Resultado" con el método drop().
#En cambio agregamos la columna Resultado en la variable y. Ejecutamos X.shape para comprobar la dimensión de nuestra
#matriz con datos de entrada de 769 registros por 6 columnas.
X = np.array(data.drop(['Resultado'],axis=1))
y = np.array(data['Resultado'])

x1 = np.array(data.drop(['Resultado'],axis=1))
y = np.array(data['Resultado'])
x = sm.add_constant(x1)
reg_log = sm.Logit(y,x) ##Predicciones de p(x)/1-p(x)
results_log = reg_log.fit() #Ajustar todo en funcion de p(x)/1 P(x)

# print(results_log.summary())
predicciones = results_log.predict() #predicciones que hace el logit
# print(predicciones[0:25])

matriz_confusion=results_log.pred_table()
VP = matriz_confusion[0][0];
FP = matriz_confusion[1][0];
FN = matriz_confusion[0][1];
VN = matriz_confusion[1][1];
# print(VP,FN,FP,VN);

# exactitud = (VP + VN)/(VP + VN + FP+ FN);
# print("La exactitud es: ", exactitud)

# # Precisión = VP/VP + FP

# precision = VP/(VP + FP);
# print("La precision es: ", precision);

# # Sensibilidad= VP / VP + FN
# sensibilidad = VP/(VP + FN);
# print("La sensibilidad es: ", sensibilidad);

# # Especificidad= VN / VN + FP
# especificidad = VN / (VN + FP);
# print("La especificidad es: ", especificidad);

from sklearn import linear_model
from sklearn import model_selection
model = linear_model.LogisticRegression()
model.fit(X,y)

#Una vez compilado nuestro modelo, le hacemos clasificar todo nuestro conjunto de entradas X utilizando
#el método “predict(X)” y revisamos algunas de sus salidas y vemos que coincide con las salidas
#reales de nuestro archivo csv.
predictions = model.predict(X)
# print(predictions[0:5])

#subdividimos nuestros datos de entrada en forma aleatoria (mezclados) utilizando
#70% de registros para entrenamiento y 30% para validar.
validation_size = 0.30
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, y, test_size=validation_size, random_state=seed)

#Volvemos a compilar nuestro modelo de Regresión Logística pero esta vez sólo con
#80% de los datos de entrada y calculamos el nuevo scoring que ahora nos da 76%.
name='Logistic Regression'
kfold = model_selection.KFold(n_splits=10, shuffle=True, random_state=7)
cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
# print(msg)

#ahora hacemos las predicciones -en realidad clasificación- utilizando nuestro
#“cross validation set”, es decir del subconjunto que habíamos apartado.
#En este caso vemos que los aciertos fueron del 78% pero hay que tener en cuenta
#que el tamaño de datos era pequeño.
from sklearn.metrics import accuracy_score
predictions = model.predict(X_validation)
# print(accuracy_score(Y_validation, predictions))

from sklearn.metrics import confusion_matrix
# print(confusion_matrix(Y_validation, predictions))

from sklearn.metrics import classification_report
df_report = pd.DataFrame(classification_report(Y_validation, predictions, output_dict=True))
# print(df_report.loc['precision', '1'])

import json
with open("data.json", "r") as archivo:
    datos_json = json.load(archivo)

X_new = pd.DataFrame({
    'Glucosa': [datos_json['Glucosa']], 
    'Presion arterial': [datos_json['Presion arterial']], 
    'Grosor de la piel': [datos_json['Grosor de la piel']], 
    'Insulina': [datos_json['Insulina']], 
    'IMC': [datos_json['IMC']], 
    'DiabetesPedigríFunción': [datos_json['DiabetesPedigríFunción']]
})
result = model.predict(X_new)
prediccion_json = {
    "Prediccion": result[0].tolist(),
    "PrecisionVerdadera": df_report.loc['precision', '1'],
    "PrecisionFalso": df_report.loc['precision', '0']
}

with open("prediccion.json", "w") as archivo:
    json.dump(prediccion_json, archivo)
