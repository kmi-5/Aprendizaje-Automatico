# Importar librerias
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import confusion_matrix

## Carga de datos

# Se utilizo ruta absoluta para evitar errores de path
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, 'TablaPrediccionAbandono-Entrenamiento.xlsx')
df = pd.read_excel(file_path)

print("-"*50)
print(f"\nDimensiones del dataset original: {df.shape[0]} filas x {df.shape[1]} columnas")
print("\n" + "-"*50)
print("Primeras 5 filas del dataset:")
print(df.head())

# Limpieza y preparacion de datos
df.rename(columns={'EstadoFinal': 'estado_final'}, inplace=True)
df.dropna(inplace=True)

# Convertir variables categóricas
df_cod = df.copy()
for col in df_cod.select_dtypes(include='object').columns:
    df_cod[col] = df_cod[col].astype('category').cat.codes

# Separar variables predictoras (X) y objetivo (y) 
X = df_cod.drop(columns=['estado_final'])
y = df_cod['estado_final']

# Dividir en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Entrenar el modelo 
modelo = DecisionTreeClassifier(
    criterion='entropy',
    max_depth=6,
    min_samples_leaf=5,
    random_state=42
)
modelo.fit(X_train, y_train)

# Matriz de confusion con los datos de prediccion
y_pred = modelo.predict(X_test)
cm = confusion_matrix(y_test, y_pred, labels=modelo.classes_)

# Creacion de la paleta rosa
rosa_cmap = sns.light_palette("#FF69B4", as_cmap=True)  

plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap=rosa_cmap,
    cbar=False,
    xticklabels=['Predice ' + str(cls) for cls in modelo.classes_],
    yticklabels=['Real ' + str(cls) for cls in modelo.classes_],
    annot_kws={"size": 13, "color": "black"}
)
plt.xlabel("Predicción del Modelo", fontsize=12)
plt.ylabel("Hecho Real", fontsize=12)
plt.title("Matriz de Confusión - Predicción de Abandono", fontsize=14)
plt.tight_layout()
plt.show()

## Graficos

# Distribucion del estado final
plt.figure(figsize=(6,4))
sns.countplot(
    x='estado_final', data=df,
    color="#FFB6C1"  
)
plt.title("Distribución del estado final de los estudiantes")
plt.xlabel("Estado final")
plt.ylabel("Cantidad de estudiantes")
plt.show()

# Importancia de variables
importancia = pd.DataFrame({
    'Variable': X.columns,
    'Importancia': modelo.feature_importances_
}).sort_values('Importancia', ascending=True)

importancia_filtrada = importancia[importancia['Importancia'] > 0]

plt.figure(figsize=(10,6))
plt.barh(
    importancia_filtrada['Variable'],
    importancia_filtrada['Importancia'],
    color="#FFB6C1"
)
plt.title("Importancia real de las variables en el modelo")
plt.xlabel("Importancia")
plt.ylabel("Variable")
plt.show()

# Abol de decision
plt.figure(figsize=(25,10))
plot_tree(
    modelo,
    feature_names=X.columns,
    class_names=['Continúa', 'Abandona'],
    filled=True,
    rounded=True,
    fontsize=8,
)
plt.title("Árbol de decisión - Predicción de abandono", fontsize=14)
plt.show()

# Interpretacion final
print("\n" + "-"*50)
print("\nIntepretacion final")
print("Las variables con mayor importancia para predecir el abandono fueron:")
print(importancia_filtrada.sort_values('Importancia', ascending=False).head())

print("\n" + "-"*50)
print("\nRecomendaciones para la institucion:")
print("1. Monitorear a estudiantes con baja asistencia y desempeño académico")
print("2. Implementar programas de apoyo virtuales")
print("3. Desarrollar estrategias de retención temprana basadas en estos predictores")
print("4. Atención especial a estudiantes que aprueben 1 o menos materias en primer cuatrimestre")