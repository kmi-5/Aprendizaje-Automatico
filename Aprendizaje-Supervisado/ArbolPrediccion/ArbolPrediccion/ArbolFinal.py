# Importar librerias
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

## Carga de datos

# Rutas de lectura de archivos
script_dir = os.path.dirname(os.path.abspath(__file__))

# Rutas absolutas de los archivos
archivo_entrenamiento = os.path.join(script_dir, 'TablaPrediccionAbandono-Entrenamiento.xlsx')
archivo_final = os.path.join(script_dir, 'TablaPrediccionAbandono-DatosFinal.xlsx')
archivo_predicciones = os.path.join(script_dir, "Predicciones_Final.xlsx")  # Genera la nueva tabla de excel guardando las predicciones

# Cargar datos de entrenamiento
df_train = pd.read_excel(archivo_entrenamiento)
df_train.rename(columns={'EstadoFinal':'estado_final'}, inplace=True)
df_train.dropna(inplace=True)

# Codificar variables categoricas 
for col in df_train.select_dtypes(include='object').columns:
    df_train[col] = df_train[col].astype('category').cat.codes

# Separar variables predictoras (X) y objetivo (y) 
X_train = df_train.drop(columns=['estado_final'])
y_train = df_train['estado_final']

# Entrenar el modelo
modelo = DecisionTreeClassifier(
    criterion='entropy',
    max_depth=6,
    min_samples_leaf=5,
    random_state=42
)
modelo.fit(X_train, y_train)

# Evaluacion con datos de entrenamiento 
y_pred = modelo.predict(X_train)
cm = confusion_matrix(y_train, y_pred, labels=modelo.classes_)
rosa_cmap = sns.light_palette("#FF69B4", as_cmap=True)

plt.figure(figsize=(6,5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap=rosa_cmap,
    cbar=False,
    xticklabels=['Predice No Abandona', 'Predice Abandona'],
    yticklabels=['Real No Abandona', 'Real Abandona'],
    annot_kws={"size":13, "color":"black"},
    linewidths=1,
    linecolor="white"
)
plt.xlabel("Predicción del Modelo", fontsize=12)
plt.ylabel("Hecho Real", fontsize=12)
plt.title("Matriz de Confusión - Predicción de Abandono", fontsize=14, color="#FF69B4")
plt.tight_layout()
plt.show()

print("\nReporte de clasificacion")
print(classification_report(y_train, y_pred))
print("\nPrecisión general:", round(accuracy_score(y_train, y_pred),3))

## Graficos 

# Distribucion de estados
df_train_plot = df_train.copy()
df_train_plot['estado_final'] = df_train_plot['estado_final'].map({0:'No Abandona',1:'Abandona'})

plt.figure(figsize=(6,4))
sns.countplot(x='estado_final', data=df_train_plot, color="#FFB6C1")
plt.title("Distribución del estado final de los estudiantes", color="#FF69B4")
plt.xlabel("Estado final")
plt.ylabel("Cantidad de estudiantes")
plt.show()

# Importancia de variables
importancia = pd.DataFrame({
    'Variable': X_train.columns,
    'Importancia': modelo.feature_importances_
}).sort_values('Importancia', ascending=True)

importancia_filtrada = importancia[importancia['Importancia'] > 0]

plt.figure(figsize=(10,6))
plt.barh(
    importancia_filtrada['Variable'],
    importancia_filtrada['Importancia'],
    color="#FFB6C1"
)
plt.title("Importancia de variables en el modelo", color="#FF69B4")
plt.xlabel("Importancia")
plt.ylabel("Variable")
plt.show()

# Arbol de decision
plt.figure(figsize=(25,10))
plot_tree(
    modelo,
    feature_names=X_train.columns,
    class_names=['No Abandona','Abandona'],
    filled=True,
    rounded=True,
    fontsize=8
)
plt.title("Árbol de decisión - Predicción de abandono", fontsize=14, color="#FF69B4")
plt.show()

# Prediccion en datos finales
df_final = pd.read_excel(archivo_final)
df_final_pred = df_final.copy()

# Codificacion igual que entrenamiento
for col in df_final_pred.select_dtypes(include='object').columns:
    df_final_pred[col] = df_final_pred[col].astype('category').cat.codes

X_new = df_final_pred[modelo.feature_names_in_]
df_final['PrediciónAbandono'] = modelo.predict(X_new)

# Mapeao de 0 a No y 1 a Sí
df_final['PrediciónAbandono'] = df_final['PrediciónAbandono'].map({0:'No', 1:'Sí'})

# Guardar nuevo Excel con las predicciones
df_final.to_excel(archivo_predicciones, index=False)
print(f"\nNuevo archivo con predicciones creado: {archivo_predicciones}")