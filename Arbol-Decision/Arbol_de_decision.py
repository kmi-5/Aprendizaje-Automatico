import math
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder

# Dataset
data = {
    'ID': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Edad': [24, 38, 29, 45, 52, 33, 41, 27, 36, 31],
    'Uso_de_datos': [2.5, 6.0, 3.0, 8.0, 7.5, 4.0, 5.5, 2.0, 6.5, 3.5],
    'Tiene_linea_fija': ['No', 'Sí', 'No', 'Sí', 'Sí', 'No', 'Sí', 'No', 'Sí', 'No'],
    'Acepto_oferta': ['No', 'Sí', 'No', 'Sí', 'Sí', 'No', 'Sí', 'No', 'Sí', 'No']
}

df = pd.DataFrame(data)

# calcular entropia del dataset 
def calcular_entropia(conjunto):
    total = len(conjunto)
    if total == 0:
        return 0
    
    si_count = sum(1 for x in conjunto if x == 'Sí')
    no_count = total - si_count
    
    p_si = si_count / total
    p_no = no_count / total
    
    if p_si == 0 or p_no == 0:
        return 0
    
    return - (p_si * math.log2(p_si) + p_no * math.log2(p_no))

entropia_original = calcular_entropia(df['Acepto_oferta'])
print(f"1. ENTROPÍA DEL CONJUNTO ORIGINAL: {entropia_original:.3f}")
print()

# preparacion de datos para ganancia de información
def crear_rangos_edad(edad):
    if edad <= 30:
        return 'Joven'
    elif edad <= 50:
        return 'Adulto'
    else:
        return 'Mayor'

def crear_rangos_uso(uso):
    if uso <= 3:
        return 'Bajo'
    elif uso <= 6:
        return 'Medio'
    else:
        return 'Alto'

df['Edad_rango'] = df['Edad'].apply(crear_rangos_edad)
df['Uso_rango'] = df['Uso_de_datos'].apply(crear_rangos_uso)

print("Rangos:")
print(df[['ID', 'Edad', 'Edad_rango', 'Uso_de_datos', 'Uso_rango', 'Tiene_linea_fija', 'Acepto_oferta']])
print()

# calcular ganancia de información
def calcular_ganancia_informacion(df, atributo, objetivo='Acepto_oferta'):
    entropia_total = calcular_entropia(df[objetivo])
    valores_atributo = df[atributo].unique()
    
    entropia_ponderada = 0
    print(f"Cálculo para atributo: {atributo}")
    
    for valor in valores_atributo:
        subconjunto = df[df[atributo] == valor]
        peso = len(subconjunto) / len(df)
        entropia_sub = calcular_entropia(subconjunto[objetivo])
        entropia_ponderada += peso * entropia_sub
        print(f"  {valor}: {len(subconjunto)} instancias, Entropía: {entropia_sub:.3f}")
    
    ganancia = entropia_total - entropia_ponderada
    print(f"  Ganancia de información: {entropia_total:.3f} - {entropia_ponderada:.3f} = {ganancia:.3f}")
    print()
    return ganancia

# calcular ganancias
print("-" * 50)
print("Cálculo de GANANCIAS DE INFORMACIÓN:")
print("-" * 50)

ganancia_edad = calcular_ganancia_informacion(df, 'Edad_rango')
ganancia_linea_fija = calcular_ganancia_informacion(df, 'Tiene_linea_fija')
ganancia_uso = calcular_ganancia_informacion(df, 'Uso_rango')

# mostrar resultados de ganancias
print("Resumen de GANANCIAS:")
print(f"Edad (rangos): {ganancia_edad:.3f}")
print(f"Tiene línea fija: {ganancia_linea_fija:.3f}")
print(f"Uso de datos (rangos): {ganancia_uso:.3f}")
print()

# armar el arbol de decisión
# preparar datos para el modelo
df_modelo = df.copy()

# Codificar variables categóricas
le_edad = LabelEncoder()
le_uso = LabelEncoder()
le_linea = LabelEncoder()
le_objetivo = LabelEncoder()

df_modelo['Edad_rango_encoded'] = le_edad.fit_transform(df_modelo['Edad_rango'])
df_modelo['Uso_rango_encoded'] = le_uso.fit_transform(df_modelo['Uso_rango'])
df_modelo['Tiene_linea_fija_encoded'] = le_linea.fit_transform(df_modelo['Tiene_linea_fija'])
df_modelo['Acepto_oferta_encoded'] = le_objetivo.fit_transform(df_modelo['Acepto_oferta'])

# Variables predictoras y objetivo
X = df_modelo[['Edad_rango_encoded', 'Uso_rango_encoded', 'Tiene_linea_fija_encoded']]
y = df_modelo['Acepto_oferta_encoded']

# entrenamiento del arbol de decisión
arbol = DecisionTreeClassifier(
    criterion='entropy',
    max_depth=3,
    random_state=42
)
arbol.fit(X, y)

# vvisualizacion del arbol
plt.figure(figsize=(12, 8))
plot_tree(
    arbol,
    feature_names=['Edad_rango', 'Uso_rango', 'Tiene_linea_fija'],
    class_names=['No', 'Sí'],
    filled=True,
    rounded=True,
    fontsize=10
)
plt.title("Árbol de Decisión - Predicción de Aceptación de Oferta", fontsize=14)
plt.show()

# conclusiones y reglas
print("-" * 50)
print("Conclusiones:")
print("-" * 50)
mejor_atributo = max([('Edad_rango', ganancia_edad), 
                      ('Tiene_linea_fija', ganancia_linea_fija), 
                      ('Uso_rango', ganancia_uso)], 
                     key=lambda x: x[1])

print(f"El mejor atributo para comenzar el árbol es: {mejor_atributo[0]}")
print(f"Con una ganancia de información de: {mejor_atributo[1]:.3f}")
print()

print("Reglas de predicción:")
print("• Si el cliente NO tiene línea fija → NO acepta la oferta")
print("• Si el cliente SÍ tiene línea fija → SÍ acepta la oferta")
print("• El uso de datos y la edad pueden refinar la predicción en casos específicos")

# Mostrar matriz de predicciones
print("\nPredicciones del modelo:")
predicciones = arbol.predict(X)
df_resultado = df.copy()
df_resultado['Prediccion'] = ['Sí' if p == 1 else 'No' for p in predicciones]
df_resultado['Correcto'] = df_resultado['Acepto_oferta'] == df_resultado['Prediccion']
print(df_resultado[['ID', 'Edad_rango', 'Uso_rango', 'Tiene_linea_fija', 'Acepto_oferta', 'Prediccion', 'Correcto']])

precision = sum(df_resultado['Correcto']) / len(df_resultado)
print(f"\nPrecisión del modelo: {precision:.1%}")