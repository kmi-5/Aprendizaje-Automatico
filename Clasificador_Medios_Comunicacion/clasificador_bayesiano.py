# Librerías
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Dataset
datos = {
    "texto": [
        "El presidente anunció una nueva reforma educativa",
        "Descubren que la vacuna convierte a las personas en robots",
        "La NASA confirma el hallazgo de agua en Marte",
        "Científicos afirman que la Tierra es plana",
        "El ministerio de salud lanza campaña contra el dengue",
        "Celebridades usan crema milagrosa para rejuvenecer 30 años",
        "Se inaugura el nuevo hospital en la ciudad",
        "Estudio revela que comer chocolate cura el cáncer",
        "Gobierno aprueba ley de protección ambiental",
        "Investigadores aseguran que los teléfonos espian nuestros sueños"
    ],
    "etiqueta": [
        "real", "fake", "real", "fake", "real",
        "fake", "real", "fake", "real", "fake"
    ]
}

df = pd.DataFrame(datos)

# Vectorización de texto
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df["texto"])
y = df["etiqueta"]

# Dividir datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Entrenar modelo Naive Bayes
modelo = MultinomialNB()
modelo.fit(X_train, y_train)

# Evaluar el modelo
y_pred = modelo.predict(X_test)

print("Precisión total (accuracy):", accuracy_score(y_test, y_pred))
print("\nMatriz de confusión:")
print(confusion_matrix(y_test, y_pred, labels=["real", "fake"]))
print("\nReporte de clasificación:")
print(classification_report(y_test, y_pred))

# Clasificación de nuevas noticias
nuevas_noticias = [
    "Nuevo estudio demuestra que el café mejora la memoria",
    "Expertos afirman que los gatos pueden hablar con humanos"
]

X_nuevas = vectorizer.transform(nuevas_noticias)
predicciones = modelo.predict(X_nuevas)

print("\nClasificación de nuevas noticias:")
for noticia, etiqueta in zip(nuevas_noticias, predicciones):
    print(f"'{noticia}' -> {etiqueta}")