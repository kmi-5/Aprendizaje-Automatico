# Clasificador de Acciones de Videojuegos

# 📄 Descripción

Este programa implementa una función simple para **predecir resultados en un videojuego** según el tipo de acción y su duración.
Su finalidad es mostrar cómo estructurar una lógica condicional para modelar **comportamientos básicos de inteligencia de juego**.

# ⚙️ Funcionalidad

1. **Entrada de datos:**
   El usuario ingresa una acción (`Combate`, `Exploración` o `Interacción social`) y la duración (en segundos).
2. **Clasificación lógica:**

   * **Combate:** si la duración es mayor a 100 segundos → “Victoria”, de lo contrario → “Derrota”.
   * **Exploración:** si la duración es mayor a 250 segundos → “Descubrimiento”, de lo contrario → “Sin hallazgos”.
   * **Interacción social:** siempre retorna “Mensaje enviado”.
   * Cualquier otra acción → “Resultado desconocido”.
3. **Salida:**
   El sistema imprime el resultado predicho en función de las condiciones anteriores.

# 📚 Librerías utilizadas

No se utilizan librerías externas. El código está escrito únicamente con **Python estándar**.

# 🧩 Uso

Ejecutar el script desde la terminal o un entorno interactivo.
El programa solicitará los datos por teclado y mostrará en pantalla el resultado correspondiente.
