# FOIL Gain Calculator

# Descripción
Programa de ejercico practico para Aprendizaje Supervisado que calcula a través del algoritmo **FOIL Gain** (First-Order Inductive Learner Gain), una métrica utilizada en aprendizaje automático para evaluar la ganancia de información al aplicar condiciones en algoritmos de clasificación.

# Características
    - Calcula la FOIL Gain para condiciones específicas sobre conjuntos de datos
    - Proporciona todos los valores intermedios del cálculo
    - Maneja casos especiales con la función `log2_safe`
    - Sigue la estructura estándar del algoritmo FOIL

# Fórmula de FOIL Gain
    ```
    FOIL Gain = p * [log2(p/(p+n)) - log2(P/(P+N))]
    ```

Donde:
    - **P**: Positivos antes de aplicar la condición
    - **N**: Negativos antes de aplicar la condición  
    - **p**: Positivos después de aplicar la condición
    - **n**: Negativos después de aplicar la condición

# Estructura del Código

# Datos de Entrada
    El conjunto de datos contiene registros con los siguientes atributos:
    - `edad`: Valor numérico
    - `departamento`: String (IT, RRHH, Finanzas)
    - `nivel_educativo`: String (terciario, universitario, maestría)
    - `en_formacion`: Boolean (True/False)

# Funciones Principales
    ```
    `log2_safe(x)`
    - Calcula el logaritmo base 2 de un número
    - Maneja el caso especial cuando x = 0 retornando `-inf`
    ```
    ```
    calculo principal
    1. Calcula P y N (positivos y negativos antes de la condición)
    2. Filtra datos aplicando la condición específica
    3. Calcula p y n (positivos y negativos después de la condición)
    4. Computa la FOIL Gain usando la fórmula
    ```

# Ejemplo de Uso
    ```python
    # Condición: nivel_educativo == "terciario"
    P [positivos antes] = 4
    N [negativos antes] = 4
    p [positivos después] = 3
    n [negativos después] = 0
    p / (p + n) = 1.000
    P / (P + N) = 0.500
    log2(p / (p + n)) = 0.000
    log2(P / (P + N)) = -1.000
    FOIL Gain = 3.000
    ```

# Requisitos
    - Python 3.x
    - Módulo math (incluido en la biblioteca estándar)

# Ejecución
    ```bash
    python foil_gain_calculator.py
    ```
