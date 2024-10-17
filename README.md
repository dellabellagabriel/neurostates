## Objetivo

El objetivo de esta librería es estandarizar el análisis de los estados de consciencia a través de patrones recurrentes de conectividad cerebral conocido en la literatura como "Brain States". El proyecto busca proporcionar herramientas para la detección de estados cerebrales recurrentes, sus patrones de conectividad específicos y su probabilidad de ocurrencia asociada a partir de datos preprocesados de neuroimagenes (específicamente electroencefalograma y resonancia magnética funcional).

## Que funcionalidades tiene el proyecto

- Calcular la conectividad funcional dinamica por diferentes metodos (ej. Pearson, wSMI)
- Clustering de matrices de conectividad funcional (ej. k-means, DBSCAN).
- Determinación del k optimo para k-means (ej, elbow method).
- Ordenamiento de estados cerebrales según entriopía u otras medidas (ej. similaridad estructural).
- Construcción de la distribución de probabilidad de estados cerebrales para cada sujeto.
- Visualización de datos y resultados como las matrices de conectividad dinámica, distribución de la probabilidad, matrices de transición.
- Test de significancia estadística sobre los resultados (ej. t-test, modelo de efectos mixtos)

## Como usar la libreria

```python
import matplotlib.pyplot as plt
from brainstates import BrainStates

#CARGAR DATOS DE NEUROIMAGENES
grupo_control = ... #matriz de numpy sujetos x regiones x tiempo
grupo_pacientes_1 = ...
grupo_pacientes_2 = ...

brain_states = BrainStates(grupo_control, grupo_pacientes_1, grupo_pacientes_2)

# 1) calcular la conectividad funcional dinamica
# puede ser "pearson", "lasso", "wSDM", "wSMI", "wPLI"
brain_states.dynamic_functional_connectivity(
    method="pearson",
    window_size=20,
    window_stride=5,
    tapering_function=None # aca podria ser hamming, gaussian, cosine, etc. None = rectangular
)

# como method deberias poder poner tu propia funcion. Deberia ser un callable que le das las series de tiempo
# y te devuelve la matriz de conectividad

def mi_conectividad(series_de_tiempo):
    # devuelve una matriz
    pass

# y aca la usas, y pones el resto de los parametros
brain_states.dynamic_functional_connectivity(method=mi_conectividad, ...)

# podes acceder a esas matrices generadas si te interesa verlas
matrices_de_conectividad = brain_states.dynamic_matrices_

# 2) encuentra el k optimo basado en el elbow method o algun otro
k_list, y_list = brain_states.find_optimal_k(clustering="kmeans", method="elbow")

# aca harias un plot para encontrar el optimo por ejemplo
plt.plot(k_list, y_list)

# supongamos que decidis el k, ahora queda correr el clustering
k_optim = 5
brain_states.run_clustering(clustering="kmeans", k=k_optim, subsample=None) #subsample es por recursos computacionales

# podes acceder a los centroides (brain states), en este caso van a ser k_optim matrices ordenadas por entropia
states = brain_states.states_

# podes graficarlas
plt.imshow(states[0])

# ahora clasificamos cada matriz de conectividad de acuerdo al 
# clustering, podes elegir que tipo de distancia

brain_states.classify(distance="euclidean") # podria ser euclidean, manhattan (cityblock), L3 etc

# y esto devolveria todas las distribuciones
brain_states.probability_distribution_

# eventualmente se podria incluir funciones de ploteo por ejemplo para plotear la distribucion por cada grupo
brain_states.plot_distribution()
```
