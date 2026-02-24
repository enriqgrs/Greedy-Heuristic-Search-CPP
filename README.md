# Greedy-Heuristic-Search-CPP

Implementación de un motor de optimización basado en búsqueda voraz (*Greedy Search*) y heurísticas avanzadas para la reconstrucción aproximada de imágenes. El proyecto se centra en la exploración eficiente de espacios de estados masivos, utilizando C++ de alto rendimiento para minimizar el error cuadrático medio (MSE) mediante trazados lineales interconectados.

### Contribuciones y Evolución:

El desarrollo se centró en la creación de un algoritmo de búsqueda capaz de tomar decisiones localmente óptimas bajo restricciones computacionales severas:

- **Algoritmo Voraz (Greedy Search)**: Implementación de una lógica de decisión basada en el algoritmo de **Bresenham**, que evalúa y selecciona en cada iteración el estado sucesor que maximiza la reducción del error global.
- **Optimización de Búsqueda con Top-S Heap**: Uso de colas de prioridad (**Min-Heap**) para gestionar los mejores candidatos de cada iteración, optimizando la complejidad asintótica de la búsqueda de $O(N \log N)$ a $O(N \log S)$.
- **Paralelización y Concurrencia**: Diseño de un sistema multihilo mediante **std::thread** para la evaluación paralela de candidatos, logrando un *speedup* lineal y reduciendo drásticamente el tiempo de convergencia del algoritmo.
- **Heurística de Gradiente Informada**: Integración de un mapa de importancia basado en gradientes para guiar la búsqueda hacia zonas de alta densidad de información (contornos), mejorando la fidelidad con menor coste computacional.

### Donde se analiza:

- **Complejidad y Estructuras de Datos**: Evaluación del rendimiento de vectores unidimensionales frente a matrices para maximizar la localidad de datos y la eficiencia de la caché del procesador durante la búsqueda voraz.
- **Benchmarking de Convergencia**: Análisis de la evolución del MSE frente al número de iteraciones, determinando el punto de rendimiento decreciente de la heurística.
- **Escalabilidad Multinúcleo**: Estudio del impacto de la concurrencia en la fase de simulación de candidatos, analizando la eficiencia de la sincronización de hilos en tareas de cálculo intensivo.

### Uso de tecnologías:

C++ (11/14/17), Multithreading (std::thread), Algoritmos Greedy, Heurísticas de Búsqueda, Complejidad Asintótica y Shell Scripting para automatización de benchmarks.
