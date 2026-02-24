/**
 * main_bench.cpp — String Art con heurísticas configurables en compilación
 *
 * Heurísticas activables con flags -D al compilar usando compilacion condicional:
 *   USE_WEIGHTED_NAIL   → Selección ponderada del clavo inicial
 *   USE_TOP_S_HEAP      → Min-heap para los S mejores candidatos
 *   USE_GRADIENT_MAP    → Mapa de importancia por gradiente
 *   USE_MULTITHREADING  → Evaluación paralela con std::thread
 *
 * Uso: ./ejecutable <n_clavos> <n_hilos_calculo> <n_hilos_seleccion> <imagen> <resultado> [intensidad]
 * Siendo [] opcionales
 */

#include <fstream>
#include <iostream>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <ctime>
#include <chrono>
#include <queue>
#include <cmath>
#include <numeric>
#include <string>

#ifdef USE_MULTITHREADING
#include <thread>
#include <mutex>
#endif

#include "image_loader.h"
#include "geometry.h"

using namespace std;

/**
 * Implementa el algoritmo de Bresenham con error incremental:
 * trabaja exclusivamente con enteros y evita divisiones y operaciones
 * de punto flotante en el bucle interior.
 *
 * Modos de operación controlados por solo_simular:
 *   true  → Calcula cuánto se reduciría el error cuadrático respecto a
 *            target_img si se dibujase el hilo, sin modificar el lienzo.
 *            Permite evaluar candidatos sin efecto secundario.
 *   false → Aplica la reducción de intensidad sobre el lienzo img.
 *
 * @param img            Lienzo actual (se modifica solo si !solo_simular).
 * @param X1,Y1,X2,Y2   Coordenadas de los extremos del segmento.
 * @param intensidad_hilo Valor que se resta a cada píxel del segmento.
 * @param solo_simular   Si true, solo evalúa; si false, dibuja.
 * @param target_img     Imagen objetivo (requerida en modo simulación).
 * @param mapa_importancia Ponderación por píxel para el cálculo de error.
 * @return Reducción total de error cuadrático acumulada (0 si dibuja).
 */
long LineaBresenham(Imagen& img, int X1, int Y1, int X2, int Y2,
                    int intensidad_hilo,
                    bool solo_simular,
                    const Imagen* target_img,
                    const vector<double>* mapa_importancia)
{
    long reduccion_total = 0;

    /**
     * Lambda interna que procesa un único píxel.
     * En modo simulación: acumula la diferencia de error cuadrático
     *   (error_antes − error_despues) ponderada por mapa_importancia.
     * En modo dibujo: aplica la resta de intensidad con saturación a 0.
     */
    auto procesar_pixel = [&](int X, int Y) {
        if (X < 0 || X >= img.ancho || Y < 0 || Y >= img.alto) return;
        int intensidadActual = img.at(X, Y);

        if (solo_simular && target_img) {
            int valor_objetivo = target_img->at(X, Y);
            int lienzo_nuevo   = max(0, intensidadActual - intensidad_hilo);
            // Error cuadrático antes y después de aplicar el hilo
            long error_antes   = (long)(intensidadActual - valor_objetivo) * (intensidadActual - valor_objetivo);
            long error_despues = (long)(lienzo_nuevo     - valor_objetivo) * (lienzo_nuevo     - valor_objetivo);
            double importancia = mapa_importancia ? (*mapa_importancia)[Y * img.ancho + X] : 1.0;
            reduccion_total += (long)((error_antes - error_despues) * importancia);
        } else {
            img.at(X, Y) = max(0, intensidadActual - intensidad_hilo);
        }
    };

    // Caso degenerado: ambos extremos coinciden, procesar un solo píxel
    if (X1 == X2 && Y1 == Y2) {
        procesar_pixel(X1, Y1);
        return reduccion_total;
    }

    // Calcular desplazamientos y sentidos de avance para cada eje
    int dY = (Y2 - Y1), dX = (X2 - X1);
    int IncYi = (dY >= 0) ? 1 : (dY = -dY, -1); // incremento irregular en Y
    int IncXi = (dX >= 0) ? 1 : (dX = -dX, -1); // incremento irregular en X

    int IncYr, IncXr;
    if (dX >= dY) { IncYr = 0; IncXr = IncXi; }
    else          { IncXr = 0; IncYr = IncYi; swap(dX, dY); }

    int X = X1, Y = Y1;
    /**
     * Variables de error incremental de Bresenham (puramente enteras):
     *   avR = 2*dY       → avance del error en el paso regular
     *   av  = avR - dX   → error inicial
     *   avI = av  - dX   → avance del error en el paso irregular
     * El signo de av decide si dar el paso regular o el irregular.
     */
    int avR = 2 * dY;
    int av  = avR - dX;
    int avI = av  - dX;

    do {
        procesar_pixel(X, Y);
        // av >= 0: el error acumulado supera el umbral → paso irregular (diagonal)
        // av <  0: aún dentro del umbral → paso regular (eje dominante)
        if (av >= 0) { X += IncXi; Y += IncYi; av += avI; }
        else         { X += IncXr; Y += IncYr; av += avR; }
    } while (X != X2 || Y != Y2);
    procesar_pixel(X2, Y2); // asegurar que el extremo final queda procesado

    return reduccion_total;
}

/**
 * calcularMSE — Mean Squared Error entre el lienzo generado y el objetivo.
 *
 * MSE = (1/N) · Σ (resultado[i] − objetivo[i])²
 */
double calcularMSE(const Imagen& resultado, const Imagen& objetivo) {
    double suma = 0.0;
    size_t n = resultado.pixeles.size();
    for (size_t i = 0; i < n; ++i) {
        double d = resultado.pixeles[i] - objetivo.pixeles[i];
        suma += d * d;
    }
    return suma / n;
}

int main(int argc, char* argv[]) {

    if (argc < 6) {
        cerr << "Uso: " << argv[0]
             << " <n_clavos> <n_hilos_calculo> <n_hilos_seleccion> <imagen> <resultado> [intensidad_hilo]\n";
        return 1;
    }

    int    n_clavos         = atoi(argv[1]);
    int    n_hilos_calculo  = atoi(argv[2]);
    int    n_hilos_seleccion= atoi(argv[3]);
    string imagen_entrada   = argv[4];
    string archivo_salida   = argv[5];
    int    intensidad_hilo  = (argc >= 7) ? atoi(argv[6]) : 17;

    // Imprimir qué heurísticas están activas
    cerr << "=== Heurísticas activas ===\n";
#ifdef USE_WEIGHTED_NAIL
    cerr << "  [ON]  Selección ponderada de clavo inicial\n";
#else
    cerr << "  [OFF] Selección ponderada de clavo inicial (aleatorio)\n";
#endif
#ifdef USE_TOP_S_HEAP
    cerr << "  [ON]  Min-Heap top-S candidatos\n";
#else
    cerr << "  [OFF] Min-Heap top-S (vector + sort completo)\n";
#endif
#ifdef USE_GRADIENT_MAP
    cerr << "  [ON]  Mapa de importancia por gradiente\n";
#else
    cerr << "  [OFF] Mapa de importancia por gradiente (uniforme=1.0)\n";
#endif
#ifdef USE_MULTITHREADING
    cerr << "  [ON]  Multithreading\n";
#else
    cerr << "  [OFF] Multithreading (secuencial)\n";
#endif

    Imagen img = cargarPGM(imagen_entrada);
    if (img.ancho == 0 || img.alto == 0) {
        cerr << "Error al cargar la imagen.\n";
        return 1;
    }

    Imagen target_img = img;
    fill(img.pixeles.begin(), img.pixeles.end(), 255); // lienzo blanco

    vector<Punto> clavos = generarClavos(n_clavos, img.ancho, img.alto);
    int num_clavos_reales = (int)clavos.size();

    vector<double> mapa_importancia(img.ancho * img.alto, 1.0);

#ifdef USE_GRADIENT_MAP
    /**
     * Mapa de importancia por gradiente:
     * Para cada píxel interior se estima la magnitud del gradiente local
     * usando diferencias centrales en ambos ejes:
     *   dx = |f(x+1,y) − f(x−1,y)|   (gradiente horizontal)
     *   dy = |f(x,y+1) − f(x,y−1)|   (gradiente vertical)
     *   magnitud = sqrt(dx² + dy²)
     *
     * La importancia resultante escala desde 1.0 (zona plana) hasta 3.0
     * (zona de alto contraste).
     *
     * Efecto: los hilos colocados sobre bordes y texturas puntúan más alto,
     * priorizando la fidelidad en las zonas de mayor detalle visual.
     */
    for (int y = 1; y < target_img.alto - 1; ++y) {
        for (int x = 1; x < target_img.ancho - 1; ++x) {
            int dx = abs(target_img.at(x+1, y) - target_img.at(x-1, y));
            int dy = abs(target_img.at(x, y+1) - target_img.at(x, y-1));
            double magnitud = sqrt((double)(dx*dx + dy*dy));
            double importancia = 1.0 + (magnitud / 120.0);
            importancia = min(3.0, importancia); // cap para evitar dominancia excesiva de bordes
            mapa_importancia[y * img.ancho + x] = importancia;
        }
    }
    // Si USE_GRADIENT_MAP no está definido, mapa_importancia permanece todo a 1.0 (uniforme)
#endif

    // ── Pesos para selección del clavo inicial ─────────────────────────────
    vector<double> pesos_oscuridad(num_clavos_reales, 1.0); // base uniforme

#ifdef USE_WEIGHTED_NAIL
    {
        /**
         * Cálculo del peso de oscuridad por clavo:
         * Cada clavo se sitúa en el perímetro del círculo. Para decidir con qué
         * probabilidad seleccionarlo como punto de partida, se mide la oscuridad
         * media de los píxeles situados en la zona interior próxima a él.
         *
         * El radio de muestreo se deriva del arco entre clavos consecutivos,
         * de modo que se examina aproximadamente medio espacio interclavos.
         *
         * El vector (vx, vy) = centro − clavo apunta hacia el interior del círculo.
         * El producto escalar dx·vx + dy·vy > 0 garantiza que solo se consideran
         * píxeles que se encuentran en el hemiciclo interior al clavo,
         * descartando los que quedan "detrás" (fuera del círculo).
         *
         * Peso = oscuridad media = promedio de (255 − valor_píxel) en la zona:
         *   255 → negro puro → oscuridad máxima → más probable que se elija.
         *   0   → blanco puro → oscuridad nula → menos probable.
         */
        double radio_circulo          = min(img.ancho, img.alto) / 2.0;
        double distancia_entre_clavos = (2 * M_PI * radio_circulo) / num_clavos_reales;
        int radio_muestreo = max(1, (int)(distancia_entre_clavos / 2.0));
        int centro_x = img.ancho / 2, centro_y = img.alto / 2;

        for (int i = 0; i < num_clavos_reales; ++i) {
            int clavo_x = clavos[i].x, clavo_y = clavos[i].y;
            // Vector desde el clavo hacia el centro: define el hemiciclo interior
            int vx = centro_x - clavo_x, vy = centro_y - clavo_y;
            double osc = 0.0; int cnt = 0;
            for (int dy = -radio_muestreo; dy <= radio_muestreo; ++dy)
                for (int dx = -radio_muestreo; dx <= radio_muestreo; ++dx) {
                    // Producto escalar: descarta píxeles en el hemiciclo exterior
                    if (dx * vx + dy * vy <= 0) continue;
                    int px = clavo_x + dx, py = clavo_y + dy;
                    if (px < 0 || px >= target_img.ancho || py < 0 || py >= target_img.alto) continue;
                    osc += (255 - target_img.at(px, py)); // mayor valor → píxel más oscuro
                    cnt++;
                }
            pesos_oscuridad[i] = cnt > 0 ? osc / cnt : 0.0;
        }
    }
#endif

    /**
     * Construcción de la distribución acumulada de pesos (prefix sum):
     * pesos_acumulados[i] = suma de pesos_oscuridad[0..i].
     * Permite muestrear la distribución de probabilidad no uniforme
     * de forma eficiente mediante búsqueda binaria en O(log N).
     */
    vector<double> pesos_acumulados(num_clavos_reales);
    pesos_acumulados[0] = pesos_oscuridad[0];
    for (int i = 1; i < num_clavos_reales; ++i)
        pesos_acumulados[i] = pesos_acumulados[i-1] + pesos_oscuridad[i];
    double peso_total = pesos_acumulados[num_clavos_reales - 1];

    srand(69); // semilla fija para reproducibilidad en benchmark

    /**
     * seleccionar_clavo — Muestreo ponderado por búsqueda binaria.
     *
     * Con USE_WEIGHTED_NAIL:
     *   Se genera un número aleatorio r uniformemente en [0, peso_total).
     *   Se busca el primer índice i tal que pesos_acumulados[i] >= r
     *   mediante búsqueda binaria sobre el prefix sum (O(log N)).
     *   Esto equivale a muestrear la distribución discreta definida por
     *   los pesos de oscuridad sin construir explícitamente la distribución.
     *
     * Sin USE_WEIGHTED_NAIL:
     *   Selección aleatoria uniforme entre todos los clavos.
     */
    auto seleccionar_clavo = [&]() -> int {
#ifdef USE_WEIGHTED_NAIL
        if (peso_total <= 0) return rand() % num_clavos_reales;
        double r = ((double)rand() / RAND_MAX) * peso_total;
        int izq = 0, der = num_clavos_reales - 1;
        // Búsqueda binaria: busca el primer i con pesos_acumulados[i] >= r
        while (izq < der) {
            int mid = (izq + der) / 2;
            if (pesos_acumulados[mid] < r) izq = mid + 1;
            else der = mid;
        }
        return izq;
#else
        return rand() % num_clavos_reales;
#endif
    };

    struct Candidato { int u, v; long reduccion; };

    // Bucle principal greedy
    int fallos_consecutivos = 0;
    const int MAX_FALLOS    = 5;
    int iteracion           = 0;
    int total_hilos_dibujados = 0;

    vector<pair<int,int>> lineas_resultado;

    auto t_inicio = chrono::high_resolution_clock::now();
    // Condiciones de parada
    while (fallos_consecutivos < MAX_FALLOS && total_hilos_dibujados < n_hilos_calculo) {
        iteracion++;
        int u = seleccionar_clavo();
        int hilos_restantes = n_hilos_calculo - total_hilos_dibujados;
        int n_seleccionar   = min(n_hilos_seleccion, hilos_restantes);

        vector<Candidato> mejores_candidatos;

#ifdef USE_TOP_S_HEAP
        /**
         * Selección Top-S mediante Min-Heap — O(N log S) por iteración:
         * Se mantiene un heap de tamaño máximo S cuya raíz es siempre
         * el candidato con MENOR reducción entre los S mejores actuales.
         * Para cada candidato nuevo:
         *   - Si el heap tiene menos de S elementos, se inserta directamente.
         *   - Si su reducción supera la de la raíz (el peor de los S mejores),
         *     se extrae la raíz y se inserta el nuevo candidato.
         * Coste: O(log S) por candidato, O(N log S) total (vs O(N log N) con sort).
         * El comparador cmp_heap invierte el orden (mayor arriba → min-heap
         * sobre reduccion invierte para tener en la raíz el mínimo de los S).
         */
        auto cmp_heap = [](const Candidato& a, const Candidato& b){ return a.reduccion > b.reduccion; };

#ifdef USE_MULTITHREADING
        /**
         * Variante multihilo del heap Top-S:
         * Se reparte el rango de clavos [0, num_clavos_reales) en num_threads
         * particiones contiguas (con distribución uniforme del resto via rem).
         * Cada hilo mantiene su propio heap local (sin contención de mutex),
         * lo que permite explorar candidatos en paralelo sin sincronización.
         *
         * Partición: chunk = N/T, los primeros rem hilos reciben un clavo extra
         * para cubrir el residuo de la división entera.
         *
         * Tras el join de todos los hilos, los heaps locales se fusionan en uno
         * global de tamaño S aplicando la misma lógica de inserción condicional.
         * Coste de fusión: O(T·S·log S), despreciable frente a O(N log S).
         */
        unsigned int num_threads = thread::hardware_concurrency();
        if (num_threads == 0) num_threads = 4; // fallback si el SO no reporta núcleos

        // Un heap local por hilo para evitar contención entre threads
        vector<priority_queue<Candidato, vector<Candidato>, decltype(cmp_heap)>> heaps_locales;
        for (unsigned i = 0; i < num_threads; ++i)
            heaps_locales.emplace_back(cmp_heap);

        // Función de trabajo: evalúa los candidatos del rango [start, end)
        auto worker = [&](int tid, int start, int end) {
            auto& h = heaps_locales[tid];
            for (int v = start; v < end; ++v) {
                if (v == u) continue; // no conectar un clavo consigo mismo
                if (clavos[u].x == clavos[v].x && clavos[u].y == clavos[v].y) continue; // clavos coincidentes
                long red = LineaBresenham(img, clavos[u].x, clavos[u].y,
                                         clavos[v].x, clavos[v].y,
                                         intensidad_hilo, true, &target_img, &mapa_importancia);
                if (red <= 0) continue; // descartar candidatos que no mejoran el error
                // Mantener el heap de tamaño máximo S con los mejores candidatos
                if ((int)h.size() < n_seleccionar) h.push({u, v, red});
                else if (red > h.top().reduccion)  { h.pop(); h.push({u, v, red}); }
            }
        };

        // Lanzar hilos con partición balanceada del array de clavos
        vector<thread> threads;
        int chunk = num_clavos_reales / num_threads;
        int rem   = num_clavos_reales % num_threads;
        int st = 0;
        for (unsigned i = 0; i < num_threads; ++i) {
            int en = st + chunk + (i < (unsigned)rem ? 1 : 0); // primeros rem hilos tienen un elemento extra
            threads.emplace_back(worker, i, st, en);
            st = en;
        }
        for (auto& t : threads) t.join(); // esperar a que todos terminen

        // Fusionar los heaps locales en uno global de tamaño S
        priority_queue<Candidato, vector<Candidato>, decltype(cmp_heap)> heap_global(cmp_heap);
        for (auto& hl : heaps_locales)
            while (!hl.empty()) {
                auto c = hl.top(); hl.pop();
                if ((int)heap_global.size() < n_seleccionar) heap_global.push(c);
                else if (c.reduccion > heap_global.top().reduccion) { heap_global.pop(); heap_global.push(c); }
            }

        // Extraer los mejores candidatos del heap global al vector de resultados
        while (!heap_global.empty()) {
            mejores_candidatos.push_back(heap_global.top());
            heap_global.pop();
        }

#else  // USE_TOP_S_HEAP secuencial (sin multithreading)
        /**
         * Versión secuencial del heap TopS:
         * Misma lógica que la variante multihilo pero sobre un único heap
         * que recorre todos los clavos candidatos en orden.
         */
        priority_queue<Candidato, vector<Candidato>, decltype(cmp_heap)> heap(cmp_heap);
        for (int v = 0; v < num_clavos_reales; ++v) {
            if (v == u) continue;
            if (clavos[u].x == clavos[v].x && clavos[u].y == clavos[v].y) continue;
            long red = LineaBresenham(img, clavos[u].x, clavos[u].y,
                                      clavos[v].x, clavos[v].y,
                                      intensidad_hilo, true, &target_img, &mapa_importancia);
            if (red <= 0) continue;
            if ((int)heap.size() < n_seleccionar) heap.push({u, v, red});
            else if (red > heap.top().reduccion)  { heap.pop(); heap.push({u, v, red}); }
        }
        while (!heap.empty()) {
            mejores_candidatos.push_back(heap.top());
            heap.pop();
        }
#endif // USE_MULTITHREADING (dentro de USE_TOP_S_HEAP)

#else  // Sin USE_TOP_S_HEAP: vector completo + sort O(N log N)
        /**
         * Fallback sin heap: se evalúan todos los candidatos, se almacenan
         * en un vector y se ordenan de mayor a menor reducción.
         * Coste: O(N log N) por iteración, más costoso que el heap O(N log S)
         * cuando S << N, pero más simple de implementar.
         * Se conservan solo los primeros S elementos tras el sort.
         */
        vector<Candidato> todos_candidatos;
        todos_candidatos.reserve(num_clavos_reales);

        for (int v = 0; v < num_clavos_reales; ++v) {
            if (v == u) continue;
            if (clavos[u].x == clavos[v].x && clavos[u].y == clavos[v].y) continue;
            long red = LineaBresenham(img, clavos[u].x, clavos[u].y,
                                      clavos[v].x, clavos[v].y,
                                      intensidad_hilo, true, &target_img, &mapa_importancia);
            if (red > 0) todos_candidatos.push_back({u, v, red});
        }

        // Ordenar de mayor a menor reducción y conservar los S mejores
        sort(todos_candidatos.begin(), todos_candidatos.end(),
             [](const Candidato& a, const Candidato& b){ return a.reduccion > b.reduccion; });

        int n_tomar = min(n_seleccionar, (int)todos_candidatos.size());
        mejores_candidatos.assign(todos_candidatos.begin(), todos_candidatos.begin() + n_tomar);
#endif // USE_TOP_S_HEAP

        if (mejores_candidatos.empty()) {
            fallos_consecutivos++;
            continue;
        }
        fallos_consecutivos = 0;

        // Aplicar hilos seleccionados
        for (int i = 0; i < (int)mejores_candidatos.size() && total_hilos_dibujados < n_hilos_calculo; ++i) {
            int cu = mejores_candidatos[i].u;
            int cv = mejores_candidatos[i].v;
            LineaBresenham(img, clavos[cu].x, clavos[cu].y, clavos[cv].x, clavos[cv].y, intensidad_hilo,
                           false, nullptr, nullptr);
            lineas_resultado.push_back({cu, cv});
            total_hilos_dibujados++;
        }
    }

    auto t_fin   = chrono::high_resolution_clock::now();
    long ms      = chrono::duration_cast<chrono::milliseconds>(t_fin - t_inicio).count();

    // Calcular MSE final (lienzo vs. objetivo)
    double mse = calcularMSE(img, target_img);

    // Guardar resultado
    guardarPGM(archivo_salida, img);

    // Línea BENCH parseable por el script
    cerr << "BENCH: tiempo_ms=" << ms
         << " hilos_dibujados=" << total_hilos_dibujados
         << " mse=" << fixed << mse << "\n";

    // Salida estándar resumida
    cout << "Hilos: " << total_hilos_dibujados
         << " | Tiempo: " << ms << " ms"
         << " | MSE: " << fixed << mse << "\n";

    return 0;
}
