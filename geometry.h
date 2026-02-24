#ifndef GEOMETRY_H
#define GEOMETRY_H

#include <vector>
#include <cmath>
#include <algorithm>

struct Punto {
    int x, y;
};

// Genera N clavos dispuestos en círculo centrado en la imagen.
inline std::vector<Punto> generarClavos(int n, int anchura, int altura) {
    std::vector<Punto> clavos;
    clavos.reserve(n);
    
    double centroX = anchura / 2.0;
    double centroY = altura / 2.0;
    double radio = std::min(anchura, altura) / 2.0 - 1.0; // -1 para evitar salirnos de rango
    for (int i = 0; i < n; ++i) {
        double angulo = 2.0 * M_PI * i / n;
        int x = static_cast<int>(centroX + radio * std::cos(angulo));
        int y = static_cast<int>(centroY + radio * std::sin(angulo));
        
        // Ajustar a coordenadas válidas por si acaso
        x = std::max(0, std::min(x, anchura - 1));
        y = std::max(0, std::min(y, altura - 1));
        
        // Evitar duplicados consecutivos si la resolución no da para más
        if (!clavos.empty() && clavos.back().x == x && clavos.back().y == y) {
            continue;
        }

        clavos.push_back({x, y});
    }

    // Opcional: Verificar que el último no sea igual al primero (cierre del círculo)
    if (clavos.size() > 1 && clavos.back().x == clavos.front().x && clavos.back().y == clavos.front().y) {
        clavos.pop_back();
    }
    return clavos;
}

#endif // GEOMETRY_H
