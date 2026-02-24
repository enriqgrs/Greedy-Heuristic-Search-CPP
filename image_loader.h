#ifndef IMAGE_LOADER_H
#define IMAGE_LOADER_H

#include <vector>
#include <string>

//Estructura que almacena la imagen en un vector plano.
struct Imagen {
    int ancho = 0;
    int alto = 0;

    //Vector aplanado que representa la totalidad de la imagen
    std::vector<int> pixeles;

    //Intermediario para acceder a los píxeles aplanados de forma directa, aumentando la legibilidad.
    //Operacion de modificacion para la definicion del resultado
    int& at(int x, int y) {
        return pixeles[y * ancho + x];
    }
    
    //Operacion de consulta
    int at(int x, int y) const {
        return pixeles[y * ancho + x];
    }
};

// Carga una imagen PGM (ASCII P2)
Imagen cargarPGM(const std::string& filename);

// Guarda una imagen en formato PGM (ASCII P2)
void guardarPGM(const std::string& filename, const Imagen& img);

#endif // IMAGE_LOADER_H
