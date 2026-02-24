/**
 * calc_mse.cpp — Calcula MSE y PSNR entre dos imágenes PGM
 * Uso: ./calc_mse <original.pgm> <resultado.pgm>
 * Salida: MSE=X PSNR=Y
 */

#include <iostream>
#include <cmath>
#include "image_loader.h"

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Uso: " << argv[0] << " <original.pgm> <resultado.pgm>\n";
        return 1;
    }

    Imagen orig = cargarPGM(argv[1]);
    Imagen res  = cargarPGM(argv[2]);

    if (orig.ancho != res.ancho || orig.alto != res.alto) {
        std::cerr << "Error: las imágenes tienen distintas dimensiones.\n";
        return 1;
    }

    double suma = 0.0;
    size_t n = orig.pixeles.size();
    for (size_t i = 0; i < n; ++i) {
        double d = orig.pixeles[i] - res.pixeles[i];
        suma += d * d;
    }

    double mse  = suma / n;
    double psnr = (mse > 0) ? 10.0 * std::log10((255.0 * 255.0) / mse) : 999.0;

    std::cout << "MSE=" << mse << " PSNR=" << psnr << "\n";
    return 0;
}
