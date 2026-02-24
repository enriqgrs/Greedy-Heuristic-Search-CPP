#include "image_loader.h"
#include <fstream>
#include <iostream>
#include <sstream>

using namespace std;

Imagen cargarPGM(const string& filename) {
    ifstream file(filename);
    Imagen img;
    
    if (!file.is_open()) {
        cerr << "Error: No se pudo abrir el archivo " << filename << endl;
        return img;
    }
    
    string line;
    string type;
    int maxVal;
    
    // Leer tipo mágico (P2)
    while (getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;
        stringstream ss(line);
        ss >> type;
        if (!type.empty()) break;
    }
    
    if (type != "P2") {
        cerr << "Error: Formato de archivo no soportado (se espera P2)" << endl;
        return img;
    }
    
    // Leer dimensiones
    while (getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;
        stringstream ss(line);
        if (ss >> img.ancho >> img.alto) break;
    }
    
    // Leer valor máximo
    while (getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;
        stringstream ss(line);
        if (ss >> maxVal) break;
    }
    
    // Redimensionar vector de píxeles
    img.pixeles.resize(img.ancho * img.alto);
    
    // Leer datos de píxeles
    int val;
    int count = 0;
    while (file >> val) {
        if (count < img.ancho * img.alto) {
            img.pixeles[count++] = val;
        }
    }
    
    if (count != img.ancho * img.alto) {
        cerr << "Advertencia: El número de píxeles leídos no coincide con las dimensiones" << endl;
    }
    
    return img;
}

void guardarPGM(const string& filename, const Imagen& img) {
    ofstream file(filename);
    
    if (!file.is_open()) {
        cerr << "Error: No se pudo crear el archivo " << filename << endl;
        return;
    }
    
    // Escribir cabecera PGM (P2 ASCII), estructura igual a la del test.
    file << "P2" << endl;
    file << "# Generado por algoritmo de arte con hilos" << endl;
    file << img.ancho << " " << img.alto << endl;
    file << "255" << endl;
    
    // Escribir píxeles
    for (int i = 0; i < img.alto * img.ancho; ++i) {
        file << img.pixeles[i];
        if ((i + 1) % img.ancho == 0) {
            file << endl;
        } else {
            file << " ";
        }
    }
    
    file.close();
}
