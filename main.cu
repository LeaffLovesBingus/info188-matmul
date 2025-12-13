#include <cuda.h>
#include <omp.h>
#include <cstdlib>
#include <iostream>

// Colores :)
#define ROJO "\033[31m"
#define VERDE "\033[32m"
#define AMARILLO "\033[33m"
#define AZUL "\033[34m"
#define MAGENTA "\033[35m"
#define CIAN "\033[36m"
#define NARANJO "\033[38;5;208m"
#define RESET "\033[0m"


int main(int argc, char **argv) 
{
    if (argc != 4) {
        std::cerr << AMARILLO << "Ejecutar como ./prog <n> <nt> <ALG>" << RESET << std::endl;
        exit(EXIT_FAILURE);
    }

    int n = atoi(argv[1]);
    int nt = atoi(argv[2]);
    int alg = atoi(argv[3]);

    if (alg < 1 || alg > 4) {
        std::cerr << ROJO << "Algoritmo " << MAGENTA << "[" << alg << "]" << ROJO << " inválido" << RESET << std::endl;
        exit(EXIT_FAILURE);
    }

    return EXIT_SUCCESS;
}