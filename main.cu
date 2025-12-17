#include <cuda.h>
#include <omp.h>
#include <cstdlib>
#include <iostream>
#include <string>
#include <random>

// Colores :)
#define ROJO "\033[31m"
#define VERDE "\033[32m"
#define AMARILLO "\033[33m"
#define AZUL "\033[34m"
#define MAGENTA "\033[35m"
#define CIAN "\033[36m"
#define NARANJO "\033[38;5;208m"
#define RESET "\033[0m"

// Define tamaño bloque a procesar en shared memory (Ej: sub-bloques 16*16)
// 16 es valor más comun y eficnete en mayoria de GPUS
// 8 es útil en GPUs con menos recursos
// 32 puede aprovechar mejor GPUs recientes, pero usa más recursos por bloque
#define TILE_SIZE 16

void print_matriz(float *m, int n, const char *msg){
	if(n > 40){
		return;
	}
	printf("%s: \n", msg);
	for(int i=0; i<n; ++i){
		for(int j=0; j<n; ++j){
			printf("%f ", m[i*n +j]);
		}
		printf("\n");
	}
	printf("\n");
}

void matmul_cpu(int n, float *A, float *B, float *C){
	#pragma omp parallel for
	for(int i=0; i<n; ++i){
		for(int j=0; j<n; ++j){
			// fila por columna
			float sum = 0.0f;
			for(int k=0; k<n; ++k){
				sum += A[i*n + k] * B[k*n + j];
			}
			C[i*n + j] = sum;
		}
	}
}

__global__ void kernel_matmul_gpu(int n, float *A, float *B, float *C){
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int ty = blockIdx.y * blockDim.y + threadIdx.y;
    float sum = 0.0f;

    for (int k = 0; k < n; ++k)
        sum += A[ty*n + k] * B[k*n + tx];

    C[ty*n + tx] = sum;
}

__global__ void kernel_matmul_gpusm(int n, float *A, float *B, float *C){
    // Declaración memoria compartida
    __shared__ float tile_A[TILE_SIZE*TILE_SIZE];
    __shared__ float tile_B[TILE_SIZE*TILE_SIZE];

    // Calculo Índices de Threads
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;

    for (int i = 0; i < (n + TILE_SIZE -1)/TILE_SIZE; ++i){
        // Cargando sub-bloques en memoria compartida
        // Cada thread carga un elemto de A y B a la variable de memoria compartida
        // Si el índice esta fuera de limites, se le asigna 0 para los casos donde la matriz
        // no es un múltiplo de TILE_SIZE
        if (row < n && (i * TILE_SIZE + threadIdx.x) < n)
            tile_A[threadIdx.y * TILE_SIZE + threadIdx.x] = A[row * n + i * TILE_SIZE + threadIdx.x];
        else
            tile_A[threadIdx.y * TILE_SIZE + threadIdx.x] = 0.0f;
    
        if (col < n && (i * TILE_SIZE + threadIdx.y) < n)
            tile_B[threadIdx.y * TILE_SIZE + threadIdx.x] = B[(i * TILE_SIZE + threadIdx.y) * n + col];
        else
            tile_B[threadIdx.y * TILE_SIZE + threadIdx.x] = 0.0f;
        // Sincronización de Threads
        __syncthreads();
        // Multiplicación
        for (int j = 0; j < TILE_SIZE; j++)
            sum += tile_A[threadIdx.y * TILE_SIZE + j] * tile_B[j * TILE_SIZE + threadIdx.x];

        // Sincronización de Threads
        __syncthreads();
    }
     // Guardando resultado devuelta a memoria global
    if(row < n && col < n)    
        C[row * n + col] = sum; 

}


int main(int argc, char **argv) 
{
    // 1) argumentos
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

    // 2) alocar memoria arrays A, B, C
	float *A = new float[n*n];  
	float *B = new float[n*n];  
	float *C = new float[n*n];  

    // 3) init arrays A, B
	std::mt19937 gen(1313); 
    std::uniform_real_distribution<> dis(0.0, 1.0); 

	for(long i=0; i<n*n; ++i){
		A[i] = (float) dis(gen);
		//A[i] = (float) (rand() % 10);
		B[i] = (float) dis(gen);
		//B[i] = (float) (rand() % 10);
	}

    print_matriz(A, n, "A");	
    print_matriz(B, n, "B");	
    
    // 4.1) Preparar CPU
	omp_set_num_threads(nt);

    // 4.2) Prepara GPU

    cudaSetDevice(0);

    float *dA;
    float *dB;
    float *dC;

    cudaMalloc(&dA, sizeof(float)*n*n);
    cudaMalloc(&dB, sizeof(float)*n*n);
    cudaMalloc(&dC, sizeof(float)*n*n);

    cudaMemcpy(dA, A, sizeof(float)*n*n, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B, sizeof(float)*n*n, cudaMemcpyHostToDevice);

    
    // 5) Ejecutar Algoritmo Seleccionado
    double t1, t2;
    t1 = omp_get_wtime();
    std::string algoritmo_usado;

	switch (alg){
        case 1:
            algoritmo_usado = "CPU Multicore";
		    matmul_cpu(n, A, B, C);
            break;
        case 2:
            algoritmo_usado = "GPU Básico";
            dim3 block(512, 1, 1);
            dim3 grid( (n + block.x - 1)/block.x, 1, 1);
            kernel_matmul_gpu<<grid, block>>(n, dA, dB, dC);
            cudaDeviceSynchronize();
            break;
        case 3:
            algoritmo_usado = "GPU Memoria Compartida";
            dim3 block(TILE_SIZE, TILE_SIZE);
            dim3 grid((n + TILE_SIZE - 1) / TILE_SIZE, (n + TILE_SIZE - 1) / TILE_SIZE);
            kernel_matmul_gpusm<<grid, block>>(n, dA, dB, dC);
            cudaDeviceSynchronize();
            break;
        case 4:
            algoritmo_usado = "GPU Tensor Cores";
            break;
	}
	t2 = omp_get_wtime();
    cudaDeviceSynchronize();

    cudaMemcpy(C, dC, sizeof(float)*n*n, cudaMemcpyDeviceToHost);

	// 6) Ver resultado	
    print_matriz(C, n, "C");	
	printf("Algoritmo %s listo: %f secs\n", algoritmo_usado, t2-t1);

    // 7) Liberar memoria
    delete[] A;
    delete[] B;
    delete[] C;
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    
    return EXIT_SUCCESS;
}