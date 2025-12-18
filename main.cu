#include <cuda.h>
#include <omp.h>
#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <string>
#include <random>
#include <cuda_fp16.h>  // Tipo half
#include <mma.h>        // Tensor cores


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
// 16 es valor más comun y eficiente en mayoria de GPUS
// 8 es útil en GPUs con menos recursos
// 32 puede aprovechar mejor GPUs recientes, pero usa más recursos por bloque
#define TILE_SIZE 16

//
#define TENSOR_TILE_SIZE 16
#define WARP_SIZE 32


void print_matriz(float *m, int n, const char *msg){
    if(n > 30){
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


void matmul_gpu(float *a, float *b, float *c, int n) {
    dim3 block(512, 1, 1);
    dim3 grid( (n + block.x - 1)/block.x, 1, 1);
    kernel_matmul_gpu<<<grid, block>>>(n, a, b, c);
    cudaDeviceSynchronize();
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


void matmul_gpu_shared_memory(float *a, float *b, float *c, int n) {
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((n + TILE_SIZE - 1) / TILE_SIZE, (n + TILE_SIZE - 1) / TILE_SIZE);
    kernel_matmul_gpusm<<<grid, block>>>(n, a, b, c);
    cudaDeviceSynchronize();
}


__global__ void kernel_matmul_gputs(half *a, half *b, float *c, int n) {
    using namespace nvcuda;

    // Identidad del tile
    int tileRow = blockIdx.y;
    int tileCol = blockIdx.x;

    int row = tileRow * TENSOR_TILE_SIZE;
    int col = tileCol * TENSOR_TILE_SIZE;

    // Creación e inicialización del fragmento acumulador
    // Como solo queremos multiplicar las matrices, la matriz acumuladora la llenamos de 0
    wmma::fragment<wmma::accumulator, TENSOR_TILE_SIZE, TENSOR_TILE_SIZE, TENSOR_TILE_SIZE, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);

    for (int k0 = 0; k0 < n; k0 += TENSOR_TILE_SIZE) {
        // Punteros al inicio de la submatriz (tile)
        const half *a_tile = a + row * n + k0;
        const half *b_tile = b + k0 * n + col;

        // Fragmentos de entrada para los tensor cores
        wmma::fragment<wmma::matrix_a, TENSOR_TILE_SIZE, TENSOR_TILE_SIZE, TENSOR_TILE_SIZE, half, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, TENSOR_TILE_SIZE, TENSOR_TILE_SIZE, TENSOR_TILE_SIZE, half, wmma::col_major> b_frag;

        // Cargar matrices desde la memoria global hacia los registros del warp
        wmma::load_matrix_sync(a_frag, a_tile, n);
        wmma::load_matrix_sync(b_frag, b_tile, n);

        // Multiplicar las matrices (D = A * B + C)
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    // Guardar resultado
    float *c_tile = c + row * n + col;
    wmma::store_matrix_sync(c_tile, c_frag, n, wmma::mem_row_major);
}


__global__ void float_to_half(const float *input, half *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n * n) {
        output[idx] = __float2half(input[idx]);
    }
}


void matmul_gpu_tensor_cores(float *a, float *b, float *c, int n) {
    // Transformar A y B a half*
    half *dA, *dB;
    cudaMalloc(&dA, sizeof(half) * n * n);
    cudaMalloc(&dB, sizeof(half) * n * n);

    int threads = 256;
    int blocks = (n * n + threads - 1) / threads;

    float_to_half<<<blocks, threads>>>(a, dA, n);
    float_to_half<<<blocks, threads>>>(b, dB, n);

    cudaDeviceSynchronize();

    // Lanzar MATMUL
    dim3 block(WARP_SIZE, 1, 1);    // 1 warp
    int tiles = (n + TENSOR_TILE_SIZE - 1) / TENSOR_TILE_SIZE;
    dim3 grid(tiles, tiles, 1);

    kernel_matmul_gputs<<<grid, block>>>(dA, dB, c, n);

    cudaDeviceSynchronize();

    cudaFree(dA);
    cudaFree(dB);
}


int main(int argc, char **argv) 
{
    std::cout << std::fixed << std::setprecision(10);

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

    // 4.2) Preparar GPU
    cudaSetDevice(0);

    float *dA;
    float *dB;
    float *dC;

    float *B_T;
    int n_pad;
    
    if (alg == 2 || alg == 3) {
        cudaMalloc(&dA, sizeof(float) * n * n);
        cudaMalloc(&dB, sizeof(float) * n * n);
        cudaMalloc(&dC, sizeof(float) * n * n);

        cudaMemcpy(dA, A, sizeof(float) * n * n, cudaMemcpyHostToDevice);
        cudaMemcpy(dB, B, sizeof(float) * n * n, cudaMemcpyHostToDevice);
    }

    else if (alg == 4) {
        // Trasponer la matriz B para que quede como columnas contiguas en memoria
        B_T = new float[n*n];
        for (int i = 0; i < n; ++i){
            for (int j = 0; j < n; ++j){
                B_T[j*n + i] = B[i*n + j];
            }
        }

        // Crear las matrices con padding para que estas puedan alcanzar un múltiplo de TENSOR_TILE_SIZE
        n_pad = ((n + TENSOR_TILE_SIZE - 1) / TENSOR_TILE_SIZE) * TENSOR_TILE_SIZE;

        float *a_pad = new float[n_pad * n_pad];
        float *b_pad = new float[n_pad * n_pad];

        for (int i = 0; i < n_pad * n_pad; ++i) {
            a_pad[i] = 0.0f;
            b_pad[i] = 0.0f;
        }

        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                a_pad[i * n_pad + j] = A[i * n + j];
                b_pad[i * n_pad + j] = B_T[i * n + j];
            }
        }

        // Crear matrices con padding en GPU
        cudaMalloc(&dA, sizeof(float) * n_pad * n_pad);
        cudaMalloc(&dB, sizeof(float) * n_pad * n_pad);
        cudaMalloc(&dC, sizeof(float) * n_pad * n_pad);

        cudaMemset(dA, 0, sizeof(float) * n_pad * n_pad);
        cudaMemset(dB, 0, sizeof(float) * n_pad * n_pad);

        cudaMemcpy(dA, a_pad, sizeof(float) * n_pad * n_pad, cudaMemcpyHostToDevice);
        cudaMemcpy(dB, b_pad, sizeof(float) * n_pad * n_pad, cudaMemcpyHostToDevice);

        delete [] a_pad;
        delete [] b_pad;
    }

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
            matmul_gpu(dA, dB, dC, n);
            break;
        case 3:
            algoritmo_usado = "GPU Memoria Compartida";
            matmul_gpu_shared_memory(dA, dB, dC, n);
            break;
        case 4:
            algoritmo_usado = "GPU Tensor Cores";
            matmul_gpu_tensor_cores(dA, dB, dC, n_pad);
            break;
	}
	t2 = omp_get_wtime();
    cudaDeviceSynchronize();

    if (alg == 2 || alg == 3) {
        cudaMemcpy(C, dC, sizeof(float) * n * n, cudaMemcpyDeviceToHost);
    }
    else if (alg == 4) {
        float *c_pad = new float[n_pad * n_pad];

        cudaMemcpy(c_pad, dC, sizeof(float) * n_pad * n_pad, cudaMemcpyDeviceToHost);
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                C[i*n + j] = c_pad[i*n_pad + j];
            }
        }

        delete [] c_pad;
    }

	// 6) Ver resultado
    print_matriz(C, n, "C");	
	//printf("%sAlgoritmo [%s] listo: %f secs\n", AZUL, algoritmo_usado, t2-t1);
    std::cout << AZUL << "Algoritmo " << CIAN << "[" + algoritmo_usado + "]" << AZUL << " terminó en " << t2-t1 << " segundos" << RESET << std::endl;

    // 7) Liberar memoria
    delete[] A;
    delete[] B;
    delete[] C;
    if (alg == 4) delete[] B_T;
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    
    return EXIT_SUCCESS;
}