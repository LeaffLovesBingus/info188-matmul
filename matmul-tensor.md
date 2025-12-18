# Programación de MATMUL con _Tensor cores_
## Tensor cores
Los _tensor cores_ son núcleos de la GPU especialmente diseñados para el cálculo rápido de multiplicaciones y acumulaciones de matrices. Descritos en la página oficial de Nvidia como "unidades programables de multiplicación y acumulación de matrices". Estos núcleos solo son capaces de realizar multiplicaciones y acumulaciones de matrices de forma mucho más eficiente que al programar en CUDA cores.

![Figura 1](img/a100-ampere-architecture.png)
En esta imagen se puede apreciar la arquitectura _ampere_ de NVIDIA, en particular e la GPU `NVIDIA GA100` que contiene 108 _streaming multiprocessors_ (SM).

![Figura 2](img/a100-streaming-multiprocessor.png)
Este diagrama es un _close-up_ de cada SM. Cada SM contiene 4 tensor cores capaces de ejecutar múltiples operaciones MMA (_Matrix Multiply Accumulate_) por ciclo de reloj, por lo que la GPU mencionada cuenta con 432 Tensor cores en total.

_Sean las matrices `A`, `B` y `C` de tamaño `n * n`._
Los tensor cores realizan la operación `D = A * B + C`, donde las matrices multiplicadoras `A` y `B` deben ser números de punto flotante de 16 bits (`FP16` o tipo `half`) y la matriz de acumulación `C` puede ser tanto half como un número flotante de 32 bits (`FP32` o tipo `float`).

![Figura 3](img/tensor-operation.png)

Estas operaciones son realizadas por un `warp`, el cual es un bloque de 32 threads. A diferencia de la programación sobre _CUDA cores_, donde se programa la operación que se debe realizar en cada hilo, en los tensor cores se "orquesta" las submatrices a operar en cada warp.

Cada warp calcula un bloque de `16 x 16` de la matriz `A * B` utilizando operaciones `WMMA` _(Warp Matrix Multiply Accumulate*)_, que activan directamente los Tensor Cores de la tarjeta gráfica.

`WWMA` está definido en el header:
```cpp
#include <mma.h>
```

_**\*WMMA (Warp Matrix Multiply Accumulate)**: API de CUDA que permite utilizar tensor cores sin necesidad de escribir código ensamblador._

## MATMUL en tensor cores

Para multiplicar matrices con tensor cores, un warp completo coopera para calcular un _tile_ entero de `A * B`. Ese _tile_ se calcula con las operaciones de tipo `D = A * B + C` vistas anteriormente utilizando instrucciones de hardware. Como en este caso solo queremos programar la multiplicación de matrices, la matriz `C` será inicialmente una matriz nula, luego, tendrá los resultados parciales del cálculo de cada _tile_.

#### Restricciones para programar en Tensor cores
- Se opera sobre bloques fijos de threads (warps).
- Los inputs deben ser `half` o flotantes de 16 bits `FP16`.
- La acumulación es `FP32`.
- El cómputo ocurre por warp.

Al programar con `WMMA`, aparece el concepto de _fragmento_, el cual es una estructura lógica que representa una submatriz (en este caso a lo que nos referimos como _tile_) cuyo contenido está distribuido internamente entre los 32 threads del warp, utilizando registros de la GPU. 

En palabras simples, es una abstracción de `WMMA` para que el programador no se tenga que ocupar de:
- Qué thread guarda/procesa qué elementos.
- Cómo se mapean los registros.
- Qué datos se le pasa a cada tensor core.

### Kernel MATMUL en Tensor Cores
- Como primer paso, se define el namespace `nvcuda`:
```cpp
using namespace nvcuda;
```
`WMMA` vive dentro de `nvcuda::wmma`, por lo que dentro del kernel se asigna el namespace nvcuda para simplificar los llamados a la API.

- Se define la identidad del tile:
```cpp
// Identidad del tile
int tileRow = blockIdx.y;
int tileCol = blockIdx.x;

int row = tileRow * TENSOR_TILE_SIZE;
int col = tileCol * TENSOR_TILE_SIZE;
```

- Luego se define el fragmento para la matriz de acumulación. Como se mencionó anteriormente, como solo se va a realizar la multiplicación de matrices, el fragmento funciona inicialmente como una matriz nula:
```cpp
wmma::fragment<
    wmma::accumulator, 
    TENSOR_TILE_SIZE, 
    TENSOR_TILE_SIZE, 
    TENSOR_TILE_SIZE, 
    float
> c_frag;
wmma::fill_fragment(c_frag, 0.0f);
```
`wmma::accumulator` define el rol del fragmento como la matriz acumuladora.

`TENSOR_TILE_SIZE` está definido como 16, es el tamaño de la submatriz cuadrada a ser procesada.

`float` define el tipo de los datos para el fragmento, ya que la matriz acumuladora puede ser `FP32`.

- Se ejecuta el for que opera sobre todos los tiles:
```cpp
for (int k0 = 0; k0 < n; k0 += TENSOR_TILE_SIZE) {
    ...
}
```
El bucle divide la multiplicación en bloques de tamaño fijo `TENSOR_TILE_SIZE` compatibles con los _Tensor cores_.

#### Dentro del for...
- Se definen punteros a las submatrices con las que se va a operar:
```cpp
// Punteros al inicio de la submatriz (tile)
const half *a_tile = a + row * n + k0;
const half *b_tile = b + k0 * n + col;
```

- Se definen los fragmentos para las matrices `A` y `B`.
```cpp
// Fragmentos de entrada para los tensor cores
wmma::fragment<
    wmma::matrix_a,
    TENSOR_TILE_SIZE,
    TENSOR_TILE_SIZE,
    TENSOR_TILE_SIZE,
    half,
    wmma::row_major
> a_frag;
wmma::fragment<
    wmma::matrix_b,
    TENSOR_TILE_SIZE,
    TENSOR_TILE_SIZE,
    TENSOR_TILE_SIZE,
    half,
    wmma::col_major
> b_frag;
```
`wmma::matrix_a` y `wmma::matrix_b` definen los roles de cada matriz respectivamente

`half` define el tipo de dato de cada fragmento. En este caso se usa `half` puesto que las matrices a multiplicar deben ser `FP16`.

`wmma::row_major` y `wmma::col_major` definen el orden de lectura de los datos. Se leerán como columnas o filas contiguas en memoria respectivamente.

- Se cargan los _tiles_ a los fragmentos creados anteriormente, efectivamente moviendo los tiles a los registros del warp:
```cpp
// Cargar matrices desde la memoria global hacia los registros del warp
wmma::load_matrix_sync(a_frag, a_tile, n);
wmma::load_matrix_sync(b_frag, b_tile, n);
```

- Se ejecuta la multiplicación habiendo movido los datos:
```cpp
// Multiplicar las matrices (D = A * B + C)
wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
```

#### Habiendo ejecutado el for...
- Se guarda el resultado de la operación en la matriz acumuladora `C`:
```cpp
// Guardar resultado
float *c_tile = c + row * n + col;
wmma::store_matrix_sync(c_tile, c_frag, n, wmma::mem_row_major);
```
El puntero `c_tile` apunta a una submatriz específica de la matriz `C`.

`n` define los límites de la matriz para que no se escriban datos fuera del mismo.

`store_matrix_sync` guarda los datos del fragmento acumulador en la memoria global (gracias al puntero a la submatriz `c_tile`).
`mem_row_major` indica que la matriz está almacenada como filas contiguas en memoria.

|Specs GPU Bingus (GPU tests tensor)||
|------|--------------|
|Nombre|RTX 2060 Super|
|Procesador gráfico|TU106|
|Arquitectura|Turing|
|Reloj base|1470 MHz|
|Reloj boost|1650 MHz|
|VRAM|8 GB|
|Tipo VRAM|GDDR6|
|Ancho de banda VRAM|448.0 GB/s|
|CUDA Cores|2176|
|Tensor Cores|272|
|Rendimiento FP16 (half)|14.36 TFLOPS|
|Rendimiento FP32 (float)|7.181 TFLOPS|
|Rendimiento FP64 (double)|224.4 GFLOPS|
|Versión CUDA|7.5|

## Bibliografía
- [Documentación oficial de Nvidia sobre Tensor cores](https://developer.nvidia.com/blog/programming-tensor-cores-cuda-9/)
- [Blog explicativo sobre la programación sobre Tensor cores](https://leimao.github.io/blog/NVIDIA-Tensor-Core-Programming/)
- [Video explicativo sobre programación en Tensor cores](https://youtu.be/Yt1A-vaWTck?si=3m5EwRk-dz3hGI70)
- [Especificaciones RTX 2060 Super](https://www.techpowerup.com/gpu-specs/geforce-rtx-2060-super.c3441)
- Aclaración de definiciones específicas y traducción de material con _ChatGPT_