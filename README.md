# info188-matmul

## Hardware utilizado para los tests
### CPU
|Especificaciones CPU||
|-|-|
|Marca|AMD|
|Modelo|Ryzen 5 3600|
|Frecuencia|3.6 GHz|
|Frecuencia turbo|Hasta 4.2 GHz|
|Núcleos|6|
|Hilos|12|
|Cache L1|64 KB (por núcleo)|
|Cache L2|512 KB (por núcleo)|
|Cache L3|32 MB (compartido)|

### RAM
|Especificaciones RAM||
|-|-|
|Capacidad|32 GB (_Dual channel, 4x8 GB_)|
|Tipo|DDR4|
|Frecuencia|3000 MHz|

### GPU
|Especificaciones GPU||
|-|-|
|Marca|NVIDIA|
|Modelo|RTX 2060 Super|
|Procesador gráfico|TU106|
|Arquitectura|_Turing_|
|Reloj base|1470 MHz|
|Reloj boost|1650 MHz|
|VRAM|8 GB|
|Tipo VRAM|GDDR6|
|Ancho de banda VRAM|448.0 GB/s|
|CUDA Cores|2176|
|Tensor Cores|272|
|Tensor Cores por SM|8|
|Rendimiento FP16 (half)|14.36 TFLOPS|
|Rendimiento FP32 (float)|7.181 TFLOPS|
|Rendimiento FP64 (double)|224.4 GFLOPS|
|Versión CUDA|7.5|

## Referencias
#### Shared memory
- [Mutliplicación matrices memoria compartida](https://medium.com/@dhanushg295/mastering-cuda-matrix-multiplication-an-introduction-to-shared-memory-tile-memory-coalescing-and-d7979499b9c5)

#### Tensor cores
- [Documentación oficial de Nvidia sobre Tensor cores](https://developer.nvidia.com/blog/programming-tensor-cores-cuda-9/)
- [Blog explicativo sobre la programación sobre Tensor cores](https://leimao.github.io/blog/NVIDIA-Tensor-Core-Programming/)
- [Video explicativo sobre programación en Tensor cores](https://youtu.be/Yt1A-vaWTck?si=3m5EwRk-dz3hGI70)

#### Hardware
- [Especificaciones RTX 2060 Super](https://www.techpowerup.com/gpu-specs/geforce-rtx-2060-super.c3441)
- [Especificaciones Ryzen 5 3600](https://www.techpowerup.com/cpu-specs/ryzen-5-3600.c2132)