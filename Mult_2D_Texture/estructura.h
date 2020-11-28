#include <stdio.h>

//Se define una estructura
typedef struct
{
	int width;
	int height;
	float* elements;
}   Matrix;

//Tamaño del bloque de hilos
#define BLOCK_SIZE 16

__global__ void MatMulKernel(Matrix, Matrix, Matrix);