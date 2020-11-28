#include <stdio.h>

//Se define una estructura
typedef struct
{
	int width;
	int height;
	int deep;
	float* elements;
}   Matrix;

//Tama�o del bloque de hilos
#define BLOCK_SIZE 16

__global__ void MatMulKernel(const Matrix, Matrix);