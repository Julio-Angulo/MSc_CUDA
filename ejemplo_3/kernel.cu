//PROGRAMA QUE SUMA DOS VECTORES (a y b) Y ALMACENA EL RESULTADO EN EL VECTOR (c)
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#define N 1000

__global__ void add(int *a, int *b, int *c)
{
	//int tid = blockIdx.x;
	int tid = threadIdx.x;
	if(tid < N)
	{
		c[tid] = a[tid] + b[tid];
	}
    
}

int main()
{
	int a[N], b[N], c[N],i,j;
	int *dev_a, *dev_b, *dev_c;

	//Reservar memoria en el GPU
	cudaMalloc((void**)&dev_a,N*sizeof(int));
	cudaMalloc((void**)&dev_b,N*sizeof(int));
	cudaMalloc((void**)&dev_c,N*sizeof(int));

	//Se rellenan los arreglos 'a' y 'b' en el CPU
	for(i=0; i<N;i++)
	{
		a[i]=i+1;
		b[i]=i*i;
	}

	//Se copian los arreglos 'a' y 'b' a la GPU
	cudaMemcpy(dev_a,a,N*sizeof(int),cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b,b,N*sizeof(int),cudaMemcpyHostToDevice);

	//Se manda llamar el kernel
	//add<<<N,1>>>(dev_a,dev_b,dev_c);
	add<<<1,N>>>(dev_a,dev_b,dev_c);

	//Se copia el arreglo 'c' de la GPU al CPU
	cudaMemcpy(c,dev_c,N*sizeof(int),cudaMemcpyDeviceToHost);

	//Se muestran los resultados
	for(j=0;j<N;j++)
	{
		printf("%d + %d = %d\n",a[j],b[j],c[j]);
	}
    
	//Se libera la memoria reservada en la GPU
	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);

	return 0;
}

