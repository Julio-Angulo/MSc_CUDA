//PROGRAMA QUE UTILIZA EL PARALELISMO DINAMICO
//PROGRAMA QUE SUMA DOS VECTORES (a y b) Y ALMACENA EL RESULTADO EN EL VECTOR (c)
//PROGRAMA QUE MULTIPLICA DOS VECTORES (a y b) Y ALMACENA EL RESULTADO EN EL VECTOR (d)

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#define N 5

__global__ void kernel_hijo(int *a, int *b, int *d)
{
	int Idx = threadIdx.x;
	if(Idx < N)
	{
		d[Idx] = a[Idx] * b[Idx];
	} 
}


__global__ void kernel_padre(int *a, int *b, int *c, int *d)
{
	int tid = threadIdx.x;
	if(tid < N)
	{
		c[tid] = a[tid] + b[tid];
	}

	if(threadIdx.x == 0)
	{
		kernel_hijo<<<1,N>>>(a, b, d);
	}
    
}

int main()
{
	int a[N], b[N], c[N],d[N],i,j;
	int *dev_a, *dev_b, *dev_c, *dev_d;

	//Reservar memoria en el GPU
	cudaMalloc((void**)&dev_a,N*sizeof(int));
	cudaMalloc((void**)&dev_b,N*sizeof(int));
	cudaMalloc((void**)&dev_c,N*sizeof(int));
	cudaMalloc((void**)&dev_d,N*sizeof(int));

	//Se rellenan los arreglos 'a' y 'b' en el CPU
	for(i=0; i<N;i++)
	{
		a[i]=i+1;
		b[i]=i+1;
	}

	//Se muestran los valores del vector "a"
	printf("\n\n---VECTOR A---\n");
	for(j=0;j<N;j++)
	{
		printf(" %d ",a[j]);
	}
	printf("\n");

	//Se muestran los valores del vector "b"
	printf("\n\n---VECTOR B---\n");
	for(j=0;j<N;j++)
	{
		printf(" %d ",b[j]);
	}
	printf("\n");

	//Se copian los arreglos 'a' y 'b' a la GPU
	cudaMemcpy(dev_a,a,N*sizeof(int),cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b,b,N*sizeof(int),cudaMemcpyHostToDevice);

	//Se manda llamar el kernel
	//add<<<N,1>>>(dev_a,dev_b,dev_c);
	kernel_padre<<<1,N>>>(dev_a,dev_b,dev_c,dev_d);

	//Se copia el arreglo 'c' de la GPU al CPU
	cudaMemcpy(c,dev_c,N*sizeof(int),cudaMemcpyDeviceToHost);

	//Se copia el arreglo 'd' de la GPU al CPU
	cudaMemcpy(d,dev_d,N*sizeof(int),cudaMemcpyDeviceToHost);

	//Se muestran los resultados del vector "c"
	printf("\n\n---VECTOR C---\n");
	for(j=0;j<N;j++)
	{
		printf(" %d ",c[j]);
	}
	printf("\n");
    
	//Se muestran los resultados del vector "d"
	printf("\n\n---VECTOR D---\n");
	for(j=0;j<N;j++)
	{
		printf(" %d ",d[j]);
	}
	printf("\n");

	//Se libera la memoria reservada en la GPU
	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);
	cudaFree(dev_d);	
}

