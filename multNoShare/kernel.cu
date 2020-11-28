
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdlib.h>
#include "multNoShare.h"



//Multiplicacion de matriz - Host code
//Se asume que las dimensiones de las matrices son multiplos de BLOCK_SIZE
void MatMul(const Matrix A,const Matrix B, Matrix C)
{
	//Se cargan las matrices A y B a la memoria de la GPU
	Matrix d_A;
	d_A.width=A.width;
	d_A.height=A.height;
	size_t size=A.width*A.height*sizeof(float);
	cudaError_t err=cudaMalloc(&d_A.elements,size);
	printf("CUDA malloc A: %s\n",cudaGetErrorString(err));
	err=cudaMemcpy(d_A.elements,A.elements,size,cudaMemcpyHostToDevice);
	printf("CUDA A to device: %s\n",cudaGetErrorString(err));

	Matrix d_B;
	d_B.width=B.width;
	d_B.height=B.height;
	size=B.width*B.height*sizeof(float);
	err=cudaMalloc(&d_B.elements,size);
	printf("CUDA malloc B: %s\n",cudaGetErrorString(err));
	err=cudaMemcpy(d_B.elements,B.elements,size,cudaMemcpyHostToDevice);
	printf("CUDA B to device: %s\n",cudaGetErrorString(err));

	//Se le reserva memoria a la matriz C en la la memoria de la GPU
	Matrix d_C;
	d_C.width=C.width;
	d_C.height=C.height;
	size=C.width*C.height*sizeof(float);
	err=cudaMalloc(&d_C.elements,size);
	printf("CUDA malloc C: %s\n",cudaGetErrorString(err));

	//Se invoca el kernel
	dim3 dimBlock(BLOCK_SIZE,BLOCK_SIZE);
	dim3 dimGrid((B.width+dimBlock.x-1)/dimBlock.x,(A.height+dimBlock.y-1)/dimBlock.y);
	MatMulKernel<<<dimGrid, dimBlock>>> (d_A,d_B,d_C);
	err=cudaThreadSynchronize();
	printf("Run kernel: %s\n", cudaGetErrorString(err));

	//Se lee la matriz C de la memoria de la GPU
	err=cudaMemcpy(C.elements,d_C.elements,size,cudaMemcpyDeviceToHost);
	printf("Copy C off of device: %s\n",cudaGetErrorString(err));

	//Se libera memoria de la GPU
	cudaFree(d_A.elements);
	cudaFree(d_B.elements);
	cudaFree(d_C.elements);
}

//Kernel para multiplicar las matrices llamada por Matmul()
__global__ void MatMulKernel(Matrix A,Matrix B,Matrix C)
{
	int e;
	//Cada hilo calcula un elemento de C acomulando los resultados en Cvalue
	float Cvalue=0;
	int row=(blockIdx.y*blockDim.y)+threadIdx.y;
	int col=(blockIdx.x*blockDim.x)+threadIdx.x;
	if(row >= A.height || col >= B.width) return;
			for(e=0;e<A.width;e=e+1)
			{
				Cvalue = (A.elements[row*A.width + e]) * (B.elements[e*B.width+col]) + Cvalue;
			}
			C.elements[row*C.width+col]=Cvalue;
}

//Usage: multNoShare a1 a2 b2
int main(int argc, char* argv[])
{
	int i,j;
	Matrix A,B,C;
	int a1,a2,b1,b2;

	//Se leen valores de la linea de comandos
	a1=atoi(argv[1]); //alto de matriz A
	a2=atoi(argv[2]); //ancho de matriz A
	b1=a2;            //alto de matriz B
	b2=atoi(argv[3]); //ancho de matriz B

	A.height=a1;
	A.width=a2;
	A.elements=(float*)malloc(A.width*A.height*sizeof(float));

	B.height=b1;
	B.width=b2;
	B.elements=(float*)malloc(B.width*B.height*sizeof(float));

	C.height=A.height;
	C.width=B.width;
	C.elements=(float*)malloc(C.width*C.height*sizeof(float));


	for(i=0;i<A.height;i++) //Se genera la matriz A de forma aleatoria
		for(j=0;j<A.width;j++)
			A.elements[i*A.width+j]=(float)(rand() % 3);

	for(i=0;i<B.height;i++) //Se genera la matriz B de forma aleatoria
		for(j=0;j<B.width;j++)
			B.elements[i*B.width+j]=(float)(rand() % 2);




	MatMul(A,B,C); //Se manda llamar la funcion "Matmul"

	//Se imprimen las tres matrices

	for(i=0;i<A.height;i++)
	{
		for(j=0;j<A.width;j++)
			printf("%f ",A.elements[i*A.width+j]);
		printf("\n");
	}
	printf("\n");

	for(i=0;i<B.height;i++)
	{
		for(j=0;j<B.width;j++)
			printf("%f ",B.elements[i*B.width+j]);
		printf("\n");
	}
	printf("\n");

	for(i=0;i<C.height;i++)
	{
		for(j=0;j<C.width;j++)
			printf("%f ",C.elements[i*C.width+j]);
		printf("\n");
	}
	printf("\n");


}

