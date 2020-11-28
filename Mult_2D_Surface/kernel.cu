
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <math.h>
#include <stdlib.h>
#include "estructura.h"

//2D surfaces
surface<void,cudaSurfaceType2D> A_SurfRef;
surface<void,cudaSurfaceType2D> B_SurfRef;
surface<void,cudaSurfaceType2D> C_SurfRef;

//Multiplicacion de matriz - Host code
//Se asume que las dimensiones de las matrices son multiplos de BLOCK_SIZE
void MatMul(const Matrix A,const Matrix B, Matrix C)
{
	//Allocate CUDA arrays in device memory
	Matrix d_A;
	d_A.width = A.width;
	d_A.height = A.height;

	Matrix d_B;
	d_B.width = B.width;
	d_B.height = B.height;

	Matrix d_C;
	d_C.width = C.width;
	d_C.height = C.height;


	cudaChannelFormatDesc channelDesc;
	channelDesc = cudaCreateChannelDesc<float>();
	cudaArray* matriz_A;
	cudaArray* matriz_B;
	cudaArray* matriz_C;
	cudaMallocArray(&matriz_A,&channelDesc,A.width,A.height,cudaArraySurfaceLoadStore);
	cudaMallocArray(&matriz_B,&channelDesc,B.width,B.height,cudaArraySurfaceLoadStore);
	cudaMallocArray(&matriz_C,&channelDesc,C.width,C.height,cudaArraySurfaceLoadStore);

	//Copy to device memory some data located at address A.elements and B.elements in host memory
	cudaMemcpyToArray(matriz_A,0,0,A.elements,A.width*A.height*sizeof(float),cudaMemcpyHostToDevice);
	cudaMemcpyToArray(matriz_B,0,0,B.elements,B.width*B.height*sizeof(float),cudaMemcpyHostToDevice);

	//Bind the arrays to the surface references
	cudaBindSurfaceToArray(A_SurfRef,matriz_A);
	cudaBindSurfaceToArray(B_SurfRef,matriz_B);
	cudaBindSurfaceToArray(C_SurfRef,matriz_C);
	
	//Se invoca el kernel
	dim3 dimBlock(BLOCK_SIZE,BLOCK_SIZE);
	dim3 dimGrid((B.width+dimBlock.x-1)/dimBlock.x,(A.height+dimBlock.y-1)/dimBlock.y);
	MatMulKernel<<<dimGrid, dimBlock>>> (d_A,d_B,d_C);
	
	//Copy to host memory some data located at address C_SurfRef in device memory
	cudaMemcpyFromArray(C.elements,matriz_C,0,0,C.width*C.height*sizeof(float),cudaMemcpyDeviceToHost);

	//Se libera memoria de la GPU
	cudaFree(matriz_A);
	cudaFree(matriz_B);
	cudaFree(matriz_C);
}

//Kernel para multiplicar las matrices llamada por Matmul()
__global__ void MatMulKernel(Matrix A,Matrix B,Matrix C)
{
	int e;
	float data_1,data_2,Cvalue=0;
	
	//Calculate surface coordinates
	int x=(blockIdx.x*blockDim.x)+threadIdx.x;
	int y=(blockIdx.y*blockDim.y)+threadIdx.y;

	//Cada hilo calcula un elemento de C acomulando los resultados en Cvalue
	if( y >= A.height || x >= B.width) return;
			for(e=0;e<A.width;e=e+1)
			{
				surf2Dread(&data_1,A_SurfRef,e*4,y);
				surf2Dread(&data_2,B_SurfRef,x*4,e);
				Cvalue = data_1*data_2 + Cvalue;
			}
			surf2Dwrite(Cvalue,C_SurfRef,x*4,y);
}

//Usage: Mult_2D_Surface a1 a2 b2
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


