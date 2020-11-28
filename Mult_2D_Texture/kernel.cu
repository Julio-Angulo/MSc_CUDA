#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdlib.h>
#include <iostream>
#include <math.h>
#include "estructura.h"



//declare texture reference
texture<float,cudaTextureType2D> textReference_A;
texture<float,cudaTextureType2D> textReference_B;

//Multiplicacion de matriz - Host code
//Se asume que las dimensiones de las matrices son multiplos de BLOCK_SIZE
void MatMul(const Matrix A,const Matrix B ,Matrix C)
{
	//Se le reserva memoria a la matriz C en la memoria global de la GPU
	Matrix d_C;
	d_C.width=C.width;
	d_C.height=C.height;
	size_t size=C.width*C.height*sizeof(float);
	cudaError_t err=cudaMalloc(&d_C.elements,size);
	printf("CUDA malloc C: %s\n",cudaGetErrorString(err));

	//Parametros reservar memoria a la matriz A y B en la memoria de texturas de la GPU
	cudaArray *cudaArray_A;
	cudaArray *cudaArray_B;
	cudaChannelFormatDesc channel;
	channel = cudaCreateChannelDesc<float>();

	//Se reserva memoria de texturas para cudaArray_A
	cudaMallocArray(&cudaArray_A,&channel,A.width,A.height);

	//Se reserva memoria de texturas para cudaArray_B
	cudaMallocArray(&cudaArray_B,&channel,B.width,B.height);

    //Copy to device memory some data located at address A.elements
	cudaMemcpyToArray(cudaArray_A,0,0,A.elements,A.width*A.height*sizeof(float),cudaMemcpyHostToDevice);

	//Copy to device memory some data located at address B.elements
	cudaMemcpyToArray(cudaArray_B,0,0,B.elements,B.width*B.height*sizeof(float),cudaMemcpyHostToDevice);

	//Set texture filter mode property for Matriz A
	//Use cudaFilterModePoint of cudaFilterModeLinear
	textReference_A.filterMode = cudaFilterModePoint;

	//Set texture filter mode property for Matriz B
	//Use cudaFilterModePoint of cudaFilterModeLinear
	textReference_B.filterMode = cudaFilterModePoint;

	//Set texture address mode property for Matriz A
	//Use cudaAddressModeClamp or cudaAddressModeWrap for integer coordinates
	textReference_A.addressMode[0] = cudaAddressModeClamp;
	textReference_A.addressMode[1] = cudaAddressModeClamp;

	//Set texture address mode property for Matriz B
	//Use cudaAddressModeClamp or cudaAddressModeWrap for integer coordinates
	textReference_B.addressMode[0] = cudaAddressModeClamp;
	textReference_B.addressMode[1] = cudaAddressModeClamp;


	//Bind texture reference with cudaArray_A and cudaArray_B
	cudaBindTextureToArray(textReference_A,cudaArray_A, channel);
	cudaBindTextureToArray(textReference_B,cudaArray_B, channel);
	

	//Se invoca el kernel
	//dim3 dimBlock(C.width,C.height);
	//dim3 dimGrid((B.width + dimBlock.x-1)/dimBlock.x,(A.height + dimBlock.y-1)/dimBlock.y);

	dim3 dimBlock(4,3);
	dim3 dimGrid(2147483648,10);

	MatMulKernel<<<dimGrid, dimBlock>>> (A,B,d_C);
	err=cudaThreadSynchronize();
	printf("Run kernel: %s\n", cudaGetErrorString(err));

	//Unbind texture reference to free resource
	cudaUnbindTexture(textReference_A);
	cudaUnbindTexture(textReference_B);

	//Se lee la matriz C de la memoria de la GPU
	err=cudaMemcpy(C.elements,d_C.elements,C.width*C.height*sizeof(float),cudaMemcpyDeviceToHost);
	printf("Copy C off of device: %s\n",cudaGetErrorString(err));

	//Se libera memoria de la GPU
	cudaFree(d_C.elements);
	cudaFreeArray(cudaArray_A);
	cudaFreeArray(cudaArray_B);
}

//Kernel para multiplicar las matrices llamada por Matmul()
__global__ void MatMulKernel(Matrix A, Matrix B, Matrix d_C)
{
	int xIndex;
	int yIndex;
	int e;
	float CValue = 0.0;
	
	
	//Calcule each thread global index
	xIndex = (blockIdx.x * blockDim.x) + threadIdx.x ;
	yIndex = (blockIdx.y * blockDim.y) + threadIdx.y ;

	//xIndex = threadIdx.x ;
	//yIndex = threadIdx.y ;
	
	if((xIndex < d_C.width) && (yIndex < d_C.height) && blockIdx.x == 0)
	{
		for(e=0;e<A.width;e=e+1)
		{
			CValue= tex2D(textReference_A,e,yIndex) * tex2D(textReference_B,xIndex,e) + CValue;

		}
		d_C.elements[yIndex*d_C.width + xIndex]=CValue;
	}
	else
	{}
	
}

//Usage: multNoShare a1 a2 b2
int main(int argc, char* argv[])
{
	int j,k;
	Matrix A,B,C;
	int a1,a2;
	int b1,b2;

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


	for(j=0;j<A.height;j++) //Se genera la matriz A de forma aleatoria
	{
		for(k=0;k<A.width;k++)
			A.elements[(j*A.width)+k]=(float)(rand() % 3);
	}

	for(j=0;j<B.height;j++) //Se genera la matriz B de forma aleatoria
	{
		for(k=0;k<B.width;k++)
			B.elements[(j*B.width)+k]=(float)(rand() % 2);
	}

	
	MatMul(A,B,C); //Se manda llamar la funcion "Matmul"

	//Se imprimen las tres matrices
	printf("\nMATRIZ A:\n");
	for(j=0;j<A.height;j++)
	{
		for(k=0;k<A.width;k++)
			printf("%f ",A.elements[(j*A.width)+k]);
		printf("\n");
	}
	printf("\n\n");

	printf("\nMATRIZ B:\n");
	for(j=0;j<B.height;j++)
	{
		for(k=0;k<B.width;k++)
			printf("%f ",B.elements[(j*B.width)+k]);
		printf("\n");
	}
	printf("\n\n");
	

	printf("\nMATRIZ C:\n");
	for(j=0;j<C.height;j++)
	{
		for(k=0;k<C.width;k++)
			printf("%f ",C.elements[(j*C.width)+k]);
		printf("\n");
	}
	printf("\n\n");
}

