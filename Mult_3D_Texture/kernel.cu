
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdlib.h>
#include <iostream>
#include <math.h>
#include "estructura.h"

//declare texture reference
texture<float,cudaTextureType3D,cudaReadModeElementType> textReference_A;


//Multiplicacion de matriz - Host code
//Se asume que las dimensiones de las matrices son multiplos de BLOCK_SIZE
void MatMul( Matrix A, Matrix C)
{
	//Se le reserva memoria a la matriz C en la memoria global de la GPU
	Matrix d_C;
	d_C.width=C.width;
	d_C.height=C.height;
	d_C.deep=C.deep;
	size_t size=C.width*C.height*C.deep*sizeof(float);
	cudaError_t err=cudaMalloc(&d_C.elements,size);
	printf("CUDA malloc C: %s\n",cudaGetErrorString(err));

	//Parametros reservar memoria a la matriz A en la memoria de texturas de la GPU
	cudaArray *cudaArray_A;
	cudaExtent volumeSize_A;
	cudaChannelFormatDesc channel;

	cudaMemcpy3DParms copyparms_A={0};

	volumeSize_A = make_cudaExtent(A.width,A.height,A.deep);
	
	channel = cudaCreateChannelDesc<float>();

	//Se reserva memoria de texturas para cudaArray_A
	cudaMalloc3DArray(&cudaArray_A,&channel,volumeSize_A);

	//Set cuda array copy parameters for matriz A
	
	copyparms_A.extent = volumeSize_A;
	copyparms_A.dstArray = cudaArray_A;
	copyparms_A.kind = cudaMemcpyHostToDevice;

	//3D copy from A.elements to cudaArray_A
	copyparms_A.srcPtr = make_cudaPitchedPtr((void*)A.elements,A.width*sizeof(float),A.height,A.deep);
	cudaMemcpy3D(&copyparms_A);

	//Set texture filter mode property for Matriz A
	//Use cudaFilterModePoint of cudaFilterModeLinear
	textReference_A.filterMode = cudaFilterModePoint;

	//Set texture address mode property for Matriz A
	//Use cudaAddressModeClamp or cudaAddressModeWrap for integer coordinates
	textReference_A.addressMode[0] = cudaAddressModeClamp;
	textReference_A.addressMode[1] = cudaAddressModeClamp;
	textReference_A.addressMode[2] = cudaAddressModeClamp;

	//Bind texture reference with cudaArray_A and cudaArray_B
	cudaBindTextureToArray(textReference_A,cudaArray_A,channel);


	//Se invoca el kernel
	//dim3 dimBlock(C.width,C.height,C.deep);
	//dim3 dimGrid((A.width + dimBlock.x-1)/dimBlock.x,(A.height + dimBlock.y-1)/dimBlock.y,(A.deep + dimBlock.z-1)/dimBlock.z);
	dim3 dimBlock(C.width,C.height,C.deep);
	dim3 dimGrid(1,1,1);

	MatMulKernel<<<dimGrid, dimBlock>>> (A,d_C);
	err=cudaThreadSynchronize();
	printf("Run kernel: %s\n", cudaGetErrorString(err));

	//Unbind texture reference to free resource
	cudaUnbindTexture(textReference_A);

	//Se lee la matriz C de la memoria de la GPU
	err=cudaMemcpy(C.elements,d_C.elements,C.width*C.height*C.deep*sizeof(float),cudaMemcpyDeviceToHost);
	printf("Copy C off of device: %s\n",cudaGetErrorString(err));

	//Se libera memoria de la GPU
	cudaFree(d_C.elements);
	cudaFreeArray(cudaArray_A);
}

//Kernel para multiplicar las matrices llamada por Matmul()
__global__ void MatMulKernel(Matrix A, Matrix d_C)
{
	int xIndex;
	int yIndex;
	int zIndex;

	//float Cvalue = 0.0;
	
	//Calcule each thread global index
	xIndex = (blockIdx.x * blockDim.x) + threadIdx.x ;
	yIndex = (blockIdx.y * blockDim.y) + threadIdx.y ;
	zIndex = (blockIdx.z * blockDim.z) + threadIdx.z ;


	if((xIndex == d_C.width) && (yIndex < d_C.height) && (zIndex < d_C.deep))
	{

		d_C.elements[zIndex*d_C.height*d_C.width + yIndex*d_C.width + xIndex]=tex3D(textReference_A,xIndex,yIndex,zIndex) ;
		//d_C.elements[zIndex*d_C.height*d_C.width + yIndex*d_C.width + xIndex]= 2;
	}

	return;	
}

//Usage: multNoShare a1 a2 a3
int main(int argc, char* argv[])
{
	int i,j,k;
	Matrix A,C;
	int a1,a2,a3;
	

	//Se leen valores de la linea de comandos
	a1=atoi(argv[1]); //alto de matriz A
	a2=atoi(argv[2]); //ancho de matriz A
	a3=atoi(argv[3]); //profundidad de matriz A
	
	A.height=a1;
	A.width=a2;
	A.deep=a3;
	A.elements=(float*)malloc(A.width*A.height*A.deep*sizeof(float));

	C.height=A.height;
	C.width=A.width;
	C.deep=A.deep;
	C.elements=(float*)malloc(C.width*C.height*C.deep*sizeof(float));


	for(i=0;i<A.deep;i++) //Se genera la matriz A de forma aleatoria
	{
		for(j=0;j<A.height;j++)
		{
			for(k=0;k<A.width;k++)
				A.elements[(i*A.height*A.width)+(j*A.width)+k]=(float)(rand() % 3);
		}

	}


		
	MatMul(A,C); //Se manda llamar la funcion "Matmul"

	//Se imprimen las 2 matrices
	printf("\nMATRIZ A:\n");
	for(i=0;i<A.deep;i++) 
	{
		for(j=0;j<A.height;j++)
		{
			for(k=0;k<A.width;k++)
				printf("%f ",A.elements[(i*A.height*A.width)+(j*A.width)+k]);
			printf("\n");
		}
		printf("\n\n");
	}
	printf("\n");

	printf("\nMATRIZ C:\n");
	for(i=0;i<C.deep;i++) 
	{
		for(j=0;j<C.height;j++)
		{
			for(k=0;k<C.width;k++)
				printf("%f ",C.elements[(i*C.height*C.width)+(j*C.width)+k]);
			printf("\n");
		}
		printf("\n\n");
	}
	printf("\n");
}
