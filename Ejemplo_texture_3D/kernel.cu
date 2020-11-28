
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <math.h>

//declare texture reference
texture<float,cudaTextureType3D,cudaReadModeElementType> textReference_A;
texture<float,cudaTextureType3D,cudaReadModeElementType> textReference_B;

// kernel function
__global__ void kernel(float *C_device, int alto, int ancho, int prof)
{
	int xIndex;
	int yIndex;
	int zIndex;

	//calculate each thread global index
	xIndex = threadIdx.x + blockIdx.x * blockDim.x;
	yIndex = threadIdx.y + blockIdx.y * blockDim.y;
	zIndex = threadIdx.z + blockIdx.z * blockDim.z;

	
	C_device[(zIndex*alto*ancho) + (yIndex*ancho) + xIndex] = tex3D(textReference_A,xIndex,yIndex,zIndex) + tex3D(textReference_B,xIndex,yIndex,zIndex);
		
}

int main(int argc, char* argv[])
{
	float *A_host;
	float *B_host;
	float *C_host;
	float *C_device;
	cudaArray *cudaArray_A;
	cudaArray *cudaArray_B;
	cudaExtent volumeSize_A;
	cudaExtent volumeSize_B;
	cudaChannelFormatDesc channel_A;
	cudaChannelFormatDesc channel_B;
	cudaMemcpy3DParms copyparms_A={0};
	cudaMemcpy3DParms copyparms_B={0};
	
	int alto,ancho,prof;

	alto = atoi(argv[1]);
	ancho = atoi(argv[2]);
	prof = atoi(argv[3]);

	//allocate host and device memory
	A_host = (float*)malloc(sizeof(float)*alto*ancho*prof);
	B_host = (float*)malloc(sizeof(float)*alto*ancho*prof);
	C_host = (float*)malloc(sizeof(float)*alto*ancho*prof);
	cudaMalloc((void**)&C_device,sizeof(float)*alto*ancho*prof);

	//initialize A_host matrix before usage
	for(int loop=0; loop<alto*ancho*prof;loop++)
		A_host[loop] = (float)(rand() % 3);
	
	//initialize B_host matrix before usage
	for(int loop=0; loop<alto*ancho*prof;loop++)
		B_host[loop] = (float)(rand() % 5);
    

	printf("A_host:\n");
	for(int i = 0;i<prof;i++)
	{
		for(int j = 0;j<alto;j++)
		{
			for(int k = 0;k<ancho;k++)
			{
				printf(" %f ",A_host[i*alto*ancho + j*ancho + k]);
			}
			printf("\n");
		}
		printf("\n\n");
	}
	
	printf("\n");
	printf("B_host:\n");
	for(int i = 0;i<prof;i++)
	{
		for(int j = 0;j<alto;j++)
		{
			for(int k = 0;k<ancho;k++)
			{
				printf(" %f ",B_host[i*alto*ancho + j*ancho + k]);
			}
			printf("\n");
		}
		printf("\n\n");
	}
	

	//set cuda array volume size
	volumeSize_A = make_cudaExtent(ancho,alto,prof);
	volumeSize_B = make_cudaExtent(ancho,alto,prof);

	//create channel to describe data type
	channel_A = cudaCreateChannelDesc<float>();
	channel_B = cudaCreateChannelDesc<float>();

	//allocate device memory for cuda array
	cudaMalloc3DArray(&cudaArray_A,&channel_A,volumeSize_A);
	cudaMalloc3DArray(&cudaArray_B,&channel_B,volumeSize_B);

	//set cuda array copy parameters
	copyparms_A.extent = volumeSize_A;
	copyparms_A.dstArray = cudaArray_A;
	copyparms_A.kind = cudaMemcpyHostToDevice;
	copyparms_B.extent = volumeSize_B;
	copyparms_B.dstArray = cudaArray_B;
	copyparms_B.kind = cudaMemcpyHostToDevice;

	// 3D copy from host_CubeMatrix to cudaArray
	copyparms_A.srcPtr = make_cudaPitchedPtr((void*)A_host,ancho*sizeof(float),ancho,alto);
	cudaMemcpy3D(&copyparms_A);
	copyparms_B.srcPtr = make_cudaPitchedPtr((void*)B_host,ancho*sizeof(float),ancho,alto);
	cudaMemcpy3D(&copyparms_B);

	//set texture filter mode property
	//use cudaFilterModePoint of cudaFilterModeLinear
	textReference_A.filterMode = cudaFilterModePoint;
	textReference_B.filterMode = cudaFilterModePoint;

	//set texture address mode property
	//use cudaAddressModeClamp or cudaAddressModeWrap for integer coordinates
	textReference_A.addressMode[0] = cudaAddressModeClamp;
	textReference_A.addressMode[1] = cudaAddressModeClamp;
	textReference_A.addressMode[2] = cudaAddressModeClamp;
	textReference_B.addressMode[0] = cudaAddressModeClamp;
	textReference_B.addressMode[1] = cudaAddressModeClamp;
	textReference_B.addressMode[2] = cudaAddressModeClamp;

	//bind texture reference with cuda array
	cudaBindTextureToArray(textReference_A,cudaArray_A, channel_A);
	cudaBindTextureToArray(textReference_B,cudaArray_B, channel_B);

	// preparing kernel launch
	dim3 blockDim; dim3 gridDim;
	blockDim.x = ancho; blockDim.y = alto; blockDim.z = prof;
	gridDim.x = 1; gridDim.y = 1; gridDim.z = 1;

	//execute device kernel
	kernel<<< gridDim , blockDim >>>(C_device, alto, ancho, prof);
	cudaThreadSynchronize();

	//unbind texture reference to free resource
	cudaUnbindTexture(textReference_A);
	cudaUnbindTexture(textReference_B);

	//copy result matrix from device to host memory
	cudaMemcpy(C_host, C_device, sizeof(float)*alto*ancho*prof, cudaMemcpyDeviceToHost);
	
	

	printf("\n");
	printf("C_host:\n");
	for(int i = 0;i<prof;i++)
	{
		for(int j = 0;j<alto;j++)
		{
			for(int k = 0;k<ancho;k++)
			{
				printf(" %f ",C_host[i*alto*ancho + j*ancho + k]);
			}
			printf("\n");
		}
		printf("\n\n");
	}
	
	//free host and device memory
	free(A_host);
	free(B_host);
	free(C_host);
	cudaFree(C_device);
	cudaFreeArray(cudaArray_A);
	cudaFreeArray(cudaArray_B);
}