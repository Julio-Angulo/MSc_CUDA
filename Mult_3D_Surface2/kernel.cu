
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <math.h>

//declare texture reference
surface<void,cudaSurfaceType3D> surfReference_A;
surface<void,cudaSurfaceType3D> surfReference_B;
surface<void,cudaSurfaceType3D> surfReference_C;
// kernel function
__global__ void kernel(int alto, int ancho, int prof)
{
	int xIndex;
	int yIndex;
	int zIndex;

	//calculate each thread global index
	xIndex = threadIdx.x + blockIdx.x * blockDim.x;
	yIndex = threadIdx.y + blockIdx.y * blockDim.y;
	zIndex = threadIdx.z + blockIdx.z * blockDim.z;

	float data1,data2;
	surf3Dread(&data1, surfReference_A, xIndex*4, yIndex, zIndex, cudaBoundaryModeTrap);
	surf3Dread(&data2, surfReference_B, xIndex*4, yIndex, zIndex, cudaBoundaryModeTrap);
	surf3Dwrite(data1 + data2, surfReference_C, xIndex*4, yIndex, zIndex,cudaBoundaryModeTrap);

	printf(" %f \n",data);
	//surf3Dwrite(data1 + data2, surfReference_C, xIndex*4, yIndex, zIndex);
	
		
}

int main(int argc, char* argv[])
{
	float *A_host;
	float *B_host;
	float *C_host;
	
	cudaArray *cudaArray_A;
	cudaArray *cudaArray_B;
	cudaArray *cudaArray_C;

	cudaExtent volumeSize_A;
	cudaExtent volumeSize_B;
	cudaExtent volumeSize_C;

	cudaChannelFormatDesc channel_A;
	cudaChannelFormatDesc channel_B;
	cudaChannelFormatDesc channel_C;

	cudaMemcpy3DParms copyparms_A={0};
	cudaMemcpy3DParms copyparms_B={0};
	cudaMemcpy3DParms copyparms_C={0};
	
	int alto,ancho,prof;

	alto = atoi(argv[1]);
	ancho = atoi(argv[2]);
	prof = atoi(argv[3]);

	//allocate host and device memory
	A_host = (float*)malloc(sizeof(float)*alto*ancho*prof);
	B_host = (float*)malloc(sizeof(float)*alto*ancho*prof);
	C_host = (float*)malloc(sizeof(float)*alto*ancho*prof);
	

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
	volumeSize_C = make_cudaExtent(ancho,alto,prof);

	//create channel to describe data type
	channel_A = cudaCreateChannelDesc<float>();
	channel_B = cudaCreateChannelDesc<float>();
	channel_C = cudaCreateChannelDesc<float>();

	//allocate device memory for cuda array
	cudaMalloc3DArray(&cudaArray_A,&channel_A,volumeSize_A,cudaArraySurfaceLoadStore);
	cudaMalloc3DArray(&cudaArray_B,&channel_B,volumeSize_B,cudaArraySurfaceLoadStore);
	cudaMalloc3DArray(&cudaArray_C,&channel_C,volumeSize_C,cudaArraySurfaceLoadStore);

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



	//bind texture reference with cuda array
	cudaBindSurfaceToArray(surfReference_A,cudaArray_A, channel_A);
	cudaBindSurfaceToArray(surfReference_B,cudaArray_B, channel_B);
	cudaBindSurfaceToArray(surfReference_C,cudaArray_C, channel_C);

	// preparing kernel launch
	dim3 blockDim; dim3 gridDim;
	blockDim.x = ancho; blockDim.y = alto; blockDim.z = prof;
	gridDim.x = 1; gridDim.y = 1; gridDim.z = 1;

	//execute device kernel
	kernel<<< gridDim , blockDim >>>(alto, ancho, prof);
	cudaThreadSynchronize();

	//unbind texture reference to free resource
	

	//copy result matrix from device to host memory
	//cudaMemcpyFromArray(C_host, cudaArray_C, 0, 0, sizeof(float)*alto*ancho*prof, cudaMemcpyDeviceToHost);

	//set cuda array copy parameters
	copyparms_C.srcArray = cudaArray_C;
	copyparms_C.extent = volumeSize_C;
	copyparms_C.dstPtr.ptr = C_host;
	copyparms_C.kind = cudaMemcpyDeviceToHost;
	


	// 3D copy from host_CubeMatrix to cudaArray
	//copyparms_C.srcPtr = make_cudaPitchedPtr((void*)cudaArray_C,ancho*sizeof(float),ancho,alto);
	cudaMemcpy3D(&copyparms_C);

	
	

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
	cudaFreeArray(cudaArray_A);
	cudaFreeArray(cudaArray_B);
	cudaFreeArray(cudaArray_C);
}
