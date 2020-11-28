#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <math.h>

using namespace std;

// 2D surfaces
surface<void, cudaSurfaceType2D> inputSurfRef;
surface<void, cudaSurfaceType2D> outputSurfRef;

// kernel: copy and increment by one
__global__ void copyKernel(int width, int height)
{
	// Calculate surface coordinates
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x < width && y < height)
	{
		float data_1,data_2,Cvalue=0;
		int e;
		for(e=0;e<width;e++)
		{
			surf2Dread(&data_1, inputSurfRef, e*4, y);
			surf2Dread(&data_2, inputSurfRef, x*4, e);
			Cvalue=data_1*data_2 + Cvalue;
			
		}
		surf2Dwrite(Cvalue, outputSurfRef, x*4, y);
		// Read from input surface
		//surf2Dread(&data, inputSurfRef, x*4, y);
		// Write to output surface
		//data = data + 2;
		//surf2Dwrite(data, outputSurfRef, x*4, y);
	}
}

// Host code
int main()
{
	int width = 3;
	int height = 3;
	int size = sizeof(float)*width*height;
	//allocate host and device memory
	float *h_data;
	float *h_data_out;
	h_data = (float*)malloc(sizeof(float)*height*width);
	h_data_out = (float*)malloc(sizeof(float)*height*width);
	//initialize host matrix before usage
	for(int loop=0; loop<width*height;loop++)
	h_data[loop] = (float)rand()/(float)(RAND_MAX-1);
	cout<<"datos entrada : "<<endl<<endl;
	for(int i = 0;i<width*height;i++)
	{
		cout<<h_data[i]<<endl;
	}
	// Allocate CUDA arrays in device memory
	cudaChannelFormatDesc channelDesc;
	channelDesc = cudaCreateChannelDesc<float>();
	cudaArray* cuInputArray; cudaArray* cuOutputArray;
	cudaMallocArray(&cuInputArray, &channelDesc, width, height,cudaArraySurfaceLoadStore);
	cudaMallocArray(&cuOutputArray, &channelDesc, width, height, cudaArraySurfaceLoadStore);
	// Copy to device memory some data located at address h_data in host memory
	cudaMemcpyToArray(cuInputArray, 0, 0, h_data, size, cudaMemcpyHostToDevice);
	// Bind the arrays to the surface references
	cudaBindSurfaceToArray(inputSurfRef, cuInputArray);
	cudaBindSurfaceToArray(outputSurfRef, cuOutputArray);
	// Invoke kernel
	dim3 dimBlock(3, 3, 1);
	dim3 dimGrid(1,1,1);
	copyKernel<<<dimGrid, dimBlock>>>(width, height);
	// Copy to host memory some data located at address outputSurfRef in device memory
	cudaMemcpyFromArray(h_data_out,cuOutputArray,0,0 , size, cudaMemcpyDeviceToHost);
	// Display
	cout<<endl<<"datos de salida : "<<endl<<endl;
	for(int i = 0;i<width*height;i++)
	{
		cout<<h_data_out[i]<<endl;
	}
	// Free device memory
	free(h_data);
	cudaFreeArray(cuInputArray);
	cudaFreeArray(cuOutputArray);
	system("pause");
	return 0;
}