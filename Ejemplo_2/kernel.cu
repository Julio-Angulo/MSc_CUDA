
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>



/*
__global__ void add(int a, int b, int *c)
{
    *c=a+b;
}
*/
int main(void)
{
	/*
	int c;
	int *dev_c;
	cudaMalloc((void**)&dev_c, sizeof(int));
	add<<<1,1>>>(20,7,dev_c);
	cudaMemcpy(&c,dev_c,sizeof(int),cudaMemcpyDeviceToHost);
	printf("20 + 7 = %d\n",c);
	cudaFree(dev_c);
	*/
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop,0);
	printf("Numero de multiprocesadores = %d \n",prop.multiProcessorCount);
	return 0;



}

