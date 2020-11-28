
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <cuComplex.h>
#include <math.h>
#include <math_constants.h>
#include <iostream>

__global__ void VectorAddKernel(int *a, int *b, int *c, int N)
{
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;
	int z = blockDim.z * blockIdx.z + threadIdx.z;
	int tid = x + (y * blockDim.x * gridDim.x) + (z * blockDim.x * gridDim.x * blockDim.y * gridDim.y);
	if (tid < N)
	{
		c[tid] = a[tid] + b[tid];
	}

}

int main()
{
	int i,j,loop,muestras;
	unsigned long N,k;
	float suma;

	//Numero de iteraciones
	loop = 500;

	//N�mero de muestras
    muestras=130;
	
	float promedio[130];

	///Se crean los archivos binarios donde se guardar�n los datos
    FILE *da;
    da = fopen("Suma_vectores_500_iteraciones_paralelizado.bin","a+b"); //Crea o sobre escribe archivo

	//Pausa
	printf("\n---PRESIONA UNA TECLA PARA CONTINUAR---\n\n");
	getchar();

	for(i=1;i<=muestras;i++)
    {
        ///N�mero de elementos de los vectores a,b y c
        N=2048*i;
    
        suma=0.0;
        for(j=0;j<loop;j++)
        {
			//Comandos necesarios para medir el tiempo
			float elapsedTime_app;
			cudaEvent_t start_app, stop_app;
			cudaEventCreate(&start_app);
			cudaEventCreate(&stop_app);

			//Declaraci�n de variables
            int *a_host;
            int *b_host;
            int *c_host;
			int *a_device;
            int *b_device;
            int *c_device;
            
			//Se reserva memoria en host y device para a, b y c
            a_host = (int*) malloc(sizeof(int)*N);
            b_host = (int*) malloc(sizeof(int)*N);
            c_host = (int*) malloc(sizeof(int)*N);
			cudaMalloc((void**)&a_device,sizeof(int)*N);
			cudaMalloc((void**)&b_device,sizeof(int)*N);
			cudaMalloc((void**)&c_device,sizeof(int)*N);

			//Se dan valores a los vectores a y b
            for(k=0;k<N;k++)
            {
                a_host[k]=rand()%11;
                b_host[k]=rand()%21;
            }

			//Dimensionamiento del grid para la funci�n kernel
			dim3 gridDim(1,1,1);
			dim3 blockDim(1,1,1);
			int maxNumThreads = 1024;
			dim3 maxNumBlocks(2147483647,65535,65535);
			if(N <= maxNumThreads)
			{
				blockDim.x = N;
			}
			else
			{
				blockDim.x = maxNumThreads;
				if(ceil(N/blockDim.x) <= maxNumBlocks.x)
				{
					gridDim.x = ceil(N/blockDim.x);
				}
				else
				{
					gridDim.x = maxNumBlocks.x;
					if(ceil(N/(blockDim.x*gridDim.x)) <= maxNumBlocks.y)
					{
						gridDim.y = ceil(N/(blockDim.x*gridDim.x));
					}
					else
					{
						gridDim.y = maxNumBlocks.y;
						if(ceil(N/(blockDim.x*gridDim.x*blockDim.y*gridDim.y)) <=maxNumBlocks.z)
						{
							gridDim.z = ceil(N/(blockDim.x*gridDim.x*blockDim.y*gridDim.y));
						}
						else
						{
							printf("El n�mero de datos excede la capacidad del grid");
						}
					}
				}
			}

			//Env�o de los arreglos a y b al device
			cudaMemcpy(a_device,a_host,sizeof(int)*N,cudaMemcpyHostToDevice);
			cudaMemcpy(b_device,b_host,sizeof(int)*N,cudaMemcpyHostToDevice);

			//---------------------------------------------------------------------------------------------
			//Se empieza a medir el tiempo de ejecucion de la aplicacion
			cudaEventRecord(start_app,0);

			//Lanzamiento del kernel
			VectorAddKernel<<<gridDim,blockDim>>>(a_device,b_device,c_device,N);
			cudaThreadSynchronize();

			//------------------------------------------------------------------------------------------
			//Comandos necesarios para medir el tiempo de la aplicacion (app)
			cudaEventRecord(stop_app,0);
			cudaEventSynchronize(stop_app);
			cudaEventElapsedTime(&elapsedTime_app,start_app,stop_app);

			//Se leen los resultados de la GPU
			cudaMemcpy(c_host,c_device,sizeof(int)*N,cudaMemcpyDeviceToHost);

			//Se liberan memorias del Host y Device
			free(a_host);
			free(b_host);
			free(c_host);
			cudaFree(a_device);
			cudaFree(b_device);
			cudaFree(c_device);

			//Suma de todos los tiempos
			suma = suma + elapsedTime_app;

			//Se destruyen los eventos que miden el tiempo de la aplicacion
			cudaEventDestroy(start_app);
			cudaEventDestroy(stop_app);
		}
        promedio[i-1] = suma/(float)loop;
        printf("%d - Tiempo promedio para N = %ld >>> %f mS\n",i,N,promedio[i-1]);

    }
    fwrite(promedio,sizeof(float),muestras,da);
    fclose(da);
}
