///VERSION PARALELISMO EXTERNO CON M THREADS (FILAS)
///Este programa encuentra la matriz transpuesta de "A" y el resultado lo guarda en la matriz "B"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <cuComplex.h>
#include <math.h>
#include <math_constants.h>
#include <iostream>

//2D surfaces
surface<void, cudaSurfaceType2D> A_surface;
surface<void, cudaSurfaceType2D> B_surface;

//Kernel
__global__ void MatrixTransposeKernel(int rows,int columns)
{
	int data = 0;
	int i,j;
	i = blockDim.y * blockIdx.y + threadIdx.y;
	if(i < rows)
	{
		for(j=0;j<columns;j++)
		{
			//A_host[r + (k*N)]=rand()%10;
			surf2Dread(&data,A_surface,i*4,j);
			surf2Dwrite(data,B_surface,j*4,i);
		}
	}

}

int main()
{
	int i,j,loop;
	unsigned long N,M,k,r;
	float suma;

	//Numero de iteraciones
	loop = 100;

	//Número de muestras
	const int muestras = 14;
	
	float promedio[muestras];

	///Se crean los archivos binarios donde se guardarán los datos
    FILE *da;
    da = fopen("Matriz_transpuesta_paralelismo_externo_M.bin","a+b"); //Crea o sobre escribe archivo

	//Pausa
	printf("\n---PRESIONA UNA TECLA PARA CONTINUAR---\n\n");
	getchar();

	N=1;
	for(i=1;i<=muestras;i++)
    {
        ///Número de columnas de la matriz "A"
        N=N*2;

		///Número de renglones de la matriz "A"
		M=N;
    
        suma=0.0;
        for(j=0;j<loop;j++)
        {
			//Comandos necesarios para medir el tiempo
			float elapsedTime_app;
			cudaEvent_t start_app, stop_app;
			cudaEventCreate(&start_app);
			cudaEventCreate(&stop_app);

			//Declaración de variables
            int* A_host;
            int* B_host;

            //Se reserva memoria en host y device para las matrices A y B
            A_host = (int*) malloc(sizeof(int)*N*M);
            B_host = (int*) malloc(sizeof(int)*N*M);
           
			//Se dan valores a la matriz "A"
			for(k=0;k<M;k++)
            {
				for(r=0;r<N;r++)
				{
					A_host[r + (k*N)]=rand()%10;
				}
            }

			/*
			//Se imprime la matriz "A"
			printf("\n\n---MATRIZ A---\n\n");
			for(k=0;k<M;k++)
            {
				for(r=0;r<N;r++)
				{
					printf(" %d ",A_host[r + (k*N)]);
				}
				printf("\n");
            }
			printf("\n");
			*/

			//Allocate CUDA arrays in device memory
			cudaChannelFormatDesc channelDesc;
			channelDesc = cudaCreateChannelDesc<int>();
			cudaArray* A_array;
			cudaArray* B_array;
			cudaMallocArray(&A_array,&channelDesc,N,M,cudaArraySurfaceLoadStore);
			cudaMallocArray(&B_array,&channelDesc,N,M,cudaArraySurfaceLoadStore);

			//Copy to device memory some data located at address A_host in host memory
			cudaMemcpyToArray(A_array,0,0,A_host,N*M*sizeof(int),cudaMemcpyHostToDevice);

			//Bind the arrays to the surface references
			cudaBindSurfaceToArray(A_surface,A_array);
			cudaBindSurfaceToArray(B_surface,B_array);

			//Dimensionamiento del grid para la función kernel
			//Dimensionamiento del Grid
			dim3 gridDim;
			gridDim.x = 1;
			gridDim.z = 1;
			//Dimensionamiento del block
			dim3 blockDim;
			blockDim.x = 1;
			blockDim.z = 1;

			if(M < 1024)
			{
				blockDim.y = M;
				gridDim.y = 1;
			}
			else
			{
				blockDim.y = 1024;
				gridDim.y = ceil(M/blockDim.y);
			}

			//---------------------------------------------------------------------------------------------
			//Se empieza a medir el tiempo de ejecucion de la aplicacion
			cudaEventRecord(start_app,0);
			
			//Lanzamiento del kernel
			MatrixTransposeKernel<<<gridDim,blockDim >>>(M,N);
			cudaThreadSynchronize();

			//---------------------------------------------------------------------------------------------
			//Comandos necesarios para medir el tiempo de la aplicacion (app)
			cudaEventRecord(stop_app,0);
			cudaEventSynchronize(stop_app);
			cudaEventElapsedTime(&elapsedTime_app,start_app,stop_app);

			//Se leen los resultados de la GPU
			cudaMemcpyFromArray(B_host,B_array,0,0,M*N*sizeof(int),cudaMemcpyDeviceToHost);

			/*
			//Se imprime la matriz "B"
			printf("\n\n---MATRIZ B---\n\n");
			for(k=0;k<M;k++)
            {
				for(r=0;r<N;r++)
				{
					printf(" %d ",B_host[r + (k*N)]);
				}
				printf("\n");
            }
			printf("\n");
			*/

			//Se liberan memorias del Host y Device
			free(A_host);
			free(B_host);
			cudaFreeArray(A_array);
			cudaFreeArray(B_array);

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
